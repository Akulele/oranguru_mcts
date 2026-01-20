#!/usr/bin/env python3
"""
Build switch-correction trajectories from clean switch opportunity logs.

These trajectories can be fed into the switch-focused BC phase to
directly target missed good switches in bad matchups.
"""

import argparse
import json
import pickle
from pathlib import Path


def infer_tag(path: Path) -> str:
    stem = path.stem
    prefix = "switch_opportunities_"
    if stem.startswith(prefix):
        return stem[len(prefix):]
    return stem


def iter_log_paths(log_path: str | None, log_dir: str | None) -> list[Path]:
    paths: list[Path] = []
    if log_dir:
        base = Path(log_dir)
        paths.extend(sorted(base.glob("switch_opportunities_*.jsonl")))
    if log_path:
        paths.append(Path(log_path))
    return paths


def build_trajectories(
    paths: list[Path],
    weight: float,
    include_switched: bool,
    stay_only: bool,
    min_best_delta: float,
    matchup_threshold: float | None,
    bad_matchup_weight: float | None,
    limit: int,
) -> tuple[list[dict], dict]:
    trajectories: list[dict] = []
    stats = {
        "lines": 0,
        "kept": 0,
        "skipped_no_features": 0,
        "skipped_no_best_action": 0,
        "skipped_decision_switch": 0,
        "skipped_not_stay": 0,
        "skipped_illegal_action": 0,
        "skipped_best_delta": 0,
        "skipped_matchup": 0,
    }

    for path in paths:
        tag = infer_tag(path)
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if limit and stats["kept"] >= limit:
                    return trajectories, stats
                line = line.strip()
                if not line:
                    continue
                stats["lines"] += 1
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if stay_only and data.get("decision") != "stay":
                    stats["skipped_not_stay"] += 1
                    continue
                if not include_switched and data.get("decision") == "switch":
                    stats["skipped_decision_switch"] += 1
                    continue
                features = data.get("features")
                mask = data.get("mask")
                if features is None or mask is None:
                    stats["skipped_no_features"] += 1
                    continue
                best_action = data.get("best_switch_action")
                if best_action is None:
                    stats["skipped_no_best_action"] += 1
                    continue
                best_delta = data.get("best_delta")
                if best_delta is not None and best_delta < min_best_delta:
                    stats["skipped_best_delta"] += 1
                    continue
                matchup = data.get("matchup")
                if matchup_threshold is not None:
                    if matchup is None or matchup > matchup_threshold:
                        stats["skipped_matchup"] += 1
                        continue
                if best_action >= len(mask) or not mask[best_action]:
                    stats["skipped_illegal_action"] += 1
                    continue

                traj_weight = weight
                if bad_matchup_weight is not None and matchup is not None and matchup_threshold is not None:
                    if matchup <= matchup_threshold:
                        traj_weight = bad_matchup_weight

                trajectories.append({
                    "features": [features],
                    "masks": [mask],
                    "actions": [best_action],
                    "rewards": [0.0],
                    "dones": [True],
                    "weight": traj_weight,
                    "tag": f"switch_correction_{tag}",
                })
                stats["kept"] += 1

    return trajectories, stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Build switch-correction trajectories")
    parser.add_argument("--log", type=str, default=None,
                        help="Single opportunity log JSONL")
    parser.add_argument("--log-dir", type=str, default=None,
                        help="Directory containing switch_opportunities_*.jsonl")
    parser.add_argument("--output", type=str, required=True,
                        help="Output pickle path for trajectories")
    parser.add_argument("--weight", type=float, default=2.5,
                        help="Trajectory weight for switch corrections")
    parser.add_argument("--include-switched", action="store_true",
                        help="Include cases where the agent already switched")
    parser.add_argument("--stay-only", action="store_true",
                        help="Only keep entries where the agent stayed")
    parser.add_argument("--min-best-delta", type=float, default=0.3,
                        help="Minimum best_delta threshold to keep")
    parser.add_argument("--matchup-threshold", type=float, default=None,
                        help="Only keep entries with matchup <= threshold")
    parser.add_argument("--bad-matchup-weight", type=float, default=None,
                        help="Optional weight override for bad matchups")
    parser.add_argument("--limit", type=int, default=0,
                        help="Optional max number of trajectories to keep (0 = no limit)")
    args = parser.parse_args()

    if not args.log and not args.log_dir:
        parser.error("--log or --log-dir is required")

    paths = iter_log_paths(args.log, args.log_dir)
    if not paths:
        print("No log files found.")
        return 1

    limit = args.limit if args.limit and args.limit > 0 else 0
    trajectories, stats = build_trajectories(
        paths,
        weight=args.weight,
        include_switched=args.include_switched,
        stay_only=args.stay_only,
        min_best_delta=args.min_best_delta,
        matchup_threshold=args.matchup_threshold,
        bad_matchup_weight=args.bad_matchup_weight,
        limit=limit,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        pickle.dump(trajectories, handle)

    print("Saved trajectories:", len(trajectories))
    print("Output:", output_path)
    for key, value in stats.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
