#!/usr/bin/env python3
"""Prepare cleaned ladder trajectories for offline BC/RL training.

This script merges raw ladder trajectory pickles (emitted by
`evaluation/ladder_rulebot.py --collect-data`) and applies strict filtering
for noisy samples before writing a training-ready pickle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from collections import Counter
from glob import glob
from pathlib import Path
from statistics import mean


N_ACTIONS = 13


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        action="append",
        default=[],
        help="Input pickle path, directory, or glob. Repeatable.",
    )
    parser.add_argument(
        "--output",
        default="data/ladder_trajectories.pkl",
        help="Output pickle path.",
    )
    parser.add_argument(
        "--summary-output",
        default="",
        help="Optional JSON summary output path (defaults to <output>.summary.json).",
    )
    parser.add_argument("--min-actions", type=int, default=4)
    parser.add_argument("--min-turns", type=int, default=8)
    parser.add_argument("--min-player-rating", type=int, default=0)
    parser.add_argument("--min-opponent-rating", type=int, default=0)
    parser.add_argument("--min-either-rating", type=int, default=0)
    parser.add_argument("--max-illegal-rate", type=float, default=0.05)
    parser.add_argument("--skip-forfeit", action="store_true", default=True)
    parser.add_argument(
        "--allow-forfeit",
        action="store_true",
        help="Disable forfeit filtering.",
    )
    parser.add_argument(
        "--allow-invalid-mask-action",
        action="store_true",
        help="Keep trajectories with any invalid action/mask pairs.",
    )
    parser.add_argument(
        "--disable-dedup",
        action="store_true",
        help="Disable trajectory de-duplication.",
    )
    parser.add_argument("--rating-baseline", type=int, default=1500)
    parser.add_argument("--rating-scale", type=float, default=1000.0)
    parser.add_argument("--rating-weight-floor", type=float, default=0.75)
    parser.add_argument("--rating-weight-cap", type=float, default=1.8)
    parser.add_argument("--win-multiplier", type=float, default=1.0)
    parser.add_argument("--loss-multiplier", type=float, default=1.0)
    return parser.parse_args()


def resolve_inputs(inputs: list[str]) -> list[Path]:
    paths: list[Path] = []
    seen: set[str] = set()
    for entry in inputs:
        if not entry:
            continue
        expanded = sorted(Path(p) for p in glob(entry))
        if expanded:
            for p in expanded:
                if p.is_dir():
                    for child in sorted(p.glob("*.pkl")):
                        key = str(child.resolve())
                        if key not in seen:
                            seen.add(key)
                            paths.append(child)
                elif p.is_file() and p.suffix == ".pkl":
                    key = str(p.resolve())
                    if key not in seen:
                        seen.add(key)
                        paths.append(p)
            continue

        p = Path(entry)
        if p.is_dir():
            for child in sorted(p.glob("*.pkl")):
                key = str(child.resolve())
                if key not in seen:
                    seen.add(key)
                    paths.append(child)
        elif p.is_file() and p.suffix == ".pkl":
            key = str(p.resolve())
            if key not in seen:
                seen.add(key)
                paths.append(p)

    return paths


def load_trajectory_list(path: Path) -> list[dict]:
    with path.open("rb") as f:
        payload = pickle.load(f)
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if isinstance(payload, dict):
        if isinstance(payload.get("trajectories"), list):
            return [x for x in payload["trajectories"] if isinstance(x, dict)]
    return []


def is_win_trajectory(traj: dict) -> bool:
    rewards = traj.get("rewards") or []
    if rewards:
        try:
            return float(rewards[-1]) > 0.0
        except Exception:
            pass
    return bool(traj.get("won", False))


def validate_lengths(traj: dict) -> tuple[bool, str]:
    features = traj.get("features") or []
    masks = traj.get("masks") or []
    actions = traj.get("actions") or []
    rewards = traj.get("rewards") or []
    dones = traj.get("dones") or []

    n = len(actions)
    if n == 0:
        return False, "empty"
    if len(features) != n:
        return False, "len_features"
    if len(masks) != n:
        return False, "len_masks"
    if rewards and len(rewards) != n:
        return False, "len_rewards"
    if dones and len(dones) != n:
        return False, "len_dones"
    return True, ""


def action_mask_valid(traj: dict) -> bool:
    masks = traj.get("masks") or []
    actions = traj.get("actions") or []
    for mask, action in zip(masks, actions):
        if not isinstance(mask, (list, tuple)):
            return False
        if len(mask) != N_ACTIONS:
            return False
        if not (0 <= int(action) < N_ACTIONS):
            return False
        try:
            if not bool(mask[int(action)]):
                return False
        except Exception:
            return False
    return True


def trajectory_key(traj: dict) -> str:
    # Stable key to drop duplicate battle captures from reconnect/retry flows.
    actions = traj.get("actions") or []
    rewards = traj.get("rewards") or []
    dones = traj.get("dones") or []
    turns = traj.get("turns")
    rating = traj.get("rating")
    opp_rating = traj.get("opponent_rating")
    payload = {
        "a": actions,
        "r": rewards,
        "d": dones,
        "t": turns,
        "pr": rating,
        "or": opp_rating,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def rating_weight(
    player_rating: int | None,
    opp_rating: int | None,
    baseline: int,
    scale: float,
    floor: float,
    cap: float,
) -> float:
    ratings = [r for r in [player_rating, opp_rating] if isinstance(r, (int, float))]
    if not ratings:
        return 1.0
    ref = min(ratings)
    w = 1.0 + (float(ref) - float(baseline)) / float(scale)
    return max(floor, min(cap, w))


def prepare(args: argparse.Namespace) -> tuple[list[dict], dict]:
    if args.allow_forfeit:
        args.skip_forfeit = False

    input_paths = resolve_inputs(args.input)
    if not input_paths:
        raise SystemExit("No input trajectory files found. Pass --input <path|glob|dir>.")

    counters = Counter()
    by_source = Counter()
    by_tag = Counter()
    action_counts = Counter()
    kept: list[dict] = []
    seen_keys: set[str] = set()

    for path in input_paths:
        rows = load_trajectory_list(path)
        counters["input_trajectories"] += len(rows)
        by_source[str(path)] += len(rows)

        for traj in rows:
            ok, reason = validate_lengths(traj)
            if not ok:
                counters[f"drop_{reason}"] += 1
                continue

            actions = traj.get("actions") or []
            turns = int(traj.get("turns") or len(actions))
            if len(actions) < args.min_actions:
                counters["drop_min_actions"] += 1
                continue
            if turns < args.min_turns:
                counters["drop_min_turns"] += 1
                continue

            player_rating = traj.get("rating")
            opp_rating = traj.get("opponent_rating")

            if args.min_player_rating and (player_rating is None or player_rating < args.min_player_rating):
                counters["drop_player_rating"] += 1
                continue
            if args.min_opponent_rating and (opp_rating is None or opp_rating < args.min_opponent_rating):
                counters["drop_opp_rating"] += 1
                continue
            if args.min_either_rating:
                candidate = player_rating if player_rating is not None else opp_rating
                if candidate is None or candidate < args.min_either_rating:
                    counters["drop_either_rating"] += 1
                    continue

            illegal_rate = float(traj.get("illegal_rate", 0.0) or 0.0)
            if args.max_illegal_rate > 0 and illegal_rate > args.max_illegal_rate:
                counters["drop_illegal_rate"] += 1
                continue

            if args.skip_forfeit and bool(traj.get("forfeit", False)):
                counters["drop_forfeit"] += 1
                continue

            if not args.allow_invalid_mask_action and not action_mask_valid(traj):
                counters["drop_invalid_mask_action"] += 1
                continue

            if not args.disable_dedup:
                key = trajectory_key(traj)
                if key in seen_keys:
                    counters["drop_duplicate"] += 1
                    continue
                seen_keys.add(key)

            base_weight = float(traj.get("weight", 1.0) or 1.0)
            w_rating = rating_weight(
                player_rating,
                opp_rating,
                baseline=args.rating_baseline,
                scale=args.rating_scale,
                floor=args.rating_weight_floor,
                cap=args.rating_weight_cap,
            )
            w_outcome = args.win_multiplier if is_win_trajectory(traj) else args.loss_multiplier

            traj = dict(traj)
            traj["weight"] = base_weight * w_rating * w_outcome
            traj["turns"] = turns
            traj["_source_file"] = str(path)

            kept.append(traj)
            counters["kept_trajectories"] += 1
            counters["kept_steps"] += len(actions)
            by_tag[str(traj.get("tag", ""))] += 1
            action_counts.update(int(a) for a in actions)

    weights = [float(t.get("weight", 1.0)) for t in kept]
    ratings = [
        int(t.get("rating"))
        for t in kept
        if isinstance(t.get("rating"), (int, float))
    ]
    opp_ratings = [
        int(t.get("opponent_rating"))
        for t in kept
        if isinstance(t.get("opponent_rating"), (int, float))
    ]

    summary = {
        "inputs": [str(p) for p in input_paths],
        "counts": dict(counters),
        "weights": {
            "avg": mean(weights) if weights else 0.0,
            "min": min(weights) if weights else 0.0,
            "max": max(weights) if weights else 0.0,
        },
        "ratings": {
            "player_min": min(ratings) if ratings else None,
            "player_max": max(ratings) if ratings else None,
            "opp_min": min(opp_ratings) if opp_ratings else None,
            "opp_max": max(opp_ratings) if opp_ratings else None,
        },
        "tags_top": by_tag.most_common(20),
        "actions_top": action_counts.most_common(20),
        "sources": dict(by_source),
        "filters": {
            "min_actions": args.min_actions,
            "min_turns": args.min_turns,
            "min_player_rating": args.min_player_rating,
            "min_opponent_rating": args.min_opponent_rating,
            "min_either_rating": args.min_either_rating,
            "max_illegal_rate": args.max_illegal_rate,
            "skip_forfeit": args.skip_forfeit,
            "allow_invalid_mask_action": args.allow_invalid_mask_action,
            "dedup_enabled": not args.disable_dedup,
        },
    }
    return kept, summary


def main() -> int:
    args = parse_args()
    trajectories, summary = prepare(args)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        pickle.dump(trajectories, f)

    summary_path = Path(args.summary_output) if args.summary_output else output_path.with_suffix(
        output_path.suffix + ".summary.json"
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print(f"Wrote {len(trajectories)} trajectories to {output_path}")
    print(f"Summary: {summary_path}")
    counts = summary.get("counts", {})
    print(
        "Drops:",
        {k: v for k, v in counts.items() if k.startswith("drop_") and v > 0},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
