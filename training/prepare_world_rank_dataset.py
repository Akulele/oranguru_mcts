#!/usr/bin/env python3
"""
Build a candidate-world ranking dataset from enriched Oranguru search traces.

Each output row represents one sampled hidden world from a single root decision.
The main supervision target is how much probability mass that world assigned to the
final aggregated chosen action.
"""

from __future__ import annotations

import argparse
import glob
import json
import pickle
from collections import Counter
from pathlib import Path
import sys
from typing import Iterable

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.search_assist_utils import (  # noqa: E402
    _normalize_policy_target,
    safe_float,
    validate_search_assist_example,
)


def _iter_examples(path: str) -> Iterable[dict]:
    p = Path(path)
    if p.suffix == ".jsonl":
        with p.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except Exception:
                    continue
        return

    with p.open("rb") as handle:
        obj = pickle.load(handle)
    if isinstance(obj, list):
        for item in obj:
            if isinstance(item, dict):
                yield item


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare candidate-world rank dataset.")
    parser.add_argument("--input", action="append", default=[], help="Trace pickle/JSONL path or glob. Repeatable.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-out", default="")
    parser.add_argument("--min-turn", type=int, default=1)
    parser.add_argument("--min-world-total", type=float, default=1.0)
    parser.add_argument("--keep-lowconf-only", action="store_true")
    args = parser.parse_args()

    paths: list[str] = []
    for pattern in args.input:
        matches = sorted(glob.glob(pattern))
        if matches:
            paths.extend(matches)
        elif Path(pattern).exists():
            paths.append(pattern)

    if not paths:
        raise SystemExit("No input files matched.")

    counters = Counter()
    rows: list[dict] = []
    board_dim = 0
    world_dim = 0

    for path in paths:
        for ex in _iter_examples(path):
            counters["rows_seen"] += 1
            ok, reason = validate_search_assist_example(ex)
            if not ok:
                counters[f"drop_parent_{reason}"] += 1
                continue
            turn = int(ex.get("turn", 0) or 0)
            if turn < args.min_turn:
                counters["drop_min_turn"] += 1
                continue

            confidence = safe_float(ex.get("policy_confidence", 0.0), 0.0)
            threshold = safe_float(ex.get("policy_threshold", 1.0), 1.0)
            if args.keep_lowconf_only and confidence >= threshold:
                counters["drop_not_lowconf"] += 1
                continue

            mask = [bool(v) for v in ex.get("action_mask", [])]
            final_policy = _normalize_policy_target(ex.get("visit_counts"), ex.get("policy_target"), mask)
            if final_policy is None:
                counters["drop_missing_final_policy"] += 1
                continue

            chosen_action = int(ex.get("chosen_action", -1) or -1)
            if chosen_action < 0 or chosen_action >= len(mask) or not mask[chosen_action]:
                counters["drop_bad_chosen_action"] += 1
                continue

            worlds = ex.get("world_candidates") or []
            if not isinstance(worlds, list) or not worlds:
                counters["drop_no_world_candidates"] += 1
                continue

            kept_for_parent = 0
            for world in worlds:
                if not isinstance(world, dict):
                    counters["drop_bad_world_entry"] += 1
                    continue
                world_features = world.get("world_features")
                if not isinstance(world_features, list) or not world_features:
                    counters["drop_missing_world_features"] += 1
                    continue

                world_policy = _normalize_policy_target(
                    world.get("visit_counts"),
                    world.get("policy_target"),
                    mask,
                )
                if world_policy is None:
                    counters["drop_missing_world_policy"] += 1
                    continue

                world_total = sum(max(0.0, safe_float(v, 0.0)) for v in (world.get("visit_counts") or []))
                if (world.get("visit_counts") is not None) and world_total < args.min_world_total:
                    counters["drop_min_world_total"] += 1
                    continue

                top_choice_idx = world.get("top_choice_idx")
                if top_choice_idx is None:
                    top_choice_idx = -1
                else:
                    top_choice_idx = int(top_choice_idx)

                row = {
                    "battle_id": str(ex.get("battle_id", "unknown")),
                    "turn": turn,
                    "world_index": int(world.get("index", 0) or 0),
                    "board_features": [safe_float(v, 0.0) for v in ex.get("board_features", [])],
                    "world_features": [safe_float(v, 0.0) for v in world_features],
                    "action_mask": mask,
                    "final_policy": [float(v) for v in final_policy],
                    "world_policy": [float(v) for v in world_policy],
                    "chosen_action": chosen_action,
                    "chosen_choice": str(ex.get("chosen_choice", "")),
                    "world_top_choice": str(world.get("top_choice", "")),
                    "world_top_choice_idx": top_choice_idx,
                    "world_top_choice_kind": str(world.get("top_choice_kind", "unknown")),
                    "world_top_choice_prob": safe_float(world.get("top_choice_prob", 0.0), 0.0),
                    "target_score": float(world_policy[chosen_action]),
                    "agreement_label": int(top_choice_idx == chosen_action),
                    "sample_weight": safe_float(world.get("sample_weight", 1.0), 1.0),
                    "world_total_visits": safe_float(world.get("total_visits", world_total), 0.0),
                    "policy_confidence": confidence,
                    "policy_threshold": threshold,
                    "selection_path": str(ex.get("selection_path", "")),
                    "phase": str(ex.get("phase", "mid")),
                    "source": str(ex.get("source", "")),
                    "tag": str(ex.get("tag", "")),
                    "value_target": safe_float(ex.get("value_target", 0.0), 0.0),
                }
                rows.append(row)
                kept_for_parent += 1
                counters["world_rows_kept"] += 1
                board_dim = max(board_dim, len(row["board_features"]))
                world_dim = max(world_dim, len(row["world_features"]))

            if kept_for_parent > 0:
                counters["parents_kept"] += 1

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as handle:
        pickle.dump(rows, handle)

    print(f"Wrote {len(rows)} world-rank rows -> {out_path}")
    print(f"Stats: {dict(counters)}")

    if args.summary_out:
        summary = {
            "inputs": paths,
            "output": str(out_path),
            "rows": len(rows),
            "board_dim": board_dim,
            "world_dim": world_dim,
            "counters": dict(counters),
            "min_turn": args.min_turn,
            "min_world_total": args.min_world_total,
            "keep_lowconf_only": args.keep_lowconf_only,
        }
        summary_path = Path(args.summary_out)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Summary -> {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
