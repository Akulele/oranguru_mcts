#!/usr/bin/env python3
"""
Prepare a passive-breaker dataset from enriched search traces.

This keeps only turns where the MCTS top-1 action is passive and rewrites the
policy target to the top-k root actions so the model learns to break passive
lines only within a narrow candidate set.
"""

from __future__ import annotations

import argparse
import glob
import json
import pickle
from collections import Counter
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.search_assist_utils import (
    load_search_assist_examples,
    safe_float,
    validate_search_assist_example,
)


PASSIVE_KINDS = {"protect", "recovery", "status", "setup"}
ACTIVE_KINDS = {"attack", "tera_attack", "switch"}


def _iter_inputs(patterns: list[str]) -> list[str]:
    paths: list[str] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            paths.extend(matches)
        elif Path(pattern).exists():
            paths.append(pattern)
    return paths


def _load_rows(path: str) -> list[dict]:
    p = Path(path)
    if p.suffix == ".pkl":
        return load_search_assist_examples(path)
    rows: list[dict] = []
    with p.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _choice_to_idx(ex: dict, choice: str) -> int | None:
    top_actions = ex.get("top_actions") or []
    for row in top_actions:
        if row.get("choice") == choice and isinstance(row.get("choice_idx"), int):
            return int(row["choice_idx"])
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare passive-breaker training data.")
    parser.add_argument("--input", action="append", required=True, help="Pickle/JSONL path or glob.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-out", default="")
    parser.add_argument("--min-turn", type=int, default=1)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--min-active-mass", type=float, default=0.10)
    parser.add_argument("--phase", action="append", default=[], help="Optional allowed phases.")
    args = parser.parse_args()

    paths = _iter_inputs(args.input)
    if not paths:
        raise SystemExit("No input files matched.")

    allowed_phases = {p.strip().lower() for p in args.phase if p.strip()}
    topk = max(2, args.topk)

    counters = Counter()
    examples: list[dict] = []

    for path in paths:
        for ex in _load_rows(path):
            counters["rows_seen"] += 1
            ok, reason = validate_search_assist_example(ex)
            if not ok:
                counters[f"drop_{reason}"] += 1
                continue
            if int(ex.get("turn", 0) or 0) < args.min_turn:
                counters["drop_min_turn"] += 1
                continue
            phase = str(ex.get("phase", "") or "").lower()
            if allowed_phases and phase not in allowed_phases:
                counters["drop_phase"] += 1
                continue
            if not bool(ex.get("passive_top1", False)):
                counters["drop_not_passive_top1"] += 1
                continue

            top_actions = ex.get("top_actions") or []
            if not isinstance(top_actions, list) or len(top_actions) < 2:
                counters["drop_top_actions"] += 1
                continue

            top_actions = top_actions[:topk]
            active_choices = [row for row in top_actions if str(row.get("kind", "")).lower() in ACTIVE_KINDS]
            if not active_choices:
                counters["drop_no_active_choice"] += 1
                continue

            mask = [bool(v) for v in ex.get("action_mask", [])]
            visit_counts = [safe_float(v, 0.0) for v in (ex.get("visit_counts") or [])]
            if len(mask) != 13 or len(visit_counts) != 13:
                counters["drop_bad_dims"] += 1
                continue

            filtered_mask = [False] * len(mask)
            filtered_target = [0.0] * len(mask)
            candidate_mass = 0.0
            active_mass = 0.0
            for row in top_actions:
                choice = str(row.get("choice", "") or "")
                idx = _choice_to_idx(ex, choice)
                if idx is None or idx < 0 or idx >= len(mask) or not mask[idx]:
                    continue
                filtered_mask[idx] = True
                filtered_target[idx] = max(0.0, visit_counts[idx])
                candidate_mass += filtered_target[idx]
                if str(row.get("kind", "")).lower() in ACTIVE_KINDS:
                    active_mass += filtered_target[idx]

            if candidate_mass <= 0:
                counters["drop_zero_candidate_mass"] += 1
                continue
            if active_mass / candidate_mass < args.min_active_mass:
                counters["drop_min_active_mass"] += 1
                continue

            policy_target = [
                (v / candidate_mass) if filtered_mask[i] else 0.0
                for i, v in enumerate(filtered_target)
            ]
            row = dict(ex)
            row["action_mask"] = filtered_mask
            row["visit_counts"] = None
            row["policy_target"] = policy_target
            row["candidate_mass"] = float(candidate_mass)
            row["active_mass_ratio"] = float(active_mass / candidate_mass)
            examples.append(row)
            counters["kept"] += 1

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as handle:
        pickle.dump(examples, handle)

    print(f"Wrote {len(examples)} passive-break examples -> {out_path}")
    print(f"Stats: {dict(counters)}")

    if args.summary_out:
        payload = {
            "inputs": paths,
            "output": str(out_path),
            "examples": len(examples),
            "counters": dict(counters),
            "min_turn": args.min_turn,
            "topk": topk,
            "min_active_mass": args.min_active_mass,
            "allowed_phases": sorted(allowed_phases),
        }
        summary_path = Path(args.summary_out)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Summary -> {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
