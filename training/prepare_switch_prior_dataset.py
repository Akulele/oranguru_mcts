#!/usr/bin/env python3
"""
Prepare a switch-only prior dataset from search-assist traces.

The output keeps the existing search-assist schema so it can be trained with
`training/train_search_prior_value.py`, but only legal switch slots remain in
the action mask / policy target.
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare switch-only prior training data.")
    parser.add_argument("--input", action="append", required=True, help="Pickle/JSONL path or glob.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-out", default="")
    parser.add_argument("--min-turn", type=int, default=1)
    parser.add_argument("--min-switch-mass", type=float, default=0.05)
    parser.add_argument("--require-multi-switch", action="store_true")
    parser.add_argument("--keep-passive-top1-only", action="store_true")
    args = parser.parse_args()

    paths = _iter_inputs(args.input)
    if not paths:
        raise SystemExit("No input files matched.")

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
            if args.keep_passive_top1_only and not bool(ex.get("passive_top1", False)):
                counters["drop_not_passive_top1"] += 1
                continue

            mask = [bool(v) for v in ex.get("action_mask", [])]
            if len(mask) != 13:
                counters["drop_bad_mask_len"] += 1
                continue
            switch_mask = [False] * len(mask)
            for idx in range(4, min(9, len(mask))):
                switch_mask[idx] = mask[idx]
            switch_count = sum(1 for ok in switch_mask if ok)
            if switch_count <= 0:
                counters["drop_no_switch"] += 1
                continue
            if args.require_multi_switch and switch_count < 2:
                counters["drop_single_switch"] += 1
                continue

            visit_counts = [safe_float(v, 0.0) for v in (ex.get("visit_counts") or [])]
            if len(visit_counts) != len(mask):
                counters["drop_bad_visits_len"] += 1
                continue
            switch_mass = sum(max(0.0, visit_counts[idx]) for idx in range(4, 9) if idx < len(visit_counts))
            total_mass = sum(max(0.0, v) for v in visit_counts)
            if total_mass <= 0:
                counters["drop_zero_mass"] += 1
                continue
            if switch_mass / total_mass < args.min_switch_mass:
                counters["drop_min_switch_mass"] += 1
                continue

            switch_target = [0.0] * len(mask)
            for idx in range(4, 9):
                if idx < len(mask) and switch_mask[idx]:
                    switch_target[idx] = max(0.0, visit_counts[idx])
            if sum(switch_target) <= 0:
                counters["drop_zero_switch_target"] += 1
                continue

            row = dict(ex)
            row["action_mask"] = switch_mask
            row["visit_counts"] = switch_target
            row["policy_target"] = None
            row["switch_mass"] = float(switch_mass / total_mass)
            row["weight"] = float(row.get("weight", 1.0))
            examples.append(row)
            counters["kept"] += 1

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as handle:
        pickle.dump(examples, handle)

    print(f"Wrote {len(examples)} switch-prior examples -> {out_path}")
    print(f"Stats: {dict(counters)}")

    if args.summary_out:
        payload = {
            "inputs": paths,
            "output": str(out_path),
            "examples": len(examples),
            "counters": dict(counters),
            "min_turn": args.min_turn,
            "min_switch_mass": args.min_switch_mass,
            "require_multi_switch": args.require_multi_switch,
            "keep_passive_top1_only": args.keep_passive_top1_only,
        }
        summary_path = Path(args.summary_out)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Summary -> {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
