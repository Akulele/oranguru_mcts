#!/usr/bin/env python3
"""
Convert live search-trace JSONL files into a cleaned pickle dataset.
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

from training.search_assist_utils import validate_search_assist_example, safe_float


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare search-assist dataset from JSONL traces.")
    parser.add_argument("--input", action="append", default=[], help="JSONL path or glob. Repeatable.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-out", default="")
    parser.add_argument("--min-turn", type=int, default=1)
    parser.add_argument("--min-visits", type=float, default=1.0)
    parser.add_argument("--allow-zero-value", action="store_true")
    args = parser.parse_args()

    paths: list[str] = []
    for pattern in args.input:
        matches = sorted(glob.glob(pattern))
        if matches:
            paths.extend(matches)
        elif Path(pattern).exists():
            paths.append(pattern)

    if not paths:
        raise SystemExit("No input JSONL files matched.")

    counters = Counter()
    examples: list[dict] = []

    for path in paths:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                counters["rows_seen"] += 1
                try:
                    ex = json.loads(line)
                except Exception:
                    counters["drop_parse"] += 1
                    continue
                ok, reason = validate_search_assist_example(ex)
                if not ok:
                    counters[f"drop_{reason}"] += 1
                    continue
                if int(ex.get("turn", 0) or 0) < args.min_turn:
                    counters["drop_min_turn"] += 1
                    continue
                visits = ex.get("visit_counts") or []
                total_visits = sum(max(0.0, safe_float(v, 0.0)) for v in visits)
                if total_visits < args.min_visits:
                    counters["drop_min_visits"] += 1
                    continue
                value = safe_float(ex.get("value_target", 0.0), 0.0)
                if not args.allow_zero_value and abs(value) < 1e-9:
                    counters["drop_zero_value"] += 1
                    continue
                examples.append(ex)
                counters["kept"] += 1

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as handle:
        pickle.dump(examples, handle)

    print(f"Wrote {len(examples)} examples -> {out_path}")
    print(f"Stats: {dict(counters)}")

    if args.summary_out:
        summary = {
            "inputs": paths,
            "output": str(out_path),
            "examples": len(examples),
            "counters": dict(counters),
            "min_turn": args.min_turn,
            "min_visits": args.min_visits,
            "allow_zero_value": args.allow_zero_value,
        }
        summary_path = Path(args.summary_out)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Summary -> {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
