#!/usr/bin/env python3
"""
Merge imitation demo pickle files with optional per-input weight multipliers.

Input format:
- --input path_or_glob
- --input path_or_glob::multiplier

Examples:
  --input data/highelo_demos_gen9random.pkl
  --input data/foulplay_selfplay_*.pkl::8
"""

from __future__ import annotations

import argparse
import glob
import json
import pickle
import random
from collections import Counter
from pathlib import Path


def _expand_input(spec: str) -> tuple[list[Path], float]:
    if "::" in spec:
        path_spec, mult_s = spec.rsplit("::", 1)
        multiplier = float(mult_s)
    else:
        path_spec = spec
        multiplier = 1.0
    paths = [Path(p).resolve() for p in sorted(glob.glob(path_spec))]
    if not paths and Path(path_spec).exists():
        paths = [Path(path_spec).resolve()]
    return paths, multiplier


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        action="append",
        default=[],
        help="Input pickle path/glob, optionally with ::multiplier",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-out", default="")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-per-input",
        type=int,
        default=0,
        help="If >0, cap number of rows loaded from each input file.",
    )
    args = parser.parse_args()

    if not args.input:
        raise SystemExit("Pass at least one --input.")

    random.seed(args.seed)
    stats = Counter()
    merged: list[dict] = []
    used: list[dict] = []

    for spec in args.input:
        paths, multiplier = _expand_input(spec)
        if not paths:
            stats["missing_inputs"] += 1
            continue
        for p in paths:
            stats["files_seen"] += 1
            try:
                rows = pickle.load(p.open("rb"))
            except Exception:
                stats["file_load_error"] += 1
                continue
            if not isinstance(rows, list):
                stats["file_not_list"] += 1
                continue
            if args.max_per_input > 0 and len(rows) > args.max_per_input:
                rows = rows[: args.max_per_input]
                stats["rows_capped"] += 1

            for row in rows:
                if not isinstance(row, dict):
                    stats["rows_invalid"] += 1
                    continue
                out = dict(row)
                base_w = out.get("weight", 1.0)
                try:
                    base_w = float(base_w)
                except Exception:
                    base_w = 1.0
                out["weight"] = base_w * multiplier
                merged.append(out)
            stats["rows_loaded"] += len(rows)
            used.append(
                {
                    "path": str(p),
                    "rows": len(rows),
                    "multiplier": multiplier,
                }
            )

    if args.shuffle:
        random.shuffle(merged)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        pickle.dump(merged, f)

    summary = {
        "inputs": used,
        "output": str(out_path),
        "rows_out": len(merged),
        "stats": dict(stats),
        "shuffle": bool(args.shuffle),
        "seed": args.seed,
        "max_per_input": args.max_per_input,
    }
    summary_path = (
        Path(args.summary_out)
        if args.summary_out
        else out_path.with_suffix(out_path.suffix + ".summary.json")
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Wrote {len(merged)} rows -> {out_path}")
    print(f"Summary -> {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

