#!/usr/bin/env python3
"""
Merge multiple search-assist pickle datasets with optional per-input weight scaling.

Example:
  venv/bin/python training/merge_search_assist_pickles.py \
    --input "data/search_assist_bootstrap.pkl::0.25" \
    --input "data/search_assist_live.pkl::1.0" \
    --output data/search_assist_mixed.pkl
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
import random
import sys

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _parse_input(spec: str) -> tuple[str, float]:
    if "::" not in spec:
        return spec, 1.0
    path, scale = spec.rsplit("::", 1)
    return path, float(scale)


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge search-assist pickle datasets.")
    parser.add_argument("--input", action="append", required=True, help="PATH or PATH::WEIGHT_SCALE")
    parser.add_argument("--output", required=True)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    merged = []
    for spec in args.input:
        path, scale = _parse_input(spec)
        with open(path, "rb") as handle:
            rows = pickle.load(handle)
        if not isinstance(rows, list):
            raise SystemExit(f"Expected list pickle: {path}")
        for row in rows:
            if isinstance(row, dict):
                out = dict(row)
                out["weight"] = float(out.get("weight", 1.0)) * float(scale)
                merged.append(out)

    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(merged)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as handle:
        pickle.dump(merged, handle)

    print(f"Merged {len(merged)} examples -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
