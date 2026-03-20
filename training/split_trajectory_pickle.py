#!/usr/bin/env python3
"""
Split trajectory pickle into train/val files.

Uses battle-id hashing first; if that yields empty validation, falls back to
random per-trajectory split.
"""

from __future__ import annotations

import argparse
import pickle
import random
from pathlib import Path

from training.sequence_utils import stable_split_by_battle


def main() -> int:
    parser = argparse.ArgumentParser(description="Split trajectory pickle into train/val.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--train-output", required=True)
    parser.add_argument("--val-output", required=True)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.input, "rb") as handle:
        rows = pickle.load(handle)
    if not isinstance(rows, list):
        raise SystemExit(f"Expected list pickle: {args.input}")

    train, val = stable_split_by_battle(rows, holdout_ratio=args.val_ratio, seed=args.seed)
    if not val and train and args.val_ratio > 0:
        rng = random.Random(args.seed)
        idx = list(range(len(train)))
        rng.shuffle(idx)
        val_n = max(1, int(len(train) * args.val_ratio))
        val_ids = set(idx[:val_n])
        val = [t for i, t in enumerate(train) if i in val_ids]
        train = [t for i, t in enumerate(train) if i not in val_ids]
        print(
            f"[warn] battle-id split produced empty val; fallback random split "
            f"(train={len(train)} val={len(val)})"
        )

    Path(args.train_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.val_output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.train_output, "wb") as handle:
        pickle.dump(train, handle)
    with open(args.val_output, "wb") as handle:
        pickle.dump(val, handle)

    print(f"Wrote train={len(train)} -> {args.train_output}")
    print(f"Wrote val={len(val)} -> {args.val_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

