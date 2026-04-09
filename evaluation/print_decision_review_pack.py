#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Pretty-print ranked decision review rows.")
    parser.add_argument("--input", required=True, help="Review pack JSON path")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--category", default="", help="Optional issue category filter")
    args = parser.parse_args()

    with Path(args.input).open("r", encoding="utf-8") as handle:
        obj = json.load(handle)

    rows = obj.get("rows") or []
    if args.category:
        rows = [row for row in rows if str(row.get("category", "") or "") == args.category]
    rows = rows[: max(1, args.limit)]

    print(f"Rows: {len(rows)}")
    for row in rows:
        print(f"{float(row.get('priority', 0.0) or 0.0):6.1f}  {str(row.get('category', '')):32}  {row.get('battle_id')} t{row.get('turn')}  {row.get('choice')}")
        print(f"        {row.get('review_blurb', '')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
