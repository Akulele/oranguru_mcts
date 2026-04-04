#!/usr/bin/env python3
"""Report parsed replay JSON files that do not contain winner information."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path


def _winner_value(obj: dict) -> str:
    outcome = ((obj.get("metadata") or {}).get("outcome") or {})
    for key in ("winner_side", "winner_player", "winner", "winner_name"):
        value = outcome.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def main() -> int:
    parser = argparse.ArgumentParser(description="Find parsed replay files missing winner data.")
    parser.add_argument("--input-dir", default="data/showdown/parsed")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--summary-out", default="")
    args = parser.parse_args()

    paths = sorted(Path(args.input_dir).glob("*.json"))
    bad: list[str] = []
    json_failures = 0
    for path in paths:
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            json_failures += 1
            continue
        if not _winner_value(obj):
            bad.append(str(path))

    print(f"files_seen {len(paths)}")
    print(f"json_failures {json_failures}")
    print(f"no_winner_files {len(bad)}")
    for path in bad[: max(0, args.limit)]:
        print(path)

    if args.summary_out:
        summary = {
            "files_seen": len(paths),
            "json_failures": json_failures,
            "no_winner_files": len(bad),
            "sample": bad[: max(0, args.limit)],
        }
        out_path = Path(args.summary_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")
        print(f"Summary -> {args.summary_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
