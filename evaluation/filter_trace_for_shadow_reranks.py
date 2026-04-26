#!/usr/bin/env python3
"""Filter search traces down to states with shadow tactical take candidates."""

from __future__ import annotations

import argparse
from collections import Counter
import glob
import json
from pathlib import Path
import sys
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.players.oranguru_rerank_gate import RERANK_WINDOWS, TAKE_TARGET_KEYS, action_for_choice, top_actions


def _resolve_inputs(patterns: list[str]) -> list[str]:
    paths: list[str] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            paths.extend(matches)
        elif Path(pattern).exists():
            paths.append(pattern)
    return paths


def _iter_rows(paths: Iterable[str]):
    for path in paths:
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                if isinstance(row, dict):
                    yield row


def _take_sources(row: dict) -> list[str]:
    actions = top_actions(row)
    if not actions:
        return []
    top1_choice = str(actions[0].get("choice", "") or "")
    if not top1_choice:
        return []
    sources: list[str] = []
    for source, key in RERANK_WINDOWS:
        payload = row.get(key)
        if not isinstance(payload, dict):
            continue
        reason = str(payload.get("reason", "") or "")
        if not reason.startswith("take_"):
            continue
        for target_key in TAKE_TARGET_KEYS:
            candidate_choice = str(payload.get(target_key, "") or "")
            if not candidate_choice or candidate_choice == top1_choice:
                continue
            if action_for_choice(row, candidate_choice) is None:
                continue
            sources.append(f"{source}:{reason}")
    return sources


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="Trace JSONL paths or glob patterns")
    parser.add_argument("--output", required=True, help="Filtered trace JSONL output")
    parser.add_argument("--source", action="append", default=[], help="Optional source filter; repeatable")
    parser.add_argument("--limit", type=int, default=0, help="Maximum rows to write; 0 means no limit")
    parser.add_argument("--summary-out", default="", help="Optional JSON summary path")
    args = parser.parse_args()

    paths = _resolve_inputs(args.inputs)
    if not paths:
        raise SystemExit("No input traces matched.")

    source_filter = set(args.source)
    counters = Counter()
    by_source = Counter()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in _iter_rows(paths):
            counters["rows_seen"] += 1
            sources = _take_sources(row)
            if not sources:
                counters["drop_no_take_candidate"] += 1
                continue
            if source_filter and not any(source in source_filter for source in sources):
                counters["drop_source_filter"] += 1
                continue
            for source in sources:
                by_source[source] += 1
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            counters["rows_written"] += 1
            if args.limit and counters["rows_written"] >= args.limit:
                break

    summary = {
        "inputs": paths,
        "output": str(output),
        "counters": dict(counters),
        "by_source": dict(by_source),
        "source_filter": sorted(source_filter),
    }
    print(f"Inputs: {len(paths)}")
    print(f"Rows seen: {counters['rows_seen']}")
    print(f"Rows written: {counters['rows_written']}")
    print(f"By source: {dict(by_source)}")
    print(f"Output -> {output}")
    if args.summary_out:
        summary_path = Path(args.summary_out)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Summary -> {summary_path}")


if __name__ == "__main__":
    main()
