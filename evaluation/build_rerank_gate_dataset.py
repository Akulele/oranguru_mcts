#!/usr/bin/env python3
"""Build a weakly supervised dataset for tactical rerank gate experiments.

The dataset is intentionally about *supporting* MCTS: rows are accepted reranks
where the candidate displaced the MCTS top action.  The label is battle outcome,
so it is noisy, but useful for calibrating which rerank sources and score-drop
profiles are worth trusting.
"""

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

from src.players.oranguru_rerank_gate import build_trace_rerank_gate_example


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


def build_dataset(paths: list[str], *, sources: set[str] | None = None, limit: int = 0) -> tuple[list[dict], dict]:
    examples: list[dict] = []
    seen = 0
    skipped = Counter()
    labels = Counter()
    by_source = Counter()
    for row in _iter_rows(paths):
        seen += 1
        example = build_trace_rerank_gate_example(row)
        if example is None:
            skipped["not_training_example"] += 1
            continue
        if sources and example["source"] not in sources:
            skipped["source_filter"] += 1
            continue
        examples.append(example)
        labels[str(example["label"])] += 1
        by_source[str(example["source"])] += 1
        if limit and len(examples) >= limit:
            break
    summary = {
        "rows_seen": seen,
        "examples": len(examples),
        "labels": dict(labels),
        "by_source": dict(by_source),
        "skipped": dict(skipped),
    }
    return examples, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build rerank gate JSONL examples from search traces.")
    parser.add_argument("inputs", nargs="+", help="Trace JSONL paths or glob patterns")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        help="Optional rerank source filter, e.g. setup_window:take_setup. Repeatable.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Maximum examples to write; 0 means no limit")
    parser.add_argument("--summary-out", default="", help="Optional JSON summary path")
    args = parser.parse_args()

    paths = _resolve_inputs(args.inputs)
    if not paths:
        raise SystemExit("No input traces matched.")

    examples, summary = build_dataset(paths, sources=set(args.source) if args.source else None, limit=args.limit)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example, sort_keys=True) + "\n")

    if args.summary_out:
        summary_path = Path(args.summary_out)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"Inputs: {len(paths)}")
    print(f"Rows seen: {summary['rows_seen']}")
    print(f"Examples: {summary['examples']}")
    print(f"Labels: {summary['labels']}")
    print(f"By source: {summary['by_source']}")
    print(f"Output -> {out_path}")
    if args.summary_out:
        print(f"Summary -> {args.summary_out}")


if __name__ == "__main__":
    main()
