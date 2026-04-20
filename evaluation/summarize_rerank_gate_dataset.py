#!/usr/bin/env python3
"""Summarize rerank-gate JSONL datasets.

This avoids shell heredocs for inspecting the dataset produced by
`evaluation/build_rerank_gate_dataset.py`.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Iterable


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _iter_rows(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
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


def _score_drop_bucket(value: float) -> str:
    if value <= 0.02:
        return "(0,0.02]"
    if value <= 0.05:
        return "(0.02,0.05]"
    if value <= 0.10:
        return "(0.05,0.10]"
    if value <= 0.20:
        return "(0.10,0.20]"
    return ">0.20"


def _heuristic_bucket(value: float) -> str:
    if value < 5.0:
        return "<5"
    if value < 25.0:
        return "[5,25)"
    if value < 60.0:
        return "[25,60)"
    if value < 100.0:
        return "[60,100)"
    return ">=100"


def _add(bucket: dict, key: object, label: int) -> None:
    wins, rows = bucket[key]
    bucket[key] = [wins + label, rows + 1]


def _print_rate_table(title: str, bucket: dict, *, min_count: int = 1, limit: int = 0) -> None:
    print(title)
    rows = [
        (key, wins, count)
        for key, (wins, count) in bucket.items()
        if count >= min_count
    ]
    rows.sort(key=lambda item: (-item[2], str(item[0])))
    if limit > 0:
        rows = rows[:limit]
    if not rows:
        print("  none")
        return
    for key, wins, count in rows:
        rate = 100.0 * wins / max(1, count)
        print(f"  {str(key):80s} n={count:4d} win_label%={rate:5.1f}")


def summarize(path: Path, *, min_count: int = 1, top_buckets: int = 80) -> dict:
    by_source: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    by_source_kind: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    by_bucket: dict[tuple[str, str, str], list[int]] = defaultdict(lambda: [0, 0])
    label_counts = defaultdict(int)
    total = 0

    for row in _iter_rows(path):
        total += 1
        label = int(row.get("label", 0) or 0)
        label = 1 if label > 0 else 0
        label_counts[str(label)] += 1
        source = str(row.get("source", "unknown") or "unknown")
        kind = str(row.get("candidate_kind", "unknown") or "unknown")
        features = row.get("features") if isinstance(row.get("features"), dict) else {}
        score_drop = _score_drop_bucket(_safe_float(features.get("score_drop_top1_minus_candidate", 0.0)))
        heur_delta = _heuristic_bucket(_safe_float(features.get("heuristic_delta_candidate_minus_top1", 0.0)))

        _add(by_source, source, label)
        _add(by_source_kind, f"{source}/{kind}", label)
        _add(by_bucket, (source, score_drop, heur_delta), label)

    summary = {
        "rows": total,
        "labels": dict(label_counts),
        "positive_rate": label_counts["1"] / max(1, total),
    }

    print(f"Rows: {total}")
    print(f"Labels: {dict(label_counts)}")
    print(f"Positive label rate: {summary['positive_rate'] * 100.0:.1f}%")
    print()
    _print_rate_table("By source:", by_source, min_count=min_count)
    print()
    _print_rate_table("By source/kind:", by_source_kind, min_count=min_count)
    print()
    pretty_bucket = {str(key): value for key, value in by_bucket.items()}
    _print_rate_table(
        "By source / score-drop bucket / heuristic-delta bucket:",
        pretty_bucket,
        min_count=min_count,
        limit=top_buckets,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize rerank-gate JSONL label rates.")
    parser.add_argument("input", help="Rerank-gate JSONL from build_rerank_gate_dataset.py")
    parser.add_argument("--min-count", type=int, default=1, help="Minimum rows for displayed groups")
    parser.add_argument("--top-buckets", type=int, default=80, help="Maximum bucket rows to print")
    parser.add_argument("--json-out", default="", help="Optional compact JSON summary output")
    args = parser.parse_args()

    summary = summarize(Path(args.input), min_count=max(1, args.min_count), top_buckets=max(0, args.top_buckets))
    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Summary -> {out_path}")


if __name__ == "__main__":
    main()
