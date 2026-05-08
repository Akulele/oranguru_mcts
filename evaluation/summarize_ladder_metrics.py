#!/usr/bin/env python3
"""Summarize ladder metrics JSONL logs.

Example:
  python evaluation/summarize_ladder_metrics.py logs/ladder/ladder_metrics.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


METRIC_RESULTS = {"win", "loss", "tie"}


def _is_metric_row(row: dict[str, Any]) -> bool:
    return row.get("schema_version") == 1 and str(row.get("result") or "") in METRIC_RESULTS


def _safe_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(str(row.get("result") or "unknown") for row in rows)
    residuals = [v for row in rows if (v := _safe_float(row.get("rating_residual"))) is not None]
    expected = [v for row in rows if (v := _safe_float(row.get("expected_score"))) is not None]
    deltas = [v for row in rows if (v := _safe_float(row.get("player_rating_delta"))) is not None]
    opp_pre = [v for row in rows if (v := _safe_float(row.get("opponent_rating_pre"))) is not None]
    turns = [v for row in rows if (v := _safe_float(row.get("turns"))) is not None]
    battles = len(rows)
    wins = counts.get("win", 0)
    return {
        "battles": battles,
        "wins": wins,
        "losses": counts.get("loss", 0),
        "ties": counts.get("tie", 0),
        "win_rate": wins / battles if battles else 0.0,
        "avg_expected_score": mean(expected) if expected else None,
        "avg_rating_residual": mean(residuals) if residuals else None,
        "rating_residual_sum": sum(residuals) if residuals else None,
        "rated_battles": len(residuals),
        "rating_delta_sum": sum(deltas) if deltas else None,
        "rating_delta_battles": len(deltas),
        "avg_opponent_rating_pre": mean(opp_pre) if opp_pre else None,
        "avg_turns": mean(turns) if turns else None,
    }


def _print_summary(label: str, rows: list[dict[str, Any]]) -> None:
    summary = _summarize(rows)
    print(f"\n{label}")
    print(f"  Battles: {summary['battles']}")
    print(
        "  W/L/T: {wins}/{losses}/{ties} ({win_rate:.1%})".format(**summary)
    )
    if summary["avg_opponent_rating_pre"] is not None:
        print(f"  Avg opp pre-rating: {summary['avg_opponent_rating_pre']:.1f}")
    if summary["avg_expected_score"] is not None:
        print(f"  Avg expected score: {summary['avg_expected_score']:.3f}")
    if summary["avg_rating_residual"] is not None:
        print(
            f"  Avg residual: {summary['avg_rating_residual']:+.3f} "
            f"sum={summary['rating_residual_sum']:+.2f} n={summary['rated_battles']}"
        )
    if summary["rating_delta_sum"] is not None:
        print(
            f"  Rating delta sum: {summary['rating_delta_sum']:+.0f} "
            f"n={summary['rating_delta_battles']}"
        )
    if summary["avg_turns"] is not None:
        print(f"  Avg turns: {summary['avg_turns']:.1f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="Ladder metrics JSONL file(s)")
    parser.add_argument("--by-version", action="store_true", help="Print per bot_version summary")
    parser.add_argument("--by-account", action="store_true", help="Print per account summary")
    parser.add_argument("--json-out", default="", help="Optional summary JSON output")
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    skipped = 0
    for raw in args.inputs:
        path = Path(raw)
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    row = json.loads(line)
                    if _is_metric_row(row):
                        rows.append(row)
                    else:
                        skipped += 1

    _print_summary("Overall", rows)
    if skipped:
        print(f"\nSkipped non-metric rows: {skipped}")
    grouped: dict[str, list[dict[str, Any]]] = {}
    if args.by_version:
        by_version: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_version[str(row.get("bot_version") or "unknown")].append(row)
        for key, group in sorted(by_version.items()):
            _print_summary(f"bot_version={key}", group)
        grouped["by_version"] = {key: _summarize(group) for key, group in by_version.items()}
    if args.by_account:
        by_account: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_account[str(row.get("account") or "unknown")].append(row)
        for key, group in sorted(by_account.items()):
            _print_summary(f"account={key}", group)
        grouped["by_account"] = {key: _summarize(group) for key, group in by_account.items()}

    if args.json_out:
        out = {"overall": _summarize(rows), **grouped}
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"\nSummary -> {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
