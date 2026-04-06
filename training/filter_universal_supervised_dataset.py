#!/usr/bin/env python3
"""Filter/export slices from a universal supervised dataset pickle."""

from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter
from pathlib import Path


def _keep_action_kind(row: dict, allowed: set[str]) -> bool:
    if not allowed:
        return True
    return str(row.get("chosen_action_kind", "")) in allowed


def _keep_terminal_reason(row: dict, allowed: set[str]) -> bool:
    if not allowed:
        return True
    return str(row.get("terminal_reason", "") or "normal") in allowed


def main() -> int:
    parser = argparse.ArgumentParser(description="Filter a universal supervised dataset pickle.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-out", default="")
    parser.add_argument("--action-kind", action="append", default=[], help="Repeatable: move, switch, tera_move")
    parser.add_argument("--terminal-reason", action="append", default=[], help="Repeatable: normal, forfeit, inactivity")
    parser.add_argument("--min-rating", type=float, default=0.0)
    parser.add_argument("--max-rating", type=float, default=0.0)
    parser.add_argument("--min-turn", type=int, default=0)
    parser.add_argument("--max-turn", type=int, default=0)
    parser.add_argument("--min-total-turns", type=int, default=0)
    parser.add_argument("--max-total-turns", type=int, default=0)
    parser.add_argument("--only-decision-index", type=int, default=-1)
    parser.add_argument("--exclude-forfeit", action="store_true")
    parser.add_argument("--exclude-inactivity", action="store_true")
    args = parser.parse_args()

    with open(args.input, "rb") as handle:
        rows = pickle.load(handle)
    if not isinstance(rows, list):
        raise SystemExit(f"Expected list pickle: {args.input}")

    action_kinds = {str(x) for x in args.action_kind if str(x)}
    terminal_reasons = {str(x) for x in args.terminal_reason if str(x)}
    out_rows = []
    counters = Counter()
    kept_action = Counter()
    kept_terminal = Counter()

    for row in rows:
        counters["rows_seen"] += 1

        if not _keep_action_kind(row, action_kinds):
            counters["drop_action_kind"] += 1
            continue
        if not _keep_terminal_reason(row, terminal_reasons):
            counters["drop_terminal_reason"] += 1
            continue

        rating = row.get("rating")
        if args.min_rating > 0 and not isinstance(rating, (int, float)):
            counters["drop_missing_rating"] += 1
            continue
        if args.min_rating > 0 and float(rating) < args.min_rating:
            counters["drop_min_rating"] += 1
            continue
        if args.max_rating > 0 and isinstance(rating, (int, float)) and float(rating) > args.max_rating:
            counters["drop_max_rating"] += 1
            continue

        turn = int(row.get("turn", 0) or 0)
        if args.min_turn > 0 and turn < args.min_turn:
            counters["drop_min_turn"] += 1
            continue
        if args.max_turn > 0 and turn > args.max_turn:
            counters["drop_max_turn"] += 1
            continue

        total_turns = int(row.get("total_turns", 0) or 0)
        if args.min_total_turns > 0 and total_turns < args.min_total_turns:
            counters["drop_min_total_turns"] += 1
            continue
        if args.max_total_turns > 0 and total_turns > args.max_total_turns:
            counters["drop_max_total_turns"] += 1
            continue

        decision_index = int(row.get("decision_index", 0) or 0)
        if args.only_decision_index >= 0 and decision_index != args.only_decision_index:
            counters["drop_decision_index"] += 1
            continue

        if args.exclude_forfeit and bool(row.get("ended_by_forfeit", False)):
            counters["drop_forfeit"] += 1
            continue
        if args.exclude_inactivity and bool(row.get("ended_by_inactivity", False)):
            counters["drop_inactivity"] += 1
            continue

        out_rows.append(row)
        counters["kept"] += 1
        kept_action[str(row.get("chosen_action_kind", ""))] += 1
        kept_terminal[str(row.get("terminal_reason", "") or "normal")] += 1

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as handle:
        pickle.dump(out_rows, handle)

    summary = {
        "input": args.input,
        "output": args.output,
        "rows": len(out_rows),
        "stats": dict(counters),
        "action_kind_counts": dict(kept_action),
        "terminal_reason_counts": dict(kept_terminal),
        "filters": {
            "action_kind": sorted(action_kinds),
            "terminal_reason": sorted(terminal_reasons),
            "min_rating": args.min_rating,
            "max_rating": args.max_rating,
            "min_turn": args.min_turn,
            "max_turn": args.max_turn,
            "min_total_turns": args.min_total_turns,
            "max_total_turns": args.max_total_turns,
            "only_decision_index": args.only_decision_index,
            "exclude_forfeit": bool(args.exclude_forfeit),
            "exclude_inactivity": bool(args.exclude_inactivity),
        },
    }

    print(f"Wrote {len(out_rows)} rows -> {args.output}")
    print(f"Stats: {dict(counters)}")
    print(f"Action kinds: {dict(kept_action)}")
    print(f"Terminal reasons: {dict(kept_terminal)}")

    if args.summary_out:
        Path(args.summary_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary_out).write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")
        print(f"Summary -> {args.summary_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
