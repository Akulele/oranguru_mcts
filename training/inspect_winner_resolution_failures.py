#!/usr/bin/env python3
"""Inspect parsed replay files that fail winner-side resolution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.check_winner_resolution import _build_min_state
from training.prepare_search_assist_bootstrap import _norm, _resolve_winner_side


def _last_event_rows(obj: dict, limit: int) -> list[dict]:
    rows: list[dict] = []
    for turn in obj.get("turns", []) or []:
        turn_number = turn.get("turn_number")
        for event in turn.get("events", []) or []:
            rows.append(
                {
                    "turn": turn_number,
                    "type": event.get("type"),
                    "player": event.get("player"),
                    "raw_parts": event.get("raw_parts"),
                }
            )
    return rows[-max(0, limit):]


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect parsed replay files with unresolved winners.")
    parser.add_argument("--input-dir", default="data/showdown/parsed")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--tail-events", type=int, default=8)
    args = parser.parse_args()

    paths = sorted(Path(args.input_dir).glob("*.json"))
    shown = 0
    bad = 0
    for path in paths:
        obj = json.loads(path.read_text(encoding="utf-8"))
        state = _build_min_state(obj)
        winner = _resolve_winner_side(obj, state)
        if winner is not None:
            continue
        bad += 1
        if shown >= args.limit:
            continue
        shown += 1
        outcome = ((obj.get("metadata") or {}).get("outcome") or {})
        players = obj.get("players") or {}
        p1_name = ((players.get("p1") or {}).get("name") or "")
        p2_name = ((players.get("p2") or {}).get("name") or "")
        print(f"PATH {path}")
        print(f"  outcome: {json.dumps(outcome, ensure_ascii=True)}")
        print(f"  p1_name: {p1_name!r} norm={_norm(p1_name)!r}")
        print(f"  p2_name: {p2_name!r} norm={_norm(p2_name)!r}")
        print(f"  metadata keys: {list((obj.get('metadata') or {}).keys())}")
        print("  last events:")
        for row in _last_event_rows(obj, args.tail_events):
            print(f"    turn={row['turn']} type={row['type']} player={row['player']} raw={row['raw_parts']}")
        print()

    print(f"resolve_bad_count {bad}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
