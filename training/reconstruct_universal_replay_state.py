#!/usr/bin/env python3
"""Reconstruct and verify public replay state for universal dataset rows.

This replays parsed replay events up to a row's recorded decision boundary:
- `source_path`
- `turn`
- `decision_event_seq`

It verifies that the reconstructed public state matches the stored
`state_snapshot`. This is the prerequisite for future search relabel work.
"""

from __future__ import annotations

import argparse
import copy
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.build_universal_supervised_dataset import _state_snapshot
from training.prepare_search_assist_bootstrap import _parse_effect_event


def _init_state_from_obj(obj: dict) -> dict:
    teams = (obj.get("team_revelation", {}) or {}).get("teams", {}) or {}
    team_order = {
        side: [mon.get("pokemon_uid") for mon in teams.get(side, []) if mon.get("pokemon_uid")]
        for side in ("p1", "p2")
    }
    state = {
        "team_order": team_order,
        "uid_side": {uid: side for side in ("p1", "p2") for uid in team_order.get(side, [])},
        "active": {"p1": None, "p2": None},
        "hp": {},
        "max_hp": {},
        "status": defaultdict(bool),
        "species": {},
        "move_slots": defaultdict(dict),
        "hazards": {
            "p1": {"spikes": 0, "toxicspikes": 0, "stealthrock": 0, "stickyweb": 0},
            "p2": {"spikes": 0, "toxicspikes": 0, "stealthrock": 0, "stickyweb": 0},
        },
        "tera_used": {"p1": False, "p2": False},
    }
    for side in ("p1", "p2"):
        for mon in teams.get(side, []):
            uid = mon.get("pokemon_uid")
            if not uid:
                continue
            hp = (mon.get("base_stats", {}) or {}).get("hp")
            if isinstance(hp, (int, float)):
                state["hp"][uid] = int(hp)
                state["max_hp"][uid] = int(hp)
            species = mon.get("species")
            if species:
                state["species"][uid] = "".join(ch.lower() for ch in str(species) if ch.isalnum())
    return state


def _apply_event(state: dict, event: dict) -> None:
    etype = event.get("type")
    if etype == "switch":
        side = event.get("player")
        into_uid = event.get("into_uid") or event.get("pokemon_uid")
        if side in ("p1", "p2") and into_uid:
            state["active"][side] = into_uid
        return

    if etype == "move":
        side = event.get("player")
        uid = event.get("pokemon_uid")
        move_id = str(event.get("move_id", "") or "")
        if side in ("p1", "p2") and uid:
            state["active"][side] = uid
            if move_id:
                slots = state["move_slots"][uid]
                if move_id not in slots and len(slots) < 4:
                    slots[move_id] = len(slots)
        return

    if etype in {"damage", "heal"}:
        uid = event.get("target_uid")
        hp_after = event.get("hp_after")
        max_hp = event.get("max_hp")
        if uid and isinstance(hp_after, (int, float)):
            state["hp"][uid] = int(hp_after)
        if uid and isinstance(max_hp, (int, float)) and max_hp > 0:
            state["max_hp"][uid] = int(max_hp)
        return

    if etype == "faint":
        uid = event.get("target_uid")
        if uid:
            state["hp"][uid] = 0
        return

    if etype == "status_start":
        uid = event.get("target_uid")
        if uid:
            state["status"][uid] = True
        return

    if etype == "status_end":
        uid = event.get("target_uid")
        if uid:
            state["status"][uid] = False
        return

    _parse_effect_event(state, event)


def _event_seq(event: dict) -> int:
    value = event.get("seq", -1)
    try:
        return int(value)
    except Exception:
        return -1


def _apply_decision_prefix(state: dict, event: dict) -> None:
    """Mirror builder semantics for the stored pre-decision snapshot.

    For move decisions, the builder records state after:
    - setting the acting side's active uid
    - revealing / slotting the move
    but before downstream move effects.

    For switch decisions, the builder records state before changing the active.
    """
    if event.get("type") != "move":
        return
    side = event.get("player")
    uid = event.get("pokemon_uid")
    move_id = str(event.get("move_id", "") or "")
    if side in ("p1", "p2") and uid:
        state["active"][side] = uid
        if move_id:
            slots = state["move_slots"][uid]
            if move_id not in slots and len(slots) < 4:
                slots[move_id] = len(slots)


def _reconstruct_state(obj: dict, turn_number: int, decision_event_seq: int) -> dict:
    state = _init_state_from_obj(obj)
    for turn in obj.get("turns", []) or []:
        tnum = int(turn.get("turn_number") or 0)
        for event in (turn.get("events") or []):
            seq = _event_seq(event)
            if tnum == turn_number and seq == decision_event_seq:
                _apply_decision_prefix(state, event)
                return state
            _apply_event(state, event)
    return state


def main() -> int:
    parser = argparse.ArgumentParser(description="Reconstruct public replay state for universal dataset rows.")
    parser.add_argument("--input", required=True, help="Universal dataset pickle")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--summary-out", default="")
    args = parser.parse_args()

    with open(args.input, "rb") as handle:
        rows = pickle.load(handle)
    if not isinstance(rows, list):
        raise SystemExit(f"Expected list pickle: {args.input}")

    counters = Counter()
    mismatches: list[dict] = []
    cache: dict[str, dict] = {}

    for row in rows[: max(0, len(rows))]:
        counters["rows_seen"] += 1
        source_path = str(row.get("source_path", "") or "")
        turn = int(row.get("turn", 0) or 0)
        seq_value = row.get("decision_event_seq", -1)
        try:
            seq = int(seq_value)
        except Exception:
            seq = -1
        stored = row.get("state_snapshot")
        if not source_path or not isinstance(stored, dict):
            counters["drop_missing_source_or_snapshot"] += 1
            continue
        if seq < 0:
            counters["drop_missing_seq"] += 1
            continue

        obj = cache.get(source_path)
        if obj is None:
            try:
                obj = json.loads(Path(source_path).read_text(encoding="utf-8"))
            except Exception:
                counters["json_load_failed"] += 1
                continue
            cache[source_path] = obj

        rebuilt = _state_snapshot(_reconstruct_state(obj, turn, seq))
        if rebuilt == stored:
            counters["verified"] += 1
        else:
            counters["mismatch"] += 1
            if len(mismatches) < args.limit:
                mismatches.append(
                    {
                        "battle_id": row.get("battle_id"),
                        "decision_key": row.get("decision_key"),
                        "source_path": source_path,
                        "turn": turn,
                        "decision_event_seq": seq,
                        "decision_event_type": row.get("decision_event_type"),
                    }
                )

    print(f"rows_seen {counters['rows_seen']}")
    print(f"verified {counters['verified']}")
    print(f"mismatch {counters['mismatch']}")
    print(f"json_load_failed {counters['json_load_failed']}")
    print(f"drop_missing_source_or_snapshot {counters['drop_missing_source_or_snapshot']}")
    print(f"drop_missing_seq {counters['drop_missing_seq']}")
    for item in mismatches:
        print(json.dumps(item, ensure_ascii=True))

    if args.summary_out:
        out = {
            "input": args.input,
            "stats": dict(counters),
            "sample_mismatches": mismatches,
        }
        Path(args.summary_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary_out).write_text(json.dumps(out, ensure_ascii=True, indent=2), encoding="utf-8")
        print(f"Summary -> {args.summary_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
