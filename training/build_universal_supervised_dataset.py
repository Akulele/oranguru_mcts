#!/usr/bin/env python3
"""Build a canonical supervised dataset from parsed replay JSON.

The output is a reusable per-decision dataset intended to support future:
- policy priors
- value models
- shortlist/pruning hooks
- belief/world models via replay context

It keeps the current lightweight feature views for compatibility while also
storing richer action labels and basic replay metadata.
"""

from __future__ import annotations

import argparse
import copy
import glob
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.prepare_search_assist_bootstrap import (
    N_ACTIONS,
    _avg_pre_rating,
    _battle_text,
    _build_action_features,
    _build_action_mask,
    _build_features,
    _parse_effect_event,
    _resolve_winner_side,
)

SCHEMA_VERSION = 1


def _norm(value: str | None) -> str:
    if not value:
        return ""
    return "".join(ch.lower() for ch in str(value) if ch.isalnum())


def _build_action_labels(state: dict, side: str, mask: list[bool]) -> list[str]:
    labels = [""] * N_ACTIONS
    active_uid = state["active"].get(side)
    active_slots = state["move_slots"].get(active_uid, {}) if active_uid else {}
    slot_to_move = {int(slot): move_id for move_id, slot in active_slots.items()}
    bench = [
        uid
        for uid in state["team_order"].get(side, [])
        if uid != active_uid and (state["hp"].get(uid, 1) > 0)
    ]
    for idx in range(4):
        labels[idx] = slot_to_move.get(idx, f"move_slot_{idx + 1}")
    for idx, uid in enumerate(bench[:5], start=4):
        labels[idx] = f"switch {state['species'].get(uid, uid)}"
    for idx in range(9, 13):
        move_idx = idx - 9
        move_id = slot_to_move.get(move_idx, f"move_slot_{move_idx + 1}")
        labels[idx] = f"{move_id}-tera"
    for idx, ok in enumerate(mask):
        if not ok:
            labels[idx] = ""
    return labels


def _append_row(
    rows: list[dict],
    *,
    obj: dict,
    state: dict,
    side: str,
    action_index: int,
    turn_number: int,
    decision_index: int,
    battle_id: str,
    rating: float | None,
    winner_side: str,
    source_path: str,
    source_tag: str,
) -> None:
    if side not in ("p1", "p2"):
        return
    if action_index < 0 or action_index >= N_ACTIONS:
        return
    mask = _build_action_mask(state, side)
    if not mask[action_index]:
        mask[action_index] = True
    action_features = _build_action_features(state, side, mask)
    action_labels = _build_action_labels(state, side, mask)
    policy_target = [0.0] * N_ACTIONS
    policy_target[action_index] = 1.0
    chosen_label = action_labels[action_index]
    if action_index < 4:
        chosen_kind = "move"
    elif action_index < 9:
        chosen_kind = "switch"
    else:
        chosen_kind = "tera_move"

    outcome = copy.deepcopy((obj.get("metadata", {}) or {}).get("outcome", {}))
    total_turns = int((obj.get("metadata", {}) or {}).get("total_turns") or 0)
    rows.append(
        {
            "schema_version": SCHEMA_VERSION,
            "battle_id": battle_id,
            "format_id": str(obj.get("format_id", "")),
            "turn": int(turn_number),
            "decision_index": int(decision_index),
            "decision_key": f"{battle_id}|{side}|{int(turn_number)}|{int(decision_index)}",
            "player": side,
            "rating": rating,
            "total_turns": total_turns,
            "winner_side": winner_side,
            "value_target": 1.0 if side == winner_side else -1.0,
            "board_features": _build_features(state, side, turn_number),
            "action_features": action_features,
            "action_mask": mask,
            "action_labels": action_labels,
            "policy_target": policy_target,
            "chosen_action_index": int(action_index),
            "chosen_action_label": chosen_label,
            "chosen_action_kind": chosen_kind,
            "chosen_terastallize": bool(action_index >= 9),
            "weight": 1.0,
            "source": source_tag,
            "tag": source_tag,
            "source_path": source_path,
            "terminal_reason": str(outcome.get("terminal_reason", "") or "normal"),
            "ended_by_forfeit": bool(outcome.get("ended_by_forfeit", False)),
            "ended_by_inactivity": bool(outcome.get("ended_by_inactivity", False)),
            "metadata": {
                "players": copy.deepcopy(obj.get("players", {})),
                "outcome": outcome,
                "timestamp_unix": (obj.get("metadata", {}) or {}).get("timestamp_unix"),
                "replay_rating": (obj.get("metadata", {}) or {}).get("replay_rating"),
                "total_turns": total_turns,
            },
        }
    )


def _process_replay(obj: dict, args: argparse.Namespace, counters: Counter, source_path: str) -> list[dict]:
    turns = obj.get("turns", []) or []
    if len(turns) < args.min_turns:
        counters["skip_short_turns"] += 1
        return []

    avg_rating = _avg_pre_rating(obj)
    fallback_rating = (obj.get("metadata", {}) or {}).get("replay_rating")
    effective_rating = avg_rating if isinstance(avg_rating, (int, float)) else fallback_rating
    if args.min_rating > 0 and (effective_rating is None or float(effective_rating) < args.min_rating):
        counters["skip_low_rating"] += 1
        return []

    body = _battle_text(obj)
    if args.skip_forfeit and "forfeit" in body:
        counters["skip_forfeit"] += 1
        return []
    if args.skip_inactivity and "lost due to inactivity" in body:
        counters["skip_inactivity"] += 1
        return []

    teams = (obj.get("team_revelation", {}) or {}).get("teams", {}) or {}
    team_order = {
        side: [mon.get("pokemon_uid") for mon in teams.get(side, []) if mon.get("pokemon_uid")]
        for side in ("p1", "p2")
    }
    if not team_order["p1"] or not team_order["p2"]:
        counters["skip_missing_team"] += 1
        return []

    battle_id = str(obj.get("battle_id", "unknown"))
    players = obj.get("players", {}) or {}
    side_ratings = {
        "p1": (players.get("p1", {}) or {}).get("ladder_rating_pre") or effective_rating,
        "p2": (players.get("p2", {}) or {}).get("ladder_rating_pre") or effective_rating,
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
            state["species"][uid] = _norm(mon.get("species"))

    pending_rows: list[dict] = []
    decision_counters: dict[tuple[str, int], int] = {}
    tera_pending = {"p1": False, "p2": False}
    for turn in turns:
        turn_number = int(turn.get("turn_number") or 0)
        for event in (turn.get("events") or []):
            etype = event.get("type")

            if etype == "switch":
                side = event.get("player")
                into_uid = event.get("into_uid") or event.get("pokemon_uid")
                if side not in ("p1", "p2") or not into_uid:
                    continue
                prev_active = state["active"].get(side)
                if prev_active and into_uid != prev_active:
                    bench = [
                        uid
                        for uid in state["team_order"].get(side, [])
                        if uid != prev_active and (state["hp"].get(uid, 1) > 0)
                    ]
                    if into_uid in bench:
                        idx = bench.index(into_uid)
                        if 0 <= idx < 5:
                            key = (side, turn_number)
                            decision_index = decision_counters.get(key, 0)
                            decision_counters[key] = decision_index + 1
                            pending_rows.append(
                                {
                                    "state": copy.deepcopy(state),
                                    "side": side,
                                    "action_index": 4 + idx,
                                    "turn_number": turn_number,
                                    "decision_index": decision_index,
                                }
                            )
                state["active"][side] = into_uid
                tera_pending[side] = False
                continue

            if etype == "move":
                side = event.get("player")
                uid = event.get("pokemon_uid")
                move_id = _norm(event.get("move_id"))
                if side not in ("p1", "p2") or not uid or not move_id:
                    continue
                state["active"][side] = uid
                slots = state["move_slots"][uid]
                if move_id not in slots and len(slots) < 4:
                    slots[move_id] = len(slots)
                if move_id in slots:
                    slot = slots[move_id]
                    if 0 <= slot < 4:
                        action_index = (9 + slot) if tera_pending.get(side, False) else slot
                        key = (side, turn_number)
                        decision_index = decision_counters.get(key, 0)
                        decision_counters[key] = decision_index + 1
                        pending_rows.append(
                            {
                                "state": copy.deepcopy(state),
                                "side": side,
                                "action_index": action_index,
                                "turn_number": turn_number,
                                "decision_index": decision_index,
                            }
                        )
                if side in tera_pending:
                    tera_pending[side] = False
                continue

            if etype in {"damage", "heal"}:
                uid = event.get("target_uid")
                hp_after = event.get("hp_after")
                max_hp = event.get("max_hp")
                if uid and isinstance(hp_after, (int, float)):
                    state["hp"][uid] = int(hp_after)
                if uid and isinstance(max_hp, (int, float)) and max_hp > 0:
                    state["max_hp"][uid] = int(max_hp)
                continue

            if etype == "faint":
                uid = event.get("target_uid")
                if uid:
                    state["hp"][uid] = 0
                continue

            if etype == "status_start":
                uid = event.get("target_uid")
                if uid:
                    state["status"][uid] = True
                continue

            if etype == "status_end":
                uid = event.get("target_uid")
                if uid:
                    state["status"][uid] = False
                continue

            if etype == "effect":
                if str(event.get("effect_type", "")) == "terastallize":
                    side = event.get("player")
                    if side in ("p1", "p2"):
                        tera_pending[side] = True

            _parse_effect_event(state, event)

        tera_pending["p1"] = False
        tera_pending["p2"] = False

    winner_side = _resolve_winner_side(obj, state)
    if winner_side is None:
        counters["skip_no_winner"] += 1
        return []

    rows: list[dict] = []
    for item in pending_rows:
        _append_row(
            rows,
            obj=obj,
            state=item["state"],
            side=item["side"],
            action_index=item["action_index"],
            turn_number=item["turn_number"],
            decision_index=item["decision_index"],
            battle_id=battle_id,
            rating=side_ratings.get(item["side"]),
            winner_side=winner_side,
            source_path=source_path,
            source_tag=args.source_tag,
        )

    if not rows:
        counters["skip_no_rows"] += 1
        return []

    counters["rows_built"] += len(rows)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a universal supervised dataset from parsed replay JSON.")
    parser.add_argument("--input", nargs="+", required=True, help="Files, dirs, or globs of parsed replay JSON")
    parser.add_argument("--output", default="data/universal_supervised_gen9random.pkl")
    parser.add_argument("--summary-out", default="logs/replay_audit/universal_supervised_gen9random.summary.json")
    parser.add_argument("--source-tag", default="showdown_gen9randombattle")
    parser.add_argument("--min-turns", type=int, default=10)
    parser.add_argument("--min-rating", type=float, default=1500.0)
    parser.add_argument("--skip-forfeit", action="store_true")
    parser.add_argument("--skip-inactivity", action="store_true")
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_out).parent.mkdir(parents=True, exist_ok=True)

    input_paths: list[Path] = []
    for spec in args.input:
        path = Path(spec)
        if path.is_dir():
            input_paths.extend(sorted(path.glob("*.json")))
        elif path.exists():
            input_paths.append(path)
        else:
            input_paths.extend(Path(p) for p in sorted(glob.glob(spec)))
    seen: set[str] = set()
    unique_paths: list[Path] = []
    for path in input_paths:
        key = str(path.resolve())
        if key not in seen:
            seen.add(key)
            unique_paths.append(path)

    counters = Counter()
    rows: list[dict] = []
    kind_counter = Counter()
    for path in unique_paths:
        counters["files_seen"] += 1
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            counters["json_load_failed"] += 1
            continue
        built = _process_replay(obj, args, counters, str(path))
        rows.extend(built)
        for row in built:
            kind_counter[str(row.get("chosen_action_kind", ""))] += 1

    with open(args.output, "wb") as handle:
        pickle.dump(rows, handle)

    summary = {
        "rows": len(rows),
        "files_seen": counters.get("files_seen", 0),
        "stats": dict(counters),
        "action_kind_counts": dict(kind_counter),
        "output": args.output,
        "generated_at_unix": int(__import__("time").time()),
    }
    Path(args.summary_out).write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")
    print(f"Wrote {len(rows)} rows -> {args.output}")
    print(f"Summary -> {args.summary_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
