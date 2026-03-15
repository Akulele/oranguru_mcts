#!/usr/bin/env python3
"""
Convert parsed Gen9 random battle replay JSON into bootstrap prior/value examples.

This produces weak search-assist supervision:
- policy target: one-hot chosen action from replay
- value target: final battle outcome from current player's perspective
- board features: coarse 272-dim replay features
- action features: simple per-slot descriptors

It is intended for warm-start training only. The main dataset should come from
live search traces with root visit counts.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.search_assist_utils import SCHEMA_VERSION


BOARD_DIM = 272
N_ACTIONS = 13
ACTION_DIM = 16


def _norm(value: str | None) -> str:
    if not value:
        return ""
    return "".join(ch.lower() for ch in str(value) if ch.isalnum())


def _avg_pre_rating(obj: dict) -> float | None:
    players = obj.get("players", {}) or {}
    p1 = (players.get("p1", {}) or {}).get("ladder_rating_pre")
    p2 = (players.get("p2", {}) or {}).get("ladder_rating_pre")
    if isinstance(p1, (int, float)) and isinstance(p2, (int, float)):
        return (float(p1) + float(p2)) / 2.0
    return None


def _battle_text(obj: dict) -> str:
    chunks: list[str] = []
    for turn in obj.get("turns", []):
        for event in turn.get("events", []):
            raw = event.get("raw_parts") or []
            if raw:
                chunks.append(" ".join(str(x) for x in raw).lower())
    return "\n".join(chunks)


def _species_hash(species: str) -> float:
    if not species:
        return 0.0
    digest = hashlib.sha1(species.encode("utf-8")).hexdigest()
    value = int(digest[:8], 16) % 997
    return value / 996.0


def _hazard_total(side_hazards: dict) -> float:
    return float(
        side_hazards.get("spikes", 0)
        + side_hazards.get("toxicspikes", 0)
        + side_hazards.get("stealthrock", 0)
        + side_hazards.get("stickyweb", 0)
    )


def _build_features(state: dict, side: str, turn_number: int) -> list[float]:
    other = "p2" if side == "p1" else "p1"
    my_active = state["active"].get(side)
    opp_active = state["active"].get(other)

    my_hp = state["hp"].get(my_active)
    my_max = state["max_hp"].get(my_active)
    opp_hp = state["hp"].get(opp_active)
    opp_max = state["max_hp"].get(opp_active)

    my_hp_frac = (
        float(my_hp) / float(my_max)
        if isinstance(my_hp, (int, float)) and isinstance(my_max, (int, float)) and my_max > 0
        else 1.0
    )
    opp_hp_frac = (
        float(opp_hp) / float(opp_max)
        if isinstance(opp_hp, (int, float)) and isinstance(opp_max, (int, float)) and opp_max > 0
        else 1.0
    )

    my_remaining = 0
    opp_remaining = 0
    for uid in state["team_order"].get(side, []):
        hp = state["hp"].get(uid)
        if not isinstance(hp, (int, float)) or hp > 0:
            my_remaining += 1
    for uid in state["team_order"].get(other, []):
        hp = state["hp"].get(uid)
        if not isinstance(hp, (int, float)) or hp > 0:
            opp_remaining += 1

    my_species = state["species"].get(my_active, "")
    opp_species = state["species"].get(opp_active, "")
    my_known_moves = len(state["move_slots"].get(my_active, {}))
    opp_known_moves = len(state["move_slots"].get(opp_active, {}))
    my_status = 1.0 if state["status"].get(my_active, False) else 0.0
    opp_status = 1.0 if state["status"].get(opp_active, False) else 0.0

    my_hazards = _hazard_total(state["hazards"][side]) / 7.0
    opp_hazards = _hazard_total(state["hazards"][other]) / 7.0

    feat = [0.0] * BOARD_DIM
    feat[0] = my_hp_frac
    feat[1] = opp_hp_frac
    feat[2] = my_hp_frac - opp_hp_frac
    feat[3] = min(float(turn_number), 60.0) / 60.0
    feat[4] = min(6.0, float(my_remaining)) / 6.0
    feat[5] = min(6.0, float(opp_remaining)) / 6.0
    feat[6] = my_status
    feat[7] = opp_status
    feat[8] = my_hazards
    feat[9] = opp_hazards
    feat[10] = _species_hash(my_species)
    feat[11] = _species_hash(opp_species)
    feat[12] = min(4.0, float(my_known_moves)) / 4.0
    feat[13] = min(4.0, float(opp_known_moves)) / 4.0
    feat[14] = 1.0 if state["tera_used"].get(side, False) else 0.0
    feat[15] = 1.0 if state["tera_used"].get(other, False) else 0.0
    return feat


def _build_action_mask(state: dict, side: str) -> list[bool]:
    mask = [False] * N_ACTIONS
    active_uid = state["active"].get(side)
    if active_uid:
        for idx in range(4):
            mask[idx] = True
        if not state["tera_used"].get(side, False):
            for idx in range(4):
                mask[9 + idx] = True
    bench = [
        uid
        for uid in state["team_order"].get(side, [])
        if uid != active_uid and (state["hp"].get(uid, 1) > 0)
    ]
    for i in range(min(5, len(bench))):
        mask[4 + i] = True
    return mask


def _build_action_features(state: dict, side: str, mask: list[bool]) -> list[list[float]]:
    other = "p2" if side == "p1" else "p1"
    active_uid = state["active"].get(side)
    opp_uid = state["active"].get(other)
    active_slots = state["move_slots"].get(active_uid, {})
    bench = [
        uid
        for uid in state["team_order"].get(side, [])
        if uid != active_uid and (state["hp"].get(uid, 1) > 0)
    ]
    opp_hp = state["hp"].get(opp_uid)
    opp_max = state["max_hp"].get(opp_uid)
    opp_hp_frac = (
        float(opp_hp) / float(opp_max)
        if isinstance(opp_hp, (int, float)) and isinstance(opp_max, (int, float)) and opp_max > 0
        else 1.0
    )
    my_hazard_load = _hazard_total(state["hazards"][side]) / 7.0

    rows: list[list[float]] = []
    for idx in range(N_ACTIONS):
        row = [0.0] * ACTION_DIM
        row[0] = 1.0 if idx < 4 else 0.0
        row[1] = 1.0 if 4 <= idx < 9 else 0.0
        row[2] = 1.0 if 9 <= idx < 13 else 0.0
        row[3] = float(idx) / float(N_ACTIONS - 1)
        row[4] = 1.0 if mask[idx] else 0.0
        row[5] = opp_hp_frac
        row[6] = my_hazard_load

        if idx < 4:
            row[7] = 1.0 if idx < len(active_slots) else 0.0
            row[8] = 1.0 if idx >= len(active_slots) else 0.0
        elif 4 <= idx < 9:
            bench_idx = idx - 4
            if bench_idx < len(bench):
                uid = bench[bench_idx]
                hp = state["hp"].get(uid)
                mx = state["max_hp"].get(uid)
                row[9] = (
                    float(hp) / float(mx)
                    if isinstance(hp, (int, float)) and isinstance(mx, (int, float)) and mx > 0
                    else 1.0
                )
                row[10] = _species_hash(state["species"].get(uid, ""))
                row[11] = 1.0
            else:
                row[11] = 0.0
        else:
            move_idx = idx - 9
            row[12] = 1.0 if move_idx < len(active_slots) else 0.0
            row[13] = 1.0 if not state["tera_used"].get(side, False) else 0.0

        row[14] = 1.0 if (idx < 4 or idx >= 9) else 0.0
        row[15] = 1.0 if (4 <= idx < 9) else 0.0
        rows.append(row)
    return rows


def _side_from_raw(token: object) -> str | None:
    if not isinstance(token, str):
        return None
    low = token.strip().lower()
    if low.startswith("p1"):
        return "p1"
    if low.startswith("p2"):
        return "p2"
    return None


def _parse_effect_event(state: dict, event: dict) -> None:
    if event.get("type") != "effect":
        return
    effect_type = _norm(event.get("effect_type"))
    raw = event.get("raw_parts") or []
    if len(raw) < 3:
        return
    side = _side_from_raw(raw[1])
    if not side:
        return
    hazards = state["hazards"][side]
    move_blob = _norm(str(raw[2]))

    if effect_type == "sidestart":
        if "stickyweb" in move_blob:
            hazards["stickyweb"] = 1
        elif "stealthrock" in move_blob:
            hazards["stealthrock"] = 1
        elif "toxicspikes" in move_blob:
            hazards["toxicspikes"] = min(2, hazards.get("toxicspikes", 0) + 1)
        elif "spikes" in move_blob:
            hazards["spikes"] = min(3, hazards.get("spikes", 0) + 1)
    elif effect_type == "sideend":
        if "stickyweb" in move_blob:
            hazards["stickyweb"] = 0
        elif "stealthrock" in move_blob:
            hazards["stealthrock"] = 0
        elif "toxicspikes" in move_blob:
            hazards["toxicspikes"] = 0
        elif "spikes" in move_blob:
            hazards["spikes"] = 0

    if effect_type == "terastallize":
        state["tera_used"][side] = True
    elif raw and isinstance(raw[0], str) and "terastallize" in raw[0].lower():
        who = _side_from_raw(raw[1]) if len(raw) > 1 else None
        if who:
            state["tera_used"][who] = True


def _resolve_winner_side(obj: dict, state: dict) -> str | None:
    players = obj.get("players", {}) or {}
    outcome = (obj.get("metadata", {}) or {}).get("outcome", {}) or {}
    candidate = outcome.get("winner_side") or outcome.get("winner")
    cand = _norm(str(candidate)) if candidate is not None else ""
    if cand in {"p1", "p2"}:
        return cand
    if cand:
        for side in ("p1", "p2"):
            entry = players.get(side, {}) or {}
            name = entry.get("name") or entry.get("username") or entry.get("id")
            if cand == _norm(name):
                return side

    alive = {}
    for side in ("p1", "p2"):
        c = 0
        for uid in state["team_order"].get(side, []):
            hp = state["hp"].get(uid)
            if not isinstance(hp, (int, float)) or hp > 0:
                c += 1
        alive[side] = c
    if alive["p1"] > alive["p2"]:
        return "p1"
    if alive["p2"] > alive["p1"]:
        return "p2"
    return None


def _append_example(examples: list[dict], state: dict, side: str, action: int, turn_number: int, battle_id: str, rating: float | None, source: str) -> None:
    if side not in ("p1", "p2"):
        return
    if not isinstance(action, int) or action < 0 or action >= N_ACTIONS:
        return
    board = _build_features(state, side, turn_number)
    mask = _build_action_mask(state, side)
    if not mask[action]:
        mask[action] = True
    policy_target = [0.0] * N_ACTIONS
    policy_target[action] = 1.0
    action_features = _build_action_features(state, side, mask)
    examples.append(
        {
            "schema_version": SCHEMA_VERSION,
            "battle_id": battle_id,
            "turn": int(turn_number),
            "rating": rating,
            "board_features": board,
            "action_features": action_features,
            "action_mask": mask,
            "policy_target": policy_target,
            "value_target": 0.0,  # filled later
            "weight": 1.0,
            "source": source,
            "tag": source,
            "player": side,
        }
    )


def _process_replay(obj: dict, args: argparse.Namespace, counters: Counter) -> list[dict]:
    turns = obj.get("turns", []) or []
    if len(turns) < args.min_turns:
        counters["skip_short_turns"] += 1
        return []

    avg_rating = _avg_pre_rating(obj)
    if args.min_avg_rating > 0 and (avg_rating is None or avg_rating < args.min_avg_rating):
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
        "p1": (players.get("p1", {}) or {}).get("ladder_rating_pre"),
        "p2": (players.get("p2", {}) or {}).get("ladder_rating_pre"),
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

    examples: list[dict] = []

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
                            _append_example(
                                examples,
                                state,
                                side,
                                4 + idx,
                                turn_number,
                                battle_id,
                                side_ratings.get(side),
                                args.source_tag,
                            )
                state["active"][side] = into_uid
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
                        _append_example(
                            examples,
                            state,
                            side,
                            slot,
                            turn_number,
                            battle_id,
                            side_ratings.get(side),
                            args.source_tag,
                        )
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

            _parse_effect_event(state, event)

    winner_side = _resolve_winner_side(obj, state)
    if winner_side is None:
        counters["skip_no_winner"] += 1
        return []

    for ex in examples:
        side = ex["player"]
        ex["value_target"] = 1.0 if side == winner_side else -1.0

    if examples:
        counters["used_files"] += 1
        counters["built_examples"] += len(examples)
    return examples


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare bootstrap search-assist dataset from gen9random replays.")
    parser.add_argument("--input-dir", default="data/gen9random")
    parser.add_argument("--output", default="data/search_assist_bootstrap.pkl")
    parser.add_argument("--summary-out", default="logs/replay_audit/search_assist_bootstrap.summary.json")
    parser.add_argument("--min-turns", type=int, default=8)
    parser.add_argument("--min-avg-rating", type=int, default=1400)
    parser.add_argument("--skip-forfeit", action="store_true")
    parser.add_argument("--skip-inactivity", action="store_true")
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--source-tag", default="human_gen9random_bootstrap")
    args = parser.parse_args()

    paths = sorted(glob.glob(str(Path(args.input_dir) / "*.json")))
    if args.max_files > 0:
        paths = paths[: args.max_files]

    counters = Counter()
    examples: list[dict] = []

    for idx, path in enumerate(paths, start=1):
        try:
            obj = json.load(open(path, "r", encoding="utf-8"))
        except Exception:
            counters["parse_errors"] += 1
            continue
        built = _process_replay(obj, args, counters)
        examples.extend(built)
        if idx % 1000 == 0:
            print(f"processed {idx}/{len(paths)} files, examples={len(examples)}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as handle:
        pickle.dump(examples, handle)

    summary = {
        "input_dir": args.input_dir,
        "output": str(out_path),
        "files_seen": len(paths),
        "examples": len(examples),
        "counters": dict(counters),
        "board_dim": BOARD_DIM,
        "action_dim": ACTION_DIM,
        "n_actions": N_ACTIONS,
        "schema_version": SCHEMA_VERSION,
        "note": "Bootstrap-only dataset from replay actions; not search visit targets.",
    }
    summary_path = Path(args.summary_out)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote {len(examples)} examples -> {out_path}")
    print(f"Summary -> {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
