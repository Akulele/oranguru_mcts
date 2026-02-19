#!/usr/bin/env python3
"""
Build imitation-learning demos from parsed Gen9 random battle replay JSON files.

Output format matches training/train_replay_imitation.py expectations:
[
  {
    "features": [float x 272],
    "action": int (0..12),
    "mask": [bool x 13],
    "rating": int | None,
    "weight": float,
    "battle_id": str,
    "player": "p1" | "p2",
  },
  ...
]
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path

FEATURE_DIM = 272
N_ACTIONS = 13


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


def _side_from_raw(token: object) -> str | None:
    if not isinstance(token, str):
        return None
    raw = token.strip().lower()
    if raw.startswith("p1"):
        return "p1"
    if raw.startswith("p2"):
        return "p2"
    return None


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

    my_hp_frac = float(my_hp) / float(my_max) if isinstance(my_hp, (int, float)) and isinstance(my_max, (int, float)) and my_max > 0 else 1.0
    opp_hp_frac = float(opp_hp) / float(opp_max) if isinstance(opp_hp, (int, float)) and isinstance(opp_max, (int, float)) and opp_max > 0 else 1.0

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

    feat = [0.0] * FEATURE_DIM
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


def _parse_hazard_event(state: dict, event: dict) -> None:
    if event.get("type") != "effect":
        return
    raw = event.get("raw_parts") or []
    if len(raw) < 3:
        return
    effect_type = _norm(event.get("effect_type"))
    side = _side_from_raw(raw[1])
    move_blob = _norm(str(raw[2]))
    if not side:
        return
    hazards = state["hazards"][side]

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

    if raw and isinstance(raw[0], str) and "terastallize" in raw[0].lower():
        who = _side_from_raw(raw[1]) if len(raw) > 1 else None
        if who:
            state["tera_used"][who] = True


def _process_replay(obj: dict, args: argparse.Namespace, counters: Counter) -> list[dict]:
    players = obj.get("players", {}) or {}
    turns = obj.get("turns", []) or []
    if len(turns) < args.min_turns:
        counters["skip_short"] += 1
        return []

    avg_rating = _avg_pre_rating(obj)
    if args.min_avg_rating > 0 and (avg_rating is None or avg_rating < args.min_avg_rating):
        counters["skip_low_rating"] += 1
        return []

    text = _battle_text(obj)
    if args.skip_forfeit and "forfeit" in text:
        counters["skip_forfeit"] += 1
        return []
    if args.skip_inactivity and "lost due to inactivity" in text:
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

    state = {
        "team_order": team_order,
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

    side_ratings = {
        "p1": (players.get("p1", {}) or {}).get("ladder_rating_pre"),
        "p2": (players.get("p2", {}) or {}).get("ladder_rating_pre"),
    }
    battle_id = str(obj.get("battle_id", "unknown"))
    demos: list[dict] = []

    for turn in turns:
        turn_number = int(turn.get("turn_number") or 0)
        events = turn.get("events", []) or []
        for event in events:
            etype = event.get("type")
            if etype == "switch":
                side = event.get("player")
                into_uid = event.get("into_uid") or event.get("pokemon_uid")
                if side not in ("p1", "p2") or not into_uid:
                    continue
                prev_active = state["active"].get(side)
                if prev_active and into_uid != prev_active:
                    bench = [
                        uid for uid in state["team_order"].get(side, [])
                        if uid != prev_active and (state["hp"].get(uid, 1) > 0)
                    ]
                    if into_uid in bench:
                        idx = bench.index(into_uid)
                        if 0 <= idx < 5:
                            demos.append(
                                {
                                    "features": _build_features(state, side, turn_number),
                                    "action": 4 + idx,
                                    "mask": [True] * N_ACTIONS,
                                    "rating": side_ratings.get(side),
                                    "weight": args.win_weight,
                                    "battle_id": battle_id,
                                    "player": side,
                                }
                            )
                state["active"][side] = into_uid
            elif etype == "move":
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
                        demos.append(
                            {
                                "features": _build_features(state, side, turn_number),
                                "action": slot,
                                "mask": [True] * N_ACTIONS,
                                "rating": side_ratings.get(side),
                                "weight": args.win_weight,
                                "battle_id": battle_id,
                                "player": side,
                            }
                        )
            elif etype == "damage" or etype == "heal":
                uid = event.get("target_uid")
                hp_after = event.get("hp_after")
                max_hp = event.get("max_hp")
                if uid and isinstance(hp_after, (int, float)):
                    state["hp"][uid] = int(hp_after)
                if uid and isinstance(max_hp, (int, float)) and max_hp > 0:
                    state["max_hp"][uid] = int(max_hp)
            elif etype == "faint":
                uid = event.get("target_uid")
                if uid:
                    state["hp"][uid] = 0
            elif etype == "status_start":
                uid = event.get("target_uid")
                if uid:
                    state["status"][uid] = True
            elif etype == "status_end":
                uid = event.get("target_uid")
                if uid:
                    state["status"][uid] = False

            _parse_hazard_event(state, event)

    if not demos:
        counters["skip_no_demos"] += 1
        return []

    counters["used_files"] += 1
    counters["demos"] += len(demos)
    return demos


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare Gen9 random replay demos.")
    parser.add_argument("--input-dir", default="data/gen9random")
    parser.add_argument("--output", default="data/highelo_demos_gen9random.pkl")
    parser.add_argument("--summary-out", default="")
    parser.add_argument("--min-turns", type=int, default=8)
    parser.add_argument("--min-avg-rating", type=int, default=1400)
    parser.add_argument("--skip-forfeit", action="store_true")
    parser.add_argument("--skip-inactivity", action="store_true")
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--max-demos", type=int, default=0)
    parser.add_argument("--win-weight", type=float, default=1.0)
    args = parser.parse_args()

    paths = sorted(glob.glob(str(Path(args.input_dir) / "*.json")))
    if args.max_files > 0:
        paths = paths[: args.max_files]
    if not paths:
        raise SystemExit(f"No replay JSON files found in {args.input_dir}")

    counters = Counter()
    all_demos: list[dict] = []
    for idx, path in enumerate(paths, start=1):
        counters["files_seen"] += 1
        try:
            with open(path, "r", encoding="utf-8") as handle:
                obj = json.load(handle)
        except Exception:
            counters["parse_error"] += 1
            continue
        demos = _process_replay(obj, args, counters)
        if demos:
            all_demos.extend(demos)
        if args.max_demos > 0 and len(all_demos) >= args.max_demos:
            all_demos = all_demos[: args.max_demos]
            break
        if idx % 1000 == 0:
            print(f"processed {idx}/{len(paths)} files, demos={len(all_demos)}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as handle:
        pickle.dump(all_demos, handle)

    summary = {
        "input_dir": args.input_dir,
        "output": str(out_path),
        "files_seen": counters["files_seen"],
        "used_files": counters["used_files"],
        "demos": len(all_demos),
        "skips": {k: v for k, v in counters.items() if k.startswith("skip_") and v > 0},
        "parse_error": counters["parse_error"],
        "filters": {
            "min_turns": args.min_turns,
            "min_avg_rating": args.min_avg_rating,
            "skip_forfeit": bool(args.skip_forfeit),
            "skip_inactivity": bool(args.skip_inactivity),
        },
    }

    summary_path = Path(args.summary_out) if args.summary_out else out_path.with_suffix(out_path.suffix + ".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    print(f"Wrote {len(all_demos)} demos -> {out_path}")
    print(f"Summary -> {summary_path}")
    print(f"Skips: {summary['skips']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

