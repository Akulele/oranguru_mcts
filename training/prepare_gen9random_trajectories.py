#!/usr/bin/env python3
"""
Build sequence trajectories from parsed Gen9 random battle replay JSON files.

Output trajectory schema:
{
  "battle_id": str,
  "player": "p1" | "p2",
  "rating": int | None,
  "features": List[List[float]],   # [T, 272]
  "masks": List[List[bool]],       # [T, 13]
  "actions": List[int],            # [T]
  "rewards": List[float],          # [T]
  "dones": List[bool],             # [T]
  "prev_actions": List[int],       # [T]
  "prev_rewards": List[float],     # [T]
  "weight": float,
  "tag": str,
  "source": str,
}
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

from training.sequence_utils import (
    DEFAULT_FEATURE_DIM,
    DEFAULT_N_ACTIONS,
    stable_split_by_battle,
)


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
        if isinstance(my_hp, (int, float))
        and isinstance(my_max, (int, float))
        and my_max > 0
        else 1.0
    )
    opp_hp_frac = (
        float(opp_hp) / float(opp_max)
        if isinstance(opp_hp, (int, float))
        and isinstance(opp_max, (int, float))
        and opp_max > 0
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

    feat = [0.0] * DEFAULT_FEATURE_DIM
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
    mask = [False] * DEFAULT_N_ACTIONS
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


def _player_display_name(players: dict, side: str) -> str:
    entry = players.get(side, {}) or {}
    for key in ("name", "username", "id", "player"):
        val = entry.get(key)
        if isinstance(val, str) and val.strip():
            return val
    return side


def _resolve_winner_side(obj: dict, state: dict) -> str | None:
    players = obj.get("players", {}) or {}
    candidate = None
    outcome = (obj.get("metadata", {}) or {}).get("outcome", {}) or {}
    for key in ("winner_side", "winner", "winning_side"):
        if outcome.get(key):
            candidate = outcome.get(key)
            break
    if candidate is None:
        for key in ("winner_side", "winner", "winning_side"):
            if obj.get(key):
                candidate = obj.get(key)
                break

    cand = _norm(str(candidate)) if candidate is not None else ""
    if cand in {"p1", "p2"}:
        return cand
    if cand:
        for side in ("p1", "p2"):
            if cand == _norm(_player_display_name(players, side)):
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


def _append_step(
    trajectories: dict[str, dict],
    last_idx: dict[str, int | None],
    state: dict,
    side: str,
    action: int,
    turn_number: int,
) -> None:
    if side not in ("p1", "p2"):
        return
    if not isinstance(action, int) or action < 0 or action >= DEFAULT_N_ACTIONS:
        return
    traj = trajectories[side]
    feats = _build_features(state, side, turn_number)
    mask = _build_action_mask(state, side)
    if not mask[action]:
        mask[action] = True

    prev_action = traj["actions"][-1] if traj["actions"] else -1
    prev_reward = traj["rewards"][-1] if traj["rewards"] else 0.0

    traj["features"].append(feats)
    traj["masks"].append(mask)
    traj["actions"].append(action)
    traj["rewards"].append(0.0)
    traj["prev_actions"].append(prev_action)
    traj["prev_rewards"].append(float(prev_reward))
    traj["turns"].append(int(turn_number))
    last_idx[side] = len(traj["actions"]) - 1


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

    trajectories = {
        side: {
            "battle_id": battle_id,
            "player": side,
            "rating": side_ratings.get(side),
            "features": [],
            "masks": [],
            "actions": [],
            "rewards": [],
            "dones": [],
            "prev_actions": [],
            "prev_rewards": [],
            "turns": [],
            "tag": args.source_tag,
            "source": args.source_tag,
        }
        for side in ("p1", "p2")
    }
    last_idx: dict[str, int | None] = {"p1": None, "p2": None}

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
                            _append_step(
                                trajectories, last_idx, state, side, 4 + idx, turn_number
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
                        _append_step(trajectories, last_idx, state, side, slot, turn_number)
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
                    fainted_side = state["uid_side"].get(uid)
                    if fainted_side in ("p1", "p2"):
                        other = "p2" if fainted_side == "p1" else "p1"
                        if last_idx.get(other) is not None:
                            trajectories[other]["rewards"][last_idx[other]] += args.ko_reward
                        if last_idx.get(fainted_side) is not None:
                            trajectories[fainted_side]["rewards"][last_idx[fainted_side]] += (
                                args.faint_penalty
                            )
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
    built: list[dict] = []
    for side in ("p1", "p2"):
        traj = trajectories[side]
        n = len(traj["actions"])
        if n < args.min_actions:
            counters["skip_short_actions"] += 1
            continue
        if n == 0:
            continue

        if winner_side in ("p1", "p2"):
            if winner_side == side:
                traj["rewards"][-1] += args.terminal_win_reward
                traj["weight"] = float(args.win_weight)
            else:
                traj["rewards"][-1] += args.terminal_loss_reward
                traj["weight"] = float(args.loss_weight)
        else:
            traj["weight"] = 1.0

        traj["dones"] = [False] * n
        traj["dones"][-1] = True
        traj["winner_side"] = winner_side
        built.append(traj)

    if built:
        counters["used_files"] += 1
        counters["built_trajectories"] += len(built)
    return built


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare sequence trajectories from gen9random JSON.")
    parser.add_argument("--input-dir", default="data/gen9random")
    parser.add_argument("--output-train", default="data/gen9random_trajectories_train.pkl")
    parser.add_argument("--output-holdout", default="data/gen9random_trajectories_holdout.pkl")
    parser.add_argument("--summary-out", default="logs/replay_audit/gen9random_trajectories.summary.json")
    parser.add_argument("--min-turns", type=int, default=8)
    parser.add_argument("--min-actions", type=int, default=4)
    parser.add_argument("--min-avg-rating", type=int, default=1400)
    parser.add_argument("--skip-forfeit", action="store_true")
    parser.add_argument("--skip-inactivity", action="store_true")
    parser.add_argument("--holdout-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--max-trajectories", type=int, default=0)
    parser.add_argument("--source-tag", default="human_gen9random")
    parser.add_argument("--terminal-win-reward", type=float, default=1.0)
    parser.add_argument("--terminal-loss-reward", type=float, default=-1.0)
    parser.add_argument("--ko-reward", type=float, default=0.08)
    parser.add_argument("--faint-penalty", type=float, default=-0.08)
    parser.add_argument("--win-weight", type=float, default=1.0)
    parser.add_argument("--loss-weight", type=float, default=1.0)
    args = parser.parse_args()

    paths = sorted(glob.glob(str(Path(args.input_dir) / "*.json")))
    if args.max_files > 0:
        paths = paths[: args.max_files]
    if not paths:
        raise SystemExit(f"No replay JSON files found in {args.input_dir}")

    counters = Counter()
    all_trajectories: list[dict] = []
    for idx, path in enumerate(paths, start=1):
        counters["files_seen"] += 1
        try:
            obj = json.load(open(path, "r", encoding="utf-8"))
        except Exception:
            counters["parse_error"] += 1
            continue
        built = _process_replay(obj, args, counters)
        if built:
            all_trajectories.extend(built)
        if args.max_trajectories > 0 and len(all_trajectories) >= args.max_trajectories:
            all_trajectories = all_trajectories[: args.max_trajectories]
            break
        if idx % 1000 == 0:
            print(
                f"processed {idx}/{len(paths)} files, trajectories={len(all_trajectories)}",
                flush=True,
            )

    train, holdout = stable_split_by_battle(all_trajectories, args.holdout_ratio, args.seed)

    out_train = Path(args.output_train)
    out_holdout = Path(args.output_holdout)
    out_train.parent.mkdir(parents=True, exist_ok=True)
    out_holdout.parent.mkdir(parents=True, exist_ok=True)
    with out_train.open("wb") as handle:
        pickle.dump(train, handle)
    with out_holdout.open("wb") as handle:
        pickle.dump(holdout, handle)

    train_steps = sum(len(t.get("actions", [])) for t in train)
    holdout_steps = sum(len(t.get("actions", [])) for t in holdout)
    summary = {
        "input_dir": args.input_dir,
        "outputs": {"train": str(out_train), "holdout": str(out_holdout)},
        "counts": {
            "files_seen": counters["files_seen"],
            "used_files": counters["used_files"],
            "trajectories_total": len(all_trajectories),
            "trajectories_train": len(train),
            "trajectories_holdout": len(holdout),
            "steps_train": train_steps,
            "steps_holdout": holdout_steps,
        },
        "skips": {k: v for k, v in counters.items() if (k.startswith("skip_") and v > 0)},
        "parse_error": counters["parse_error"],
        "filters": {
            "min_turns": args.min_turns,
            "min_actions": args.min_actions,
            "min_avg_rating": args.min_avg_rating,
            "skip_forfeit": bool(args.skip_forfeit),
            "skip_inactivity": bool(args.skip_inactivity),
        },
        "rewards": {
            "terminal_win_reward": args.terminal_win_reward,
            "terminal_loss_reward": args.terminal_loss_reward,
            "ko_reward": args.ko_reward,
            "faint_penalty": args.faint_penalty,
        },
        "weights": {
            "win_weight": args.win_weight,
            "loss_weight": args.loss_weight,
        },
        "split": {"holdout_ratio": args.holdout_ratio, "seed": args.seed},
    }

    summary_path = Path(args.summary_out)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    print(f"Wrote train trajectories: {len(train)} -> {out_train}")
    print(f"Wrote holdout trajectories: {len(holdout)} -> {out_holdout}")
    print(f"Summary -> {summary_path}")
    print(
        "Skips:",
        {k: v for k, v in counters.items() if (k.startswith("skip_") and v > 0)},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
