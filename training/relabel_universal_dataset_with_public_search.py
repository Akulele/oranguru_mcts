#!/usr/bin/env python3
"""Relabel universal replay rows with a public-state random-battle search teacher.

This uses the replay-reconstructible row payloads:
- `state_snapshot`
- `public_team_snapshot`
- `player`

It builds an approximate Foul Play random-battle state from the acting player's
public view, samples consistent opponent worlds, runs poke-engine search, and
replaces `policy_target` with the aggregated teacher distribution.

This is a real public-state relabel pass. It is not self-distillation.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import hashlib
import json
import math
import pickle
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

FP_ROOT = PROJECT_ROOT / "third_party" / "foul-play"
if str(FP_ROOT) not in sys.path:
    sys.path.insert(0, str(FP_ROOT))

import constants  # noqa: E402
from fp.battle import Battle as FPBattle, Battler, LastUsedMove, Pokemon as FPPokemon  # noqa: E402
from fp.search.helpers import populate_pkmn_from_set  # noqa: E402
from fp.search.poke_engine_helpers import battle_to_poke_engine_state  # noqa: E402
from fp.search.random_battles import prepare_random_battles  # noqa: E402
from poke_engine import State as PokeEngineState, monte_carlo_tree_search  # noqa: E402
from data.pkmn_sets import RandomBattleTeamDatasets  # noqa: E402
from data import all_move_json  # noqa: E402

from training.search_assist_utils import safe_float


def _norm(value: object) -> str:
    if value is None:
        return ""
    return "".join(ch.lower() for ch in str(value) if ch.isalnum())


def _stable_seed(base_seed: int, battle_id: str, turn: int, player: str) -> int:
    payload = f"{base_seed}|{battle_id}|{turn}|{player}".encode("utf-8", "ignore")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _status_to_fp(value: object) -> Optional[str]:
    status = _norm(value)
    if not status:
        return None
    mapping = {
        "brn": constants.BURN,
        "burn": constants.BURN,
        "par": constants.PARALYZED,
        "paralysis": constants.PARALYZED,
        "psn": constants.POISON,
        "poison": constants.POISON,
        "tox": constants.TOXIC,
        "toxic": constants.TOXIC,
        "slp": constants.SLEEP,
        "sleep": constants.SLEEP,
        "frz": constants.FROZEN,
        "freeze": constants.FROZEN,
    }
    return mapping.get(status, None)


def _boost_key(stat: object) -> Optional[str]:
    mapping = {
        "atk": constants.ATTACK,
        "attack": constants.ATTACK,
        "def": constants.DEFENSE,
        "defense": constants.DEFENSE,
        "spa": constants.SPECIAL_ATTACK,
        "specialattack": constants.SPECIAL_ATTACK,
        "spd": constants.SPECIAL_DEFENSE,
        "specialdefense": constants.SPECIAL_DEFENSE,
        "spe": constants.SPEED,
        "speed": constants.SPEED,
        "accuracy": constants.ACCURACY,
        "evasion": constants.EVASION,
    }
    return mapping.get(_norm(stat), None)


def _terrain_to_fp(value: object) -> Optional[str]:
    terrain = _norm(value)
    mapping = {
        "electricterrain": constants.ELECTRIC_TERRAIN,
        "grassyterrain": constants.GRASSY_TERRAIN,
        "mistyterrain": constants.MISTY_TERRAIN,
        "psychicterrain": constants.PSYCHIC_TERRAIN,
    }
    return mapping.get(terrain, None)


def _weather_to_fp(value: object) -> Optional[str]:
    weather = _norm(value)
    mapping = {
        "raindance": constants.RAIN,
        "rain": constants.RAIN,
        "sunnyday": constants.SUN,
        "sun": constants.SUN,
        "sandstorm": constants.SAND,
        "hail": constants.HAIL,
        "snowscape": constants.SNOW,
    }
    return mapping.get(weather, None)


def _iter_rows(path: str) -> list[dict]:
    with open(path, "rb") as handle:
        rows = pickle.load(handle)
    if not isinstance(rows, list):
        raise SystemExit(f"Expected list pickle: {path}")
    return [row for row in rows if isinstance(row, dict)]


def _sample_weighted_index(weights: list[float], rng: random.Random) -> int:
    total = sum(max(0.0, w) for w in weights)
    if total <= 0:
        return rng.randrange(len(weights))
    target = rng.random() * total
    acc = 0.0
    for idx, weight in enumerate(weights):
        acc += max(0.0, weight)
        if acc >= target:
            return idx
    return len(weights) - 1


def _species_hash(species: object) -> float:
    value = _norm(species)
    if not value:
        return 0.0
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()
    return float(int(digest[:8], 16) % 997) / 996.0


def _move_hash(move_id: object) -> float:
    value = _norm(move_id)
    if not value:
        return 0.0
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()
    return float(int(digest[:8], 16) % 997) / 996.0


def _move_meta(move_id: object) -> tuple[float, float, float]:
    move = all_move_json.get(_norm(move_id), {}) or {}
    category = _norm(move.get(constants.CATEGORY))
    is_status = 1.0 if category == "status" else 0.0
    is_damaging = 1.0 if category in {"physical", "special"} else 0.0
    try:
        priority = int(move.get(constants.PRIORITY, 0) or 0)
    except Exception:
        priority = 0
    priority_norm = max(-7.0, min(7.0, float(priority))) / 7.0
    return is_status, is_damaging, priority_norm


def _restore_known_runtime_state(mon: FPPokemon, *, hp: object, max_hp: object, status: object, boosts: dict, volatile: list[str], can_tera: bool, tera_type: object) -> None:
    hp_val = int(hp) if isinstance(hp, (int, float)) else mon.hp
    max_hp_val = int(max_hp) if isinstance(max_hp, (int, float)) and int(max_hp) > 0 else mon.max_hp
    mon.max_hp = max(1, int(max_hp_val))
    mon.hp = max(0, min(mon.max_hp, int(hp_val)))
    mon.fainted = mon.hp <= 0
    mon.status = _status_to_fp(status)
    mon.boosts = defaultdict(lambda: 0)
    for stat, value in dict(boosts or {}).items():
        key = _boost_key(stat)
        if key:
            mon.boosts[key] = max(-6, min(6, int(value)))
    mon.volatile_statuses = [_norm(v) for v in list(volatile or []) if _norm(v)]
    mon.can_terastallize = bool(can_tera)
    norm_tera = _norm(tera_type)
    if norm_tera:
        mon.tera_type = norm_tera


def _sample_revealed_set(mon: FPPokemon, move_slots: dict[str, int], rng: random.Random) -> None:
    remaining = RandomBattleTeamDatasets.get_all_remaining_sets(mon)
    if not remaining:
        return
    idx = _sample_weighted_index([float(getattr(s.pkmn_set, "count", 1.0) or 1.0) for s in remaining], rng)
    sampled = remaining[idx]

    known_hp = mon.hp
    known_max_hp = mon.max_hp
    known_status = mon.status
    known_boosts = dict(mon.boosts)
    known_volatile = list(mon.volatile_statuses)
    known_tera_type = mon.tera_type
    known_can_tera = mon.can_terastallize

    populate_pkmn_from_set(mon, sampled, source="public-search")

    if move_slots:
        sampled_moves = { _norm(m.name): m for m in list(mon.moves or []) }
        ordered = [None, None, None, None]
        used = set()
        for move_id, slot in sorted(move_slots.items(), key=lambda item: int(item[1])):
            if 0 <= int(slot) < 4 and move_id in sampled_moves:
                ordered[int(slot)] = sampled_moves[move_id]
                used.add(move_id)
        fill = [m for m in list(mon.moves or []) if _norm(m.name) not in used]
        fill_idx = 0
        for i in range(4):
            if ordered[i] is None and fill_idx < len(fill):
                ordered[i] = fill[fill_idx]
                fill_idx += 1
        mon.moves = [m for m in ordered if m is not None]

    _restore_known_runtime_state(
        mon,
        hp=known_hp,
        max_hp=known_max_hp,
        status=known_status,
        boosts=known_boosts,
        volatile=known_volatile,
        can_tera=known_can_tera,
        tera_type=known_tera_type,
    )


def _build_fp_pokemon(mon_payload: dict, state_snapshot: dict, uid: str, can_tera: bool, rng: random.Random) -> FPPokemon:
    species = str(mon_payload.get("species", "") or "")
    level = int(mon_payload.get("level", 100) or 100)
    mon = FPPokemon(species, level)
    move_slots = dict((state_snapshot.get("move_slots", {}) or {}).get(uid, {}) or {})
    for move_id, _slot in sorted(move_slots.items(), key=lambda item: int(item[1])):
        if move_id:
            mon.add_move(str(move_id))
    _restore_known_runtime_state(
        mon,
        hp=(state_snapshot.get("hp", {}) or {}).get(uid),
        max_hp=(state_snapshot.get("max_hp", {}) or {}).get(uid),
        status=(state_snapshot.get("status", {}) or {}).get(uid, ""),
        boosts=(state_snapshot.get("boosts", {}) or {}).get(uid, {}),
        volatile=(state_snapshot.get("volatile_statuses", {}) or {}).get(uid, []),
        can_tera=can_tera,
        tera_type=mon_payload.get("known_tera_type"),
    )
    _sample_revealed_set(mon, move_slots, rng)
    return mon


def _apply_hazards(dest: dict, side_hazards: dict) -> None:
    if not isinstance(side_hazards, dict):
        return
    for key in (constants.SPIKES, constants.TOXIC_SPIKES, constants.STEALTH_ROCK, constants.STICKY_WEB):
        value = side_hazards.get(key)
        if isinstance(value, (int, float)) and int(value) > 0:
            dest[key] = int(value)


def _build_battler(
    team_payload: list[dict],
    side: str,
    state_snapshot: dict,
    *,
    can_tera_now: bool,
    rng: random.Random,
) -> Battler:
    battler = Battler()
    side_hazards = (state_snapshot.get("hazards", {}) or {}).get(side, {}) or {}
    _apply_hazards(battler.side_conditions, side_hazards)

    mons: list[FPPokemon] = []
    active_idx = 0
    for idx, mon_payload in enumerate(list(team_payload or [])):
        uid = str(mon_payload.get("pokemon_uid", "") or "")
        if not uid:
            continue
        mon = _build_fp_pokemon(
            mon_payload,
            state_snapshot,
            uid,
            can_tera=bool(can_tera_now and mon_payload.get("active", False)),
            rng=rng,
        )
        mons.append(mon)
        if mon_payload.get("active", False):
            active_idx = len(mons) - 1

    if mons:
        battler.active = mons[active_idx]
        battler.reserve = [mon for i, mon in enumerate(mons) if i != active_idx]
    battler.last_used_move = LastUsedMove("", "", 0)
    battler.last_selected_move = LastUsedMove("", "", 0)
    return battler


def _prepare_sample_battles(base_battle: FPBattle, num_battles: int) -> list[tuple[FPBattle, float]]:
    if base_battle.battle_type == constants.BattleType.RANDOM_BATTLE:
        return list(prepare_random_battles(base_battle, max(1, num_battles)))
    return [(copy.deepcopy(base_battle), 1.0)]


def _battle_choice_maps(battle: FPBattle, action_mask: list[bool]) -> dict[str, int]:
    choice_to_idx: dict[str, int] = {}
    active = getattr(battle.user, "active", None)
    if active is not None:
        for slot, move in enumerate(list(getattr(active, "moves", []) or [])[:4]):
            move_id = _norm(getattr(move, "name", "") or "")
            if not move_id:
                continue
            if slot < len(action_mask) and action_mask[slot]:
                choice_to_idx[move_id] = slot
            tera_idx = 9 + slot
            if tera_idx < len(action_mask) and action_mask[tera_idx]:
                choice_to_idx[f"{move_id}-tera"] = tera_idx
    alive_reserves = [mon for mon in list(getattr(battle.user, "reserve", []) or []) if getattr(mon, "hp", 0) > 0]
    for offset, mon in enumerate(alive_reserves[:5]):
        idx = 4 + offset
        if idx < len(action_mask) and action_mask[idx]:
            choice_to_idx[f"switch {_norm(getattr(mon, 'name', '') or '')}"] = idx
    return choice_to_idx


def _build_teacher_action_labels_and_features(battle: FPBattle, action_mask: list[bool]) -> tuple[list[str], list[list[float]]]:
    n_actions = len(action_mask)
    labels = [""] * n_actions
    features: list[list[float]] = []

    active = getattr(getattr(battle, "user", None), "active", None)
    opp = getattr(getattr(battle, "opponent", None), "active", None)
    active_moves = list(getattr(active, "moves", []) or [])
    bench = [mon for mon in list(getattr(getattr(battle, "user", None), "reserve", []) or []) if getattr(mon, "hp", 0) > 0]
    my_hazards = getattr(getattr(battle, "user", None), "side_conditions", {}) or {}
    hazard_load = float(
        int(my_hazards.get(constants.SPIKES, 0) or 0)
        + int(my_hazards.get(constants.TOXIC_SPIKES, 0) or 0)
        + int(my_hazards.get(constants.STEALTH_ROCK, 0) or 0)
        + int(my_hazards.get(constants.STICKY_WEB, 0) or 0)
    ) / 7.0
    opp_hp_frac = (
        float(getattr(opp, "hp", 0) or 0) / float(max(1, int(getattr(opp, "max_hp", 1) or 1)))
        if opp is not None
        else 1.0
    )
    my_hp_frac = (
        float(getattr(active, "hp", 0) or 0) / float(max(1, int(getattr(active, "max_hp", 1) or 1)))
        if active is not None
        else 1.0
    )
    opp_status_flag = 1.0 if _status_to_fp(getattr(opp, "status", None)) else 0.0
    my_status_flag = 1.0 if _status_to_fp(getattr(active, "status", None)) else 0.0
    can_tera = 1.0 if bool(getattr(active, "can_terastallize", False)) else 0.0

    for idx in range(n_actions):
        row = [0.0] * 20
        row[0] = 1.0 if idx < 4 else 0.0
        row[1] = 1.0 if 4 <= idx < 9 else 0.0
        row[2] = 1.0 if 9 <= idx < 13 else 0.0
        row[3] = float(idx) / float(max(1, n_actions - 1))
        row[4] = 1.0 if action_mask[idx] else 0.0
        row[5] = opp_hp_frac
        row[6] = hazard_load
        row[15] = can_tera
        row[16] = float(idx % 4) / 3.0 if idx < 4 or idx >= 9 else 0.0
        row[17] = my_hp_frac
        row[18] = opp_status_flag
        row[19] = my_status_flag

        if idx < 4 or idx >= 9:
            move_idx = idx if idx < 4 else idx - 9
            if move_idx < len(active_moves):
                move = active_moves[move_idx]
                move_id = _norm(getattr(move, "name", "") or "")
                is_status, is_damaging, priority_norm = _move_meta(move_id)
                row[7] = 1.0
                row[8] = _move_hash(move_id)
                row[9] = is_status
                row[10] = is_damaging
                row[11] = priority_norm
                labels[idx] = f"{move_id}-tera" if idx >= 9 else move_id
        elif 4 <= idx < 9:
            bench_idx = idx - 4
            if bench_idx < len(bench):
                mon = bench[bench_idx]
                row[12] = float(getattr(mon, "hp", 0) or 0) / float(max(1, int(getattr(mon, "max_hp", 1) or 1)))
                row[13] = _species_hash(getattr(mon, "name", ""))
                row[14] = 1.0
                labels[idx] = f"switch {_norm(getattr(mon, 'name', '') or '')}"
        features.append(row)

    return labels, features


def _mcts_from_state(task: tuple[str, int]):
    state_str, search_ms = task
    state = PokeEngineState.from_string(state_str)
    return monte_carlo_tree_search(state, search_ms)


def _entropy(probs: list[float]) -> float:
    total = 0.0
    for p in probs:
        if p > 1e-12:
            total -= p * math.log(max(p, 1e-12))
    return total


def _infer_generation(row: dict) -> str:
    format_id = str(row.get("format_id", "") or "")
    if format_id.startswith("gen"):
        return format_id.split("randombattle", 1)[0].split("battlefactory", 1)[0] or "gen9"
    return "gen9"


def _build_base_battle(row: dict, rng: random.Random) -> Optional[FPBattle]:
    player = str(row.get("player", "") or "")
    if player not in ("p1", "p2"):
        return None
    other = "p2" if player == "p1" else "p1"
    state_snapshot = row.get("state_snapshot")
    public_team_snapshot = row.get("public_team_snapshot")
    if not isinstance(state_snapshot, dict) or not isinstance(public_team_snapshot, dict):
        return None
    my_team = list(public_team_snapshot.get(player, []) or [])
    opp_team = list(public_team_snapshot.get(other, []) or [])
    if not my_team or not opp_team:
        return None

    battle = FPBattle(str(row.get("battle_id", "") or "teacher"))
    battle.turn = int(row.get("turn", 0) or 0)
    battle.force_switch = False
    battle.team_preview = False
    battle.weather = _weather_to_fp(state_snapshot.get("weather"))
    battle.weather_turns_remaining = -1 if battle.weather else 0
    battle.field = _terrain_to_fp(state_snapshot.get("terrain"))
    battle.field_turns_remaining = 0
    battle.trick_room = bool(state_snapshot.get("trick_room", False))
    battle.trick_room_turns_remaining = -1 if battle.trick_room else 0
    battle.pokemon_format = str(row.get("format_id", "") or "gen9randombattle")
    battle.generation = _infer_generation(row)
    battle.battle_type = constants.BattleType.RANDOM_BATTLE

    my_can_tera = bool(row.get("can_tera", False))
    opp_can_tera = not bool((state_snapshot.get("tera_used", {}) or {}).get(other, False))
    battle.user = _build_battler(my_team, player, state_snapshot, can_tera_now=my_can_tera, rng=rng)
    battle.opponent = _build_battler(opp_team, other, state_snapshot, can_tera_now=opp_can_tera, rng=rng)
    if battle.user.active is None or battle.opponent.active is None:
        return None
    try:
        battle.user.lock_moves()
    except Exception:
        pass
    try:
        battle.opponent.lock_moves()
    except Exception:
        pass
    return battle


def _relabel_row(row: dict, *, search_ms: int, num_battles: int, seed: int, value_mode: str, executor) -> dict | None:
    if not bool(row.get("search_relabel_ready", False)):
        return None
    action_mask = row.get("action_mask")
    if not isinstance(action_mask, list) or not any(bool(v) for v in action_mask):
        return None

    rng = random.Random(
        _stable_seed(
            seed,
            str(row.get("battle_id", "") or ""),
            int(row.get("turn", 0) or 0),
            str(row.get("player", "") or ""),
        )
    )
    base_battle = _build_base_battle(row, rng)
    if base_battle is None:
        return None
    teacher_action_labels, teacher_action_features = _build_teacher_action_labels_and_features(
        base_battle, [bool(v) for v in action_mask]
    )

    try:
        sampled = _prepare_sample_battles(base_battle, num_battles)
    except Exception:
        return None
    if not sampled:
        return None

    tasks: list[tuple[str, int]] = []
    choice_maps: list[dict[str, int]] = []
    weights: list[float] = []
    for battle, weight in sampled:
        try:
            state_str = battle_to_poke_engine_state(battle).to_string()
        except Exception:
            continue
        tasks.append((state_str, max(1, int(search_ms))))
        choice_maps.append(_battle_choice_maps(battle, [bool(v) for v in action_mask]))
        weights.append(float(weight))
    if not tasks:
        return None

    total_weight = sum(max(0.0, w) for w in weights)
    if total_weight <= 0:
        return None
    weights = [max(0.0, w) / total_weight for w in weights]

    try:
        if executor is not None and len(tasks) > 1:
            results = list(executor.map(_mcts_from_state, tasks))
        else:
            results = [_mcts_from_state(task) for task in tasks]
    except Exception:
        return None

    policy = [0.0] * len(action_mask)
    teacher_value01 = 0.0
    total_visits = 0.0
    worlds_used = 0
    unmapped_choices = 0

    for weight, res, choice_map in zip(weights, results, choice_maps):
        visits_total = max(0.0, float(getattr(res, "total_visits", 0.0) or 0.0))
        side_one = list(getattr(res, "side_one", []) or [])
        if visits_total <= 0 or not side_one:
            continue
        worlds_used += 1
        total_visits += visits_total
        world_value01 = 0.0
        for opt in side_one:
            choice = str(getattr(opt, "move_choice", "") or "")
            visits = max(0.0, float(getattr(opt, "visits", 0.0) or 0.0))
            if visits <= 0:
                continue
            prob = visits / visits_total
            idx = choice_map.get(choice)
            if idx is None:
                unmapped_choices += 1
            elif 0 <= idx < len(policy) and action_mask[idx]:
                policy[idx] += weight * prob
            total_score = float(getattr(opt, "total_score", 0.0) or 0.0)
            avg_score = total_score / visits if visits > 0 else 0.5
            avg_score = max(0.0, min(1.0, avg_score))
            world_value01 += prob * avg_score
        teacher_value01 += weight * world_value01

    legal = [i for i, ok in enumerate(action_mask) if ok]
    if not legal:
        return None
    policy_total = sum(policy[i] for i in legal)
    if policy_total <= 0:
        uniform = 1.0 / float(len(legal))
        policy = [uniform if ok else 0.0 for ok in action_mask]
    else:
        policy = [(v / policy_total) if action_mask[i] else 0.0 for i, v in enumerate(policy)]

    teacher_value = max(-1.0, min(1.0, 2.0 * teacher_value01 - 1.0))
    top1 = max((policy[i] for i in legal), default=0.0)

    new_row = copy.deepcopy(row)
    new_row["orig_policy_target"] = list(row.get("policy_target", []) or [])
    new_row["orig_value_target"] = safe_float(row.get("value_target", 0.0), 0.0)
    new_row["orig_action_labels"] = list(row.get("action_labels", []) or [])
    new_row["orig_action_features"] = copy.deepcopy(row.get("action_features", []) or [])
    new_row["action_labels"] = teacher_action_labels
    new_row["action_features"] = teacher_action_features
    new_row["teacher_policy_target"] = list(policy)
    new_row["teacher_value_target"] = float(teacher_value)
    new_row["teacher_value01"] = float(teacher_value01)
    new_row["teacher_top1_prob"] = float(top1)
    new_row["teacher_entropy"] = float(_entropy([policy[i] for i in legal]))
    new_row["teacher_total_visits"] = float(total_visits)
    new_row["teacher_worlds_used"] = int(worlds_used)
    new_row["teacher_unmapped_choices"] = int(unmapped_choices)
    new_row["teacher_source"] = "public_fp_random_search"
    new_row["teacher_search_ms"] = int(search_ms)
    new_row["teacher_num_battles"] = int(num_battles)
    new_row["policy_target"] = list(policy)
    new_row["value_target"] = (
        float(teacher_value) if value_mode == "teacher" else new_row["orig_value_target"]
    )
    new_row["value_target_source"] = str(value_mode)
    return new_row


def main() -> int:
    parser = argparse.ArgumentParser(description="Relabel universal replay rows with a public-state random-battle search teacher.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-out", default="")
    parser.add_argument("--search-ms", type=int, default=600)
    parser.add_argument("--num-battles", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--value-mode", choices=["orig", "teacher"], default="orig")
    parser.add_argument("--min-turn", type=int, default=1)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=100)
    args = parser.parse_args()

    rows = _iter_rows(args.input)
    RandomBattleTeamDatasets.initialize("gen9")

    counters = Counter()
    out_rows: list[dict] = []
    top1_probs: list[float] = []
    entropies: list[float] = []
    teacher_values: list[float] = []

    executor = None
    if args.workers > 1:
        executor = concurrent.futures.ProcessPoolExecutor(max_workers=args.workers)

    try:
        for row in rows:
            counters["rows_seen"] += 1
            if int(row.get("turn", 0) or 0) < int(args.min_turn):
                counters["drop_min_turn"] += 1
                continue
            relabeled = _relabel_row(
                row,
                search_ms=max(1, int(args.search_ms)),
                num_battles=max(1, int(args.num_battles)),
                seed=int(args.seed),
                value_mode=str(args.value_mode),
                executor=executor,
            )
            if relabeled is None:
                counters["drop_unrelabelable"] += 1
                continue
            out_rows.append(relabeled)
            counters["kept"] += 1
            top1_probs.append(float(relabeled.get("teacher_top1_prob", 0.0) or 0.0))
            entropies.append(float(relabeled.get("teacher_entropy", 0.0) or 0.0))
            teacher_values.append(float(relabeled.get("teacher_value_target", 0.0) or 0.0))
            if args.progress_every > 0 and counters["kept"] % int(args.progress_every) == 0:
                print(f"Relabeled {counters['kept']} rows", flush=True)
            if args.max_rows > 0 and counters["kept"] >= int(args.max_rows):
                break
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as handle:
        pickle.dump(out_rows, handle)

    summary = {
        "input": args.input,
        "output": str(out_path),
        "search_ms": int(args.search_ms),
        "num_battles": int(args.num_battles),
        "value_mode": str(args.value_mode),
        "stats": dict(counters),
        "teacher_top1_mean": (sum(top1_probs) / len(top1_probs)) if top1_probs else 0.0,
        "teacher_entropy_mean": (sum(entropies) / len(entropies)) if entropies else 0.0,
        "teacher_value_mean": (sum(teacher_values) / len(teacher_values)) if teacher_values else 0.0,
    }
    if args.summary_out:
        summary_path = Path(args.summary_out)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")

    print(f"Wrote {len(out_rows)} relabeled rows -> {out_path}")
    print(f"Stats: {dict(counters)}")
    print(f"Teacher top1 mean: {summary['teacher_top1_mean']:.4f}")
    print(f"Teacher entropy mean: {summary['teacher_entropy_mean']:.4f}")
    print(f"Teacher value mean: {summary['teacher_value_mean']:.4f}")
    if args.summary_out:
        print(f"Summary -> {args.summary_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
