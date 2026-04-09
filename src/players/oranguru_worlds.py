#!/usr/bin/env python3
"""
World-sampling and FP-battle helpers for OranguruEnginePlayer.

This module keeps random-battle world construction, world ranking, and MCTS
world aggregation separate from the engine orchestration layer.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import List, Optional, Tuple

from poke_env.battle import Battle, Pokemon
from poke_env.battle.field import Field

from src.utils.damage_calc import normalize_name

FP_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "foul-play"
if str(FP_ROOT) not in sys.path:
    sys.path.insert(0, str(FP_ROOT))

from fp.battle import Battle as FPBattle, Battler, Pokemon as FPPokemon, LastUsedMove  # noqa: E402
from fp.search.poke_engine_helpers import battle_to_poke_engine_state  # noqa: E402
from fp.search.random_battles import prepare_random_battles  # noqa: E402
from poke_engine import State as PokeEngineState, monte_carlo_tree_search  # noqa: E402
import constants  # noqa: E402


def _run_mcts(state_str: str, search_time_ms: int):
    try:
        state = PokeEngineState.from_string(state_str)
        return monte_carlo_tree_search(state, search_time_ms)
    except Exception:
        return None


def build_world_rank_features(self, fp_battle: FPBattle) -> List[float]:
    opponent = getattr(fp_battle, "opponent", None)
    slots: List[Optional[FPPokemon]] = []
    if opponent is not None:
        slots.append(getattr(opponent, "active", None))
        slots.extend(list(getattr(opponent, "reserve", []) or [])[:5])
    while len(slots) < 6:
        slots.append(None)

    try:
        opp_alive = sum(1 for mon in slots if mon is not None and getattr(mon, "hp", 0) > 0)
    except Exception:
        opp_alive = 0
    try:
        opp_revealed = sum(1 for mon in slots if mon is not None)
    except Exception:
        opp_revealed = 0

    features: List[float] = [
        float(int(bool(getattr(opponent, "active", None)))),
        min(1.0, opp_alive / 6.0),
        min(1.0, opp_revealed / 6.0),
        float(int(bool(getattr(getattr(opponent, "active", None), "terastallized", False)))),
    ]

    for mon in slots:
        if mon is None:
            features.extend([0.0] * 13)
            continue

        stats = getattr(mon, "stats", {}) or {}
        hp = float(getattr(mon, "hp", 0) or 0.0)
        max_hp = float(getattr(mon, "max_hp", 0) or 0.0)
        hp_frac = 0.0 if max_hp <= 0 else max(0.0, min(1.0, hp / max_hp))
        spe = max(0.0, min(1.0, float(stats.get(constants.SPEED, 0) or 0.0) / 500.0))
        atk = max(0.0, min(1.0, float(stats.get(constants.ATTACK, 0) or 0.0) / 500.0))
        spa = max(0.0, min(1.0, float(stats.get(constants.SPECIAL_ATTACK, 0) or 0.0) / 500.0))
        move_count = max(0.0, min(1.0, len(getattr(mon, "moves", []) or []) / 4.0))
        features.extend(
            [
                1.0,
                float(int(hp > 0)),
                hp_frac,
                max(0.0, min(1.0, float(getattr(mon, "level", 100) or 100.0) / 100.0)),
                spe,
                atk,
                spa,
                move_count,
                self._search_trace_status_code(getattr(mon, "status", None)),
                self._search_trace_token_hash(getattr(mon, "name", "")),
                self._search_trace_token_hash(getattr(mon, "item", "")),
                self._search_trace_token_hash(getattr(mon, "ability", "")),
                self._search_trace_token_hash(getattr(mon, "tera_type", "")),
            ]
        )
    return features


def build_world_plausibility_features(self, fp_battle: FPBattle) -> List[float]:
    opponent = getattr(fp_battle, "opponent", None)
    active = getattr(opponent, "active", None)
    reserve = list(getattr(opponent, "reserve", []) or [])[:5]
    slots: List[Optional[FPPokemon]] = [active, *reserve]
    while len(slots) < 6:
        slots.append(None)

    present = [mon for mon in slots if mon is not None]
    known_species = sum(1 for mon in present if getattr(mon, "name", ""))
    known_items = sum(1 for mon in present if getattr(mon, "item", ""))
    known_abilities = sum(1 for mon in present if getattr(mon, "ability", ""))
    known_tera = sum(1 for mon in present if getattr(mon, "tera_type", ""))
    known_moves = sum(len(getattr(mon, "moves", []) or []) for mon in present)
    reserve_present = [mon for mon in reserve if mon is not None]
    reserve_alive = sum(1 for mon in reserve_present if getattr(mon, "hp", 0) > 0)
    active_hp = float(getattr(active, "hp", 0) or 0.0)
    active_max_hp = float(getattr(active, "max_hp", 0) or 0.0)
    active_hp_frac = 0.0 if active_max_hp <= 0 else max(0.0, min(1.0, active_hp / active_max_hp))
    active_move_count = len(getattr(active, "moves", []) or []) if active is not None else 0

    return [
        min(1.0, known_species / 6.0),
        min(1.0, known_items / 6.0),
        min(1.0, known_abilities / 6.0),
        min(1.0, known_tera / 6.0),
        min(1.0, known_moves / 24.0),
        min(1.0, max(0, 24 - known_moves) / 24.0),
        min(1.0, len(reserve_present) / 5.0),
        min(1.0, reserve_alive / 5.0),
        active_hp_frac,
        min(1.0, active_move_count / 4.0),
        float(int(bool(getattr(active, "item", "")))),
        float(int(bool(getattr(active, "ability", "")))),
        float(int(bool(getattr(active, "tera_type", "")))),
    ]


def is_low_uncertainty_turn(self, battle: Battle) -> bool:
    opponent = getattr(battle, "opponent_active_pokemon", None)
    opp_known_moves = len(self._get_known_moves(opponent)) if opponent is not None else 0
    opponent_team = list(getattr(battle, "opponent_team", {}).values())
    opp_revealed = len([mon for mon in opponent_team if mon is not None])
    return (
        opp_revealed >= max(1, self.LOW_UNCERTAINTY_MIN_REVEALED)
        and opp_known_moves >= max(0, self.LOW_UNCERTAINTY_MIN_KNOWN_MOVES)
    )


def is_endgame_turn(self, battle: Battle) -> bool:
    opponent_team = list(getattr(battle, "opponent_team", {}).values())
    my_team = list(getattr(battle, "team", {}).values())
    opp_alive = len([mon for mon in opponent_team if mon is not None and not getattr(mon, "fainted", False)])
    my_alive = len([mon for mon in my_team if mon is not None and not getattr(mon, "fainted", False)])
    return my_alive <= 2 or opp_alive <= 2


def world_ranker_turn_allowed(self, battle: Battle) -> bool:
    low_uncertainty = self._is_low_uncertainty_turn(battle)
    endgame = self._is_endgame_turn(battle)
    allow_low_uncertainty = self.WORLD_RANKER_LOW_UNCERTAINTY_ONLY and low_uncertainty
    allow_endgame = self.WORLD_RANKER_ENDGAME_ONLY and endgame
    if self.WORLD_RANKER_LOW_UNCERTAINTY_ONLY or self.WORLD_RANKER_ENDGAME_ONLY:
        return allow_low_uncertainty or allow_endgame
    return True


def build_world_ranker_input_features(
    self,
    battle: Battle,
    fp_battle: FPBattle,
    sample_weight: float,
) -> tuple[list[float], list[float]] | tuple[None, None]:
    feature_builder = self._init_search_trace_builder()
    if feature_builder is None:
        return None, None
    try:
        board_features = feature_builder.build(battle)
        board_features = [
            0.0 if (not isinstance(f, (int, float)) or f != f or f > 1e6 or f < -1e6) else float(f)
            for f in board_features
        ]
    except Exception:
        return None, None

    mask, _, _ = self._build_rl_action_mask_and_maps(battle)
    valid_actions = max(1, sum(1 for v in mask if v))
    turn = int(getattr(battle, "turn", 0) or 0)
    phase = self._search_trace_phase(battle)
    world_features = self._build_world_rank_features(fp_battle)
    plausibility_features = self._build_world_plausibility_features(fp_battle)
    world_features = [float(v) for v in world_features] + [float(v) for v in plausibility_features] + [
        float(sample_weight),
        min(1.0, valid_actions / max(1, len(mask))),
        min(1.0, turn / 30.0),
        1.0 if phase == "opening" else 0.0,
        1.0 if phase == "mid" else 0.0,
        1.0 if phase == "end" else 0.0,
    ]
    return board_features, world_features


def rank_and_trim_worlds(
    self,
    battle: Battle,
    fp_battles: List[FPBattle],
    weights: List[float],
) -> tuple[List[FPBattle], List[float]]:
    if not self.WORLD_RANKER_ENABLED:
        return fp_battles, weights
    if len(fp_battles) < max(1, self.WORLD_RANKER_MIN_CANDIDATES):
        return fp_battles, weights
    if len(fp_battles) <= max(1, self.WORLD_RANKER_KEEP_TOPK):
        return fp_battles, weights
    if not self._world_ranker_turn_allowed(battle):
        return fp_battles, weights
    if not self._init_world_ranker():
        return fp_battles, weights

    torch = self._world_ranker_torch
    model = self._world_ranker_model
    if torch is None or model is None:
        return fp_battles, weights

    pairs: List[tuple[float, int]] = []
    try:
        for index, (fp_battle, weight) in enumerate(zip(fp_battles, weights)):
            board_features, world_features = self._build_world_ranker_input_features(
                battle,
                fp_battle,
                sample_weight=float(weight),
            )
            if board_features is None or world_features is None:
                return fp_battles, weights
            board_t = torch.tensor([board_features], dtype=torch.float32, device=self._world_ranker_device)
            world_t = torch.tensor([world_features], dtype=torch.float32, device=self._world_ranker_device)
            with torch.no_grad():
                score = float(model(board_t, world_t)[0].detach().cpu().item())
            pairs.append((score, index))
    except Exception:
        self._world_ranker_failed = True
        return fp_battles, weights

    keep_topk = max(1, min(len(fp_battles), self.WORLD_RANKER_KEEP_TOPK))
    kept_indices = {idx for _, idx in sorted(pairs, key=lambda item: item[0], reverse=True)[:keep_topk]}
    if len(kept_indices) >= len(fp_battles):
        return fp_battles, weights

    self._mcts_stats["world_ranker_used"] += 1
    self._mcts_stats["world_ranker_pruned"] += max(0, len(fp_battles) - len(kept_indices))
    trimmed_battles = [fp_b for idx, fp_b in enumerate(fp_battles) if idx in kept_indices]
    trimmed_weights = [w for idx, w in enumerate(weights) if idx in kept_indices]
    total_weight = sum(max(0.0, float(w)) for w in trimmed_weights)
    if total_weight > 0:
        trimmed_weights = [float(w) / total_weight for w in trimmed_weights]
    return trimmed_battles, trimmed_weights


def build_world_candidate_summary(
    self,
    battle: Battle,
    fp_battle: FPBattle,
    sample_weight: float,
    result,
    state_str: str = "",
) -> dict:
    mask, move_map, switch_map = self._build_rl_action_mask_and_maps(battle)
    world_visits = [0.0] * len(mask)
    total_visits = 0.0
    top_choice = ""
    top_choice_visits = 0.0

    for opt in getattr(result, "side_one", []) or []:
        choice = getattr(opt, "move_choice", "") or ""
        visits = max(0.0, float(getattr(opt, "visits", 0) or 0.0))
        if visits <= 0:
            continue
        idx = self._choice_to_rl_action_idx(choice, mask, move_map, switch_map)
        if idx is None or idx >= len(world_visits):
            continue
        world_visits[idx] += visits
        total_visits += visits
        if visits > top_choice_visits:
            top_choice_visits = visits
            top_choice = choice

    top_choice_idx = self._choice_to_rl_action_idx(top_choice, mask, move_map, switch_map)
    top_choice_prob = 0.0 if total_visits <= 0 else top_choice_visits / total_visits
    summary = {
        "index": 0,
        "sample_weight": float(sample_weight),
        "world_features": [float(v) for v in self._build_world_rank_features(fp_battle)],
        "plausibility_features": [
            float(v) for v in self._build_world_plausibility_features(fp_battle)
        ],
        "visit_counts": [float(v) for v in world_visits],
        "total_visits": float(total_visits),
        "top_choice": top_choice,
        "top_choice_idx": int(top_choice_idx) if top_choice_idx is not None else None,
        "top_choice_kind": self._search_trace_choice_kind(battle, top_choice),
        "top_choice_prob": float(top_choice_prob),
    }
    if self.SEARCH_TRACE_INCLUDE_STATE_STR and state_str:
        summary["state_str"] = str(state_str)
    return summary


def apply_world_budget_controls(self, battle: Battle, sample_states: int) -> int:
    requested = max(1, int(sample_states))
    budgeted = requested
    low_uncertainty = self.LOW_UNCERTAINTY_WORLD_REDUCTION and self._is_low_uncertainty_turn(battle)
    endgame = self.ENDGAME_WORLD_REDUCTION and self._is_endgame_turn(battle)

    if low_uncertainty:
        budgeted = min(budgeted, max(1, self.LOW_UNCERTAINTY_MAX_STATES))
        self._mcts_stats["low_uncertainty_turns"] += 1
    if endgame:
        budgeted = min(budgeted, max(1, self.ENDGAME_MAX_STATES))
        if budgeted < requested:
            self._mcts_stats["endgame_reduction_turns"] += 1

    saved = max(0, requested - budgeted)
    if low_uncertainty and saved > 0:
        self._mcts_stats["low_uncertainty_worlds_saved"] += saved
    if endgame and saved > 0:
        self._mcts_stats["endgame_worlds_saved"] += saved
    self._mcts_stats["sample_states_requested_total"] += requested
    self._mcts_stats["sample_states_budgeted_total"] += budgeted
    return budgeted


def extract_last_opponent_move(self, battle: Battle) -> Optional[Tuple[str, str, int]]:
    last_turn = battle.turn - 1
    observations = getattr(battle, "observations", {})
    obs = observations.get(last_turn)
    role = getattr(battle, "player_role", None)
    if obs is None or not role or last_turn < 1:
        return None
    for event in obs.events:
        if len(event) < 4:
            continue
        if event[1] != "move":
            continue
        who = event[2]
        if who.startswith(role):
            continue
        species = self._species_from_event(battle, event) or ""
        move_id = normalize_name(event[3])
        if move_id:
            return species, move_id, last_turn
    return None


def sample_set_for_species(
    self,
    species: str,
    battle: Battle,
    mon: Optional[Pokemon] = None,
    rng: Optional[random.Random] = None,
) -> Optional[dict]:
    picker = rng or random
    if mon is not None:
        candidates = self._candidate_randombattle_sets(mon, battle)
        if candidates:
            weights = [c for _, c in candidates]
            choice = picker.choices(candidates, weights=weights, k=1)[0][0]
            return choice
    data = self._load_randombattle_sets()
    sets = data.get(species, {})
    if not sets:
        return None
    keys = list(sets.keys())
    weights = [sets[k] for k in keys]
    key = picker.choices(keys, weights=weights, k=1)[0]
    return self._parse_randombattle_set_key(key)


def sample_unknown_opponents(
    self,
    battle: Battle,
    taken: set,
    count: int,
    rng: Optional[random.Random] = None,
) -> List[FPPokemon]:
    picker = rng or random
    data = self._load_randombattle_sets()
    if not data:
        return []
    names = []
    weights = []
    for name, sets in data.items():
        if name in taken:
            continue
        total = sum(sets.values())
        if total <= 0:
            continue
        names.append(name)
        weights.append(total)
    result = []
    for _ in range(count):
        if not names:
            break
        species = picker.choices(names, weights=weights, k=1)[0]
        set_info = self._sample_set_for_species(species, battle, rng=picker)
        if not set_info:
            continue
        fp_mon = FPPokemon(species, set_info.get("level", 100))
        fp_mon.ability = set_info.get("ability") or fp_mon.ability
        fp_mon.item = set_info.get("item") or fp_mon.item
        fp_mon.tera_type = set_info.get("tera") or fp_mon.tera_type
        self._fill_moves_from_set(fp_mon, set_info, set())
        result.append(fp_mon)
        taken.add(species)
    return result


def build_fp_battle(self, battle: Battle, seed: int, fill_opponent_sets: bool = False) -> FPBattle:
    rng = random.Random(seed)
    fp_battle = FPBattle(battle.battle_tag)
    fp_battle.turn = battle.turn
    fp_battle.force_switch = battle.force_switch
    fp_battle.team_preview = getattr(battle, "team_preview", None)
    if fp_battle.team_preview is None:
        fp_battle.team_preview = getattr(battle, "teampreview", False)
    fp_battle.weather = self._map_weather(battle)
    fp_battle.field = self._map_terrain(battle)
    fp_battle.trick_room = Field.TRICK_ROOM in (battle.fields or {})
    fp_battle.trick_room_turns_remaining = -1 if fp_battle.trick_room else 0
    fp_battle.weather_turns_remaining = self._weather_turns_remaining(battle)
    fp_battle.field_turns_remaining = self._terrain_turns_remaining(battle)
    fp_battle.pokemon_format = getattr(battle, "format", None) or "gen9randombattle"
    fp_battle.generation = f"gen{getattr(battle, 'gen', 9) or 9}"
    if fp_battle.pokemon_format and "randombattle" in fp_battle.pokemon_format:
        fp_battle.battle_type = constants.BattleType.RANDOM_BATTLE

    user = Battler()
    opponent = Battler()
    user.last_used_move = LastUsedMove("", "", 0)
    opponent.last_used_move = LastUsedMove("", "", 0)
    user.last_selected_move = LastUsedMove("", "", 0)
    opponent.last_selected_move = LastUsedMove("", "", 0)

    active = battle.active_pokemon
    try:
        user.trapped = bool(getattr(battle, "trapped", False))
    except Exception:
        user.trapped = False
    if active and self._is_trapped(active):
        user.trapped = True

    self._map_side_conditions(battle.side_conditions, user.side_conditions)
    self._map_side_conditions(battle.opponent_side_conditions, opponent.side_conditions)

    if active:
        user.active = self._poke_env_to_fp(active, battle, None)
        user.active.can_terastallize = bool(getattr(battle, "can_tera", False))
    reserves = []
    for mon in battle.team.values():
        if mon is None or mon == active:
            continue
        reserves.append(self._poke_env_to_fp(mon, battle, None))
    user.reserve = reserves
    if user.active and battle.available_moves:
        available = {normalize_name(m.id) for m in battle.available_moves}
        for mv in user.active.moves:
            move_id = self._fp_move_id(mv)
            mv.disabled = bool(move_id) and move_id not in available
        if self._sleep_clause_blocked(battle):
            for mv in user.active.moves:
                if self._fp_move_inflicts_sleep(mv.name):
                    mv.disabled = True
        mem = self._get_battle_memory(battle)
        if mem.get("last_action") == "move":
            last_turn = mem.get("last_action_turn", 0)
            move_id = mem.get("last_move_id") or ""
            if last_turn and move_id:
                user.last_used_move = LastUsedMove(user.active.name, move_id, last_turn)
                user.last_selected_move = user.last_used_move
    self._clear_invalid_encore(user)

    opp_active = battle.opponent_active_pokemon
    if opp_active:
        set_info = None
        if fill_opponent_sets:
            set_info = self._sample_set_for_species(
                normalize_name(opp_active.species), battle, opp_active, rng=rng
            )
        opponent.active = self._poke_env_to_fp(opp_active, battle, set_info)
        opponent.active.can_terastallize = not bool(getattr(battle, "opponent_used_tera", False))
        self._apply_opponent_item_flags(opponent.active, battle)
        self._apply_opponent_ability_flags(opponent.active, battle)
        self._apply_known_opponent_moves(opponent.active, battle)
        self._apply_opponent_switch_memory(opponent.active, battle)
        self._apply_speed_bounds(opponent.active, battle)
        opponent.trapped = self._is_trapped(opp_active)
    opp_reserves = []
    taken = set()
    if opp_active:
        taken.add(normalize_name(opp_active.species))
    for mon in battle.opponent_team.values():
        if mon is None or mon == opp_active:
            continue
        set_info = None
        if fill_opponent_sets:
            set_info = self._sample_set_for_species(normalize_name(mon.species), battle, mon, rng=rng)
        fp_mon = self._poke_env_to_fp(mon, battle, set_info)
        self._apply_opponent_item_flags(fp_mon, battle)
        self._apply_opponent_ability_flags(fp_mon, battle)
        self._apply_known_opponent_moves(fp_mon, battle)
        self._apply_opponent_switch_memory(fp_mon, battle)
        self._apply_speed_bounds(fp_mon, battle)
        opp_reserves.append(fp_mon)
        taken.add(normalize_name(mon.species))
    if fill_opponent_sets:
        missing = max(0, 6 - (len(opp_reserves) + (1 if opp_active else 0)))
        opp_reserves.extend(self._sample_unknown_opponents(battle, taken, missing, rng=rng))
    opponent.reserve = opp_reserves

    last_opp = self._extract_last_opponent_move(battle)
    if last_opp and opponent.active:
        species, move_id, last_turn = last_opp
        opponent.last_used_move = LastUsedMove(species or opponent.active.name, move_id, last_turn)
        opponent.last_selected_move = opponent.last_used_move
    self._clear_invalid_encore(opponent)

    mem = self._get_battle_memory(battle)
    switch_flags = mem.get("switch_flags", {})
    if isinstance(switch_flags, dict):
        self_flags = switch_flags.get("self", {})
        opp_flags = switch_flags.get("opp", {})
        user.baton_passing = bool(self_flags.get("baton", False))
        user.shed_tailing = bool(self_flags.get("shed", False))
        opponent.baton_passing = bool(opp_flags.get("baton", False))
        opponent.shed_tailing = bool(opp_flags.get("shed", False))
    wish = mem.get("pending_wish", {})
    future_sight = mem.get("pending_future_sight", {})
    if isinstance(wish, dict):
        user.wish = wish.get("self", user.wish)
        opponent.wish = wish.get("opp", opponent.wish)
    if isinstance(future_sight, dict):
        user.future_sight = future_sight.get("self", user.future_sight)
        opponent.future_sight = future_sight.get("opp", opponent.future_sight)

    fp_battle.user = user
    fp_battle.opponent = opponent
    if user.active:
        try:
            user.lock_moves()
        except Exception:
            pass
    if opponent.active:
        try:
            opponent.lock_moves()
        except Exception:
            pass
    return fp_battle


def fp_move_id(self, move) -> str:
    if move is None:
        return ""
    for attr in ("id", "move_id"):
        try:
            value = getattr(move, attr)
        except Exception:
            value = None
        if value:
            return normalize_name(str(value))
    try:
        name = getattr(move, "name", "")
    except Exception:
        name = ""
    return normalize_name(str(name)) if name else ""


def collect_mcts_results(
    self,
    battle: Battle,
    sample_states: int,
    search_time_ms: int,
    base_fp_battle: Optional[FPBattle] = None,
) -> Tuple[List[Tuple[object, float]], List[dict]]:
    fp_battle = base_fp_battle or self._build_fp_battle(
        battle, seed=0, fill_opponent_sets=False
    )
    fp_battles: List[FPBattle] = []
    weights: List[float] = []

    if fp_battle.battle_type == constants.BattleType.RANDOM_BATTLE:
        try:
            self._ensure_randbats_sets(battle)
            samples = prepare_random_battles(fp_battle, sample_states)
            fp_battles = [b for b, _ in samples]
            weights = [w for _, w in samples]
        except Exception:
            fp_battles = [fp_battle]
            weights = [1.0]
    else:
        seeds = [random.randint(1, 1_000_000) for _ in range(sample_states)]
        for seed in seeds:
            fp_battles.append(
                self._build_fp_battle(battle, seed, fill_opponent_sets=True)
            )
        weights = [1.0 / len(fp_battles)] * len(fp_battles)

    self._mcts_stats["worlds_generated_total"] += len(fp_battles)
    fp_battles, weights = self._rank_and_trim_worlds(battle, fp_battles, weights)
    self._mcts_stats["worlds_searched_total"] += len(fp_battles)

    states = [battle_to_poke_engine_state(b).to_string() for b in fp_battles]
    self._mcts_stats["states_sampled"] += len(states)

    results: List[Tuple[object, float]] = []
    world_candidates: List[dict] = []
    workers = min(self.PARALLELISM, len(states))
    if workers > 1:
        executor = self._get_mcts_pool(workers)
        if executor is None:
            workers = 1
    if workers > 1:
        futures = [
            executor.submit(_run_mcts, state, search_time_ms)  # type: ignore[union-attr]
            for state in states
        ]
        for index, (fut, weight, fp_state, state_str) in enumerate(zip(futures, weights, fp_battles, states)):
            try:
                timeout_sec = max(1.0, (search_time_ms / 1000.0) * 2.0 + 2.0)
                res = fut.result(timeout=timeout_sec)
            except Exception:
                self._mcts_stats["result_errors"] += 1
                continue
            if res is None:
                self._mcts_stats["result_none"] += 1
                continue
            results.append((res, weight))
            summary = self._build_world_candidate_summary(
                battle, fp_state, weight, res, state_str=state_str
            )
            summary["index"] = int(index)
            world_candidates.append(summary)
            self._mcts_stats["results_kept"] += 1
    else:
        for index, (state, weight, fp_state) in enumerate(zip(states, weights, fp_battles)):
            res = _run_mcts(state, search_time_ms)
            if res is None:
                self._mcts_stats["result_none"] += 1
                continue
            results.append((res, weight))
            summary = self._build_world_candidate_summary(
                battle, fp_state, weight, res, state_str=state
            )
            summary["index"] = int(index)
            world_candidates.append(summary)
            self._mcts_stats["results_kept"] += 1
    return results, world_candidates
