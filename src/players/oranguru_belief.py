#!/usr/bin/env python3
"""
Random-battle belief and opponent-set inference helpers for OranguruEnginePlayer.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

from poke_env.battle import Battle, Pokemon, SideCondition

from src.players.rule_bot import RuleBotPlayer
from src.utils.damage_calc import normalize_name
from src.utils.damage_belief import score_set_damage_consistency

FP_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "foul-play"
if str(FP_ROOT) not in sys.path:
    sys.path.insert(0, str(FP_ROOT))

from fp.battle import Pokemon as FPPokemon  # noqa: E402
from data import all_move_json as FP_MOVE_JSON  # noqa: E402
from data.pkmn_sets import RandomBattleTeamDatasets  # noqa: E402


def damage_belief_has_unmodeled_state(self, battle: Battle) -> bool:
    opp_sc = set((battle.opponent_side_conditions or {}).keys())
    screen_conds = {SideCondition.REFLECT, SideCondition.LIGHT_SCREEN}
    aurora = getattr(SideCondition, "AURORA_VEIL", None)
    if aurora is not None:
        screen_conds.add(aurora)
    if opp_sc.intersection(screen_conds):
        return True

    for w in battle.weather:
        wn = normalize_name(w.name if hasattr(w, "name") else str(w))
        if "sun" in wn or "sunnyday" in wn or "rain" in wn or "raindance" in wn:
            continue
        if wn:
            return True
    return False


def damage_belief_observations(self, battle: Battle, species: str) -> List[dict]:
    mem = self._get_battle_memory(battle)
    obs = list(mem.get("damage_observations", {}).get(species, []) or [])
    if not obs:
        return []
    if self.DAMAGE_BELIEF_STRICT_ONLY:
        obs = [entry for entry in obs if entry.get("high_confidence", False)]
    return obs


def ensure_randbats_sets(self, battle: Battle) -> str:
    gen_num = getattr(battle, "gen", None) or 9
    gen_name = f"gen{gen_num}"
    if not self._randbats_initialized or self._randbats_gen != gen_name:
        RandomBattleTeamDatasets.initialize(gen_name)
        self._randbats_initialized = True
        self._randbats_gen = gen_name
        self._randbats_sanitized = False
    if not self._randbats_sanitized:
        self._sanitize_randbats_moves()
    return gen_name


def sanitize_randbats_moves(self) -> None:
    invalid = set()
    for pkmn_sets in RandomBattleTeamDatasets.pkmn_sets.values():
        for predicted in pkmn_sets:
            moves = list(predicted.pkmn_moveset.moves)
            filtered = [m for m in moves if m in FP_MOVE_JSON]
            if not filtered:
                invalid.update(m for m in moves if m not in FP_MOVE_JSON)
                continue
            if len(filtered) != len(moves):
                invalid.update(m for m in moves if m not in FP_MOVE_JSON)
                predicted.pkmn_moveset.moves = tuple(filtered)
    self._randbats_sanitized = True
    if invalid:
        pass


def belief_weight_for_set(
    self,
    fp_mon: FPPokemon,
    set_info: dict,
    battle: Battle,
    apply_damage: bool = True,
) -> float:
    species = normalize_name(fp_mon.name)
    mem = self._get_battle_memory(battle)
    ability = normalize_name(set_info.get("ability", "") or "")

    multiplier = 1.0
    immune_types = mem.get("immune_types", {}).get(species, set())
    if immune_types and ability:
        types = set(fp_mon.types or [])
        for immune_type in immune_types:
            if immune_type in types:
                continue
            candidates = self.IMMUNITY_ABILITY_MAP.get(immune_type, set())
            if not candidates:
                continue
            if ability in candidates:
                multiplier *= self.BELIEF_IMMUNITY_MATCH
            else:
                multiplier *= self.BELIEF_IMMUNITY_MISS

    if self.DAMAGE_BELIEF and apply_damage:
        dmg_obs = self._damage_belief_observations(battle, species)
        if dmg_obs:
            set_item = normalize_name(set_info.get("item", "") or "")
            base_stats = {}
            if fp_mon.base_stats:
                raw = fp_mon.base_stats
                base_stats = {
                    "hp": raw.get("hp", 80),
                    "atk": raw.get("attack", raw.get("atk", 80)),
                    "def": raw.get("defense", raw.get("def", 80)),
                    "spa": raw.get("special-attack", raw.get("spa", 80)),
                    "spd": raw.get("special-defense", raw.get("spd", 80)),
                    "spe": raw.get("speed", raw.get("spe", 80)),
                }
            if not base_stats:
                base_stats = {"hp": 80, "atk": 80, "def": 80, "spa": 80, "spd": 80, "spe": 80}
            level = int(set_info.get("level", 0) or 0) or int(fp_mon.level or 100)
            sp_types = list(fp_mon.types or [])
            dmg_weight = score_set_damage_consistency(
                observations=dmg_obs,
                set_ability=ability,
                set_item=set_item,
                species_base_stats=base_stats,
                species_level=level,
                species_types=sp_types,
                mode=self.DAMAGE_BELIEF_MODE,
                per_obs_min=self.DAMAGE_BELIEF_PER_OBS_MIN,
                per_obs_max=self.DAMAGE_BELIEF_PER_OBS_MAX,
                final_min=self.DAMAGE_BELIEF_FINAL_MIN,
                final_max=self.DAMAGE_BELIEF_FINAL_MAX,
            )
            multiplier *= dmg_weight

    return multiplier


def candidate_randombattle_sets(self, opponent: Pokemon, battle: Battle) -> List[Tuple[dict, float]]:
    if not self.BELIEF_SAMPLING:
        return RuleBotPlayer._candidate_randombattle_sets(self, opponent, battle)
    if opponent is None:
        return []
    try:
        self._ensure_randbats_sets(battle)
    except Exception:
        return RuleBotPlayer._candidate_randombattle_sets(self, opponent, battle)

    fp_mon = self._poke_env_to_fp(opponent, battle, None)
    self._apply_opponent_item_flags(fp_mon, battle)
    self._apply_opponent_ability_flags(fp_mon, battle)
    self._apply_known_opponent_moves(fp_mon, battle)
    self._apply_speed_bounds(fp_mon, battle)

    predicted_sets = RandomBattleTeamDatasets.get_all_remaining_sets(fp_mon)
    if not predicted_sets:
        return RuleBotPlayer._candidate_randombattle_sets(self, opponent, battle)

    candidates: List[Tuple[dict, float]] = []
    for predicted in predicted_sets:
        pset = predicted.pkmn_set
        moves = [m for m in predicted.pkmn_moveset.moves if m in FP_MOVE_JSON]
        if not moves:
            continue
        set_info = {
            "level": int(pset.level or fp_mon.level or 100),
            "item": normalize_name(pset.item or ""),
            "ability": normalize_name(pset.ability or ""),
            "moves": [normalize_name(m) for m in moves],
            "tera": normalize_name(pset.tera_type or ""),
        }
        weight = float(pset.count or 0)
        if weight <= 0:
            continue
        weight = weight * self._belief_weight_for_set(
            fp_mon,
            set_info,
            battle,
            apply_damage=False,
        )
        if weight <= 0:
            continue
        candidates.append((set_info, weight))

    if not candidates:
        return RuleBotPlayer._candidate_randombattle_sets(self, opponent, battle)

    candidates.sort(key=lambda x: x[1], reverse=True)
    if self.DAMAGE_BELIEF:
        species = normalize_name(fp_mon.name)
        dmg_obs = self._damage_belief_observations(battle, species)
        if len(dmg_obs) >= self.DAMAGE_BELIEF_MIN_OBS:
            topk = max(1, min(self.DAMAGE_BELIEF_TOPK, len(candidates)))
            base_stats = {}
            if fp_mon.base_stats:
                raw = fp_mon.base_stats
                base_stats = {
                    "hp": raw.get("hp", 80),
                    "atk": raw.get("attack", raw.get("atk", 80)),
                    "def": raw.get("defense", raw.get("def", 80)),
                    "spa": raw.get("special-attack", raw.get("spa", 80)),
                    "spd": raw.get("special-defense", raw.get("spd", 80)),
                    "spe": raw.get("speed", raw.get("spe", 80)),
                }
            if not base_stats:
                base_stats = {"hp": 80, "atk": 80, "def": 80, "spa": 80, "spd": 80, "spe": 80}
            sp_types = list(fp_mon.types or [])

            reranked = list(candidates)
            for idx in range(topk):
                set_info, base_weight = reranked[idx]
                set_ability = normalize_name(set_info.get("ability", "") or "")
                set_item = normalize_name(set_info.get("item", "") or "")
                level = int(set_info.get("level", 0) or 0) or int(fp_mon.level or 100)
                dmg_weight = score_set_damage_consistency(
                    observations=dmg_obs,
                    set_ability=set_ability,
                    set_item=set_item,
                    species_base_stats=base_stats,
                    species_level=level,
                    species_types=sp_types,
                    mode=self.DAMAGE_BELIEF_MODE,
                    per_obs_min=self.DAMAGE_BELIEF_PER_OBS_MIN,
                    per_obs_max=self.DAMAGE_BELIEF_PER_OBS_MAX,
                    final_min=self.DAMAGE_BELIEF_FINAL_MIN,
                    final_max=self.DAMAGE_BELIEF_FINAL_MAX,
                )
                reranked[idx] = (set_info, base_weight * dmg_weight)
            candidates = reranked
            candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[:20]
