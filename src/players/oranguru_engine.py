#!/usr/bin/env python3
"""
OranguruEnginePlayer - MCTS via poke-engine using a lightweight state builder.

Builds a poke-engine State from poke_env battle + randombattle set sampling
and runs MCTS to choose moves, similar in spirit to foul-play.
"""

from __future__ import annotations

import math
import os
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys

from poke_env.battle import AbstractBattle, Battle, Pokemon, SideCondition, MoveCategory
from poke_env.battle.effect import Effect
from poke_env.battle.field import Field
from poke_env.battle.weather import Weather

from src.players.rule_bot import RuleBotPlayer
from src.utils.damage_calc import normalize_name

FP_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "foul-play"
if str(FP_ROOT) not in sys.path:
    sys.path.insert(0, str(FP_ROOT))

from fp.battle import Battle as FPBattle, Battler, Pokemon as FPPokemon, LastUsedMove, StatRange  # noqa: E402
from fp.search.poke_engine_helpers import battle_to_poke_engine_state  # noqa: E402
from fp.search.random_battles import prepare_random_battles  # noqa: E402
from data import all_move_json as FP_MOVE_JSON  # noqa: E402
from data.pkmn_sets import RandomBattleTeamDatasets  # noqa: E402
import constants  # noqa: E402
from poke_engine import State as PokeEngineState, monte_carlo_tree_search  # noqa: E402


def _run_mcts(state_str: str, search_time_ms: int):
    try:
        state = PokeEngineState.from_string(state_str)
        return monte_carlo_tree_search(state, search_time_ms)
    except Exception:
        return None


def _maybe_effect(name: str) -> Optional[Effect]:
    return getattr(Effect, name, None)


class OranguruEnginePlayer(RuleBotPlayer):
    CPU_COUNT = os.cpu_count() or 2
    SEARCH_TIME_MS = int(os.getenv("ORANGURU_SEARCH_MS", "200"))
    PARALLELISM = int(os.getenv("ORANGURU_PARALLELISM", str(max(1, min(3, CPU_COUNT // 2 or 1)))))
    SAMPLE_STATES = int(os.getenv("ORANGURU_SAMPLE_STATES", str(max(1, min(PARALLELISM, 3)))))
    MAX_SAMPLE_STATES = int(
        os.getenv("ORANGURU_SAMPLE_STATES_MAX", str(max(6, PARALLELISM * 4)))
    )
    DYNAMIC_SAMPLING = bool(int(os.getenv("ORANGURU_DYNAMIC_SAMPLING", "1")))
    HEURISTIC_BLEND = float(os.getenv("ORANGURU_HEURISTIC_BLEND", "0.35"))
    MIN_HEURISTIC_BLEND = float(os.getenv("ORANGURU_MIN_HEURISTIC_BLEND", "0.0"))
    POLICY_CUTOFF = float(os.getenv("ORANGURU_POLICY_CUTOFF", "0.75"))
    STATUS_KO_GUARD = bool(int(os.getenv("ORANGURU_STATUS_KO_GUARD", "0")))
    STATUS_KO_THRESHOLD = float(os.getenv("ORANGURU_STATUS_KO_THRESHOLD", "200.0"))
    IMMUNITY_INFER = bool(int(os.getenv("ORANGURU_IMMUNITY_INFER", "0")))
    MCTS_DETERMINISTIC = bool(int(os.getenv("ORANGURU_MCTS_DETERMINISTIC", "0")))
    MCTS_DETERMINISTIC_EVAL_ONLY = bool(int(os.getenv("ORANGURU_MCTS_DETERMINISTIC_EVAL_ONLY", "0")))
    BELIEF_SAMPLING = bool(int(os.getenv("ORANGURU_BELIEF_SAMPLING", "1")))
    BELIEF_IMMUNITY_MATCH = float(os.getenv("ORANGURU_BELIEF_IMMUNITY_MATCH", "1.5"))
    BELIEF_IMMUNITY_MISS = float(os.getenv("ORANGURU_BELIEF_IMMUNITY_MISS", "0.7"))
    MCTS_CONFIDENCE_THRESHOLD = float(os.getenv("ORANGURU_MCTS_CONFIDENCE", "0.6"))
    GATE_MODE = os.getenv("ORANGURU_GATE_MODE", "hard").lower()
    SELECTION_MODE = os.getenv("ORANGURU_SELECTION_MODE", "blend").lower()
    RERANK_TOPK = int(os.getenv("ORANGURU_RERANK_TOPK", "3"))
    SLEEP_STATUS_IDS = {"slp", "sleep"}
    SLEEP_CLAUSE_ENABLED = bool(int(os.getenv("ORANGURU_SLEEP_CLAUSE", "1")))
    STALL_SHUTDOWN_BOOST = bool(int(os.getenv("ORANGURU_STALL_SHUTDOWN_BOOST", "1")))
    AUTO_TERA = bool(int(os.getenv("ORANGURU_AUTO_TERA", "1")))
    SPEED_BOUNDS_ENABLED = bool(int(os.getenv("ORANGURU_SPEED_BOUNDS", "1")))
    _VOLATILE_RAW = {
        _maybe_effect("CONFUSION"): constants.CONFUSION,
        _maybe_effect("LEECH_SEED"): constants.LEECH_SEED,
        _maybe_effect("SUBSTITUTE"): constants.SUBSTITUTE,
        _maybe_effect("TAUNT"): constants.TAUNT,
        _maybe_effect("ENCORE"): "encore",
        _maybe_effect("LOCKED_MOVE"): constants.LOCKED_MOVE,
        _maybe_effect("YAWN"): constants.YAWN,
        _maybe_effect("SLOW_START"): constants.SLOW_START,
        _maybe_effect("PROTECT"): constants.PROTECT,
        _maybe_effect("BANEFUL_BUNKER"): constants.BANEFUL_BUNKER,
        _maybe_effect("SPIKY_SHIELD"): constants.SPIKY_SHIELD,
        _maybe_effect("SILK_TRAP"): constants.SILK_TRAP,
        _maybe_effect("ENDURE"): constants.ENDURE,
        _maybe_effect("PARTIALLY_TRAPPED"): constants.PARTIALLY_TRAPPED,
        _maybe_effect("ROOST"): constants.ROOST,
        _maybe_effect("DYNAMAX"): constants.DYNAMAX,
        _maybe_effect("TRANSFORM"): constants.TRANSFORM,
    }
    VOLATILE_EFFECT_MAP = {k: v for k, v in _VOLATILE_RAW.items() if k is not None}
    IMMUNITY_ABILITY_MAP = {
        "electric": {"lightningrod", "motordrive", "voltabsorb"},
        "water": {"waterabsorb", "stormdrain", "dryskin"},
        "fire": {"flashfire", "wellbakedbody"},
        "ground": {"levitate", "eartheater"},
        "grass": {"sapsipper"},
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._mcts_pool = None
        self._pool_workers = 0
        self._randbats_initialized = False
        self._randbats_gen = None
        self._randbats_sanitized = False
        self._mcts_stats = {
            "calls": 0,
            "states_sampled": 0,
            "results_kept": 0,
            "result_none": 0,
            "result_errors": 0,
            "empty_results": 0,
            "deterministic_decisions": 0,
            "stochastic_decisions": 0,
            "fallback_super": 0,
            "fallback_random": 0,
        }

    def get_mcts_stats(self) -> Dict[str, float]:
        stats = dict(self._mcts_stats)
        calls = max(1, int(stats.get("calls", 0)))
        sampled = max(1, int(stats.get("states_sampled", 0)))
        stats["empty_results_rate"] = float(stats.get("empty_results", 0)) / calls
        stats["fallback_super_rate"] = float(stats.get("fallback_super", 0)) / calls
        stats["fallback_random_rate"] = float(stats.get("fallback_random", 0)) / calls
        stats["state_failure_rate"] = float(
            stats.get("result_none", 0) + stats.get("result_errors", 0)
        ) / sampled
        return stats

    def _get_mcts_pool(self, desired_workers: int) -> Optional[ProcessPoolExecutor]:
        if desired_workers <= 1:
            return None
        if self._mcts_pool is None or self._pool_workers != desired_workers:
            if self._mcts_pool is not None:
                try:
                    self._mcts_pool.shutdown(wait=False, cancel_futures=True)
                except Exception:
                    pass
            self._mcts_pool = ProcessPoolExecutor(max_workers=desired_workers)
            self._pool_workers = desired_workers
        return self._mcts_pool

    def close(self):
        if self._mcts_pool is not None:
            try:
                self._mcts_pool.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass
            self._mcts_pool = None
            self._pool_workers = 0

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _status_to_fp(self, status) -> Optional[str]:
        if status is None:
            return None
        status_id = normalize_name(str(status))
        if status_id in {"slp", "sleep"}:
            return constants.SLEEP
        if status_id in {"brn", "burn"}:
            return constants.BURN
        if status_id in {"frz", "freeze"}:
            return constants.FROZEN
        if status_id in {"par", "paralysis"}:
            return constants.PARALYZED
        if status_id in {"psn", "poison"}:
            return constants.POISON
        if status_id in {"tox", "toxic"}:
            return constants.TOXIC
        return None

    def _weather_turns_remaining(self, battle: Battle) -> int:
        mem = self._get_battle_memory(battle)
        state = mem.get("weather_state")
        if not state or not battle.weather:
            return 0
        if not isinstance(state, dict):
            return -1
        weather_id = state.get("type")
        current_weather = self._map_weather(battle)
        mapped = None
        if weather_id:
            mapping = {
                "raindance": constants.RAIN,
                "rain": constants.RAIN,
                "sunnyday": constants.SUN,
                "sun": constants.SUN,
                "sandstorm": constants.SAND,
                "hail": constants.HAIL,
                "snowscape": constants.SNOW,
                "snow": constants.SNOW,
            }
            mapped = mapping.get(weather_id)
        if mapped and current_weather and mapped != current_weather:
            return -1
        start = state.get("start")
        duration = state.get("duration", 5)
        if not isinstance(start, int):
            return -1
        remaining = duration - max(0, battle.turn - start)
        return max(0, remaining)

    def _terrain_turns_remaining(self, battle: Battle) -> int:
        mem = self._get_battle_memory(battle)
        state = mem.get("terrain_state")
        if not state or not battle.fields:
            return 0
        if not isinstance(state, dict):
            return -1
        terrain_id = state.get("type")
        current_terrain = self._map_terrain(battle)
        mapped = None
        if terrain_id:
            mapping = {
                "electricterrain": constants.ELECTRIC_TERRAIN,
                "grassyterrain": constants.GRASSY_TERRAIN,
                "mistyterrain": constants.MISTY_TERRAIN,
                "psychicterrain": constants.PSYCHIC_TERRAIN,
            }
            mapped = mapping.get(terrain_id)
        if mapped and current_terrain and mapped != current_terrain:
            return -1
        start = state.get("start")
        duration = state.get("duration", 5)
        if not isinstance(start, int):
            return -1
        remaining = duration - max(0, battle.turn - start)
        return max(0, remaining)

    def _is_trapped(self, mon: Optional[Pokemon]) -> bool:
        if mon is None:
            return False
        effects = getattr(mon, "effects", None) or {}
        return Effect.TRAPPED in effects or Effect.PARTIALLY_TRAPPED in effects

    def _boosts_to_fp(self, boosts: Dict[str, int]) -> Dict[str, int]:
        result = {}
        if not boosts:
            return result
        mapping = {
            "atk": constants.ATTACK,
            "def": constants.DEFENSE,
            "spa": constants.SPECIAL_ATTACK,
            "spd": constants.SPECIAL_DEFENSE,
            "spe": constants.SPEED,
            "accuracy": "accuracy",
            "evasion": "evasion",
        }
        for key, val in boosts.items():
            fp_key = mapping.get(key)
            if fp_key:
                result[fp_key] = int(val)
        return result

    def _fill_moves_from_set(self, fp_mon: FPPokemon, set_info: dict, known_moves: set):
        if not set_info:
            return
        for move_id in set_info.get("moves", []):
            if not move_id:
                continue
            if move_id not in FP_MOVE_JSON:
                continue
            if fp_mon.get_move(move_id) is not None:
                continue
            if not known_moves or move_id in known_moves or len(known_moves) < 4:
                fp_mon.add_move(move_id)
        if len(fp_mon.moves) < 4:
            for move_id in set_info.get("moves", []):
                if move_id and fp_mon.get_move(move_id) is None:
                    if move_id not in FP_MOVE_JSON:
                        continue
                    fp_mon.add_move(move_id)
                if len(fp_mon.moves) >= 4:
                    break

    def _poke_env_to_fp(self, mon: Pokemon, battle: Battle, set_info: Optional[dict]) -> FPPokemon:
        species = normalize_name(getattr(mon, "species", "") or "")
        level = getattr(mon, "level", None) or (set_info.get("level") if set_info else 100)
        fp_mon = FPPokemon(species, level)

        ability = self._canon_id(getattr(mon, "ability", None)) if getattr(mon, "ability", None) else ""
        item = self._canon_id(getattr(mon, "item", None)) if getattr(mon, "item", None) else ""
        if set_info:
            if set_info.get("ability"):
                fp_mon.ability = set_info["ability"]
            if set_info.get("item"):
                fp_mon.item = set_info["item"]
            if set_info.get("tera"):
                fp_mon.tera_type = set_info["tera"]
        if ability:
            fp_mon.ability = ability
        if item:
            fp_mon.item = item
        tera_type = getattr(mon, "tera_type", None)
        if tera_type is not None:
            fp_mon.tera_type = normalize_name(tera_type.name if hasattr(tera_type, "name") else str(tera_type))
        terastallized = getattr(mon, "terastallized", None)
        if terastallized:
            fp_mon.terastallized = True
            if isinstance(terastallized, str):
                fp_mon.tera_type = normalize_name(terastallized)
        if getattr(mon, "is_terastallized", False):
            fp_mon.terastallized = True

        # Moves
        known_moves: List[str] = []
        for move in self._get_known_moves(mon):
            move_id = normalize_name(move.id)
            if move_id and move_id not in known_moves:
                known_moves.append(move_id)
                if move_id not in FP_MOVE_JSON:
                    continue
                fp_move = fp_mon.add_move(move_id)
                if fp_move is not None and hasattr(move, "current_pp"):
                    try:
                        fp_move.current_pp = int(move.current_pp)
                    except Exception:
                        pass
        if len(fp_mon.moves) < 4 and set_info:
            self._fill_moves_from_set(fp_mon, set_info, set(known_moves))

        # HP
        max_hp = getattr(mon, "max_hp", None)
        current_hp = getattr(mon, "current_hp", None)
        if max_hp:
            fp_mon.max_hp = int(max_hp)
            if current_hp is not None:
                fp_mon.hp = int(current_hp)
            elif mon.current_hp_fraction is not None:
                fp_mon.hp = int(fp_mon.max_hp * mon.current_hp_fraction)
        elif mon.current_hp_fraction is not None and fp_mon.max_hp:
            fp_mon.hp = max(1, int(fp_mon.max_hp * mon.current_hp_fraction))

        # Stats
        stats = getattr(mon, "stats", None) or {}
        if stats:
            stat_map = {
                constants.ATTACK: stats.get("atk"),
                constants.DEFENSE: stats.get("def"),
                constants.SPECIAL_ATTACK: stats.get("spa"),
                constants.SPECIAL_DEFENSE: stats.get("spd"),
                constants.SPEED: stats.get("spe"),
            }
            for key, value in stat_map.items():
                if value is not None:
                    fp_mon.stats[key] = int(value)
            if stats.get("hp") is not None:
                fp_mon.max_hp = int(stats["hp"])
                if current_hp is not None:
                    fp_mon.hp = int(current_hp)
                elif mon.current_hp_fraction is not None:
                    fp_mon.hp = int(fp_mon.max_hp * mon.current_hp_fraction)

        # Types (honor temporary/tera types from poke_env)
        try:
            type_1 = getattr(mon, "type_1", None)
            type_2 = getattr(mon, "type_2", None)
            types = []
            if type_1 is not None:
                types.append(normalize_name(type_1.name if hasattr(type_1, "name") else str(type_1)))
            if type_2 is not None:
                types.append(normalize_name(type_2.name if hasattr(type_2, "name") else str(type_2)))
            if types:
                fp_mon.types = tuple(types)
        except Exception:
            pass

        if getattr(mon, "fainted", False):
            fp_mon.hp = 0
            fp_mon.fainted = True

        # Status
        fp_status = self._status_to_fp(getattr(mon, "status", None))
        if fp_status:
            fp_mon.status = fp_status
            if fp_status == constants.SLEEP:
                try:
                    fp_mon.sleep_turns = int(getattr(mon, "status_counter", 0))
                except Exception:
                    fp_mon.sleep_turns = 0

        # Boosts
        boosts = self._boosts_to_fp(getattr(mon, "boosts", {}) or {})
        for key, val in boosts.items():
            fp_mon.boosts[key] = val

        # Volatile statuses
        effects = getattr(mon, "effects", None) or {}
        for effect, count in effects.items():
            mapped = self.VOLATILE_EFFECT_MAP.get(effect)
            if not mapped:
                continue
            if mapped not in fp_mon.volatile_statuses:
                fp_mon.volatile_statuses.append(mapped)
            if count is not None:
                try:
                    fp_mon.volatile_status_durations[mapped] = int(count)
                except Exception:
                    pass
        # Poke-engine rejects taunt duration 3; clamp to a safe range.
        if constants.TAUNT in fp_mon.volatile_statuses:
            try:
                taunt_turns = int(fp_mon.volatile_status_durations[constants.TAUNT])
                fp_mon.volatile_status_durations[constants.TAUNT] = max(0, min(2, taunt_turns))
            except Exception:
                fp_mon.volatile_status_durations[constants.TAUNT] = 2

        # Tera
        if getattr(mon, "terastallized", False):
            fp_mon.terastallized = True
            if getattr(mon, "tera_type", None):
                fp_mon.tera_type = normalize_name(mon.tera_type.name)

        # Substitute state (track whether it took a hit)
        try:
            mem = self._get_battle_memory(battle)
            sub_state = mem.get("substitute_state", {})
            side_key = "self" if mon is battle.active_pokemon or mon in battle.team.values() else "opp"
            entry = sub_state.get(side_key, {}).get(normalize_name(fp_mon.name))
            if entry and Effect.SUBSTITUTE in (getattr(mon, "effects", None) or {}):
                fp_mon.substitute_hit = bool(entry.get("hit", False))
        except Exception:
            pass

        return fp_mon

    def _map_side_conditions(self, src: dict, dest: dict) -> None:
        if not src:
            return
        if SideCondition.SPIKES in src:
            dest[constants.SPIKES] = int(src[SideCondition.SPIKES])
        if SideCondition.STEALTH_ROCK in src:
            dest[constants.STEALTH_ROCK] = int(src[SideCondition.STEALTH_ROCK])
        if SideCondition.TOXIC_SPIKES in src:
            dest[constants.TOXIC_SPIKES] = int(src[SideCondition.TOXIC_SPIKES])
        if SideCondition.STICKY_WEB in src:
            dest[constants.STICKY_WEB] = int(src[SideCondition.STICKY_WEB])
        if SideCondition.REFLECT in src:
            dest[constants.REFLECT] = int(src[SideCondition.REFLECT])
        if SideCondition.LIGHT_SCREEN in src:
            dest[constants.LIGHT_SCREEN] = int(src[SideCondition.LIGHT_SCREEN])
        if SideCondition.AURORA_VEIL in src:
            dest[constants.AURORA_VEIL] = int(src[SideCondition.AURORA_VEIL])
        if SideCondition.TAILWIND in src:
            dest[constants.TAILWIND] = int(src[SideCondition.TAILWIND])
        if SideCondition.SAFEGUARD in src:
            dest[constants.SAFEGUARD] = int(src[SideCondition.SAFEGUARD])
        if SideCondition.MIST in src:
            dest[constants.MIST] = int(src[SideCondition.MIST])

    def _map_weather(self, battle: Battle) -> Optional[str]:
        weather = getattr(battle, "weather", None)
        if not weather:
            return None
        if isinstance(weather, dict):
            if not weather:
                return None
            weather = next(iter(weather.keys()))
        if isinstance(weather, Weather):
            name = normalize_name(weather.name)
        else:
            name = normalize_name(str(weather))
        mapping = {
            "raindance": constants.RAIN,
            "rain": constants.RAIN,
            "sunnyday": constants.SUN,
            "sun": constants.SUN,
            "sandstorm": constants.SAND,
            "hail": constants.HAIL,
            "snowscape": constants.SNOW,
        }
        return mapping.get(name, None)

    def _map_terrain(self, battle: Battle) -> Optional[str]:
        fields = getattr(battle, "fields", None) or {}
        for field in fields:
            if field == Field.ELECTRIC_TERRAIN:
                return constants.ELECTRIC_TERRAIN
            if field == Field.GRASSY_TERRAIN:
                return constants.GRASSY_TERRAIN
            if field == Field.MISTY_TERRAIN:
                return constants.MISTY_TERRAIN
            if field == Field.PSYCHIC_TERRAIN:
                return constants.PSYCHIC_TERRAIN
        return None

    def _clear_invalid_encore(self, battler: Battler) -> None:
        if battler.active is None:
            return
        if "encore" not in battler.active.volatile_statuses:
            return
        last_move = getattr(battler.last_used_move, "move", "") or ""
        if not last_move or last_move.startswith("switch"):
            battler.active.volatile_statuses = [
                v for v in battler.active.volatile_statuses if v != "encore"
            ]
            battler.active.volatile_status_durations["encore"] = 0

    def _sleep_clause_active(self, battle: Battle) -> bool:
        if not self.SLEEP_CLAUSE_ENABLED:
            return False
        fmt = normalize_name(getattr(battle, "format", "") or "")
        return "randombattle" in fmt

    def _opponent_has_sleeping_mon(self, battle: Battle) -> bool:
        for mon in battle.opponent_team.values():
            if mon is None:
                continue
            status = getattr(mon, "status", None)
            if status is None:
                continue
            status_id = getattr(status, "value", None) or normalize_name(str(status))
            if status_id in self.SLEEP_STATUS_IDS:
                return True
        opp_active = getattr(battle, "opponent_active_pokemon", None)
        if opp_active is not None:
            status = getattr(opp_active, "status", None)
            if status is not None:
                status_id = getattr(status, "value", None) or normalize_name(str(status))
                if status_id in self.SLEEP_STATUS_IDS:
                    return True
        return False

    def _sleep_clause_blocked(self, battle: Battle) -> bool:
        return self._sleep_clause_active(battle) and self._opponent_has_sleeping_mon(battle)

    def _move_inflicts_sleep(self, move) -> bool:
        if move is None:
            return False
        entry = self._get_move_entry(move)
        if entry.get("self", {}).get("status") == "slp":
            return False
        status = normalize_name(entry.get("status", ""))
        if status in self.SLEEP_STATUS_IDS:
            return True
        status_type = self.STATUS_MOVES.get(move.id)
        return status_type == "sleep"

    def _fp_move_inflicts_sleep(self, move_id: str) -> bool:
        entry = FP_MOVE_JSON.get(move_id, {})
        if entry.get("self", {}).get("status") == "slp":
            return False
        status = normalize_name(entry.get("status", ""))
        return status in self.SLEEP_STATUS_IDS

    def _sleep_clause_banned_choices(self, battle: Battle) -> set:
        if not self._sleep_clause_blocked(battle):
            return set()
        banned = set()
        for move in battle.available_moves or []:
            if self._move_inflicts_sleep(move):
                banned.add(move.id)
                banned.add(f"{move.id}-tera")
        return banned

    def _apply_opponent_item_flags(self, fp_mon: FPPokemon, battle: Battle) -> None:
        mem = self._get_battle_memory(battle)
        flags = mem.get("opponent_item_flags", {}).get(normalize_name(fp_mon.name))
        if not flags:
            return
        if flags.get("no_choice"):
            fp_mon.can_have_choice_item = False
        if flags.get("no_assaultvest"):
            fp_mon.impossible_items.add("assaultvest")
        if flags.get("no_boots"):
            fp_mon.impossible_items.add("heavydutyboots")
        if flags.get("has_boots"):
            fp_mon.item = "heavydutyboots"
            fp_mon.item_inferred = True
        known_item = flags.get("known_item")
        if known_item:
            fp_mon.item = known_item
            fp_mon.item_inferred = True
        removed_item = flags.get("removed_item")
        if removed_item:
            fp_mon.removed_item = removed_item
            if fp_mon.item in {"", constants.UNKNOWN_ITEM}:
                fp_mon.knocked_off = True

    def _apply_opponent_ability_flags(self, fp_mon: FPPokemon, battle: Battle) -> None:
        mem = self._get_battle_memory(battle)
        ability = mem.get("opponent_abilities", {}).get(normalize_name(fp_mon.name))
        if ability:
            fp_mon.ability = ability
            if not fp_mon.original_ability:
                fp_mon.original_ability = ability
        impossible = mem.get("opponent_impossible_abilities", {}).get(normalize_name(fp_mon.name))
        if impossible:
            fp_mon.impossible_abilities.update(impossible)

    def _apply_known_opponent_moves(self, fp_mon: FPPokemon, battle: Battle) -> None:
        mem = self._get_battle_memory(battle)
        moves = mem.get("opponent_moves", {}).get(normalize_name(fp_mon.name), set())
        if not moves:
            return
        for move_id in moves:
            if len(fp_mon.moves) >= 4:
                break
            if move_id in FP_MOVE_JSON and fp_mon.get_move(move_id) is None:
                fp_mon.add_move(move_id)

    def _apply_opponent_switch_memory(self, fp_mon: FPPokemon, battle: Battle) -> None:
        mem = self._get_battle_memory(battle)
        species = normalize_name(fp_mon.name)
        info = mem.get("opp_switch_info", {}).get(species)
        if info:
            hp_at_switch = info.get("hp")
            if isinstance(hp_at_switch, (int, float)) and hp_at_switch > 0:
                try:
                    fp_mon.hp_at_switch_in = int(hp_at_switch)
                except Exception:
                    pass
            status_at_switch = info.get("status")
            if status_at_switch:
                fp_status = self._status_to_fp(status_at_switch)
                if fp_status:
                    fp_mon.status_at_switch_in = fp_status
        moves_since = mem.get("opp_moves_since_switch", {}).get(species, set())
        if moves_since:
            fp_mon.moves_used_since_switch_in = set(moves_since)
            if len(moves_since) >= 2:
                fp_mon.can_have_choice_item = False

    def _apply_speed_bounds(self, fp_mon: FPPokemon, battle: Battle) -> None:
        if not self.SPEED_BOUNDS_ENABLED:
            return
        mem = self._get_battle_memory(battle)
        bounds = mem.get("speed_bounds", {}).get(normalize_name(fp_mon.name))
        if not bounds:
            return
        min_speed = bounds.get("min", 0)
        max_speed = bounds.get("max", float("inf"))
        if min_speed <= 0 and max_speed == float("inf"):
            return
        if min_speed > max_speed:
            return
        fp_mon.speed_range = StatRange(min=min_speed, max=max_speed)

    def _ensure_randbats_sets(self, battle: Battle) -> str:
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

    def _sanitize_randbats_moves(self) -> None:
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

    def _belief_weight_for_set(
        self,
        fp_mon: FPPokemon,
        set_info: dict,
        battle: Battle,
    ) -> float:
        species = normalize_name(fp_mon.name)
        mem = self._get_battle_memory(battle)
        immune_types = mem.get("immune_types", {}).get(species, set())
        if not immune_types:
            return 1.0
        ability = normalize_name(set_info.get("ability", "") or "")
        if not ability:
            return 1.0
        types = set(fp_mon.types or [])
        multiplier = 1.0
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
        return multiplier

    def _candidate_randombattle_sets(self, opponent: Pokemon, battle: Battle) -> List[Tuple[dict, float]]:
        if not self.BELIEF_SAMPLING:
            return super()._candidate_randombattle_sets(opponent, battle)
        if opponent is None:
            return []
        try:
            self._ensure_randbats_sets(battle)
        except Exception:
            return super()._candidate_randombattle_sets(opponent, battle)

        fp_mon = self._poke_env_to_fp(opponent, battle, None)
        self._apply_opponent_item_flags(fp_mon, battle)
        self._apply_opponent_ability_flags(fp_mon, battle)
        self._apply_known_opponent_moves(fp_mon, battle)
        self._apply_speed_bounds(fp_mon, battle)

        predicted_sets = RandomBattleTeamDatasets.get_all_remaining_sets(fp_mon)
        if not predicted_sets:
            return super()._candidate_randombattle_sets(opponent, battle)

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
            weight = weight * self._belief_weight_for_set(fp_mon, set_info, battle)
            if weight <= 0:
                continue
            candidates.append((set_info, weight))

        if not candidates:
            return super()._candidate_randombattle_sets(opponent, battle)

        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:20]

    def _extract_last_opponent_move(self, battle: Battle) -> Optional[Tuple[str, str, int]]:
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

    def _sample_set_for_species(
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

    def _sample_unknown_opponents(
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

    def _build_fp_battle(self, battle: Battle, seed: int, fill_opponent_sets: bool = False) -> FPBattle:
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
        # User team
        active = battle.active_pokemon
        try:
            user.trapped = bool(getattr(battle, "trapped", False))
        except Exception:
            user.trapped = False
        if active and self._is_trapped(active):
            user.trapped = True

        # Side conditions
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

        # Opponent team
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

    @staticmethod
    def _fp_move_id(move) -> str:
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

    def _heuristic_action_score(self, battle: Battle, choice: str) -> Optional[float]:
        active = battle.active_pokemon
        opponent = battle.opponent_active_pokemon
        if active is None or opponent is None:
            return None

        if choice.startswith("switch "):
            target = normalize_name(choice.split("switch ", 1)[1])
            for sw in battle.available_switches:
                if normalize_name(sw.species) == target:
                    return max(0.0, self._score_switch(sw, opponent, battle))
            return None

        move_id = normalize_name(choice.replace("-tera", ""))
        for move in battle.available_moves:
            if move.id != move_id:
                continue
            if move.id in self.PROTECT_MOVES:
                reply_score = self._estimate_best_reply_score(opponent, active, battle)
                if self._should_use_protect(battle, reply_score):
                    hp_frac = active.current_hp_fraction or 0.0
                    threshold = max(1.0, 220.0 * max(hp_frac, 0.1))
                    danger = min(1.0, reply_score / threshold)
                    return max(0.0, 120.0 * danger)
                return 0.0
            if self._is_recovery_move(move):
                reply_score = self._estimate_best_reply_score(opponent, active, battle)
                safe_recover = reply_score < 150
                if self._estimate_matchup(active, opponent) > 0.35 and (active.current_hp_fraction or 0.0) < 0.4:
                    safe_recover = True
                if getattr(self, "RECOVERY_KO_GUARD", False):
                    best_damage = self._estimate_best_damage_score(active, opponent, battle)
                    opp_hp = opponent.current_hp_fraction or 0.0
                    threshold = getattr(self, "RECOVERY_KO_THRESHOLD", 220.0) * max(opp_hp, 0.05)
                    if best_damage >= threshold:
                        return 0.0
                hp_frac = active.current_hp_fraction or 0.0
                if hp_frac < 0.65 and safe_recover:
                    missing = 1.0 - hp_frac
                    return max(0.0, 140.0 * missing)
                return 0.0
            if self._sleep_clause_blocked(battle) and self._move_inflicts_sleep(move):
                return 0.0
            if move.category == MoveCategory.STATUS:
                if self.STATUS_KO_GUARD:
                    opp_hp = opponent.current_hp_fraction or 0.0
                    best_damage = self._estimate_best_damage_score(active, opponent, battle)
                    threshold = self.STATUS_KO_THRESHOLD * max(opp_hp, 0.05)
                    if best_damage >= threshold:
                        return 0.0
                boosts = getattr(move, "boosts", None) or {}
                if boosts and move.target and "self" in str(move.target).lower():
                    if self._should_setup_move(move, active, opponent):
                        matchup = self._estimate_matchup(active, opponent)
                        hp_frac = active.current_hp_fraction or 0.0
                        base = 80.0 + 40.0 * max(0.0, min(1.0, matchup + 0.5))
                        return max(0.0, base * max(0.0, min(1.0, hp_frac + 0.1)))
                score = self._should_use_status_move(move, active, opponent, battle)
                status_type = self.STATUS_MOVES.get(move.id)
                if status_type is None:
                    status_type = self._status_from_move_entry(self._get_move_entry(move))
                if (
                    self.STALL_SHUTDOWN_BOOST
                    and (status_type in {"taunt", "encore"} or move.id in self.ANTI_SETUP_MOVES)
                    and (
                        self._opponent_is_stallish(opponent)
                        or (opponent.boosts and sum(opponent.boosts.values()) > 0)
                    )
                ):
                    score *= 1.25
                return max(0.0, score)
            predicted_switch = self._predict_opponent_switch(battle)
            return max(
                0.0,
                self._score_move_with_prediction(
                    move, active, opponent, predicted_switch, battle
                ),
            )
        return None

    def _select_move_from_results(
        self,
        results: List[Tuple[object, float]],
        battle: Battle,
        banned_choices: Optional[set] = None,
    ) -> str:
        battle_tag = str(getattr(battle, "battle_tag", "")).lower()
        is_eval_tag = any(key in battle_tag for key in ("eval", "evaluation", "heuristic", "oranguru"))
        env_eval_mode = os.getenv("ORANGURU_EVAL_MODE", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        deterministic = self.MCTS_DETERMINISTIC and (
            not self.MCTS_DETERMINISTIC_EVAL_ONLY or is_eval_tag or env_eval_mode
        )
        if deterministic:
            self._mcts_stats["deterministic_decisions"] += 1
        else:
            self._mcts_stats["stochastic_decisions"] += 1
        final_policy = {}
        for res, weight in results:
            total_visits = res.total_visits or 1
            for opt in res.side_one:
                final_policy[opt.move_choice] = final_policy.get(opt.move_choice, 0.0) + (
                    weight * (opt.visits / total_visits)
                )
        if not final_policy:
            return ""
        if banned_choices:
            filtered_policy = {k: v for k, v in final_policy.items() if k not in banned_choices}
            if filtered_policy:
                final_policy = filtered_policy
        ordered = sorted(final_policy.items(), key=lambda x: x[1], reverse=True)
        total_policy = sum(w for _, w in ordered)
        if total_policy <= 0:
            return ordered[0][0]

        def _pick_choice(choices: List[str], weights: List[float]) -> str:
            if not choices:
                return ""
            total = sum(weights) if weights else 0.0
            if deterministic:
                if total <= 0:
                    return choices[0]
                best_idx = max(range(len(choices)), key=lambda i: weights[i])
                return choices[best_idx]
            if total <= 0:
                return random.choice(choices)
            return random.choices(choices, weights=weights, k=1)[0]

        best = ordered[0][1]
        confidence = best / total_policy if total_policy > 0 else 0.0
        if len(ordered) > 1:
            second = ordered[1][1]
            margin = (best - second) / total_policy if total_policy > 0 else 0.0
            confidence = max(confidence, margin)

        cutoff = best * 0.75
        filtered = [o for o in ordered if o[1] >= cutoff]

        threshold = max(0.0, min(1.0, self.MCTS_CONFIDENCE_THRESHOLD))
        if self.SELECTION_MODE == "policy":
            cutoff_ratio = max(0.0, min(1.0, self.POLICY_CUTOFF))
            cutoff = best * cutoff_ratio
            policy_choices = [(choice, weight) for choice, weight in ordered if weight >= cutoff]
            if not policy_choices:
                policy_choices = [ordered[0]]
            choices = [choice for choice, _ in policy_choices]
            policy_weights = [max(0.0, weight) for _, weight in policy_choices]
            if confidence < threshold:
                heuristic_weights = []
                for choice in choices:
                    score = self._heuristic_action_score(battle, choice)
                    heuristic_weights.append(max(0.0, score or 0.0))
                heur_total = sum(heuristic_weights)
                if heur_total > 0:
                    return _pick_choice(choices, heuristic_weights)
            return _pick_choice(choices, policy_weights)

        if self.SELECTION_MODE == "rerank" and confidence < threshold:
            candidates = filtered[: max(1, self.RERANK_TOPK)]
            scored = []
            for choice, weight in candidates:
                score = self._heuristic_action_score(battle, choice)
                if score is None:
                    continue
                scored.append((score, weight, choice))
            if scored:
                scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
                return scored[0][2]

        blend = max(0.0, min(1.0, self.HEURISTIC_BLEND))
        min_blend = max(0.0, min(1.0, self.MIN_HEURISTIC_BLEND))
        if self.GATE_MODE == "entropy":
            probs = [w / total_policy for _, w in ordered]
            if len(probs) <= 1:
                normalized_entropy = 0.0
            else:
                entropy = -sum(p * math.log(p) for p in probs if p > 0)
                normalized_entropy = entropy / math.log(len(probs))
            if threshold > 0:
                gate = min(1.0, normalized_entropy / threshold)
            else:
                gate = normalized_entropy
            blend *= gate
        elif self.GATE_MODE == "soft":
            if threshold > 0:
                gate = max(0.0, min(1.0, (threshold - confidence) / threshold))
            else:
                gate = 1.0
            blend *= gate
        else:
            confidence = best / total_policy if total_policy > 0 else 0.0
            if len(ordered) > 1:
                second = ordered[1][1]
                margin = (best - second) / total_policy if total_policy > 0 else 0.0
                confidence = max(confidence, margin)
            if confidence >= threshold:
                blend = 0.0
        if min_blend > 0.0:
            blend = max(blend, min_blend)

        choices = []
        mcts_weights = []
        for choice, weight in filtered:
            choices.append(choice)
            mcts_weights.append(max(0.0, weight))

        if not choices:
            return ordered[0][0]

        mcts_total = sum(mcts_weights)
        if mcts_total <= 0:
            mcts_norm = [1.0 / len(choices)] * len(choices)
        else:
            mcts_norm = [w / mcts_total for w in mcts_weights]

        if blend <= 0:
            combined = mcts_norm
        else:
            heuristic_weights = []
            for choice in choices:
                score = self._heuristic_action_score(battle, choice)
                heuristic_weights.append(max(0.0, score or 0.0))
            heur_total = sum(heuristic_weights)
            if heur_total <= 0:
                combined = mcts_norm
            else:
                heur_norm = [w / heur_total for w in heuristic_weights]
                combined = [
                    (1.0 - blend) * m + blend * h for m, h in zip(mcts_norm, heur_norm)
                ]

        return _pick_choice(choices, combined)

    @staticmethod
    def _empty_order_if_no_choices(battle: Battle):
        # Compatibility guard: some engine branches call this helper directly.
        try:
            orders = getattr(battle, "valid_orders", None)
            if orders is not None and len(orders) == 0:
                from poke_env.player.battle_order import _EmptyBattleOrder

                return _EmptyBattleOrder()
        except Exception:
            pass
        return None

    def _is_switch_churn_risk(self, battle: Battle) -> bool:
        # Compatibility shim for mixed commit states where RuleBot may or may not
        # define anti-switch-churn logic.
        try:
            checker = getattr(RuleBotPlayer, "_is_switch_churn_risk", None)
            if checker is None:
                return False
            return bool(checker(self, battle))
        except Exception:
            return False

    def choose_move(self, battle: AbstractBattle):
        if not isinstance(battle, Battle):
            return self.choose_random_move(battle)
        noop_order = self._empty_order_if_no_choices(battle)
        if noop_order is not None:
            return noop_order
        if getattr(battle, "_wait", False) or getattr(battle, "teampreview", False):
            return self.choose_random_move(battle)

        self._current_battle = battle
        self._update_immunity_memory(battle)
        self._update_active_turns(battle)
        self._update_battle_memory(battle)
        self._update_speed_order_memory(battle)
        self._update_switch_in_memory(battle)
        self._update_opponent_item_memory(battle)
        self._update_opponent_move_history(battle)
        self._update_opponent_ability_memory(battle)
        self._update_opponent_ability_constraints(battle)
        if self.IMMUNITY_INFER:
            self._infer_ability_from_immunity(battle)
        self._update_field_memory(battle)
        self._update_switch_flags(battle)
        self._update_substitute_memory(battle)
        self._cleanup_battle_memory(battle)

        active = battle.active_pokemon
        opponent = battle.opponent_active_pokemon
        if active is None or opponent is None:
            return self.choose_random_move(battle)

        if battle.force_switch:
            if battle.available_switches:
                best_switch = max(
                    battle.available_switches,
                    key=lambda s: self._score_switch(s, opponent, battle),
                )
                return self._commit_order(battle, self.create_order(best_switch))
            return self.choose_random_move(battle)

        self._mcts_stats["calls"] += 1
        fp_battles = []
        weights = []
        sample_states = self.SAMPLE_STATES
        search_time_ms = self.SEARCH_TIME_MS
        if self.DYNAMIC_SAMPLING:
            opp_known_moves = len(self._get_known_moves(opponent))
            revealed = len([m for m in battle.opponent_team.values() if m is not None])
            opp_hp = opponent.current_hp_fraction or 0.0
            time_remaining = getattr(battle, "time_remaining", None)
            in_time_pressure = time_remaining is not None and time_remaining <= 60

            if revealed <= 3 and opp_hp > 0 and opp_known_moves == 0:
                multiplier = 2 if in_time_pressure else 4
                sample_states = max(sample_states, self.PARALLELISM * multiplier)
                search_time_ms = max(80, int(self.SEARCH_TIME_MS * 0.5))
            else:
                multiplier = 1 if in_time_pressure else 2
                sample_states = max(sample_states, self.PARALLELISM * multiplier)

            sample_states = min(self.MAX_SAMPLE_STATES, sample_states)

        base_fp_battle = self._build_fp_battle(battle, seed=0, fill_opponent_sets=False)
        if base_fp_battle.battle_type == constants.BattleType.RANDOM_BATTLE:
            try:
                self._ensure_randbats_sets(battle)
                samples = prepare_random_battles(base_fp_battle, sample_states)
                fp_battles = [b for b, _ in samples]
                weights = [w for _, w in samples]
            except Exception:
                fp_battles = [base_fp_battle]
                weights = [1.0]
        else:
            seeds = [random.randint(1, 1_000_000) for _ in range(sample_states)]
            for seed in seeds:
                fp_battles.append(self._build_fp_battle(battle, seed, fill_opponent_sets=True))
            weights = [1.0 / len(fp_battles)] * len(fp_battles)

        states = [battle_to_poke_engine_state(b).to_string() for b in fp_battles]
        self._mcts_stats["states_sampled"] += len(states)

        results = []
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
            for fut, weight in zip(futures, weights):
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
                self._mcts_stats["results_kept"] += 1
        else:
            for state, weight in zip(states, weights):
                res = _run_mcts(state, search_time_ms)
                if res is None:
                    self._mcts_stats["result_none"] += 1
                    continue
                results.append((res, weight))
                self._mcts_stats["results_kept"] += 1

        if not results:
            self._mcts_stats["empty_results"] += 1
            self._mcts_stats["fallback_super"] += 1
            return super().choose_move(battle)

        banned_choices = self._sleep_clause_banned_choices(battle)
        choice = self._select_move_from_results(results, battle, banned_choices=banned_choices)
        if not choice:
            self._mcts_stats["fallback_super"] += 1
            return super().choose_move(battle)

        if choice.startswith("switch "):
            if self._is_switch_churn_risk(battle) and battle.available_moves:
                emergency_order = self._choose_emergency_non_switch_order(
                    battle,
                    active,
                    opponent,
                    len([m for m in battle.team.values() if not m.fainted]),
                )
                if emergency_order is not None:
                    mem = self._get_battle_memory(battle)
                    mem["switch_churn_breaks"] = int(mem.get("switch_churn_breaks", 0) or 0) + 1
                    return self._commit_order(battle, emergency_order)
            switch_name = normalize_name(choice.split("switch ", 1)[1])
            for sw in battle.available_switches:
                if normalize_name(sw.species) == switch_name:
                    return self._commit_order(battle, self.create_order(sw))
            self._mcts_stats["fallback_random"] += 1
            return self.choose_random_move(battle)

        tera = False
        if choice.endswith("-tera"):
            choice = choice.replace("-tera", "")
            tera = bool(getattr(battle, "can_tera", False))
        move_id = normalize_name(choice)
        for move in battle.available_moves:
            if move.id == move_id:
                return self._commit_order(
                    battle,
                    self.create_order(
                        move,
                        terastallize=(
                            tera
                            or (
                                self.AUTO_TERA
                                and getattr(battle, "can_tera", False)
                                and self._should_terastallize(battle, move)
                            )
                        ),
                        dynamax=self._should_dynamax(battle, len([m for m in battle.team.values() if not m.fainted])),
                    ),
                )
        self._mcts_stats["fallback_random"] += 1
        return self.choose_random_move(battle)
