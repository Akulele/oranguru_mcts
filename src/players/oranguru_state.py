from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

from poke_env.battle import Battle, Pokemon, SideCondition, Field
from poke_env.battle.effect import Effect
from poke_env.battle.weather import Weather

from src.players import oranguru_belief
from src.players import oranguru_memory
from src.players import oranguru_tactical
from src.players import oranguru_worlds
from src.utils.damage_calc import normalize_name

import constants
from fp.battle import Battle as FPBattle, Battler, Pokemon as FPPokemon, StatRange
from data import all_move_json as FP_MOVE_JSON


def status_to_fp(self, status) -> Optional[str]:
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


def is_trapped(self, mon: Optional[Pokemon]) -> bool:
    if mon is None:
        return False
    effects = getattr(mon, "effects", None) or {}
    return Effect.TRAPPED in effects or Effect.PARTIALLY_TRAPPED in effects


def boosts_to_fp(self, boosts: Dict[str, int]) -> Dict[str, int]:
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


def fill_moves_from_set(self, fp_mon: FPPokemon, set_info: dict, known_moves: set):
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


def poke_env_to_fp(self, mon: Pokemon, battle: Battle, set_info: Optional[dict]) -> FPPokemon:
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

    fp_status = self._status_to_fp(getattr(mon, "status", None))
    if fp_status:
        fp_mon.status = fp_status
        if fp_status == constants.SLEEP:
            try:
                fp_mon.sleep_turns = int(getattr(mon, "status_counter", 0))
            except Exception:
                fp_mon.sleep_turns = 0

    boosts = self._boosts_to_fp(getattr(mon, "boosts", {}) or {})
    for key, val in boosts.items():
        fp_mon.boosts[key] = val

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
    if constants.TAUNT in fp_mon.volatile_statuses:
        try:
            taunt_turns = int(fp_mon.volatile_status_durations[constants.TAUNT])
            fp_mon.volatile_status_durations[constants.TAUNT] = max(0, min(2, taunt_turns))
        except Exception:
            fp_mon.volatile_status_durations[constants.TAUNT] = 2

    if getattr(mon, "terastallized", False):
        fp_mon.terastallized = True
        if getattr(mon, "tera_type", None):
            fp_mon.tera_type = normalize_name(mon.tera_type.name)

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


def map_side_conditions(self, src: dict, dest: dict) -> None:
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


def map_weather(self, battle: Battle) -> Optional[str]:
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


def map_terrain(self, battle: Battle) -> Optional[str]:
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


def clear_invalid_encore(self, battler: Battler) -> None:
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


def sleep_clause_active(self, battle: Battle) -> bool:
    if not self.SLEEP_CLAUSE_ENABLED:
        return False
    fmt = normalize_name(getattr(battle, "format", "") or "")
    return "randombattle" in fmt


def opponent_has_sleeping_mon(self, battle: Battle) -> bool:
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


def sleep_clause_blocked(self, battle: Battle) -> bool:
    return self._sleep_clause_active(battle) and self._opponent_has_sleeping_mon(battle)


def move_inflicts_sleep(self, move) -> bool:
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


def fp_move_inflicts_sleep(self, move_id: str) -> bool:
    entry = FP_MOVE_JSON.get(move_id, {})
    if entry.get("self", {}).get("status") == "slp":
        return False
    status = normalize_name(entry.get("status", ""))
    return status in self.SLEEP_STATUS_IDS


def sleep_clause_banned_choices(self, battle: Battle) -> set:
    if not self._sleep_clause_blocked(battle):
        return set()
    banned = set()
    for move in battle.available_moves or []:
        if self._move_inflicts_sleep(move):
            banned.add(move.id)
            banned.add(f"{move.id}-tera")
    return banned


def apply_opponent_item_flags(self, fp_mon: FPPokemon, battle: Battle) -> None:
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


def apply_opponent_ability_flags(self, fp_mon: FPPokemon, battle: Battle) -> None:
    mem = self._get_battle_memory(battle)
    ability = mem.get("opponent_abilities", {}).get(normalize_name(fp_mon.name))
    if ability:
        fp_mon.ability = ability
        if not fp_mon.original_ability:
            fp_mon.original_ability = ability
    impossible = mem.get("opponent_impossible_abilities", {}).get(normalize_name(fp_mon.name))
    if impossible:
        fp_mon.impossible_abilities.update(impossible)


def apply_known_opponent_moves(self, fp_mon: FPPokemon, battle: Battle) -> None:
    mem = self._get_battle_memory(battle)
    moves = mem.get("opponent_moves", {}).get(normalize_name(fp_mon.name), set())
    if not moves:
        return
    for move_id in moves:
        if len(fp_mon.moves) >= 4:
            break
        if move_id in FP_MOVE_JSON and fp_mon.get_move(move_id) is None:
            fp_mon.add_move(move_id)


def apply_opponent_switch_memory(self, fp_mon: FPPokemon, battle: Battle) -> None:
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


def apply_speed_bounds(self, fp_mon: FPPokemon, battle: Battle) -> None:
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


def side_hazard_pressure(self, battle: Battle) -> float:
    side_conditions = getattr(battle, "side_conditions", None) or {}
    pressure = 0.0
    if SideCondition.STEALTH_ROCK in side_conditions:
        pressure += 0.125
    spikes_layers = int(side_conditions.get(SideCondition.SPIKES, 0) or 0)
    if spikes_layers == 1:
        pressure += 0.125
    elif spikes_layers == 2:
        pressure += 1.0 / 6.0
    elif spikes_layers >= 3:
        pressure += 0.25
    tspikes_layers = int(side_conditions.get(SideCondition.TOXIC_SPIKES, 0) or 0)
    if tspikes_layers > 0:
        pressure += 0.08
    if SideCondition.STICKY_WEB in side_conditions:
        pressure += 0.05
    return pressure


def opponent_progress_markers(self, battle: Battle, opponent: Optional[Pokemon]) -> dict:
    status = normalize_name(str(getattr(opponent, "status", "") or "")) if opponent is not None else ""
    opp_sc = getattr(battle, "opponent_side_conditions", None) or {}
    return {
        "status": status,
        "rocks": int(SideCondition.STEALTH_ROCK in opp_sc),
        "web": int(SideCondition.STICKY_WEB in opp_sc),
        "spikes": int(opp_sc.get(SideCondition.SPIKES, 0) or 0),
        "tspikes": int(opp_sc.get(SideCondition.TOXIC_SPIKES, 0) or 0),
    }


def resolve_passive_progress(self, battle: Battle) -> None:
    return oranguru_tactical.resolve_passive_progress(self, battle)


def passive_choice_kind(self, move) -> str:
    return oranguru_tactical.passive_choice_kind(self, move)


def progress_need_score(self, battle: Battle, active: Pokemon, opponent: Pokemon, best_damage_score: float) -> int:
    return oranguru_tactical.progress_need_score(self, battle, active, opponent, best_damage_score)


def ensure_randbats_sets(self, battle: Battle) -> str:
    return oranguru_belief.ensure_randbats_sets(self, battle)


def sanitize_randbats_moves(self) -> None:
    return oranguru_belief.sanitize_randbats_moves(self)


def belief_weight_for_set(
    self,
    fp_mon: FPPokemon,
    set_info: dict,
    battle: Battle,
    apply_damage: bool = True,
) -> float:
    return oranguru_belief.belief_weight_for_set(self, fp_mon, set_info, battle, apply_damage=apply_damage)


def candidate_randombattle_sets(self, opponent: Pokemon, battle: Battle) -> List[Tuple[dict, float]]:
    return oranguru_belief.candidate_randombattle_sets(self, opponent, battle)


def extract_last_opponent_move(self, battle: Battle) -> Optional[Tuple[str, str, int]]:
    return oranguru_worlds.extract_last_opponent_move(self, battle)


def sample_set_for_species(
    self,
    species: str,
    battle: Battle,
    mon: Optional[Pokemon] = None,
    rng: Optional[random.Random] = None,
) -> Optional[dict]:
    return oranguru_worlds.sample_set_for_species(self, species, battle, mon=mon, rng=rng)


def sample_unknown_opponents(
    self,
    battle: Battle,
    taken: set,
    count: int,
    rng: Optional[random.Random] = None,
) -> List[FPPokemon]:
    return oranguru_worlds.sample_unknown_opponents(self, battle, taken, count, rng=rng)


def build_fp_battle(self, battle: Battle, seed: int, fill_opponent_sets: bool = False) -> FPBattle:
    return oranguru_worlds.build_fp_battle(self, battle, seed, fill_opponent_sets=fill_opponent_sets)


def fp_move_id(self, move) -> str:
    return oranguru_worlds.fp_move_id(None, move)
