#!/usr/bin/env python3
"""
Battle-memory and state-tracking helpers for OranguruEnginePlayer.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from poke_env.battle import Battle, MoveCategory

from src.players.rule_bot import RuleBotPlayer
from src.utils.damage_calc import normalize_name

FP_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "foul-play"
if str(FP_ROOT) not in sys.path:
    sys.path.insert(0, str(FP_ROOT))

import constants


def record_last_action(self, battle: Battle, order) -> None:
    mem = self._get_battle_memory(battle)
    prev_action = mem.get("last_action")
    prev_turn = int(mem.get("last_action_turn", -99) or -99)
    prev_move_id = normalize_name(str(mem.get("last_move_id", "") or ""))
    prev_active_species = normalize_name(str(mem.get("last_active_species", "") or ""))
    prev_opponent_species = normalize_name(str(mem.get("last_opponent_species", "") or ""))
    active_species = normalize_name(getattr(getattr(battle, "active_pokemon", None), "species", "") or "")
    opponent_species = normalize_name(
        getattr(getattr(battle, "opponent_active_pokemon", None), "species", "") or ""
    )

    RuleBotPlayer._record_last_action(self, battle, order)
    mem = self._get_battle_memory(battle)
    order_obj = getattr(order, "order", None)
    mem["last_active_species"] = active_species
    if opponent_species:
        mem["last_opponent_species"] = opponent_species
    if hasattr(order_obj, "category"):
        move_id = normalize_name(getattr(order_obj, "id", "") or "")
        same_matchup_repeat = bool(
            prev_action == "move"
            and prev_turn == int(getattr(battle, "turn", 0) or 0) - 1
            and prev_move_id == move_id
            and prev_active_species == active_species
            and prev_opponent_species == opponent_species
        )
        if same_matchup_repeat:
            mem["same_move_repeat_streak"] = int(mem.get("same_move_repeat_streak", 1) or 1) + 1
        else:
            mem["same_move_repeat_streak"] = 1
    else:
        mem["same_move_repeat_streak"] = 0
    passive_kind = self._passive_choice_kind(order_obj)
    if passive_kind:
        mem["pending_passive_action"] = {
            "turn": int(getattr(battle, "turn", 0) or 0),
            "kind": passive_kind,
            "self_hp": getattr(getattr(battle, "active_pokemon", None), "current_hp_fraction", None),
            "opp_hp": getattr(
                getattr(battle, "opponent_active_pokemon", None), "current_hp_fraction", None
            ),
            "opp_markers": self._opponent_progress_markers(
                battle, getattr(battle, "opponent_active_pokemon", None)
            ),
        }
    else:
        mem.pop("pending_passive_action", None)
        mem["passive_no_progress_streak"] = 0
    if not self.DAMAGE_BELIEF:
        return
    if not hasattr(order_obj, "category"):
        mem.pop("_dmg_pending", None)
        return
    move = order_obj
    if move.category == MoveCategory.STATUS:
        mem.pop("_dmg_pending", None)
        return
    move_id = normalize_name(getattr(move, "id", "") or "")
    if move_id in self.DAMAGE_BELIEF_UNSTABLE_MOVES:
        mem.pop("_dmg_pending", None)
        return

    active = battle.active_pokemon
    opponent = battle.opponent_active_pokemon
    if active is None or opponent is None:
        mem.pop("_dmg_pending", None)
        return
    if self._damage_belief_has_unmodeled_state(battle):
        mem.pop("_dmg_pending", None)
        return

    entry = self._get_move_entry(move)
    bp = int(entry.get("basePower", 0) or 0) or (move.base_power or 0)
    if bp <= 0:
        mem.pop("_dmg_pending", None)
        return

    multihit = entry.get("multihit")
    if multihit:
        mem.pop("_dmg_pending", None)
        return

    move_cat = "physical" if move.category == MoveCategory.PHYSICAL else "special"
    move_type_str = self._move_type_id(move) or ""
    if not move_type_str:
        mem.pop("_dmg_pending", None)
        return

    atk_stats = active.stats or {}
    stat_key = "atk" if move_cat == "physical" else "spa"
    raw_stat = atk_stats.get(stat_key) or 100

    boost_val = (active.boosts or {}).get(stat_key, 0) or 0
    if boost_val > 0:
        eff_stat = int(raw_stat * (2 + boost_val) / 2)
    elif boost_val < 0:
        eff_stat = int(raw_stat * 2 / (2 - boost_val))
    else:
        eff_stat = raw_stat

    atk_types = [t.name.lower() for t in active.types if t is not None] if active.types else []
    atk_ability_str = normalize_name(str(active.ability)) if active.ability else ""
    atk_item_str = normalize_name(str(active.item)) if active.item else ""
    atk_status_str = normalize_name(str(active.status)) if active.status else ""
    opp_types = [t.name.lower() for t in opponent.types if t is not None] if opponent.types else []
    opp_boosts = dict(opponent.boosts) if opponent.boosts else {}

    weather_str = ""
    for w in battle.weather:
        weather_str = w.name.lower() if hasattr(w, "name") else str(w).lower()
        break

    terrain_str = ""
    for f in battle.fields:
        fn = f.name.lower() if hasattr(f, "name") else str(f).lower()
        if "terrain" in fn:
            terrain_str = fn
            break

    mem["_dmg_pending"] = {
        "turn": battle.turn,
        "move_id": move_id,
        "move_bp": bp,
        "move_type": move_type_str,
        "move_category": move_cat,
        "attacker_stat": eff_stat,
        "attacker_level": active.level,
        "attacker_types": atk_types,
        "attacker_boosts": {stat_key: boost_val},
        "attacker_status": atk_status_str,
        "attacker_ability": atk_ability_str,
        "attacker_item": atk_item_str,
        "defender_types": opp_types,
        "defender_boosts": opp_boosts,
        "opponent_species": normalize_name(opponent.species),
        "weather": weather_str,
        "terrain": terrain_str,
        "high_confidence": True,
    }


def update_damage_observation(self, battle: Battle) -> None:
    if not self.DAMAGE_BELIEF:
        return
    mem = self._get_battle_memory(battle)
    mem.setdefault("damage_observations", {})

    pending = mem.get("_dmg_pending")
    if not pending:
        return
    pending_turn = pending.get("turn", -1)
    last_obs_turn = mem.get("_dmg_last_obs_turn", -1)
    if pending_turn <= last_obs_turn:
        return
    last_turn = pending_turn
    observations = getattr(battle, "observations", {})
    obs = observations.get(last_turn)
    if obs is None:
        return

    role = getattr(battle, "player_role", None)
    if not role:
        return

    mem["_dmg_last_obs_turn"] = pending_turn

    species = pending["opponent_species"]
    opp_prefix = "p2" if role == "p1" else "p1"
    our_prefix = role

    tracked_hp = None
    found_our_move = False
    pre_hit_hp = None
    post_hit_hp = None
    was_crit = False
    skip = False

    for event in obs.events:
        if len(event) < 2:
            continue
        kind = event[1]

        if not found_our_move:
            if kind in ("switch", "drag") and len(event) >= 5:
                who = event[2]
                if who.startswith(opp_prefix):
                    hp_str = event[4] if len(event) >= 5 else ""
                    tracked_hp = self._parse_hp_fraction(hp_str)
                    ev_species = self._species_from_event(battle, event)
                    if ev_species and ev_species != species:
                        skip = True
                        break

            if kind in ("-damage", "-heal") and len(event) >= 4:
                who = event[2]
                if who.startswith(opp_prefix):
                    tracked_hp = self._parse_hp_fraction(event[3])

            if kind == "move" and len(event) >= 4:
                who = event[2]
                if who.startswith(our_prefix):
                    if tracked_hp is None:
                        opp = battle.opponent_active_pokemon
                        if opp and opp.species and normalize_name(opp.species) == species:
                            tracked_hp = mem.get("last_opponent_hp")
                    pre_hit_hp = tracked_hp
                    found_our_move = True
                    continue

        if found_our_move:
            if kind == "-miss":
                skip = True
                break
            if kind == "-immune":
                skip = True
                break
            if kind == "-fail":
                skip = True
                break
            if kind == "-activate" and len(event) >= 4:
                act = event[3].lower() if len(event) >= 4 else ""
                if "protect" in act or "substitute" in act:
                    skip = True
                    break

            if kind == "-crit":
                was_crit = True
                continue

            if kind in ("-damage", "damage") and len(event) >= 4:
                who = event[2]
                if who.startswith(opp_prefix):
                    ev_lower = " ".join(event).lower()
                    has_from = "[from]" in ev_lower
                    if has_from and "move:" not in ev_lower:
                        continue
                    post_hit_hp = self._parse_hp_fraction(event[3])
                    break

            if kind == "move":
                break
            if kind in ("switch", "drag"):
                break

    if skip or pre_hit_hp is None or post_hit_hp is None:
        return

    observed_frac = pre_hit_hp - post_hit_hp
    if observed_frac <= 0.01:
        return

    obs_record = {
        "move_id": pending.get("move_id", ""),
        "move_bp": pending["move_bp"],
        "move_type": pending["move_type"],
        "move_category": pending["move_category"],
        "attacker_stat": pending["attacker_stat"],
        "attacker_level": pending["attacker_level"],
        "attacker_types": pending["attacker_types"],
        "attacker_boosts": pending.get("attacker_boosts", {}),
        "attacker_status": pending.get("attacker_status", ""),
        "attacker_ability": pending.get("attacker_ability", ""),
        "attacker_item": pending.get("attacker_item", ""),
        "defender_boosts": pending.get("defender_boosts", {}),
        "defender_hp_frac": pre_hit_hp,
        "observed_frac": observed_frac,
        "weather": pending.get("weather", ""),
        "terrain": pending.get("terrain", ""),
        "is_crit": was_crit,
        "high_confidence": bool(pending.get("high_confidence", False)),
    }
    mem["damage_observations"].setdefault(species, []).append(obs_record)
    if len(mem["damage_observations"][species]) > 8:
        mem["damage_observations"][species] = mem["damage_observations"][species][-8:]


def parse_hp_fraction(hp_str: str) -> Optional[float]:
    if not hp_str:
        return None
    try:
        hp_str = hp_str.strip().split()[0]
        if "/" in hp_str:
            parts = hp_str.split("/")
            cur = float(parts[0])
            mx = float(parts[1])
            if mx <= 0:
                return 0.0
            return cur / mx
        val = float(hp_str)
        if val > 1.0:
            return val / 100.0
        return val
    except Exception:
        return None


def weather_turns_remaining(self, battle: Battle) -> int:
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


def terrain_turns_remaining(self, battle: Battle) -> int:
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
