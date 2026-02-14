#!/usr/bin/env python3
"""
Enhanced RuleBot - Improved SimpleHeuristicsPlayer with smarter decisions.

Key improvements over SimpleHeuristicsPlayer:
1. Better matchup estimation with actual stat consideration
2. Smarter switch decisions based on predicted opponent moves
3. Priority move awareness
4. Better setup move timing

This should beat SimpleHeuristics by exploiting its predictable behavior.
"""

import json
import os
from pathlib import Path
from functools import lru_cache
from typing import List, Tuple, Optional, Dict
from poke_env.player import Player
from poke_env.player.battle_order import _EmptyBattleOrder
from poke_env.battle import AbstractBattle, Battle, Pokemon, Move, MoveCategory, SideCondition
from poke_env.battle.field import Field
from poke_env.battle.weather import Weather
from poke_env.battle.effect import Effect

from src.utils.damage_calc import ABILITY_DAMAGE_MODS, normalize_name, get_type_effectiveness
from src.utils.features import load_abilities, load_moves
from src.utils.moveset_priors import load_moveset_priors

class RuleBotPlayer(Player):
    """Enhanced heuristics player designed to beat SimpleHeuristicsPlayer."""

    ENTRY_HAZARDS = {
        "spikes": SideCondition.SPIKES,
        "stealthrock": SideCondition.STEALTH_ROCK,
        "stickyweb": SideCondition.STICKY_WEB,
        "toxicspikes": SideCondition.TOXIC_SPIKES,
    }

    ANTI_HAZARDS_MOVES = {"rapidspin", "defog", "tidyup", "mortalspin", "courtchange"}

    # Tuned coefficients
    SPEED_TIER_COEFICIENT = 0.15  # Value speed more
    HP_FRACTION_COEFICIENT = 0.35
    SWITCH_OUT_MATCHUP_THRESHOLD = -1.5  # Balanced switch threshold

    # Priority moves for special handling
    PRIORITY_MOVES = {
        'extremespeed', 'fakeout', 'firstimpression', 'machpunch',
        'bulletpunch', 'aquajet', 'iceshard', 'shadowsneak',
        'suckerpunch', 'accelerock', 'quickattack', 'jetpunch'
    }

    # Status moves that cripple opponents - SimpleHeuristics doesn't use or account for these!
    STATUS_MOVES = {
        'toxic': 'poison',       # Great vs bulky mons
        'willowisp': 'burn',     # Halves physical damage
        'thunderwave': 'para',   # Halves speed + chance to skip
        'glare': 'para',
        'stunspore': 'para',
        'spore': 'sleep',        # Puts to sleep
        'sleeppowder': 'sleep',
        'hypnosis': 'sleep',
        'darkvoid': 'sleep',
        'yawn': 'yawn',          # Forces switch or sleep
    }

    # Pivot moves that let us switch after attacking
    PIVOT_MOVES = {'uturn', 'voltswitch', 'flipturn', 'partingshot', 'teleport', 'batonpass'}

    PROTECT_MOVES = {
        "protect", "detect", "kingsshield", "spikyshield", "banefulbunker",
        "silktrap", "obstruct", "maxguard"
    }

    ANTI_SETUP_MOVES = {
        "haze", "clearsmog", "topsyturvy",
        "roar", "whirlwind", "dragontail", "circlethrow",
    }

    FUTURE_SIGHT_MOVES = {"futuresight", "doomdesire"}

    # High-recoil moves (33% recoil or more)
    HIGH_RECOIL_MOVES = {
        "bravebird", "doubleedge", "flareblitz", "headcharge", "headsmash",
        "volttackle", "woodhammer", "wildcharge", "takedown", "submission",
        "wavecrash", "chloroblast"
    }

    RECOVERY_MOVES = {
        "recover", "roost", "slackoff", "softboiled", "moonlight", "morningsun",
        "shoreup", "strengthsap", "synthesis", "rest", "milkdrink", "healorder"
    }

    PRIOR_MOVE_LOOKAHEAD = 6
    OPP_MOVE_PRIOR_WEIGHT = 0.65
    OPP_KNOWN_MOVE_WEIGHT = 1.15
    OPP_SWITCH_MAX_WEIGHT = 0.6
    CHOICE_ITEMS = {"choiceband", "choicescarf", "choicespecs"}
    UNLIKELY_CHOICE_INFER = bool(int(os.getenv("ORANGURU_UNLIKELY_CHOICE_INFER", "1")))
    CHOICE_UNLIKELY_EXTENDED = bool(int(os.getenv("ORANGURU_CHOICE_UNLIKELY_EXTENDED", "0")))
    CHOICE_UNLIKELY_PIVOT = bool(int(os.getenv("ORANGURU_CHOICE_UNLIKELY_PIVOT", "0")))
    CHOICE_UNLIKELY_EXTRA = {
        "protect",
        "detect",
        "endure",
        "kingsshield",
        "spikyshield",
        "banefulbunker",
        "silktrap",
        "obstruct",
        "substitute",
        "roost",
        "recover",
        "wish",
        "rest",
        "healingwish",
        "lunarblessing",
        "morningsun",
        "moonlight",
        "synthesis",
        "strengthsap",
        "taunt",
        "encore",
        "haze",
        "aromatherapy",
        "healbell",
        "trickroom",
    }
    CHOICE_UNLIKELY_HAZARDS = {
        "stealthrock",
        "spikes",
        "toxicspikes",
        "stickyweb",
        "defog",
        "rapidspin",
        "mortalspin",
        "tidyup",
        "courtchange",
    }
    CHOICE_UNLIKELY_PIVOT_MOVES = {
        "uturn",
        "voltswitch",
        "flipturn",
        "partingshot",
        "batonpass",
        "teleport",
        "chillyreception",
    }

    DEBUG_STATUS = True
    STATUS_SKIP_COUNTS = {"skipped": 0, "available": 0}
    STATUS_AVAILABLE_TURNS = 0
    ANTI_SWITCH_CHURN = bool(int(os.getenv("ORANGURU_ANTI_SWITCH_CHURN", "1")))
    PARA_FINISH_GUARD = bool(int(os.getenv("ORANGURU_PARA_FINISH_GUARD", "1")))
    PARA_FINISH_MAX_OPP_HP = float(os.getenv("ORANGURU_PARA_FINISH_MAX_OPP_HP", "0.6"))
    PARA_FINISH_KO_THRESHOLD = float(os.getenv("ORANGURU_PARA_FINISH_KO_THRESHOLD", "220.0"))
    SETUP_BOOST_CAPS = {
        "atk": 2,
        "spa": 2,
        "def": 2,
        "spd": 2,
        "spe": 2,
    }

    @staticmethod
    def _canon_id(value) -> str:
        if value is None:
            return ""
        for attr in ("id", "value"):
            try:
                cand = getattr(value, attr)
            except Exception:
                cand = None
            if cand:
                return normalize_name(str(cand))
        return normalize_name(str(value))

    def _get_battle_memory(self, battle: Battle) -> dict:
        if not hasattr(self, "_battle_memory"):
            self._battle_memory = {}
        mem = self._battle_memory.setdefault(battle.battle_tag, {})
        mem.setdefault("last_protect_turn", -99)
        mem.setdefault("last_future_sight_turn", -99)
        mem.setdefault("opponent_hp", {})
        mem.setdefault("team_hp", {})
        mem.setdefault("last_action_turn", -1)
        mem.setdefault("last_action", None)
        mem.setdefault("last_move_id", None)
        mem.setdefault("last_move_type", None)
        mem.setdefault("last_move_accuracy", None)
        mem.setdefault("last_move_category", None)
        mem.setdefault("last_opponent_species", None)
        mem.setdefault("last_opponent_hp", None)
        mem.setdefault("active_species", None)
        mem.setdefault("active_turns", 0)
        mem.setdefault("immune_moves", {})
        mem.setdefault("immune_types", {})
        mem.setdefault("no_damage_counts", {})
        mem.setdefault("speed_hints", {})
        mem.setdefault("last_speed_turn", -1)
        mem.setdefault("opponent_item_flags", {})
        mem.setdefault("opponent_moves", {})
        mem.setdefault("last_item_turn", -1)
        mem.setdefault("opponent_abilities", {})
        mem.setdefault("last_ability_turn", -1)
        mem.setdefault("opponent_impossible_abilities", {})
        mem.setdefault("last_ability_constraint_turn", -1)
        mem.setdefault("pending_wish", {"self": (0, 0), "opp": (0, 0)})
        mem.setdefault("pending_future_sight", {"self": (0, ""), "opp": (0, "")})
        mem.setdefault("last_field_turn", -1)
        mem.setdefault("last_ability_infer_turn", -1)
        mem.setdefault("switch_flags", {"self": {"baton": False, "shed": False}, "opp": {"baton": False, "shed": False}})
        mem.setdefault("substitute_state", {"self": {}, "opp": {}})
        mem.setdefault("last_substitute_turn", -1)
        mem.setdefault("last_switch_flag_turn", -1)
        mem.setdefault("weather_state", None)
        mem.setdefault("terrain_state", None)
        mem.setdefault("opp_switch_info", {})
        mem.setdefault("opp_moves_since_switch", {})
        mem.setdefault("last_switch_turn", -1)
        mem.setdefault("last_opp_move_turn", -1)
        mem.setdefault("self_switch_streak", 0)
        mem.setdefault("switch_churn_breaks", 0)
        return mem

    @staticmethod
    @lru_cache(maxsize=1)
    def _load_randombattle_sets() -> Dict[str, Dict[str, int]]:
        root = Path(__file__).resolve().parents[2]
        path = root / "third_party" / "foul-play" / "data" / "pkmn_sets_cache" / "gen9randombattle.json"
        if not path.exists():
            return {}
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            if isinstance(data, dict):
                return data
        except Exception:
            return {}
        return {}

    @staticmethod
    @lru_cache(maxsize=1)
    def _load_pokedex() -> Dict[str, Dict]:
        root = Path(__file__).resolve().parents[2]
        path = root / "data" / "pokedex.json"
        if not path.exists():
            return {}
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            if isinstance(data, dict):
                return data
        except Exception:
            return {}
        return {}

    def _parse_randombattle_set_key(self, key: str) -> dict:
        parts = [normalize_name(p) for p in key.split(",")]
        if len(parts) < 6:
            return {}
        try:
            level = int(parts[0])
        except Exception:
            level = 80
        item = parts[1] if len(parts) > 1 else ""
        ability = parts[2] if len(parts) > 2 else ""
        tera_types = {
            "normal", "fire", "water", "electric", "grass", "ice", "fighting",
            "poison", "ground", "flying", "psychic", "bug", "rock", "ghost",
            "dragon", "dark", "steel", "fairy", "stellar",
        }
        tera = ""
        if len(parts) >= 8:
            moves = parts[3:7]
            tera = parts[7]
        elif len(parts) == 7:
            # Some randbats entries have 3 moves + tera type
            if parts[6] in tera_types:
                moves = parts[3:6]
                tera = parts[6]
            else:
                moves = parts[3:7]
        else:
            moves = parts[3:7]
        return {"level": level, "item": item, "ability": ability, "moves": moves, "tera": tera}

    def _candidate_randombattle_sets(self, opponent: Pokemon, battle: Battle) -> List[Tuple[dict, int]]:
        if opponent is None:
            return []
        species = normalize_name(getattr(opponent, "species", ""))
        if not species:
            return []
        data = self._load_randombattle_sets()
        if not data:
            return []
        raw_sets = data.get(species, {})
        if not isinstance(raw_sets, dict) or not raw_sets:
            return []

        known_moves = {normalize_name(m.id) for m in self._get_known_moves(opponent)}
        mem = self._get_battle_memory(battle)
        observed_moves = mem.get("opponent_moves", {}).get(species, set())
        known_moves |= {normalize_name(m) for m in observed_moves}

        ability_id = self._get_ability_id(opponent)
        item_id = self._canon_id(getattr(opponent, "item", None)) if getattr(opponent, "item", None) else ""
        level = getattr(opponent, "level", None)
        tera_id = ""
        if getattr(opponent, "terastallized", False):
            tera_type = getattr(opponent, "tera_type", None)
            if tera_type is not None:
                tera_id = normalize_name(tera_type.name if hasattr(tera_type, "name") else str(tera_type))

        flags = mem.get("opponent_item_flags", {}).get(species, {})
        known_item = normalize_name(flags.get("known_item") or flags.get("removed_item") or "")
        if not item_id and known_item:
            item_id = known_item
        known_ability = mem.get("opponent_abilities", {}).get(species)
        if not ability_id and known_ability:
            ability_id = normalize_name(known_ability)
        no_choice = flags.get("no_choice", False)
        no_av = flags.get("no_assaultvest", False)
        no_boots = flags.get("no_boots", False)
        must_choice = flags.get("choice_item", False)
        impossible_abilities = mem.get("opponent_impossible_abilities", {}).get(species, set())

        candidates: List[Tuple[dict, int]] = []
        for key, count in raw_sets.items():
            if count <= 0:
                continue
            parsed = self._parse_randombattle_set_key(key)
            if not parsed:
                continue
            if level and parsed["level"] and int(parsed["level"]) != int(level):
                continue
            if tera_id and parsed["tera"] and tera_id != parsed["tera"]:
                continue
            if ability_id and parsed["ability"] and ability_id != parsed["ability"]:
                continue
            if item_id and parsed["item"] and item_id != parsed["item"]:
                continue
            if no_choice and parsed["item"] in self.CHOICE_ITEMS:
                continue
            if no_av and parsed["item"] == "assaultvest":
                continue
            if no_boots and parsed["item"] == "heavydutyboots":
                continue
            if must_choice and parsed["item"] not in self.CHOICE_ITEMS:
                continue
            if impossible_abilities and parsed["ability"] in impossible_abilities:
                continue
            if known_moves and not known_moves.issubset(set(parsed["moves"])):
                continue
            candidates.append((parsed, int(count)))

        if not candidates:
            for key, count in raw_sets.items():
                parsed = self._parse_randombattle_set_key(key)
                if parsed:
                    candidates.append((parsed, int(count)))

        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:20]

    def _update_battle_memory(self, battle: Battle) -> None:
        mem = self._get_battle_memory(battle)
        for mon in battle.opponent_team.values():
            if mon is None:
                continue
            mem["opponent_hp"][mon.species] = mon.current_hp_fraction
        for mon in battle.team.values():
            if mon is None:
                continue
            mem["team_hp"][mon.species] = mon.current_hp_fraction

    def _cleanup_battle_memory(self, battle: Battle) -> None:
        if not hasattr(self, "_battle_memory"):
            return
        if getattr(battle, "finished", False):
            self._battle_memory.pop(battle.battle_tag, None)
            if getattr(self, "_current_battle", None) is battle:
                self._current_battle = None

    def _update_active_turns(self, battle: Battle) -> None:
        mem = self._get_battle_memory(battle)
        active = battle.active_pokemon
        if not active:
            return
        species = active.species
        if mem.get("active_species") != species:
            mem["active_species"] = species
            mem["active_turns"] = 0
        else:
            mem["active_turns"] = mem.get("active_turns", 0) + 1

    def _update_immunity_memory(self, battle: Battle) -> None:
        mem = self._get_battle_memory(battle)
        last_turn = mem.get("last_action_turn", -1)
        if last_turn < 0 or battle.turn <= last_turn:
            return
        if mem.get("last_action") != "move":
            return
        last_move_id = mem.get("last_move_id")
        last_move_type = mem.get("last_move_type")
        last_move_accuracy = mem.get("last_move_accuracy")
        last_move_category = mem.get("last_move_category")
        last_opponent_species = mem.get("last_opponent_species")
        last_opponent_hp = mem.get("last_opponent_hp")
        opponent = battle.opponent_active_pokemon
        if not opponent or opponent.species != last_opponent_species:
            return
        if last_move_category == "status":
            return
        if last_opponent_hp is None or opponent.current_hp_fraction is None:
            return
        if abs(opponent.current_hp_fraction - last_opponent_hp) > 0.01:
            return

        if isinstance(last_move_accuracy, (int, float)):
            acc = float(last_move_accuracy)
            acc = acc / 100.0 if acc > 1 else acc
        else:
            acc = 1.0
        threshold = 1 if acc >= 0.95 else 2

        counts = mem.setdefault("no_damage_counts", {}).setdefault(last_opponent_species, {})
        if last_move_id:
            counts[last_move_id] = counts.get(last_move_id, 0) + 1
            if counts[last_move_id] >= threshold:
                mem.setdefault("immune_moves", {}).setdefault(last_opponent_species, set()).add(last_move_id)
        if last_move_type:
            type_key = f"type:{last_move_type}"
            counts[type_key] = counts.get(type_key, 0) + 1
            if counts[type_key] >= threshold:
                mem.setdefault("immune_types", {}).setdefault(last_opponent_species, set()).add(last_move_type)

    def _record_last_action(self, battle: Battle, order) -> None:
        mem = self._get_battle_memory(battle)
        mem["last_action_turn"] = battle.turn
        order_obj = getattr(order, "order", None)
        opponent = battle.opponent_active_pokemon
        if hasattr(order_obj, "category"):
            move = order_obj
            mem["last_action"] = "move"
            mem["last_move_id"] = move.id
            mem["last_move_type"] = self._move_type_id(move)
            mem["last_move_accuracy"] = move.accuracy
            mem["last_move_category"] = "status" if move.category == MoveCategory.STATUS else "damage"
            mem["self_switch_streak"] = 0
        else:
            mem["last_action"] = "switch"
            mem["last_move_id"] = None
            mem["last_move_type"] = None
            mem["last_move_accuracy"] = None
            mem["last_move_category"] = None
            mem["self_switch_streak"] = int(mem.get("self_switch_streak", 0) or 0) + 1
        if opponent:
            mem["last_opponent_species"] = opponent.species
            mem["last_opponent_hp"] = opponent.current_hp_fraction

    def _commit_order(self, battle: Battle, order):
        self._record_last_action(battle, order)
        return order

    @staticmethod
    def _empty_order_if_no_choices(battle: Battle):
        try:
            orders = getattr(battle, "valid_orders", None)
            if orders is not None and len(orders) == 0:
                return _EmptyBattleOrder()
        except Exception:
            pass
        return None

    def _estimate_matchup(self, mon: Pokemon, opponent: Pokemon) -> float:
        """Estimate matchup score - positive means we have advantage."""
        if mon is None or opponent is None:
            return 0.0

        # Type matchup: How well can we hit them vs how well they hit us
        my_types = self._effective_types(mon)
        opp_types = self._effective_types(opponent)
        my_offensive = max([opponent.damage_multiplier(t) for t in my_types], default=1.0)
        their_offensive = max([mon.damage_multiplier(t) for t in opp_types], default=1.0)
        score = my_offensive - their_offensive

        # Speed comparison with actual stats if available
        my_speed = self._get_effective_speed(mon)
        opp_speed = self._get_effective_speed(opponent)

        if self._is_trick_room_active():
            if my_speed < opp_speed:
                score += self.SPEED_TIER_COEFICIENT
            elif opp_speed < my_speed:
                score -= self.SPEED_TIER_COEFICIENT
        else:
            if my_speed > opp_speed:
                score += self.SPEED_TIER_COEFICIENT
            elif opp_speed > my_speed:
                score -= self.SPEED_TIER_COEFICIENT

        # HP advantage
        score += mon.current_hp_fraction * self.HP_FRACTION_COEFICIENT
        score -= opponent.current_hp_fraction * self.HP_FRACTION_COEFICIENT

        return score

    def _effective_types(self, mon: Pokemon) -> List:
        if mon is None:
            return []
        terastallized = getattr(mon, "terastallized", None)
        if terastallized:
            return [terastallized]
        if getattr(mon, "is_terastallized", False):
            tera_type = getattr(mon, "tera_type", None)
            if tera_type is not None:
                return [tera_type]
        return [t for t in (mon.types or []) if t is not None]

    def _predict_opponent_switch(self, battle: Battle) -> Optional[Pokemon]:
        """Predict which Pokemon opponent is likely to switch to using SimpleHeuristics logic."""
        opponent = battle.opponent_active_pokemon
        active = battle.active_pokemon

        if not active or not opponent:
            return None

        # SimpleHeuristics switches when matchup < -2
        # Check if current opponent has bad matchup against us
        current_matchup = self._estimate_matchup(opponent, active)  # Their perspective

        if current_matchup >= -2:
            return None  # They probably won't switch

        # Find their best switch based on what we can see
        best_switch = None
        best_matchup = -999

        for mon_id, mon in battle.opponent_team.items():
            if mon.fainted or mon == opponent:
                continue
            matchup = self._estimate_matchup(mon, active)
            if matchup > best_matchup:
                best_matchup = matchup
                best_switch = mon

        return best_switch

    def _priority_from_move_id(self, move_id: str) -> int:
        if not move_id:
            return 0
        entry = self._get_move_entry_by_id(move_id)
        try:
            return int(entry.get("priority", 0) or 0)
        except Exception:
            return 0

    def _extract_obs_active(self, obs_poke):
        if isinstance(obs_poke, list):
            for mon in obs_poke:
                if mon is not None:
                    return mon
            return None
        return obs_poke

    def _update_speed_order_memory(self, battle: Battle) -> None:
        mem = self._get_battle_memory(battle)
        last_turn = battle.turn - 1
        if last_turn <= mem.get("last_speed_turn", -1):
            return
        mem["last_speed_turn"] = last_turn
        if last_turn < 1:
            return
        if "trick room" in str(battle.fields).lower():
            return
        if (
            battle.side_conditions.get(SideCondition.TAILWIND)
            or battle.opponent_side_conditions.get(SideCondition.TAILWIND)
        ):
            return
        observations = getattr(battle, "observations", {})
        obs = observations.get(last_turn) if isinstance(observations, dict) else None
        if obs is None:
            return
        role = getattr(battle, "player_role", None)
        if not role:
            return

        my_obs = self._extract_obs_active(getattr(obs, "active_pokemon", None))
        opp_obs = self._extract_obs_active(getattr(obs, "opponent_active_pokemon", None))
        if my_obs is None or opp_obs is None:
            return
        my_boosts = getattr(my_obs, "boosts", {}) or {}
        opp_boosts = getattr(opp_obs, "boosts", {}) or {}
        if my_boosts.get("spe", 0) != 0 or opp_boosts.get("spe", 0) != 0:
            return
        if normalize_name(str(getattr(my_obs, "status", ""))) in {"par", "paralysis"}:
            return
        if normalize_name(str(getattr(opp_obs, "status", ""))) in {"par", "paralysis"}:
            return
        if self._can_have_speed_modified(battle, my_obs):
            return
        if self._can_have_speed_modified(battle, opp_obs):
            return

        saw_switch = False
        self_move_id = None
        opp_move_id = None
        self_idx = None
        opp_idx = None
        events = getattr(obs, "events", None)
        if not isinstance(events, list):
            return
        for idx, event in enumerate(events):
            if len(event) < 3:
                continue
            kind = event[1]
            if kind in {"switch", "drag", "replace"}:
                saw_switch = True
            if kind != "move" or len(event) < 4:
                continue
            who = event[2]
            move_id = normalize_name(event[3])
            if not move_id:
                continue
            if who.startswith(role):
                if self_move_id is None:
                    self_move_id = move_id
                    self_idx = idx
            else:
                if opp_move_id is None:
                    opp_move_id = move_id
                    opp_idx = idx

        if saw_switch or self_move_id is None or opp_move_id is None:
            return
        if self_idx is None or opp_idx is None:
            return
        if self._priority_from_move_id(self_move_id) != self._priority_from_move_id(opp_move_id):
            return
        faster = self_idx < opp_idx
        key_pair = f"{normalize_name(my_obs.species)}|{normalize_name(opp_obs.species)}"
        key_opp = f"opp:{normalize_name(opp_obs.species)}"
        for key in (key_pair, key_opp):
            entry = mem.setdefault("speed_hints", {}).setdefault(
                key, {"faster": 0, "slower": 0}
            )
            if faster:
                entry["faster"] += 1
            else:
                entry["slower"] += 1

        my_speed = None
        if my_obs.stats and my_obs.stats.get("spe"):
            try:
                my_speed = int(my_obs.stats["spe"])
            except Exception:
                my_speed = None
        if my_speed is None:
            return
        opp_species = normalize_name(opp_obs.species)
        bounds = mem.setdefault("speed_bounds", {}).setdefault(
            opp_species, {"min": 0, "max": float("inf")}
        )
        if faster:
            bounds["max"] = min(bounds["max"], max(0, my_speed - 1))
        else:
            bounds["min"] = max(bounds["min"], my_speed + 1)

    def _species_from_event(self, battle: Battle, event: List[str]) -> Optional[str]:
        if not event or len(event) < 3:
            return None
        try:
            mon = battle.get_pokemon(event[2])
            if mon and mon.species:
                return normalize_name(mon.species)
        except Exception:
            pass
        if ": " in event[2]:
            return normalize_name(event[2].split(": ", 1)[1])
        return None

    def _update_switch_in_memory(self, battle: Battle) -> None:
        mem = self._get_battle_memory(battle)
        last_turn = battle.turn - 1
        if last_turn <= mem.get("last_switch_turn", -1):
            return
        mem["last_switch_turn"] = last_turn
        if last_turn < 1:
            return
        observations = getattr(battle, "observations", {})
        obs = observations.get(last_turn)
        if obs is None:
            return
        role = getattr(battle, "player_role", None)
        if not role:
            return

        for event in obs.events:
            if len(event) < 3:
                continue
            kind = event[1]
            if kind not in {"switch", "drag", "replace"}:
                continue
            who = event[2]
            if who.startswith(role):
                continue
            species = self._species_from_event(battle, event)
            if not species and len(event) >= 4:
                species = normalize_name(str(event[3]).split(",", 1)[0])
            if not species:
                continue

            mon = None
            try:
                opp_active = getattr(battle, "opponent_active_pokemon", None)
                if opp_active and normalize_name(opp_active.species) == species:
                    mon = opp_active
                else:
                    for candidate in battle.opponent_team.values():
                        if candidate and normalize_name(candidate.species) == species:
                            mon = candidate
                            break
            except Exception:
                mon = None

            hp_at_switch = None
            status_at_switch = None
            if mon is not None:
                try:
                    if getattr(mon, "current_hp", None) is not None:
                        hp_at_switch = int(mon.current_hp)
                    elif mon.current_hp_fraction is not None and getattr(mon, "max_hp", None):
                        hp_at_switch = int(mon.max_hp * mon.current_hp_fraction)
                except Exception:
                    hp_at_switch = None
                try:
                    status_at_switch = getattr(mon, "status", None)
                except Exception:
                    status_at_switch = None

            mem.setdefault("opp_switch_info", {})[species] = {
                "hp": hp_at_switch,
                "status": status_at_switch,
                "turn": last_turn,
            }
            mem.setdefault("opp_moves_since_switch", {})[species] = set()

            flags = mem.setdefault("opponent_item_flags", {}).setdefault(
                species,
                {
                    "no_choice": False,
                    "no_assaultvest": False,
                    "no_boots": False,
                    "choice_item": False,
                    "has_boots": False,
                    "known_item": None,
                    "removed_item": None,
                    "last_move_id": None,
                    "last_move_turn": -1,
                },
            )
            flags["last_move_id"] = None
            flags["last_move_turn"] = last_turn

    def _update_opponent_move_history(self, battle: Battle) -> None:
        mem = self._get_battle_memory(battle)
        last_turn = battle.turn - 1
        if last_turn <= mem.get("last_opp_move_turn", -1):
            return
        mem["last_opp_move_turn"] = last_turn
        if last_turn < 1:
            return
        observations = getattr(battle, "observations", {})
        obs = observations.get(last_turn)
        if obs is None:
            return
        role = getattr(battle, "player_role", None)
        if not role:
            return

        for event in obs.events:
            if len(event) < 4:
                continue
            kind = event[1]
            if kind != "move":
                continue
            who = event[2]
            if who.startswith(role):
                continue
            species = self._species_from_event(battle, event)
            if not species:
                continue
            move_id = normalize_name(event[3])
            if not move_id:
                continue
            moves = mem.setdefault("opp_moves_since_switch", {}).setdefault(species, set())
            moves.add(move_id)

    def _update_opponent_item_memory(self, battle: Battle) -> None:
        mem = self._get_battle_memory(battle)
        last_turn = battle.turn - 1
        if last_turn <= mem.get("last_item_turn", -1):
            return
        mem["last_item_turn"] = last_turn
        if last_turn < 1:
            return
        observations = getattr(battle, "observations", {})
        obs = observations.get(last_turn)
        if obs is None:
            return
        role = getattr(battle, "player_role", None)
        if not role:
            return

        for event in obs.events:
            if len(event) < 3:
                continue
            kind = event[1]
            who = event[2]
            if who.startswith(role):
                continue
            species = self._species_from_event(battle, event)
            if not species:
                continue
            flags = mem.setdefault("opponent_item_flags", {}).setdefault(
                species,
                {
                    "no_choice": False,
                    "no_assaultvest": False,
                    "no_boots": False,
                    "choice_item": False,
                    "has_boots": False,
                    "known_item": None,
                    "removed_item": None,
                    "last_move_id": None,
                    "last_move_turn": -1,
                },
            )
            moveset = mem.setdefault("opponent_moves", {}).setdefault(species, set())

            if kind == "move" and len(event) >= 4:
                move_id = normalize_name(event[3])
                if move_id:
                    if flags.get("last_move_id") and flags["last_move_id"] != move_id:
                        flags["no_choice"] = True
                    flags["last_move_id"] = move_id
                    flags["last_move_turn"] = last_turn
                    moveset.add(move_id)
                    entry = self._get_move_entry_by_id(move_id)
                    if str(entry.get("category", "")).lower() == "status":
                        flags["no_assaultvest"] = True
                    if self.UNLIKELY_CHOICE_INFER and self._unlikely_choice_move(move_id):
                        flags["no_choice"] = True
                continue

            if kind in {"-damage", "damage"}:
                lower = " ".join(event).lower()
                if "[from] stealth rock" in lower or "[from] spikes" in lower:
                    flags["no_boots"] = True
                if "[from] item: life orb" in lower:
                    flags["no_choice"] = True
                for part in event:
                    if "item:" in part.lower():
                        item_id = normalize_name(part.split("item:", 1)[1])
                        if item_id:
                            flags["known_item"] = item_id
                        break
                continue

            if "item" in kind:
                lower = " ".join(event).lower()
                if len(event) >= 4:
                    item_id = normalize_name(event[3])
                    if item_id:
                        flags["known_item"] = item_id
                        if item_id in self.CHOICE_ITEMS:
                            flags["choice_item"] = True
                        if item_id == "heavydutyboots":
                            flags["has_boots"] = True
                        if kind == "-enditem":
                            flags["removed_item"] = item_id
                if "heavy-duty boots" in lower or "heavydutyboots" in lower:
                    flags["has_boots"] = True
                if "choiceband" in lower or "choicespecs" in lower or "choicescarf" in lower:
                    flags["choice_item"] = True

    def _update_opponent_ability_memory(self, battle: Battle) -> None:
        mem = self._get_battle_memory(battle)
        last_turn = battle.turn - 1
        if last_turn <= mem.get("last_ability_turn", -1):
            return
        mem["last_ability_turn"] = last_turn
        if last_turn < 1:
            return
        observations = getattr(battle, "observations", {})
        obs = observations.get(last_turn)
        if obs is None:
            return
        role = getattr(battle, "player_role", None)
        if not role:
            return

        for event in obs.events:
            if len(event) < 3:
                continue
            who = event[2]
            if who.startswith(role):
                continue
            species = self._species_from_event(battle, event)
            if not species:
                continue
            ability_id = None
            kind = event[1]
            if kind in {"-ability", "ability"} and len(event) >= 4:
                ability_id = normalize_name(event[3])
            if not ability_id:
                for part in event:
                    if isinstance(part, str) and "ability:" in part.lower():
                        ability_id = normalize_name(part.split("ability:", 1)[1])
                        break
            if ability_id:
                mem.setdefault("opponent_abilities", {})[species] = ability_id

    def _update_opponent_ability_constraints(self, battle: Battle) -> None:
        mem = self._get_battle_memory(battle)
        last_turn = mem.get("last_action_turn", -1)
        if last_turn <= mem.get("last_ability_constraint_turn", -1):
            return
        mem["last_ability_constraint_turn"] = last_turn
        if last_turn < 1:
            return
        if mem.get("last_action") != "move":
            return
        if mem.get("last_move_category") != "damage":
            return
        last_move_id = mem.get("last_move_id")
        last_move_type = mem.get("last_move_type")
        last_opponent_species = mem.get("last_opponent_species")
        last_opponent_hp = mem.get("last_opponent_hp")
        opponent = battle.opponent_active_pokemon
        if not opponent or not last_opponent_species:
            return
        if opponent.species != last_opponent_species:
            return
        if last_opponent_hp is None or opponent.current_hp_fraction is None:
            return
        if abs(opponent.current_hp_fraction - last_opponent_hp) <= 0.01:
            return
        if not last_move_type:
            return
        entry = self._get_move_entry_by_id(last_move_id or "")
        if entry.get("ignoreAbility"):
            return
        if last_move_id in {"thousandarrows"}:
            return

        immune_map = {
            "water": {"waterabsorb", "stormdrain", "dryskin"},
            "electric": {"voltabsorb", "lightningrod", "motordrive"},
            "grass": {"sapsipper"},
            "fire": {"flashfire", "wellbakedbody"},
            "ground": {"eartheater", "levitate"},
        }
        blocked = immune_map.get(last_move_type, set())
        if not blocked:
            return
        abilities = mem.setdefault("opponent_impossible_abilities", {}).setdefault(
            normalize_name(last_opponent_species), set()
        )
        abilities.update(blocked)

    def _infer_ability_from_immunity(self, battle: Battle) -> None:
        mem = self._get_battle_memory(battle)
        last_turn = mem.get("last_action_turn", -1)
        if last_turn <= mem.get("last_ability_infer_turn", -1):
            return
        mem["last_ability_infer_turn"] = last_turn
        if last_turn < 1:
            return
        if mem.get("last_action") != "move":
            return
        if mem.get("last_move_category") != "damage":
            return
        last_move_id = mem.get("last_move_id")
        last_move_type = mem.get("last_move_type")
        last_opponent_species = mem.get("last_opponent_species")
        last_opponent_hp = mem.get("last_opponent_hp")
        opponent = battle.opponent_active_pokemon
        if not opponent or not last_opponent_species:
            return
        if opponent.species != last_opponent_species:
            return
        if last_opponent_hp is None or opponent.current_hp_fraction is None:
            return
        if abs(opponent.current_hp_fraction - last_opponent_hp) > 0.01:
            return
        if not last_move_type:
            return

        effects = getattr(opponent, "effects", None) or {}
        for shield in (
            Effect.PROTECT,
            Effect.BANEFUL_BUNKER,
            Effect.SPIKY_SHIELD,
            Effect.SILK_TRAP,
            Effect.ENDURE,
            Effect.SUBSTITUTE,
        ):
            if shield in effects:
                return

        entry = self._get_move_entry_by_id(last_move_id or "")
        if entry.get("ignoreAbility"):
            return
        if last_move_id in {"thousandarrows"}:
            return

        # If type immunity explains it, do not infer ability.
        try:
            move_obj = None
            for mv in self._get_known_moves(battle.active_pokemon):
                if mv.id == last_move_id:
                    move_obj = mv
                    break
            if move_obj and opponent.damage_multiplier(move_obj) <= 0:
                return
        except Exception:
            pass

        immune_map = {
            "water": {"waterabsorb", "stormdrain", "dryskin"},
            "electric": {"voltabsorb", "lightningrod", "motordrive"},
            "grass": {"sapsipper"},
            "fire": {"flashfire", "wellbakedbody"},
            "ground": {"eartheater", "levitate"},
        }
        candidates = immune_map.get(last_move_type)
        if not candidates:
            return

        # Only infer if exactly one of the remaining randombattle sets matches a blocking ability.
        possible_sets = self._candidate_randombattle_sets(opponent, battle)
        if not possible_sets:
            return
        possible_abilities = {parsed.get("ability") for parsed, _ in possible_sets if parsed.get("ability")}
        plausible = possible_abilities & set(candidates)
        if len(plausible) == 1:
            ability_id = next(iter(plausible))
            mem.setdefault("opponent_abilities", {})[normalize_name(last_opponent_species)] = ability_id

    def _update_field_memory(self, battle: Battle) -> None:
        mem = self._get_battle_memory(battle)
        last_turn = battle.turn - 1
        if last_turn <= mem.get("last_field_turn", -1):
            return
        mem["last_field_turn"] = last_turn
        if last_turn < 1:
            return
        observations = getattr(battle, "observations", {})
        obs = observations.get(last_turn)
        if obs is None:
            return
        role = getattr(battle, "player_role", None)
        if not role:
            return

        wish = mem.get("pending_wish", {"self": (0, 0), "opp": (0, 0)})
        future_sight = mem.get("pending_future_sight", {"self": (0, ""), "opp": (0, "")})

        for side in ("self", "opp"):
            turns, amount = wish.get(side, (0, 0))
            if turns > 0:
                wish[side] = (max(0, turns - 1), amount)
            turns, source = future_sight.get(side, (0, ""))
            if turns > 0:
                future_sight[side] = (max(0, turns - 1), source)

        for event in obs.events:
            if len(event) < 4:
                continue
            if event[1] != "move":
                continue
            who = event[2]
            move_id = normalize_name(event[3])
            if not move_id:
                continue
            side_key = "self" if who.startswith(role) else "opp"
            if move_id == "wish":
                if side_key == "self":
                    active = battle.active_pokemon
                else:
                    active = battle.opponent_active_pokemon
                max_hp = getattr(active, "max_hp", None) if active else None
                amount = int(max_hp / 2) if max_hp else 0
                wish[side_key] = (2, amount)
            elif move_id in {"futuresight", "doomdesire"}:
                species = self._species_from_event(battle, event) or ""
                future_sight[side_key] = (3, normalize_name(species))

        mem["pending_wish"] = wish
        mem["pending_future_sight"] = future_sight
        mem["weather_state"] = self._update_weather_state(battle, obs, mem.get("weather_state"))
        mem["terrain_state"] = self._update_terrain_state(battle, obs, mem.get("terrain_state"))

    def _update_weather_state(self, battle: Battle, obs, current: Optional[dict]) -> Optional[dict]:
        role = getattr(battle, "player_role", None)
        if not role:
            return current
        state = current if isinstance(current, dict) else None
        for event in obs.events:
            if len(event) < 3:
                continue
            if event[1] != "-weather":
                continue
            weather_id = normalize_name(event[2])
            if weather_id == "none":
                return None
            source = self._parse_event_source(event, role, battle)
            if source.get("side") == "opp" and source.get("ability") and source.get("species"):
                mem = self._get_battle_memory(battle)
                mem.setdefault("opponent_abilities", {})[source["species"]] = source["ability"]
            state = {
                "type": weather_id,
                "start": battle.turn - 1,
                "source": source,
                "duration": self._estimate_weather_duration(source, battle),
            }
        return state

    def _update_terrain_state(self, battle: Battle, obs, current: Optional[dict]) -> Optional[dict]:
        role = getattr(battle, "player_role", None)
        if not role:
            return current
        state = current if isinstance(current, dict) else None
        for event in obs.events:
            if len(event) < 3:
                continue
            if event[1] == "-fieldend":
                terrain_id = normalize_name(event[2])
                if state and state.get("type") == terrain_id:
                    state = None
                continue
            if event[1] != "-fieldstart":
                continue
            terrain_id = normalize_name(event[2])
            source = self._parse_event_source(event, role, battle)
            if source.get("side") == "opp" and source.get("ability") and source.get("species"):
                mem = self._get_battle_memory(battle)
                mem.setdefault("opponent_abilities", {})[source["species"]] = source["ability"]
            state = {
                "type": terrain_id,
                "start": battle.turn - 1,
                "source": source,
                "duration": self._estimate_terrain_duration(source, battle),
            }
        return state

    def _parse_event_source(self, event: List[str], role: str, battle: Battle) -> dict:
        source = {"side": None, "species": "", "ability": "", "move": ""}
        for part in event:
            if not isinstance(part, str):
                continue
            lower = part.lower()
            if "ability:" in lower:
                source["ability"] = normalize_name(part.split("ability:", 1)[1])
            if "move:" in lower:
                source["move"] = normalize_name(part.split("move:", 1)[1])
            if "[of]" in lower:
                token = part.split("[of]", 1)[1].strip()
                if token.startswith(role):
                    source["side"] = "self"
                elif token.startswith("p1") or token.startswith("p2"):
                    source["side"] = "opp"
                try:
                    mon = battle.get_pokemon(token)
                    if mon and mon.species:
                        source["species"] = normalize_name(mon.species)
                except Exception:
                    if ": " in token:
                        source["species"] = normalize_name(token.split(": ", 1)[1])
        return source

    def _estimate_weather_duration(self, source: dict, battle: Battle) -> int:
        duration = 5
        if not source or not source.get("move"):
            return duration
        item_id = self._known_item_for_source(source, battle)
        if item_id in {"damprock", "heatrock", "smoothrock", "icyrock"}:
            return 8
        return duration

    def _estimate_terrain_duration(self, source: dict, battle: Battle) -> int:
        duration = 5
        if not source or not source.get("move"):
            return duration
        item_id = self._known_item_for_source(source, battle)
        if item_id == "terrainextender":
            return 8
        return duration

    def _known_item_for_source(self, source: dict, battle: Battle) -> str:
        side = source.get("side")
        species = source.get("species", "")
        if side == "self":
            for mon in battle.team.values():
                if mon and normalize_name(mon.species) == species:
                    return normalize_name(str(getattr(mon, "item", "") or ""))
        elif side == "opp":
            mem = self._get_battle_memory(battle)
            flags = mem.get("opponent_item_flags", {}).get(species, {})
            item_id = flags.get("known_item")
            if item_id:
                return normalize_name(item_id)
        return ""

    def _update_switch_flags(self, battle: Battle) -> None:
        mem = self._get_battle_memory(battle)
        last_turn = battle.turn - 1
        if last_turn <= mem.get("last_switch_flag_turn", -1):
            return
        mem["last_switch_flag_turn"] = last_turn
        if last_turn < 1:
            return
        observations = getattr(battle, "observations", {})
        obs = observations.get(last_turn)
        if obs is None:
            return
        role = getattr(battle, "player_role", None)
        if not role:
            return

        flags = mem.get("switch_flags", {"self": {"baton": False, "shed": False}, "opp": {"baton": False, "shed": False}})
        flags["self"]["baton"] = False
        flags["self"]["shed"] = False
        flags["opp"]["baton"] = False
        flags["opp"]["shed"] = False

        for event in obs.events:
            if len(event) < 4:
                continue
            if event[1] != "move":
                continue
            who = event[2]
            move_id = normalize_name(event[3])
            if not move_id:
                continue
            side_key = "self" if who.startswith(role) else "opp"
            if move_id == "batonpass":
                flags[side_key]["baton"] = True
            elif move_id == "shedtail":
                flags[side_key]["shed"] = True

        mem["switch_flags"] = flags

    def _update_substitute_memory(self, battle: Battle) -> None:
        mem = self._get_battle_memory(battle)
        last_turn = battle.turn - 1
        if last_turn <= mem.get("last_substitute_turn", -1):
            return
        mem["last_substitute_turn"] = last_turn
        if last_turn < 1:
            return
        observations = getattr(battle, "observations", {})
        obs = observations.get(last_turn)
        if obs is None:
            return
        role = getattr(battle, "player_role", None)
        if not role:
            return

        state = mem.get("substitute_state", {"self": {}, "opp": {}})

        for event in obs.events:
            if len(event) < 3:
                continue
            kind = event[1]
            who = event[2]
            side_key = "self" if who.startswith(role) else "opp"
            species = self._species_from_event(battle, event)
            if not species:
                continue
            entry = state[side_key].setdefault(species, {"has": False, "hit": False})
            lower = " ".join(str(p) for p in event).lower()

            if kind == "-start" and "substitute" in lower:
                entry["has"] = True
                entry["hit"] = False
            elif kind == "-end" and "substitute" in lower:
                entry["has"] = False
                entry["hit"] = False
            elif kind in {"-activate", "-damage"} and "substitute" in lower:
                entry["has"] = True
                entry["hit"] = True

        mem["substitute_state"] = state

    def _speed_hint(self, active: Pokemon, opponent: Pokemon, battle: Battle) -> int:
        if active is None or opponent is None:
            return 0
        mem = self._get_battle_memory(battle)
        pair_key = f"{normalize_name(active.species)}|{normalize_name(opponent.species)}"
        for key in (pair_key, f"opp:{normalize_name(opponent.species)}"):
            counts = mem.get("speed_hints", {}).get(key)
            if not counts:
                continue
            diff = counts.get("faster", 0) - counts.get("slower", 0)
            if diff > 0:
                return 1
            if diff < 0:
                return -1
        return 0

    def _opponent_choice_locked(self, opponent: Pokemon, battle: Battle) -> tuple[bool, Optional[str]]:
        if opponent is None or battle is None:
            return False, None
        mem = self._get_battle_memory(battle)
        species = normalize_name(getattr(opponent, "species", ""))
        flags = mem.get("opponent_item_flags", {}).get(species, {})
        last_move_id = flags.get("last_move_id")
        last_turn = flags.get("last_move_turn", -1)
        if flags.get("choice_item"):
            return True, last_move_id
        if flags.get("no_choice"):
            return False, None
        if not last_move_id or last_turn < battle.turn - 1:
            return False, None
        moveset = mem.get("opponent_moves", {}).get(species, set())
        if len(moveset) <= 1:
            return True, last_move_id
        return False, None

    def _switch_likelihood(self, opponent: Pokemon, target: Pokemon, battle: Battle) -> float:
        """Estimate how likely the opponent is to switch instead of attacking."""
        if opponent is None or target is None:
            return 0.0
        matchup = self._estimate_matchup(opponent, target)
        if matchup >= -0.5:
            return 0.0
        # Scale from mild disadvantage to hard counters.
        raw = (-matchup - 0.5) / 2.5
        return max(0.0, min(raw * self.OPP_SWITCH_MAX_WEIGHT, self.OPP_SWITCH_MAX_WEIGHT))

    def _classify_status_entry(self, entry: dict) -> str:
        if not entry:
            return "status"
        if entry.get("heal"):
            return "recovery"
        if entry.get("sideCondition") or (entry.get("self") or {}).get("sideCondition"):
            return "hazard"
        boosts = entry.get("boosts") or (entry.get("self") or {}).get("boosts")
        if boosts and any(delta > 0 for delta in boosts.values()):
            return "setup"
        return "status"

    def _status_entry_score(self, kind: str, opponent: Pokemon) -> float:
        if kind == "setup":
            return 130.0 if not opponent.boosts or sum(opponent.boosts.values()) <= 1 else 80.0
        if kind == "recovery":
            if opponent.current_hp_fraction is not None and opponent.current_hp_fraction < 0.6:
                return 120.0
            return 70.0
        if kind == "hazard":
            return 90.0
        return 80.0

    def _predict_opponent_move(self, opponent: Pokemon, target: Pokemon, battle: Battle) -> dict:
        """Predict opponent's most likely move intent."""
        if opponent is None or target is None:
            return {"kind": "unknown", "score": 0.0, "move_id": None, "priority": False, "damage_score": 0.0}

        locked, locked_move = self._opponent_choice_locked(opponent, battle)
        if locked and locked_move:
            entry = self._get_move_entry_by_id(locked_move)
            category = str(entry.get("category", "")).lower()
            if category == "status":
                kind = self._classify_status_entry(entry)
                base = self._status_entry_score(kind, opponent)
                damage_score = 0.0
            else:
                kind = "damage"
                base = self._estimate_entry_damage_score(entry, opponent, target)
                damage_score = base
            return {
                "kind": kind,
                "score": base,
                "move_id": locked_move,
                "priority": self._entry_priority_value(entry) > 0,
                "damage_score": damage_score,
            }

        candidates = []
        known_moves = self._get_known_moves(opponent)
        known_ids = {m.id for m in known_moves}

        for move in known_moves:
            entry = self._get_move_entry(move)
            if move.category == MoveCategory.STATUS:
                kind = self._classify_status_entry(entry)
                base = self._status_entry_score(kind, opponent)
            else:
                kind = "damage"
                base = self._calculate_move_score(
                    move, opponent, target, battle, apply_recoil=False, respect_immunity_memory=False
                )
            priority = entry.get("priority", 0) > 0
            candidates.append((base, kind, move.id, priority))

        prior_moves = self._get_prior_moves(opponent)
        if prior_moves:
            max_count = max(c for _, c in prior_moves) if prior_moves else 1
            for move_id, count in prior_moves:
                if move_id in known_ids:
                    continue
                entry = self._get_move_entry_by_id(move_id)
                category = str(entry.get("category", "")).lower()
                if category == "status":
                    kind = self._classify_status_entry(entry)
                    base = self._status_entry_score(kind, opponent)
                else:
                    kind = "damage"
                    base = self._estimate_entry_damage_score(entry, opponent, target)
                weight = 0.5 + 0.5 * (count / max_count)
                priority = entry.get("priority", 0) > 0
                candidates.append((base * weight, kind, move_id, priority))

        if not candidates:
            return {"kind": "unknown", "score": 0.0, "move_id": None, "priority": False, "damage_score": 0.0}

        best = max(candidates, key=lambda x: x[0])
        best_damage = max((c[0] for c in candidates if c[1] == "damage"), default=0.0)
        return {
            "kind": best[1],
            "score": best[0],
            "move_id": best[2],
            "priority": best[3],
            "damage_score": best_damage,
        }

    def _move_priority_value(self, move: Move) -> int:
        if move is None:
            return 0
        entry = self._get_move_entry(move)
        try:
            return int(entry.get("priority", 0) or 0)
        except Exception:
            return 0

    def _entry_priority_value(self, entry: dict) -> int:
        if not entry:
            return 0
        try:
            return int(entry.get("priority", 0) or 0)
        except Exception:
            return 0

    def _opponent_action_distribution(
        self,
        opponent: Pokemon,
        target: Pokemon,
        battle: Battle,
        predicted_switch: Optional[Pokemon] = None,
    ) -> List[dict]:
        """Build a lightweight distribution over likely opponent actions."""
        if opponent is None or target is None:
            return []

        locked, locked_move = self._opponent_choice_locked(opponent, battle)
        candidates: List[dict] = []
        known_moves = self._get_known_moves(opponent)
        known_ids = {m.id for m in known_moves}

        for move in known_moves:
            entry = self._get_move_entry(move)
            if move.category == MoveCategory.STATUS:
                kind = self._classify_status_entry(entry)
                base = self._status_entry_score(kind, opponent)
                damage_score = 0.0
            else:
                kind = "damage"
                base = self._calculate_move_score(
                    move, opponent, target, battle, apply_recoil=False, respect_immunity_memory=False
                )
                damage_score = base
            if base <= 0:
                continue
            weight = max(base, 1.0) * self.OPP_KNOWN_MOVE_WEIGHT
            if locked and locked_move:
                if move.id == locked_move:
                    weight *= 1.6
                else:
                    weight *= 0.45
            candidates.append(
                {
                    "type": "move",
                    "kind": kind,
                    "move_id": move.id,
                    "priority": self._move_priority_value(move),
                    "damage_score": damage_score,
                    "weight": weight,
                }
            )

        prior_moves = self._get_prior_moves(opponent)
        if prior_moves:
            max_count = max(c for _, c in prior_moves) if prior_moves else 1
            for move_id, count in prior_moves:
                if move_id in known_ids:
                    continue
                entry = self._get_move_entry_by_id(move_id)
                category = str(entry.get("category", "")).lower()
                if category == "status":
                    kind = self._classify_status_entry(entry)
                    base = self._status_entry_score(kind, opponent)
                    damage_score = 0.0
                else:
                    kind = "damage"
                    base = self._estimate_entry_damage_score(entry, opponent, target)
                    damage_score = base
                if base <= 0:
                    continue
                weight_factor = 0.5 + 0.5 * (count / max_count)
                weight = max(base, 1.0) * weight_factor * self.OPP_MOVE_PRIOR_WEIGHT
                if locked and locked_move:
                    if move_id == locked_move:
                        weight *= 1.6
                    else:
                        weight *= 0.45
                candidates.append(
                    {
                        "type": "move",
                        "kind": kind,
                        "move_id": move_id,
                        "priority": self._entry_priority_value(entry),
                        "damage_score": damage_score,
                        "weight": weight,
                    }
                )

        switch_target = predicted_switch or self._predict_opponent_switch(battle)
        switch_chance = self._switch_likelihood(opponent, target, battle)
        if switch_target is not None and switch_chance > 0:
            penalty = self._hazard_switch_penalty_for_opponent(battle, switch_target)
            switch_chance = max(0.0, switch_chance - (0.35 * penalty))
        if switch_target is not None and switch_chance > 0:
            avg_weight = (
                sum(c["weight"] for c in candidates) / len(candidates)
                if candidates
                else 120.0
            )
            candidates.append(
                {
                    "type": "switch",
                    "switch": switch_target,
                    "weight": switch_chance * avg_weight,
                }
            )

        total = sum(c["weight"] for c in candidates)
        if total <= 0:
            return []
        for cand in candidates:
            cand["weight"] /= total
        return candidates

    def _ko_likelihood(self, damage_score: float, opponent: Pokemon) -> float:
        if opponent is None or opponent.current_hp_fraction is None:
            return 0.0
        if damage_score <= 0:
            return 0.0
        hp_threshold = 220 * opponent.current_hp_fraction
        if hp_threshold <= 0:
            return 0.0
        ratio = damage_score / hp_threshold
        return max(0.0, min(1.0, (ratio - 0.7) / 0.5))

    def _ko_likelihood_vs_hp(self, damage_score: float, hp_fraction: Optional[float]) -> float:
        if hp_fraction is None or damage_score <= 0:
            return 0.0
        hp_threshold = 220 * hp_fraction
        if hp_threshold <= 0:
            return 0.0
        ratio = damage_score / hp_threshold
        return max(0.0, min(1.0, (ratio - 0.7) / 0.5))

    def _expected_reply_for_move(
        self,
        move: Move,
        damage_score: float,
        active: Pokemon,
        opponent: Pokemon,
        battle: Battle,
        action_dist: List[dict],
    ) -> float:
        if not action_dist:
            return 0.0
        my_priority = self._move_priority_value(move)
        my_speed = self._get_effective_speed(active)
        speed_hint = self._speed_hint(active, opponent, battle)
        total_reply = 0.0
        for action in action_dist:
            if action.get("type") != "move":
                continue
            if action.get("kind") != "damage":
                continue
            opp_damage = action.get("damage_score", 0.0)
            if opp_damage <= 0:
                continue
            opp_priority = action.get("priority", 0)
            if my_priority > opp_priority:
                first_chance = 0.85
            elif my_priority < opp_priority:
                first_chance = 0.15
            else:
                if speed_hint > 0:
                    first_chance = 0.7
                elif speed_hint < 0:
                    first_chance = 0.3
                else:
                    opp_speed = self._get_effective_speed(opponent)
                    if my_speed > opp_speed:
                        first_chance = 0.65
                    elif opp_speed > my_speed:
                        first_chance = 0.35
                    else:
                        first_chance = 0.5

            ko_chance = self._ko_likelihood(damage_score, opponent) if damage_score > 0 else 0.0
            adjusted = opp_damage * (1.0 - first_chance * ko_chance)
            total_reply += action["weight"] * adjusted

        return total_reply

    def _expected_net_value(
        self,
        move: Move,
        damage_score: float,
        active: Pokemon,
        opponent: Pokemon,
        battle: Battle,
        action_dist: List[dict],
        predicted_switch: Optional[Pokemon],
    ) -> float:
        if move is None or not action_dist:
            return damage_score

        my_priority = self._move_priority_value(move)
        my_speed = self._get_effective_speed(active)
        speed_hint = self._speed_hint(active, opponent, battle)
        my_ko = self._ko_likelihood_vs_hp(damage_score, opponent.current_hp_fraction)

        switch_damage = None
        if predicted_switch is not None:
            switch_damage = self._calculate_move_score(
                move, active, predicted_switch, battle, apply_recoil=True
            )
            hazard_bonus = 80.0 * self._hazard_switch_penalty_for_opponent(battle, predicted_switch)
        else:
            hazard_bonus = 0.0

        expected_value = 0.0
        for action in action_dist:
            if action.get("type") == "switch":
                if switch_damage is None:
                    continue
                net = switch_damage + hazard_bonus
                expected_value += action["weight"] * net
                continue

            if action.get("kind") != "damage":
                expected_value += action["weight"] * damage_score
                continue

            opp_damage = action.get("damage_score", 0.0)
            if opp_damage <= 0:
                expected_value += action["weight"] * damage_score
                continue
            opp_priority = action.get("priority", 0)
            if my_priority > opp_priority:
                first_chance = 0.85
            elif my_priority < opp_priority:
                first_chance = 0.15
            else:
                if speed_hint > 0:
                    first_chance = 0.7
                elif speed_hint < 0:
                    first_chance = 0.3
                else:
                    opp_speed = self._get_effective_speed(opponent)
                    if my_speed > opp_speed:
                        first_chance = 0.65
                    elif opp_speed > my_speed:
                        first_chance = 0.35
                    else:
                        first_chance = 0.5

            opp_ko = self._ko_likelihood_vs_hp(opp_damage, active.current_hp_fraction)
            expected_my = damage_score * (1.0 - (1.0 - first_chance) * opp_ko)
            expected_opp = opp_damage * (1.0 - first_chance * my_ko)
            net = expected_my - expected_opp + (30.0 * my_ko)
            expected_value += action["weight"] * net

        return expected_value

    def _estimate_best_damage_score_for_target(
        self, active: Pokemon, target: Pokemon, battle: Battle
    ) -> float:
        if not active or not target or not battle.available_moves:
            return 0.0
        best_score = 0.0
        for move in battle.available_moves:
            try:
                if move.category == MoveCategory.STATUS:
                    continue
            except Exception:
                continue
            score = self._calculate_move_score(move, active, target, battle)
            if score > best_score:
                best_score = score
        return best_score

    def _expected_two_ply_value(
        self,
        move: Move,
        damage_score: float,
        active: Pokemon,
        opponent: Pokemon,
        battle: Battle,
        action_dist: List[dict],
        predicted_switch: Optional[Pokemon],
    ) -> float:
        if move is None or not action_dist:
            return damage_score

        my_priority = self._move_priority_value(move)
        my_speed = self._get_effective_speed(active)
        speed_hint = self._speed_hint(active, opponent, battle)
        my_ko = self._ko_likelihood_vs_hp(damage_score, opponent.current_hp_fraction)

        switch_damage = None
        if predicted_switch is not None:
            switch_damage = self._calculate_move_score(
                move, active, predicted_switch, battle, apply_recoil=True
            )
            hazard_bonus = 80.0 * self._hazard_switch_penalty_for_opponent(battle, predicted_switch)
        else:
            hazard_bonus = 0.0

        expected_net = 0.0
        followup = 0.0
        active_hp = active.current_hp_fraction if active.current_hp_fraction is not None else 1.0
        opp_hp = opponent.current_hp_fraction if opponent.current_hp_fraction is not None else 1.0
        best_followup_vs_opp = self._estimate_best_damage_score_for_target(active, opponent, battle)
        best_followup_vs_switch = (
            self._estimate_best_damage_score_for_target(active, predicted_switch, battle)
            if predicted_switch is not None
            else 0.0
        )
        reply_score = self._estimate_best_reply_score(opponent, active, battle)

        for action in action_dist:
            weight = action["weight"]
            if action.get("type") == "switch":
                if switch_damage is None:
                    continue
                opp_hp_after = max(0.0, opp_hp - switch_damage / 220.0)
                expected_net += weight * (switch_damage + hazard_bonus)
                if opp_hp_after > 0:
                    ko_next = self._ko_likelihood_vs_hp(best_followup_vs_switch, opp_hp_after)
                    followup += weight * (35.0 * ko_next)
                continue

            if action.get("kind") != "damage":
                opp_hp_after = max(0.0, opp_hp - damage_score / 220.0)
                expected_net += weight * damage_score
                if opp_hp_after > 0:
                    ko_next = self._ko_likelihood_vs_hp(best_followup_vs_opp, opp_hp_after)
                    followup += weight * (30.0 * ko_next)
                continue

            opp_damage = action.get("damage_score", 0.0)
            if opp_damage <= 0:
                opp_hp_after = max(0.0, opp_hp - damage_score / 220.0)
                expected_net += weight * damage_score
                if opp_hp_after > 0:
                    ko_next = self._ko_likelihood_vs_hp(best_followup_vs_opp, opp_hp_after)
                    followup += weight * (30.0 * ko_next)
                continue
            opp_priority = action.get("priority", 0)
            if my_priority > opp_priority:
                first_chance = 0.85
            elif my_priority < opp_priority:
                first_chance = 0.15
            else:
                if speed_hint > 0:
                    first_chance = 0.7
                elif speed_hint < 0:
                    first_chance = 0.3
                else:
                    opp_speed = self._get_effective_speed(opponent)
                    if my_speed > opp_speed:
                        first_chance = 0.65
                    elif opp_speed > my_speed:
                        first_chance = 0.35
                    else:
                        first_chance = 0.5

            opp_ko = self._ko_likelihood_vs_hp(opp_damage, active.current_hp_fraction)
            expected_my = damage_score * (1.0 - (1.0 - first_chance) * opp_ko)
            expected_opp = opp_damage * (1.0 - first_chance * my_ko)
            expected_net += weight * (expected_my - expected_opp)

            opp_hp_after = max(0.0, opp_hp - expected_my / 220.0)
            self_hp_after = max(0.0, active_hp - expected_opp / 220.0)
            if opp_hp_after > 0:
                ko_next = self._ko_likelihood_vs_hp(best_followup_vs_opp, opp_hp_after)
                followup += weight * (30.0 * ko_next)
            if self_hp_after > 0:
                opp_ko_next = self._ko_likelihood_vs_hp(reply_score, self_hp_after)
                followup -= weight * (25.0 * opp_ko_next)

        return expected_net + followup

    def _expected_move_value(
        self,
        move: Move,
        base_score: float,
        active: Pokemon,
        opponent: Pokemon,
        battle: Battle,
        action_dist: List[dict],
        predicted_switch: Optional[Pokemon],
    ) -> float:
        if not action_dist:
            return base_score
        switch_prob = sum(a["weight"] for a in action_dist if a.get("type") == "switch")
        if move.category == MoveCategory.STATUS:
            entry = self._get_move_entry(move)
            kind = self._classify_status_entry(entry)
            if kind in {"setup", "recovery", "hazard"}:
                return base_score
            return base_score * (1.0 - switch_prob)

        if predicted_switch is None:
            return base_score
        switch_score = self._calculate_move_score(move, active, predicted_switch, battle)
        expected = 0.0
        for action in action_dist:
            if action.get("type") == "switch":
                expected += action["weight"] * switch_score
            else:
                expected += action["weight"] * base_score
        return expected

    def _get_known_moves(self, mon: Pokemon) -> List[Move]:
        """Return known moves for a Pokemon if available."""
        if mon is None:
            return []
        try:
            mon_moves = getattr(mon, "moves", None)
            if isinstance(mon_moves, dict):
                moves = list(mon_moves.values())
            elif mon_moves:
                moves = list(mon_moves)
            else:
                moves = []
        except Exception:
            moves = []
        return [m for m in moves if m is not None]

    def _get_prior_moves(self, mon: Pokemon) -> List[Tuple[str, int]]:
        """Return top move-id priors for a Pokemon species."""
        if mon is None:
            return []
        battle = getattr(self, "_current_battle", None)
        # Prefer randombattle set priors if available
        if battle is not None:
            candidates = self._candidate_randombattle_sets(mon, battle)
            if candidates:
                move_counts: Dict[str, int] = {}
                for parsed, count in candidates:
                    moves = parsed.get("moves", []) if isinstance(parsed, dict) else []
                    for move_id in moves:
                        if move_id:
                            move_counts[move_id] = move_counts.get(move_id, 0) + count
                ordered = sorted(move_counts.items(), key=lambda x: x[1], reverse=True)
                return ordered[: self.PRIOR_MOVE_LOOKAHEAD]

        try:
            priors = load_moveset_priors()
        except Exception:
            return []
        if not priors:
            return []
        species = normalize_name(getattr(mon, "species", ""))
        entries = priors.get(species, {})
        if not isinstance(entries, dict):
            return []
        ordered = sorted(entries.items(), key=lambda x: x[1], reverse=True)
        return ordered[: self.PRIOR_MOVE_LOOKAHEAD]

    def _get_move_entry_by_id(self, move_id: str) -> dict:
        """Return moves.json entry for a move id."""
        if not move_id:
            return {}
        return load_moves().get(move_id, {})

    def _unlikely_choice_move(self, move_id: str) -> bool:
        if not move_id:
            return False
        if move_id in {"substitute", "roost", "recover"}:
            return True
        entry = self._get_move_entry_by_id(move_id)
        if not entry:
            return False
        category = str(entry.get("category", "")).lower()
        boosts = entry.get("boosts") or {}
        if category == "status" and bool(boosts):
            return True
        if self.CHOICE_UNLIKELY_EXTENDED:
            if move_id in self.CHOICE_UNLIKELY_EXTRA or move_id in self.CHOICE_UNLIKELY_HAZARDS:
                return True
            if self.CHOICE_UNLIKELY_PIVOT and move_id in self.CHOICE_UNLIKELY_PIVOT_MOVES:
                return True
        return False

    def _expected_hits_from_entry(self, entry: dict) -> float:
        """Estimate expected hits from moves.json multihit data."""
        multihit = entry.get("multihit")
        if isinstance(multihit, list) and len(multihit) == 2:
            try:
                expected = (float(multihit[0]) + float(multihit[1])) / 2.0
            except Exception:
                return 1.0
            return max(1.0, expected)
        if isinstance(multihit, int):
            try:
                return max(1.0, float(multihit))
            except Exception:
                return 1.0
        return 1.0

    def _estimate_entry_damage_score(self, entry: dict, attacker: Pokemon, defender: Pokemon) -> float:
        """Approximate damage score using moves.json entry data."""
        if not entry or attacker is None or defender is None:
            return 0.0
        category = str(entry.get("category", "")).lower()
        if category == "status":
            return 0.0
        base_power = entry.get("basePower") or 0
        if base_power <= 0:
            return 0.0
        if category == "physical":
            stat_ratio = self._stat_estimation(attacker, "atk") / max(self._stat_estimation(defender, "def"), 1)
        else:
            stat_ratio = self._stat_estimation(attacker, "spa") / max(self._stat_estimation(defender, "spd"), 1)

        move_type = normalize_name(entry.get("type", ""))
        stab = 1.0
        if move_type and attacker.types:
            for t in attacker.types:
                if t is not None and t.name.lower() == move_type:
                    stab = 1.5
                    break

        def_types = [t.name.lower() for t in (defender.types or []) if t is not None]
        type_mult = get_type_effectiveness(move_type, def_types) if move_type else 1.0

        accuracy = entry.get("accuracy")
        if accuracy is True or accuracy is None:
            acc_mult = 1.0
        else:
            acc_mult = max(min(float(accuracy) / 100.0, 1.0), 0.0)

        expected_hits = self._expected_hits_from_entry(entry)

        return base_power * stab * stat_ratio * acc_mult * expected_hits * type_mult

    def _move_recoil_rate(self, move: Move) -> float:
        """Estimate recoil rate as fraction of damage dealt."""
        if move is None:
            return 0.0
        entry = self._get_move_entry(move)
        recoil = entry.get("recoil")
        if isinstance(recoil, list) and len(recoil) == 2:
            try:
                r0 = float(recoil[0])
                r1 = float(recoil[1])
                if r1:
                    return r0 / r1
            except Exception:
                return 0.0
        if move.id in self.HIGH_RECOIL_MOVES:
            return 0.33
        return 0.0

    def _opponent_likely_has_priority(self, opponent: Pokemon) -> bool:
        """Check known + prior moves for priority threats."""
        known_ids = self._opponent_known_move_ids(opponent)
        if known_ids & self.PRIORITY_MOVES:
            return True
        for move_id, _ in self._get_prior_moves(opponent):
            if move_id in self.PRIORITY_MOVES:
                return True
        return False

    def _estimate_best_reply_score(self, opponent: Pokemon, target: Pokemon, battle: Battle) -> float:
        """Estimate opponent's best known damaging reply against a target."""
        if opponent is None or target is None:
            return 0.0

        locked, locked_move = self._opponent_choice_locked(opponent, battle)
        if locked and locked_move:
            entry = self._get_move_entry_by_id(locked_move)
            score = self._estimate_entry_damage_score(entry, opponent, target)
            if score > 0:
                return score

        moves = self._get_known_moves(opponent)
        known_ids = {m.id for m in moves}
        best_score = 0.0

        for move in moves:
            try:
                if move.category == MoveCategory.STATUS:
                    continue
            except Exception:
                continue
            score = self._calculate_move_score(
                move, opponent, target, battle, apply_recoil=False, respect_immunity_memory=False
            )
            if score > best_score:
                best_score = score

        prior_moves = self._get_prior_moves(opponent)
        if prior_moves:
            max_count = max(c for _, c in prior_moves) if prior_moves else 1
            for move_id, count in prior_moves:
                if move_id in known_ids:
                    continue
                entry = self._get_move_entry_by_id(move_id)
                score = self._estimate_entry_damage_score(entry, opponent, target)
                if score <= 0:
                    continue
                weight = 0.5 + 0.5 * (count / max_count)
                score *= weight
                if score > best_score:
                    best_score = score

        if best_score <= 0:
            type_mult = max(
                [target.damage_multiplier(t) for t in opponent.types if t is not None],
                default=1.0
            )
            return 80.0 * type_mult

        return best_score

    def _estimate_best_damage_score(self, active: Pokemon, opponent: Pokemon, battle: Battle) -> float:
        """Estimate our best damaging move score."""
        if not active or not opponent:
            return 0.0
        available_moves = getattr(battle, "available_moves", None) if battle else None
        if not available_moves:
            return 0.0
        best_score = 0.0
        for move in available_moves:
            try:
                if move.category == MoveCategory.STATUS:
                    continue
            except Exception:
                continue
            score = self._calculate_move_score(move, active, opponent, battle)
            if score > best_score:
                best_score = score
        return best_score

    def _opponent_known_move_ids(self, opponent: Pokemon) -> set:
        """Return a set of known move ids for opponent."""
        return {m.id for m in self._get_known_moves(opponent)}

    def _has_heavy_duty_boots(self, mon: Pokemon) -> bool:
        if mon is None:
            return False
        item = getattr(mon, "item", None)
        return normalize_name(str(item)) == "heavydutyboots" if item else False

    def _hazard_switch_penalty(self, battle: Battle, mon: Pokemon) -> float:
        if battle is None or mon is None:
            return 0.0
        if self._has_heavy_duty_boots(mon):
            return 0.0
        side_conditions = getattr(battle, "side_conditions", None) or {}
        if not side_conditions:
            return 0.0

        hazard_fraction = 0.0
        if SideCondition.STEALTH_ROCK in side_conditions:
            def_types = [t.name.lower() for t in (mon.types or []) if t is not None]
            rock_mult = get_type_effectiveness("rock", def_types)
            hazard_fraction += (1.0 / 8.0) * rock_mult

        try:
            grounded = battle.is_grounded(mon)
        except Exception:
            grounded = True

        if grounded:
            spikes_layers = side_conditions.get(SideCondition.SPIKES, 0)
            if spikes_layers == 1:
                hazard_fraction += 1.0 / 8.0
            elif spikes_layers == 2:
                hazard_fraction += 1.0 / 6.0
            elif spikes_layers >= 3:
                hazard_fraction += 1.0 / 4.0

        penalty = hazard_fraction * 2.2

        if grounded and SideCondition.TOXIC_SPIKES in side_conditions:
            has_poison = self._opponent_has_type(mon, "poison")
            has_steel = self._opponent_has_type(mon, "steel")
            if not has_poison and not has_steel and mon.status is None:
                penalty += 0.35

        if grounded and SideCondition.STICKY_WEB in side_conditions:
            penalty += 0.25

        return penalty

    def _hazard_switch_penalty_for_opponent(self, battle: Battle, mon: Pokemon) -> float:
        if battle is None or mon is None:
            return 0.0
        mem = self._get_battle_memory(battle)
        species = normalize_name(getattr(mon, "species", ""))
        flags = mem.get("opponent_item_flags", {}).get(species, {})
        if flags.get("has_boots"):
            return 0.0

        side_conditions = getattr(battle, "opponent_side_conditions", None) or {}
        if not side_conditions:
            return 0.0

        hazard_fraction = 0.0
        if SideCondition.STEALTH_ROCK in side_conditions:
            def_types = [t.name.lower() for t in (mon.types or []) if t is not None]
            rock_mult = get_type_effectiveness("rock", def_types)
            hazard_fraction += (1.0 / 8.0) * rock_mult

        try:
            grounded = battle.is_grounded(mon)
        except Exception:
            grounded = True

        if grounded:
            spikes_layers = side_conditions.get(SideCondition.SPIKES, 0)
            if spikes_layers == 1:
                hazard_fraction += 1.0 / 8.0
            elif spikes_layers == 2:
                hazard_fraction += 1.0 / 6.0
            elif spikes_layers >= 3:
                hazard_fraction += 1.0 / 4.0

        penalty = hazard_fraction * 2.2

        if grounded and SideCondition.TOXIC_SPIKES in side_conditions:
            has_poison = self._opponent_has_type(mon, "poison")
            has_steel = self._opponent_has_type(mon, "steel")
            if not has_poison and not has_steel and mon.status is None:
                penalty += 0.35

        if grounded and SideCondition.STICKY_WEB in side_conditions:
            penalty += 0.25

        if flags.get("no_boots"):
            return penalty
        return penalty * 0.5

    def _score_switch(self, switch: Pokemon, opponent: Pokemon, battle: Battle) -> float:
        """Score a potential switch based on matchup and incoming damage risk."""
        if not switch or not opponent:
            return -999.0
        matchup = self._estimate_matchup(switch, opponent)
        reply = self._estimate_best_reply_score(opponent, switch, battle)
        hp = switch.current_hp_fraction if switch.current_hp_fraction is not None else 0.5
        hazard_penalty = self._hazard_switch_penalty(battle, switch)
        return matchup + hp * 0.2 - (reply / 400.0) - hazard_penalty

    def _opponent_is_stallish(self, opponent: Pokemon) -> bool:
        """Heuristic: detect stall tendencies based on known moves and bulk."""
        if opponent is None:
            return False
        move_ids = {m.id for m in self._get_known_moves(opponent)}
        prior_ids = {move_id for move_id, _ in self._get_prior_moves(opponent)}
        if (move_ids | prior_ids) & (self.PROTECT_MOVES | self.RECOVERY_MOVES):
            return True
        bulk = self._stat_estimation(opponent, "def") + self._stat_estimation(opponent, "spd")
        return bulk > 260 and opponent.current_hp_fraction > 0.6


    def _score_move_with_prediction(self, move: Move, active: Pokemon,
                                     opponent: Pokemon, predicted_switch: Optional[Pokemon],
                                     battle: Battle, switch_weight: Optional[float] = None) -> float:
        """Score a move considering likely opponent response."""
        base_score = self._calculate_move_score(move, active, opponent, battle)

        if predicted_switch is not None:
            # They might switch - how does this move do vs their switch-in?
            switch_score = self._calculate_move_score(move, active, predicted_switch, battle)

            # Blend scores with a switch-likelihood weight
            if switch_weight is None:
                switch_weight = 0.4
            switch_weight = max(0.2, min(0.7, switch_weight))
            blended_score = (1.0 - switch_weight) * base_score + switch_weight * switch_score

            # Bonus for trapping moves or moves that punish switching
            if move.id in {'pursuit', 'uturn', 'voltswitch', 'flipturn'}:
                blended_score *= 1.2

            return blended_score

        return base_score

    def _get_effective_speed(self, mon: Pokemon) -> int:
        """Get Pokemon's effective speed including boosts."""
        if mon is None:
            return 100

        # Try to get actual speed stat
        if mon.stats and mon.stats.get('spe'):
            base_speed = mon.stats['spe']
        elif mon.base_stats and mon.base_stats.get('spe'):
            base_speed = mon.base_stats['spe']
        else:
            base_speed = 100

        # Apply boosts
        if mon.boosts:
            boost = mon.boosts.get('spe', 0)
            if boost > 0:
                base_speed = int(base_speed * (2 + boost) / 2)
            elif boost < 0:
                base_speed = int(base_speed * 2 / (2 - boost))

        # Paralysis halves speed
        if mon.status:
            status_id = normalize_name(str(mon.status))
        else:
            status_id = ""
        if status_id in {"par", "paralysis"}:
            base_speed = base_speed // 2

        return base_speed

    def _is_trick_room_active(self, battle: Optional[Battle] = None) -> bool:
        if battle is None:
            battle = getattr(self, "_current_battle", None)
        if battle is None:
            return False
        fields = getattr(battle, "fields", None) or {}
        if Field.TRICK_ROOM in fields:
            try:
                return fields[Field.TRICK_ROOM] > 0
            except Exception:
                return True
        return "trick room" in str(fields).lower()

    def _should_setup_move(self, move: Move, active: Pokemon, opponent: Pokemon) -> bool:
        """Return True if a boosting move is still worth using."""
        if move is None or not move.boosts or active is None:
            return False
        if move.id == "noretreat":
            effects = getattr(active, "effects", None) or {}
            if Effect.NO_RETREAT in effects:
                return False
        for stat, delta in move.boosts.items():
            if delta <= 0:
                continue
            current = active.boosts.get(stat, 0) if active.boosts else 0
            cap = self.SETUP_BOOST_CAPS.get(stat, 2)
            if current >= cap:
                continue
            if stat == "spe" and opponent is not None:
                if self._is_trick_room_active():
                    continue
                if self._get_effective_speed(active) > self._get_effective_speed(opponent) * 1.05:
                    continue
            return True
        return False

    def _is_stellar_type(self, tera_type) -> bool:
        if tera_type is None:
            return False
        name = tera_type.name if hasattr(tera_type, "name") else str(tera_type)
        return normalize_name(name) == "stellar"

    def _opponent_is_stellar_tera(self, opponent: Pokemon) -> bool:
        if opponent is None:
            return False
        terastallized = getattr(opponent, "terastallized", None)
        if terastallized:
            return normalize_name(str(terastallized)) == "stellar"
        return False

    def _opponent_is_set_up(self, opponent: Pokemon) -> bool:
        if opponent is None or not opponent.boosts:
            return False
        positives = [b for b in opponent.boosts.values() if b > 0]
        if not positives:
            return False
        return max(positives) >= 2 or sum(positives) >= 3

    def _is_switch_churn_risk(self, battle: Battle) -> bool:
        if not self.ANTI_SWITCH_CHURN:
            return False
        if battle is None or battle.force_switch:
            return False
        if not battle.available_switches or not battle.available_moves:
            return False
        active = battle.active_pokemon
        opponent = battle.opponent_active_pokemon
        if active is None or opponent is None:
            return False

        mem = self._get_battle_memory(battle)
        switch_streak = int(mem.get("self_switch_streak", 0) or 0)
        if switch_streak < 2:
            return False

        last_opp = normalize_name(mem.get("last_opponent_species"))
        current_opp = normalize_name(getattr(opponent, "species", ""))
        if last_opp and current_opp and last_opp != current_opp:
            return False

        last_opp_hp = mem.get("last_opponent_hp")
        current_opp_hp = opponent.current_hp_fraction
        if not isinstance(last_opp_hp, (int, float)) or not isinstance(current_opp_hp, (int, float)):
            return False
        if abs(current_opp_hp - last_opp_hp) > 0.05:
            return False

        current_matchup = self._estimate_matchup(active, opponent)
        current_reply = self._estimate_best_reply_score(opponent, active, battle)
        active_hp = active.current_hp_fraction if active.current_hp_fraction is not None else 0.5
        current_score = current_matchup + active_hp * 0.2 - (current_reply / 400.0)
        best_switch_score = max(self._score_switch(sw, opponent, battle) for sw in battle.available_switches)
        no_real_upgrade = best_switch_score <= current_score + 0.2

        speed_boost = int((opponent.boosts or {}).get("spe", 0) or 0) >= 2
        speed_gap = self._get_effective_speed(opponent) > self._get_effective_speed(active) * 1.05
        setup_threat = self._opponent_is_set_up(opponent)
        # Only break switch loops for true speed-control snowballs where switching
        # no longer meaningfully improves the position.
        return no_real_upgrade and (speed_boost or (setup_threat and speed_gap))

    def _choose_emergency_non_switch_order(
        self,
        battle: Battle,
        active: Pokemon,
        opponent: Pokemon,
        n_remaining_mons: int,
    ):
        if not battle.available_moves:
            return None
        best_move = None
        best_score = -1.0
        my_speed = self._get_effective_speed(active)
        opp_speed = self._get_effective_speed(opponent)
        opp_set_up = self._opponent_is_set_up(opponent)

        for move in battle.available_moves:
            move_entry = self._get_move_entry(move)
            status_kind = (
                move.category == MoveCategory.STATUS
                or self._status_from_move_entry(move_entry) is not None
            )
            if self._sleep_clause_blocked(battle) and self._move_inflicts_sleep(move):
                continue

            if move.id in self.ANTI_SETUP_MOVES and opp_set_up:
                score = 420.0
            elif status_kind:
                score = self._should_use_status_move(move, active, opponent, battle)
                if score <= 0:
                    continue
                status_type = self.STATUS_MOVES.get(move.id) or self._status_from_move_entry(move_entry)
                if opp_set_up and status_type in {"burn", "para", "sleep", "taunt", "encore", "yawn"}:
                    score *= 1.25
            else:
                score = self._calculate_move_score(move, active, opponent, battle, apply_recoil=False)
                if score <= 0:
                    continue

            try:
                move_priority = int(move_entry.get("priority", 0) or 0)
            except Exception:
                move_priority = 0
            if opp_speed > my_speed and move_priority > 0:
                score *= 1.2

            if score > best_score:
                best_score = score
                best_move = move

        if best_move is None:
            best_move = battle.available_moves[0]

        return self.create_order(
            best_move,
            dynamax=self._should_dynamax(battle, n_remaining_mons),
            terastallize=self._should_terastallize(battle, best_move),
        )

    def _should_use_protect(self, battle: Battle, reply_score: float) -> bool:
        mem = self._get_battle_memory(battle)
        last_protect_turn = mem.get("last_protect_turn", -99)
        if battle.turn - last_protect_turn <= 1:
            return False
        active = battle.active_pokemon
        opponent = battle.opponent_active_pokemon
        if not active or not opponent:
            return False
        unknown_moves = len(self._get_known_moves(opponent)) == 0
        if unknown_moves and battle.turn <= 2 and active.current_hp_fraction >= 0.6:
            return True
        if reply_score >= 220 * active.current_hp_fraction and active.current_hp_fraction >= 0.5:
            return True
        return False

    def _future_sight_recent(self, battle: Battle) -> bool:
        mem = self._get_battle_memory(battle)
        last_turn = mem.get("last_future_sight_turn", -99)
        return battle.turn - last_turn <= 3

    def _self_debuff_penalty(self, move: Move, active: Pokemon) -> float:
        entry = self._get_move_entry(move)
        boosts = (entry.get("self") or {}).get("boosts", {})
        if not boosts:
            return 1.0
        penalty = 1.0
        for stat, delta in boosts.items():
            if delta >= 0:
                continue
            current = active.boosts.get(stat, 0) if active.boosts else 0
            if current <= -4:
                penalty *= 0.3
            elif current <= -2:
                penalty *= 0.55
            elif current <= -1:
                penalty *= 0.75
            else:
                penalty *= 0.9
        return penalty

    def _stat_estimation(self, mon: Pokemon, stat: str) -> float:
        """Estimate effective stat value including boosts."""
        if mon is None:
            return 100.0

        # Get base stat
        if mon.stats and stat in mon.stats and mon.stats[stat]:
            base = mon.stats[stat]
        elif mon.base_stats and stat in mon.base_stats:
            base = mon.base_stats[stat]
        else:
            base = 100
        try:
            base = float(base)
        except Exception:
            base = 100.0

        # Apply boosts
        boost = 0
        try:
            boosts = mon.boosts or {}
            if isinstance(boosts, dict):
                boost = boosts.get(stat, 0) or 0
            boost = int(boost)
        except Exception:
            boost = 0
        if boost > 6:
            boost = 6
        elif boost < -6:
            boost = -6
        if boost > 0:
            multiplier = (2 + boost) / 2
        else:
            multiplier = 2 / (2 - boost)

        return base * multiplier

    def _should_switch_out(self, battle: Battle) -> bool:
        """Determine if we should switch out."""
        active = battle.active_pokemon
        opponent = battle.opponent_active_pokemon

        if not active or not opponent:
            return False
        if self._is_switch_churn_risk(battle):
            return False

        # Check if there's a better switch option
        if not battle.available_switches:
            return False
        current_matchup = self._estimate_matchup(active, opponent)
        current_reply = self._estimate_best_reply_score(opponent, active, battle)
        active_hp = active.current_hp_fraction if active.current_hp_fraction is not None else 0.5
        current_score = current_matchup + active_hp * 0.2 - (current_reply / 400.0)
        best_switch = max(
            battle.available_switches,
            key=lambda m: self._score_switch(m, opponent, battle)
        )
        best_score = self._score_switch(best_switch, opponent, battle)
        if current_score < 0.15 and (best_score - current_score) >= 0.5:
            return True

        # Switch if badly debuffed
        if active.boosts:
            if active.boosts.get("def", 0) <= -3 or active.boosts.get("spd", 0) <= -3:
                return True
            # Physical attacker with -3 Atk
            if active.boosts.get("atk", 0) <= -3 and self._stat_estimation(active, "atk") >= self._stat_estimation(active, "spa"):
                return True
            # Special attacker with -3 SpA
            if active.boosts.get("spa", 0) <= -3 and self._stat_estimation(active, "atk") <= self._stat_estimation(active, "spa"):
                return True

        # Very bad matchup
        if current_matchup < self.SWITCH_OUT_MATCHUP_THRESHOLD:
            return True

        return False

    def _should_use_status_move(self, move: Move, active: Pokemon, opponent: Pokemon, battle: Battle) -> float:
        """Return a score for using a status move. Higher = better opportunity."""
        move_entry = self._get_move_entry(move)
        status_type = self.STATUS_MOVES.get(move.id)
        if status_type is None:
            status_type = self._status_from_move_entry(move_entry)
        if status_type is None:
            return 0.0

        # Don't status already statused Pokemon (except seed/taunt/encore)
        if opponent.status is not None and status_type not in {"seed", "taunt", "encore"}:
            return 0.0

        if getattr(self, "STATUS_KO_GUARD", False) and battle is not None:
            best_damage = self._estimate_best_damage_score(active, opponent, battle)
            if best_damage >= getattr(self, "STATUS_KO_THRESHOLD", 200.0):
                return 0.0

        score = 0.0

        # Check accuracy
        accuracy = (move.accuracy / 100.0) if move.accuracy else 1.0
        if accuracy < 0.7:  # Too risky
            return 0.0

        # Sleep is great but only use Spore (100% accurate)
        if status_type == 'sleep':
            if move.id == 'spore':  # 100% accurate sleep
                score = 400.0
            else:  # Other sleep moves are risky
                score = 140.0

        # Burn is situational - only vs clear physical attackers at high HP
        elif status_type == 'burn':
            opp_atk = self._stat_estimation(opponent, "atk")
            opp_spa = self._stat_estimation(opponent, "spa")
            if opp_atk > opp_spa * 1.1 and opponent.current_hp_fraction > 0.5:
                if self._opponent_has_type(opponent, "fire"):
                    score = 0.0
                else:
                    score = 180.0
            else:
                score = 0.0

        # Toxic only vs slow/bulky mons
        elif status_type == 'poison':
            opp_speed = self._get_effective_speed(opponent)
            opp_def = self._stat_estimation(opponent, "def")
            opp_spd = self._stat_estimation(opponent, "spd")
            if self._opponent_has_type(opponent, "steel") or self._opponent_has_type(opponent, "poison"):
                score = 0.0
            elif opponent.current_hp_fraction > 0.5 and (opp_speed < 100 or (opp_def + opp_spd) > 220):
                score = 160.0
            else:
                score = 0.0

        # Thunder Wave only vs significantly faster threats
        elif status_type == 'para':
            if self.PARA_FINISH_GUARD and battle is not None:
                opp_hp = opponent.current_hp_fraction or 0.0
                if opp_hp <= self.PARA_FINISH_MAX_OPP_HP:
                    best_damage = self._estimate_best_damage_score(active, opponent, battle)
                    threshold = self.PARA_FINISH_KO_THRESHOLD * max(opp_hp, 0.05)
                    if best_damage >= threshold:
                        return 0.0
            if self._is_trick_room_active():
                score = 0.0
            else:
                opp_speed = self._get_effective_speed(opponent)
                my_speed = self._get_effective_speed(active)
                if opp_speed > my_speed * 1.05:
                    if move.id == "thunderwave" and (
                        self._opponent_has_type(opponent, "ground") or
                        self._opponent_has_type(opponent, "electric")
                    ):
                        score = 0.0
                    else:
                        score = 140.0
                else:
                    score = 0.0

        # Yawn is situational
        elif status_type == 'yawn':
            if opponent.current_hp_fraction > 0.5:
                score = 110.0
            else:
                score = 0.0

        # Leech Seed: strong vs bulky non-Grass
        elif status_type == 'seed':
            opp_def = self._stat_estimation(opponent, "def")
            opp_spd = self._stat_estimation(opponent, "spd")
            if self._opponent_has_type(opponent, "grass"):
                score = 0.0
            elif opponent.current_hp_fraction > 0.6 and (opp_def + opp_spd) > 220:
                score = 150.0
            else:
                score = 80.0

        # Taunt/Encore: suppress setup/stall
        elif status_type in {"taunt", "encore"}:
            if self._opponent_is_stallish(opponent) or (opponent.boosts and sum(opponent.boosts.values()) > 1):
                score = 130.0
            else:
                score = 60.0

        # Strength Sap: heal + drop attack
        if status_type == "sap" or move.id == "strengthsap":
            opp_atk = self._stat_estimation(opponent, "atk")
            if active.current_hp_fraction < 0.75 and opp_atk > 120:
                score = max(score, 200.0)

        # Apply accuracy penalty
        score *= accuracy

        if self._opponent_is_stallish(opponent):
            if status_type in {"poison", "seed", "taunt", "encore", "yawn"}:
                score *= 1.6
            elif status_type in {"burn", "para", "sleep"}:
                score *= 1.3
            else:
                score *= 1.1

        # Role-aware adjustments
        my_role = self._get_role(active)
        opp_role = self._get_role(opponent)
        if my_role in {"wall", "tank"}:
            score *= 1.2
        if my_role in {"sweeper", "fast_attacker"} and status_type in {"poison", "seed", "yawn", "taunt", "encore"}:
            score *= 0.7
        if opp_role == "sweeper" and status_type in {"para", "burn", "sleep"}:
            score *= 1.25
        if opp_role in {"wall", "tank"} and status_type in {"poison", "seed"}:
            score *= 1.25

        known_moves = self._opponent_known_move_ids(opponent)
        if known_moves & (self.RECOVERY_MOVES | self.PROTECT_MOVES):
            if status_type in {"poison", "seed", "taunt", "encore", "yawn"}:
                score *= 1.15

        # Status bonus vs bulky/faster opponents when we are healthy
        opp_def = self._stat_estimation(opponent, "def")
        opp_spd = self._stat_estimation(opponent, "spd")
        opp_speed = self._get_effective_speed(opponent)
        my_speed = self._get_effective_speed(active)
        if active.current_hp_fraction >= 0.6:
            if (opp_def + opp_spd) > 220 or opp_speed > my_speed:
                score *= 1.2

        # Reduce value if we're low HP (should attack instead)
        if active.current_hp_fraction < 0.4:
            score *= 0.5

        return score

    def _calculate_move_score(
        self,
        move: Move,
        active: Pokemon,
        opponent: Pokemon,
        battle: Battle,
        apply_recoil: bool = True,
        respect_immunity_memory: bool = True,
    ) -> float:
        """Calculate move score - similar to SimpleHeuristics but with improvements."""
        # Skip status moves entirely - focus on damage
        if move.category == MoveCategory.STATUS:
            return 0.0

        base_power = move.base_power or 0
        if base_power == 0:
            return 0.0

        if respect_immunity_memory and battle and opponent:
            mem = self._get_battle_memory(battle)
            species = opponent.species if hasattr(opponent, "species") else None
            if species:
                immune_moves = mem.get("immune_moves", {}).get(species, set())
                if move.id in immune_moves:
                    return 0.0
                immune_types = mem.get("immune_types", {}).get(species, set())
                move_type_id = self._move_type_id(move)
                if move_type_id in immune_types:
                    return 0.0

        # Stat ratio for damage calculation
        if move.category == MoveCategory.PHYSICAL:
            stat_ratio = self._stat_estimation(active, "atk") / max(self._stat_estimation(opponent, "def"), 1)
        else:
            stat_ratio = self._stat_estimation(active, "spa") / max(self._stat_estimation(opponent, "spd"), 1)

        # STAB bonus
        stab = 1.5 if move.type and active.types and move.type in active.types else 1.0

        # Type effectiveness
        if opponent and self._opponent_is_stellar_tera(opponent):
            type_mult = 1.0
        else:
            type_mult = opponent.damage_multiplier(move) if opponent else 1.0

        # Ability-based immunities and reductions
        if opponent and self._ability_blocks_move(move, opponent):
            return 0.0
        type_mult *= self._ability_damage_multiplier(move, opponent)

        # Accuracy
        accuracy = (move.accuracy / 100.0) if move.accuracy else 1.0

        # Expected hits (for multi-hit moves)
        expected_hits = move.expected_hits if hasattr(move, 'expected_hits') and move.expected_hits else 1.0

        raw_score = base_power * stab * stat_ratio * accuracy * expected_hits * type_mult
        score = raw_score

        # Avoid spamming Future Sight / Doom Desire
        if move.id in self.FUTURE_SIGHT_MOVES and self._future_sight_recent(battle):
            score *= 0.3

        # Penalize repeated self-debuff moves (Draco Meteor, Close Combat, etc.)
        score *= self._self_debuff_penalty(move, active)

        # Penalize low-accuracy moves unless they are clearly strong
        if accuracy < 0.85:
            score *= 0.85
        elif accuracy < 0.9:
            score *= 0.92

        # Bonus for priority moves when we're slower or at low HP
        if move.id in self.PRIORITY_MOVES:
            my_speed = self._get_effective_speed(active)
            opp_speed = self._get_effective_speed(opponent)
            if opp_speed > my_speed:
                score *= 1.3  # Bonus for going first when normally slower
            if active.current_hp_fraction < 0.4:
                score *= 1.5  # Big bonus at low HP

        # Recoil awareness: avoid self-KO unless it wins immediately
        if apply_recoil:
            recoil_rate = self._move_recoil_rate(move)
            if recoil_rate > 0 and opponent is not None:
                expected_damage_fraction = min(1.0, raw_score / 220.0)
                expected_recoil = recoil_rate * expected_damage_fraction
                likely_ko = raw_score >= 220 * opponent.current_hp_fraction
                if active.current_hp_fraction <= expected_recoil + 0.02:
                    score *= 0.7 if likely_ko else 0.25
                elif active.current_hp_fraction < 0.35:
                    score *= 0.85

        return score

    def _move_type_id(self, move: Move) -> Optional[str]:
        if move is None or move.type is None:
            return None
        move_type = move.type
        if hasattr(move_type, "name"):
            return move_type.name.lower()
        return str(move_type).lower()

    def _get_move_entry(self, move: Move) -> dict:
        """Return moves.json entry for a move if available."""
        if move is None:
            return {}
        moves_data = load_moves()
        return moves_data.get(move.id, {})

    def _is_recovery_move(self, move: Move) -> bool:
        """Detect recovery moves via id or moves.json heal data."""
        if move is None:
            return False
        if move.id in self.RECOVERY_MOVES:
            return True
        entry = self._get_move_entry(move)
        return bool(entry.get("heal"))

    def _status_from_move_entry(self, entry: dict) -> Optional[str]:
        """Map move entry status/volatileStatus to internal status types."""
        if not entry:
            return None
        status = entry.get("status")
        if status:
            status = normalize_name(status)
            if status in {"psn", "tox"}:
                return "poison"
            if status == "brn":
                return "burn"
            if status == "par":
                return "para"
            if status == "slp":
                return "sleep"
        vol = entry.get("volatileStatus")
        if vol:
            vol = normalize_name(vol)
            if vol == "yawn":
                return "yawn"
            if vol == "leechseed":
                return "seed"
            if vol in {"taunt", "encore"}:
                return vol
        name = normalize_name(entry.get("name", ""))
        if name == "strengthsap":
            return "sap"
        return None

    def _canonicalize_move_id(self, move_id: str) -> str:
        if not move_id:
            return ""
        if ":" in move_id:
            move_id = move_id.split(":", 1)[1]
        return normalize_name(move_id)

    def _opponent_has_type(self, opponent: Pokemon, type_id: str) -> bool:
        if opponent is None:
            return False
        type_id = normalize_name(type_id)
        terastallized = getattr(opponent, "terastallized", None)
        if terastallized:
            return normalize_name(str(terastallized)) == type_id
        if getattr(opponent, "is_terastallized", False):
            tera_type = getattr(opponent, "tera_type", None)
            if tera_type is not None:
                name = tera_type.name if hasattr(tera_type, "name") else str(tera_type)
                return normalize_name(name) == type_id
        for t in (opponent.types or []):
            if t is None:
                continue
            if normalize_name(t.name if hasattr(t, "name") else str(t)) == type_id:
                return True
        return False

    def _opponent_is_ghost(self, opponent: Pokemon) -> bool:
        if opponent is None:
            return False
        if self._opponent_has_type(opponent, "ghost"):
            return True
        terastallized = getattr(opponent, "terastallized", None)
        if terastallized and normalize_name(str(terastallized)) == "ghost":
            return True
        return False

    def _priority_blocked(self, battle: Battle, opponent: Pokemon) -> bool:
        if not opponent or not battle:
            return False
        ability_id = self._get_ability_id(opponent)
        if ability_id in {"queenlymajesty", "dazzling", "armortail"}:
            return True
        terrain = str(battle.fields).lower() if battle.fields else ""
        if "psychic" in terrain:
            if self._opponent_has_type(opponent, "flying"):
                return False
            if ability_id == "levitate":
                return False
            return True
        return False

    def _active_has_move_id(self, active: Pokemon, move_id: str) -> bool:
        if active is None:
            return False
        return any(m.id == move_id for m in self._get_known_moves(active))

    def _get_role(self, mon: Pokemon) -> str:
        """Classify a Pokemon into a coarse role for heuristic scoring."""
        if mon is None:
            return "unknown"
        try:
            hp = (mon.stats.get("hp") or 100) if mon.stats else 100
            atk = (mon.stats.get("atk") or 100) if mon.stats else 100
            spa = (mon.stats.get("spa") or 100) if mon.stats else 100
            dfn = (mon.stats.get("def") or 100) if mon.stats else 100
            spd = (mon.stats.get("spd") or 100) if mon.stats else 100
            spe = (mon.stats.get("spe") or 100) if mon.stats else 100
        except Exception:
            hp, atk, spa, dfn, spd, spe = 100, 100, 100, 100, 100, 100

        offense = max(atk, spa)
        bulk = (dfn + spd) / 2

        if spe >= 110 and offense >= 110 and bulk < 90:
            return "sweeper"
        if bulk >= 125 and offense < 95:
            return "wall"
        if bulk >= 110 and offense >= 100:
            return "tank"
        if spe >= 100 and offense >= 95:
            return "fast_attacker"
        return "balanced"

    def _get_ability_id(self, opponent: Pokemon) -> Optional[str]:
        ability = getattr(opponent, "ability", None)
        if not ability:
            return None
        ability_id = self._canon_id(ability)
        abilities_data = load_abilities()
        if abilities_data and ability_id not in abilities_data:
            return None
        return ability_id

    def _possible_abilities(self, opponent: Pokemon) -> set:
        species = normalize_name(getattr(opponent, "species", ""))
        if not species:
            return set()
        dex = self._load_pokedex()
        entry = dex.get(species, {})
        abilities = entry.get("abilities", {}) if isinstance(entry, dict) else {}
        result = set()
        if isinstance(abilities, dict):
            for val in abilities.values():
                if val:
                    result.add(normalize_name(val))
        return result

    def _can_have_speed_modified(self, battle: Battle, opponent: Pokemon) -> bool:
        if opponent is None:
            return False
        ability_id = self._get_ability_id(opponent)
        possible = {ability_id} if ability_id else self._possible_abilities(opponent)
        if not possible:
            return False

        try:
            weather = getattr(battle, "weather", None)
            if isinstance(weather, dict) and weather:
                weather = next(iter(weather.keys()))
            if isinstance(weather, Weather):
                weather_id = normalize_name(weather.name)
            else:
                weather_id = normalize_name(str(weather)) if weather else ""
        except Exception:
            weather_id = ""
        if weather_id in {"raindance", "rain"} and "swiftswim" in possible:
            return True
        if weather_id in {"sunnyday", "sun"} and "chlorophyll" in possible:
            return True
        if weather_id in {"sandstorm"} and "sandrush" in possible:
            return True
        if weather_id in {"hail", "snowscape", "snow"} and "slushrush" in possible:
            return True

        try:
            fields = getattr(battle, "fields", None) or {}
            if Field.ELECTRIC_TERRAIN in fields and "surgesurfer" in possible:
                return True
        except Exception:
            pass

        item = getattr(opponent, "item", None)
        if not item and "unburden" in possible:
            return True

        status = normalize_name(str(getattr(opponent, "status", "")))
        if status in {"par", "paralysis"} and "quickfeet" in possible:
            return True

        return False

    def _ability_blocks_move(self, move: Move, opponent: Pokemon) -> bool:
        """Return True if opponent's revealed ability makes the move ineffective."""
        if opponent is None:
            return False
        ability_id = self._get_ability_id(opponent)
        if not ability_id:
            return False
        if getattr(move, "ignore_ability", False):
            return False

        move_type = self._move_type_id(move)
        if not move_type:
            return False

        ability_mods = ABILITY_DAMAGE_MODS.get(ability_id, {})
        immune_key = f"{move_type}_immune"
        if ability_mods.get(immune_key):
            return True

        # Use abilities.json names to broaden immunity checks
        abilities_data = load_abilities()
        ability_name = ""
        if abilities_data and ability_id in abilities_data:
            ability_name = normalize_name(abilities_data[ability_id].get("name", ""))

        name_based_immunities = {
            "waterabsorb": "water",
            "stormdrain": "water",
            "dryskin": "water",
            "voltabsorb": "electric",
            "lightningrod": "electric",
            "motordrive": "electric",
            "sapsipper": "grass",
            "flashfire": "fire",
            "wellbakedbody": "fire",
            "eartheater": "ground",
            "levitate": "ground",
        }
        if ability_id in name_based_immunities and move_type == name_based_immunities[ability_id]:
            return True
        if ability_name in name_based_immunities and move_type == name_based_immunities[ability_name]:
            return True

        if ability_id == "wonderguard":
            try:
                if opponent.damage_multiplier(move) <= 1.0:
                    return True
            except Exception:
                pass

        return False

    def _ability_damage_multiplier(self, move: Move, opponent: Pokemon) -> float:
        """Return a damage multiplier for revealed abilities (non-immune)."""
        if opponent is None:
            return 1.0
        ability_id = self._get_ability_id(opponent)
        if not ability_id or getattr(move, "ignore_ability", False):
            return 1.0

        move_type = self._move_type_id(move)
        if not move_type:
            return 1.0

        ability_mods = ABILITY_DAMAGE_MODS.get(ability_id, {})
        if ability_mods.get("damage_reduction"):
            if not ability_mods.get("at_full_hp") or opponent.current_hp_fraction >= 0.99:
                return float(ability_mods["damage_reduction"])
        if ability_id == "thickfat" and move_type in {"fire", "ice"}:
            return 0.5
        return 1.0

    def _should_terastallize(self, battle: Battle, move: Move) -> bool:
        """Decide whether to terastallize."""
        active = battle.active_pokemon
        opponent = battle.opponent_active_pokemon

        if not battle.can_tera or not active or not opponent:
            return False

        n_remaining = len([m for m in battle.team.values() if m is not None and not m.fainted])
        n_opp_remaining = len([m for m in battle.opponent_team.values() if m is not None and not m.fainted])
        endgame = n_remaining <= 2 or n_opp_remaining <= 2
        move_damage = self._calculate_move_score(
            move, active, opponent, battle, apply_recoil=False
        )
        likely_ko = move_damage >= 220 * opponent.current_hp_fraction

        # Calculate offensive and defensive tera value
        try:
            current_offensive = opponent.damage_multiplier(move.type)
            if active.tera_type is not None:
                tera_offensive = (
                    opponent.damage_multiplier(active.tera_type)
                    if move.type != active.tera_type
                    else current_offensive
                )
            else:
                tera_offensive = current_offensive

            # Simple heuristic: tera if it improves offense significantly
            if active.tera_type is not None and tera_offensive > current_offensive * 1.4:
                if endgame or likely_ko or opponent.current_hp_fraction > 0.6 or n_remaining < n_opp_remaining:
                    return True

            if active.tera_type is not None and tera_offensive > current_offensive * 1.25:
                if endgame and (likely_ko or opponent.current_hp_fraction > 0.35):
                    return True

            # Or if we get STAB bonus from tera
            if active.tera_type is not None and active.tera_type == move.type and active.tera_type not in active.types:
                if endgame or likely_ko or n_remaining < n_opp_remaining:
                    return True

            # Defensive tera: gain immunity or strong resistance to opponent's likely types
            if active.tera_type is not None:
                try:
                    max_incoming = 1.0
                    max_incoming_tera = 1.0
                    for opp_type in (opponent.types or []):
                        opp_type_name = opp_type.name.lower()
                        def_types = [t.name.lower() for t in (active.types or []) if t is not None]
                        max_incoming = max(max_incoming, get_type_effectiveness(opp_type_name, def_types))
                        max_incoming_tera = max(
                        max_incoming_tera,
                        get_type_effectiveness(opp_type_name, [active.tera_type.name.lower()])
                    )

                    if max_incoming >= 2.0 and max_incoming_tera <= 0.5:
                        if endgame or active.current_hp_fraction < 0.55:
                            return True
                except Exception:
                    pass

            # Fallback: if tera type is hidden, allow it for big swings
            if active.tera_type is None:
                if endgame and (likely_ko or active.current_hp_fraction < 0.35):
                    return True
                if n_remaining < n_opp_remaining and (likely_ko or current_offensive >= 1.5):
                    return True
        except:
            pass

        return False

    def _should_dynamax(self, battle: Battle, n_remaining_mons: int) -> bool:
        """Decide whether to dynamax."""
        if not battle.can_dynamax:
            return False

        active = battle.active_pokemon
        opponent = battle.opponent_active_pokemon

        if not active or not opponent:
            return False

        # Dynamax on last mon
        if n_remaining_mons == 1:
            return True

        # Dynamax with matchup advantage at full HP
        if (self._estimate_matchup(active, opponent) > 0.5 and
            active.current_hp_fraction == 1 and
            opponent.current_hp_fraction > 0.5):
            return True

        # Last full HP mon
        full_hp_mons = len([m for m in battle.team.values() if m.current_hp_fraction == 1])
        if full_hp_mons == 1 and active.current_hp_fraction == 1:
            return True

        return False

    def choose_move(self, battle: AbstractBattle):
        """Main move selection logic with switch prediction and status usage."""
        if not isinstance(battle, Battle):
            return self.choose_random_move(battle)
        noop_order = self._empty_order_if_no_choices(battle)
        if noop_order is not None:
            return noop_order

        self._current_battle = battle
        self._update_immunity_memory(battle)
        self._update_active_turns(battle)
        self._update_battle_memory(battle)
        self._update_speed_order_memory(battle)
        self._update_opponent_item_memory(battle)
        self._cleanup_battle_memory(battle)

        active = battle.active_pokemon
        opponent = battle.opponent_active_pokemon

        if active is None or opponent is None:
            return self.choose_random_move(battle)

        # Force switch handling
        if battle.force_switch:
            if battle.available_switches:
                best_switch = max(
                    battle.available_switches,
                    key=lambda s: self._score_switch(s, opponent, battle)
                )
                return self._commit_order(battle, self.create_order(best_switch))
            return self.choose_random_move(battle)

        mem = self._get_battle_memory(battle)

        # Fake Out: use on first active turn unless blocked or immune
        if battle.available_moves and mem.get("active_turns", 0) == 0:
            for move in battle.available_moves:
                if move.id == "fakeout":
                    if not self._opponent_is_ghost(opponent) and not self._priority_blocked(battle, opponent):
                        return self._commit_order(battle, self.create_order(move))

        # Count remaining mons
        n_remaining = len([m for m in battle.team.values() if not m.fainted])
        n_opp_remaining = 6 - len([m for m in battle.opponent_team.values() if m.fainted])
        is_endgame = n_remaining <= 2 or n_opp_remaining <= 2
        ahead = n_remaining > n_opp_remaining
        behind = n_remaining < n_opp_remaining
        play_safe = is_endgame and ahead
        opp_set_up = self._opponent_is_set_up(opponent)
        opp_stallish = self._opponent_is_stallish(opponent)
        predicted_move = self._predict_opponent_move(opponent, active, battle)
        predicted_kind = predicted_move.get("kind")
        predicted_damage = predicted_move.get("damage_score", 0.0)
        switch_churn = self._is_switch_churn_risk(battle)

        # Check if we should switch out
        if battle.available_moves and (switch_churn or not self._should_switch_out(battle) or not battle.available_switches):
            # Priority 0: punish boosted opponents
            if self._opponent_is_set_up(opponent):
                for move in battle.available_moves:
                    if move.id in self.ANTI_SETUP_MOVES:
                        return self._commit_order(battle, self.create_order(move))
                for move in battle.available_moves:
                    move_entry = self._get_move_entry(move)
                    status_kind = (
                        move.category == MoveCategory.STATUS or
                        self._status_from_move_entry(move_entry) is not None
                    )
                    if not status_kind:
                        continue
                    status_score = self._should_use_status_move(move, active, opponent, battle)
                    if status_score > 0:
                        return self._commit_order(battle, self.create_order(move))
                if battle.available_switches:
                    best_switch = max(
                        battle.available_switches,
                        key=lambda s: self._score_switch(s, opponent, battle)
                    )
                    if self._estimate_matchup(best_switch, opponent) > self._estimate_matchup(active, opponent) + 0.3:
                        return self._commit_order(battle, self.create_order(best_switch))

            # Priority 0.5: Protect scouting
            if any(m.id in self.PROTECT_MOVES for m in battle.available_moves):
                reply_score = self._estimate_best_reply_score(opponent, active, battle)
                if self._should_use_protect(battle, reply_score):
                    for move in battle.available_moves:
                        if move.id in self.PROTECT_MOVES:
                            mem = self._get_battle_memory(battle)
                            mem["last_protect_turn"] = battle.turn
                            return self._commit_order(battle, self.create_order(move))

            # Priority 1: Entry hazard setup
            for move in battle.available_moves:
                if (not is_endgame and n_opp_remaining >= 3 and
                    move.id in self.ENTRY_HAZARDS and
                    self.ENTRY_HAZARDS[move.id] not in battle.opponent_side_conditions):
                    reply_score = self._estimate_best_reply_score(opponent, active, battle)
                    if behind and reply_score >= 200 * active.current_hp_fraction:
                        continue
                    return self._commit_order(battle, self.create_order(move))

            # Priority 2: Hazard removal
            if battle.side_conditions:
                for move in battle.available_moves:
                    if move.id in self.ANTI_HAZARDS_MOVES and n_remaining >= 2:
                        return self._commit_order(battle, self.create_order(move))

            # Priority 2.5: Recovery when safe
            if active.current_hp_fraction < 0.55:
                reply_score = self._estimate_best_reply_score(opponent, active, battle)
                safe_recover = reply_score < 150
                if self._estimate_matchup(active, opponent) > 0.35 and active.current_hp_fraction < 0.4:
                    safe_recover = True
                if behind and reply_score > 140:
                    safe_recover = False
                if safe_recover:
                    for move in battle.available_moves:
                        if self._is_recovery_move(move):
                            if move.id == "rest" and not self._active_has_move_id(active, "sleeptalk"):
                                if n_remaining > 1 or reply_score >= 140:
                                    continue
                            return self._commit_order(battle, self.create_order(move))

            # Priority 3: Setup moves (but be smarter about timing)
            if active.current_hp_fraction >= 0.9 and not play_safe and not opp_set_up:
                can_setup = not (
                    predicted_kind == "damage" and predicted_damage >= 180 * active.current_hp_fraction
                )
                if behind and predicted_damage >= 140 * active.current_hp_fraction:
                    can_setup = False
                if can_setup:
                    matchup = self._estimate_matchup(active, opponent)
                    if matchup > 0.3:  # Good matchup
                        for move in battle.available_moves:
                            if (move.boosts and
                                move.target and "self" in str(move.target).lower() and
                                sum(move.boosts.values()) >= 2):
                                if self._should_setup_move(move, active, opponent):
                                    reply_score = self._estimate_best_reply_score(opponent, active, battle)
                                    if reply_score < 180 or active.current_hp_fraction >= 0.95:
                                        return self._commit_order(battle, self.create_order(move))

            # Anti-stall behavior: punish protect/recover loops
            if self._opponent_is_stallish(opponent):
                if active.status is not None and battle.available_switches:
                    best_switch = max(
                        battle.available_switches,
                        key=lambda s: self._estimate_matchup(s, opponent)
                    )
                    if self._estimate_matchup(best_switch, opponent) > self._estimate_matchup(active, opponent) + 0.3:
                        return self._commit_order(battle, self.create_order(best_switch))

                if active.current_hp_fraction >= 0.6 and not opp_set_up:
                    for move in battle.available_moves:
                        if (move.boosts and
                            move.target and "self" in str(move.target).lower() and
                            sum(move.boosts.values()) >= 1):
                            if self._should_setup_move(move, active, opponent):
                                return self._commit_order(battle, self.create_order(move))

                for move in battle.available_moves:
                    if move.category == MoveCategory.STATUS:
                        status_score = self._should_use_status_move(move, active, opponent, battle)
                        if status_score > 0:
                            return self._commit_order(battle, self.create_order(move))

            predicted_switch = self._predict_opponent_switch(battle)
            switch_weight = self._switch_likelihood(opponent, active, battle) if predicted_switch else 0.0
            action_dist = self._opponent_action_distribution(
                opponent, active, battle, predicted_switch=predicted_switch
            )
            matchup = self._estimate_matchup(active, opponent)
            reply_score = self._estimate_best_reply_score(opponent, active, battle)
            best_damage_score = self._estimate_best_damage_score(active, opponent, battle)
            my_speed = self._get_effective_speed(active)
            opp_speed = self._get_effective_speed(opponent)
            opp_priority_threat = self._opponent_likely_has_priority(opponent) or predicted_move.get("priority", False)

            # Endgame: prioritize safe KO lines
            if is_endgame and battle.available_moves:
                endgame_ko_move = None
                endgame_ko_risk = -1.0
                endgame_ko_score = -1.0
                for move in battle.available_moves:
                    if move.category == MoveCategory.STATUS:
                        continue
                    raw_score = self._calculate_move_score(
                        move, active, opponent, battle, apply_recoil=False
                    )
                    if raw_score < 220 * opponent.current_hp_fraction:
                        continue
                    accuracy = move.accuracy
                    if accuracy is None or accuracy is True:
                        acc_mult = 1.0
                    else:
                        acc_mult = max(min(float(accuracy) / 100.0, 1.0), 0.0)
                    recoil_rate = self._move_recoil_rate(move)
                    risk_score = acc_mult
                    if recoil_rate > 0:
                        expected_damage_fraction = min(1.0, raw_score / 220.0)
                        expected_recoil = recoil_rate * expected_damage_fraction
                        if active.current_hp_fraction <= expected_recoil + 0.02:
                            risk_score *= 0.4
                        else:
                            risk_score *= 0.85
                    if risk_score > endgame_ko_risk or (
                        abs(risk_score - endgame_ko_risk) < 1e-6 and raw_score > endgame_ko_score
                    ):
                        endgame_ko_move = move
                        endgame_ko_risk = risk_score
                        endgame_ko_score = raw_score

                if endgame_ko_move is not None:
                    return self._commit_order(battle, self.create_order(
                        endgame_ko_move,
                        dynamax=self._should_dynamax(battle, n_remaining),
                        terastallize=self._should_terastallize(battle, endgame_ko_move),
                    ))

            # Priority KO checks when speed control matters
            priority_ko_move = None
            priority_ko_score = -1.0
            for move in battle.available_moves:
                move_entry = self._get_move_entry(move)
                move_priority = move_entry.get("priority", 0)
                if move.id not in self.PRIORITY_MOVES and move_priority <= 0:
                    continue
                raw_score = self._calculate_move_score(
                    move, active, opponent, battle, apply_recoil=False
                )
                if raw_score < 220 * opponent.current_hp_fraction:
                    continue
                recoil_rate = self._move_recoil_rate(move)
                if recoil_rate > 0:
                    expected_damage_fraction = min(1.0, raw_score / 220.0)
                    expected_recoil = recoil_rate * expected_damage_fraction
                    if active.current_hp_fraction <= expected_recoil + 0.02 and n_opp_remaining > 1:
                        continue
                if raw_score > priority_ko_score:
                    priority_ko_score = raw_score
                    priority_ko_move = move

            if priority_ko_move is not None:
                if opp_speed > my_speed or opp_priority_threat or active.current_hp_fraction < 0.4 or is_endgame:
                    return self._commit_order(battle, self.create_order(
                        priority_ko_move,
                        dynamax=self._should_dynamax(battle, n_remaining),
                        terastallize=self._should_terastallize(battle, priority_ko_move),
                    ))

            # Avoid overextending when likely to get KO'd and we can pivot
            if battle.available_switches and n_remaining > 1 and not switch_churn:
                if reply_score >= 240 * active.current_hp_fraction and best_damage_score < 200 * opponent.current_hp_fraction:
                    best_switch = max(
                        battle.available_switches,
                        key=lambda s: self._score_switch(s, opponent, battle)
                    )
                    current_score = (
                        self._estimate_matchup(active, opponent)
                        + (active.current_hp_fraction or 0.5) * 0.2
                        - (reply_score / 400.0)
                    )
                    if self._score_switch(best_switch, opponent, battle) > current_score + 0.2:
                        return self._commit_order(battle, self.create_order(best_switch))

            best_status_move = None
            best_status_score = 0.0
            for move in battle.available_moves:
                move_entry = self._get_move_entry(move)
                status_kind = (
                    move.category == MoveCategory.STATUS or
                    self._status_from_move_entry(move_entry) is not None
                )
                if not status_kind:
                    continue
                status_score = self._should_use_status_move(move, active, opponent, battle)
                if status_score <= 0:
                    continue
                status_type = self.STATUS_MOVES.get(move.id) or self._status_from_move_entry(move_entry)
                if opp_set_up and status_type not in {"burn", "para", "sleep", "taunt", "encore"}:
                    status_score *= 0.7
                if opp_stallish and status_type in {"poison", "seed", "taunt", "encore", "yawn"}:
                    status_score *= 1.2
                if play_safe:
                    status_score *= 0.8
                if predicted_switch is not None:
                    status_score *= 1.3
                if matchup >= -0.3:
                    status_score *= 1.2
                if opponent.boosts and max(opponent.boosts.values()) >= 2:
                    status_score *= 1.2
                if predicted_kind == "damage":
                    status_score *= 0.85
                if status_score > best_status_score:
                    best_status_score = status_score
                    best_status_move = move

            # Priority 0.8: proactive status when safe and opponent is unstatused
            if opponent.status is None and active.current_hp_fraction >= 0.5 and best_status_move:
                can_ko = best_damage_score >= 260 * opponent.current_hp_fraction
                safe = reply_score < 240 * active.current_hp_fraction or active.current_hp_fraction >= 0.75
                strong_status = best_status_score >= 110 or predicted_switch is not None
                if behind and reply_score >= 200 * active.current_hp_fraction:
                    safe = False
                if play_safe and best_damage_score >= 180 * opponent.current_hp_fraction:
                    pass
                elif safe and (not can_ko or strong_status):
                    return self._commit_order(battle, self.create_order(best_status_move))

            # Priority 4: Best overall move (damage + status) with light 2-ply risk check
            candidates = []
            status_available = False
            for move in battle.available_moves:
                move_entry = self._get_move_entry(move)
                status_kind = (
                    move.category == MoveCategory.STATUS or
                    self._status_from_move_entry(move_entry) is not None
                )
                if status_kind:
                    base_score = self._should_use_status_move(move, active, opponent, battle)
                    status_available = True
                    if base_score > 0:
                        status_type = self.STATUS_MOVES.get(move.id) or self._status_from_move_entry(move_entry)
                        if opp_set_up and status_type not in {"burn", "para", "sleep", "taunt", "encore"}:
                            base_score *= 0.75
                        if opp_stallish and status_type in {"poison", "seed", "taunt", "encore", "yawn"}:
                            base_score *= 1.2
                        if predicted_switch is not None:
                            base_score *= 1.2
                        if opponent.status is None and active.current_hp_fraction >= 0.6 and matchup >= -0.3:
                            base_score *= 1.25
                        if best_status_move is move:
                            base_score *= 1.15
                        if predicted_kind == "damage":
                            base_score *= 0.85
                        if behind and reply_score >= 220 * active.current_hp_fraction and best_damage_score < 220 * opponent.current_hp_fraction:
                            base_score *= 0.4
                        if best_damage_score >= 200 * opponent.current_hp_fraction:
                            base_score *= 0.6
                    if base_score > 0:
                        base_score = self._expected_move_value(
                            move, base_score, active, opponent, battle, action_dist, predicted_switch
                        )
                else:
                    raw_damage = self._calculate_move_score(move, active, opponent, battle)
                    if action_dist:
                        expected_net = self._expected_net_value(
                            move, raw_damage, active, opponent, battle, action_dist, predicted_switch
                        )
                        base_score = raw_damage + (0.35 * max(0.0, expected_net)) + 0.1 * raw_damage
                    else:
                        base_score = self._score_move_with_prediction(
                            move, active, opponent, predicted_switch, battle, switch_weight=switch_weight
                        )
                    if predicted_kind in {"setup", "recovery", "status", "hazard"}:
                        base_score *= 1.1
                    if opp_set_up:
                        base_score *= 1.15
                    if play_safe and self._move_recoil_rate(move) > 0:
                        raw_score = self._calculate_move_score(
                            move, active, opponent, battle, apply_recoil=False
                        )
                        if raw_score < 220 * opponent.current_hp_fraction:
                            base_score *= 0.7
                if base_score > 0:
                    candidates.append((move, base_score))

            if status_available:
                self.STATUS_AVAILABLE_TURNS += 1

            if candidates:
                candidates.sort(key=lambda x: x[1], reverse=True)
                top_candidates = candidates[:3]
                # reply_score and best_damage_score computed above

                best_move = None
                best_net = -1e9
                best_status = 0.0
                for move, base_score in top_candidates:
                    if move.category == MoveCategory.STATUS:
                        reply_scale = 0.75
                        if base_score > best_status:
                            best_status = base_score
                    else:
                        if base_score >= 220 * opponent.current_hp_fraction:
                            reply_scale = 0.35
                        elif base_score >= 150 * opponent.current_hp_fraction:
                            reply_scale = 0.6
                        else:
                            reply_scale = 0.9

                    damage_score = 0.0
                    if move.category != MoveCategory.STATUS:
                        damage_score = self._calculate_move_score(
                            move, active, opponent, battle, apply_recoil=False
                        )
                    expected_reply = (
                        self._expected_reply_for_move(
                            move, damage_score, active, opponent, battle, action_dist
                        )
                        if action_dist
                        else reply_score
                    )
                    reply_component = reply_score if play_safe else max(expected_reply, reply_score * 0.6)

                    if move.category != MoveCategory.STATUS and action_dist:
                        net_score = base_score
                    else:
                        net_score = base_score - (reply_component * 0.35 * reply_scale)

                    # Prefer likely KO lines
                    if best_damage_score >= 220 * opponent.current_hp_fraction:
                        if move.category != MoveCategory.STATUS:
                            net_score *= 1.15

                    # Avoid bad trades if opponent likely KOs us
                    if reply_score >= 220 * active.current_hp_fraction and base_score < 180:
                        net_score *= 0.7

                    if active.current_hp_fraction < 0.4:
                        net_score -= reply_score * 0.05

                    if net_score > best_net:
                        best_net = net_score
                        best_move = move

                if best_move:
                    if status_available:
                        self.STATUS_AVAILABLE_TURNS += 1
                    if (self.DEBUG_STATUS and best_move.category != MoveCategory.STATUS and
                        any(m.category == MoveCategory.STATUS for m, _ in top_candidates)):
                        if best_status > 0:
                            self.STATUS_SKIP_COUNTS["skipped"] += 1
                            self.STATUS_SKIP_COUNTS["available"] += 1
                            print(
                                f"[status-skip] {battle.battle_tag} turn={battle.turn} "
                                f"best_status={best_status:.1f} best_net={best_net:.1f}"
                            )
                    elif self.DEBUG_STATUS and any(m.category == MoveCategory.STATUS for m, _ in top_candidates):
                        self.STATUS_SKIP_COUNTS["available"] += 1

                    order = self.create_order(
                        best_move,
                        dynamax=self._should_dynamax(battle, n_remaining),
                        terastallize=self._should_terastallize(battle, best_move)
                    )
                    if hasattr(order, "order") and hasattr(order.order, "id"):
                        if order.order.id in self.FUTURE_SIGHT_MOVES:
                            mem = self._get_battle_memory(battle)
                            mem["last_future_sight_turn"] = battle.turn
                    return self._commit_order(battle, order)

            # Fallback to any move if no damaging or status moves score
            if battle.available_moves:
                return self._commit_order(battle, self.create_order(battle.available_moves[0]))

        # Switch to best matchup
        if battle.available_switches:
            if switch_churn and battle.available_moves:
                emergency_order = self._choose_emergency_non_switch_order(
                    battle, active, opponent, n_remaining
                )
                if emergency_order is not None:
                    mem["switch_churn_breaks"] = int(mem.get("switch_churn_breaks", 0) or 0) + 1
                    return self._commit_order(battle, emergency_order)
            best_switch = max(
                battle.available_switches,
                key=lambda s: self._score_switch(s, opponent, battle)
            )
            return self._commit_order(battle, self.create_order(best_switch))

        return self.choose_random_move(battle)
