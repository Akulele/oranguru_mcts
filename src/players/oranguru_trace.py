#!/usr/bin/env python3
"""
Search-trace helpers for OranguruEnginePlayer.

This module isolates trace-building and trace-flushing logic from the main
engine so the online decision code stays easier to navigate.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple

from poke_env.battle import Battle, MoveCategory

from src.utils.damage_calc import normalize_name

FP_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "foul-play"
if str(FP_ROOT) not in sys.path:
    sys.path.insert(0, str(FP_ROOT))

from fp.battle import Battle as FPBattle, Battler, Pokemon as FPPokemon, LastUsedMove  # noqa: E402


def search_trace_token_hash(token: str) -> float:
    if not token:
        return 0.0
    digest = hashlib.sha1(token.encode("utf-8")).hexdigest()
    value = int(digest[:8], 16) % 997
    return value / 996.0


def search_trace_species_hash(self, species: str) -> float:
    return search_trace_token_hash(species)


def init_search_trace_builder(self):
    if self._search_trace_builder_failed:
        return None
    if self._search_trace_builder is not None:
        return self._search_trace_builder
    try:
        from src.utils.features import EnhancedFeatureBuilder
    except Exception:
        self._search_trace_builder_failed = True
        return None
    self._search_trace_builder = EnhancedFeatureBuilder(enable_prediction_features=False)
    return self._search_trace_builder


def build_search_trace_action_features(
    self,
    battle: Battle,
    mask: List[bool],
) -> List[List[float]]:
    rows: List[List[float]] = []
    my_hazard_load = 0.0
    try:
        my_hazard_load = sum(
            float(v)
            for v in getattr(battle, "side_conditions", {}).values()
            if isinstance(v, (int, float))
        )
    except Exception:
        my_hazard_load = 0.0
    my_hazard_load = min(my_hazard_load / 7.0, 1.0)

    opp = battle.opponent_active_pokemon
    opp_hp_frac = float(getattr(opp, "current_hp_fraction", 1.0) or 1.0)
    active_slots = list(battle.available_moves or [])
    bench = list(battle.available_switches or [])

    for idx in range(13):
        row = [0.0] * 16
        row[0] = 1.0 if idx < 4 else 0.0
        row[1] = 1.0 if 4 <= idx < 9 else 0.0
        row[2] = 1.0 if idx >= 9 else 0.0
        row[3] = float(idx) / 12.0
        row[4] = 1.0 if mask[idx] else 0.0
        row[5] = opp_hp_frac
        row[6] = my_hazard_load

        if idx < 4:
            row[7] = 1.0 if idx < len(active_slots) else 0.0
            row[8] = 1.0 if idx >= len(active_slots) else 0.0
        elif 4 <= idx < 9:
            bench_idx = idx - 4
            if bench_idx < len(bench):
                sw = bench[bench_idx]
                row[9] = float(getattr(sw, "current_hp_fraction", 1.0) or 1.0)
                row[10] = self._search_trace_species_hash(normalize_name(sw.species))
                row[11] = 1.0
            else:
                row[11] = 0.0
        else:
            move_idx = idx - 9
            row[12] = 1.0 if move_idx < len(active_slots) else 0.0
            row[13] = 1.0 if bool(getattr(battle, "can_tera", False)) else 0.0

        row[14] = 1.0 if (idx < 4 or idx >= 9) else 0.0
        row[15] = 1.0 if (4 <= idx < 9) else 0.0
        rows.append(row)
    return rows


def search_trace_phase(self, battle: Battle) -> str:
    try:
        my_remaining = sum(
            1 for mon in (battle.team or {}).values() if not getattr(mon, "fainted", False)
        )
    except Exception:
        my_remaining = 6
    try:
        opp_remaining = sum(
            1 for mon in (battle.opponent_team or {}).values() if not getattr(mon, "fainted", False)
        )
    except Exception:
        opp_remaining = 6
    turn = int(getattr(battle, "turn", 0) or 0)
    if my_remaining <= 2 or opp_remaining <= 2:
        return "end"
    if turn <= 6:
        return "opening"
    return "mid"


def search_trace_status_code(self, status: object) -> float:
    status_name = normalize_name(str(status or ""))
    if not status_name:
        return 0.0
    mapping = {
        "brn": 1.0 / 6.0,
        "burn": 1.0 / 6.0,
        "par": 2.0 / 6.0,
        "paralysis": 2.0 / 6.0,
        "psn": 3.0 / 6.0,
        "poison": 3.0 / 6.0,
        "tox": 4.0 / 6.0,
        "toxic": 4.0 / 6.0,
        "slp": 5.0 / 6.0,
        "sleep": 5.0 / 6.0,
        "frz": 1.0,
        "freeze": 1.0,
    }
    return mapping.get(status_name, 0.0)


def build_search_trace_action_labels(
    self,
    battle: Battle,
    mask: List[bool],
) -> List[str]:
    labels = [""] * len(mask)
    for i in range(min(4, len(battle.available_moves or []))):
        if i >= len(labels):
            break
        move = battle.available_moves[i]
        move_id = normalize_name(getattr(move, "id", "") or "")
        if not move_id:
            continue
        if mask[i]:
            labels[i] = move_id
        tera_idx = 9 + i
        if tera_idx < len(labels) and mask[tera_idx]:
            labels[tera_idx] = f"{move_id}-tera"
    for i in range(min(5, len(battle.available_switches or []))):
        idx = 4 + i
        if idx >= len(labels):
            break
        sw = battle.available_switches[i]
        sw_id = normalize_name(getattr(sw, "species", "") or "")
        if sw_id and mask[idx]:
            labels[idx] = f"switch {sw_id}"
    return labels


def serialize_fp_last_used_move(move: LastUsedMove) -> dict:
    return {
        "pokemon_name": str(getattr(move, "pokemon_name", "") or ""),
        "move": str(getattr(move, "move", "") or ""),
        "turn": int(getattr(move, "turn", 0) or 0),
    }


def serialize_fp_move(move) -> dict:
    return {
        "name": str(getattr(move, "name", "") or ""),
        "disabled": bool(getattr(move, "disabled", False)),
        "can_z": bool(getattr(move, "can_z", False)),
        "current_pp": int(getattr(move, "current_pp", 0) or 0),
        "max_pp": int(getattr(move, "max_pp", 0) or 0),
    }


def serialize_fp_pokemon(mon: Optional[FPPokemon]) -> Optional[dict]:
    if mon is None:
        return None
    speed_range = getattr(mon, "speed_range", None)
    return {
        "name": str(getattr(mon, "name", "") or ""),
        "nickname": getattr(mon, "nickname", None),
        "base_name": str(getattr(mon, "base_name", "") or ""),
        "level": int(getattr(mon, "level", 0) or 0),
        "nature": str(getattr(mon, "nature", "serious") or "serious"),
        "evs": [int(v) for v in tuple(getattr(mon, "evs", ()) or ())],
        "base_stats": dict(getattr(mon, "base_stats", {}) or {}),
        "stats": dict(getattr(mon, "stats", {}) or {}),
        "max_hp": int(getattr(mon, "max_hp", 0) or 0),
        "hp": int(getattr(mon, "hp", 0) or 0),
        "substitute_hit": bool(getattr(mon, "substitute_hit", False)),
        "ability": getattr(mon, "ability", None),
        "types": [str(t) for t in tuple(getattr(mon, "types", ()) or ())],
        "item": getattr(mon, "item", None),
        "removed_item": getattr(mon, "removed_item", None),
        "unknown_forme": bool(getattr(mon, "unknown_forme", False)),
        "moves_used_since_switch_in": sorted(
            str(v) for v in set(getattr(mon, "moves_used_since_switch_in", set()) or set())
        ),
        "zoroark_disguised_as": getattr(mon, "zoroark_disguised_as", None),
        "hp_at_switch_in": int(getattr(mon, "hp_at_switch_in", 0) or 0),
        "status_at_switch_in": getattr(mon, "status_at_switch_in", None),
        "terastallized": bool(getattr(mon, "terastallized", False)),
        "tera_type": getattr(mon, "tera_type", None),
        "forme_changed": bool(getattr(mon, "forme_changed", False)),
        "original_ability": getattr(mon, "original_ability", None),
        "fainted": bool(getattr(mon, "fainted", False)),
        "reviving": bool(getattr(mon, "reviving", False)),
        "moves": [serialize_fp_move(m) for m in list(getattr(mon, "moves", []) or [])],
        "status": getattr(mon, "status", None),
        "volatile_statuses": [str(v) for v in list(getattr(mon, "volatile_statuses", []) or [])],
        "volatile_status_durations": {
            str(k): int(v) for k, v in dict(getattr(mon, "volatile_status_durations", {}) or {}).items()
        },
        "boosts": {
            str(k): int(v) for k, v in dict(getattr(mon, "boosts", {}) or {}).items()
        },
        "rest_turns": int(getattr(mon, "rest_turns", 0) or 0),
        "sleep_turns": int(getattr(mon, "sleep_turns", 0) or 0),
        "knocked_off": bool(getattr(mon, "knocked_off", False)),
        "can_mega_evo": bool(getattr(mon, "can_mega_evo", False)),
        "can_ultra_burst": bool(getattr(mon, "can_ultra_burst", False)),
        "can_dynamax": bool(getattr(mon, "can_dynamax", False)),
        "can_terastallize": bool(getattr(mon, "can_terastallize", False)),
        "is_mega": bool(getattr(mon, "is_mega", False)),
        "mega_name": getattr(mon, "mega_name", None),
        "can_have_choice_item": bool(getattr(mon, "can_have_choice_item", True)),
        "item_inferred": bool(getattr(mon, "item_inferred", False)),
        "gen_3_consecutive_sleep_talks": int(
            getattr(mon, "gen_3_consecutive_sleep_talks", 0) or 0
        ),
        "impossible_items": sorted(
            str(v) for v in set(getattr(mon, "impossible_items", set()) or set())
        ),
        "impossible_abilities": sorted(
            str(v) for v in set(getattr(mon, "impossible_abilities", set()) or set())
        ),
        "speed_range": {
            "min": int(getattr(speed_range, "min", 0) or 0),
            "max": float(getattr(speed_range, "max", float("inf")) or float("inf")),
        },
        "hidden_power_possibilities": sorted(
            str(v)
            for v in set(getattr(mon, "hidden_power_possibilities", set()) or set())
        ),
    }


def serialize_fp_battler(self, battler: Battler) -> dict:
    return {
        "active": self._serialize_fp_pokemon(getattr(battler, "active", None)),
        "reserve": [
            self._serialize_fp_pokemon(mon)
            for mon in list(getattr(battler, "reserve", []) or [])
        ],
        "side_conditions": {
            str(k): int(v) for k, v in dict(getattr(battler, "side_conditions", {}) or {}).items()
        },
        "name": getattr(battler, "name", None),
        "trapped": bool(getattr(battler, "trapped", False)),
        "baton_passing": bool(getattr(battler, "baton_passing", False)),
        "shed_tailing": bool(getattr(battler, "shed_tailing", False)),
        "wish": list(tuple(getattr(battler, "wish", (0, 0)) or (0, 0))),
        "future_sight": list(tuple(getattr(battler, "future_sight", (0, "")) or (0, ""))),
        "account_name": getattr(battler, "account_name", None),
        "team_dict": getattr(battler, "team_dict", None),
        "last_selected_move": self._serialize_fp_last_used_move(
            getattr(battler, "last_selected_move", LastUsedMove("", "", 0))
        ),
        "last_used_move": self._serialize_fp_last_used_move(
            getattr(battler, "last_used_move", LastUsedMove("", "", 0))
        ),
    }


def serialize_fp_battle(self, battle: FPBattle) -> dict:
    return {
        "battle_tag": str(getattr(battle, "battle_tag", "") or ""),
        "turn": int(getattr(battle, "turn", 0) or 0),
        "weather": getattr(battle, "weather", None),
        "weather_turns_remaining": int(getattr(battle, "weather_turns_remaining", -1) or -1),
        "weather_source": getattr(battle, "weather_source", ""),
        "field": getattr(battle, "field", None),
        "field_turns_remaining": int(getattr(battle, "field_turns_remaining", 0) or 0),
        "trick_room": bool(getattr(battle, "trick_room", False)),
        "trick_room_turns_remaining": int(
            getattr(battle, "trick_room_turns_remaining", 0) or 0
        ),
        "gravity": bool(getattr(battle, "gravity", False)),
        "team_preview": bool(getattr(battle, "team_preview", False)),
        "started": bool(getattr(battle, "started", False)),
        "rqid": getattr(battle, "rqid", None),
        "force_switch": bool(getattr(battle, "force_switch", False)),
        "wait": bool(getattr(battle, "wait", False)),
        "battle_type": getattr(battle, "battle_type", None),
        "pokemon_format": getattr(battle, "pokemon_format", None),
        "generation": getattr(battle, "generation", None),
        "time_remaining": getattr(battle, "time_remaining", None),
        "user": self._serialize_fp_battler(getattr(battle, "user", Battler())),
        "opponent": self._serialize_fp_battler(getattr(battle, "opponent", Battler())),
    }


def search_trace_choice_kind(self, battle: Battle, choice: str) -> str:
    if not choice:
        return "unknown"
    if choice.startswith("switch "):
        return "switch"
    move_id = normalize_name(choice.replace("-tera", ""))
    for move in battle.available_moves or []:
        if normalize_name(getattr(move, "id", "")) != move_id:
            continue
        if move.id in self.PROTECT_MOVES:
            return "protect"
        if self._is_recovery_move(move):
            return "recovery"
        try:
            if move.category == MoveCategory.STATUS:
                boosts = getattr(move, "boosts", None) or {}
                if boosts and move.target and "self" in str(move.target).lower():
                    return "setup"
                return "status"
        except Exception:
            return "unknown"
        if choice.endswith("-tera"):
            return "tera_attack"
        return "attack"
    return "unknown"


def append_search_trace_example(
    self,
    battle: Battle,
    ordered: List[Tuple[str, float]],
    chosen_choice: str,
    confidence: float,
    threshold: float,
    path: str,
    world_candidates: Optional[List[dict]] = None,
) -> None:
    if not self.SEARCH_TRACE_ENABLED:
        return
    if self.SEARCH_TRACE_SKIP_FALLBACK and path.startswith("fallback"):
        return
    builder = self._init_search_trace_builder()
    if builder is None:
        return
    mask, move_map, switch_map = self._build_rl_action_mask_and_maps(battle)
    if not any(mask):
        return
    board_features = builder.build(battle)
    action_features = self._build_search_trace_action_features(battle, mask)
    action_labels = self._build_search_trace_action_labels(battle, mask)
    visit_counts = [0.0] * len(mask)
    total_policy = 0.0
    for choice, weight in ordered:
        idx = self._choice_to_rl_action_idx(choice, mask, move_map, switch_map)
        if idx is None or idx >= len(visit_counts):
            continue
        w = max(0.0, float(weight))
        visit_counts[idx] += w
        total_policy += w
    if total_policy < max(0.0, self.SEARCH_TRACE_MIN_TOTAL):
        return
    chosen_idx = self._choice_to_rl_action_idx(chosen_choice, mask, move_map, switch_map)
    if chosen_idx is None:
        return
    active = battle.active_pokemon
    opponent = battle.opponent_active_pokemon
    phase = self._search_trace_phase(battle)
    matchup_score = 0.0
    best_reply_score = 0.0
    hazard_load = 0.0
    if active is not None and opponent is not None:
        try:
            matchup_score = float(self._estimate_matchup(active, opponent))
        except Exception:
            matchup_score = 0.0
        try:
            best_reply_score = float(self._estimate_best_reply_score(opponent, active, battle))
        except Exception:
            best_reply_score = 0.0
    try:
        hazard_load = float(self._side_hazard_pressure(battle))
    except Exception:
        hazard_load = 0.0
    top_actions = []
    for choice, weight in ordered[:5]:
        choice_idx = self._choice_to_rl_action_idx(choice, mask, move_map, switch_map)
        try:
            heuristic_score = self._heuristic_action_score(battle, choice)
        except Exception:
            heuristic_score = None
        try:
            risk_penalty = self._adaptive_choice_risk_penalty(battle, choice)
        except Exception:
            risk_penalty = None
        top_actions.append(
            {
                "choice": choice,
                "choice_idx": int(choice_idx) if choice_idx is not None else None,
                "weight": float(weight),
                "score": float(weight),
                "heuristic_score": None if heuristic_score is None else float(heuristic_score),
                "risk_penalty": None if risk_penalty is None else float(risk_penalty),
                "kind": self._search_trace_choice_kind(battle, choice),
            }
        )
    top1_kind = top_actions[0]["kind"] if top_actions else "unknown"
    switch_candidate_count = sum(1 for idx in range(4, 9) if idx < len(mask) and mask[idx])
    tera_candidate_count = sum(1 for idx in range(9, 13) if idx < len(mask) and mask[idx])
    state_value_features = self._build_state_value_features(
        battle,
        phase=phase,
        hazard_load=hazard_load,
        matchup_score=matchup_score,
        best_reply_score=best_reply_score,
    )
    mem = self._get_battle_memory(battle)
    examples = mem.setdefault("search_trace_examples", [])
    finish_blow = mem.get("finish_blow_last") if isinstance(mem, dict) else None
    if not isinstance(finish_blow, dict):
        finish_blow = None
    recovery_window = mem.get("recovery_window_last") if isinstance(mem, dict) else None
    if not isinstance(recovery_window, dict):
        recovery_window = None
    setup_window = mem.get("setup_window_last") if isinstance(mem, dict) else None
    if not isinstance(setup_window, dict):
        setup_window = None
    switch_guard = mem.get("switch_guard_last") if isinstance(mem, dict) else None
    if not isinstance(switch_guard, dict):
        switch_guard = None
    progress_window = mem.get("progress_window_last") if isinstance(mem, dict) else None
    if not isinstance(progress_window, dict):
        progress_window = None
    passive_breaker = mem.get("passive_breaker_last") if isinstance(mem, dict) else None
    if not isinstance(passive_breaker, dict):
        passive_breaker = None
    fp_oracle_battle = None
    if self.SEARCH_TRACE_INCLUDE_FP_ORACLE:
        try:
            fp_oracle_battle = self._serialize_fp_battle(
                self._build_fp_battle(battle, seed=0, fill_opponent_sets=False)
            )
        except Exception:
            fp_oracle_battle = None
    examples.append(
        {
            "battle_id": str(getattr(battle, "battle_tag", "")),
            "turn": int(getattr(battle, "turn", 0) or 0),
            "rating": None,
            "board_features": [float(v) for v in board_features],
            "action_features": action_features,
            "action_labels": action_labels,
            "action_mask": [bool(v) for v in mask],
            "visit_counts": [float(v) for v in visit_counts],
            "value_target": 0.0,
            "weight": 1.0,
            "source": self.SEARCH_TRACE_SOURCE,
            "tag": self.SEARCH_TRACE_TAG,
            "player": "self",
            "chosen_action": int(chosen_idx),
            "chosen_choice": chosen_choice,
            "policy_confidence": float(confidence),
            "policy_threshold": float(threshold),
            "selection_path": path,
            "can_tera": bool(getattr(battle, "can_tera", False)),
            "top_actions": top_actions,
            "top1_kind": top1_kind,
            "passive_top1": top1_kind in {"protect", "recovery", "status", "setup"},
            "switch_candidate_count": int(switch_candidate_count),
            "tera_candidate_count": int(tera_candidate_count),
            "hazard_load": float(hazard_load),
            "matchup_score": float(matchup_score),
            "best_reply_score": float(best_reply_score),
            "phase": phase,
            "state_value_features": state_value_features,
            "world_candidates": list(world_candidates or []),
            "finish_blow": finish_blow,
            "recovery_window": recovery_window,
            "setup_window": setup_window,
            "switch_guard": switch_guard,
            "progress_window": progress_window,
            "passive_breaker": passive_breaker,
            "fp_oracle_battle": fp_oracle_battle,
        }
    )


def flush_finished_search_traces(self) -> None:
    if not self.SEARCH_TRACE_ENABLED:
        return
    battles = getattr(self, "battles", {}) or {}
    if not battles:
        return
    battle_memory = getattr(self, "_battle_memory", {}) or {}
    out_path = Path(self.SEARCH_TRACE_OUT)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for tag, battle in battles.items():
        if tag in self._search_trace_finished_battle_tags:
            continue
        if not getattr(battle, "finished", False):
            continue
        self._search_trace_finished_battle_tags.add(tag)
        mem = battle_memory.get(tag, {}) if isinstance(battle_memory, dict) else {}
        if not isinstance(mem, dict):
            continue
        examples = list(mem.get("search_trace_examples", []) or [])
        if not examples:
            continue
        value = 0.0
        if bool(getattr(battle, "won", False)):
            value = 1.0
        elif bool(getattr(battle, "lost", False)):
            value = -1.0
        with out_path.open("a", encoding="utf-8") as handle:
            for ex in examples:
                ex["value_target"] = value
                handle.write(json.dumps(ex, separators=(",", ":")) + "\n")
