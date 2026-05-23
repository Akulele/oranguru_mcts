#!/usr/bin/env python3
"""
Tactical safety helpers for OranguruEnginePlayer.

This module keeps the passive-loop and move-safety logic separate from the
main engine/search code so behavior tuning can happen in a smaller surface
area.
"""

from __future__ import annotations

from poke_env.battle import Battle, MoveCategory, Pokemon, SideCondition

from src.utils.damage_calc import get_type_effectiveness, normalize_name

_PIVOT_MOVES = {
    "batonpass",
    "chillyreception",
    "flipturn",
    "partingshot",
    "shedtail",
    "teleport",
    "uturn",
    "voltswitch",
}


def switch_faints_to_entry_hazards(self, battle: Battle, mon: Pokemon) -> bool:
    if battle is None or mon is None:
        return False
    if self._has_heavy_duty_boots(mon):
        return False
    hp_frac = mon.current_hp_fraction if mon.current_hp_fraction is not None else 1.0
    if hp_frac <= 0:
        return True
    side_conditions = getattr(battle, "side_conditions", None) or {}
    if not side_conditions:
        return False

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

    return hazard_fraction >= max(0.0, hp_frac - 1e-6)


def resolve_passive_progress(self, battle: Battle) -> None:
    mem = self._get_battle_memory(battle)
    pending = mem.get("pending_passive_action")
    if not isinstance(pending, dict):
        return
    pending_turn = int(pending.get("turn", -1) or -1)
    current_turn = int(getattr(battle, "turn", 0) or 0)
    if current_turn <= pending_turn:
        return

    opponent = battle.opponent_active_pokemon
    active = battle.active_pokemon
    prev_opp_hp = pending.get("opp_hp")
    cur_opp_hp = getattr(opponent, "current_hp_fraction", None)
    opp_hp_drop = 0.0
    if isinstance(prev_opp_hp, (int, float)) and isinstance(cur_opp_hp, (int, float)):
        opp_hp_drop = max(0.0, float(prev_opp_hp) - float(cur_opp_hp))

    prev_self_hp = pending.get("self_hp")
    cur_self_hp = getattr(active, "current_hp_fraction", None)
    self_hp_gain = 0.0
    if isinstance(prev_self_hp, (int, float)) and isinstance(cur_self_hp, (int, float)):
        self_hp_gain = max(0.0, float(cur_self_hp) - float(prev_self_hp))

    prev_markers = pending.get("opp_markers", {}) or {}
    cur_markers = self._opponent_progress_markers(battle, opponent)
    progress = opp_hp_drop >= 0.06
    progress = progress or cur_markers.get("status") != prev_markers.get("status")
    for key in ("rocks", "web", "spikes", "tspikes"):
        if int(cur_markers.get(key, 0) or 0) > int(prev_markers.get(key, 0) or 0):
            progress = True
            break

    kind = pending.get("kind", "")
    if kind == "recovery" and self_hp_gain >= 0.20:
        progress = True

    if progress:
        mem["passive_no_progress_streak"] = 0
    else:
        mem["passive_no_progress_streak"] = int(mem.get("passive_no_progress_streak", 0) or 0) + 1
        if self.DECISION_DIAG_ENABLED:
            self._mcts_stats["diag_passive_no_progress_turns"] += 1
            mem["diag_passive_no_progress_turns"] = int(
                mem.get("diag_passive_no_progress_turns", 0) or 0
            ) + 1
    mem.pop("pending_passive_action", None)


def passive_choice_kind(self, move) -> str:
    if move is None:
        return ""
    if not hasattr(move, "id"):
        return ""
    move_id = normalize_name(getattr(move, "id", "") or "")
    if move_id in self.PROTECT_MOVES:
        return "protect"
    if self._is_recovery_move(move):
        return "recovery"
    try:
        if move.category == MoveCategory.STATUS:
            return "status"
    except Exception:
        return ""
    return ""


def _best_non_pivot_damage_move(self, battle: Battle, active: Pokemon, opponent: Pokemon, blocked_move_id: str):
    best_move = None
    best_score = 0.0
    for move in getattr(battle, "available_moves", None) or []:
        move_id = normalize_name(getattr(move, "id", "") or "")
        if not move_id or move_id == blocked_move_id or move_id in _PIVOT_MOVES:
            continue
        try:
            if move.category == MoveCategory.STATUS:
                continue
        except Exception:
            continue
        try:
            score = float(self._calculate_move_score(move, active, opponent, battle, apply_recoil=False) or 0.0)
        except TypeError:
            try:
                score = float(self._calculate_move_score(move, active, opponent, battle) or 0.0)
            except Exception:
                score = 0.0
        except Exception:
            score = 0.0
        if move_id == "knockoff" and getattr(opponent, "item", None):
            score += 12.0
        if best_move is None or score > best_score:
            best_move = move
            best_score = score
    return best_move, best_score


def progress_need_score(
    self,
    battle: Battle,
    active: Pokemon,
    opponent: Pokemon,
    best_damage_score: float,
) -> int:
    score = 0
    if self._estimate_matchup(active, opponent) <= self.PROGRESS_NEED_MATCHUP:
        score += 1
    if self._side_hazard_pressure(battle) > 0:
        score += 1
    if self._estimate_best_reply_score(opponent, active, battle) >= self.PROGRESS_NEED_REPLY:
        score += 1
    if best_damage_score >= self.PROGRESS_NEED_DAMAGE and (opponent.current_hp_fraction or 0.0) >= 0.25:
        score += 1
    return score


def status_choice_is_obviously_bad(
    self,
    battle: Battle,
    move,
    active: Pokemon,
    opponent: Pokemon,
) -> bool:
    move_entry = self._get_move_entry(move)
    status_type = self.STATUS_MOVES.get(move.id)
    if status_type is None:
        status_type = self._status_from_move_entry(move_entry)
    if status_type is None:
        return False

    if self._ability_blocks_move(move, opponent):
        return True
    if self._target_blocks_status_move(move, opponent):
        return True
    if opponent.status is not None and status_type in {"poison", "burn", "para", "sleep", "yawn"}:
        return True
    if self._status_pivot_absorber_blocks(battle, move, status_type, opponent):
        return True
    if status_type == "sleep" and self._sleep_clause_blocked(battle):
        return True
    if status_type == "poison" and (
        self._opponent_has_type(opponent, "steel") or self._opponent_has_type(opponent, "poison")
    ):
        return True
    if status_type == "burn" and self._opponent_has_type(opponent, "fire"):
        return True
    if status_type == "para":
        if move.id == "thunderwave" and (
            self._opponent_has_type(opponent, "ground")
            or self._opponent_has_type(opponent, "electric")
        ):
            return True
    if status_type == "sap" or move.id == "strengthsap":
        opp_atk = self._stat_estimation(opponent, "atk")
        opp_spa = self._stat_estimation(opponent, "spa")
        hp_frac = active.current_hp_fraction or 0.0
        if opp_atk <= opp_spa and hp_frac > 0.45:
            return True

    mem = self._get_battle_memory(battle)
    if (
        int(mem.get("status_stall_streak", 0) or 0) >= self.STATUS_STALL_MAX
        and mem.get("last_move_id") == move.id
    ):
        return True
    return False


def _alive_count(team: object) -> int:
    if not isinstance(team, dict):
        return 0
    return sum(1 for mon in team.values() if mon is not None and not bool(getattr(mon, "fainted", False)))


def last_opponent_move_id(self, battle: Battle, opponent: Pokemon) -> str:
    if battle is None or opponent is None:
        return ""
    mem = self._get_battle_memory(battle)
    species = normalize_name(getattr(opponent, "species", "") or "")
    flags = (mem.get("opponent_item_flags", {}) or {}).get(species, {}) or {}
    last_turn = int(flags.get("last_move_turn", -99) or -99)
    if last_turn >= int(getattr(battle, "turn", 0) or 0) - 1:
        return normalize_name(flags.get("last_move_id", "") or "")
    return ""


def move_id_is_damaging(self, move_id: str) -> bool:
    if not move_id:
        return False
    entry = self._get_move_entry_by_id(move_id)
    return str(entry.get("category", "")).lower() not in {"", "status"}


def move_has_recharge(self, move) -> bool:
    if move is None:
        return False
    entry = self._get_move_entry(move)
    flags = entry.get("flags", {}) if isinstance(entry, dict) else {}
    self_data = entry.get("self", {}) if isinstance(entry, dict) else {}
    return bool(
        (isinstance(flags, dict) and flags.get("recharge"))
        or (isinstance(self_data, dict) and self_data.get("volatileStatus") == "mustrecharge")
    )


def best_non_recharge_damage_move(self, battle: Battle, active: Pokemon, opponent: Pokemon):
    best_move = None
    best_score = -1.0
    for move in battle.available_moves or []:
        try:
            if move.category == MoveCategory.STATUS:
                continue
        except Exception:
            pass
        if self._move_has_recharge(move):
            continue
        try:
            score = float(self._calculate_move_score(move, active, opponent, battle, apply_recoil=False) or 0.0)
        except TypeError:
            try:
                score = float(self._calculate_move_score(move, active, opponent, battle) or 0.0)
            except Exception:
                score = 0.0
        except Exception:
            score = 0.0
        if score > best_score:
            best_move = move
            best_score = score
    return best_move, best_score


def boosted_offense_pressure(mon: Pokemon) -> float:
    boosts = getattr(mon, "boosts", {}) or {}
    if not isinstance(boosts, dict):
        return 0.0
    vals = []
    for keys in (("attack", "atk"), ("special-attack", "spa"), ("speed", "spe")):
        cur = 0.0
        for key in keys:
            try:
                cur = max(cur, float(boosts.get(key, 0) or 0.0))
            except Exception:
                pass
        vals.append(cur)
    return vals[0] + vals[1] + max(0.0, vals[2] - 1.0)


def apply_tactical_safety(self, battle: Battle, choice: str, active: Pokemon, opponent: Pokemon) -> str:
    if not self.MOVE_SAFETY_GUARD:
        return choice

    best_damage_move, best_damage_score = self._best_damaging_move(battle, active, opponent)
    reply_score = self._estimate_best_reply_score(opponent, active, battle)
    opp_hp = opponent.current_hp_fraction or 0.0
    active_hp = active.current_hp_fraction or 0.0
    ko_threshold = self.TACTICAL_KO_THRESHOLD * max(opp_hp, 0.05)
    has_ko_window = best_damage_move is not None and best_damage_score >= ko_threshold

    if choice.startswith("switch "):
        switch_name = normalize_name(choice.split("switch ", 1)[1])
        chosen_switch = None
        for sw in battle.available_switches:
            if normalize_name(sw.species) == switch_name:
                chosen_switch = sw
                break
        if chosen_switch is not None and self._switch_faints_to_entry_hazards(battle, chosen_switch):
            survivable = [sw for sw in battle.available_switches if not self._switch_faints_to_entry_hazards(battle, sw)]
            if survivable:
                best_sw = max(survivable, key=lambda s: self._score_switch(s, opponent, battle))
                return f"switch {normalize_name(best_sw.species)}"
            if best_damage_move is not None and not battle.force_switch:
                return best_damage_move.id
        if (
            chosen_switch is not None
            and not battle.force_switch
            and bool(getattr(self, "LATEGAME_SACK_SWITCH_GUARD", True))
        ):
            my_alive = _alive_count(getattr(battle, "team", {}) or {})
            if my_alive <= 0:
                my_alive = 1 + sum(
                    1 for sw in (battle.available_switches or []) if not bool(getattr(sw, "fainted", False))
                )
            switch_hp = chosen_switch.current_hp_fraction if chosen_switch.current_hp_fraction is not None else 1.0
            active_hp_for_switch = active.current_hp_fraction if active.current_hp_fraction is not None else 1.0
            switch_reply = float(self._estimate_best_reply_score(opponent, chosen_switch, battle) or 0.0)
            switch_fatal = switch_reply >= float(getattr(self, "LATEGAME_SACK_SWITCH_REPLY_KO", 185.0)) * max(
                switch_hp,
                0.05,
            )
            switch_score = float(self._score_switch(chosen_switch, opponent, battle) or 0.0)
            active_matchup = float(self._estimate_matchup(active, opponent) or 0.0)
            min_damage = float(getattr(self, "LATEGAME_SACK_SWITCH_MIN_DAMAGE", 20.0))
            endgame_mons = int(getattr(self, "LATEGAME_SACK_SWITCH_MAX_MY_ALIVE", 3))
            boosted_pressure = boosted_offense_pressure(opponent)
            switch_large_hit = switch_reply >= float(getattr(self, "LATEGAME_BAD_SWITCH_REPLY", 120.0)) * max(
                switch_hp,
                0.05,
            )
            if (
                (switch_fatal or (switch_large_hit and boosted_pressure >= 2.0))
                and my_alive <= endgame_mons
                and best_damage_move is not None
                and best_damage_score >= min_damage
                and switch_hp >= active_hp_for_switch + 0.12
                and switch_score <= active_matchup + float(getattr(self, "LATEGAME_SACK_SWITCH_MIN_GAIN", 0.35))
            ):
                mem = self._get_battle_memory(battle)
                mem["sack_switch_last"] = {
                    "reason": "reject_lategame_sack_switch",
                    "choice": choice,
                    "replacement": best_damage_move.id,
                    "switch_reply": float(switch_reply),
                    "switch_hp": float(switch_hp),
                    "active_hp": float(active_hp_for_switch),
                    "my_alive": int(my_alive),
                    "boosted_pressure": float(boosted_pressure),
                    "best_damage_score": float(best_damage_score),
                }
                return best_damage_move.id
            if (
                bool(getattr(self, "BOOSTED_FOE_DAMAGE_DOMINANCE_GUARD", True))
                and boosted_pressure >= float(getattr(self, "BOOSTED_FOE_DAMAGE_DOMINANCE_MIN_PRESSURE", 4.0))
                and best_damage_move is not None
                and best_damage_score >= float(getattr(self, "BOOSTED_FOE_DAMAGE_DOMINANCE_MIN_DAMAGE", 28.0))
                and switch_reply >= float(getattr(self, "BOOSTED_FOE_DAMAGE_DOMINANCE_MIN_REPLY", 85.0)) * max(
                    switch_hp,
                    0.05,
                )
            ):
                mem = self._get_battle_memory(battle)
                mem["boosted_damage_dominance_last"] = {
                    "reason": "reject_switch_into_boosted_foe",
                    "choice": choice,
                    "replacement": best_damage_move.id,
                    "switch_reply": float(switch_reply),
                    "switch_hp": float(switch_hp),
                    "boosted_pressure": float(boosted_pressure),
                    "best_damage_score": float(best_damage_score),
                }
                return best_damage_move.id
        return choice

    tera_suffix = choice.endswith("-tera")
    move_id = normalize_name(choice.replace("-tera", ""))
    selected_move = None
    for move in battle.available_moves:
        if move.id == move_id:
            selected_move = move
            break
    if selected_move is None:
        return choice

    try:
        is_status = selected_move.category == MoveCategory.STATUS
    except Exception:
        is_status = False
    is_recovery = self._is_recovery_move(selected_move)
    is_setup = bool(getattr(selected_move, "boosts", None) or {}) and (
        selected_move.target and "self" in str(selected_move.target).lower()
    )
    is_protect = selected_move.id in self.PROTECT_MOVES
    is_phaze = move_id in {"roar", "whirlwind", "dragontail", "circlethrow"}
    mem = self._get_battle_memory(battle)
    stall = int(mem.get("status_stall_streak", 0) or 0)
    passive_streak = int(mem.get("passive_no_progress_streak", 0) or 0)
    passive_kind = self._passive_choice_kind(selected_move)
    same_matchup_repeat = bool(
        mem.get("last_action") == "move"
        and int(mem.get("last_action_turn", -99) or -99) == int(getattr(battle, "turn", 0) or 0) - 1
        and normalize_name(str(mem.get("last_move_id", "") or "")) == move_id
        and normalize_name(str(mem.get("last_active_species", "") or "")) == normalize_name(active.species)
        and normalize_name(str(mem.get("last_opponent_species", "") or "")) == normalize_name(opponent.species)
    )
    same_move_repeat_streak = int(mem.get("same_move_repeat_streak", 0) or 0)
    progress_need = self._progress_need_score(
        battle,
        active,
        opponent,
        best_damage_score,
    )

    try:
        selected_damage_score = float(
            self._calculate_move_score(selected_move, active, opponent, battle, apply_recoil=False) or 0.0
        )
    except TypeError:
        try:
            selected_damage_score = float(self._calculate_move_score(selected_move, active, opponent, battle) or 0.0)
        except Exception:
            selected_damage_score = 0.0
    except Exception:
        selected_damage_score = 0.0

    if (
        tera_suffix
        and bool(getattr(self, "NONBENEFICIAL_ATTACK_TERA_GUARD", True))
        and not is_status
        and not is_recovery
        and not is_setup
        and not is_protect
        and not is_phaze
    ):
        tera_id = self._type_id(getattr(active, "tera_type", None))
        move_type_id = self._move_type_id(selected_move)
        has_offensive_gain = bool(tera_id and (move_type_id == tera_id or move_id == "terablast"))
        has_defensive_gain = bool(self._tera_has_immediate_defensive_gain(battle))
        if tera_id and not has_offensive_gain and not has_defensive_gain:
            mem["tera_sanity_last"] = {
                "reason": "strip_nonbeneficial_attack_tera",
                "choice": choice,
                "move": move_id,
                "tera_type": tera_id,
                "move_type": move_type_id,
                "selected_damage_score": float(selected_damage_score),
            }
            return move_id

    if (
        tera_suffix
        and bool(getattr(self, "LOW_HP_DEFENSIVE_TERA_GUARD", True))
        and active_hp <= float(getattr(self, "LOW_HP_DEFENSIVE_TERA_MAX_HP", 0.35))
        and selected_damage_score < ko_threshold
        and best_damage_score < ko_threshold
    ):
        mem["tera_sanity_last"] = {
            "reason": "strip_low_hp_non_ko_defensive_tera",
            "choice": choice,
            "move": move_id,
            "active_hp": float(active_hp),
            "opp_hp": float(opp_hp),
            "selected_damage_score": float(selected_damage_score),
            "best_damage_score": float(best_damage_score),
            "ko_threshold": float(ko_threshold),
        }
        return move_id

    if (
        bool(getattr(self, "RECHARGE_MOVE_GUARD", True))
        and self._move_has_recharge(selected_move)
        and selected_damage_score < ko_threshold
        and not battle.force_switch
    ):
        alt_move, alt_score = self._best_non_recharge_damage_move(battle, active, opponent)
        min_ratio = float(getattr(self, "RECHARGE_MOVE_MIN_ALT_RATIO", 0.35))
        if alt_move is not None and alt_move.id != move_id and alt_score >= max(8.0, selected_damage_score * min_ratio):
            mem["recharge_guard_last"] = {
                "reason": "reject_non_ko_recharge",
                "choice": choice,
                "replacement": alt_move.id,
                "selected_damage_score": float(selected_damage_score),
                "alt_damage_score": float(alt_score),
                "ko_threshold": float(ko_threshold),
            }
            return alt_move.id

    if move_id == "rapidspin" and bool(getattr(self, "RAPID_SPIN_VALUE_GUARD", True)):
        hazard_pressure = float(self._side_hazard_pressure(battle) or 0.0)
        min_reply_factor = float(getattr(self, "RAPID_SPIN_MIN_REPLY_FACTOR", 95.0))
        min_alt_gain = float(getattr(self, "RAPID_SPIN_MIN_ALT_DAMAGE_GAIN", 8.0))
        if (
            hazard_pressure <= float(getattr(self, "RAPID_SPIN_MAX_HAZARD_PRESSURE", 0.01))
            and selected_damage_score < ko_threshold
            and reply_score >= min_reply_factor * max(active_hp, 0.05)
        ):
            alt_move, alt_score = _best_non_pivot_damage_move(self, battle, active, opponent, move_id)
            if alt_move is not None and alt_score >= selected_damage_score + min_alt_gain:
                mem["rapid_spin_guard_last"] = {
                    "reason": "reject_low_value_speed_spin",
                    "choice": choice,
                    "replacement": normalize_name(getattr(alt_move, "id", "") or ""),
                    "hazard_pressure": float(hazard_pressure),
                    "selected_damage_score": float(selected_damage_score),
                    "alt_damage_score": float(alt_score),
                    "reply_score": float(reply_score),
                }
                return normalize_name(getattr(alt_move, "id", "") or "")

    if move_id in _PIVOT_MOVES and bool(getattr(self, "PIVOT_CHURN_GUARD", True)):
        pivot_streak = int(mem.get("pivot_move_streak", 0) or 0)
        min_streak = max(1, int(getattr(self, "PIVOT_CHURN_MIN_STREAK", 2)))
        min_alt_gain = float(getattr(self, "PIVOT_CHURN_MIN_ALT_DAMAGE_GAIN", 6.0))
        if pivot_streak >= min_streak and selected_damage_score < ko_threshold:
            alt_move, alt_score = _best_non_pivot_damage_move(self, battle, active, opponent, move_id)
            if alt_move is not None and alt_score >= selected_damage_score + min_alt_gain:
                mem["pivot_churn_guard_last"] = {
                    "reason": "reject_repeated_low_value_pivot",
                    "choice": choice,
                    "replacement": normalize_name(getattr(alt_move, "id", "") or ""),
                    "pivot_streak": int(pivot_streak),
                    "selected_damage_score": float(selected_damage_score),
                    "alt_damage_score": float(alt_score),
                }
                return normalize_name(getattr(alt_move, "id", "") or "")

    wallbreaker_switch = self._high_defense_counter_switch(battle, active, opponent, best_damage_score)
    if wallbreaker_switch is not None and not battle.force_switch:
        mem["high_defense_counter_last"] = {
            "reason": "take_counter_switch",
            "choice": choice,
            "switch": normalize_name(wallbreaker_switch.species),
            "best_damage_score": float(best_damage_score),
        }
        return f"switch {normalize_name(wallbreaker_switch.species)}"

    if tera_suffix and self._passive_tera_is_bad(battle, selected_move):
        mem["tera_sanity_last"] = {
            "reason": "passive_tera_no_immediate_defensive_gain",
            "choice": choice,
            "move": move_id,
        }
        return move_id

    if (
        tera_suffix
        and bool(getattr(self, "NO_UNNECESSARY_TERA_KO_GUARD", True))
        and not is_status
        and selected_damage_score >= ko_threshold
        and opp_hp <= float(getattr(self, "NO_UNNECESSARY_TERA_KO_MAX_HP", 0.15))
        and reply_score < float(getattr(self, "NO_UNNECESSARY_TERA_REPLY_KO", 185.0)) * max(active_hp, 0.05)
    ):
        mem["tera_sanity_last"] = {
            "reason": "strip_unnecessary_low_hp_ko_tera",
            "choice": choice,
            "move": move_id,
            "opp_hp": float(opp_hp),
            "selected_damage_score": float(selected_damage_score),
            "reply_score": float(reply_score),
        }
        return move_id

    if (
        bool(getattr(self, "BOOSTED_FOE_DAMAGE_DOMINANCE_GUARD", True))
        and boosted_offense_pressure(opponent) >= float(getattr(self, "BOOSTED_FOE_DAMAGE_DOMINANCE_MIN_PRESSURE", 4.0))
        and reply_score >= float(getattr(self, "BOOSTED_FOE_DAMAGE_DOMINANCE_MIN_REPLY", 85.0))
        and best_damage_move is not None
        and best_damage_move.id != move_id
        and best_damage_score >= float(getattr(self, "BOOSTED_FOE_DAMAGE_DOMINANCE_MIN_DAMAGE", 28.0))
        and not battle.force_switch
        and (is_status or is_recovery or is_setup or is_protect or is_phaze)
    ):
        mem["boosted_damage_dominance_last"] = {
            "reason": "reject_passive_into_boosted_foe",
            "choice": choice,
            "replacement": best_damage_move.id,
            "reply_score": float(reply_score),
            "boosted_pressure": float(boosted_offense_pressure(opponent)),
            "best_damage_score": float(best_damage_score),
        }
        return best_damage_move.id

    if is_phaze and self._bad_endgame_phaze(battle, selected_move, opponent):
        mem["phase_endgame_last"] = {
            "reason": "reject_endgame_phaze",
            "choice": choice,
            "opponent": normalize_name(getattr(opponent, "species", "") or ""),
        }
        if best_damage_move is not None and best_damage_move.id != move_id:
            return best_damage_move.id
        if battle.available_switches:
            best_sw = max(battle.available_switches, key=lambda s: self._score_switch(s, opponent, battle))
            return f"switch {normalize_name(best_sw.species)}"

    if (
        bool(getattr(self, "ENCORE_CONVERSION_GUARD", True))
        and self._has_effect(opponent, "encore")
        and battle.available_switches
        and ((active.current_hp_fraction or 0.0) <= 0.55 or getattr(active, "status", None) is not None)
    ):
        best_sw = max(battle.available_switches, key=lambda s: self._score_switch(s, opponent, battle))
        if self._score_switch(best_sw, opponent, battle) > self._estimate_matchup(active, opponent) - 0.15:
            mem["encore_conversion_last"] = {
                "reason": "pivot_on_encored_target",
                "choice": choice,
                "switch": normalize_name(best_sw.species),
            }
            return f"switch {normalize_name(best_sw.species)}"

    if (
        bool(getattr(self, "BAD_ENCORE_ATTACK_LOCK_GUARD", True))
        and move_id == "encore"
        and best_damage_move is not None
        and not battle.force_switch
    ):
        last_opp_move = self._last_opponent_move_id(battle, opponent)
        predicted = self._predict_opponent_move(opponent, active, battle)
        predicted_kind = str((predicted or {}).get("kind", "") or "")
        predicted_damage = float((predicted or {}).get("damage_score", 0.0) or 0.0)
        last_was_damage = self._move_id_is_damaging(last_opp_move)
        fatal_reply = reply_score >= float(getattr(self, "BAD_ENCORE_ATTACK_LOCK_REPLY_KO", 185.0)) * max(
            active_hp,
            0.05,
        )
        min_damage = float(getattr(self, "BAD_ENCORE_ATTACK_LOCK_MIN_DAMAGE", 20.0))
        if (
            fatal_reply
            and best_damage_score >= min_damage
            and (last_was_damage or predicted_kind == "damage" or predicted_damage >= reply_score * 0.70)
        ):
            mem["bad_encore_last"] = {
                "reason": "reject_locking_lethal_attack",
                "choice": choice,
                "replacement": best_damage_move.id,
                "last_opp_move": last_opp_move,
                "reply_score": float(reply_score),
                "active_hp": float(active_hp),
                "best_damage_score": float(best_damage_score),
            }
            return best_damage_move.id

    if (
        self.LOOP_BREAKER_ENABLED
        and stall >= max(1, self.LOOP_BREAKER_STALL_STREAK)
        and (is_status or is_recovery or is_protect)
        and best_damage_move is not None
        and best_damage_move.id != move_id
        and not battle.force_switch
    ):
        loop_threshold = max(
            self.LOOP_BREAKER_MIN_SCORE,
            self.TACTICAL_KO_THRESHOLD
            * max(0.0, self.LOOP_BREAKER_KO_FRACTION)
            * max(opp_hp, 0.05),
        )
        if best_damage_score >= loop_threshold:
            return best_damage_move.id

    if is_status and self._status_choice_is_obviously_bad(battle, selected_move, active, opponent):
        if best_damage_move is not None:
            return best_damage_move.id
        if battle.available_switches:
            best_sw = max(battle.available_switches, key=lambda s: self._score_switch(s, opponent, battle))
            return f"switch {normalize_name(best_sw.species)}"

    if is_setup and (self._has_effect(active, "encore") or self._setup_move_is_encore_bait(selected_move, active, opponent)):
        if best_damage_move is not None and best_damage_move.id != move_id:
            return best_damage_move.id
        if battle.available_switches:
            best_sw = max(battle.available_switches, key=lambda s: self._score_switch(s, opponent, battle))
            return f"switch {normalize_name(best_sw.species)}"

    if (
        bool(getattr(self, "BOOSTED_FOE_SETUP_RACE_GUARD", True))
        and is_setup
        and best_damage_move is not None
        and best_damage_move.id != move_id
        and boosted_offense_pressure(opponent) >= float(getattr(self, "BOOSTED_FOE_SETUP_RACE_MIN_PRESSURE", 2.0))
        and reply_score >= float(getattr(self, "BOOSTED_FOE_SETUP_RACE_MIN_REPLY", 70.0))
    ):
        mem["setup_race_last"] = {
            "reason": "reject_setup_into_boosted_foe",
            "choice": choice,
            "replacement": best_damage_move.id,
            "reply_score": float(reply_score),
            "boosted_pressure": float(boosted_offense_pressure(opponent)),
        }
        return best_damage_move.id

    if (
        is_setup
        and best_damage_move is not None
        and best_damage_move.id != move_id
        and not battle.force_switch
        and active_hp <= self.SETUP_PRESSURE_HP_MAX
        and reply_score >= self.SETUP_PRESSURE_REPLY
    ):
        return best_damage_move.id

    if (
        passive_kind
        and best_damage_move is not None
        and best_damage_move.id != move_id
        and not battle.force_switch
    ):
        if same_matchup_repeat:
            if passive_kind == "protect":
                if reply_score < self.PROGRESS_NEED_REPLY or same_move_repeat_streak >= 1:
                    return best_damage_move.id
            elif passive_kind == "recovery":
                if active_hp >= self.PASSIVE_REPEAT_HIGH_HP_MAX or same_move_repeat_streak >= 1:
                    return best_damage_move.id
            elif passive_kind == "status":
                if stall >= 1 or passive_streak >= 1 or same_move_repeat_streak >= 1:
                    return best_damage_move.id
        if passive_kind == "protect" and self._should_use_protect(battle, reply_score):
            if progress_need <= 1 and passive_streak <= 0:
                pass
            elif best_damage_score >= self.PROGRESS_NEED_DAMAGE:
                return best_damage_move.id
        elif passive_kind == "recovery":
            safe_low_hp_recover = (
                active_hp <= self.PASSIVE_BREAK_RECOVERY_HP_MAX and reply_score < self.PROGRESS_NEED_REPLY
            )
            if not safe_low_hp_recover and progress_need >= 2 and (
                passive_streak >= 1 or active_hp >= 0.55
            ):
                if best_damage_score >= self.PROGRESS_NEED_DAMAGE:
                    return best_damage_move.id
        elif passive_kind == "status":
            if progress_need >= 2 and (stall >= 1 or passive_streak >= 1):
                if best_damage_score >= self.PROGRESS_NEED_DAMAGE:
                    return best_damage_move.id

    if has_ko_window and (is_status or is_recovery or is_setup):
        if best_damage_move is not None:
            return best_damage_move.id

    if has_ko_window and is_protect and best_damage_move is not None:
        return best_damage_move.id

    if tera_suffix:
        return f"{move_id}-tera"
    return choice
