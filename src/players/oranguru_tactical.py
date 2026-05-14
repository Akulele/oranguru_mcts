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

    if opponent.status is not None and status_type in {"poison", "burn", "para", "sleep", "yawn"}:
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
