from __future__ import annotations

import math
import os
import random
from typing import Dict, List, Optional, Tuple

from poke_env.battle import Battle, Pokemon, MoveCategory

from src.utils.damage_calc import normalize_name


_COMMIT_AVOIDING_MOVES = {
    "batonpass",
    "chillyreception",
    "flipturn",
    "partingshot",
    "shedtail",
    "teleport",
    "uturn",
    "voltswitch",
}


def heuristic_action_score(self, battle: Battle, choice: str) -> Optional[float]:
    active = battle.active_pokemon
    opponent = battle.opponent_active_pokemon
    if active is None or opponent is None:
        return None
    mem = self._get_battle_memory(battle)

    if choice.startswith("switch "):
        target = normalize_name(choice.split("switch ", 1)[1])
        for sw in battle.available_switches:
            if normalize_name(sw.species) == target:
                score = max(0.0, self._score_switch(sw, opponent, battle))
                switch_streak = int(mem.get("self_switch_streak", 0) or 0)
                if switch_streak >= 2:
                    score -= 20.0 * min(4, switch_streak - 1)
                return max(0.0, score)
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
                stall = int(mem.get("status_stall_streak", 0) or 0)
                penalty = 18.0 * min(3, stall)
                return max(0.0, 120.0 * danger - penalty)
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
                stall = int(mem.get("status_stall_streak", 0) or 0)
                penalty = 20.0 * min(3, stall)
                return max(0.0, 140.0 * missing - penalty)
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
            stall = int(mem.get("status_stall_streak", 0) or 0)
            if stall >= 1 and status_type not in {"taunt", "encore"}:
                score -= 20.0 * min(3, stall)
            return max(0.0, score)
        predicted_switch = self._predict_opponent_switch(battle)
        return max(
            0.0,
            self._score_move_with_prediction(move, active, opponent, predicted_switch, battle),
        )
    return None


def adaptive_choice_risk_penalty(self, battle: Battle, choice: str) -> float:
    active = battle.active_pokemon
    opponent = battle.opponent_active_pokemon
    if active is None or opponent is None:
        return 0.0

    mem = self._get_battle_memory(battle)
    risk = 0.0

    if choice.startswith("switch "):
        target = normalize_name(choice.split("switch ", 1)[1])
        selected = None
        for sw in battle.available_switches or []:
            if normalize_name(sw.species) == target:
                selected = sw
                break
        if selected is None:
            return 80.0
        if self._switch_faints_to_entry_hazards(battle, selected):
            risk += 150.0
        cur_match = self._estimate_matchup(active, opponent)
        new_match = self._estimate_matchup(selected, opponent)
        if new_match < cur_match - 0.15:
            risk += 45.0
        switch_streak = int(mem.get("self_switch_streak", 0) or 0)
        if switch_streak >= 2:
            risk += 25.0 * min(4, switch_streak - 1)
        return risk

    move_id = normalize_name(choice.replace("-tera", ""))
    selected_move = None
    for move in battle.available_moves or []:
        if move.id == move_id:
            selected_move = move
            break
    if selected_move is None:
        return 80.0

    best_damage_move, best_damage_score = self._best_damaging_move(battle, active, opponent)
    opp_hp = opponent.current_hp_fraction or 0.0
    ko_threshold = self.TACTICAL_KO_THRESHOLD * max(opp_hp, 0.05)
    has_ko_window = best_damage_move is not None and best_damage_score >= ko_threshold

    try:
        is_status = selected_move.category == MoveCategory.STATUS
    except Exception:
        is_status = False
    is_recovery = self._is_recovery_move(selected_move)
    is_setup = bool(getattr(selected_move, "boosts", None) or {}) and (
        selected_move.target and "self" in str(selected_move.target).lower()
    )

    if is_status and self._status_choice_is_obviously_bad(battle, selected_move, active, opponent):
        risk += 90.0
    if has_ko_window and (is_status or is_recovery or is_setup or selected_move.id in self.PROTECT_MOVES):
        risk += 120.0
    if is_recovery and (active.current_hp_fraction or 0.0) > 0.75:
        risk += 35.0
    if selected_move.id in self.PROTECT_MOVES:
        reply_score = self._estimate_best_reply_score(opponent, active, battle)
        if not self._should_use_protect(battle, reply_score):
            risk += 70.0

    stall = int(mem.get("status_stall_streak", 0) or 0)
    if stall >= 1 and (is_status or is_recovery or selected_move.id in self.PROTECT_MOVES):
        risk += 25.0 * min(3, stall)
    return risk


def adaptive_rerank_choice(
    self,
    battle: Battle,
    ordered: List[Tuple[str, float]],
    topk: int,
) -> str:
    if not ordered:
        return ""
    candidates = ordered[: max(1, topk)]
    total = sum(max(0.0, w) for _, w in candidates)
    if total <= 0:
        return candidates[0][0]

    heur_weight = max(0.0, self.ADAPTIVE_RERANK_HEUR_WEIGHT)
    risk_weight = max(0.0, self.ADAPTIVE_RERANK_RISK_WEIGHT)
    max_policy_drop = max(0.0, min(1.0, self.ADAPTIVE_RERANK_MAX_POLICY_DROP))
    min_score_gain = max(0.0, self.ADAPTIVE_RERANK_MIN_SCORE_GAIN)
    min_risk_delta = max(0.0, self.ADAPTIVE_RERANK_MIN_RISK_DELTA)

    scored: List[dict] = []
    for choice, weight in candidates:
        mcts_term = max(0.0, weight) / total
        heur = max(0.0, float(self._heuristic_action_score(battle, choice) or 0.0))
        risk = max(0.0, float(self._adaptive_choice_risk_penalty(battle, choice)))
        score = mcts_term + heur_weight * (heur / 100.0) - risk_weight * (risk / 100.0)
        scored.append(
            {
                "choice": choice,
                "weight": max(0.0, float(weight)),
                "risk": risk,
                "score": score,
            }
        )
    if not scored:
        return candidates[0][0]

    baseline = scored[0]
    winner = max(scored, key=lambda row: row["score"])
    if winner["choice"] == baseline["choice"]:
        return baseline["choice"]

    policy_drop = (baseline["weight"] - winner["weight"]) / total
    if policy_drop > max_policy_drop:
        return baseline["choice"]

    score_gain = float(winner["score"]) - float(baseline["score"])
    if score_gain < min_score_gain:
        return baseline["choice"]

    risk_delta = float(baseline["risk"]) - float(winner["risk"])
    if risk_delta < min_risk_delta:
        return baseline["choice"]

    return winner["choice"]


def maybe_reduce_negative_matchup_switch(
    self,
    battle: Battle,
    ordered: List[Tuple[str, float]],
    chosen_choice: str,
) -> str:
    def _record(reason: str, **extra) -> None:
        try:
            mem = self._get_battle_memory(battle)
        except Exception:
            return
        if not isinstance(mem, dict):
            return
        payload = {
            "reason": reason,
            "chosen_choice": str(chosen_choice or ""),
        }
        payload.update(extra)
        mem["switch_guard_last"] = payload

    if not chosen_choice or getattr(battle, "force_switch", False):
        _record("forced_or_empty")
        return chosen_choice
    if not chosen_choice.startswith("switch "):
        _record("not_switch")
        return chosen_choice

    active = battle.active_pokemon
    opponent = battle.opponent_active_pokemon
    active_hp = (active.current_hp_fraction or 0.0) if active is not None else 0.0

    switch_weight = 0.0
    for choice, weight in ordered:
        if choice == chosen_choice:
            switch_weight = float(weight or 0.0)
            break

    best_move_choice = ""
    best_move_weight = 0.0
    for choice, weight in ordered:
        if choice.startswith("switch "):
            continue
        if not self._is_damaging_move_choice(battle, choice):
            continue
        best_move_choice = choice
        best_move_weight = float(weight or 0.0)
        break

    if not best_move_choice:
        _record(
            "no_attack_candidate",
            active_hp=float(active_hp),
            switch_weight=float(switch_weight),
        )
        return chosen_choice

    switch_heur = float(self._heuristic_action_score(battle, chosen_choice) or 0.0)
    move_heur = float(self._heuristic_action_score(battle, best_move_choice) or 0.0)
    switch_risk = float(self._adaptive_choice_risk_penalty(battle, chosen_choice) or 0.0)
    if switch_weight <= 0.0:
        weight_ratio = 1.0 if best_move_weight > 0.0 else 0.0
    else:
        weight_ratio = best_move_weight / max(switch_weight, 1e-6)

    def _chosen_switch_target():
        try:
            switch_name = normalize_name(chosen_choice.split("switch ", 1)[1])
        except Exception:
            return None
        for sw in getattr(battle, "available_switches", None) or []:
            if normalize_name(getattr(sw, "species", "")) == switch_name:
                return sw
        return None

    def _details(**extra) -> dict:
        payload = {
            "active_hp": float(active_hp),
            "attack_choice": best_move_choice,
            "switch_weight": float(switch_weight),
            "attack_weight": float(best_move_weight),
            "policy_ratio": float(weight_ratio),
            "switch_heuristic": float(switch_heur),
            "attack_heuristic": float(move_heur),
            "heuristic_delta": float(move_heur - switch_heur),
            "switch_risk": float(switch_risk),
        }
        if opponent is not None:
            payload["opp_hp"] = float(opponent.current_hp_fraction or 0.0)
        payload.update(extra)
        return payload

    min_active_hp = float(getattr(self, "SWITCH_GUARD_MIN_ACTIVE_HP", 0.45))
    if active_hp < min_active_hp:
        switch_target = _chosen_switch_target()
        target_hp = getattr(switch_target, "current_hp_fraction", None) if switch_target is not None else None
        if not isinstance(target_hp, (int, float)):
            _record("low_active_hp", **_details(min_active_hp=min_active_hp))
            return chosen_choice
        min_target_hp = float(getattr(self, "SWITCH_GUARD_LOW_HP_TARGET_MIN_HP", 0.35))
        min_hp_gain = float(getattr(self, "SWITCH_GUARD_LOW_HP_MIN_HP_GAIN", 0.15))
        hp_gain = float(target_hp) - float(active_hp)
        if target_hp >= min_target_hp and hp_gain >= min_hp_gain:
            _record(
                "low_active_hp",
                **_details(
                    min_active_hp=min_active_hp,
                    switch_target_hp=float(target_hp),
                    switch_hp_gain=float(hp_gain),
                    min_target_hp=min_target_hp,
                    min_hp_gain=min_hp_gain,
                ),
            )
            return chosen_choice

        low_hp_policy_ratio = float(getattr(self, "SWITCH_GUARD_LOW_HP_POLICY_RATIO", 0.45))
        low_hp_heur_floor = float(getattr(self, "SWITCH_GUARD_LOW_HP_HEUR_FLOOR", 0.0))
        if weight_ratio >= low_hp_policy_ratio and move_heur >= low_hp_heur_floor:
            _record(
                "take_low_hp_attack",
                **_details(
                    min_active_hp=min_active_hp,
                    switch_target_hp=float(target_hp),
                    switch_hp_gain=float(hp_gain),
                    min_target_hp=min_target_hp,
                    min_hp_gain=min_hp_gain,
                    min_policy_ratio=low_hp_policy_ratio,
                    min_attack_heuristic=low_hp_heur_floor,
                ),
            )
            return best_move_choice

        _record(
            "low_active_hp",
            **_details(
                min_active_hp=min_active_hp,
                switch_target_hp=float(target_hp),
                switch_hp_gain=float(hp_gain),
                min_target_hp=min_target_hp,
                min_hp_gain=min_hp_gain,
            ),
        )
        return chosen_choice

    heur_delta = move_heur - switch_heur
    policy_ratio = float(getattr(self, "SWITCH_GUARD_POLICY_RATIO", 0.70))
    min_heur_gain = float(getattr(self, "SWITCH_GUARD_HEUR_GAIN", 1.0))
    if weight_ratio >= policy_ratio and heur_delta >= min_heur_gain:
        _record(
            "take_live_attack",
            **_details(min_policy_ratio=policy_ratio, min_heur_gain=min_heur_gain),
        )
        return best_move_choice

    risk_policy_ratio = float(getattr(self, "SWITCH_GUARD_RISK_POLICY_RATIO", 0.60))
    risk_min_risk = float(getattr(self, "SWITCH_GUARD_RISK_MIN_RISK", 20.0))
    risk_heur_floor = float(getattr(self, "SWITCH_GUARD_RISK_HEUR_FLOOR", -0.5))
    if switch_risk >= risk_min_risk and weight_ratio >= risk_policy_ratio and heur_delta >= risk_heur_floor:
        _record(
            "take_risk_attack",
            **_details(
                min_policy_ratio=risk_policy_ratio,
                min_switch_risk=risk_min_risk,
                min_heur_gain=risk_heur_floor,
            ),
        )
        return best_move_choice

    if weight_ratio >= 0.75 and move_heur >= switch_heur + 10.0:
        _record("take_legacy_attack", **_details(min_policy_ratio=0.75, min_heur_gain=10.0))
        return best_move_choice
    if weight_ratio >= 0.95 and move_heur >= switch_heur - 5.0 and move_heur > 0.0:
        _record("take_legacy_near_tie_attack", **_details(min_policy_ratio=0.95, min_heur_gain=-5.0))
        return best_move_choice
    if switch_risk >= 35.0 and weight_ratio >= 0.60 and move_heur >= max(0.0, switch_heur - 10.0):
        _record("take_legacy_risk_attack", **_details(min_policy_ratio=0.60, min_switch_risk=35.0))
        return best_move_choice
    _record(
        "policy_or_heuristic",
        **_details(min_policy_ratio=policy_ratio, min_heur_gain=min_heur_gain),
    )
    return chosen_choice


def maybe_commit_late_game_attack_choice(
    self,
    battle: Battle,
    ordered: List[Tuple[str, float]],
    chosen_choice: str,
) -> str:
    def _record(reason: str, **extra) -> None:
        try:
            mem = self._get_battle_memory(battle)
        except Exception:
            return
        if not isinstance(mem, dict):
            return
        payload = {
            "reason": reason,
            "chosen_choice": str(chosen_choice or ""),
        }
        payload.update(extra)
        mem["late_game_attack_guard_last"] = payload

    def _alive_count(team: object) -> tuple[int, int]:
        if not isinstance(team, dict):
            return 0, 0
        visible = 0
        alive = 0
        for mon in team.values():
            visible += 1
            if not bool(getattr(mon, "fainted", False)):
                alive += 1
        return visible, alive

    def _move_for_choice(choice: str):
        move_id = normalize_name(choice.replace("-tera", ""))
        for move in battle.available_moves or []:
            if normalize_name(getattr(move, "id", "")) == move_id:
                return move
        return None

    def _is_commit_avoiding_choice(choice: str) -> bool:
        if not choice:
            return False
        if choice.startswith("switch "):
            return True
        move = _move_for_choice(choice)
        if move is None:
            return False
        move_id = normalize_name(getattr(move, "id", ""))
        protect_moves = getattr(self, "PROTECT_MOVES", set())
        if move_id in _COMMIT_AVOIDING_MOVES or move_id in protect_moves:
            return True
        try:
            if self._is_recovery_move(move):
                return True
        except Exception:
            pass
        return getattr(move, "category", None) == MoveCategory.STATUS

    def _is_direct_attack_choice(choice: str) -> bool:
        move = _move_for_choice(choice)
        if move is None:
            return False
        move_id = normalize_name(getattr(move, "id", ""))
        if move_id in _COMMIT_AVOIDING_MOVES:
            return False
        if not self._is_damaging_move_choice(battle, choice):
            return False
        min_base_power = float(getattr(self, "LATEGAME_ATTACK_MIN_BASE_POWER", 50.0))
        base_power = float(getattr(move, "base_power", 0.0) or 0.0)
        damage_attr = getattr(move, "damage", None)
        return base_power >= min_base_power or damage_attr not in (None, 0, "0", False)

    if not chosen_choice or getattr(battle, "force_switch", False):
        _record("forced_or_empty")
        return chosen_choice
    if not _is_commit_avoiding_choice(chosen_choice):
        _record("already_committed")
        return chosen_choice

    active = battle.active_pokemon
    opponent = battle.opponent_active_pokemon
    if active is None or opponent is None:
        _record("missing_active")
        return chosen_choice

    my_visible, my_alive = _alive_count(getattr(battle, "team", {}) or {})
    opp_visible, opp_alive = _alive_count(getattr(battle, "opponent_team", {}) or {})
    opp_hidden = max(0, 6 - opp_visible)
    allow_hidden = bool(getattr(self, "LATEGAME_ATTACK_ALLOW_HIDDEN", False))
    turn = int(getattr(battle, "turn", 0) or 0)
    min_turn = int(getattr(self, "LATEGAME_ATTACK_MIN_TURN", 12))

    known_endgame = opp_hidden == 0 and my_alive <= 2 and opp_alive <= 2
    late_known_close = (
        opp_hidden == 0
        and turn >= min_turn
        and my_alive <= 3
        and opp_alive <= 3
    )
    hidden_desperation = allow_hidden and turn >= min_turn and my_alive <= 2 and opp_alive <= 2
    if not (known_endgame or late_known_close or hidden_desperation):
        _record(
            "not_lategame",
            turn=int(turn),
            min_turn=int(min_turn),
            my_visible=int(my_visible),
            my_alive=int(my_alive),
            opp_visible=int(opp_visible),
            opp_alive=int(opp_alive),
            opp_hidden=int(opp_hidden),
        )
        return chosen_choice

    weights = {choice: float(weight or 0.0) for choice, weight in ordered}
    chosen_weight = weights.get(chosen_choice, 0.0)
    chosen_heur = float(self._heuristic_action_score(battle, chosen_choice) or 0.0)

    direct_candidates: Dict[str, Dict[str, float]] = {}
    for choice, weight in ordered:
        if not _is_direct_attack_choice(choice):
            continue
        heur = float(self._heuristic_action_score(battle, choice) or 0.0)
        candidate_weight = float(weight or 0.0)
        direct_candidates[choice] = {
            "weight": candidate_weight,
            "heuristic": heur,
        }

    best_choice = ""
    best_weight = 0.0
    best_heur = 0.0
    selected_damage_score = 0.0

    try:
        best_damage_move, best_damage_score = self._best_damaging_move(battle, active, opponent)
    except Exception:
        best_damage_move = None
        best_damage_score = 0.0
    if best_damage_move is not None:
        best_damage_id = normalize_name(getattr(best_damage_move, "id", ""))
        matching_choices = [
            choice
            for choice in (best_damage_id, f"{best_damage_id}-tera")
            if choice in direct_candidates
        ]
        if matching_choices:
            best_choice = max(
                matching_choices,
                key=lambda choice: (
                    float(direct_candidates[choice]["heuristic"]),
                    float(direct_candidates[choice]["weight"]),
                ),
            )
            best_weight = float(direct_candidates[best_choice]["weight"])
            best_heur = float(direct_candidates[best_choice]["heuristic"])
            selected_damage_score = float(best_damage_score or 0.0)

    if not best_choice:
        for choice, candidate in direct_candidates.items():
            candidate_weight = float(candidate["weight"])
            heur = float(candidate["heuristic"])
            if not best_choice or (heur, candidate_weight) > (best_heur, best_weight):
                best_choice = choice
                best_weight = candidate_weight
                best_heur = heur

    if not best_choice:
        _record(
            "no_direct_attack_candidate",
            turn=int(turn),
            my_alive=int(my_alive),
            opp_alive=int(opp_alive),
            opp_hidden=int(opp_hidden),
            chosen_weight=float(chosen_weight),
        )
        return chosen_choice
    if best_choice == chosen_choice:
        _record("chosen_direct_attack", attack_choice=best_choice)
        return chosen_choice

    min_heuristic = float(getattr(self, "LATEGAME_ATTACK_MIN_HEURISTIC", 0.75))
    if best_heur < min_heuristic:
        _record(
            "weak_attack_heuristic",
            attack_choice=best_choice,
            attack_heuristic=float(best_heur),
            min_heuristic=float(min_heuristic),
        )
        return chosen_choice

    if chosen_weight <= 0.0:
        policy_ratio = 1.0 if best_weight > 0.0 else 0.0
    else:
        policy_ratio = best_weight / max(chosen_weight, 1e-6)
    score_drop = max(0.0, chosen_weight - best_weight)
    min_ratio = float(getattr(self, "LATEGAME_ATTACK_MIN_POLICY_RATIO", 0.05))
    max_drop = float(getattr(self, "LATEGAME_ATTACK_MAX_SCORE_DROP", 0.40))
    if policy_ratio < min_ratio or score_drop > max_drop:
        _record(
            "policy_ratio",
            attack_choice=best_choice,
            chosen_weight=float(chosen_weight),
            attack_weight=float(best_weight),
            policy_ratio=float(policy_ratio),
            min_policy_ratio=float(min_ratio),
            score_drop=float(score_drop),
            max_score_drop=float(max_drop),
        )
        return chosen_choice

    attack_risk = float(self._adaptive_choice_risk_penalty(battle, best_choice) or 0.0)
    max_risk = float(getattr(self, "LATEGAME_ATTACK_MAX_RISK", 35.0))
    if attack_risk > max_risk:
        _record(
            "attack_risk",
            attack_choice=best_choice,
            attack_risk=float(attack_risk),
            max_attack_risk=float(max_risk),
        )
        return chosen_choice

    _record(
        "take_late_attack",
        attack_choice=best_choice,
        turn=int(turn),
        my_alive=int(my_alive),
        opp_alive=int(opp_alive),
        opp_hidden=int(opp_hidden),
        active_hp=float(active.current_hp_fraction or 0.0),
        opp_hp=float(opponent.current_hp_fraction or 0.0),
        chosen_weight=float(chosen_weight),
        attack_weight=float(best_weight),
        policy_ratio=float(policy_ratio),
        score_drop=float(score_drop),
        chosen_heuristic=float(chosen_heur),
        attack_heuristic=float(best_heur),
        attack_damage_score=float(selected_damage_score),
        attack_risk=float(attack_risk),
    )
    return best_choice


def maybe_take_progress_when_behind_choice(
    self,
    battle: Battle,
    ordered: List[Tuple[str, float]],
    chosen_choice: str,
) -> str:
    def _record(reason: str, **extra) -> None:
        try:
            mem = self._get_battle_memory(battle)
        except Exception:
            return
        if not isinstance(mem, dict):
            return
        payload = {
            "reason": reason,
            "chosen_choice": str(chosen_choice or ""),
        }
        payload.update(extra)
        mem["progress_window_last"] = payload

    if not chosen_choice or getattr(battle, "force_switch", False):
        _record("forced_or_empty")
        return chosen_choice
    if chosen_choice.startswith("switch ") or not self._is_damaging_move_choice(battle, chosen_choice):
        _record("not_damaging_choice")
        return chosen_choice

    active = battle.active_pokemon
    opponent = battle.opponent_active_pokemon
    if active is None or opponent is None:
        _record("missing_active")
        return chosen_choice

    active_hp = active.current_hp_fraction or 0.0
    min_active_hp = float(getattr(self, "PROGRESS_WINDOW_MIN_ACTIVE_HP", 0.50))
    if active_hp < min_active_hp:
        _record("low_hp", active_hp=float(active_hp), min_hp=min_active_hp)
        return chosen_choice

    my_alive = sum(
        1
        for mon in (getattr(battle, "team", {}) or {}).values()
        if mon is not None and not getattr(mon, "fainted", False)
    )
    opp_alive = sum(
        1
        for mon in (getattr(battle, "opponent_team", {}) or {}).values()
        if mon is not None and not getattr(mon, "fainted", False)
    )
    revealed_opp_fainted = sum(
        1
        for mon in (getattr(battle, "opponent_team", {}) or {}).values()
        if mon is not None and getattr(mon, "fainted", False)
    )
    # Unrevealed random-battle opponents are still alive; matching RuleBot's
    # convention avoids treating most early/midgame boards as "not behind".
    inferred_opp_alive = max(opp_alive, 6 - revealed_opp_fainted)
    if my_alive <= 0:
        my_alive = 1 + len([sw for sw in (battle.available_switches or []) if not getattr(sw, "fainted", False)])
    opp_alive = max(1, inferred_opp_alive)
    if my_alive >= opp_alive:
        _record("not_behind", active_hp=float(active_hp), my_alive=int(my_alive), opp_alive=int(opp_alive))
        return chosen_choice

    opp_hp = opponent.current_hp_fraction or 0.0
    min_opp_hp = float(getattr(self, "PROGRESS_WINDOW_MIN_OPP_HP", 0.55))
    if opp_hp <= min_opp_hp:
        _record("opponent_low", active_hp=float(active_hp), opp_hp=float(opp_hp), min_opp_hp=min_opp_hp)
        return chosen_choice

    reply_score = float(self._estimate_best_reply_score(opponent, active, battle) or 0.0)
    max_reply = float(getattr(self, "PROGRESS_WINDOW_MAX_REPLY", 110.0))
    if reply_score > max_reply:
        _record(
            "unsafe_reply",
            active_hp=float(active_hp),
            opp_hp=float(opp_hp),
            reply_score=float(reply_score),
            max_reply=max_reply,
        )
        return chosen_choice

    best_damage_score = float(self._estimate_best_damage_score(active, opponent, battle) or 0.0)
    ko_threshold = self.TACTICAL_KO_THRESHOLD * max(opp_hp, 0.05)
    if best_damage_score >= ko_threshold:
        _record(
            "ko_guard",
            active_hp=float(active_hp),
            opp_hp=float(opp_hp),
            reply_score=float(reply_score),
            best_damage_score=float(best_damage_score),
            ko_threshold=float(ko_threshold),
        )
        return chosen_choice

    chosen_weight = 0.0
    for choice, weight in ordered:
        if choice == chosen_choice:
            chosen_weight = float(weight or 0.0)
            break
    chosen_heur = float(self._heuristic_action_score(battle, chosen_choice) or 0.0)

    best_progress_choice = ""
    best_progress_weight = 0.0
    best_progress_heur = 0.0
    best_progress_kind = ""
    opp_side_conditions = getattr(battle, "opponent_side_conditions", {}) or {}
    for choice, weight in ordered:
        if choice == chosen_choice or choice.startswith("switch "):
            continue
        move_id = normalize_name(choice.replace("-tera", ""))
        selected_move = None
        for move in battle.available_moves or []:
            if normalize_name(getattr(move, "id", "")) == move_id:
                selected_move = move
                break
        if selected_move is None:
            continue

        progress_kind = ""
        hazard_condition = self.ENTRY_HAZARDS.get(move_id)
        if hazard_condition is not None and hazard_condition not in opp_side_conditions and opp_alive >= 3:
            progress_kind = "hazard"
        else:
            boosts = getattr(selected_move, "boosts", None) or {}
            is_self_setup = bool(boosts) and selected_move.target and "self" in str(selected_move.target).lower()
            if is_self_setup and self._should_setup_move(selected_move, active, opponent):
                progress_kind = "setup"
            else:
                try:
                    is_status = selected_move.category == MoveCategory.STATUS
                except Exception:
                    is_status = False
                if (
                    is_status
                    and not self._is_recovery_move(selected_move)
                    and selected_move.id not in self.PROTECT_MOVES
                    and not self._status_choice_is_obviously_bad(battle, selected_move, active, opponent)
                    and float(self._should_use_status_move(selected_move, active, opponent, battle) or 0.0) > 0.0
                ):
                    progress_kind = "status"
        if not progress_kind:
            continue

        heur = float(self._heuristic_action_score(battle, choice) or 0.0)
        if heur <= best_progress_heur:
            continue
        best_progress_choice = choice
        best_progress_weight = float(weight or 0.0)
        best_progress_heur = heur
        best_progress_kind = progress_kind

    if not best_progress_choice:
        _record(
            "no_progress_candidate",
            active_hp=float(active_hp),
            opp_hp=float(opp_hp),
            my_alive=int(my_alive),
            opp_alive=int(opp_alive),
            reply_score=float(reply_score),
            chosen_weight=float(chosen_weight),
        )
        return chosen_choice

    heur_gain = best_progress_heur - chosen_heur
    min_heur_gain = float(getattr(self, "PROGRESS_WINDOW_MIN_HEUR_GAIN", 1.0))
    if heur_gain < min_heur_gain:
        _record(
            "heuristic_gain",
            active_hp=float(active_hp),
            opp_hp=float(opp_hp),
            my_alive=int(my_alive),
            opp_alive=int(opp_alive),
            reply_score=float(reply_score),
            chosen_weight=float(chosen_weight),
            progress_choice=best_progress_choice,
            progress_weight=float(best_progress_weight),
            progress_kind=best_progress_kind,
            chosen_heuristic=float(chosen_heur),
            progress_heuristic=float(best_progress_heur),
            min_heur_gain=min_heur_gain,
        )
        return chosen_choice

    high_heur_gain = float(getattr(self, "PROGRESS_WINDOW_HIGH_HEUR_GAIN", 10.0))
    high_gain = heur_gain >= high_heur_gain
    min_ratio = (
        float(getattr(self, "PROGRESS_WINDOW_HIGH_GAIN_MIN_POLICY_RATIO", 0.30))
        if high_gain
        else float(getattr(self, "PROGRESS_WINDOW_MIN_POLICY_RATIO", 0.65))
    )
    if best_progress_weight < chosen_weight * min_ratio:
        _record(
            "policy_ratio",
            active_hp=float(active_hp),
            opp_hp=float(opp_hp),
            my_alive=int(my_alive),
            opp_alive=int(opp_alive),
            reply_score=float(reply_score),
            chosen_weight=float(chosen_weight),
            progress_choice=best_progress_choice,
            progress_weight=float(best_progress_weight),
            progress_kind=best_progress_kind,
            chosen_heuristic=float(chosen_heur),
            progress_heuristic=float(best_progress_heur),
            high_gain=bool(high_gain),
            min_policy_ratio=float(min_ratio),
        )
        return chosen_choice

    _record(
        "take_progress",
        active_hp=float(active_hp),
        opp_hp=float(opp_hp),
        my_alive=int(my_alive),
        opp_alive=int(opp_alive),
        reply_score=float(reply_score),
        chosen_weight=float(chosen_weight),
        progress_choice=best_progress_choice,
        progress_weight=float(best_progress_weight),
        progress_kind=best_progress_kind,
        chosen_heuristic=float(chosen_heur),
        progress_heuristic=float(best_progress_heur),
        high_gain=bool(high_gain),
    )
    return best_progress_choice


def maybe_force_finish_blow_choice(
    self,
    battle: Battle,
    ordered: List[Tuple[str, float]],
    chosen_choice: str,
) -> str:
    def _record(reason: str, **extra) -> None:
        try:
            mem = self._get_battle_memory(battle)
        except Exception:
            return
        if not isinstance(mem, dict):
            return
        payload = {
            "reason": reason,
            "chosen_choice": str(chosen_choice or ""),
        }
        payload.update(extra)
        mem["finish_blow_last"] = payload

    def _move_for_choice(choice: str):
        move_id = normalize_name(choice.replace("-tera", ""))
        for move in battle.available_moves or []:
            if normalize_name(getattr(move, "id", "")) == move_id:
                return move
        return None

    def _damage_score(move) -> float:
        try:
            return float(
                self._calculate_move_score(
                    move,
                    active,
                    opponent,
                    battle,
                    apply_recoil=False,
                )
                or 0.0
            )
        except TypeError:
            try:
                return float(self._calculate_move_score(move, active, opponent, battle) or 0.0)
            except Exception:
                return 0.0
        except Exception:
            return 0.0

    def _accuracy(move) -> float:
        raw = getattr(move, "accuracy", None)
        if raw is None or raw is True:
            return 1.0
        try:
            return max(0.0, min(1.0, float(raw) / 100.0))
        except Exception:
            return 1.0

    def _crash_rate(move) -> float:
        move_id = normalize_name(getattr(move, "id", ""))
        if move_id in {"axekick", "highjumpkick", "jumpkick", "supercellslam"}:
            return 0.5
        try:
            entry = self._get_move_entry(move)
        except Exception:
            entry = {}
        crash = entry.get("crash") if isinstance(entry, dict) else None
        if isinstance(crash, list) and len(crash) == 2:
            try:
                return float(crash[0]) / max(float(crash[1]), 1.0)
            except Exception:
                return 0.5
        if isinstance(entry, dict) and entry.get("hasCrashDamage"):
            return 0.5
        return 0.0

    def _recoil_rate(move) -> float:
        try:
            return max(0.0, float(self._move_recoil_rate(move) or 0.0))
        except Exception:
            return 0.0

    def _finish_risk(move) -> float:
        acc = _accuracy(move)
        return max(0.0, 1.0 - acc) + _crash_rate(move) + _recoil_rate(move)

    def _maybe_take_safer_ko(
        active: Pokemon,
        opponent: Pokemon,
        opp_hp: float,
        ko_threshold: float,
    ) -> str:
        if not bool(getattr(self, "SAFE_KO_GUARD", True)):
            _record("chosen_damaging")
            return chosen_choice
        chosen_move = _move_for_choice(chosen_choice)
        if chosen_move is None:
            _record("chosen_damaging")
            return chosen_choice
        chosen_score = _damage_score(chosen_move)
        if chosen_score < ko_threshold:
            _record(
                "chosen_damaging",
                opp_hp=float(opp_hp),
                chosen_damage_score=float(chosen_score),
                ko_threshold=float(ko_threshold),
            )
            return chosen_choice

        weights = {choice: float(weight or 0.0) for choice, weight in ordered}
        chosen_weight = weights.get(chosen_choice, 0.0)
        chosen_risk = _finish_risk(chosen_move)
        min_overkill = float(getattr(self, "SAFE_KO_MIN_OVERKILL", 1.0))
        min_risk_delta = float(getattr(self, "SAFE_KO_MIN_RISK_DELTA", 0.10))
        min_policy_ratio = float(getattr(self, "SAFE_KO_MIN_POLICY_RATIO", 0.02))
        best_choice = chosen_choice
        best_move = chosen_move
        best_score = chosen_score
        best_risk = chosen_risk
        best_weight = chosen_weight

        candidate_choices = list(ordered)
        seen = {choice for choice, _ in candidate_choices}
        for move in battle.available_moves or []:
            candidate = normalize_name(getattr(move, "id", ""))
            if candidate and candidate not in seen:
                candidate_choices.append((candidate, 0.0))

        for choice, weight in candidate_choices:
            if choice == chosen_choice or choice.startswith("switch "):
                continue
            if not self._is_damaging_move_choice(battle, choice):
                continue
            move = _move_for_choice(choice)
            if move is None:
                continue
            score = _damage_score(move)
            if score < ko_threshold * min_overkill:
                continue
            candidate_weight = float(weight or 0.0)
            if chosen_weight > 0.0 and candidate_weight / max(chosen_weight, 1e-6) < min_policy_ratio:
                continue
            risk = _finish_risk(move)
            risk_delta = chosen_risk - risk
            if risk_delta < min_risk_delta:
                continue
            if (
                best_choice == chosen_choice
                or (risk, -score, -candidate_weight) < (best_risk, -best_score, -best_weight)
            ):
                best_choice = choice
                best_move = move
                best_score = score
                best_risk = risk
                best_weight = candidate_weight

        if best_choice == chosen_choice:
            _record(
                "chosen_damaging",
                opp_hp=float(opp_hp),
                chosen_damage_score=float(chosen_score),
                ko_threshold=float(ko_threshold),
                chosen_accuracy=float(_accuracy(chosen_move)),
                chosen_finish_risk=float(chosen_risk),
            )
            return chosen_choice

        _record(
            "take_safe_ko",
            opp_hp=float(opp_hp),
            ko_threshold=float(ko_threshold),
            chosen_damage_score=float(chosen_score),
            safe_damage_score=float(best_score),
            chosen_accuracy=float(_accuracy(chosen_move)),
            safe_accuracy=float(_accuracy(best_move)),
            chosen_finish_risk=float(chosen_risk),
            safe_finish_risk=float(best_risk),
            chosen_weight=float(chosen_weight),
            safe_weight=float(best_weight),
            finish_choice=best_choice,
        )
        return best_choice

    if not chosen_choice or getattr(battle, "force_switch", False):
        _record("forced_or_empty")
        return chosen_choice

    active = battle.active_pokemon
    opponent = battle.opponent_active_pokemon
    if active is None or opponent is None:
        _record("missing_active")
        return chosen_choice

    opp_hp = opponent.current_hp_fraction or 0.0
    ko_threshold = self.TACTICAL_KO_THRESHOLD * max(opp_hp, 0.05)
    if self._is_damaging_move_choice(battle, chosen_choice):
        return _maybe_take_safer_ko(active, opponent, opp_hp, ko_threshold)

    best_damage_score = float(self._estimate_best_damage_score(active, opponent, battle) or 0.0)
    if best_damage_score < ko_threshold:
        _record(
            "no_ko_window",
            active_hp=float(active.current_hp_fraction or 0.0),
            opp_hp=float(opp_hp),
            best_damage_score=float(best_damage_score),
            ko_threshold=float(ko_threshold),
        )
        return chosen_choice

    best_damage_choice = ""
    best_damage_heur = 0.0
    for choice, weight in ordered:
        if choice.startswith("switch "):
            continue
        if not self._is_damaging_move_choice(battle, choice):
            continue
        heur = float(self._heuristic_action_score(battle, choice) or 0.0)
        if not best_damage_choice or heur > best_damage_heur:
            best_damage_choice = choice
            best_damage_heur = heur

    if not best_damage_choice:
        for move in battle.available_moves or []:
            try:
                if move.category == MoveCategory.STATUS:
                    continue
            except Exception:
                continue
            candidate = normalize_name(getattr(move, "id", ""))
            if not candidate:
                continue
            heur = float(self._heuristic_action_score(battle, candidate) or 0.0)
            if not best_damage_choice or heur > best_damage_heur:
                best_damage_choice = candidate
                best_damage_heur = heur

    if not best_damage_choice:
        _record(
            "no_damage_choice",
            active_hp=float(active.current_hp_fraction or 0.0),
            opp_hp=float(opp_hp),
            best_damage_score=float(best_damage_score),
            ko_threshold=float(ko_threshold),
        )
        return chosen_choice
    if chosen_choice.startswith("switch "):
        _record(
            "take_switch_finish",
            active_hp=float(active.current_hp_fraction or 0.0),
            opp_hp=float(opp_hp),
            best_damage_score=float(best_damage_score),
            ko_threshold=float(ko_threshold),
            finish_choice=best_damage_choice,
            finish_heuristic=float(best_damage_heur),
        )
        return best_damage_choice
    chosen_heur = float(self._heuristic_action_score(battle, chosen_choice) or 0.0)
    if best_damage_heur > 0.0:
        _record(
            "take_passive_finish",
            active_hp=float(active.current_hp_fraction or 0.0),
            opp_hp=float(opp_hp),
            best_damage_score=float(best_damage_score),
            ko_threshold=float(ko_threshold),
            finish_choice=best_damage_choice,
            finish_heuristic=float(best_damage_heur),
            chosen_heuristic=float(chosen_heur),
        )
        return best_damage_choice
    _record(
        "passive_heuristic_guard",
        active_hp=float(active.current_hp_fraction or 0.0),
        opp_hp=float(opp_hp),
        best_damage_score=float(best_damage_score),
        ko_threshold=float(ko_threshold),
        finish_choice=best_damage_choice,
        finish_heuristic=float(best_damage_heur),
        chosen_heuristic=float(chosen_heur),
    )
    return chosen_choice


def maybe_take_setup_window_choice(
    self,
    battle: Battle,
    ordered: List[Tuple[str, float]],
    chosen_choice: str,
) -> str:
    def _record(reason: str, **extra) -> None:
        try:
            mem = self._get_battle_memory(battle)
        except Exception:
            return
        if not isinstance(mem, dict):
            return
        payload = {
            "reason": reason,
            "chosen_choice": str(chosen_choice or ""),
        }
        payload.update(extra)
        mem["setup_window_last"] = payload

    if not chosen_choice or getattr(battle, "force_switch", False):
        _record("forced_or_empty")
        return chosen_choice
    if chosen_choice.startswith("switch ") or not self._is_damaging_move_choice(battle, chosen_choice):
        _record("not_damaging_choice")
        return chosen_choice

    active = battle.active_pokemon
    opponent = battle.opponent_active_pokemon
    if active is None or opponent is None:
        _record("missing_active")
        return chosen_choice

    active_hp = active.current_hp_fraction or 0.0
    if active_hp < self.SETUP_WINDOW_MIN_HP:
        _record("low_hp", active_hp=float(active_hp), min_hp=float(self.SETUP_WINDOW_MIN_HP))
        return chosen_choice

    reply_score = float(self._estimate_best_reply_score(opponent, active, battle) or 0.0)
    if reply_score > self.SETUP_WINDOW_MAX_REPLY:
        _record(
            "unsafe_reply",
            active_hp=float(active_hp),
            reply_score=float(reply_score),
            max_reply=float(self.SETUP_WINDOW_MAX_REPLY),
        )
        return chosen_choice

    opp_hp = opponent.current_hp_fraction or 0.0
    best_damage_score = float(self._estimate_best_damage_score(active, opponent, battle) or 0.0)
    ko_threshold = self.TACTICAL_KO_THRESHOLD * max(opp_hp, 0.05)
    if best_damage_score >= ko_threshold:
        _record(
            "ko_guard",
            active_hp=float(active_hp),
            opp_hp=float(opp_hp),
            reply_score=float(reply_score),
            best_damage_score=float(best_damage_score),
            ko_threshold=float(ko_threshold),
        )
        return chosen_choice

    chosen_weight = 0.0
    for choice, weight in ordered:
        if choice == chosen_choice:
            chosen_weight = float(weight or 0.0)
            break

    best_setup_choice = ""
    best_setup_weight = 0.0
    best_setup_heur = 0.0
    for choice, weight in ordered:
        if choice.startswith("switch "):
            continue
        move_id = normalize_name(choice.replace("-tera", ""))
        selected_move = None
        for move in battle.available_moves or []:
            if normalize_name(getattr(move, "id", "")) == move_id:
                selected_move = move
                break
        if selected_move is None:
            continue
        boosts = getattr(selected_move, "boosts", None) or {}
        if not boosts or not selected_move.target or "self" not in str(selected_move.target).lower():
            continue
        if not self._should_setup_move(selected_move, active, opponent):
            continue
        heur = float(self._heuristic_action_score(battle, choice) or 0.0)
        if heur <= best_setup_heur:
            continue
        best_setup_choice = choice
        best_setup_weight = float(weight or 0.0)
        best_setup_heur = heur

    if not best_setup_choice:
        _record(
            "no_setup_candidate",
            active_hp=float(active_hp),
            opp_hp=float(opp_hp),
            reply_score=float(reply_score),
            best_damage_score=float(best_damage_score),
            ko_threshold=float(ko_threshold),
            chosen_weight=float(chosen_weight),
        )
        return chosen_choice

    chosen_heur = float(self._heuristic_action_score(battle, chosen_choice) or 0.0)
    if best_setup_heur < chosen_heur + self.SETUP_WINDOW_MIN_HEUR_GAIN:
        _record(
            "heuristic_gain",
            active_hp=float(active_hp),
            opp_hp=float(opp_hp),
            reply_score=float(reply_score),
            best_damage_score=float(best_damage_score),
            ko_threshold=float(ko_threshold),
            chosen_weight=float(chosen_weight),
            setup_choice=best_setup_choice,
            setup_weight=float(best_setup_weight),
            chosen_heuristic=float(chosen_heur),
            setup_heuristic=float(best_setup_heur),
            min_heur_gain=float(self.SETUP_WINDOW_MIN_HEUR_GAIN),
        )
        return chosen_choice
    if best_setup_weight < chosen_weight * self.SETUP_WINDOW_MIN_POLICY_RATIO:
        high_gain = best_setup_heur >= chosen_heur + self.SETUP_WINDOW_HIGH_HEUR_GAIN
        if not high_gain or best_setup_weight < chosen_weight * self.SETUP_WINDOW_HIGH_GAIN_MIN_POLICY_RATIO:
            _record(
                "policy_ratio",
                active_hp=float(active_hp),
                opp_hp=float(opp_hp),
                reply_score=float(reply_score),
                best_damage_score=float(best_damage_score),
                ko_threshold=float(ko_threshold),
                chosen_weight=float(chosen_weight),
                setup_choice=best_setup_choice,
                setup_weight=float(best_setup_weight),
                chosen_heuristic=float(chosen_heur),
                setup_heuristic=float(best_setup_heur),
                high_gain=bool(high_gain),
                min_policy_ratio=float(
                    self.SETUP_WINDOW_HIGH_GAIN_MIN_POLICY_RATIO if high_gain else self.SETUP_WINDOW_MIN_POLICY_RATIO
                ),
            )
            return chosen_choice
    _record(
        "take_setup",
        active_hp=float(active_hp),
        opp_hp=float(opp_hp),
        reply_score=float(reply_score),
        best_damage_score=float(best_damage_score),
        ko_threshold=float(ko_threshold),
        chosen_weight=float(chosen_weight),
        setup_choice=best_setup_choice,
        setup_weight=float(best_setup_weight),
        chosen_heuristic=float(chosen_heur),
        setup_heuristic=float(best_setup_heur),
        high_gain=bool(best_setup_heur >= chosen_heur + self.SETUP_WINDOW_HIGH_HEUR_GAIN),
    )
    return best_setup_choice


def maybe_take_safe_recovery_choice(
    self,
    battle: Battle,
    ordered: List[Tuple[str, float]],
    chosen_choice: str,
) -> str:
    def _record(reason: str, **extra) -> None:
        try:
            mem = self._get_battle_memory(battle)
        except Exception:
            return
        if not isinstance(mem, dict):
            return
        payload = {
            "reason": reason,
            "chosen_choice": str(chosen_choice or ""),
        }
        payload.update(extra)
        mem["recovery_window_last"] = payload

    if not chosen_choice or getattr(battle, "force_switch", False):
        _record("forced_or_empty")
        return chosen_choice

    active = battle.active_pokemon
    opponent = battle.opponent_active_pokemon
    if active is None or opponent is None:
        _record("missing_active")
        return chosen_choice

    active_hp = active.current_hp_fraction or 0.0
    if active_hp > self.RECOVERY_WINDOW_MAX_HP:
        _record("high_hp", active_hp=float(active_hp), max_hp=float(self.RECOVERY_WINDOW_MAX_HP))
        return chosen_choice

    opp_hp = opponent.current_hp_fraction or 0.0
    if opp_hp <= self.RECOVERY_WINDOW_MIN_OPP_HP:
        _record("opponent_low", active_hp=float(active_hp), opp_hp=float(opp_hp))
        return chosen_choice

    reply_score = float(self._estimate_best_reply_score(opponent, active, battle) or 0.0)
    if reply_score > self.RECOVERY_WINDOW_MAX_REPLY:
        _record(
            "unsafe_reply",
            active_hp=float(active_hp),
            opp_hp=float(opp_hp),
            reply_score=float(reply_score),
            max_reply=float(self.RECOVERY_WINDOW_MAX_REPLY),
        )
        return chosen_choice

    best_damage_score = float(self._estimate_best_damage_score(active, opponent, battle) or 0.0)
    ko_threshold = self.TACTICAL_KO_THRESHOLD * max(opp_hp, 0.05)
    if best_damage_score >= ko_threshold:
        _record(
            "ko_guard",
            active_hp=float(active_hp),
            opp_hp=float(opp_hp),
            reply_score=float(reply_score),
            best_damage_score=float(best_damage_score),
            ko_threshold=float(ko_threshold),
        )
        return chosen_choice

    chosen_weight = 0.0
    for choice, weight in ordered:
        if choice == chosen_choice:
            chosen_weight = float(weight or 0.0)
            break

    best_recovery_choice = ""
    best_recovery_weight = 0.0
    best_recovery_heur = 0.0
    for choice, weight in ordered:
        if choice.startswith("switch "):
            continue
        move_id = normalize_name(choice.replace("-tera", ""))
        selected_move = None
        for move in battle.available_moves or []:
            if normalize_name(getattr(move, "id", "")) == move_id:
                selected_move = move
                break
        if selected_move is None or not self._is_recovery_move(selected_move):
            continue
        heur = float(self._heuristic_action_score(battle, choice) or 0.0)
        if heur <= best_recovery_heur:
            continue
        best_recovery_choice = choice
        best_recovery_weight = float(weight or 0.0)
        best_recovery_heur = heur

    if not best_recovery_choice:
        _record(
            "no_recovery_candidate",
            active_hp=float(active_hp),
            opp_hp=float(opp_hp),
            reply_score=float(reply_score),
            best_damage_score=float(best_damage_score),
            ko_threshold=float(ko_threshold),
            chosen_weight=float(chosen_weight),
        )
        return chosen_choice

    if best_recovery_choice == chosen_choice:
        _record("chosen_recovery", active_hp=float(active_hp), opp_hp=float(opp_hp))
        return chosen_choice

    chosen_heur = float(self._heuristic_action_score(battle, chosen_choice) or 0.0)
    heur_gain = best_recovery_heur - chosen_heur
    if heur_gain < self.RECOVERY_WINDOW_MIN_HEUR_GAIN:
        _record(
            "heuristic_gain",
            active_hp=float(active_hp),
            opp_hp=float(opp_hp),
            reply_score=float(reply_score),
            chosen_weight=float(chosen_weight),
            recovery_choice=best_recovery_choice,
            recovery_weight=float(best_recovery_weight),
            chosen_heuristic=float(chosen_heur),
            recovery_heuristic=float(best_recovery_heur),
            min_heur_gain=float(self.RECOVERY_WINDOW_MIN_HEUR_GAIN),
        )
        return chosen_choice

    high_gain = heur_gain >= self.RECOVERY_WINDOW_HIGH_HEUR_GAIN
    min_ratio = (
        self.RECOVERY_WINDOW_HIGH_GAIN_MIN_POLICY_RATIO
        if high_gain
        else self.RECOVERY_WINDOW_MIN_POLICY_RATIO
    )
    critical_hp = active_hp <= float(getattr(self, "RECOVERY_WINDOW_CRITICAL_HP", 0.30))
    if critical_hp and high_gain:
        min_ratio = min(
            float(min_ratio),
            float(getattr(self, "RECOVERY_WINDOW_CRITICAL_MIN_POLICY_RATIO", min_ratio)),
        )
    if best_recovery_weight < chosen_weight * min_ratio:
        _record(
            "policy_ratio",
            active_hp=float(active_hp),
            opp_hp=float(opp_hp),
            reply_score=float(reply_score),
            chosen_weight=float(chosen_weight),
            recovery_choice=best_recovery_choice,
            recovery_weight=float(best_recovery_weight),
            chosen_heuristic=float(chosen_heur),
            recovery_heuristic=float(best_recovery_heur),
            high_gain=bool(high_gain),
            critical_hp=bool(critical_hp),
            min_policy_ratio=float(min_ratio),
        )
        return chosen_choice

    _record(
        "take_recovery",
        active_hp=float(active_hp),
        opp_hp=float(opp_hp),
        reply_score=float(reply_score),
        best_damage_score=float(best_damage_score),
        ko_threshold=float(ko_threshold),
        chosen_weight=float(chosen_weight),
        recovery_choice=best_recovery_choice,
        recovery_weight=float(best_recovery_weight),
        chosen_heuristic=float(chosen_heur),
        recovery_heuristic=float(best_recovery_heur),
        high_gain=bool(high_gain),
        critical_hp=bool(critical_hp),
    )
    return best_recovery_choice


def maybe_take_critical_recovery_choice(
    self,
    battle: Battle,
    ordered: List[Tuple[str, float]],
    chosen_choice: str,
) -> str:
    def _record(reason: str, **extra) -> None:
        try:
            mem = self._get_battle_memory(battle)
        except Exception:
            return
        if not isinstance(mem, dict):
            return
        payload = {
            "reason": reason,
            "chosen_choice": str(chosen_choice or ""),
        }
        payload.update(extra)
        mem["critical_recovery_last"] = payload
        mem["recovery_window_last"] = payload

    if not chosen_choice or getattr(battle, "force_switch", False):
        _record("critical_forced_or_empty")
        return chosen_choice

    active = battle.active_pokemon
    opponent = battle.opponent_active_pokemon
    if active is None or opponent is None:
        _record("critical_missing_active")
        return chosen_choice

    active_hp = active.current_hp_fraction or 0.0
    max_hp = float(getattr(self, "CRITICAL_RECOVERY_MAX_HP", 0.35))
    if active_hp > max_hp:
        _record("critical_high_hp", active_hp=float(active_hp), max_hp=float(max_hp))
        return chosen_choice

    opp_hp = opponent.current_hp_fraction or 0.0
    min_opp_hp = float(getattr(self, "CRITICAL_RECOVERY_MIN_OPP_HP", 0.30))
    if opp_hp <= min_opp_hp:
        _record("critical_opponent_low", active_hp=float(active_hp), opp_hp=float(opp_hp))
        return chosen_choice

    reply_score = float(self._estimate_best_reply_score(opponent, active, battle) or 0.0)
    max_reply = float(getattr(self, "CRITICAL_RECOVERY_MAX_REPLY", 100.0))
    if reply_score > max_reply:
        _record(
            "critical_unsafe_reply",
            active_hp=float(active_hp),
            opp_hp=float(opp_hp),
            reply_score=float(reply_score),
            max_reply=float(max_reply),
        )
        return chosen_choice

    best_damage_score = float(self._estimate_best_damage_score(active, opponent, battle) or 0.0)
    ko_threshold = self.TACTICAL_KO_THRESHOLD * max(opp_hp, 0.05)
    if best_damage_score >= ko_threshold:
        _record(
            "critical_ko_guard",
            active_hp=float(active_hp),
            opp_hp=float(opp_hp),
            reply_score=float(reply_score),
            best_damage_score=float(best_damage_score),
            ko_threshold=float(ko_threshold),
        )
        return chosen_choice

    weights = {choice: float(weight or 0.0) for choice, weight in ordered}
    chosen_weight = weights.get(chosen_choice, 0.0)
    chosen_heur = float(self._heuristic_action_score(battle, chosen_choice) or 0.0)

    allow_rest = bool(getattr(self, "CRITICAL_RECOVERY_ALLOW_REST", False))
    best_recovery_choice = ""
    best_recovery_weight = 0.0
    best_recovery_heur = 0.0
    for choice, weight in ordered:
        if choice.startswith("switch "):
            continue
        move_id = normalize_name(choice.replace("-tera", ""))
        if move_id == "rest" and not allow_rest:
            continue
        selected_move = None
        for move in battle.available_moves or []:
            if normalize_name(getattr(move, "id", "")) == move_id:
                selected_move = move
                break
        if selected_move is None or not self._is_recovery_move(selected_move):
            continue
        heur = float(self._heuristic_action_score(battle, choice) or 0.0)
        if heur <= best_recovery_heur:
            continue
        best_recovery_choice = choice
        best_recovery_weight = float(weight or 0.0)
        best_recovery_heur = heur

    if not best_recovery_choice:
        _record(
            "critical_no_recovery_candidate",
            active_hp=float(active_hp),
            opp_hp=float(opp_hp),
            reply_score=float(reply_score),
            best_damage_score=float(best_damage_score),
            ko_threshold=float(ko_threshold),
            chosen_weight=float(chosen_weight),
        )
        return chosen_choice

    if best_recovery_choice == chosen_choice:
        _record("critical_chosen_recovery", active_hp=float(active_hp), opp_hp=float(opp_hp))
        return chosen_choice

    min_heuristic = float(getattr(self, "CRITICAL_RECOVERY_MIN_HEURISTIC", 110.0))
    heur_gain = best_recovery_heur - chosen_heur
    min_heur_gain = float(getattr(self, "CRITICAL_RECOVERY_MIN_HEUR_GAIN", 85.0))
    if best_recovery_heur < min_heuristic or heur_gain < min_heur_gain:
        _record(
            "critical_heuristic_gain",
            active_hp=float(active_hp),
            opp_hp=float(opp_hp),
            reply_score=float(reply_score),
            chosen_weight=float(chosen_weight),
            recovery_choice=best_recovery_choice,
            recovery_weight=float(best_recovery_weight),
            chosen_heuristic=float(chosen_heur),
            recovery_heuristic=float(best_recovery_heur),
            min_heuristic=float(min_heuristic),
            min_heur_gain=float(min_heur_gain),
        )
        return chosen_choice

    score_drop = max(0.0, chosen_weight - best_recovery_weight)
    max_score_drop = float(getattr(self, "CRITICAL_RECOVERY_MAX_SCORE_DROP", 0.12))
    min_ratio = float(getattr(self, "CRITICAL_RECOVERY_MIN_POLICY_RATIO", 0.40))
    if score_drop > max_score_drop or best_recovery_weight < chosen_weight * min_ratio:
        _record(
            "critical_policy_ratio",
            active_hp=float(active_hp),
            opp_hp=float(opp_hp),
            reply_score=float(reply_score),
            chosen_weight=float(chosen_weight),
            recovery_choice=best_recovery_choice,
            recovery_weight=float(best_recovery_weight),
            chosen_heuristic=float(chosen_heur),
            recovery_heuristic=float(best_recovery_heur),
            score_drop=float(score_drop),
            max_score_drop=float(max_score_drop),
            min_policy_ratio=float(min_ratio),
        )
        return chosen_choice

    _record(
        "take_critical_recovery",
        active_hp=float(active_hp),
        opp_hp=float(opp_hp),
        reply_score=float(reply_score),
        best_damage_score=float(best_damage_score),
        ko_threshold=float(ko_threshold),
        chosen_weight=float(chosen_weight),
        recovery_choice=best_recovery_choice,
        recovery_weight=float(best_recovery_weight),
        chosen_heuristic=float(chosen_heur),
        recovery_heuristic=float(best_recovery_heur),
        score_drop=float(score_drop),
    )
    return best_recovery_choice


def aggregate_policy_from_results(
    self,
    results: List[Tuple[object, float]],
    banned_choices: Optional[set] = None,
) -> Tuple[List[Tuple[str, float]], float, float, float]:
    final_policy: Dict[str, float] = {}
    for res, weight in results:
        total_visits = res.total_visits or 1
        for opt in res.side_one:
            final_policy[opt.move_choice] = final_policy.get(opt.move_choice, 0.0) + (
                weight * (opt.visits / total_visits)
            )
    if banned_choices:
        filtered_policy = {k: v for k, v in final_policy.items() if k not in banned_choices}
        if filtered_policy:
            final_policy = filtered_policy
    ordered = sorted(final_policy.items(), key=lambda x: x[1], reverse=True)
    total_policy = sum(w for _, w in ordered)
    confidence = 0.0
    if ordered and total_policy > 0:
        best = ordered[0][1]
        confidence = best / total_policy
        if len(ordered) > 1:
            second = ordered[1][1]
            margin = (best - second) / total_policy
            confidence = max(confidence, margin)
    threshold = max(0.0, min(1.0, self.MCTS_CONFIDENCE_THRESHOLD))
    return ordered, total_policy, confidence, threshold


def select_move_from_results(
    self,
    results: List[Tuple[object, float]],
    battle: Battle,
    banned_choices: Optional[set] = None,
    world_candidates: Optional[List[dict]] = None,
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
    fp_close_top_selection = bool(getattr(self, "FP_CLOSE_TOP_SELECTION", False))
    if deterministic and not fp_close_top_selection:
        self._mcts_stats["deterministic_decisions"] += 1
    else:
        self._mcts_stats["stochastic_decisions"] += 1
    ordered, total_policy, confidence, threshold = self._aggregate_policy_from_results(
        results,
        banned_choices=banned_choices,
    )
    if not ordered:
        return ""

    def _return_choice(chosen_choice: str, path: str) -> str:
        tactical_reranks_enabled = bool(getattr(self, "TACTICAL_RERANKS_ENABLED", True))
        finish_blow_guard_enabled = bool(getattr(self, "FINISH_BLOW_GUARD_ENABLED", False))

        def _apply_finish_blow_guard(current_choice: str, current_path: str) -> tuple[str, str]:
            adjusted_choice = self._maybe_force_finish_blow_choice(
                battle,
                ordered,
                current_choice,
            )
            adjusted_choice = self._maybe_accept_rerank_choice(
                battle,
                ordered,
                current_choice,
                adjusted_choice,
                confidence,
                threshold,
            )
            if adjusted_choice != current_choice:
                current_choice = adjusted_choice
                current_path = "rerank" if current_path == "mcts" else current_path
            return current_choice, current_path

        def _apply_rerank_candidate(current_choice: str, current_path: str, candidate_fn) -> tuple[str, str]:
            adjusted_choice = candidate_fn(battle, ordered, current_choice)
            adjusted_choice = self._maybe_accept_rerank_choice(
                battle,
                ordered,
                current_choice,
                adjusted_choice,
                confidence,
                threshold,
            )
            if adjusted_choice != current_choice:
                current_choice = adjusted_choice
                current_path = "rerank" if current_path == "mcts" else current_path
            return current_choice, current_path

        def _record_shadow_windows(current_choice: str) -> None:
            if not bool(getattr(self, "TACTICAL_SHADOW_WINDOWS_ENABLED", False)):
                return
            if not current_choice:
                return
            if not bool(getattr(self, "SETUP_WINDOW_ENABLED", tactical_reranks_enabled)):
                self._maybe_take_setup_window_choice(battle, ordered, current_choice)
            if not bool(getattr(self, "RECOVERY_WINDOW_ENABLED", tactical_reranks_enabled)):
                self._maybe_take_safe_recovery_choice(battle, ordered, current_choice)
            if not bool(getattr(self, "PROGRESS_WINDOW_ENABLED", tactical_reranks_enabled)):
                self._maybe_take_progress_when_behind_choice(battle, ordered, current_choice)
            if not bool(getattr(self, "SWITCH_GUARD_ENABLED", tactical_reranks_enabled)):
                self._maybe_reduce_negative_matchup_switch(battle, ordered, current_choice)

        if chosen_choice and finish_blow_guard_enabled:
            chosen_choice, path = _apply_finish_blow_guard(chosen_choice, path)
        if chosen_choice and bool(getattr(self, "CRITICAL_RECOVERY_GUARD_ENABLED", False)):
            chosen_choice, path = _apply_rerank_candidate(
                chosen_choice,
                path,
                self._maybe_take_critical_recovery_choice,
            )
        if chosen_choice and bool(getattr(self, "SETUP_WINDOW_ENABLED", tactical_reranks_enabled)):
            chosen_choice, path = _apply_rerank_candidate(
                chosen_choice,
                path,
                self._maybe_take_setup_window_choice,
            )
        if chosen_choice and bool(getattr(self, "RECOVERY_WINDOW_ENABLED", tactical_reranks_enabled)):
            chosen_choice, path = _apply_rerank_candidate(
                chosen_choice,
                path,
                self._maybe_take_safe_recovery_choice,
            )
        if chosen_choice and bool(getattr(self, "PROGRESS_WINDOW_ENABLED", tactical_reranks_enabled)):
            chosen_choice, path = _apply_rerank_candidate(
                chosen_choice,
                path,
                self._maybe_take_progress_when_behind_choice,
            )
        if chosen_choice and bool(getattr(self, "SWITCH_GUARD_ENABLED", tactical_reranks_enabled)):
            chosen_choice, path = _apply_rerank_candidate(
                chosen_choice,
                path,
                self._maybe_reduce_negative_matchup_switch,
            )
        if chosen_choice and finish_blow_guard_enabled:
            chosen_choice, path = _apply_finish_blow_guard(chosen_choice, path)
        if chosen_choice and bool(getattr(self, "LATEGAME_ATTACK_GUARD_ENABLED", False)):
            adjusted_choice = self._maybe_commit_late_game_attack_choice(
                battle,
                ordered,
                chosen_choice,
            )
            if adjusted_choice != chosen_choice:
                chosen_choice = adjusted_choice
                path = "late_attack"
        if chosen_choice:
            _record_shadow_windows(chosen_choice)
            self._diag_record_choice(
                battle,
                ordered,
                chosen_choice,
                confidence,
                threshold,
                path,
            )
            self._append_search_trace_example(
                battle,
                ordered,
                chosen_choice,
                confidence,
                threshold,
                path,
                world_candidates=world_candidates,
            )
        return chosen_choice

    if total_policy <= 0:
        return _return_choice(ordered[0][0], "mcts")

    def _pick_choice(choices: List[str], weights: List[float]) -> str:
        if not choices:
            return ""
        total = sum(weights) if weights else 0.0
        if fp_close_top_selection and len(choices) > 1:
            self._mcts_stats["fp_close_top_used"] = int(
                self._mcts_stats.get("fp_close_top_used", 0) or 0
            ) + 1
            if total <= 0:
                choice = random.choice(choices)
            else:
                choice = random.choices(choices, weights=weights, k=1)[0]
            if choice != choices[0]:
                self._mcts_stats["fp_close_top_non_top1"] = int(
                    self._mcts_stats.get("fp_close_top_non_top1", 0) or 0
                ) + 1
            return choice
        if deterministic:
            if total <= 0:
                return choices[0]
            best_idx = max(range(len(choices)), key=lambda i: weights[i])
            return choices[best_idx]
        if total <= 0:
            return random.choice(choices)
        return random.choices(choices, weights=weights, k=1)[0]

    best = ordered[0][1]

    close_top_ratio = max(0.0, min(1.0, float(getattr(self, "FP_CLOSE_TOP_RATIO", 0.75))))
    cutoff = best * close_top_ratio
    filtered = [o for o in ordered if o[1] >= cutoff]
    filtered = self._apply_switch_prior_prune(battle, filtered, confidence, threshold)
    filtered = self._apply_tera_prune(battle, filtered, confidence, threshold)
    if not filtered:
        filtered = [ordered[0]]
    if getattr(self, "PASSIVE_BREAKER_ENABLED", False):
        passive_break_choice = self._maybe_passive_break_choice(battle, filtered, confidence, threshold)
        if passive_break_choice:
            return _return_choice(passive_break_choice, "rerank")

    if self._should_trigger_adaptive_fallback(battle, ordered, confidence, threshold):
        mem = self._get_battle_memory(battle)
        mem["adaptive_fallback_last_turn"] = int(getattr(battle, "turn", 0) or 0)
        self._mcts_stats["adaptive_triggered"] += 1
        mode = (self.ADAPTIVE_FALLBACK_MODE or "super").strip().lower()
        if mode in {"rerank", "risk", "adaptive"}:
            reranked = self._adaptive_rerank_choice(
                battle,
                ordered,
                topk=max(1, self.ADAPTIVE_FALLBACK_TOPK),
            )
            if reranked:
                self._mcts_stats["adaptive_rerank_used"] += 1
                if self.DECISION_DIAG_ENABLED:
                    mem["diag_adaptive_triggered"] = int(mem.get("diag_adaptive_triggered", 0) or 0) + 1
                return _return_choice(reranked, "adaptive_rerank")
            self._mcts_stats["adaptive_rerank_failed"] += 1
            return _return_choice(ordered[0][0], "mcts")
        mem["adaptive_fallback_pending"] = 1
        if self.DECISION_DIAG_ENABLED:
            mem["diag_adaptive_triggered"] = int(mem.get("diag_adaptive_triggered", 0) or 0) + 1
        return ""
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
                return _return_choice(_pick_choice(choices, heuristic_weights), "policy")
        return _return_choice(_pick_choice(choices, policy_weights), "policy")

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
            return _return_choice(scored[0][2], "rerank")

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

    selection_path = "blend"
    applied_heuristic_blend = False
    applied_prior_blend = False
    if blend <= 0:
        combined = mcts_norm
        selection_path = "mcts"
    else:
        heuristic_weights = []
        for choice in choices:
            score = self._heuristic_action_score(battle, choice)
            heuristic_weights.append(max(0.0, score or 0.0))
        heur_total = sum(heuristic_weights)
        if heur_total <= 0:
            combined = mcts_norm
            selection_path = "mcts"
        else:
            heur_norm = [w / heur_total for w in heuristic_weights]
            combined = [
                (1.0 - blend) * m + blend * h for m, h in zip(mcts_norm, heur_norm)
            ]
            applied_heuristic_blend = True

    rl_blend = max(0.0, min(1.0, self.RL_PRIOR_BLEND))
    if rl_blend > 0.0 and (not self.RL_PRIOR_LOWCONF_ONLY or confidence < threshold):
        rl_priors = self._rl_choice_priors(battle, choices)
        if rl_priors:
            rl_total = sum(rl_priors)
            if rl_total > 0:
                rl_norm = [w / rl_total for w in rl_priors]
                combined = [
                    (1.0 - rl_blend) * base + rl_blend * rl
                    for base, rl in zip(combined, rl_norm)
                ]
                applied_prior_blend = True

    search_prior_blend = max(0.0, min(1.0, self.SEARCH_PRIOR_BLEND))
    if search_prior_blend > 0.0 and (not self.SEARCH_PRIOR_LOWCONF_ONLY or confidence < threshold):
        search_priors = self._search_choice_priors(battle, choices)
        if search_priors:
            prior_total = sum(search_priors)
            if prior_total > 0:
                prior_norm = [w / prior_total for w in search_priors]
                combined = [
                    (1.0 - search_prior_blend) * base + search_prior_blend * prior
                    for base, prior in zip(combined, prior_norm)
                ]
                applied_prior_blend = True

    if applied_prior_blend and not applied_heuristic_blend:
        selection_path = "policy"
    elif applied_prior_blend or applied_heuristic_blend:
        selection_path = "blend"
    else:
        selection_path = "mcts"

    return _return_choice(_pick_choice(choices, combined), selection_path)


def is_damaging_move_choice(self, battle: Battle, choice: str) -> bool:
    move_id = normalize_name(choice.replace("-tera", ""))
    for move in battle.available_moves or []:
        if normalize_name(getattr(move, "id", "")) != move_id:
            continue
        category = getattr(move, "category", None)
        base_power = float(getattr(move, "base_power", 0) or 0)
        damage_attr = getattr(move, "damage", None)
        if category == MoveCategory.STATUS:
            return False
        if base_power > 0:
            return True
        if category is not None and category != MoveCategory.STATUS:
            return True
        return bool(damage_attr not in (None, 0, "0", False))
    return False


def choose_adaptive_fallback_order(self, battle: Battle, active: Pokemon, opponent: Pokemon):
    if battle.force_switch:
        if battle.available_switches:
            best_switch = max(
                battle.available_switches,
                key=lambda s: self._score_switch(s, opponent, battle),
            )
            return self.create_order(best_switch)
        return None

    move_candidates: List[Tuple[str, float]] = []
    for move in battle.available_moves or []:
        choice = normalize_name(getattr(move, "id", ""))
        if not choice:
            continue
        score = self._heuristic_action_score(battle, choice)
        move_candidates.append((choice, float(score or 0.0)))
        if getattr(battle, "can_tera", False):
            tera_choice = f"{choice}-tera"
            tera_score = self._heuristic_action_score(battle, tera_choice)
            move_candidates.append((tera_choice, float(tera_score or 0.0)))

    switch_candidates: List[Tuple[str, float]] = []
    for sw in battle.available_switches or []:
        sw_choice = f"switch {normalize_name(sw.species)}"
        score = self._heuristic_action_score(battle, sw_choice)
        switch_candidates.append((sw_choice, float(score or 0.0)))

    if not move_candidates and not switch_candidates:
        return None

    selected: Optional[str] = None
    if move_candidates:
        move_candidates.sort(key=lambda x: x[1], reverse=True)
        best_move, best_score = move_candidates[0]
        if best_score > 0:
            damaging = [
                (c, s)
                for c, s in move_candidates
                if self._is_damaging_move_choice(battle, c) and s >= best_score * 0.85
            ]
            if damaging:
                damaging.sort(key=lambda x: x[1], reverse=True)
                selected = damaging[0][0]
            else:
                selected = best_move

    if selected is None and switch_candidates:
        switch_candidates.sort(key=lambda x: x[1], reverse=True)
        selected = switch_candidates[0][0]

    if not selected:
        return None
    if selected.startswith("switch "):
        switch_name = normalize_name(selected.split("switch ", 1)[1])
        for sw in battle.available_switches or []:
            if normalize_name(sw.species) == switch_name:
                return self.create_order(sw)
        return None

    tera = False
    if selected.endswith("-tera"):
        selected = selected.replace("-tera", "")
        tera = bool(getattr(battle, "can_tera", False))
    move_id = normalize_name(selected)
    for move in battle.available_moves or []:
        if normalize_name(getattr(move, "id", "")) == move_id:
            return self.create_order(move, terastallize=tera)
    return None
