from __future__ import annotations

import math
import os
import random
from typing import Dict, List, Optional, Tuple

from poke_env.battle import Battle, Pokemon, MoveCategory

from src.utils.damage_calc import normalize_name


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
    if not chosen_choice or not chosen_choice.startswith("switch ") or getattr(battle, "force_switch", False):
        return chosen_choice

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
        return chosen_choice

    switch_heur = float(self._heuristic_action_score(battle, chosen_choice) or 0.0)
    move_heur = float(self._heuristic_action_score(battle, best_move_choice) or 0.0)
    switch_risk = float(self._adaptive_choice_risk_penalty(battle, chosen_choice) or 0.0)
    if switch_weight <= 0.0:
        weight_ratio = 1.0 if best_move_weight > 0.0 else 0.0
    else:
        weight_ratio = best_move_weight / max(switch_weight, 1e-6)

    if weight_ratio >= 0.75 and move_heur >= switch_heur + 10.0:
        return best_move_choice
    if weight_ratio >= 0.95 and move_heur >= switch_heur - 5.0 and move_heur > 0.0:
        return best_move_choice
    if switch_risk >= 35.0 and weight_ratio >= 0.60 and move_heur >= max(0.0, switch_heur - 10.0):
        return best_move_choice
    return chosen_choice


def maybe_force_finish_blow_choice(
    self,
    battle: Battle,
    ordered: List[Tuple[str, float]],
    chosen_choice: str,
) -> str:
    if not chosen_choice or getattr(battle, "force_switch", False):
        return chosen_choice
    if self._is_damaging_move_choice(battle, chosen_choice):
        return chosen_choice

    active = battle.active_pokemon
    opponent = battle.opponent_active_pokemon
    if active is None or opponent is None:
        return chosen_choice

    opp_hp = opponent.current_hp_fraction or 0.0
    best_damage_score = float(self._estimate_best_damage_score(active, opponent, battle) or 0.0)
    ko_threshold = self.TACTICAL_KO_THRESHOLD * max(opp_hp, 0.05)
    if best_damage_score < ko_threshold:
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
        return chosen_choice
    if chosen_choice.startswith("switch "):
        return best_damage_choice
    chosen_heur = float(self._heuristic_action_score(battle, chosen_choice) or 0.0)
    if best_damage_heur > 0.0 and best_damage_heur + 5.0 >= chosen_heur:
        return best_damage_choice
    return chosen_choice


def maybe_take_setup_window_choice(
    self,
    battle: Battle,
    ordered: List[Tuple[str, float]],
    chosen_choice: str,
) -> str:
    if not chosen_choice or getattr(battle, "force_switch", False):
        return chosen_choice
    if chosen_choice.startswith("switch ") or not self._is_damaging_move_choice(battle, chosen_choice):
        return chosen_choice

    active = battle.active_pokemon
    opponent = battle.opponent_active_pokemon
    if active is None or opponent is None:
        return chosen_choice

    active_hp = active.current_hp_fraction or 0.0
    if active_hp < self.SETUP_WINDOW_MIN_HP:
        return chosen_choice

    reply_score = float(self._estimate_best_reply_score(opponent, active, battle) or 0.0)
    if reply_score > self.SETUP_WINDOW_MAX_REPLY:
        return chosen_choice

    opp_hp = opponent.current_hp_fraction or 0.0
    best_damage_score = float(self._estimate_best_damage_score(active, opponent, battle) or 0.0)
    ko_threshold = self.TACTICAL_KO_THRESHOLD * max(opp_hp, 0.05)
    if best_damage_score >= ko_threshold:
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
        return chosen_choice

    chosen_heur = float(self._heuristic_action_score(battle, chosen_choice) or 0.0)
    if best_setup_heur < chosen_heur + self.SETUP_WINDOW_MIN_HEUR_GAIN:
        return chosen_choice
    if best_setup_weight < chosen_weight * self.SETUP_WINDOW_MIN_POLICY_RATIO:
        return chosen_choice
    return best_setup_choice


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
    if deterministic:
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
        if chosen_choice:
            adjusted_choice = self._maybe_force_finish_blow_choice(
                battle,
                ordered,
                chosen_choice,
            )
            if adjusted_choice != chosen_choice:
                chosen_choice = adjusted_choice
                path = "rerank" if path == "mcts" else path
            adjusted_choice = self._maybe_take_setup_window_choice(
                battle,
                ordered,
                chosen_choice,
            )
            if adjusted_choice != chosen_choice:
                chosen_choice = adjusted_choice
                path = "rerank" if path == "mcts" else path
            adjusted_choice = self._maybe_reduce_negative_matchup_switch(
                battle,
                ordered,
                chosen_choice,
            )
            if adjusted_choice != chosen_choice:
                chosen_choice = adjusted_choice
                path = "rerank" if path == "mcts" else path
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
        if deterministic:
            if total <= 0:
                return choices[0]
            best_idx = max(range(len(choices)), key=lambda i: weights[i])
            return choices[best_idx]
        if total <= 0:
            return random.choice(choices)
        return random.choices(choices, weights=weights, k=1)[0]

    best = ordered[0][1]

    cutoff = best * 0.75
    filtered = [o for o in ordered if o[1] >= cutoff]
    filtered = self._apply_switch_prior_prune(battle, filtered, confidence, threshold)
    filtered = self._apply_tera_prune(battle, filtered, confidence, threshold)
    if not filtered:
        filtered = [ordered[0]]
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
        return bool(
            base_power > 0
            or damage_attr is not None
            or (category is not None and category != MoveCategory.STATUS)
        )
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
