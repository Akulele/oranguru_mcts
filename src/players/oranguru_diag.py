#!/usr/bin/env python3
"""
Diagnostics and MCTS-stats helpers for OranguruEnginePlayer.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from poke_env.battle import Battle


def init_mcts_stats() -> Dict[str, float]:
    return {
        "calls": 0,
        "sample_states_requested_total": 0,
        "sample_states_budgeted_total": 0,
        "worlds_generated_total": 0,
        "worlds_searched_total": 0,
        "states_sampled": 0,
        "results_kept": 0,
        "result_none": 0,
        "result_errors": 0,
        "empty_results": 0,
        "world_keep_rate_sum": 0.0,
        "low_uncertainty_turns": 0,
        "low_uncertainty_worlds_saved": 0,
        "endgame_reduction_turns": 0,
        "endgame_worlds_saved": 0,
        "deterministic_decisions": 0,
        "stochastic_decisions": 0,
        "fallback_super": 0,
        "fallback_random": 0,
        "search_prior_used": 0,
        "search_prior_calls": 0,
        "search_prior_mask_empty": 0,
        "search_prior_unmapped_choices": 0,
        "search_prior_zero_sum": 0,
        "search_prior_init_failed": 0,
        "search_prior_apply_failed": 0,
        "switch_prior_used": 0,
        "switch_prior_pruned": 0,
        "passive_breaker_used": 0,
        "tera_pruner_used": 0,
        "tera_pruner_pruned": 0,
        "world_ranker_used": 0,
        "world_ranker_pruned": 0,
        "leaf_value_used": 0,
        "leaf_value_escalated": 0,
        "leaf_value_escalate_failed": 0,
        "leaf_value_pred_sum": 0.0,
        "leaf_value_pred_count": 0,
        "adaptive_triggered": 0,
        "adaptive_heuristic_used": 0,
        "adaptive_heuristic_failed": 0,
        "adaptive_super_used": 0,
        "adaptive_rerank_used": 0,
        "adaptive_rerank_failed": 0,
        "adaptive_second_pass_used": 0,
        "adaptive_second_pass_failed": 0,
        "diag_turns": 0,
        "diag_low_conf_turns": 0,
        "diag_low_margin_turns": 0,
        "diag_non_top1_choices": 0,
        "diag_choice_delta_sum": 0.0,
        "diag_move_choices": 0,
        "diag_switch_choices": 0,
        "diag_tera_choices": 0,
        "diag_forced_switch_turns": 0,
        "diag_hazard_switch_choices": 0,
        "diag_passive_no_progress_turns": 0,
        "diag_path_mcts": 0,
        "diag_path_adaptive_rerank": 0,
        "diag_path_policy": 0,
        "diag_path_rerank": 0,
        "diag_path_blend": 0,
        "diag_path_fallback_super": 0,
        "diag_path_fallback_random": 0,
        "diag_adaptive_reason_triggered": 0,
        "diag_adaptive_reason_disabled": 0,
        "diag_adaptive_reason_empty": 0,
        "diag_adaptive_reason_early_turn": 0,
        "diag_adaptive_reason_cooldown": 0,
        "diag_adaptive_reason_confidence": 0,
        "diag_adaptive_reason_top_ratio": 0,
        "diag_adaptive_reason_no_heuristics": 0,
        "diag_adaptive_reason_high_heuristic": 0,
        "diag_adaptive_reason_damaging_available": 0,
        "diag_adaptive_reason_not_stallish": 0,
        "diag_battles_finished": 0,
        "diag_battles_won": 0,
        "diag_battles_lost": 0,
        "diag_loss_fast": 0,
        "diag_loss_low_conf": 0,
        "diag_loss_switch_heavy": 0,
        "diag_loss_status_loop": 0,
        "diag_loss_forced_switch": 0,
        "diag_loss_passive": 0,
        "diag_loss_hazard_pivot": 0,
        "diag_loss_tempo": 0,
        "diag_loss_adaptive_used": 0,
        "diag_loss_churn_breaks": 0,
        "diag_loss_other": 0,
    }


def get_mcts_stats(self) -> Dict[str, float]:
    stats = dict(self._mcts_stats)
    calls = max(1, int(stats.get("calls", 0)))
    sampled = max(1, int(stats.get("states_sampled", 0)))
    generated = max(1, int(stats.get("worlds_generated_total", sampled)))
    searched = max(1, int(stats.get("worlds_searched_total", sampled)))
    stats["empty_results_rate"] = float(stats.get("empty_results", 0)) / calls
    stats["fallback_super_rate"] = float(stats.get("fallback_super", 0)) / calls
    stats["fallback_random_rate"] = float(stats.get("fallback_random", 0)) / calls
    stats["state_failure_rate"] = float(
        stats.get("result_none", 0) + stats.get("result_errors", 0)
    ) / sampled
    stats["world_generated_rate"] = float(generated) / calls
    stats["world_search_rate"] = float(searched) / calls
    stats["world_keep_rate"] = float(searched) / generated
    stats["avg_requested_worlds_per_call"] = float(stats.get("sample_states_requested_total", 0)) / calls
    stats["avg_budgeted_worlds_per_call"] = float(stats.get("sample_states_budgeted_total", 0)) / calls
    stats["avg_generated_worlds_per_call"] = float(stats.get("worlds_generated_total", 0)) / calls
    stats["avg_kept_worlds_per_call"] = float(stats.get("worlds_searched_total", 0)) / calls
    stats["avg_leaf_value_pred"] = float(stats.get("leaf_value_pred_sum", 0.0)) / max(
        1, int(stats.get("leaf_value_pred_count", 0))
    )
    diag_turns = max(1, int(stats.get("diag_turns", 0)))
    stats["diag_low_conf_rate"] = float(stats.get("diag_low_conf_turns", 0)) / diag_turns
    stats["diag_low_margin_rate"] = float(stats.get("diag_low_margin_turns", 0)) / diag_turns
    stats["diag_non_top1_rate"] = float(stats.get("diag_non_top1_choices", 0)) / diag_turns
    stats["diag_switch_rate"] = float(stats.get("diag_switch_choices", 0)) / diag_turns
    stats["diag_choice_delta_avg"] = float(stats.get("diag_choice_delta_sum", 0.0)) / diag_turns
    return stats


def diag_record_adaptive_reason(self, reason: str) -> None:
    if not self.DECISION_DIAG_ENABLED:
        return
    if not reason:
        return
    key = f"diag_adaptive_reason_{reason}"
    self._mcts_stats[key] = int(self._mcts_stats.get(key, 0) or 0) + 1


def diag_record_choice(
    self,
    battle: Battle,
    ordered: List[Tuple[str, float]],
    chosen: str,
    confidence: float,
    threshold: float,
    path: str,
) -> None:
    if not self.DECISION_DIAG_ENABLED:
        return
    mem = self._get_battle_memory(battle)
    self._mcts_stats["diag_turns"] += 1
    mem["diag_turns"] = int(mem.get("diag_turns", 0) or 0) + 1
    mem["diag_status_stall_peak"] = max(
        int(mem.get("diag_status_stall_peak", 0) or 0),
        int(mem.get("status_stall_streak", 0) or 0),
    )

    if confidence < threshold:
        self._mcts_stats["diag_low_conf_turns"] += 1
        mem["diag_low_conf_turns"] = int(mem.get("diag_low_conf_turns", 0) or 0) + 1

    if ordered:
        total = sum(max(0.0, float(w)) for _, w in ordered)
        best_choice = ordered[0][0]
        best_weight = max(0.0, float(ordered[0][1]))
        second_weight = max(0.0, float(ordered[1][1])) if len(ordered) > 1 else 0.0
        margin = ((best_weight - second_weight) / total) if total > 0 else 0.0
        if margin < max(0.0, self.DECISION_DIAG_LOW_MARGIN):
            self._mcts_stats["diag_low_margin_turns"] += 1
            mem["diag_low_margin_turns"] = int(mem.get("diag_low_margin_turns", 0) or 0) + 1
        if chosen and chosen != best_choice:
            self._mcts_stats["diag_non_top1_choices"] += 1
            mem["diag_non_top1_choices"] = int(mem.get("diag_non_top1_choices", 0) or 0) + 1
            best_prob = (best_weight / total) if total > 0 else 0.0
            chosen_weight = 0.0
            for c, w in ordered:
                if c == chosen:
                    chosen_weight = max(0.0, float(w))
                    break
            chosen_prob = (chosen_weight / total) if total > 0 else 0.0
            delta = max(0.0, best_prob - chosen_prob)
            self._mcts_stats["diag_choice_delta_sum"] += delta
            mem["diag_choice_delta_sum"] = float(mem.get("diag_choice_delta_sum", 0.0) or 0.0) + delta

    if chosen.startswith("switch "):
        self._mcts_stats["diag_switch_choices"] += 1
        mem["diag_switch_choices"] = int(mem.get("diag_switch_choices", 0) or 0) + 1
        if self._side_hazard_pressure(battle) > 0:
            self._mcts_stats["diag_hazard_switch_choices"] += 1
            mem["diag_hazard_switch_choices"] = int(mem.get("diag_hazard_switch_choices", 0) or 0) + 1
    elif chosen:
        self._mcts_stats["diag_move_choices"] += 1
        mem["diag_move_choices"] = int(mem.get("diag_move_choices", 0) or 0) + 1
    if chosen.endswith("-tera"):
        self._mcts_stats["diag_tera_choices"] += 1
        mem["diag_tera_choices"] = int(mem.get("diag_tera_choices", 0) or 0) + 1

    path_key = f"diag_path_{path}"
    self._mcts_stats[path_key] = int(self._mcts_stats.get(path_key, 0) or 0) + 1
    mem[path_key] = int(mem.get(path_key, 0) or 0) + 1

    if self.DECISION_DIAG_ENABLED and self.DECISION_DIAG_LOG and ordered:
        topk = max(1, self.DECISION_DIAG_TOPK)
        head = ", ".join(f"{c}:{w:.3f}" for c, w in ordered[:topk])
        print(
            "[diag] turn={} conf={:.3f}/{:.3f} path={} chosen={} top={}".format(
                int(getattr(battle, "turn", 0) or 0),
                float(confidence),
                float(threshold),
                path,
                chosen or "<none>",
                head,
            )
        )


def flush_finished_battle_diags(self) -> None:
    if not self.DECISION_DIAG_ENABLED:
        return
    battles = getattr(self, "battles", {}) or {}
    if not battles:
        return
    battle_memory = getattr(self, "_battle_memory", {}) or {}
    for tag, battle in battles.items():
        if tag in self._diag_finished_battle_tags:
            continue
        if not getattr(battle, "finished", False):
            continue
        self._diag_finished_battle_tags.add(tag)
        self._mcts_stats["diag_battles_finished"] += 1
        won = bool(getattr(battle, "won", False))
        lost = bool(getattr(battle, "lost", False))
        if won:
            self._mcts_stats["diag_battles_won"] += 1
        elif lost:
            self._mcts_stats["diag_battles_lost"] += 1

        mem = battle_memory.get(tag, {}) if isinstance(battle_memory, dict) else {}
        if not isinstance(mem, dict):
            mem = {}
        if not lost:
            continue

        tags = 0
        turns = int(getattr(battle, "turn", 0) or 0)
        if 0 < turns <= 12:
            self._mcts_stats["diag_loss_fast"] += 1
            tags += 1

        diag_turns = int(mem.get("diag_turns", 0) or 0)
        low_conf = int(mem.get("diag_low_conf_turns", 0) or 0)
        switch_turns = int(mem.get("diag_switch_choices", 0) or 0)
        low_margin = int(mem.get("diag_low_margin_turns", 0) or 0)
        hazard_switch_turns = int(mem.get("diag_hazard_switch_choices", 0) or 0)
        passive_turns = int(mem.get("diag_passive_no_progress_turns", 0) or 0)
        if diag_turns > 0 and (low_conf / diag_turns) >= 0.5:
            self._mcts_stats["diag_loss_low_conf"] += 1
            tags += 1
        if diag_turns > 0 and (switch_turns / diag_turns) >= 0.45:
            self._mcts_stats["diag_loss_switch_heavy"] += 1
            tags += 1

        passive_ratio = (passive_turns / max(1, diag_turns)) if diag_turns > 0 else 0.0
        if (
            int(mem.get("diag_status_stall_peak", 0) or 0) >= 2
            or passive_ratio >= max(0.0, self.LOSS_PASSIVE_RATIO)
        ):
            self._mcts_stats["diag_loss_status_loop"] += 1
            self._mcts_stats["diag_loss_passive"] += 1
            tags += 1
        forced_turns = int(mem.get("diag_forced_switch_turns", 0) or 0)
        forced_ratio = (forced_turns / max(1, diag_turns)) if diag_turns > 0 else 0.0
        if (
            turns <= 24
            and forced_turns >= 4
            and forced_ratio >= max(0.0, min(1.0, self.LOSS_FORCED_SWITCH_RATIO))
        ):
            self._mcts_stats["diag_loss_forced_switch"] += 1
            tags += 1
        hazard_switch_ratio = (
            hazard_switch_turns / max(1, diag_turns) if diag_turns > 0 else 0.0
        )
        if (
            diag_turns > 0
            and hazard_switch_ratio >= max(0.0, self.LOSS_HAZARD_SWITCH_RATIO)
            and (switch_turns / max(1, diag_turns)) >= 0.25
        ):
            self._mcts_stats["diag_loss_hazard_pivot"] += 1
            tags += 1
        tempo_ratio = max(low_conf, low_margin) / max(1, diag_turns)
        if (
            tempo_ratio >= max(0.0, self.LOSS_TEMPO_RATIO)
            and passive_ratio < max(0.0, self.LOSS_PASSIVE_RATIO)
        ):
            self._mcts_stats["diag_loss_tempo"] += 1
            tags += 1
        if int(mem.get("adaptive_fallback_pending", 0) or 0) == 1 or int(mem.get("diag_adaptive_triggered", 0) or 0) > 0:
            self._mcts_stats["diag_loss_adaptive_used"] += 1
            tags += 1
        if int(mem.get("switch_churn_breaks", 0) or 0) > 0:
            self._mcts_stats["diag_loss_churn_breaks"] += 1
            tags += 1
        if tags == 0:
            self._mcts_stats["diag_loss_other"] += 1
