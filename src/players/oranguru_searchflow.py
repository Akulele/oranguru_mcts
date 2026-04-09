from __future__ import annotations

from typing import Tuple

from poke_env.battle import AbstractBattle, Battle

from src.players.rule_bot import RuleBotPlayer
from src.utils.damage_calc import normalize_name


def _prepare_turn_state(self, battle: Battle):
    self._flush_finished_battle_diags()
    self._flush_finished_search_traces()
    noop_order = self._empty_order_if_no_choices(battle)
    if noop_order is not None:
        return noop_order, None, None, None
    if getattr(battle, "_wait", False) or getattr(battle, "teampreview", False):
        return self.choose_random_move(battle), None, None, None

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
    self._update_damage_observation(battle)
    self._resolve_passive_progress(battle)
    self._cleanup_battle_memory(battle)
    mem = self._get_battle_memory(battle)
    last_action = mem.get("last_action")
    if last_action == "move" and mem.get("last_move_category") == "status":
        prev_hp = mem.get("last_opponent_hp")
        cur_hp = getattr(battle.opponent_active_pokemon, "current_hp_fraction", None)
        unchanged = (
            isinstance(prev_hp, (int, float))
            and isinstance(cur_hp, (int, float))
            and abs(cur_hp - prev_hp) <= 0.03
        )
        if unchanged:
            mem["status_stall_streak"] = int(mem.get("status_stall_streak", 0) or 0) + 1
        else:
            mem["status_stall_streak"] = 0
    else:
        mem["status_stall_streak"] = 0

    active = battle.active_pokemon
    opponent = battle.opponent_active_pokemon
    if active is None or opponent is None:
        return self.choose_random_move(battle), None, None, None
    return None, mem, active, opponent


def _handle_force_switch(self, battle: Battle, opponent, mem):
    if not battle.force_switch:
        return None
    if self.DECISION_DIAG_ENABLED:
        self._mcts_stats["diag_forced_switch_turns"] += 1
        mem["diag_forced_switch_turns"] = int(mem.get("diag_forced_switch_turns", 0) or 0) + 1
    if battle.available_switches:
        best_switch = max(
            battle.available_switches,
            key=lambda s: self._score_switch(s, opponent, battle),
        )
        return self._commit_order(battle, self.create_order(best_switch))
    return self.choose_random_move(battle)


def _compute_search_budget(self, battle: Battle, opponent) -> Tuple[int, int]:
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
    sample_states = self._apply_world_budget_controls(battle, sample_states)
    return sample_states, search_time_ms


def _run_search_with_escalation(self, battle: Battle, sample_states: int, search_time_ms: int):
    results, world_candidates = self._collect_mcts_results(
        battle,
        sample_states=sample_states,
        search_time_ms=search_time_ms,
    )
    if not results:
        return results, world_candidates

    banned_choices = self._sleep_clause_banned_choices(battle)
    ordered_pre, _total_pre, conf_pre, th_pre = self._aggregate_policy_from_results(
        results,
        banned_choices=banned_choices,
    )
    adaptive_escalate = False
    leaf_escalate = False
    if self.ADAPTIVE_ESCALATE_ENABLED and ordered_pre:
        adaptive_escalate = self._should_trigger_adaptive_fallback(
            battle, ordered_pre, conf_pre, th_pre, record_diag=False
        )
    if ordered_pre:
        leaf_escalate = self._should_trigger_leaf_value_escalation(
            battle,
            conf_pre,
            th_pre,
        )
    if not (adaptive_escalate or leaf_escalate):
        return results, world_candidates

    boosted_ms = search_time_ms
    boosted_states = sample_states
    if adaptive_escalate:
        boosted_ms = max(
            boosted_ms,
            int(search_time_ms * max(1.0, self.ADAPTIVE_ESCALATE_MS_MULT)),
        )
        boosted_ms = min(
            boosted_ms,
            max(search_time_ms, self.ADAPTIVE_ESCALATE_MAX_MS),
        )
        boosted_states = max(
            boosted_states,
            int(sample_states * max(1.0, self.ADAPTIVE_ESCALATE_SAMPLE_MULT)),
        )
        boosted_states = min(
            boosted_states,
            max(sample_states, self.ADAPTIVE_ESCALATE_MAX_STATES),
        )
    if leaf_escalate:
        boosted_ms = max(
            boosted_ms,
            int(search_time_ms * max(1.0, self.LEAF_VALUE_ESCALATE_MS_MULT)),
        )
        boosted_ms = min(
            boosted_ms,
            max(search_time_ms, self.LEAF_VALUE_ESCALATE_MAX_MS),
        )
        boosted_states = max(
            boosted_states,
            int(sample_states * max(1.0, self.LEAF_VALUE_ESCALATE_SAMPLE_MULT)),
        )
        boosted_states = min(
            boosted_states,
            max(sample_states, self.LEAF_VALUE_ESCALATE_MAX_STATES),
        )
    if boosted_ms <= search_time_ms and boosted_states <= sample_states:
        return results, world_candidates

    second_results, second_world_candidates = self._collect_mcts_results(
        battle,
        sample_states=boosted_states,
        search_time_ms=boosted_ms,
    )
    if second_results:
        results = second_results
        world_candidates = second_world_candidates
        if adaptive_escalate:
            self._mcts_stats["adaptive_second_pass_used"] += 1
        if leaf_escalate:
            self._mcts_stats["leaf_value_escalated"] += 1
    else:
        if adaptive_escalate:
            self._mcts_stats["adaptive_second_pass_failed"] += 1
        if leaf_escalate:
            self._mcts_stats["leaf_value_escalate_failed"] += 1
    return results, world_candidates


def _handle_empty_or_fallback_choice(self, battle: Battle, choice: str, active, opponent):
    if choice:
        return None
    mem = self._get_battle_memory(battle)
    adaptive_pending = int(mem.get("adaptive_fallback_pending", 0) or 0) == 1
    if adaptive_pending:
        mem["adaptive_fallback_pending"] = 0
        if self.ADAPTIVE_FALLBACK_MODE == "heuristic":
            adaptive_order = self._choose_adaptive_fallback_order(battle, active, opponent)
            if adaptive_order is not None:
                self._mcts_stats["adaptive_heuristic_used"] += 1
                if self.DECISION_DIAG_ENABLED:
                    self._mcts_stats["diag_path_fallback_super"] += 1
                return self._commit_order(battle, adaptive_order)
            self._mcts_stats["adaptive_heuristic_failed"] += 1
        self._mcts_stats["adaptive_super_used"] += 1
    self._mcts_stats["fallback_super"] += 1
    if self.DECISION_DIAG_ENABLED:
        self._mcts_stats["diag_path_fallback_super"] += 1
    return RuleBotPlayer.choose_move(self, battle)


def _resolve_choice_to_order(self, battle: Battle, choice: str, active, opponent):
    choice = self._apply_tactical_safety(battle, choice, active, opponent)

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
        if self.DECISION_DIAG_ENABLED:
            self._mcts_stats["diag_path_fallback_random"] += 1
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
    if self.DECISION_DIAG_ENABLED:
        self._mcts_stats["diag_path_fallback_random"] += 1
    return self.choose_random_move(battle)


def choose_move(self, battle: AbstractBattle):
    if not isinstance(battle, Battle):
        return self.choose_random_move(battle)

    early_order, mem, active, opponent = _prepare_turn_state(self, battle)
    if early_order is not None:
        return early_order

    forced = _handle_force_switch(self, battle, opponent, mem)
    if forced is not None:
        return forced

    self._mcts_stats["calls"] += 1
    sample_states, search_time_ms = _compute_search_budget(self, battle, opponent)
    results, world_candidates = _run_search_with_escalation(
        self,
        battle,
        sample_states,
        search_time_ms,
    )

    if not results:
        self._mcts_stats["empty_results"] += 1
        self._mcts_stats["fallback_super"] += 1
        if self.DECISION_DIAG_ENABLED:
            self._mcts_stats["diag_path_fallback_super"] += 1
        return RuleBotPlayer.choose_move(self, battle)

    banned_choices = self._sleep_clause_banned_choices(battle)
    choice = self._select_move_from_results(
        results,
        battle,
        banned_choices=banned_choices,
        world_candidates=world_candidates,
    )
    fallback = _handle_empty_or_fallback_choice(self, battle, choice, active, opponent)
    if fallback is not None:
        return fallback
    return _resolve_choice_to_order(self, battle, choice, active, opponent)
