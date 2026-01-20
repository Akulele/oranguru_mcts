#!/usr/bin/env python3
"""
OranguruPlayer - lightweight DUCT-style search player.

Uses opponent action priors + shallow expected-damage simulation
to approximate simultaneous-move selection without a full engine.
"""

import math
import random
from typing import List, Dict, Optional

from poke_env.player import Player
from poke_env.battle import AbstractBattle, Battle, Pokemon, Move, MoveCategory

from src.players.rule_bot import RuleBotPlayer


class OranguruPlayer(RuleBotPlayer):
    SEARCH_ITERS = 480
    EXPLORATION = 1.15
    MAX_OPP_ACTIONS = 8
    OPP_PRIOR_SCALE = 12
    MAX_SWITCH_ACTIONS = 2

    def _select_ucb(self, stats: List[dict], total_visits: int, exploration: float) -> int:
        for idx, s in enumerate(stats):
            if s["visits"] <= 0:
                return idx
        log_total = math.log(max(1, total_visits))
        best_idx = 0
        best_score = -1e9
        for idx, s in enumerate(stats):
            avg = s["total"] / s["visits"]
            bonus = exploration * math.sqrt(log_total / s["visits"])
            score = avg + bonus
            if score > best_score:
                best_score = score
                best_idx = idx
        return best_idx

    def _build_opponent_actions(
        self,
        opponent: Pokemon,
        active: Pokemon,
        battle: Battle,
        predicted_switch: Optional[Pokemon],
    ) -> List[dict]:
        candidates = self._candidate_randombattle_sets(opponent, battle)
        actions = []
        if candidates:
            move_weights: Dict[str, int] = {}
            for parsed, count in candidates:
                for move_id in parsed.get("moves", []):
                    move_weights[move_id] = move_weights.get(move_id, 0) + count
            for move_id, count in move_weights.items():
                entry = self._get_move_entry_by_id(move_id)
                actions.append(
                    {
                        "type": "move",
                        "move_id": move_id,
                        "move": None,
                        "entry": entry,
                        "kind": "damage" if str(entry.get("category", "")).lower() != "status" else "status",
                        "priority": int(entry.get("priority", 0) or 0),
                        "weight": float(count),
                    }
                )
        else:
            action_dist = self._opponent_action_distribution(
                opponent, active, battle, predicted_switch=predicted_switch
            )
            if not action_dist:
                return []
            known_moves = {m.id: m for m in self._get_known_moves(opponent)}
            for entry in action_dist:
                if entry.get("type") == "move":
                    move_id = entry.get("move_id")
                    move_obj = known_moves.get(move_id)
                    actions.append(
                        {
                            "type": "move",
                            "move_id": move_id,
                            "move": move_obj,
                            "entry": self._get_move_entry_by_id(move_id),
                            "kind": entry.get("kind", "damage"),
                            "priority": entry.get("priority", 0),
                            "weight": entry.get("weight", 0.0),
                        }
                    )
                else:
                    actions.append(
                        {
                            "type": "switch",
                            "switch": entry.get("switch"),
                            "weight": entry.get("weight", 0.0),
                        }
                    )

        if predicted_switch is not None:
            switch_weight = self._switch_likelihood(opponent, active, battle)
            if switch_weight > 0:
                penalty = self._hazard_switch_penalty_for_opponent(battle, predicted_switch)
                switch_weight = max(0.0, switch_weight - 0.35 * penalty)
                if switch_weight > 0:
                    actions.append(
                        {
                            "type": "switch",
                            "switch": predicted_switch,
                            "weight": switch_weight * 100.0,
                        }
                    )

        actions.sort(key=lambda x: x.get("weight", 0.0), reverse=True)
        return actions[: self.MAX_OPP_ACTIONS]

    def _estimate_opp_damage(
        self,
        opp_action: dict,
        opponent: Pokemon,
        target: Pokemon,
        battle: Battle,
    ) -> float:
        if opp_action.get("type") != "move":
            return 0.0
        move_obj = opp_action.get("move")
        if move_obj is not None:
            try:
                if move_obj.category == MoveCategory.STATUS:
                    return 0.0
            except Exception:
                return 0.0
            return self._calculate_move_score(
                move_obj, opponent, target, battle, apply_recoil=False, respect_immunity_memory=False
            )
        entry = opp_action.get("entry", {})
        if str(entry.get("category", "")).lower() == "status":
            return 0.0
        return self._estimate_entry_damage_score(entry, opponent, target)

    def _simulate_pair(
        self,
        my_action: dict,
        opp_action: dict,
        active: Pokemon,
        opponent: Pokemon,
        battle: Battle,
        current_score: float,
    ) -> float:
        if my_action["type"] == "switch":
            switch_mon = my_action["switch"]
            if switch_mon is None:
                return -50.0
            switch_score = self._score_switch(switch_mon, opponent, battle)
            reward = (switch_score - current_score) * 120.0
            if switch_score <= current_score:
                reward -= 40.0
            return reward

        move: Move = my_action["move"]
        if move is None:
            return -50.0
        if move.category == MoveCategory.STATUS:
            status_value = self._should_use_status_move(move, active, opponent, battle)
            if opp_action.get("type") == "switch":
                status_value *= 0.6
            opp_damage = 0.0
            if opp_action.get("type") == "move":
                opp_damage = self._estimate_opp_damage(opp_action, opponent, active, battle)
            return status_value - opp_damage

        if opp_action.get("type") == "switch" and opp_action.get("switch") is not None:
            target = opp_action["switch"]
            my_damage = self._calculate_move_score(move, active, target, battle)
            hazard_bonus = 80.0 * self._hazard_switch_penalty_for_opponent(battle, target)
            return my_damage + hazard_bonus

        my_damage = self._calculate_move_score(move, active, opponent, battle)
        opp_damage = 0.0
        if opp_action.get("type") == "move":
            opp_damage = self._estimate_opp_damage(opp_action, opponent, active, battle)

        opp_priority = opp_action.get("priority", 0)
        my_priority = self._move_priority_value(move)
        if my_priority > opp_priority:
            first_chance = 0.85
        elif my_priority < opp_priority:
            first_chance = 0.15
        else:
            speed_hint = self._speed_hint(active, opponent, battle)
            if speed_hint > 0:
                first_chance = 0.7
            elif speed_hint < 0:
                first_chance = 0.3
            else:
                my_speed = self._get_effective_speed(active)
                opp_speed = self._get_effective_speed(opponent)
                if my_speed > opp_speed:
                    first_chance = 0.65
                elif opp_speed > my_speed:
                    first_chance = 0.35
                else:
                    first_chance = 0.5

        my_ko = self._ko_likelihood_vs_hp(my_damage, opponent.current_hp_fraction)
        opp_ko = self._ko_likelihood_vs_hp(opp_damage, active.current_hp_fraction)
        expected_my = my_damage * (1.0 - (1.0 - first_chance) * opp_ko)
        expected_opp = opp_damage * (1.0 - first_chance * my_ko)
        return expected_my - expected_opp + (25.0 * my_ko)

    def _search_action(
        self,
        actions: List[dict],
        opp_actions: List[dict],
        active: Pokemon,
        opponent: Pokemon,
        battle: Battle,
        current_score: float,
    ) -> dict:
        total_visits = 0
        my_stats = [{"visits": 0, "total": 0.0} for _ in actions]
        opp_stats = []
        for action in opp_actions:
            prior = max(1, int(round(action.get("weight", 0.0) * self.OPP_PRIOR_SCALE)))
            opp_stats.append({"visits": prior, "total": 0.0})
            total_visits += prior

        for _ in range(self.SEARCH_ITERS):
            total_visits += 1
            my_idx = self._select_ucb(my_stats, total_visits, self.EXPLORATION)
            opp_idx = self._select_ucb(opp_stats, total_visits, self.EXPLORATION)
            reward = self._simulate_pair(
                actions[my_idx], opp_actions[opp_idx], active, opponent, battle, current_score
            )
            my_stats[my_idx]["visits"] += 1
            my_stats[my_idx]["total"] += reward
            opp_stats[opp_idx]["visits"] += 1
            opp_stats[opp_idx]["total"] -= reward

        best_idx = 0
        best_score = -1e9
        for idx, stat in enumerate(my_stats):
            if stat["visits"] <= 0:
                continue
            score = stat["total"] / stat["visits"]
            if score > best_score:
                best_score = score
                best_idx = idx
        return actions[best_idx]

    def choose_move(self, battle: AbstractBattle):
        if not isinstance(battle, Battle):
            return self.choose_random_move(battle)

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

        if battle.force_switch:
            if battle.available_switches:
                best_switch = max(
                    battle.available_switches,
                    key=lambda s: self._score_switch(s, opponent, battle),
                )
                return self._commit_order(battle, self.create_order(best_switch))
            return self.choose_random_move(battle)

        actions = []
        for move in battle.available_moves:
            actions.append({"type": "move", "move": move})
        if battle.available_switches and not battle.force_switch:
            reply_score = self._estimate_best_reply_score(opponent, active, battle)
            current_score = (
                self._estimate_matchup(active, opponent)
                + (active.current_hp_fraction or 0.5) * 0.2
                - (reply_score / 400.0)
            )
            switch_scores = []
            for sw in battle.available_switches:
                score = self._score_switch(sw, opponent, battle)
                if score > current_score + 0.15:
                    switch_scores.append((score, sw))
            switch_scores.sort(key=lambda x: x[0], reverse=True)
            for _, sw in switch_scores[: self.MAX_SWITCH_ACTIONS]:
                actions.append({"type": "switch", "switch": sw})
        elif battle.available_switches:
            for sw in battle.available_switches:
                actions.append({"type": "switch", "switch": sw})
        if not actions:
            return self.choose_random_move(battle)

        predicted_switch = self._predict_opponent_switch(battle)
        opp_actions = self._build_opponent_actions(opponent, active, battle, predicted_switch)
        if not opp_actions:
            return super().choose_move(battle)

        reply_score = self._estimate_best_reply_score(opponent, active, battle)
        current_score = (
            self._estimate_matchup(active, opponent)
            + (active.current_hp_fraction or 0.5) * 0.2
            - (reply_score / 400.0)
        )
        best = self._search_action(actions, opp_actions, active, opponent, battle, current_score)
        if best["type"] == "switch":
            return self._commit_order(battle, self.create_order(best["switch"]))

        move = best["move"]
        order = self.create_order(
            move,
            dynamax=self._should_dynamax(battle, len([m for m in battle.team.values() if not m.fainted])),
            terastallize=self._should_terastallize(battle, move),
        )
        return self._commit_order(battle, order)
