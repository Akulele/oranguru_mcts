#!/usr/bin/env python3
"""
🦧 ORANGURU RL - Comprehensive Evaluation

Evaluates trained RL models with statistically significant sample sizes.

Usage:
    python evaluation/evaluate.py --checkpoint checkpoints/rl/ensemble_final.pt
    python evaluation/evaluate.py --checkpoint checkpoints/rl/ensemble_final.pt --battles 1000
    python evaluation/evaluate.py --checkpoint checkpoints/rl/ensemble_final.pt --quick
"""

import asyncio
import argparse
import sys
from pathlib import Path
from dataclasses import dataclass, fields
import json
import math
import copy
from collections import Counter

import torch.nn.functional as F

import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from poke_env.player import RandomPlayer
from poke_env.player.baselines import MaxBasePowerPlayer, SimpleHeuristicsPlayer
from src.utils.server_config import get_server_configuration
from poke_env.battle import MoveCategory

# Custom config for local showdown server (override via env vars).
CustomServerConfig = get_server_configuration(default_port=8000)

from training.config import RLConfig
from src.models.actor_critic import ActorCritic, RecurrentActorCritic
from src.players.rl_player import RLPlayer
from src.players.rule_bot import RuleBotPlayer
from src.utils.features import load_moves
from src.utils.damage_calc import normalize_name

LOW_HP_THRESHOLD = 0.12
SACK_DELTA_THRESHOLD = 0.2
PRIORITY_HP_THRESHOLD = 0.25
PRIORITY_EXPECTED_RATIO = 0.8
KNOCKOFF_EXPECTED_RATIO = 0.8
BOOST_MISMATCH_RATIO = 1.1

FEATURE_MOVE_START = 16
FEATURE_MOVE_STRIDE = 12
FEATURE_SPEED_START = 124
FEATURE_ITEM_START = 156
FEATURE_FIELD_START = 172
FEATURE_BOOST_START = 220
FEATURE_STATUS_START = 244

RECOVERY_MOVES = {
    "recover",
    "roost",
    "softboiled",
    "slackoff",
    "moonlight",
    "synthesis",
    "morningsun",
    "shoreup",
    "wish",
    "rest",
    "strengthsap",
}


@dataclass
class BattleStats:
    """Detailed battle statistics."""
    wins: int = 0
    losses: int = 0
    total_remaining_pokemon: int = 0  # Sum of remaining pokemon in wins
    total_opponent_remaining: int = 0  # Sum of opponent remaining in losses
    close_wins: int = 0  # Won with 1-2 pokemon left
    dominant_wins: int = 0  # Won with 4+ pokemon left


@dataclass
class ActionStats:
    """Action-level statistics and switch-mass diagnostics."""
    actions: int = 0
    switches: int = 0
    attacks: int = 0
    status: int = 0
    setup: int = 0
    hazards: int = 0
    moves: Counter = None
    switch_targets: Counter = None
    status_available: int = 0
    switch_available: int = 0
    force_switch: int = 0
    tera_available: int = 0
    tera: int = 0
    switch_mass_sum: float = 0.0
    switch_mass_min: float = 1.0
    switch_mass_max: float = 0.0
    switch_mass_low: int = 0
    switch_mass_count: int = 0
    entropy_sum: float = 0.0
    entropy_count: int = 0
    max_prob_sum: float = 0.0
    max_prob_count: int = 0
    switch_mass_bins: Counter = None
    bad_matchup_count: int = 0
    bad_matchup_switch_available: int = 0
    bad_matchup_switch_mass_sum: float = 0.0
    bad_matchup_switch_chosen: int = 0
    bad_matchup_move_chosen: int = 0
    bad_matchup_missed_best: int = 0
    bad_matchup_attack: int = 0
    bad_matchup_status: int = 0
    bad_matchup_setup: int = 0
    bad_matchup_hazard: int = 0
    switch_delta_sum: float = 0.0
    switch_delta_count: int = 0
    switch_delta_bad: int = 0
    switch_delta_good: int = 0
    best_switch_delta_sum: float = 0.0
    best_switch_delta_count: int = 0
    best_switch_delta_ge_03: int = 0
    best_switch_delta_ge_05: int = 0
    missed_best_switch_ge_03: int = 0
    missed_best_switch_ge_05: int = 0
    matchup_bucket_counts: Counter = None
    matchup_bucket_switch: Counter = None
    matchup_bucket_switch_mass: dict = None
    bad_matchup_forced: int = 0
    bad_matchup_no_switch: int = 0
    opportunity_count: int = 0
    opportunity_switch_chosen: int = 0
    opportunity_switch_mass_sum: float = 0.0
    opportunity_switch_mass_min: float = 1.0
    opportunity_switch_mass_max: float = 0.0
    opportunity_best_delta_sum: float = 0.0
    opportunity_best_delta_min: float = float("inf")
    opportunity_best_delta_max: float = float("-inf")
    opportunity_missed: int = 0
    opportunity_attack: int = 0
    opportunity_status: int = 0
    opportunity_setup: int = 0
    opportunity_hazard: int = 0
    opportunity_moves: Counter = None
    opportunity_active: Counter = None
    opportunity_opponent: Counter = None
    opportunity_best_switch: Counter = None
    bad_stay_count: int = 0
    bad_stay_punished: int = 0
    bad_stay_fainted: int = 0
    bad_stay_hp_drop_sum: float = 0.0
    bad_stay_hp_drop_ge_25: int = 0
    bad_stay_hp_drop_ge_50: int = 0
    bad_stay_attack: int = 0
    bad_stay_status: int = 0
    bad_stay_setup: int = 0
    bad_stay_hazard: int = 0
    attack_eff_counts: Counter = None
    attack_eff_sum: float = 0.0
    attack_eff_count: int = 0
    attack_eff_low: int = 0
    attack_eff_super: int = 0
    attack_eff_immune: int = 0
    low_hp_count: int = 0
    low_hp_switch: int = 0
    low_hp_stay: int = 0
    low_hp_sack_opportunities: int = 0
    low_hp_sack_taken: int = 0
    low_hp_sack_missed: int = 0
    priority_finish_opportunities: int = 0
    priority_finish_taken: int = 0
    priority_finish_missed: int = 0
    knockoff_opportunities: int = 0
    knockoff_taken: int = 0
    knockoff_missed: int = 0
    boost_mismatch_opportunities: int = 0
    boost_mismatch_chosen: int = 0
    low_eff_opportunities: int = 0
    low_eff_chosen: int = 0
    defensive_role_actions: int = 0
    defensive_role_attack: int = 0
    defensive_role_status: int = 0
    defensive_role_hazard: int = 0
    defensive_role_setup: int = 0
    defensive_role_switch: int = 0
    defensive_role_status_missed: int = 0

    def __post_init__(self):
        if self.moves is None:
            self.moves = Counter()
        if self.switch_targets is None:
            self.switch_targets = Counter()
        if self.switch_mass_bins is None:
            self.switch_mass_bins = Counter()
        if self.matchup_bucket_counts is None:
            self.matchup_bucket_counts = Counter()
        if self.matchup_bucket_switch is None:
            self.matchup_bucket_switch = Counter()
        if self.matchup_bucket_switch_mass is None:
            self.matchup_bucket_switch_mass = {}
        if self.opportunity_moves is None:
            self.opportunity_moves = Counter()
        if self.opportunity_active is None:
            self.opportunity_active = Counter()
        if self.opportunity_opponent is None:
            self.opportunity_opponent = Counter()
        if self.opportunity_best_switch is None:
            self.opportunity_best_switch = Counter()
        if self.attack_eff_counts is None:
            self.attack_eff_counts = Counter()

    def clone(self) -> "ActionStats":
        return copy.deepcopy(self)

    def diff(self, other: "ActionStats") -> "ActionStats":
        """Return a delta ActionStats (self - other) for additive fields."""
        delta = ActionStats()
        ignore = {
            "switch_mass_min",
            "switch_mass_max",
            "opportunity_switch_mass_min",
            "opportunity_switch_mass_max",
            "opportunity_best_delta_min",
            "opportunity_best_delta_max",
        }
        for field in fields(ActionStats):
            name = field.name
            if name in ignore:
                continue
            val = getattr(self, name)
            other_val = getattr(other, name)
            if isinstance(val, Counter):
                diff_counter = Counter(val)
                diff_counter.subtract(other_val or Counter())
                diff_counter = Counter({k: v for k, v in diff_counter.items() if v > 0})
                setattr(delta, name, diff_counter)
            elif isinstance(val, dict):
                diff_dict = {}
                other_dict = other_val or {}
                for key, value in val.items():
                    diff_value = value - other_dict.get(key, 0.0)
                    if diff_value != 0:
                        diff_dict[key] = diff_value
                setattr(delta, name, diff_dict)
            elif isinstance(val, (int, float)):
                setattr(delta, name, val - (other_val or 0))
        return delta

    def merge(self, other: "ActionStats") -> None:
        """Add stats from another ActionStats into this one."""
        if other is None:
            return
        ignore = {
            "switch_mass_min",
            "switch_mass_max",
            "opportunity_switch_mass_min",
            "opportunity_switch_mass_max",
            "opportunity_best_delta_min",
            "opportunity_best_delta_max",
        }
        for field in fields(ActionStats):
            name = field.name
            if name in ignore:
                continue
            val = getattr(other, name)
            if isinstance(val, Counter):
                getattr(self, name).update(val)
            elif isinstance(val, dict):
                target = getattr(self, name)
                for key, value in val.items():
                    target[key] = target.get(key, 0.0) + value
            elif isinstance(val, (int, float)):
                setattr(self, name, getattr(self, name) + val)


@dataclass
class OutcomeActionStats:
    """Action stats split by battle outcome."""
    wins: ActionStats
    losses: ActionStats


class LoggedRLPlayer(RLPlayer):
    """RL player with optional decision logging and action summaries."""

    def __init__(
        self,
        *args,
        action_stats: ActionStats,
        decision_log: bool = False,
        decision_log_every: int = 1,
        decision_log_topk: int = 3,
        decision_log_max: int = 0,
        opportunity_log: bool = False,
        opportunity_log_topk: int = 3,
        opportunity_log_max: int = 0,
        opportunity_log_include_features: bool = False,
        opportunity_log_all: bool = False,
        opportunity_log_sink=None,
        switch_mass_warn: float = 0.05,
        matchup_threshold: float = -0.3,
        good_switch_delta: float = 0.3,
        log_sink=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.action_stats = action_stats
        self.decision_log = decision_log
        self.decision_log_every = max(1, decision_log_every)
        self.decision_log_topk = max(1, decision_log_topk)
        self.decision_log_max = decision_log_max
        self.opportunity_log = opportunity_log
        self.opportunity_log_topk = max(1, opportunity_log_topk)
        self.opportunity_log_max = opportunity_log_max
        self.opportunity_log_include_features = opportunity_log_include_features
        self.opportunity_log_all = opportunity_log_all
        self.opportunity_log_sink = opportunity_log_sink
        self.switch_mass_warn = switch_mass_warn
        self.matchup_threshold = matchup_threshold
        self.good_switch_delta = good_switch_delta
        self._decision_count = 0
        self._decision_logged = 0
        self._opportunity_logged = 0
        self.log_sink = log_sink
        self._prev_decision = None

    def _find_team_pokemon(self, battle, species: str):
        if not battle or not species:
            return None
        for mon in getattr(battle, "team", {}).values():
            if getattr(mon, "species", None) == species:
                return mon
        return None

    def _update_prev_outcome(self, battle):
        prev = self._prev_decision
        if not prev or not battle:
            return
        if getattr(battle, "battle_tag", None) != prev.get("tag"):
            return
        curr_turn = getattr(battle, "turn", None)
        if curr_turn is None or prev.get("turn") is None:
            return
        if curr_turn <= prev["turn"]:
            return
        if not prev.get("bad_stay"):
            return

        species = prev.get("active_species")
        prev_hp = prev.get("active_hp")
        if species is None or prev_hp is None:
            return
        mon = self._find_team_pokemon(battle, species)
        if not mon:
            return
        curr_hp = getattr(mon, "current_hp_fraction", None)
        if curr_hp is None:
            return
        fainted = bool(getattr(mon, "fainted", False)) or curr_hp <= 0
        drop = max(0.0, prev_hp - curr_hp)

        stats = self.action_stats
        stats.bad_stay_count += 1
        stats.bad_stay_hp_drop_sum += drop
        if drop >= 0.25:
            stats.bad_stay_hp_drop_ge_25 += 1
        if drop >= 0.5:
            stats.bad_stay_hp_drop_ge_50 += 1
        if fainted:
            stats.bad_stay_fainted += 1
        if fainted or drop >= 0.25:
            stats.bad_stay_punished += 1

        action_kind = prev.get("action_kind")
        if action_kind == "attack":
            stats.bad_stay_attack += 1
        elif action_kind == "status":
            stats.bad_stay_status += 1
        elif action_kind == "setup":
            stats.bad_stay_setup += 1
        elif action_kind == "hazard":
            stats.bad_stay_hazard += 1

    def _status_available_in_moves(self, moves):
        moves_data = load_moves()
        for move in moves or []:
            if getattr(move, "category", None) == MoveCategory.STATUS:
                return True
            entry = moves_data.get(move.id, {})
            status = normalize_name(entry.get("status", ""))
            volatile = normalize_name(entry.get("volatileStatus", ""))
            if status or volatile:
                return True
        return False

    def _is_status_move(self, move):
        if getattr(move, "category", None) == MoveCategory.STATUS:
            return True
        entry = load_moves().get(move.id, {})
        status = normalize_name(entry.get("status", ""))
        volatile = normalize_name(entry.get("volatileStatus", ""))
        return bool(status or volatile)

    def _action_label(self, battle, idx: int) -> str:
        if idx < 4:
            if idx < len(battle.available_moves):
                return f"move:{battle.available_moves[idx].id}"
            return "move:?"
        if idx < 9:
            switch_idx = idx - 4
            if switch_idx < len(battle.available_switches):
                return f"switch:{battle.available_switches[switch_idx].species}"
            return "switch:?"
        tera_idx = idx - 9
        if tera_idx < len(battle.available_moves):
            return f"tera:{battle.available_moves[tera_idx].id}"
        return "tera:?"

    def _log_decision(self, battle, action_idx, probs, mask, value):
        tag = getattr(battle, "battle_tag", "unknown")
        turn = getattr(battle, "turn", "?")
        active = getattr(battle, "active_pokemon", None)
        opponent = getattr(battle, "opponent_active_pokemon", None)
        active_name = getattr(active, "species", "unknown")
        opp_name = getattr(opponent, "species", "unknown")
        hp = getattr(active, "current_hp_fraction", None)
        opp_hp = getattr(opponent, "current_hp_fraction", None)
        hp_s = f"{hp:.2f}" if isinstance(hp, float) else "?"
        opp_hp_s = f"{opp_hp:.2f}" if isinstance(opp_hp, float) else "?"

        mask_t = torch.tensor(mask, device=probs.device)
        legal_probs = torch.where(mask_t, probs, torch.zeros_like(probs))
        switch_mass = float(legal_probs[4:9].sum().item())
        move_mass = float(legal_probs[:4].sum().item())
        tera_mass = float(legal_probs[9:13].sum().item())

        topk = min(self.decision_log_topk, len(legal_probs))
        top_vals, top_idxs = torch.topk(legal_probs, topk)
        top_parts = []
        top_entries = []
        for idx, val in zip(top_idxs.tolist(), top_vals.tolist()):
            if val <= 0:
                continue
            label = self._action_label(battle, idx)
            top_parts.append(f"{label}@{val:.2f}")
            top_entries.append({"action": label, "prob": round(val, 4)})

        chosen_label = self._action_label(battle, action_idx)
        avail_moves = len(getattr(battle, "available_moves", []))
        avail_switches = len(getattr(battle, "available_switches", []))
        force_switch = bool(getattr(battle, "force_switch", False))
        tera_available = bool(getattr(battle, "can_tera", False))
        value_s = f"{float(value.item()):.3f}" if value is not None else "?"

        print(
            "DECISION "
            f"tag={tag} turn={turn} {active_name}({hp_s}) vs {opp_name}({opp_hp_s}) "
            f"avail_m={avail_moves} avail_s={avail_switches} force={int(force_switch)} "
            f"tera={int(tera_available)} "
            f"chosen={chosen_label} "
            f"mass_m={move_mass:.2f} mass_s={switch_mass:.2f} mass_t={tera_mass:.2f} "
            f"v={value_s} "
            f"top=[{', '.join(top_parts)}]"
        )
        if self.log_sink is not None:
            payload = {
                "tag": tag,
                "turn": turn,
                "active": active_name,
                "opp": opp_name,
                "hp": hp,
                "opp_hp": opp_hp,
                "avail_moves": avail_moves,
                "avail_switches": avail_switches,
                "force_switch": int(force_switch),
                "tera_available": int(tera_available),
                "chosen": chosen_label,
                "mass_m": round(move_mass, 4),
                "mass_s": round(switch_mass, 4),
                "mass_t": round(tera_mass, 4),
                "value": float(value.item()) if value is not None else None,
                "top": top_entries,
            }
            self.log_sink.write(json.dumps(payload, ensure_ascii=True) + "\n")

    def _log_opportunity(
        self,
        battle,
        action_idx,
        probs,
        mask,
        value,
        matchup: float | None,
        best_delta: float | None,
        best_switch,
        best_switch_action: int | None,
        switch_rankings,
        switch_actions,
        switch_mass: float,
        features,
    ):
        tag = getattr(battle, "battle_tag", "unknown")
        turn = getattr(battle, "turn", "?")
        active = getattr(battle, "active_pokemon", None)
        opponent = getattr(battle, "opponent_active_pokemon", None)
        active_name = getattr(active, "species", "unknown")
        opp_name = getattr(opponent, "species", "unknown")
        hp = getattr(active, "current_hp_fraction", None)
        opp_hp = getattr(opponent, "current_hp_fraction", None)

        mask_t = torch.tensor(mask, device=probs.device)
        legal_probs = torch.where(mask_t, probs, torch.zeros_like(probs))
        topk = min(self.opportunity_log_topk, len(legal_probs))
        top_vals, top_idxs = torch.topk(legal_probs, topk)
        top_entries = []
        for idx, val in zip(top_idxs.tolist(), top_vals.tolist()):
            if val <= 0:
                continue
            label = self._action_label(battle, idx)
            top_entries.append({"action": label, "prob": round(val, 4)})

        chosen_label = self._action_label(battle, action_idx)
        avail_moves = len(getattr(battle, "available_moves", []))
        avail_switches = len(getattr(battle, "available_switches", []))
        force_switch = bool(getattr(battle, "force_switch", False))
        tera_available = bool(getattr(battle, "can_tera", False))
        action_type = "move" if action_idx < 4 else ("switch" if action_idx < 9 else "tera")
        decision = "switch" if action_type == "switch" else "stay"
        best_switch_name = getattr(best_switch, "species", None) if best_switch else None

        payload = {
            "tag": tag,
            "turn": turn,
            "active": active_name,
            "opp": opp_name,
            "hp": hp,
            "opp_hp": opp_hp,
            "avail_moves": avail_moves,
            "avail_switches": avail_switches,
            "force_switch": int(force_switch),
            "tera_available": int(tera_available),
            "matchup": round(matchup, 4) if matchup is not None else None,
            "best_delta": round(best_delta, 4) if best_delta is not None else None,
            "best_switch": best_switch_name,
            "best_switch_action": best_switch_action,
            "switch_mass": round(switch_mass, 4),
            "chosen": chosen_label,
            "chosen_action": action_idx,
            "action_type": action_type,
            "decision": decision,
            "good_switch_delta": self.good_switch_delta,
            "matchup_threshold": self.matchup_threshold,
            "top": top_entries,
        }
        if switch_rankings is not None:
            payload["switch_rankings"] = switch_rankings
        if switch_actions is not None:
            payload["switch_actions"] = switch_actions
        if self.opportunity_log_include_features:
            payload["features"] = features
            payload["mask"] = mask

        if self.opportunity_log_sink is not None:
            self.opportunity_log_sink.write(json.dumps(payload, ensure_ascii=True) + "\n")

    def choose_move(self, battle):
        self._last_battle = battle
        if battle.battle_tag != self._battle_tag:
            self._battle_tag = battle.battle_tag
            self._reset_hidden()
            self._prev_decision = None

        self._update_prev_outcome(battle)

        features = self.feature_builder.build(battle)
        features = [0.0 if (f != f or f > 1e6 or f < -1e6) else f for f in features]
        mask = self._build_mask(battle)

        features_t = torch.tensor([features], dtype=torch.float, device=self.device)
        mask_t = torch.tensor([mask], dtype=torch.bool, device=self.device)

        with torch.no_grad():
            if getattr(self.model, "is_recurrent", False):
                logits, value, next_hidden = self.model.forward_step(features_t, self._hidden)
                logits = self._apply_switch_bias(battle, logits, mask_t)
                logits = self._apply_attack_effectiveness_bias(battle, logits, mask_t)
                masked_logits = logits.masked_fill(~mask_t, -1e9)
                masked_logits = torch.clamp(masked_logits, min=-1e8, max=1e8)
                probs = F.softmax(masked_logits, dim=-1)
                probs = torch.clamp(probs, min=1e-8)
                probs = probs / probs.sum(dim=-1, keepdim=True)
                action = masked_logits.argmax(dim=-1)
                self._hidden = next_hidden
            else:
                logits, value = self.model.forward(features_t, mask_t)
                logits = self._apply_switch_bias(battle, logits, mask_t)
                logits = self._apply_attack_effectiveness_bias(battle, logits, mask_t)
                masked_logits = torch.clamp(logits, min=-1e8, max=1e8)
                probs = F.softmax(masked_logits, dim=-1)
                probs = torch.clamp(probs, min=1e-8)
                probs = probs / probs.sum(dim=-1, keepdim=True)
                action = masked_logits.argmax(dim=-1)

        action_idx = action.item()
        order = self._action_to_order(battle, action_idx)
        move_or_switch = None
        if hasattr(order, "order"):
            move_or_switch = order.order

        stats = self.action_stats
        stats.actions += 1
        if self._status_available_in_moves(getattr(battle, "available_moves", [])):
            stats.status_available += 1
        if getattr(battle, "available_switches", []):
            stats.switch_available += 1
        if getattr(battle, "force_switch", False):
            stats.force_switch += 1
        if getattr(battle, "can_tera", False):
            stats.tera_available += 1

        if move_or_switch is not None:
            if hasattr(move_or_switch, "species"):
                stats.switches += 1
                stats.switch_targets[move_or_switch.species] += 1
            elif hasattr(move_or_switch, "id"):
                move_id = move_or_switch.id
                stats.moves[move_id] += 1
                if move_id in RuleBotPlayer.ENTRY_HAZARDS or move_id in RuleBotPlayer.ANTI_HAZARDS_MOVES:
                    stats.hazards += 1
                if self._is_status_move(move_or_switch):
                    stats.status += 1
                if getattr(move_or_switch, "boosts", None):
                    stats.setup += 1
                if (not self._is_status_move(move_or_switch) and
                    not getattr(move_or_switch, "boosts", None) and
                    move_id not in RuleBotPlayer.ENTRY_HAZARDS and
                    move_id not in RuleBotPlayer.ANTI_HAZARDS_MOVES):
                    stats.attacks += 1

        if hasattr(order, "terastallize") and order.terastallize:
            stats.tera += 1

        legal_probs = torch.where(mask_t.squeeze(0), probs.squeeze(0), torch.zeros_like(probs.squeeze(0)))
        switch_mass = float(legal_probs[4:9].sum().item())
        max_prob = float(legal_probs.max().item()) if legal_probs.numel() else 0.0
        entropy = -(legal_probs * torch.log(torch.clamp(legal_probs, min=1e-8))).sum().item()
        stats.switch_mass_sum += switch_mass
        stats.switch_mass_min = min(stats.switch_mass_min, switch_mass)
        stats.switch_mass_max = max(stats.switch_mass_max, switch_mass)
        if switch_mass < self.switch_mass_warn:
            stats.switch_mass_low += 1
        stats.switch_mass_count += 1
        stats.entropy_sum += entropy
        stats.entropy_count += 1
        stats.max_prob_sum += max_prob
        stats.max_prob_count += 1

        if switch_mass < 0.05:
            stats.switch_mass_bins["<0.05"] += 1
        elif switch_mass < 0.10:
            stats.switch_mass_bins["0.05-0.10"] += 1
        elif switch_mass < 0.20:
            stats.switch_mass_bins["0.10-0.20"] += 1
        elif switch_mass < 0.40:
            stats.switch_mass_bins["0.20-0.40"] += 1
        else:
            stats.switch_mass_bins[">=0.40"] += 1

        active = getattr(battle, "active_pokemon", None)
        opponent = getattr(battle, "opponent_active_pokemon", None)
        matchup = None
        if active and opponent:
            matchup = self._estimate_matchup(active, opponent)
        if move_or_switch is not None and opponent is not None:
            if hasattr(move_or_switch, "id") and getattr(move_or_switch, "base_power", 0) > 0:
                move_id = move_or_switch.id
                is_attack = (
                    not self._is_status_move(move_or_switch)
                    and not getattr(move_or_switch, "boosts", None)
                    and move_id not in RuleBotPlayer.ENTRY_HAZARDS
                    and move_id not in RuleBotPlayer.ANTI_HAZARDS_MOVES
                )
                if is_attack:
                    move_type = getattr(move_or_switch, "type", None)
                    if move_type is not None:
                        eff = opponent.damage_multiplier(move_type)
                        stats.attack_eff_sum += eff
                        stats.attack_eff_count += 1
                        if eff == 0:
                            stats.attack_eff_immune += 1
                            stats.attack_eff_counts["0"] += 1
                        elif eff <= 0.5:
                            stats.attack_eff_low += 1
                            stats.attack_eff_counts["<=0.5"] += 1
                        elif eff < 2:
                            stats.attack_eff_counts["1"] += 1
                        elif eff < 4:
                            stats.attack_eff_super += 1
                            stats.attack_eff_counts["2"] += 1
                        else:
                            stats.attack_eff_super += 1
                            stats.attack_eff_counts[">=4"] += 1
        force_switch = bool(getattr(battle, "force_switch", False))
        available_switches = list(getattr(battle, "available_switches", []))
        switch_available = bool(available_switches)
        if matchup is not None:
            if switch_available and not force_switch:
                if matchup <= -0.5:
                    bucket = "<=-0.5"
                elif matchup < -0.2:
                    bucket = "-0.5..-0.2"
                elif matchup < 0.2:
                    bucket = "-0.2..0.2"
                else:
                    bucket = ">=0.2"
                stats.matchup_bucket_counts[bucket] += 1
                stats.matchup_bucket_switch_mass[bucket] = (
                    stats.matchup_bucket_switch_mass.get(bucket, 0.0) + switch_mass
                )
                if 4 <= action_idx < 9:
                    stats.matchup_bucket_switch[bucket] += 1

        best_delta = None
        best_switch = None
        switch_deltas = None
        if matchup is not None:
            if switch_available and not force_switch:
                switch_deltas = []
                for candidate in available_switches:
                    if not candidate:
                        continue
                    cand_score = self._estimate_matchup(candidate, opponent)
                    delta = cand_score - matchup
                    switch_deltas.append((candidate, delta))
                    if best_delta is None or delta > best_delta:
                        best_delta = delta
                        best_switch = candidate
            if best_delta is not None:
                stats.best_switch_delta_sum += best_delta
                stats.best_switch_delta_count += 1
                if best_delta >= 0.3:
                    stats.best_switch_delta_ge_03 += 1
                if best_delta >= 0.5:
                    stats.best_switch_delta_ge_05 += 1
                if not (4 <= action_idx < 9):
                    if best_delta >= 0.3:
                        stats.missed_best_switch_ge_03 += 1
                    if best_delta >= 0.5:
                        stats.missed_best_switch_ge_05 += 1

        is_opportunity = False
        if matchup is not None and switch_available and not force_switch and best_delta is not None:
            if matchup < self.matchup_threshold and best_delta >= self.good_switch_delta:
                is_opportunity = True
                stats.opportunity_count += 1
                stats.opportunity_switch_mass_sum += switch_mass
                stats.opportunity_switch_mass_min = min(stats.opportunity_switch_mass_min, switch_mass)
                stats.opportunity_switch_mass_max = max(stats.opportunity_switch_mass_max, switch_mass)
                stats.opportunity_best_delta_sum += best_delta
                stats.opportunity_best_delta_min = min(stats.opportunity_best_delta_min, best_delta)
                stats.opportunity_best_delta_max = max(stats.opportunity_best_delta_max, best_delta)

                if 4 <= action_idx < 9:
                    stats.opportunity_switch_chosen += 1
                else:
                    stats.opportunity_missed += 1
                    active_name = getattr(active, "species", "unknown")
                    opp_name = getattr(opponent, "species", "unknown")
                    stats.opportunity_active[active_name] += 1
                    stats.opportunity_opponent[opp_name] += 1
                    if best_switch is not None:
                        best_name = getattr(best_switch, "species", "unknown")
                        stats.opportunity_best_switch[best_name] += 1
                    if move_or_switch is not None and hasattr(move_or_switch, "id"):
                        move_id = move_or_switch.id
                        stats.opportunity_moves[move_id] += 1
                        if move_id in RuleBotPlayer.ENTRY_HAZARDS or move_id in RuleBotPlayer.ANTI_HAZARDS_MOVES:
                            stats.opportunity_hazard += 1
                        elif self._is_status_move(move_or_switch):
                            stats.opportunity_status += 1
                        elif getattr(move_or_switch, "boosts", None):
                            stats.opportunity_setup += 1
                        else:
                            stats.opportunity_attack += 1

        if matchup is not None and matchup < self.matchup_threshold:
            stats.bad_matchup_count += 1
            if not switch_available:
                stats.bad_matchup_no_switch += 1
            elif force_switch:
                stats.bad_matchup_forced += 1
            else:
                stats.bad_matchup_switch_available += 1
                stats.bad_matchup_switch_mass_sum += switch_mass
                if 4 <= action_idx < 9:
                    stats.bad_matchup_switch_chosen += 1
                else:
                    stats.bad_matchup_move_chosen += 1
                    if move_or_switch is not None and hasattr(move_or_switch, "id"):
                        move_id = move_or_switch.id
                        if move_id in RuleBotPlayer.ENTRY_HAZARDS or move_id in RuleBotPlayer.ANTI_HAZARDS_MOVES:
                            stats.bad_matchup_hazard += 1
                        elif self._is_status_move(move_or_switch):
                            stats.bad_matchup_status += 1
                        elif getattr(move_or_switch, "boosts", None):
                            stats.bad_matchup_setup += 1
                        else:
                            stats.bad_matchup_attack += 1
                if best_delta is not None and best_delta >= self.good_switch_delta:
                    if not (4 <= action_idx < 9):
                        stats.bad_matchup_missed_best += 1

        if self.opportunity_log and is_opportunity:
            if self.opportunity_log_all or not (4 <= action_idx < 9):
                if self.opportunity_log_max <= 0 or self._opportunity_logged < self.opportunity_log_max:
                    self._opportunity_logged += 1
                    switch_actions = None
                    best_switch_action = None
                    if available_switches:
                        switch_actions = []
                        best_switch_idx = None
                        for idx, candidate in enumerate(available_switches):
                            name = getattr(candidate, "species", "unknown")
                            switch_actions.append({"species": name, "action": 4 + idx})
                            if best_switch is not None and best_switch_idx is None:
                                if candidate == best_switch or name == getattr(best_switch, "species", None):
                                    best_switch_idx = idx
                        if best_switch_idx is not None:
                            best_switch_action = 4 + best_switch_idx
                    switch_rankings = None
                    if switch_deltas:
                        switch_rankings = []
                        for candidate, delta in sorted(
                            switch_deltas, key=lambda item: item[1], reverse=True
                        )[:5]:
                            name = getattr(candidate, "species", "unknown")
                            action = None
                            if available_switches:
                                try:
                                    idx = available_switches.index(candidate)
                                    action = 4 + idx
                                except ValueError:
                                    action = None
                            switch_rankings.append({
                                "species": name,
                                "delta": round(delta, 4),
                                "action": action,
                            })
                    self._log_opportunity(
                        battle,
                        action_idx,
                        probs.squeeze(0),
                        mask,
                        value,
                        matchup,
                        best_delta,
                        best_switch,
                        best_switch_action,
                        switch_rankings,
                        switch_actions,
                        switch_mass,
                        features,
                    )

        if not force_switch and (4 <= action_idx < 9) and switch_available and matchup is not None:
            switch_idx = action_idx - 4
            if switch_idx < len(getattr(battle, "available_switches", [])):
                chosen = battle.available_switches[switch_idx]
                if chosen and opponent:
                    delta = self._estimate_matchup(chosen, opponent) - matchup
                    stats.switch_delta_sum += delta
                    stats.switch_delta_count += 1
                    if delta < 0:
                        stats.switch_delta_bad += 1
                    if delta >= self.good_switch_delta:
                        stats.switch_delta_good += 1

        action_kind = None
        if move_or_switch is not None:
            if hasattr(move_or_switch, "species"):
                action_kind = "switch"
            elif hasattr(move_or_switch, "id"):
                move_id = move_or_switch.id
                if move_id in RuleBotPlayer.ENTRY_HAZARDS or move_id in RuleBotPlayer.ANTI_HAZARDS_MOVES:
                    action_kind = "hazard"
                elif self._is_status_move(move_or_switch):
                    action_kind = "status"
                elif getattr(move_or_switch, "boosts", None):
                    action_kind = "setup"
                else:
                    action_kind = "attack"

        moves = list(getattr(battle, "available_moves", []))
        active_hp = features[1] if len(features) > 1 else None
        opp_hp = features[2] if len(features) > 2 else None

        chosen_move_idx = None
        if action_idx < 4:
            chosen_move_idx = action_idx
        elif 9 <= action_idx < 13:
            chosen_move_idx = action_idx - 9

        chosen_move = None
        if chosen_move_idx is not None and chosen_move_idx < len(moves):
            chosen_move = moves[chosen_move_idx]

        chosen_eff = None
        chosen_expected = None
        chosen_is_physical = False
        chosen_is_special = False
        chosen_is_priority = False
        if chosen_move_idx is not None:
            off = FEATURE_MOVE_START + chosen_move_idx * FEATURE_MOVE_STRIDE
            if off + 11 < len(features):
                chosen_eff = features[off + 1] * 4
                chosen_expected = features[off + 11] * 200
                power = features[off + 2] * 150
                if power <= 0:
                    chosen_expected = 0.0
                chosen_is_physical = features[off + 4] > 0.5
                chosen_is_special = features[off + 5] > 0.5
                chosen_is_priority = features[off + 7] > 0.0

        best_expected = 0.0
        best_eff = 0.0
        best_physical_expected = 0.0
        best_special_expected = 0.0
        best_priority_expected = 0.0
        knockoff_expected = 0.0

        for idx, move in enumerate(moves[:4]):
            off = FEATURE_MOVE_START + idx * FEATURE_MOVE_STRIDE
            if off + 11 >= len(features):
                continue
            eff = features[off + 1] * 4
            expected = features[off + 11] * 200
            power = features[off + 2] * 150
            is_physical = features[off + 4] > 0.5
            is_special = features[off + 5] > 0.5
            has_priority = features[off + 7] > 0.0

            if power > 0:
                best_expected = max(best_expected, expected)
                best_eff = max(best_eff, eff)
                if is_physical:
                    best_physical_expected = max(best_physical_expected, expected)
                if is_special:
                    best_special_expected = max(best_special_expected, expected)
                if has_priority:
                    best_priority_expected = max(best_priority_expected, expected)

            if move.id == "knockoff":
                knockoff_expected = expected

        opp_blocks_priority = False
        if FEATURE_SPEED_START + 5 < len(features):
            opp_blocks_priority = features[FEATURE_SPEED_START + 5] > 0.5
        if FEATURE_FIELD_START + 6 < len(features):
            if features[FEATURE_FIELD_START + 6] > 0.5:
                opp_blocks_priority = True

        if (
            active_hp is not None
            and switch_available
            and not force_switch
            and active_hp > 0
            and active_hp <= LOW_HP_THRESHOLD
        ):
            stats.low_hp_count += 1
            if 4 <= action_idx < 9:
                stats.low_hp_switch += 1
            else:
                stats.low_hp_stay += 1
            if best_delta is not None and best_delta < SACK_DELTA_THRESHOLD:
                stats.low_hp_sack_opportunities += 1
                if 4 <= action_idx < 9:
                    stats.low_hp_sack_missed += 1
                else:
                    stats.low_hp_sack_taken += 1

        if (
            opp_hp is not None
            and opp_hp <= PRIORITY_HP_THRESHOLD
            and best_priority_expected > 0
            and best_expected > 0
            and not opp_blocks_priority
            and best_priority_expected >= best_expected * PRIORITY_EXPECTED_RATIO
        ):
            stats.priority_finish_opportunities += 1
            if chosen_is_priority:
                stats.priority_finish_taken += 1
            else:
                stats.priority_finish_missed += 1

        opp_item_known = False
        if opponent and getattr(opponent, "item", None):
            opp_item_known = True
        if not opp_item_known and FEATURE_ITEM_START + 10 <= len(features):
            opp_item_known = any(
                features[FEATURE_ITEM_START + i] > 0 for i in range(6, 10)
            )

        if (
            knockoff_expected > 0
            and opp_item_known
            and best_expected > 0
            and knockoff_expected >= best_expected * KNOCKOFF_EXPECTED_RATIO
        ):
            stats.knockoff_opportunities += 1
            if chosen_move is not None and getattr(chosen_move, "id", "") == "knockoff":
                stats.knockoff_taken += 1
            else:
                stats.knockoff_missed += 1

        opp_def_boost = 0.0
        opp_spd_boost = 0.0
        if FEATURE_BOOST_START + 11 < len(features):
            opp_def_boost = features[FEATURE_BOOST_START + 8] * 6
            opp_spd_boost = features[FEATURE_BOOST_START + 10] * 6

        if best_physical_expected > 0 and best_special_expected > 0:
            if opp_def_boost >= 1:
                stats.boost_mismatch_opportunities += 1
                if (
                    chosen_is_physical
                    and chosen_expected is not None
                    and best_special_expected >= chosen_expected * BOOST_MISMATCH_RATIO
                ):
                    stats.boost_mismatch_chosen += 1
            if opp_spd_boost >= 1:
                stats.boost_mismatch_opportunities += 1
                if (
                    chosen_is_special
                    and chosen_expected is not None
                    and best_physical_expected >= chosen_expected * BOOST_MISMATCH_RATIO
                ):
                    stats.boost_mismatch_chosen += 1

        if best_eff >= 1.0 and chosen_expected is not None and chosen_expected > 0:
            stats.low_eff_opportunities += 1
            if chosen_eff is not None and chosen_eff <= 0.5:
                stats.low_eff_chosen += 1

        has_recovery = any(getattr(move, "id", "") in RECOVERY_MOVES for move in moves)
        has_hazard = any(
            getattr(move, "id", "") in RuleBotPlayer.ENTRY_HAZARDS
            or getattr(move, "id", "") in RuleBotPlayer.ANTI_HAZARDS_MOVES
            for move in moves
        )
        has_status = any(self._is_status_move(move) for move in moves)
        damaging_moves = sum(1 for move in moves if getattr(move, "base_power", 0) > 0)
        defensive_role = (has_recovery or has_hazard) and damaging_moves <= 2

        if defensive_role:
            stats.defensive_role_actions += 1
            if action_kind == "attack":
                stats.defensive_role_attack += 1
            elif action_kind == "status":
                stats.defensive_role_status += 1
            elif action_kind == "hazard":
                stats.defensive_role_hazard += 1
            elif action_kind == "setup":
                stats.defensive_role_setup += 1
            elif action_kind == "switch":
                stats.defensive_role_switch += 1

            opp_status_any = 0.0
            if FEATURE_STATUS_START + 13 < len(features):
                opp_status_any = features[FEATURE_STATUS_START + 13]
            if has_status and opp_status_any < 0.5 and action_kind == "attack":
                stats.defensive_role_status_missed += 1

        self._prev_decision = {
            "tag": getattr(battle, "battle_tag", None),
            "turn": getattr(battle, "turn", None),
            "active_species": getattr(active, "species", None),
            "active_hp": getattr(active, "current_hp_fraction", None),
            "bad_stay": bool(is_opportunity and not (4 <= action_idx < 9)),
            "action_kind": action_kind,
        }

        if self.decision_log:
            self._decision_count += 1
            if self.decision_log_max <= 0 or self._decision_logged < self.decision_log_max:
                if self._decision_count % self.decision_log_every == 0:
                    self._decision_logged += 1
                    self._log_decision(battle, action_idx, probs.squeeze(0), mask, value)

        return order


async def evaluate_vs_opponent(
    model: ActorCritic,
    config: RLConfig,
    opponent_type: str,
    n_battles: int,
    device: str,
    decision_log: bool = False,
    decision_log_every: int = 1,
    decision_log_topk: int = 3,
    decision_log_max: int = 0,
    opportunity_log: bool = False,
    opportunity_log_topk: int = 3,
    opportunity_log_max: int = 0,
    opportunity_log_include_features: bool = False,
    opportunity_log_all: bool = False,
    action_summary: bool = False,
    switch_mass_warn: float = 0.05,
    matchup_threshold: float = -0.3,
    good_switch_delta: float = 0.3,
    decision_log_file=None,
    opportunity_log_file=None,
) -> tuple[float, BattleStats, ActionStats | None, OutcomeActionStats | None]:
    """Evaluate model against a specific opponent type with detailed stats."""

    kwargs = {
        'battle_format': 'gen9randombattle',
        'max_concurrent_battles': 1,
        'server_configuration': CustomServerConfig,
    }

    # Create opponent
    if opponent_type == "random":
        opponent = RandomPlayer(**kwargs)
    elif opponent_type == "max_power":
        opponent = MaxBasePowerPlayer(**kwargs)
    elif opponent_type == "heuristics":
        opponent = SimpleHeuristicsPlayer(**kwargs)
    else:
        raise ValueError(f"Unknown opponent type: {opponent_type}")

    action_stats = ActionStats() if (decision_log or action_summary or opportunity_log) else None
    outcome_stats = None
    if action_stats:
        outcome_stats = OutcomeActionStats(wins=ActionStats(), losses=ActionStats())
    if action_stats:
        agent = LoggedRLPlayer(
            model=model,
            config=config,
            device=device,
            training=False,
            action_stats=action_stats,
            decision_log=decision_log,
            decision_log_every=decision_log_every,
            decision_log_topk=decision_log_topk,
            decision_log_max=decision_log_max,
            opportunity_log=opportunity_log,
            opportunity_log_topk=opportunity_log_topk,
            opportunity_log_max=opportunity_log_max,
            opportunity_log_include_features=opportunity_log_include_features,
            opportunity_log_all=opportunity_log_all,
            opportunity_log_sink=opportunity_log_file,
            switch_mass_warn=switch_mass_warn,
            matchup_threshold=matchup_threshold,
            good_switch_delta=good_switch_delta,
            log_sink=decision_log_file,
            **kwargs,
        )
    else:
        agent = RLPlayer(
            model=model,
            config=config,
            device=device,
            training=False,
            **kwargs
        )

    stats = BattleStats()

    # Run battles one at a time to track detailed stats
    for i in range(n_battles):
        pre_stats = action_stats.clone() if action_stats else None
        prev_wins = agent.n_won_battles
        await agent.battle_against(opponent, n_battles=1)
        won = agent.n_won_battles > prev_wins
        battle = agent._last_battle

        if won:
            stats.wins += 1
            # Count remaining pokemon (approximate from last battle state)
            if battle:
                remaining = len([p for p in battle.team.values() if not p.fainted])
                stats.total_remaining_pokemon += remaining
                if remaining <= 2:
                    stats.close_wins += 1
                elif remaining >= 4:
                    stats.dominant_wins += 1
        else:
            stats.losses += 1
            if battle:
                opp_remaining = 6 - len([p for p in battle.opponent_team.values() if p.fainted])
                stats.total_opponent_remaining += opp_remaining

        # Clear battle history to avoid leaking battle objects
        agent._last_battle = None
        try:
            agent.reset_battles()
        except Exception:
            pass
        try:
            opponent.reset_battles()
        except Exception:
            pass

        # Progress indicator
        if (i + 1) % 100 == 0:
            wr = stats.wins / (i + 1)
            print(f"      {i+1}/{n_battles} battles, current: {wr:.1%}")

        if action_stats and outcome_stats and pre_stats:
            delta = action_stats.diff(pre_stats)
            if won:
                outcome_stats.wins.merge(delta)
            else:
                outcome_stats.losses.merge(delta)

    win_rate = stats.wins / n_battles
    return win_rate, stats, action_stats, outcome_stats


def confidence_interval(wins: int, total: int, confidence: float = 0.95) -> tuple[float, float]:
    """Calculate confidence interval for win rate using Wilson score."""
    if total == 0:
        return 0.0, 0.0

    p = wins / total
    z = 1.96 if confidence == 0.95 else 2.576  # 95% or 99%

    denominator = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denominator
    spread = z * math.sqrt((p * (1 - p) + z**2 / (4 * total)) / total) / denominator

    return max(0, center - spread), min(1, center + spread)


def print_action_summary(action_stats: ActionStats, switch_mass_warn: float):
    total_actions = max(action_stats.actions, 1)
    print("\n📌 Action Summary")
    print(f"   Switch rate: {action_stats.switches / total_actions:.1%}")
    print(f"   Attack rate: {action_stats.attacks / total_actions:.1%}")
    print(f"   Status rate: {action_stats.status / total_actions:.1%}")
    print(f"   Setup rate:  {action_stats.setup / total_actions:.1%}")
    print(f"   Hazard rate: {action_stats.hazards / total_actions:.1%}")
    print(f"   Status avail: {action_stats.status_available / total_actions:.1%}")
    print(f"   Switch avail: {action_stats.switch_available / total_actions:.1%}")
    print(f"   Force switch: {action_stats.force_switch / total_actions:.1%}")
    print(f"   Tera avail:   {action_stats.tera_available / total_actions:.1%}")
    print(f"   Tera used/avail: {action_stats.tera}/{action_stats.tera_available}")

    if action_stats.moves:
        top_moves = action_stats.moves.most_common(10)
        print("\n📌 Top Moves")
        for move_id, count in top_moves:
            print(f"   {move_id}: {count}")
    if action_stats.switch_targets:
        top_switches = action_stats.switch_targets.most_common(10)
        print("\n📌 Top Switches")
        for species, count in top_switches:
            print(f"   {species}: {count}")

    if action_stats.switch_mass_count > 0:
        avg_mass = action_stats.switch_mass_sum / action_stats.switch_mass_count
        low_pct = action_stats.switch_mass_low / action_stats.switch_mass_count
        print("\n📌 Policy Switch Mass")
        print(
            f"   avg={avg_mass:.2f} (min {action_stats.switch_mass_min:.2f} "
            f"max {action_stats.switch_mass_max:.2f}) "
            f"low<{switch_mass_warn:.2f}: {action_stats.switch_mass_low} ({low_pct:.1%})"
        )

    if action_stats.switch_mass_bins:
        total = max(action_stats.switch_mass_count, 1)
        print("\n📌 Switch Mass Histogram")
        for label in ["<0.05", "0.05-0.10", "0.10-0.20", "0.20-0.40", ">=0.40"]:
            count = action_stats.switch_mass_bins.get(label, 0)
            print(f"   {label}: {count} ({count/total:.1%})")

    if action_stats.entropy_count > 0:
        avg_entropy = action_stats.entropy_sum / action_stats.entropy_count
        avg_max_prob = action_stats.max_prob_sum / action_stats.max_prob_count
        print("\n📌 Decision Confidence")
        print(f"   Avg entropy: {avg_entropy:.3f}")
        print(f"   Avg max prob: {avg_max_prob:.1%}")

    if action_stats.attack_eff_count > 0:
        avg_eff = action_stats.attack_eff_sum / action_stats.attack_eff_count
        low_rate = action_stats.attack_eff_low / action_stats.attack_eff_count
        super_rate = action_stats.attack_eff_super / action_stats.attack_eff_count
        immune_rate = action_stats.attack_eff_immune / action_stats.attack_eff_count
        print("\n📌 Attack Effectiveness")
        print(f"   Avg multiplier: {avg_eff:.2f}")
        print(
            f"   Immune: {immune_rate:.1%} | Resisted<=0.5: {low_rate:.1%} "
            f"| Super>=2: {super_rate:.1%}"
        )
        if action_stats.attack_eff_counts:
            for label in ["0", "<=0.5", "1", "2", ">=4"]:
                count = action_stats.attack_eff_counts.get(label, 0)
                if count:
                    print(f"   {label}: {count}")

    if action_stats.bad_matchup_count > 0:
        bad_rate = action_stats.bad_matchup_count / max(action_stats.actions, 1)
        eligible = action_stats.bad_matchup_switch_available
        forced = action_stats.bad_matchup_forced
        no_switch = action_stats.bad_matchup_no_switch
        sw_rate = action_stats.bad_matchup_switch_chosen / max(eligible, 1)
        move_rate = action_stats.bad_matchup_move_chosen / max(eligible, 1)
        avg_bad_mass = action_stats.bad_matchup_switch_mass_sum / max(eligible, 1)
        missed = action_stats.bad_matchup_missed_best
        print("\n📌 Bad Matchup Behavior")
        print(f"   Bad matchup rate: {bad_rate:.1%}")
        print(
            f"   Eligible (switch avail, not forced): {eligible}/"
            f"{action_stats.bad_matchup_count} ({eligible/max(action_stats.bad_matchup_count,1):.1%})"
        )
        if forced or no_switch:
            print(f"   Ineligible: forced={forced} no_switch={no_switch}")
        print(f"   Switch when bad: {action_stats.bad_matchup_switch_chosen}/{eligible} ({sw_rate:.1%})")
        print(f"   Stay when bad:   {action_stats.bad_matchup_move_chosen}/{eligible} ({move_rate:.1%})")
        print(f"   Avg switch mass (bad): {avg_bad_mass:.2f}")
        print(f"   Missed good switch: {missed} ({missed/max(eligible,1):.1%})")
        stayed = max(action_stats.bad_matchup_move_chosen, 1)
        print("   Stay action mix:")
        print(f"     attack={action_stats.bad_matchup_attack/stayed:.1%} "
              f"status={action_stats.bad_matchup_status/stayed:.1%} "
              f"setup={action_stats.bad_matchup_setup/stayed:.1%} "
              f"hazard={action_stats.bad_matchup_hazard/stayed:.1%}")

    if action_stats.low_hp_count > 0:
        low = action_stats.low_hp_count
        sw_rate = action_stats.low_hp_switch / low
        stay_rate = action_stats.low_hp_stay / low
        sack_opp = action_stats.low_hp_sack_opportunities
        sack_taken = action_stats.low_hp_sack_taken
        print("\n📌 Low-HP Decisions")
        print(f"   Low-HP (<= {LOW_HP_THRESHOLD:.0%}) opportunities: {low}")
        print(f"   Switch at low HP: {action_stats.low_hp_switch}/{low} ({sw_rate:.1%})")
        print(f"   Stay at low HP:   {action_stats.low_hp_stay}/{low} ({stay_rate:.1%})")
        if sack_opp:
            print(
                f"   Sack opportunities: {sack_taken}/{sack_opp} "
                f"({sack_taken/max(sack_opp,1):.1%})"
            )

    if action_stats.priority_finish_opportunities > 0:
        opps = action_stats.priority_finish_opportunities
        taken = action_stats.priority_finish_taken
        print("\n📌 Priority Finisher")
        print(f"   Opportunities: {opps}")
        print(f"   Taken: {taken}/{opps} ({taken/opps:.1%})")

    if action_stats.knockoff_opportunities > 0:
        opps = action_stats.knockoff_opportunities
        taken = action_stats.knockoff_taken
        print("\n📌 Knock Off Value")
        print(f"   Opportunities: {opps}")
        print(f"   Taken: {taken}/{opps} ({taken/opps:.1%})")

    if action_stats.boost_mismatch_opportunities > 0:
        opps = action_stats.boost_mismatch_opportunities
        chosen = action_stats.boost_mismatch_chosen
        print("\n📌 Boost Mismatch")
        print(f"   Opportunities: {opps}")
        print(f"   Chosen mismatch: {chosen}/{opps} ({chosen/opps:.1%})")

    if action_stats.low_eff_opportunities > 0:
        opps = action_stats.low_eff_opportunities
        chosen = action_stats.low_eff_chosen
        print("\n📌 Low-Effect Choice")
        print(f"   Opportunities: {opps}")
        print(f"   Chosen low-eff: {chosen}/{opps} ({chosen/opps:.1%})")

    if action_stats.defensive_role_actions > 0:
        total = action_stats.defensive_role_actions
        print("\n📌 Defensive Role Usage")
        print(f"   Actions: {total}")
        print(
            f"   attack={action_stats.defensive_role_attack/total:.1%} "
            f"status={action_stats.defensive_role_status/total:.1%} "
            f"setup={action_stats.defensive_role_setup/total:.1%} "
            f"hazard={action_stats.defensive_role_hazard/total:.1%} "
            f"switch={action_stats.defensive_role_switch/total:.1%}"
        )
        if action_stats.defensive_role_status_missed:
            print(
                f"   Status missed (opp not statused): "
                f"{action_stats.defensive_role_status_missed}/{total} "
                f"({action_stats.defensive_role_status_missed/total:.1%})"
            )

    if action_stats.opportunity_count > 0:
        opp_count = action_stats.opportunity_count
        opp_taken = action_stats.opportunity_switch_chosen
        opp_missed = action_stats.opportunity_missed
        avg_opp_mass = action_stats.opportunity_switch_mass_sum / opp_count
        avg_best = action_stats.opportunity_best_delta_sum / opp_count
        min_best = action_stats.opportunity_best_delta_min
        max_best = action_stats.opportunity_best_delta_max
        print("\n📌 Switch Opportunities (clean)")
        print(f"   Opportunities: {opp_count}")
        print(f"   Switch taken: {opp_taken}/{opp_count} ({opp_taken/opp_count:.1%})")
        print(f"   Avg switch mass: {avg_opp_mass:.2f}")
        print(f"   Best delta: avg={avg_best:.2f} min={min_best:.2f} max={max_best:.2f}")
        if opp_missed:
            print("   Missed action mix:")
            stayed = max(opp_missed, 1)
            print(f"     attack={action_stats.opportunity_attack/stayed:.1%} "
                  f"status={action_stats.opportunity_status/stayed:.1%} "
                  f"setup={action_stats.opportunity_setup/stayed:.1%} "
                  f"hazard={action_stats.opportunity_hazard/stayed:.1%}")
            if action_stats.opportunity_moves:
                top_moves = action_stats.opportunity_moves.most_common(5)
                move_parts = ", ".join(f"{move}:{count}" for move, count in top_moves)
                print(f"   Top missed moves: {move_parts}")
            if action_stats.opportunity_best_switch:
                top_best = action_stats.opportunity_best_switch.most_common(5)
                best_parts = ", ".join(f"{name}:{count}" for name, count in top_best)
                print(f"   Top best switches: {best_parts}")
            if action_stats.opportunity_active:
                top_active = action_stats.opportunity_active.most_common(3)
                active_parts = ", ".join(f"{name}:{count}" for name, count in top_active)
                print(f"   Top actives: {active_parts}")
            if action_stats.opportunity_opponent:
                top_opp = action_stats.opportunity_opponent.most_common(3)
                opp_parts = ", ".join(f"{name}:{count}" for name, count in top_opp)
                print(f"   Top opponents: {opp_parts}")

    if action_stats.switch_delta_count > 0:
        avg_delta = action_stats.switch_delta_sum / action_stats.switch_delta_count
        bad_rate = action_stats.switch_delta_bad / action_stats.switch_delta_count
        good_rate = action_stats.switch_delta_good / action_stats.switch_delta_count
        print("\n📌 Switch Quality")
        print(f"   Avg matchup delta: {avg_delta:.2f}")
        print(f"   Bad switches (delta<0): {bad_rate:.1%}")
        print(f"   Good switches (delta>=0.3): {good_rate:.1%}")

    if action_stats.best_switch_delta_count > 0:
        avg_best = action_stats.best_switch_delta_sum / action_stats.best_switch_delta_count
        ge03 = action_stats.best_switch_delta_ge_03 / action_stats.best_switch_delta_count
        ge05 = action_stats.best_switch_delta_ge_05 / action_stats.best_switch_delta_count
        miss03 = action_stats.missed_best_switch_ge_03 / action_stats.best_switch_delta_count
        miss05 = action_stats.missed_best_switch_ge_05 / action_stats.best_switch_delta_count
        print("\n📌 Best Switch Opportunity")
        print(f"   Avg best delta: {avg_best:.2f}")
        print(f"   Best>=0.3: {ge03:.1%} | Missed>=0.3: {miss03:.1%}")
        print(f"   Best>=0.5: {ge05:.1%} | Missed>=0.5: {miss05:.1%}")

    if action_stats.bad_stay_count > 0:
        punished = action_stats.bad_stay_punished / action_stats.bad_stay_count
        fainted = action_stats.bad_stay_fainted / action_stats.bad_stay_count
        avg_drop = action_stats.bad_stay_hp_drop_sum / action_stats.bad_stay_count
        print("\n📌 Bad Stay Consequences")
        print(f"   Stayed in bad matchup: {action_stats.bad_stay_count}")
        print(f"   Punished (>=25% drop or faint): {action_stats.bad_stay_punished} ({punished:.1%})")
        print(f"   Fainted: {action_stats.bad_stay_fainted} ({fainted:.1%})")
        print(f"   Avg HP drop: {avg_drop:.2f}")
        if action_stats.bad_stay_count > 0:
            print("   Stayed action mix:")
            stayed = max(action_stats.bad_stay_count, 1)
            print(f"     attack={action_stats.bad_stay_attack/stayed:.1%} "
                  f"status={action_stats.bad_stay_status/stayed:.1%} "
                  f"setup={action_stats.bad_stay_setup/stayed:.1%} "
                  f"hazard={action_stats.bad_stay_hazard/stayed:.1%}")

    if action_stats.matchup_bucket_counts:
        print("\n📌 Matchup Buckets (switch-available)")
        order = ["<=-0.5", "-0.5..-0.2", "-0.2..0.2", ">=0.2"]
        for bucket in order:
            count = action_stats.matchup_bucket_counts.get(bucket, 0)
            if count == 0:
                continue
            switch_rate = action_stats.matchup_bucket_switch.get(bucket, 0) / count
            avg_mass = action_stats.matchup_bucket_switch_mass.get(bucket, 0.0) / count
            print(
                f"   {bucket}: n={count} switch_rate={switch_rate:.1%} avg_switch_mass={avg_mass:.2f}"
            )


def print_outcome_summary(outcome_stats: OutcomeActionStats) -> None:
    """Print key metrics split by wins vs losses."""
    def ratio(n: float, d: float) -> float:
        return n / d if d > 0 else 0.0

    def summarize(label: str, stats: ActionStats) -> None:
        total = max(stats.actions, 1)
        switch_rate = stats.switches / total
        opp_rate = ratio(stats.opportunity_switch_chosen, stats.opportunity_count)
        bad_rate = ratio(stats.bad_matchup_switch_chosen, stats.bad_matchup_switch_available)
        avg_mass = ratio(stats.switch_mass_sum, stats.switch_mass_count)
        avg_opp_mass = ratio(stats.opportunity_switch_mass_sum, stats.opportunity_count)
        avg_best_delta = ratio(stats.opportunity_best_delta_sum, stats.opportunity_count)
        attack_rate = stats.attacks / total
        status_rate = stats.status / total
        setup_rate = stats.setup / total
        hazard_rate = stats.hazards / total
        avg_entropy = ratio(stats.entropy_sum, stats.entropy_count)
        avg_max_prob = ratio(stats.max_prob_sum, stats.max_prob_count)
        avg_switch_delta = ratio(stats.switch_delta_sum, stats.switch_delta_count)
        bad_switch_rate = ratio(stats.switch_delta_bad, stats.switch_delta_count)
        good_switch_rate = ratio(stats.switch_delta_good, stats.switch_delta_count)
        avg_eff = ratio(stats.attack_eff_sum, stats.attack_eff_count)
        low_eff = ratio(stats.attack_eff_low, stats.attack_eff_count)
        super_eff = ratio(stats.attack_eff_super, stats.attack_eff_count)
        immune_eff = ratio(stats.attack_eff_immune, stats.attack_eff_count)
        print(
            f"   {label}: switch={switch_rate:.1%} "
            f"opp_take={opp_rate:.1%} bad_switch={bad_rate:.1%} "
            f"avg_mass={avg_mass:.2f} opp_mass={avg_opp_mass:.2f} "
            f"best_delta={avg_best_delta:.2f} "
            f"entropy={avg_entropy:.2f} max_prob={avg_max_prob:.1%}"
        )
        print(
            f"     mix: attack={attack_rate:.1%} status={status_rate:.1%} "
            f"setup={setup_rate:.1%} hazard={hazard_rate:.1%}"
        )
        print(
            f"     attack_eff: avg={avg_eff:.2f} immune={immune_eff:.1%} "
            f"resisted={low_eff:.1%} super={super_eff:.1%}"
        )
        print(
            f"     switch_quality: avg_delta={avg_switch_delta:.2f} "
            f"bad={bad_switch_rate:.1%} good={good_switch_rate:.1%}"
        )

    print("\n📌 Outcome Split (wins vs losses)")
    summarize("wins", outcome_stats.wins)
    summarize("losses", outcome_stats.losses)


def print_outcome_bucket_summary(outcome_stats: OutcomeActionStats) -> None:
    """Print matchup bucket switch rates split by wins vs losses."""
    order = ["<=-0.5", "-0.5..-0.2", "-0.2..0.2", ">=0.2"]

    def bucket_line(label: str, stats: ActionStats, bucket: str) -> str:
        count = stats.matchup_bucket_counts.get(bucket, 0)
        if count == 0:
            return f"{label} n=0"
        switch_rate = stats.matchup_bucket_switch.get(bucket, 0) / count
        avg_mass = stats.matchup_bucket_switch_mass.get(bucket, 0.0) / count
        return f"{label} n={count} sw={switch_rate:.1%} mass={avg_mass:.2f}"

    print("\n📌 Outcome Buckets (switch-available)")
    for bucket in order:
        win_line = bucket_line("wins", outcome_stats.wins, bucket)
        loss_line = bucket_line("loss", outcome_stats.losses, bucket)
        print(f"   {bucket}: {win_line} | {loss_line}")


def print_condensed_diagnostics(
    opponents: list[str],
    all_stats: dict[str, BattleStats],
    all_action_stats: dict[str, ActionStats | None],
    targets: dict[str, float],
    switch_mass_warn: float,
) -> None:
    """Print condensed diagnostics useful for policy tuning."""
    def ratio(n: float, d: float) -> float:
        return n / d if d > 0 else 0.0

    print("\n📌 Condensed Diagnostics")
    print(
        f"{'Opponent':<12} {'Sw/Avail':>9} {'St/Avail':>9} {'TeraUse':>8} "
        f"{'SwMass':>7} {'LowMass':>8} {'TopMove':>8} {'SwDiv':>7}"
    )
    print("-" * 70)

    for opponent in opponents:
        action_stats = all_action_stats.get(opponent)
        stats = all_stats.get(opponent)
        if not action_stats or not stats:
            continue
        total_actions = max(action_stats.actions, 1)
        switch_rate = action_stats.switches / total_actions
        switch_avail_rate = action_stats.switch_available / total_actions
        status_rate = action_stats.status / total_actions
        status_avail_rate = action_stats.status_available / total_actions
        tera_use = ratio(action_stats.tera, action_stats.tera_available)

        avg_mass = ratio(action_stats.switch_mass_sum, action_stats.switch_mass_count)
        low_mass = ratio(action_stats.switch_mass_low, action_stats.switch_mass_count)
        total_moves = sum(action_stats.moves.values()) or 1
        top_move_share = (max(action_stats.moves.values()) if action_stats.moves else 0) / total_moves
        switch_div = ratio(len(action_stats.switch_targets), action_stats.switches)

        print(
            f"{opponent:<12} {ratio(switch_rate, switch_avail_rate):>9.2f} "
            f"{ratio(status_rate, status_avail_rate):>9.2f} "
            f"{tera_use:>8.2f} {avg_mass:>7.2f} {low_mass:>8.1%} "
            f"{top_move_share:>8.1%} {switch_div:>7.2f}"
        )

    print("-" * 70)
    print(
        f"  Sw/Avail and St/Avail = utilization ratios; "
        f"LowMass uses <{switch_mass_warn:.2f} threshold."
    )


async def full_evaluation(
    checkpoint_path: str,
    n_battles: int = 500,
    decision_log: bool = False,
    decision_log_every: int = 1,
    decision_log_topk: int = 3,
    decision_log_max: int = 0,
    decision_log_dir: str | None = None,
    opportunity_log_dir: str | None = None,
    opportunity_log_topk: int = 3,
    opportunity_log_max: int = 0,
    opportunity_log_include_features: bool = False,
    opportunity_log_all: bool = False,
    action_summary: bool = False,
    switch_mass_warn: float = 0.05,
    matchup_threshold: float = -0.3,
    good_switch_delta: float = 0.3,
):
    """Run comprehensive evaluation against all baselines."""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = RLConfig()

    # Load checkpoint first to get config
    print(f"\n📂 Loading: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Get model dimensions from checkpoint or use defaults
    ckpt_config = checkpoint.get('config', {})
    feature_dim = ckpt_config.get('feature_dim', config.feature_dim)
    d_model = ckpt_config.get('d_model', config.d_model)
    n_actions = ckpt_config.get('n_actions', config.n_actions)
    rnn_hidden = ckpt_config.get('rnn_hidden', config.rnn_hidden)
    rnn_layers = ckpt_config.get('rnn_layers', config.rnn_layers)
    model_type = checkpoint.get('model_type', 'feedforward')

    print(f"   Using checkpoint config: feature_dim={feature_dim}, d_model={d_model}")
    config.feature_dim = feature_dim

    if model_type == "recurrent":
        model = RecurrentActorCritic(
            feature_dim=feature_dim,
            d_model=d_model,
            n_actions=n_actions,
            rnn_hidden=rnn_hidden,
            rnn_layers=rnn_layers,
        ).to(device)
    else:
        model = ActorCritic(
            feature_dim=feature_dim,
            d_model=d_model,
            n_actions=n_actions
        ).to(device)

    # Handle different checkpoint formats
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    elif 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        raise KeyError(f"Checkpoint missing model weights. Keys: {list(checkpoint.keys())}")
    model.eval()

    # Print checkpoint info
    if 'total_steps' in checkpoint:
        print(f"   Training steps: {checkpoint['total_steps']:,}")
    if 'n_demos' in checkpoint:
        print(f"   Training demos: {checkpoint['n_demos']:,}")

    print(f"\n🎮 Comprehensive Evaluation ({n_battles} battles per opponent)")
    print("   This provides statistically significant results (95% CI)\n")

    opponents = ["random", "max_power", "heuristics"]
    results = {}
    all_stats = {}
    all_action_stats = {}
    all_outcome_stats = {}

    for opponent in opponents:
        print(f"   vs {opponent}...")
        log_handle = None
        if decision_log and decision_log_dir:
            log_dir = Path(decision_log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / f"decisions_{opponent}.jsonl"
            log_handle = log_path.open("w", encoding="utf-8")
        opportunity_handle = None
        if opportunity_log_dir:
            log_dir = Path(opportunity_log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / f"switch_opportunities_{opponent}.jsonl"
            opportunity_handle = log_path.open("w", encoding="utf-8")
        win_rate, stats, action_stats, outcome_stats = await evaluate_vs_opponent(
            model,
            config,
            opponent,
            n_battles,
            device,
            decision_log=decision_log,
            decision_log_every=decision_log_every,
            decision_log_topk=decision_log_topk,
            decision_log_max=decision_log_max,
            opportunity_log=bool(opportunity_log_dir),
            opportunity_log_topk=opportunity_log_topk,
            opportunity_log_max=opportunity_log_max,
            opportunity_log_include_features=opportunity_log_include_features,
            opportunity_log_all=opportunity_log_all,
            action_summary=action_summary,
            switch_mass_warn=switch_mass_warn,
            matchup_threshold=matchup_threshold,
            good_switch_delta=good_switch_delta,
            decision_log_file=log_handle,
            opportunity_log_file=opportunity_handle,
        )
        if log_handle:
            log_handle.close()
        if opportunity_handle:
            opportunity_handle.close()
        results[opponent] = win_rate
        all_stats[opponent] = stats
        all_action_stats[opponent] = action_stats
        all_outcome_stats[opponent] = outcome_stats
        ci_low, ci_high = confidence_interval(stats.wins, n_battles)
        print(f"      Result: {win_rate:.1%} (95% CI: {ci_low:.1%} - {ci_high:.1%})\n")
        if action_summary and action_stats:
            print_action_summary(action_stats, switch_mass_warn)
            if outcome_stats:
                print_outcome_summary(outcome_stats)
                print_outcome_bucket_summary(outcome_stats)
            print()

    # Print detailed summary
    print("=" * 70)
    print("📊 COMPREHENSIVE EVALUATION RESULTS")
    print("=" * 70)

    # Targets
    targets = {
        "random": 0.90,
        "max_power": 0.70,
        "heuristics": 0.60,
    }

    print(f"\n{'Opponent':<15} {'Win Rate':>10} {'95% CI':>18} {'Target':>10} {'Status':>10}")
    print("-" * 70)

    for opponent, win_rate in results.items():
        stats = all_stats[opponent]
        ci_low, ci_high = confidence_interval(stats.wins, n_battles)
        target = targets[opponent]

        if ci_low >= target:
            status = "✅ PASS"
        elif ci_high >= target:
            status = "⚠️ CLOSE"
        else:
            status = "❌ FAIL"

        ci_str = f"[{ci_low:.1%} - {ci_high:.1%}]"
        print(f"{opponent:<15} {win_rate:>10.1%} {ci_str:>18} {target:>10.0%} {status:>10}")

    print("-" * 70)

    # Detailed stats for heuristics (the hard one)
    h_stats = all_stats["heuristics"]
    if h_stats.wins > 0:
        avg_remaining = h_stats.total_remaining_pokemon / h_stats.wins
        print(f"\n📈 Heuristics Match Quality:")
        print(f"   Avg Pokemon remaining in wins: {avg_remaining:.1f}")
        print(f"   Close wins (1-2 left): {h_stats.close_wins} ({h_stats.close_wins/max(h_stats.wins,1):.0%} of wins)")
        print(f"   Dominant wins (4+ left): {h_stats.dominant_wins} ({h_stats.dominant_wins/max(h_stats.wins,1):.0%} of wins)")

    # Overall assessment
    avg_rate = sum(results.values()) / len(results)
    print(f"\n{'Overall Average':<15} {avg_rate:>10.1%}")

    passed = sum(1 for opp in results
                 if confidence_interval(all_stats[opp].wins, n_battles)[0] >= targets[opp])

    print(f"\nTargets met: {passed}/3")

    if passed == 3:
        print("\n🎉 ALL TARGETS ACHIEVED!")
    elif passed >= 2:
        print("\n📈 Good progress. Fine-tune to hit remaining target.")
    else:
        print("\n🔧 More training needed. Consider:")
        print("   - More diverse expert demonstrations")
        print("   - Longer training epochs")
        print("   - Reward function adjustments")

    if action_summary and all_action_stats:
        print_condensed_diagnostics(
            opponents,
            all_stats,
            all_action_stats,
            targets,
            switch_mass_warn,
        )

    print("=" * 70 + "\n")

    return results


def main():
    parser = argparse.ArgumentParser(description="Comprehensive Oranguru RL Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--battles", type=int, default=500,
                        help="Number of battles per opponent (default: 500)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick evaluation with 50 battles (high variance)")
    parser.add_argument("--decision-log", action="store_true",
                        help="Log compact decision traces during evaluation")
    parser.add_argument("--decision-log-every", type=int, default=10,
                        help="Log every N decisions when --decision-log is set")
    parser.add_argument("--decision-log-topk", type=int, default=3,
                        help="Top-k actions to print in decision logs")
    parser.add_argument("--decision-log-max", type=int, default=200,
                        help="Max decision logs to print (0 = no limit)")
    parser.add_argument("--decision-log-dir", type=str, default=None,
                        help="Directory to write decision logs as JSONL")
    parser.add_argument("--opportunity-log-dir", type=str, default=None,
                        help="Directory to write clean switch opportunity logs as JSONL")
    parser.add_argument("--opportunity-log-topk", type=int, default=3,
                        help="Top-k actions to include in opportunity logs")
    parser.add_argument("--opportunity-log-max", type=int, default=0,
                        help="Max opportunity logs per opponent (0 = no limit)")
    parser.add_argument("--opportunity-log-include-features", action="store_true",
                        help="Include features/masks in opportunity logs")
    parser.add_argument("--opportunity-log-all", action="store_true",
                        help="Log all clean switch opportunities (default: missed only)")
    parser.add_argument("--action-summary", action="store_true",
                        help="Print action summaries per opponent")
    parser.add_argument("--switch-mass-warn", type=float, default=0.05,
                        help="Threshold for low switch-mass warning stats")
    parser.add_argument("--matchup-threshold", type=float, default=-0.3,
                        help="Matchup threshold to flag bad matchups")
    parser.add_argument("--good-switch-delta", type=float, default=0.3,
                        help="Delta threshold for good switch quality stats")
    args = parser.parse_args()

    n_battles = 50 if args.quick else args.battles

    print("\n⚠️ Ensure Pokemon Showdown server is running!")
    print("   docker start pokemon-showdown\n")

    if args.quick:
        print("⚡ Quick mode: 50 battles (results will have high variance)")

    asyncio.run(
        full_evaluation(
            args.checkpoint,
            n_battles,
            decision_log=args.decision_log,
            decision_log_every=args.decision_log_every,
            decision_log_topk=args.decision_log_topk,
            decision_log_max=args.decision_log_max,
            decision_log_dir=args.decision_log_dir,
            opportunity_log_dir=args.opportunity_log_dir,
            opportunity_log_topk=args.opportunity_log_topk,
            opportunity_log_max=args.opportunity_log_max,
            opportunity_log_include_features=args.opportunity_log_include_features,
            opportunity_log_all=args.opportunity_log_all,
            action_summary=args.action_summary,
            switch_mass_warn=args.switch_mass_warn,
            matchup_threshold=args.matchup_threshold,
            good_switch_delta=args.good_switch_delta,
        )
    )


if __name__ == "__main__":
    main()
