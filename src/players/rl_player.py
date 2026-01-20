#!/usr/bin/env python3
"""
🦧 ORANGURU RL PLAYER - Enhanced Version

Uses 272-dimensional feature vector with:
- Speed comparisons
- Move priority/effects
- Ability interactions
- Better matchup analysis
- Item effect features

Rewards include:
- Win/loss terminal rewards
- KO/faint rewards (subordinate)
- HP damage (capped)
- Context-aware positioning rewards (safe vs risky setups)
"""

import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from poke_env.player import Player
from poke_env.battle import AbstractBattle

from src.utils.features import EnhancedFeatureBuilder
from training.config import RLConfig


# Status mapping
STATUS_TO_IDX = {'': 0, 'brn': 1, 'par': 2, 'slp': 3, 'psn': 4, 'tox': 5, 'frz': 6}


class RLPlayer(Player):
    """RL-based Pokemon Battle Agent with enhanced features."""

    def __init__(self, model, config: RLConfig, device: str = "cpu",
                 training: bool = False, track_illegal: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.config = config
        self.device = device
        self.training = training
        self.track_illegal = track_illegal

        # Feature builder
        self.feature_builder = EnhancedFeatureBuilder(
            enable_prediction_features=getattr(config, "prediction_features_enabled", False)
        )

        # Training state
        self.rollout = {
            'features': [], 'masks': [], 'actions': [],
            'log_probs': [], 'values': [], 'rewards': [], 'dones': []
        }
        self.prev_state = {}
        self._last_battle: AbstractBattle | None = None
        self._finalized = False
        self._battle_tag = None
        self._hidden = None

        # Illegal action tracking
        self.illegal_picks = 0
        self.total_picks = 0
        self.fallback_used = 0
    
    def choose_move(self, battle: AbstractBattle):
        """Choose move using RL policy."""
        self._last_battle = battle
        if battle.battle_tag != self._battle_tag:
            self._battle_tag = battle.battle_tag
            self._reset_hidden()
        features = self.feature_builder.build(battle)
        mask = self._build_mask(battle)

        # Validate features for NaN/Inf values
        features = [0.0 if (f != f or f > 1e6 or f < -1e6) else f for f in features]

        features_t = torch.tensor([features], dtype=torch.float, device=self.device)
        mask_t = torch.tensor([mask], dtype=torch.bool, device=self.device)

        with torch.no_grad():
            if getattr(self.model, "is_recurrent", False):
                logits, value, next_hidden = self.model.forward_step(features_t, self._hidden)
                self._hidden = next_hidden
                if self.track_illegal:
                    self.total_picks += 1
                    unmasked_action = logits.argmax(dim=-1).item()
                    if not mask[unmasked_action]:
                        self.illegal_picks += 1
                logits = self._apply_switch_bias(battle, logits, mask_t)
                logits = self._apply_attack_effectiveness_bias(battle, logits, mask_t)
                action, log_prob, entropy = self._select_action_from_logits(
                    logits, mask_t, deterministic=not self.training
                )
            else:
                logits, value = self.model.forward(features_t, mask_t)
                logits = self._apply_switch_bias(battle, logits, mask_t)
                logits = self._apply_attack_effectiveness_bias(battle, logits, mask_t)
                action, log_prob, entropy = self._select_action_from_logits(
                    logits, mask_t, deterministic=not self.training
                )

                # Track what model would pick WITHOUT masking
                if self.track_illegal:
                    self.total_picks += 1
                    # Get unmasked logits to see raw preference
                    unmasked_action = self.model.get_action(
                        features_t, torch.ones_like(mask_t), deterministic=True
                    )[0].item()
                    if not mask[unmasked_action]:
                        self.illegal_picks += 1

        action_idx = action.item()
        
        if self.training:
            # Calculate reward from previous turn
            reward = self._calc_reward(battle)
            if self.rollout['features']:
                self.rollout['rewards'].append(reward)
                self.rollout['dones'].append(False)
            
            # Store current transition
            self.rollout['features'].append(features)
            self.rollout['masks'].append(mask)
            self.rollout['actions'].append(action_idx)
            self.rollout['log_probs'].append(log_prob.item())
            self.rollout['values'].append(value.item())

            self._update_state(battle)
            self._finalized = False

        return self._action_to_order(battle, action_idx)

    def _reset_hidden(self):
        """Reset recurrent state at the start of each battle."""
        if getattr(self.model, "is_recurrent", False):
            self._hidden = self.model.init_hidden(1, self.device)
        else:
            self._hidden = None

    def _select_action_from_logits(self, logits: torch.Tensor, action_mask: torch.Tensor,
                                   deterministic: bool = False):
        """Sample an action from precomputed logits and mask."""
        masked_logits = logits.masked_fill(~action_mask, -1e9)
        masked_logits = torch.clamp(masked_logits, min=-1e8, max=1e8)
        probs = F.softmax(masked_logits, dim=-1)
        probs = torch.clamp(probs, min=1e-8)
        probs = probs / probs.sum(dim=-1, keepdim=True)
        dist = Categorical(probs)
        if deterministic:
            action = masked_logits.argmax(dim=-1)
        else:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy()

    def _apply_switch_bias(self, battle: AbstractBattle, logits: torch.Tensor,
                           action_mask: torch.Tensor) -> torch.Tensor:
        if not getattr(self.config, "switch_bias_enabled", False):
            return logits
        if battle is None or battle.force_switch:
            return logits
        if not getattr(battle, "available_switches", []):
            return logits
        threshold = getattr(self.config, "switch_bias_matchup_threshold", -0.3)
        strength = getattr(self.config, "switch_bias_strength", 0.0)
        if strength <= 0:
            return logits

        my_poke = battle.active_pokemon
        opp_poke = battle.opponent_active_pokemon
        matchup = self._estimate_matchup(my_poke, opp_poke)
        if matchup is None or matchup >= threshold:
            return logits

        if getattr(self.config, "switch_bias_require_good_switch", False):
            best_delta = None
            for candidate in getattr(battle, "available_switches", []):
                if not candidate:
                    continue
                cand_score = self._estimate_matchup(candidate, opp_poke)
                if cand_score is None:
                    continue
                delta = cand_score - matchup
                if best_delta is None or delta > best_delta:
                    best_delta = delta
            delta_threshold = getattr(self.config, "switch_bias_good_switch_delta", 0.3)
            if best_delta is None or best_delta < delta_threshold:
                return logits

        bias = torch.zeros_like(logits)
        bias_mask = action_mask[..., 4:9].float()
        bias[..., 4:9] = strength * bias_mask
        logits = logits + bias

        stay_penalty = getattr(self.config, "switch_stay_penalty_strength", 0.0)
        if stay_penalty > 0:
            stay_mask = action_mask.clone()
            stay_mask[..., 4:9] = False
            logits = logits - stay_penalty * stay_mask.float()

        return logits

    def _is_damaging_move(self, move) -> bool:
        if move is None:
            return False
        if getattr(move, "base_power", 0) > 0:
            return True
        return getattr(move, "damage", None) is not None

    def _apply_attack_effectiveness_bias(self, battle: AbstractBattle, logits: torch.Tensor,
                                         action_mask: torch.Tensor) -> torch.Tensor:
        if not getattr(self.config, "attack_eff_penalty_enabled", False):
            return logits
        if battle is None or getattr(battle, "force_switch", False):
            return logits
        if getattr(self.config, "attack_eff_require_switch", True):
            if not getattr(battle, "available_switches", []):
                return logits

        my_poke = battle.active_pokemon
        opp_poke = battle.opponent_active_pokemon
        if not my_poke or not opp_poke:
            return logits

        if getattr(self.config, "attack_eff_bad_matchup_only", True):
            threshold = getattr(self.config, "attack_eff_bad_matchup_threshold", -0.2)
            matchup = self._estimate_matchup(my_poke, opp_poke)
            if matchup is None or matchup >= threshold:
                return logits

        if getattr(self.config, "attack_eff_require_good_switch", False):
            best_delta = None
            current_matchup = self._estimate_matchup(my_poke, opp_poke)
            if current_matchup is None:
                return logits
            for candidate in getattr(battle, "available_switches", []):
                if not candidate:
                    continue
                cand_score = self._estimate_matchup(candidate, opp_poke)
                if cand_score is None:
                    continue
                delta = cand_score - current_matchup
                if best_delta is None or delta > best_delta:
                    best_delta = delta
            delta_threshold = getattr(self.config, "attack_eff_good_switch_delta", 0.3)
            if best_delta is None or best_delta < delta_threshold:
                return logits

        low_penalty = getattr(self.config, "attack_eff_low_penalty", 0.0)
        immune_penalty = getattr(self.config, "attack_eff_immune_penalty", 0.0)
        if low_penalty <= 0 and immune_penalty <= 0:
            return logits

        moves = list(getattr(battle, "available_moves", []))
        if not moves:
            return logits

        low_threshold = getattr(self.config, "attack_eff_low_threshold", 0.5)
        bias = torch.zeros_like(logits)
        can_tera = bool(getattr(battle, "can_tera", False))
        for i, move in enumerate(moves[:4]):
            if not self._is_damaging_move(move):
                continue
            move_type = getattr(move, "type", None)
            if move_type is None:
                continue
            try:
                eff = opp_poke.damage_multiplier(move_type)
            except Exception:
                continue
            penalty = 0.0
            if eff <= 0:
                penalty = immune_penalty
            elif eff <= low_threshold:
                penalty = low_penalty
            if penalty <= 0:
                continue
            if action_mask[..., i].any():
                bias[..., i] -= penalty
            if can_tera and action_mask[..., i + 9].any():
                bias[..., i + 9] -= penalty

        return logits + bias
    
    def _build_mask(self, battle: AbstractBattle) -> list:
        """Build legal action mask."""
        mask = [False] * 13
        
        # Force switch - only switches are available
        if battle.force_switch:
            for i in range(min(5, len(battle.available_switches))):
                mask[4 + i] = True
        else:
            # Regular moves
            for i in range(min(4, len(battle.available_moves))):
                mask[i] = True
            
            # Switches (if not trapped)
            for i in range(min(5, len(battle.available_switches))):
                mask[4 + i] = True
            
            # Tera moves
            if battle.can_tera:
                for i in range(min(4, len(battle.available_moves))):
                    mask[9 + i] = True
        
        # Ensure at least one action
        if not any(mask):
            if battle.available_moves:
                mask[0] = True
            elif battle.available_switches:
                mask[4] = True
        
        return mask
    
    def _action_to_order(self, battle: AbstractBattle, idx: int):
        """Convert action index to battle order."""
        # Handle force switch first
        if battle.force_switch:
            if battle.available_switches:
                switch_idx = idx - 4 if idx >= 4 and idx < 9 else 0
                switch_idx = min(switch_idx, len(battle.available_switches) - 1)
                switch_idx = max(switch_idx, 0)
                return self.create_order(battle.available_switches[switch_idx])
            else:
                if self.track_illegal:
                    self.fallback_used += 1
                return self.choose_random_move(battle)

        # Regular moves (0-3)
        if idx < 4:
            if idx < len(battle.available_moves):
                return self.create_order(battle.available_moves[idx])
            elif battle.available_moves:
                if self.track_illegal:
                    self.fallback_used += 1
                return self.create_order(battle.available_moves[0])

        # Switches (4-8)
        elif idx < 9:
            switch_idx = idx - 4
            if switch_idx < len(battle.available_switches):
                return self.create_order(battle.available_switches[switch_idx])
            elif battle.available_switches:
                if self.track_illegal:
                    self.fallback_used += 1
                return self.create_order(battle.available_switches[0])

        # Tera moves (9-12)
        else:
            move_idx = idx - 9
            if battle.can_tera and move_idx < len(battle.available_moves):
                return self.create_order(battle.available_moves[move_idx], terastallize=True)
            elif battle.available_moves:
                if self.track_illegal:
                    self.fallback_used += 1
                return self.create_order(battle.available_moves[0])

        # Final fallback
        if self.track_illegal:
            self.fallback_used += 1
        return self.choose_random_move(battle)
    
    def _update_state(self, battle: AbstractBattle):
        """Store state for reward calculation."""
        my_poke = battle.active_pokemon
        opp_poke = battle.opponent_active_pokemon

        self.prev_state = {
            'my_alive': len([p for p in battle.team.values() if not p.fainted]),
            'opp_alive': 6 - len([p for p in battle.opponent_team.values() if p.fainted]),
            # Track active Pokemon for switch/matchup rewards
            'my_active_species': my_poke.species if my_poke else None,
            'opp_active_species': opp_poke.species if opp_poke else None,
            'my_matchup': self._estimate_matchup(my_poke, opp_poke),
            # Track opponent status for status-landing rewards
            'opp_status': opp_poke.status.name if opp_poke and opp_poke.status else None,
            # Track hazards for hazard-setting rewards
            'opp_hazards': len(battle.opponent_side_conditions),
        }

    def _estimate_matchup(self, my_poke, opp_poke) -> float:
        """Estimate matchup score (positive = advantage). Simplified from RuleBot."""
        if not my_poke or not opp_poke:
            return 0.0

        try:
            # Type effectiveness
            my_offensive = max([opp_poke.damage_multiplier(t) for t in my_poke.types if t], default=1.0)
            their_offensive = max([my_poke.damage_multiplier(t) for t in opp_poke.types if t], default=1.0)

            score = 0.0
            # Offensive advantage
            if my_offensive >= 2.0:
                score += 1.0
            elif my_offensive <= 0.5:
                score -= 0.5

            # Defensive advantage
            if their_offensive >= 2.0:
                score -= 1.0
            elif their_offensive <= 0.5:
                score += 0.5

            # HP consideration
            score += (my_poke.current_hp_fraction - opp_poke.current_hp_fraction) * 0.3

            return score
        except Exception:
            return 0.0
    
    def _calc_reward(self, battle: AbstractBattle) -> float:
        """Calculate TERMINAL-DOMINANT reward with matchup-based signals.

        Key insight from RuleBot: Good play is about matchups and momentum,
        not raw HP damage. Reward:
        - KOs (major signal, but still < terminal)
        - Good switches (improved matchup)
        - Landing status (burn/para/sleep cripples opponent)
        - Setting hazards (long-term advantage)
        """
        if not self.prev_state:
            return 0.0

        reward = 0.0

        my_alive = len([p for p in battle.team.values() if not p.fainted])
        opp_alive = 6 - len([p for p in battle.opponent_team.values() if p.fainted])

        # KO/Faint rewards - meaningful but can't overwhelm terminal
        opp_kos = self.prev_state['opp_alive'] - opp_alive
        my_faints = self.prev_state['my_alive'] - my_alive
        reward += opp_kos * self.config.reward_ko      # +0.5 per KO
        reward += my_faints * self.config.reward_faint  # -0.5 per faint

        # Switch reward - did we improve our matchup?
        my_poke = battle.active_pokemon
        opp_poke = battle.opponent_active_pokemon
        prev_species = self.prev_state.get('my_active_species')

        if my_poke and prev_species and my_poke.species != prev_species:
            # We switched - evaluate if it was good
            prev_matchup = self.prev_state.get('my_matchup', 0.0)
            new_matchup = self._estimate_matchup(my_poke, opp_poke)
            matchup_delta = new_matchup - prev_matchup

            if matchup_delta > 0.5:
                reward += getattr(self.config, 'reward_good_switch', 0.3)
            elif matchup_delta < -0.5:
                reward += getattr(self.config, 'reward_bad_switch', -0.2)

        # Status landing reward - we applied status to opponent
        prev_opp_status = self.prev_state.get('opp_status')
        curr_opp_status = opp_poke.status.name if opp_poke and opp_poke.status else None

        if curr_opp_status and not prev_opp_status:
            # We applied a new status!
            reward += getattr(self.config, 'reward_status_land', 0.4)

        # Hazard setting reward
        prev_hazards = self.prev_state.get('opp_hazards', 0)
        curr_hazards = len(battle.opponent_side_conditions)

        if curr_hazards > prev_hazards:
            reward += getattr(self.config, 'reward_hazard_set', 0.3)

        return reward

    def finalize_rollout(self, won: bool) -> dict:
        """Add final transition reward and terminal signal once per battle."""
        if not self.training or self._finalized or not self.rollout['features']:
            return self.rollout

        # Capture reward from the last turn (no subsequent choose_move call)
        battle = self._last_battle
        if battle:
            final_step_reward = self._calc_reward(battle)
            self.rollout['rewards'].append(final_step_reward)
            self.rollout['dones'].append(False)

        # Always add terminal reward
        terminal_reward = self.config.reward_win if won else self.config.reward_lose
        if self.rollout['rewards']:
            if len(self.rollout['rewards']) < len(self.rollout['features']):
                self.rollout['rewards'].append(terminal_reward)
                self.rollout['dones'].append(True)
            else:
                self.rollout['rewards'][-1] += terminal_reward
                if len(self.rollout['dones']) < len(self.rollout['features']):
                    self.rollout['dones'].append(True)
                else:
                    self.rollout['dones'][-1] = True
        self._finalized = True
        return self.rollout

    def end_battle(self, won: bool):
        """Hook from poke-env when a battle finishes."""
        self.finalize_rollout(won)

    def get_rollout(self) -> dict:
        """Get collected rollout data."""
        return self.rollout

    def clear_rollout(self):
        """Clear rollout data."""
        self.rollout = {
            'features': [], 'masks': [], 'actions': [],
            'log_probs': [], 'values': [], 'rewards': [], 'dones': []
        }
        self.prev_state = {}
        self._last_battle = None
        self._finalized = False
        self._battle_tag = None
        self._reset_hidden()
