#!/usr/bin/env python3
"""
🦧 ORANGURU RL - Imitation Learning Pretraining

Bootstrap the RL model by imitating RuleBotPlayer.
RuleBot beats SimpleHeuristics at 58%, so training on its demonstrations
should produce a model that also beats SimpleHeuristics.
"""

import asyncio
import sys
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from poke_env.player.baselines import SimpleHeuristicsPlayer
from src.utils.server_config import get_server_configuration

from training.config import RLConfig
from src.models.actor_critic import ActorCritic
from src.players.rl_player import RLPlayer
from src.players.rule_bot import RuleBotPlayer


class ExpertDataset(Dataset):
    """Dataset of (state, expert_action, mask, weight) tuples."""

    def __init__(self, states, actions, masks, weights=None):
        self.states = torch.tensor(states, dtype=torch.float32)
        self.actions = torch.tensor(actions, dtype=torch.long)
        self.masks = torch.tensor(masks, dtype=torch.bool)
        if weights is None:
            weights = [1.0] * len(states)
        self.weights = torch.tensor(weights, dtype=torch.float32)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx], self.masks[idx], self.weights[idx]


class ExpertDataCollector(RLPlayer):
    """Collects state-action pairs from expert demonstrator."""

    def __init__(self, expert, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.expert = expert
        self.demonstrations = []
        self._current_battle_demos = []  # Buffer for current battle
        self._battle_id = None

    def choose_move(self, battle):
        # Track battle ID to detect new battles
        if battle.battle_tag != self._battle_id:
            self._battle_id = battle.battle_tag
            self._current_battle_demos = []

        # Get our features
        features = self.feature_builder.build(battle)
        mask = self._build_mask(battle)

        # Get expert's action
        # RuleBot returns BattleOrder directly
        # SimpleHeuristics returns (SingleBattleOrder, score) tuple
        expert_result = self.expert.choose_move(battle)

        if expert_result is None:
            return self.choose_random_move(battle)

        # Handle both return types
        if isinstance(expert_result, tuple):
            expert_order = expert_result[0]  # SimpleHeuristics style
        else:
            expert_order = expert_result  # RuleBot style

        # Convert order to action index
        expert_action = self._order_to_action_idx(battle, expert_order)

        # Store demonstration in buffer
        if expert_action is not None:
            self._current_battle_demos.append({
                'features': features,
                'action': expert_action,
                'mask': mask
            })

        # Return the order directly
        return expert_order

    def commit_battle(self, won: bool, wins_only: bool = False, weight: float = 1.0,
                       pokemon_remaining: int = 0, advantage_weighted: bool = False):
        """Commit current battle demos if conditions met.

        Args:
            won: Whether the battle was won
            wins_only: If True, only commit winning battles
            weight: Base weight for all demos
            pokemon_remaining: How many Pokemon the winner had remaining
            advantage_weighted: If True, multiply weight based on dominance
        """
        if not wins_only or won:
            final_weight = weight

            if advantage_weighted and won and pokemon_remaining > 0:
                # Dominant wins (4+ Pokemon remaining) get highest weight
                if pokemon_remaining >= 5:
                    final_weight = weight * 4.0  # Crushing victory
                elif pokemon_remaining >= 4:
                    final_weight = weight * 3.0  # Dominant win
                elif pokemon_remaining >= 3:
                    final_weight = weight * 2.0  # Solid win
                elif pokemon_remaining >= 2:
                    final_weight = weight * 1.5  # Good win
                # 1 Pokemon left = base weight (close game)

            for demo in self._current_battle_demos:
                demo['weight'] = final_weight
            self.demonstrations.extend(self._current_battle_demos)
        self._current_battle_demos = []

    def _order_to_action_idx(self, battle, order):
        """Convert SingleBattleOrder to action index."""
        if order is None:
            return None

        # Get the underlying move or pokemon from the order
        move_or_pokemon = order.order if hasattr(order, 'order') else order

        if move_or_pokemon is None:
            return None

        # Check if it's a move (has base_power or category attribute)
        if hasattr(move_or_pokemon, 'base_power') or hasattr(move_or_pokemon, 'category'):
            # It's a move
            try:
                idx = battle.available_moves.index(move_or_pokemon)
                # Check for terastallize
                if hasattr(order, 'terastallize') and order.terastallize:
                    return 9 + idx
                return idx
            except ValueError:
                pass
        else:
            # Check if it's a Pokemon (switch)
            try:
                idx = battle.available_switches.index(move_or_pokemon)
                return 4 + idx
            except ValueError:
                pass

        return None


async def collect_expert_data(n_battles_per_opponent: int = 200):
    """Collect expert demonstrations from RuleBot.

    Strategy: RuleBot beats SimpleHeuristics at ~55%
    - Train model on ALL games (not just wins) for robust behavior
    - Weight heuristics games slightly higher (target opponent)
    """
    from poke_env.player import RandomPlayer
    from poke_env.player.baselines import MaxBasePowerPlayer

    # Custom server config (override via env vars).
    CustomServerConfig = get_server_configuration(default_port=8000)

    config = RLConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    kwargs = {
        'battle_format': 'gen9randombattle',
        'max_concurrent_battles': 1,
        'server_configuration': CustomServerConfig,
    }

    all_demonstrations = []

    # RuleBot as teacher - train on ALL games, not just wins
    # Format: (expert_name, ExpertClass, opponent_name, OpponentClass, n_battles, wins_only, weight)
    training_config = [
        # RuleBot vs Random (learn basics)
        ("RuleBot", RuleBotPlayer, "Random", RandomPlayer, n_battles_per_opponent, False, 1.0),

        # RuleBot vs MaxPower (learn to outplay aggression)
        ("RuleBot", RuleBotPlayer, "MaxPower", MaxBasePowerPlayer, n_battles_per_opponent, False, 1.0),

        # RuleBot vs SimpleHeuristics - ALL games with higher weight
        # Learning from both wins AND losses helps understand opponent patterns
        ("RuleBot", RuleBotPlayer, "SimpleHeuristics", SimpleHeuristicsPlayer,
         n_battles_per_opponent * 3, False, 1.5),
    ]

    for expert_name, ExpertClass, opp_name, OppClass, n_battles, wins_only, weight in training_config:
        mode_str = "(wins only)" if wins_only else f"(w={weight}x)"
        print(f"\n🎓 {expert_name} vs {opp_name} {mode_str} ({n_battles} battles)...")

        # Create dummy model (we only need features)
        model = ActorCritic(config.feature_dim, config.d_model, config.n_actions).to(device)

        # Expert player (the teacher)
        expert = ExpertClass(**kwargs)

        # Data collector
        collector = ExpertDataCollector(
            expert=expert,
            model=model,
            config=config,
            device=device,
            training=False,
            **kwargs
        )

        # Opponent
        opponent = OppClass(**kwargs)

        # Collect data one battle at a time
        wins = 0
        for i in range(n_battles):
            prev_wins = collector.n_won_battles
            await collector.battle_against(opponent, n_battles=1)
            won = collector.n_won_battles > prev_wins
            if won:
                wins += 1

            collector.commit_battle(won, wins_only=wins_only, weight=weight)

            if (i + 1) % 100 == 0:
                print(f"   Progress: {i+1}/{n_battles}, wins={wins}, demos={len(collector.demonstrations)}")

        print(f"   ✓ Collected {len(collector.demonstrations)} demos, win rate: {wins/n_battles:.1%}")

        all_demonstrations.extend(collector.demonstrations)

        # Cleanup
        del collector, opponent, expert

    print(f"\n✓ Total demonstrations: {len(all_demonstrations)}")
    return all_demonstrations


def train_imitation(model, dataset, config, device, epochs=50):
    """Train model via behavior cloning with weighted loss and learning rate scheduling."""
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    print(f"\n🎯 Training via imitation learning (weighted)...")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: 64")
    print(f"   Dataset size: {len(dataset)}")
    print(f"   Learning rate: 3e-4 -> 1e-5 (cosine annealing)")

    model.train()
    best_acc = 0
    best_state = None

    for epoch in range(epochs):
        total_loss = 0
        total_acc = 0
        n_batches = 0

        for features, actions, masks, weights in dataloader:
            features = features.to(device)
            actions = actions.to(device)
            masks = masks.to(device)
            weights = weights.to(device)

            # Forward pass - get raw logits without masking for loss calculation
            x = model.input_proj(features)
            x = F.relu(x)
            for layer in model.encoder_layers:
                residual = x
                x = layer['linear'](x)
                x = layer['norm'](x)
                x = F.relu(x)
                x = layer['dropout'](x)
                x = x + residual
            logits = model.actor(x)

            # Only mask for prediction, not for loss
            masked_logits = logits.masked_fill(~masks, float('-inf'))

            # Weighted cross-entropy loss - wins count more
            per_sample_loss = F.cross_entropy(logits, actions, reduction='none')
            loss = (per_sample_loss * weights).mean()

            # Skip if loss is invalid
            if not torch.isfinite(loss):
                continue

            # Accuracy using masked logits
            preds = masked_logits.argmax(dim=-1)
            acc = (preds == actions).float().mean()

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_acc += acc.item()
            n_batches += 1

        scheduler.step()

        if n_batches == 0:
            continue

        avg_loss = total_loss / n_batches
        avg_acc = total_acc / n_batches

        # Save best model
        if avg_acc > best_acc:
            best_acc = avg_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 5 == 0 or epoch == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f"   Epoch {epoch+1:3d}/{epochs}: Loss={avg_loss:.4f}, Acc={avg_acc:.1%}, LR={lr:.2e}")

    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"✓ Loaded best model with {best_acc:.1%} accuracy")

    print("✓ Imitation learning complete!")
    return model


async def rl_finetune_conservative(model, config, device, n_battles=200,
                                    kl_weight=0.25, loss_penalty=-0.8,
                                    turn_penalty=-0.01, lr=1e-5):
    """Conservative RL fine-tuning with KL constraint and exploitation rewards.

    Args:
        kl_weight: KL divergence penalty (higher = stay closer to reference)
        loss_penalty: Asymmetric penalty for losses (negative, harsher = more regret)
        turn_penalty: Per-turn penalty for efficiency (negative = prefer shorter games)
        lr: Learning rate
    """
    from src.players.rl_player import RLPlayer
    import gc
    import copy

    CustomServerConfig = get_server_configuration(default_port=8000)

    print(f"\n🎯 Conservative RL Fine-tuning vs Heuristics ({n_battles} battles)...")
    print(f"   KL weight: {kl_weight}, Loss penalty: {loss_penalty}, Turn penalty: {turn_penalty}")

    kwargs = {
        'battle_format': 'gen9randombattle',
        'max_concurrent_battles': 1,
        'server_configuration': CustomServerConfig,
    }

    # Store reference model for KL constraint
    reference_model = copy.deepcopy(model)
    reference_model.eval()
    for p in reference_model.parameters():
        p.requires_grad = False

    # Learning rate (can be adjusted for boss phase)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    wins = 0
    batch_features, batch_masks, batch_actions, batch_old_log_probs = [], [], [], []
    batch_won = []
    batch_n_turns = []  # Track turns per step for efficiency penalty

    for battle_num in range(n_battles):
        agent = RLPlayer(model=model, config=config, device=device, training=True, **kwargs)
        opponent = SimpleHeuristicsPlayer(**kwargs)
        agent.clear_rollout()

        try:
            prev_wins = agent.n_won_battles
            await asyncio.wait_for(agent.battle_against(opponent, n_battles=1), timeout=60)
            won = agent.n_won_battles > prev_wins
            if won:
                wins += 1
        except Exception:
            del agent, opponent
            continue

        rollout = agent.finalize_rollout(won)

        n_turns = len(rollout['features'])
        if n_turns >= 2:
            batch_features.extend(rollout['features'])
            batch_masks.extend(rollout['masks'])
            batch_actions.extend(rollout['actions'])
            batch_old_log_probs.extend(rollout['log_probs'])
            batch_won.extend([won] * n_turns)
            batch_n_turns.extend([n_turns] * n_turns)  # Track battle length

        del agent, opponent

        # Update every 20 battles
        if (battle_num + 1) % 20 == 0 and batch_features:
            features = torch.tensor(batch_features, dtype=torch.float, device=device)
            masks = torch.tensor(batch_masks, dtype=torch.bool, device=device)
            actions = torch.tensor(batch_actions, dtype=torch.long, device=device)
            won_tensor = torch.tensor(batch_won, dtype=torch.float, device=device)
            turns_tensor = torch.tensor(batch_n_turns, dtype=torch.float, device=device)

            # Compute advantages with asymmetric rewards and turn efficiency
            # Win: +1, Loss: harsh penalty (e.g., -0.8)
            base_advantage = torch.where(won_tensor == 1,
                                          torch.ones_like(won_tensor),
                                          loss_penalty * torch.ones_like(won_tensor))

            # Turn efficiency penalty: longer battles get penalized
            efficiency_penalty = turn_penalty * turns_tensor
            advantages = base_advantage + efficiency_penalty

            # Get current log probs
            log_probs, _, entropy = model.evaluate(features, masks, actions)

            # Get reference log probs for KL constraint
            with torch.no_grad():
                ref_log_probs, _, _ = reference_model.evaluate(features, masks, actions)

            # KL divergence penalty (keep close to reference)
            kl_div = (ref_log_probs.exp() * (ref_log_probs - log_probs)).mean()

            # Policy gradient with KL constraint
            policy_loss = -(log_probs * advantages).mean()
            kl_loss = kl_weight * kl_div

            loss = policy_loss + kl_loss

            if torch.isfinite(loss):
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.3)
                optimizer.step()

            batch_features, batch_masks, batch_actions, batch_old_log_probs, batch_won, batch_n_turns = [], [], [], [], [], []
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if (battle_num + 1) % 40 == 0:
            wr = wins / (battle_num + 1)
            print(f"   Battle {battle_num+1}/{n_battles}: Win rate = {wr:.1%}")

    final_wr = wins / n_battles
    print(f"   ✓ Fine-tuning complete: {wins}/{n_battles} = {final_wr:.1%}")
    return model


async def main():
    """Main pretraining pipeline: RuleBot Imitation Learning.

    Train on ALL games (wins and losses) to learn robust behavior.
    RuleBot achieves ~55% vs SimpleHeuristics.
    """
    print("=" * 60)
    print("🦧 ORANGURU RL - RuleBot Imitation Learning")
    print("=" * 60)
    print("Strategy: RuleBot (55% vs Heuristics) → All games → Pure imitation")

    config = RLConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Collect expert demonstrations (200 vs Random, 200 vs MaxPower, 600 vs Heuristics)
    demos = await collect_expert_data(n_battles_per_opponent=200)

    # Prepare dataset with weights
    states = [d['features'] for d in demos]
    actions = [d['action'] for d in demos]
    masks = [d['mask'] for d in demos]
    weights = [d.get('weight', 1.0) for d in demos]

    dataset = ExpertDataset(states, actions, masks, weights)

    # Create and train model
    model = ActorCritic(config.feature_dim, config.d_model, config.n_actions).to(device)
    model = train_imitation(model, dataset, config, device, epochs=100)

    # Save model
    save_path = f"{config.checkpoint_dir}/rulebot_imitation.pt"
    Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    torch.save({
        'model': model.state_dict(),
        'config': config,
        'n_demos': len(demos),
    }, save_path)

    print(f"\n💾 Saved model: {save_path}")
    print("\n📊 Evaluate with:")
    print(f"   python evaluation/evaluate.py --checkpoint {save_path}")


if __name__ == "__main__":
    asyncio.run(main())
