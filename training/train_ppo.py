#!/usr/bin/env python3
"""
🦧 ORANGURU RL - PPO Training (Fixed Rollout Structure)

Process one battle at a time to maintain proper episode structure.
"""

import asyncio
import sys
import time
import gc
from pathlib import Path
from collections import deque
from datetime import timedelta
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from poke_env.player import RandomPlayer
from poke_env.player.baselines import MaxBasePowerPlayer, SimpleHeuristicsPlayer
from src.utils.server_config import get_server_configuration

# Custom server config (override via env vars).
CustomServerConfig = get_server_configuration(default_port=8000)

from training.config import RLConfig
from src.models.actor_critic import ActorCritic
from src.players.rl_player import RLPlayer


class PPOBuffer:
    """Buffer with replay capability to prevent catastrophic forgetting."""

    def __init__(self, replay_size: int = 10000, replay_ratio: float = 0.25):
        self.replay_size = replay_size
        self.replay_ratio = replay_ratio
        self.clear()
        # Replay buffer stores complete transitions for mixing
        self.replay_buffer = deque(maxlen=replay_size)

    def clear(self):
        """Clear current buffer but keep replay."""
        self.features = []
        self.masks = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []

    def add_episode(self, rollout: dict):
        """Add a complete episode to the buffer."""
        n = len(rollout['features'])
        if n == 0:
            return

        # Ensure rewards and dones match features length
        rewards = list(rollout['rewards'])
        dones = list(rollout['dones'])

        # Pad if needed
        while len(rewards) < n:
            rewards.append(0.0)
        while len(dones) < n:
            dones.append(False)

        # Mark episode end
        dones[-1] = True

        # Add to current buffer
        self.features.extend(rollout['features'][:n])
        self.masks.extend(rollout['masks'][:n])
        self.actions.extend(rollout['actions'][:n])
        self.log_probs.extend(rollout['log_probs'][:n])
        self.values.extend(rollout['values'][:n])
        self.rewards.extend(rewards[:n])
        self.dones.extend(dones[:n])

        # Also add transitions to replay buffer
        for i in range(n):
            self.replay_buffer.append({
                'features': rollout['features'][i],
                'mask': rollout['masks'][i],
                'action': rollout['actions'][i],
                'log_prob': rollout['log_probs'][i],
                'value': rollout['values'][i],
                'reward': rewards[i],
                'done': dones[i],
            })

    def sample_with_replay(self):
        """Get current buffer + sampled replay experiences."""
        if not self.replay_buffer or self.replay_ratio <= 0:
            return self

        # How many replay samples to add
        n_replay = int(len(self.features) * self.replay_ratio)
        n_replay = min(n_replay, len(self.replay_buffer))

        if n_replay == 0:
            return self

        # Sample from replay
        replay_indices = np.random.choice(len(self.replay_buffer), n_replay, replace=False)

        # Create combined buffer
        combined = PPOBuffer(replay_size=0, replay_ratio=0)
        combined.features = list(self.features)
        combined.masks = list(self.masks)
        combined.actions = list(self.actions)
        combined.log_probs = list(self.log_probs)
        combined.values = list(self.values)
        combined.rewards = list(self.rewards)
        combined.dones = list(self.dones)

        for idx in replay_indices:
            trans = self.replay_buffer[idx]
            combined.features.append(trans['features'])
            combined.masks.append(trans['mask'])
            combined.actions.append(trans['action'])
            combined.log_probs.append(trans['log_prob'])
            combined.values.append(trans['value'])
            combined.rewards.append(trans['reward'])
            combined.dones.append(trans['done'])

        return combined

    def __len__(self):
        return len(self.features)


class PPOTrainer:
    """PPO trainer with proper episode handling."""
    
    def __init__(self, config: RLConfig, device: str = "cuda"):
        self.config = config
        self.device = device
        
        self.model = ActorCritic(
            feature_dim=config.feature_dim,
            d_model=config.d_model,
            n_actions=config.n_actions
        ).to(device)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr, eps=1e-5)
        
        self.total_steps = 0
        self.updates = 0
        self.current_stage = 0
        self.stage_wins = 0
        self.stage_battles = 0
        self.last_save_step = 0
        self.start_time = None
        
        self.recent_winrate = deque(maxlen=100)
        self.recent_entropy = deque(maxlen=50)
        self.recent_policy_loss = deque(maxlen=50)
        self.recent_value_loss = deque(maxlen=50)
        self.recent_rewards = deque(maxlen=100)
        
        self.entropy_coef = config.entropy_coef
        
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    def compute_gae(self, rewards, values, dones, gamma=0.99, lam=0.95):
        """Compute GAE with proper episode boundary handling."""
        advantages = []
        gae = 0
        
        # Convert to numpy for easier handling
        rewards = np.array(rewards)
        values = np.array(values)
        dones = np.array(dones, dtype=float)
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            # If this step is terminal, next_value should be 0
            next_value = next_value * (1 - dones[t])
            
            delta = rewards[t] + gamma * next_value - values[t]
            gae = delta + gamma * lam * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        
        advantages = np.array(advantages)
        returns = advantages + values
        
        return advantages, returns
    
    def update(self, buffer: PPOBuffer):
        """PPO update from buffer."""
        if len(buffer) < self.config.batch_size:
            return None
        
        # Prepare tensors
        features = torch.tensor(buffer.features, dtype=torch.float, device=self.device)
        masks = torch.tensor(buffer.masks, dtype=torch.bool, device=self.device)
        actions = torch.tensor(buffer.actions, dtype=torch.long, device=self.device)
        old_log_probs = torch.tensor(buffer.log_probs, dtype=torch.float, device=self.device)
        
        # Compute advantages
        advantages, returns = self.compute_gae(
            buffer.rewards, buffer.values, buffer.dones,
            self.config.gamma, self.config.gae_lambda
        )
        
        advantages = torch.tensor(advantages, dtype=torch.float, device=self.device)
        returns = torch.tensor(returns, dtype=torch.float, device=self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Debug: Print advantage stats
        # print(f"   [Debug] Advantages: mean={advantages.mean():.3f}, std={advantages.std():.3f}, min={advantages.min():.3f}, max={advantages.max():.3f}")
        
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        n_batches = 0
        
        indices = np.arange(len(buffer))
        
        for epoch in range(self.config.n_epochs):
            np.random.shuffle(indices)

            for start in range(0, len(buffer), self.config.batch_size):
                batch_idx = indices[start:start + self.config.batch_size]
                if len(batch_idx) < self.config.batch_size // 2:
                    continue
                
                # Get batch
                b_features = features[batch_idx]
                b_masks = masks[batch_idx]
                b_actions = actions[batch_idx]
                b_old_log_probs = old_log_probs[batch_idx]
                b_advantages = advantages[batch_idx]
                b_returns = returns[batch_idx]
                
                # Forward pass
                log_probs, values, entropy = self.model.evaluate(b_features, b_masks, b_actions)
                
                # Policy loss
                ratio = torch.exp(log_probs - b_old_log_probs)
                surr1 = ratio * b_advantages
                surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon, 
                                   1 + self.config.clip_epsilon) * b_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(values, b_returns)
                
                # Entropy loss
                entropy_loss = -entropy.mean()

                # Total loss
                loss = policy_loss + self.config.value_coef * value_loss + self.entropy_coef * entropy_loss

                # Skip obviously bad updates that could destabilize training
                if policy_loss.detach().item() > self.config.max_policy_loss:
                    continue

                # Update
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                n_batches += 1
        
        self.updates += 1
        
        if n_batches > 0:
            avg_policy_loss = total_policy_loss / n_batches
            avg_value_loss = total_value_loss / n_batches
            avg_entropy = total_entropy / n_batches

            self.recent_policy_loss.append(avg_policy_loss)
            self.recent_value_loss.append(avg_value_loss)
            self.recent_entropy.append(avg_entropy)

            # Dynamically bump entropy coefficient if policy collapses
            if avg_entropy < self.config.min_entropy:
                old_coef = self.entropy_coef
                # Cap the boost to avoid runaway exploration
                self.entropy_coef = min(old_coef * 1.1, 0.1)
                if self.entropy_coef > old_coef:
                    print(f"   🔧 Low entropy ({avg_entropy:.3f}); increasing entropy_coef to {self.entropy_coef:.3f}")

            return {
                'policy_loss': avg_policy_loss,
                'value_loss': avg_value_loss,
                'entropy': avg_entropy,
            }
        
        return None
    
    def should_advance_stage(self) -> bool:
        if self.current_stage >= len(self.config.curriculum_stages):
            return False
        
        stage = self.config.curriculum_stages[self.current_stage]
        stage_name, min_wr, min_battles, max_steps = stage
        
        if self.stage_battles < min_battles:
            return False
        
        win_rate = self.stage_wins / self.stage_battles
        return win_rate >= min_wr
    
    def advance_stage(self):
        if self.current_stage < len(self.config.curriculum_stages):
            old_name = self.config.curriculum_stages[self.current_stage][0]
            wr = self.stage_wins / max(self.stage_battles, 1)
            print(f"\n🎓 Completed '{old_name}': {wr:.1%} ({self.stage_battles} battles)")
        
        self.current_stage += 1
        self.stage_wins = 0
        self.stage_battles = 0
        
        if self.current_stage < len(self.config.curriculum_stages):
            new_name = self.config.curriculum_stages[self.current_stage][0]
            print(f"   → Moving to '{new_name}'\n")
    
    def save(self, path: str):
        torch.save({
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'total_steps': self.total_steps,
            'updates': self.updates,
            'current_stage': self.current_stage,
        }, path)
    
    def get_eta(self) -> str:
        if not self.start_time or self.total_steps == 0:
            return "..."
        elapsed = time.time() - self.start_time
        steps_per_sec = self.total_steps / elapsed
        if steps_per_sec > 0:
            remaining = (self.config.total_timesteps - self.total_steps) / steps_per_sec
            return str(timedelta(seconds=int(remaining)))
        return "?"


async def test_connection():
    print("🔌 Testing connection to port 8090...", end=" ", flush=True)
    try:
        p1 = RandomPlayer(battle_format='gen9randombattle',
                         server_configuration=CustomServerConfig)
        p2 = RandomPlayer(battle_format='gen9randombattle',
                         server_configuration=CustomServerConfig)
        await asyncio.wait_for(p1.battle_against(p2, n_battles=1), timeout=30)
        print("✓")
        return True
    except Exception as e:
        print(f"✗ ({e})")
        return False


def create_opponent(stage_name: str, config: RLConfig, device: str, model=None):
    """Create opponent for curriculum stage."""
    kwargs = {
        'battle_format': 'gen9randombattle',
        'max_concurrent_battles': 1,
        'server_configuration': CustomServerConfig,
    }
    
    if stage_name == "random":
        return RandomPlayer(**kwargs)
    elif stage_name == "max_power":
        return MaxBasePowerPlayer(**kwargs)
    elif stage_name == "heuristics":
        return SimpleHeuristicsPlayer(**kwargs)
    else:
        opp_model = ActorCritic(config.feature_dim, config.d_model, config.n_actions).to(device)
        opp_model.load_state_dict(model.state_dict())
        return RLPlayer(model=opp_model, config=config, device=device, training=False, **kwargs)


async def train(config: RLConfig):
    """Training loop with proper episode handling."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("=" * 60)
    print("🦧 ORANGURU RL - PPO Training (Fixed)")
    print("=" * 60)
    print(f"   Device: {device}")
    print(f"   Steps: {config.total_timesteps:,}")
    print(f"   LR: {config.lr}, Clip: {config.clip_epsilon}")
    
    if not await test_connection():
        print("❌ Server not responding")
        return
    
    trainer = PPOTrainer(config, device)
    trainer.start_time = time.time()

    # Create buffer with replay settings from config
    replay_size = getattr(config, 'replay_buffer_size', 10000)
    replay_ratio = getattr(config, 'replay_sample_ratio', 0.25)
    buffer = PPOBuffer(replay_size=replay_size, replay_ratio=replay_ratio)
    battles_since_update = 0
    update_every = 5  # Update every N battles
    
    stage_name = config.curriculum_stages[0][0]
    print(f"\n🎮 Starting with '{stage_name}' opponent")
    
    player_kwargs = {
        'battle_format': 'gen9randombattle',
        'max_concurrent_battles': 1,
        'server_configuration': CustomServerConfig,
    }
    
    # Create reusable players - only recreate every N battles
    agent = RLPlayer(
        model=trainer.model,
        config=config,
        device=device,
        training=True,
        **player_kwargs
    )
    opponent = create_opponent(stage_name, config, device)
    battles_with_current_players = 0
    max_battles_per_player = 100  # Recreate players every 100 battles
    
    print(f"\n{'Battle':<8} {'Steps':<10} {'Stage':<10} {'WR':<7} {'Ent':<8} {'P.Loss':<9} {'V.Loss':<9} {'Reward':<8} {'ETA':<10}")
    print("-" * 95)
    
    battle_num = 0
    
    while trainer.total_steps < config.total_timesteps:
        # Check stage advancement
        if trainer.should_advance_stage():
            trainer.advance_stage()
            if trainer.current_stage >= len(config.curriculum_stages):
                break
            stage_name = config.curriculum_stages[trainer.current_stage][0]
            # Must recreate players for new stage
            del agent, opponent
            gc.collect()
            agent = RLPlayer(model=trainer.model, config=config, device=device, 
                           training=True, **player_kwargs)
            opponent = create_opponent(stage_name, config, device, trainer.model)
            battles_with_current_players = 0
        
        # Recreate players periodically to avoid resource exhaustion
        if battles_with_current_players >= max_battles_per_player:
            del agent, opponent
            gc.collect()
            agent = RLPlayer(model=trainer.model, config=config, device=device,
                           training=True, **player_kwargs)
            opponent = create_opponent(stage_name, config, device, trainer.model)
            battles_with_current_players = 0
            print(f"   🔄 Recycled players at battle {battle_num}")

        # Reset agent rollout for new battle
        agent.clear_rollout()

        # Run battle
        try:
            prev_wins = agent.n_won_battles
            await asyncio.wait_for(
                agent.battle_against(opponent, n_battles=1),
                timeout=60
            )
            won = agent.n_won_battles > prev_wins
            battles_with_current_players += 1
        except Exception as e:
            print(f"\n   ⚠️ Battle error: {e}, recycling players...")
            del agent, opponent
            gc.collect()
            await asyncio.sleep(1)
            agent = RLPlayer(model=trainer.model, config=config, device=device,
                           training=True, **player_kwargs)
            opponent = create_opponent(stage_name, config, device, trainer.model)
            battles_with_current_players = 0
            continue
        
        battle_num += 1
        trainer.stage_battles += 1
        if won:
            trainer.stage_wins += 1
        trainer.recent_winrate.append(1 if won else 0)

        # Finalize rollout with terminal and last-step rewards
        rollout = agent.finalize_rollout(won)

        if rollout['features']:
            ep_reward = sum(rollout['rewards'])
            trainer.recent_rewards.append(ep_reward)

            # Debug: Print reward distribution every 100 battles
            if battle_num % 100 == 1:
                terminal_reward = config.reward_win if won else config.reward_lose
                print(f"\n   [Debug] Battle {battle_num}: won={won}, steps={len(rollout['features'])}")
                print(f"   [Debug] Rewards: sum={ep_reward:.1f}, terminal={terminal_reward:.1f}")
                print(f"   [Debug] Non-zero rewards: {sum(1 for r in rollout['rewards'] if abs(r) > 0.01)}/{len(rollout['rewards'])}")

            # Add to buffer
            buffer.add_episode(rollout)
            trainer.total_steps += len(rollout['features'])
            battles_since_update += 1
        
        # Update every N battles or when buffer is large enough
        if battles_since_update >= update_every and len(buffer) >= config.batch_size * 2:
            # Mix current buffer with replay samples to prevent forgetting
            update_buffer = buffer.sample_with_replay()
            losses = trainer.update(update_buffer)
            buffer.clear()  # Clear current but keep replay
            battles_since_update = 0
            
            # Log
            if battle_num % 10 == 0 and losses:
                wr = np.mean(trainer.recent_winrate) if trainer.recent_winrate else 0
                avg_reward = np.mean(trainer.recent_rewards) if trainer.recent_rewards else 0
                
                print(f"{battle_num:<8} {trainer.total_steps:<10,} {stage_name[:8]:<10} {wr:<7.1%} "
                      f"{losses['entropy']:<8.3f} {losses['policy_loss']:<9.4f} "
                      f"{losses['value_loss']:<9.3f} {avg_reward:<8.1f} {trainer.get_eta():<10}")
        
        # Save checkpoint
        if trainer.total_steps - trainer.last_save_step >= config.save_freq:
            path = f"{config.checkpoint_dir}/step_{trainer.total_steps}.pt"
            trainer.save(path)
            trainer.last_save_step = trainer.total_steps
            print(f"   💾 Saved: {path}")
        
        # Periodic cleanup
        if battle_num % 50 == 0:
            gc.collect()
    
    # Final save
    trainer.save(f"{config.checkpoint_dir}/final.pt")
    elapsed = timedelta(seconds=int(time.time() - trainer.start_time))
    print(f"\n✅ Done! {trainer.total_steps:,} steps in {elapsed}")


async def evaluate(model_path: str, n_battles: int = 50):
    """Evaluate model."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = RLConfig()
    
    model = ActorCritic(config.feature_dim, config.d_model, config.n_actions).to(device)
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    
    print(f"\n📊 Evaluating: {model_path}")
    print(f"   Steps: {ckpt.get('total_steps', '?'):,}")
    
    kwargs = {'battle_format': 'gen9randombattle', 'max_concurrent_battles': 5,
              'server_configuration': CustomServerConfig}
    
    for name, cls in [("Random", RandomPlayer), ("MaxPower", MaxBasePowerPlayer), 
                      ("Heuristics", SimpleHeuristicsPlayer)]:
        agent = RLPlayer(model=model, config=config, device=device, training=False, **kwargs)
        opp = cls(**kwargs)
        
        print(f"   vs {name}...", end=" ", flush=True)
        await agent.battle_against(opp, n_battles=n_battles)
        print(f"{agent.n_won_battles}/{n_battles} = {agent.n_won_battles/n_battles:.0%}")
        
        del agent, opp
        gc.collect()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--eval", type=str, default=None)
    args = parser.parse_args()
    
    print("\n⚠️ Ensure server is running: docker start pokemon-showdown\n")
    
    if args.eval:
        asyncio.run(evaluate(args.eval))
    else:
        asyncio.run(train(RLConfig(total_timesteps=args.timesteps)))
