#!/usr/bin/env python3
"""
Diagnose illegal action rates in the RL model.

Measures:
1. How often the model would pick an illegal action WITHOUT masking
2. How often fallback behavior is triggered despite masking
"""

import asyncio
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from poke_env.player.baselines import SimpleHeuristicsPlayer
from src.utils.server_config import get_server_configuration

from training.config import RLConfig
from src.models.actor_critic import ActorCritic
from src.players.rl_player import RLPlayer


async def diagnose_model(checkpoint_path: str, n_battles: int = 100):
    """Run battles with illegal action tracking."""

    CustomServerConfig = get_server_configuration(default_port=8080)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = RLConfig()

    # Load model
    print(f"\n Loading: {checkpoint_path}")
    model = ActorCritic(config.feature_dim, config.d_model, config.n_actions).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    kwargs = {
        'battle_format': 'gen9randombattle',
        'max_concurrent_battles': 1,
        'server_configuration': CustomServerConfig,
    }

    # Create agent with tracking enabled
    agent = RLPlayer(
        model=model,
        config=config,
        device=device,
        training=False,
        track_illegal=True,  # Enable tracking
        **kwargs
    )

    opponent = SimpleHeuristicsPlayer(**kwargs)

    print(f"\n Running {n_battles} battles vs SimpleHeuristics with illegal action tracking...\n")

    await agent.battle_against(opponent, n_battles=n_battles)

    win_rate = agent.n_won_battles / n_battles
    illegal_rate = agent.illegal_picks / max(agent.total_picks, 1)
    fallback_rate = agent.fallback_used / max(agent.total_picks, 1)

    print("=" * 60)
    print(" ILLEGAL ACTION DIAGNOSTIC")
    print("=" * 60)
    print(f"\n Win rate vs Heuristics: {win_rate:.1%}")
    print(f"\n Total decisions made: {agent.total_picks}")
    print(f"\n Illegal picks (without masking): {agent.illegal_picks} ({illegal_rate:.1%})")
    print(f"   -> These are choices the model would make if masking was disabled")
    print(f"   -> High rate = model hasn't learned legal action space")
    print(f"\n Fallback usage: {agent.fallback_used} ({fallback_rate:.1%})")
    print(f"   -> Times execution fell back to move[0] or random")
    print(f"   -> High rate = masking isn't fully aligned with execution")
    print("=" * 60)

    # Interpretation
    print("\n INTERPRETATION:")
    if illegal_rate > 0.2:
        print("   HIGH illegal rate - model relies heavily on masking")
        print("   Training the model to predict legal actions could help")
    elif illegal_rate > 0.05:
        print("   MODERATE illegal rate - some reliance on masking")
    else:
        print("   LOW illegal rate - model learned legal action space well")

    if fallback_rate > 0.01:
        print(f"   WARNING: {fallback_rate:.1%} fallback rate - mask/execution mismatch")

    return {
        'win_rate': win_rate,
        'illegal_rate': illegal_rate,
        'fallback_rate': fallback_rate,
        'total_picks': agent.total_picks
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints/rl/pretrained_final.pt")
    parser.add_argument("--battles", type=int, default=100)
    args = parser.parse_args()

    asyncio.run(diagnose_model(args.checkpoint, args.battles))
