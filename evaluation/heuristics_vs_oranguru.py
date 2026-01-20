#!/usr/bin/env python3
"""
Evaluate OranguruEnginePlayer vs SimpleHeuristicsPlayer (local server).
"""

import argparse
import asyncio
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from poke_env.player.baselines import SimpleHeuristicsPlayer  # noqa: E402
from src.players.oranguru_engine import OranguruEnginePlayer  # noqa: E402
from src.utils.server_config import get_server_configuration  # noqa: E402

CustomServerConfig = get_server_configuration(default_port=8000)


async def _safe_stop(player):
    ps_client = getattr(player, "ps_client", None)
    if ps_client is None:
        return
    stop_fn = getattr(ps_client, "stop_listening", None)
    if stop_fn is None:
        return
    try:
        await stop_fn()
    except Exception:
        pass


async def run(n_battles: int, progress_every: int, timeout_s: int, battle_format: str):
    kwargs = {
        "battle_format": battle_format,
        "max_concurrent_battles": 1,
        "server_configuration": CustomServerConfig,
    }
    wins = 0
    for i in range(n_battles):
        agent = OranguruEnginePlayer(**kwargs)
        opponent = SimpleHeuristicsPlayer(**kwargs)
        prev_wins = agent.n_won_battles
        try:
            await asyncio.wait_for(agent.battle_against(opponent, n_battles=1), timeout=timeout_s)
            if agent.n_won_battles > prev_wins:
                wins += 1
        except Exception:
            pass
        finally:
            await _safe_stop(agent)
            await _safe_stop(opponent)
            await asyncio.sleep(0.1)

        if (i + 1) % progress_every == 0:
            print(f"{i+1}/{n_battles}: {wins}/{i+1} wins ({wins/(i+1)*100:.1f}%)")

    print(f"Final: {wins}/{n_battles} wins ({wins/n_battles*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--battles", type=int, default=50)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--format", default="gen9randombattle")
    args = parser.parse_args()

    asyncio.run(run(args.battles, args.progress_every, args.timeout, args.format))


if __name__ == "__main__":
    main()
