#!/usr/bin/env python3
"""
Full Training Pipeline - Sequence BC + Offline RL

Optimized for 32GB RAM + RTX 3080 10GB
"""

import asyncio
import gc
import os
import pickle
import random
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from poke_env.player import RandomPlayer
from poke_env.player.baselines import MaxBasePowerPlayer, SimpleHeuristicsPlayer
from poke_env.ps_client.account_configuration import AccountConfiguration
from src.utils.server_config import get_server_configuration

from training.config import RLConfig
from src.models.actor_critic import RecurrentActorCritic
from src.players.rl_player import RLPlayer
from src.players.rule_bot import RuleBotPlayer
try:
    from src.players.smart_heuristics import SmartHeuristicsPlayer
    HAS_SMART_HEURISTICS = True
except Exception:
    SmartHeuristicsPlayer = None
    HAS_SMART_HEURISTICS = False

USE_SMART_HEURISTICS = False

# Server config (override via env vars).
CustomServerConfig = get_server_configuration(default_port=8000)

RUN_TAG = f"{os.getpid() % 10000:04d}{random.randint(0, 99):02d}"


def _safe_username(name: str) -> str:
    clean = "".join(ch if ch.isalnum() else "_" for ch in name)
    if not clean:
        clean = "Player"
    return clean[:8]


def player_kwargs(base_kwargs: dict, name: str) -> dict:
    kwargs = dict(base_kwargs)
    key = f"{_safe_username(name)}_{RUN_TAG}"
    kwargs["account_configuration"] = AccountConfiguration.generate(key, rand=True)
    return kwargs


def apply_quick_overrides(config: RLConfig) -> None:
    """Apply short-run overrides from environment variables."""
    if os.getenv("ORANGURU_SHORT", "").lower() not in {"1", "true", "yes"}:
        return
    config.stream_rebuild = False
    config.rulebot_battles_per_opponent = int(os.getenv("ORANGURU_SHORT_BATTLES", "800"))
    config.rulebot_min_trajectories = int(os.getenv("ORANGURU_SHORT_MIN_TRAJ", "8000"))
    config.offline_bc_epochs = int(os.getenv("ORANGURU_SHORT_BC_EPOCHS", "60"))
    config.bc_focus_epochs = int(os.getenv("ORANGURU_SHORT_FOCUS_EPOCHS", "40"))
    config.bc_focus_eval_interval = int(os.getenv("ORANGURU_SHORT_FOCUS_EVAL", "10"))
    config.replay_max_trajectories = int(os.getenv("ORANGURU_SHORT_REPLAY_MAX", "5000"))


def get_mem_available_gb() -> float | None:
    """Return available system memory in GB (Linux)."""
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    kb = int(line.split()[1])
                    return kb / (1024 * 1024)
    except Exception:
        return None
    return None


def enforce_memory_budget(trajectories: list, config: RLConfig, label: str) -> list:
    """Trim trajectories if available RAM is below threshold."""
    avail = get_mem_available_gb()
    if avail is None or avail >= config.ram_min_available_gb:
        return trajectories
    if not trajectories:
        return trajectories

    target = max(config.ram_trim_keep_min, int(len(trajectories) * config.ram_trim_fraction))
    if len(trajectories) <= target:
        return trajectories

    priority = []
    other = []
    for t in trajectories:
        tag = t.get("tag", "")
        if any(key in tag for key in config.ram_priority_tags):
            priority.append(t)
        else:
            other.append(t)

    rng = random.Random(0)
    if len(priority) >= target:
        trimmed = rng.sample(priority, target)
    else:
        remaining = target - len(priority)
        if other:
            pick = rng.sample(other, min(len(other), remaining))
            trimmed = priority + pick
        else:
            trimmed = priority

    print(
        f"  RAM low ({avail:.1f} GB). Trimming {label}: {len(trajectories)} -> {len(trimmed)}"
    )
    gc.collect()
    return trimmed


def dataloader_settings(config: RLConfig, device: str) -> tuple[int, bool]:
    avail = get_mem_available_gb()
    low = avail is not None and avail < config.ram_min_available_gb * 2
    num_workers = 0 if low else 2
    pin_memory = (device == "cuda") and not low
    return num_workers, pin_memory


def wait_for_memory(config: RLConfig, label: str) -> None:
    """Pause training if available RAM is below threshold."""
    avail = get_mem_available_gb()
    if avail is None or avail >= config.ram_pause_available_gb:
        return
    print(
        f"  Low RAM before {label}: {avail:.1f} GB available. "
        f"Pausing to avoid crash..."
    )
    while True:
        time.sleep(config.ram_pause_check_seconds)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        avail = get_mem_available_gb()
        if avail is None or avail >= config.ram_pause_available_gb:
            break
    if avail is not None:
        print(f"  RAM recovered: {avail:.1f} GB. Resuming {label}.")


class TrajectoryStreamWriter:
    """Stream trajectories to disk in chunked pickles to avoid RAM spikes."""

    def __init__(self, config: RLConfig, prefix: str, append_existing: bool = False):
        self.config = config
        self.prefix = prefix
        self.chunk_dir = Path(config.stream_cache_dir)
        self.chunk_dir.mkdir(parents=True, exist_ok=True)

        if config.stream_rebuild and not append_existing:
            for path in self.chunk_dir.glob(f"{prefix}_chunk_*.pkl"):
                path.unlink(missing_ok=True)

        self.chunk_idx = 0
        if not config.stream_rebuild or append_existing:
            existing = sorted(self.chunk_dir.glob(f"{prefix}_chunk_*.pkl"))
            if existing:
                tail = existing[-1].stem.split("_")[-1]
                try:
                    self.chunk_idx = int(tail) + 1
                except ValueError:
                    self.chunk_idx = len(existing)

        self.buffer: list = []
        self.total_added = 0
        self.total_steps = 0

    def append(self, trajectories: list) -> None:
        if not trajectories:
            return
        self.buffer.extend(trajectories)
        self.total_added += len(trajectories)
        self.total_steps += sum(len(t.get("actions", [])) for t in trajectories)

        while len(self.buffer) >= self.config.stream_chunk_size:
            chunk = self.buffer[:self.config.stream_chunk_size]
            del self.buffer[:self.config.stream_chunk_size]
            path = self.chunk_dir / f"{self.prefix}_chunk_{self.chunk_idx:05d}.pkl"
            with open(path, "wb") as f:
                pickle.dump(chunk, f)
            self.chunk_idx += 1

    def finalize(self) -> None:
        if not self.buffer:
            return
        path = self.chunk_dir / f"{self.prefix}_chunk_{self.chunk_idx:05d}.pkl"
        with open(path, "wb") as f:
            pickle.dump(self.buffer, f)
        self.chunk_idx += 1
        self.buffer = []


def write_trajectory_chunks(trajectories: list, config: RLConfig, prefix: str) -> list[Path]:
    """Write trajectories to chunked pickle files and return paths."""
    chunk_dir = Path(config.stream_cache_dir)
    chunk_dir.mkdir(parents=True, exist_ok=True)
    if config.stream_rebuild:
        for path in chunk_dir.glob(f"{prefix}_chunk_*.pkl"):
            path.unlink(missing_ok=True)

    paths: list[Path] = []
    for idx in range(0, len(trajectories), config.stream_chunk_size):
        chunk = trajectories[idx:idx + config.stream_chunk_size]
        path = chunk_dir / f"{prefix}_chunk_{idx // config.stream_chunk_size:05d}.pkl"
        with open(path, "wb") as f:
            pickle.dump(chunk, f)
        paths.append(path)
    return paths


def load_chunk(path: Path) -> list:
    with open(path, "rb") as f:
        return pickle.load(f)


def phase_after(phase: str) -> str:
    mapping = {
        "imitation": "bc_focus",
        "bc_focus": "bc_switch_focus",
        "bc_switch_focus": "offline_rl",
        "offline_rl": "offline_focus",
        "offline_focus": "selfplay",
        "offline_selfplay": "done",
    }
    return mapping.get(phase, "bc")


def battle_won_from_player(player, prev_wins: int) -> bool:
    """Determine if the last battle was a win without relying on battle history."""
    battle = getattr(player, "_last_battle", None)
    if battle is not None and getattr(battle, "finished", False):
        try:
            return bool(battle.won)
        except Exception:
            pass
    return player.n_won_battles > prev_wins


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.1f}m"
    hours = int(minutes // 60)
    mins = minutes % 60
    return f"{hours}h{mins:02.0f}m"


# ============================================================================
# PHASE 1: Trajectory Collection from RuleBot
# ============================================================================

class ExpertTrajectoryCollector(RLPlayer):
    """Collect expert trajectories with rewards for offline RL."""

    def __init__(self, expert: RuleBotPlayer, weight: float = 1.0,
                 wins_only: bool = False, advantage_weighted: bool = False,
                 min_remaining: int = 0, tag: str = "", **kwargs):
        super().__init__(**kwargs)
        self.expert = expert
        self.weight = weight
        self.wins_only = wins_only
        self.advantage_weighted = advantage_weighted
        self.min_remaining = min_remaining
        self.tag = tag
        self.trajectories = []
        self._current = None
        self._battle_id = None

    def _reset_current(self):
        self._current = {
            'features': [],
            'masks': [],
            'actions': [],
            'rewards': [],
            'dones': [],
            'matchup_prev': [],
            'matchup_new': [],
            'force_switch': [],
            'weight': self.weight,
            'tag': self.tag,
        }

    def choose_move(self, battle):
        self._last_battle = battle
        battle_id = battle.battle_tag if battle.battle_tag is not None else id(battle)
        if self._current is None or battle_id != self._battle_id:
            self._battle_id = battle_id
            self._reset_current()
            self.prev_state = {}

        if self._current['features']:
            reward = self._calc_reward(battle)
            self._current['rewards'].append(reward)
            self._current['dones'].append(False)

        features = self.feature_builder.build(battle)
        mask = self._build_mask(battle)
        my_poke = battle.active_pokemon
        opp_poke = battle.opponent_active_pokemon
        prev_matchup = None
        if getattr(self, "prev_state", None):
            prev_matchup = self.prev_state.get("my_matchup")
        new_matchup = self._estimate_matchup(my_poke, opp_poke)
        force_switch = False
        if hasattr(battle, "force_switch"):
            if isinstance(battle.force_switch, list):
                force_switch = any(battle.force_switch)
            else:
                force_switch = bool(battle.force_switch)

        expert_result = self.expert.choose_move(battle)
        if expert_result is None:
            return self.choose_random_move(battle)
        expert_order = expert_result[0] if isinstance(expert_result, tuple) else expert_result

        action = self._order_to_action_idx(battle, expert_order)
        if action is None:
            action = 0

        self._current['features'].append(features)
        self._current['masks'].append(mask)
        self._current['actions'].append(action)
        self._current['matchup_prev'].append(prev_matchup)
        self._current['matchup_new'].append(new_matchup)
        self._current['force_switch'].append(force_switch)
        self._update_state(battle)
        return expert_order

    def _finalize_current(self, won: bool):
        if not self._current or not self._current['features']:
            return
        remaining = 0
        if self._last_battle:
            remaining = len([p for p in self._last_battle.team.values() if not p.fainted])
        if self.wins_only and not won:
            self._current = None
            return
        if self.min_remaining and remaining < self.min_remaining:
            self._current = None
            return
        if self.advantage_weighted and won and self._last_battle:
            if remaining >= 5:
                self._current['weight'] *= 4.0
            elif remaining >= 4:
                self._current['weight'] *= 3.0
            elif remaining >= 3:
                self._current['weight'] *= 2.0
            elif remaining >= 2:
                self._current['weight'] *= 1.5
        if self._last_battle:
            final_step_reward = self._calc_reward(self._last_battle)
            self._current['rewards'].append(final_step_reward)
            self._current['dones'].append(False)
        terminal_reward = self.config.reward_win if won else self.config.reward_lose
        if len(self._current['rewards']) < len(self._current['features']):
            self._current['rewards'].append(terminal_reward)
            self._current['dones'].append(True)
        else:
            self._current['rewards'][-1] += terminal_reward
            if len(self._current['dones']) < len(self._current['features']):
                self._current['dones'].append(True)
            else:
                self._current['dones'][-1] = True
        self.trajectories.append(self._current)
        self._current = None

    def end_battle(self, won: bool):
        self._finalize_current(won)

    def _order_to_action_idx(self, battle, order):
        if order is None or not hasattr(order, 'order'):
            return None

        move_or_pokemon = order.order
        if move_or_pokemon is None:
            return None

        if hasattr(move_or_pokemon, 'base_power') or hasattr(move_or_pokemon, 'category'):
            try:
                idx = battle.available_moves.index(move_or_pokemon)
                if hasattr(order, 'terastallize') and order.terastallize:
                    return 9 + idx
                return idx
            except ValueError:
                pass
        else:
            try:
                idx = battle.available_switches.index(move_or_pokemon)
                return 4 + idx
            except ValueError:
                pass
        return None


async def collect_trajectories(
    config: RLConfig,
    n_battles_per_opponent: int = 500,
) -> tuple[list, dict | None]:
    """Collect expert trajectories from RuleBot vs multiple opponents."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    kwargs = {
        'battle_format': 'gen9randombattle',
        'max_concurrent_battles': 1,
        'server_configuration': CustomServerConfig,
    }

    all_trajectories: list = []
    stream_writer = None
    if config.stream_trajectories and not config.stream_keep_in_memory:
        stream_writer = TrajectoryStreamWriter(config, prefix="full")

    total_written_steps = 0

    def drain_collector(collector):
        nonlocal total_written_steps
        if not collector.trajectories:
            return
        total_written_steps += sum(len(t.get("actions", [])) for t in collector.trajectories)
        if stream_writer:
            stream_writer.append(collector.trajectories)
        else:
            all_trajectories.extend(collector.trajectories)
        collector.trajectories = []

    matchups = []
    if config.rulebot_collect_random:
        matchups.append((
            "Random", RandomPlayer, n_battles_per_opponent, config.rulebot_weight_random
        ))
    if config.rulebot_collect_maxpower:
        matchups.append((
            "MaxPower", MaxBasePowerPlayer, n_battles_per_opponent, config.rulebot_weight_maxpower
        ))
    if config.rulebot_collect_heuristics:
        matchups.append((
            "Heuristics",
            SimpleHeuristicsPlayer,
            int(n_battles_per_opponent * config.rulebot_heuristics_multiplier),
            config.rulebot_weight_heuristics,
        ))

    for opp_name, OppClass, n_battles, weight in matchups:
        print(f"\n  Collecting vs {opp_name} ({n_battles} battles, weight={weight})...")
        matchup_start = time.time()

        model = RecurrentActorCritic(
            config.feature_dim, config.d_model, config.n_actions,
            rnn_hidden=config.rnn_hidden, rnn_layers=config.rnn_layers
        ).to(device)
        expert = RuleBotPlayer(**player_kwargs(kwargs, "rulebot"))

        collector = ExpertTrajectoryCollector(
            expert=expert,
            weight=weight,
            tag=f"rulebot_vs_{opp_name.lower()}",
            model=model,
            config=config,
            device=device,
            training=False,
            **player_kwargs(kwargs, "collector")
        )

        opponent = OppClass(**player_kwargs(kwargs, f"{opp_name}_opp"))

        wins = 0
        for i in range(n_battles):
            prev_wins = collector.n_won_battles
            await collector.battle_against(opponent, n_battles=1)
            won = battle_won_from_player(collector, prev_wins)
            if won:
                wins += 1
            collector._finalize_current(won)
            collector._last_battle = None
            try:
                collector.reset_battles()
            except Exception:
                pass
            try:
                opponent.reset_battles()
            except Exception:
                pass

            if (i + 1) % 100 == 0:
                wait_for_memory(config, f"collect {opp_name} {i+1}")
                if stream_writer and len(collector.trajectories) >= config.stream_chunk_size:
                    drain_collector(collector)
                buffer_steps = sum(len(t['actions']) for t in collector.trajectories)
                total_steps = total_written_steps + buffer_steps
                wr = wins / (i + 1)
                elapsed = time.time() - matchup_start
                rate = (i + 1) / elapsed if elapsed > 0 else 0.0
                eta = (n_battles - (i + 1)) / rate if rate > 0 else 0.0
                print(
                    f"    {i+1}/{n_battles}: {wr:.1%} win rate, {total_steps} steps "
                    f"| ETA {eta/60:.1f}m"
                )

        print(f"  Done: {wins}/{n_battles} = {wins/n_battles:.1%}")
        drain_collector(collector)
        if all_trajectories:
            all_trajectories = enforce_memory_budget(
                all_trajectories, config, label=f"collect_{opp_name.lower()}"
            )

        del collector, opponent, expert, model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Extra win-only focus vs Heuristics to push above 60%
    focus_matchups = []
    if config.rulebot_collect_wins_only:
        focus_matchups.append((
            "Heuristics",
            SimpleHeuristicsPlayer,
            int(n_battles_per_opponent * config.rulebot_wins_multiplier),
            config.rulebot_weight_wins,
        ))
    for opp_name, OppClass, n_battles, weight in focus_matchups:
        print(f"\n  Collecting RuleBot (wins-only) vs {opp_name} ({n_battles} battles, weight={weight})...")
        matchup_start = time.time()

        model = RecurrentActorCritic(
            config.feature_dim, config.d_model, config.n_actions,
            rnn_hidden=config.rnn_hidden, rnn_layers=config.rnn_layers
        ).to(device)
        expert = RuleBotPlayer(**player_kwargs(kwargs, "rulebot"))

        collector = ExpertTrajectoryCollector(
            expert=expert,
            weight=weight,
            wins_only=True,
            advantage_weighted=True,
            min_remaining=config.offline_focus_min_remaining,
            tag=f"rulebot_wins_vs_{opp_name.lower()}",
            model=model,
            config=config,
            device=device,
            training=False,
            **player_kwargs(kwargs, "collector")
        )

        opponent = OppClass(**player_kwargs(kwargs, f"{opp_name}_opp"))
        wins = 0
        for i in range(n_battles):
            prev_wins = collector.n_won_battles
            await collector.battle_against(opponent, n_battles=1)
            won = battle_won_from_player(collector, prev_wins)
            if won:
                wins += 1
            collector._finalize_current(won)
            collector._last_battle = None
            try:
                collector.reset_battles()
            except Exception:
                pass
            try:
                opponent.reset_battles()
            except Exception:
                pass

            if (i + 1) % 100 == 0:
                wait_for_memory(config, f"collect wins {opp_name} {i+1}")
                if stream_writer and len(collector.trajectories) >= config.stream_chunk_size:
                    drain_collector(collector)
                buffer_steps = sum(len(t['actions']) for t in collector.trajectories)
                total_steps = total_written_steps + buffer_steps
                wr = wins / (i + 1)
                elapsed = time.time() - matchup_start
                rate = (i + 1) / elapsed if elapsed > 0 else 0.0
                eta = (n_battles - (i + 1)) / rate if rate > 0 else 0.0
                print(
                    f"    {i+1}/{n_battles}: {wr:.1%} win rate, {total_steps} steps "
                    f"| ETA {eta/60:.1f}m"
                )

        print(f"  Done: {wins}/{n_battles} = {wins/n_battles:.1%}")
        drain_collector(collector)
        if all_trajectories:
            all_trajectories = enforce_memory_budget(
                all_trajectories, config, label=f"collect_wins_{opp_name.lower()}"
            )

        del collector, opponent, expert, model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if USE_SMART_HEURISTICS and HAS_SMART_HEURISTICS:
        smart_matchups = [
            ("Heuristics", SimpleHeuristicsPlayer, n_battles_per_opponent * 2, 2.0),
        ]
        for opp_name, OppClass, n_battles, weight in smart_matchups:
            print(f"\n  Collecting SmartHeuristics vs {opp_name} ({n_battles} battles, weight={weight})...")
            matchup_start = time.time()

            model = RecurrentActorCritic(
                config.feature_dim, config.d_model, config.n_actions,
                rnn_hidden=config.rnn_hidden, rnn_layers=config.rnn_layers
            ).to(device)
            expert = SmartHeuristicsPlayer(**player_kwargs(kwargs, "smart"))

            collector = ExpertTrajectoryCollector(
                expert=expert,
                weight=weight,
                wins_only=True,
                advantage_weighted=True,
                model=model,
                config=config,
                device=device,
                training=False,
                **player_kwargs(kwargs, "collector")
            )

            opponent = OppClass(**player_kwargs(kwargs, f"{opp_name}_opp"))
            wins = 0
            for i in range(n_battles):
                prev_wins = collector.n_won_battles
                await collector.battle_against(opponent, n_battles=1)
                won = battle_won_from_player(collector, prev_wins)
                if won:
                    wins += 1
                collector._finalize_current(won)
                collector._last_battle = None
                try:
                    collector.reset_battles()
                except Exception:
                    pass
                try:
                    opponent.reset_battles()
                except Exception:
                    pass

                if (i + 1) % 100 == 0:
                    wait_for_memory(config, f"collect smart {opp_name} {i+1}")
                    if stream_writer and len(collector.trajectories) >= config.stream_chunk_size:
                        drain_collector(collector)
                    buffer_steps = sum(len(t['actions']) for t in collector.trajectories)
                    total_steps = total_written_steps + buffer_steps
                    wr = wins / (i + 1)
                    elapsed = time.time() - matchup_start
                    rate = (i + 1) / elapsed if elapsed > 0 else 0.0
                    eta = (n_battles - (i + 1)) / rate if rate > 0 else 0.0
                    print(
                        f"    {i+1}/{n_battles}: {wr:.1%} win rate, {total_steps} steps "
                        f"| ETA {eta/60:.1f}m"
                    )

            print(f"  Done: {wins}/{n_battles} = {wins/n_battles:.1%}")
            drain_collector(collector)

            del collector, opponent, expert, model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    else:
        if USE_SMART_HEURISTICS:
            print("\n  SmartHeuristicsPlayer not available; skipping expert boost.")

    stream_stats = None
    if stream_writer:
        stream_writer.finalize()
        stream_stats = {
            "total_trajectories": stream_writer.total_added,
            "total_steps": stream_writer.total_steps,
            "chunk_dir": str(stream_writer.chunk_dir),
        }
        return [], stream_stats

    return all_trajectories, stream_stats


def discounted_returns(rewards, gamma: float) -> list:
    """Compute discounted returns for a trajectory."""
    returns = []
    running = 0.0
    for r in reversed(rewards):
        running = r + gamma * running
        returns.append(running)
    returns.reverse()
    return returns


class TrajectoryDataset(Dataset):
    """Dataset of full trajectories for sequence training."""

    def __init__(self, trajectories: list, gamma: float, include_returns: bool = True):
        self.trajectories = []
        for traj in trajectories:
            actions = traj.get('actions')
            if actions is None or len(actions) == 0:
                continue
            item = dict(traj)
            if include_returns:
                returns = discounted_returns(traj['rewards'], gamma)
                try:
                    import numpy as np
                    item['returns'] = np.asarray(returns, dtype=np.float32)
                except Exception:
                    item['returns'] = returns
            self.trajectories.append(item)

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        return self.trajectories[idx]


def collate_trajectories(batch):
    """Pad trajectories to the max length in the batch."""
    batch_size = len(batch)
    max_len = max(len(item['actions']) for item in batch)
    feature_dim = len(batch[0]['features'][0])
    n_actions = len(batch[0]['masks'][0])

    features = torch.zeros(batch_size, max_len, feature_dim, dtype=torch.float32)
    masks = torch.ones(batch_size, max_len, n_actions, dtype=torch.bool)
    actions = torch.zeros(batch_size, max_len, dtype=torch.long)
    returns = torch.zeros(batch_size, max_len, dtype=torch.float32)
    valid = torch.zeros(batch_size, max_len, dtype=torch.bool)
    weights = torch.ones(batch_size, max_len, dtype=torch.float32)
    matchups = torch.full((batch_size, max_len), float("nan"), dtype=torch.float32)

    for i, traj in enumerate(batch):
        length = len(traj['actions'])
        features[i, :length] = torch.tensor(traj['features'], dtype=torch.float32)
        masks[i, :length] = torch.tensor(traj['masks'], dtype=torch.bool)
        actions[i, :length] = torch.tensor(traj['actions'], dtype=torch.long)
        if 'returns' in traj:
            returns[i, :length] = torch.tensor(traj['returns'], dtype=torch.float32)
        valid[i, :length] = True
        weights[i, :length] = float(traj.get('weight', 1.0))
        if 'matchup_new' in traj:
            try:
                matchups[i, :length] = torch.tensor(traj['matchup_new'], dtype=torch.float32)
            except Exception:
                pass

    return features, masks, actions, returns, valid, weights, matchups


def filter_trajectories(trajectories, tag_contains: list[str]):
    """Filter trajectories by tag substrings."""
    filtered = []
    for t in trajectories:
        tag = t.get('tag', '')
        if any(key in tag for key in tag_contains):
            filtered.append(t)
    return filtered


def mix_trajectories(primary: list, secondary: list, ratio: float, seed: int = 0) -> list:
    """Mix two trajectory lists with a target primary ratio."""
    if not primary and not secondary:
        return []
    if not secondary or ratio >= 1.0:
        return list(primary)
    if not primary or ratio <= 0.0:
        return list(secondary)
    rng = random.Random(seed)
    primary = list(primary)
    secondary = list(secondary)
    rng.shuffle(primary)
    rng.shuffle(secondary)
    target_primary = int((len(primary) + len(secondary)) * ratio)
    target_primary = max(1, min(len(primary), target_primary))
    target_secondary = max(1, min(len(secondary), (target_primary * (1 - ratio)) / ratio))
    mixed = primary[:target_primary] + secondary[:target_secondary]
    rng.shuffle(mixed)
    return mixed


def extract_switch_only_trajectories(
    trajectories: list,
    min_actions: int = 1,
    weight_scale: float = 1.0,
    skip_forced: bool = True,
    tag_allowlist: tuple | None = None,
) -> list:
    """Extract switch-only steps from trajectories for switch-focused BC."""
    switch_only = []
    for traj in trajectories:
        actions = traj.get("actions", [])
        if not actions:
            continue
        if tag_allowlist:
            tag = traj.get("tag", "")
            if not any(t in tag for t in tag_allowlist):
                continue
        force_flags = traj.get("force_switch", [])
        switch_indices = []
        for i, action in enumerate(actions):
            if 4 <= action < 9:
                if skip_forced and i < len(force_flags) and force_flags[i]:
                    continue
                switch_indices.append(i)
        if len(switch_indices) < min_actions:
            continue
        features = [traj["features"][i] for i in switch_indices]
        masks = [traj["masks"][i] for i in switch_indices]
        action_seq = [actions[i] for i in switch_indices]
        rewards = [0.0] * len(action_seq)
        if traj.get("rewards"):
            rewards[-1] = traj["rewards"][-1]
        dones = [False] * len(action_seq)
        dones[-1] = True
        weight = float(traj.get("weight", 1.0)) * weight_scale
        tag = traj.get("tag", "") or "switch_focus"
        switch_only.append({
            "features": features,
            "masks": masks,
            "actions": action_seq,
            "rewards": rewards,
            "dones": dones,
            "weight": weight,
            "tag": f"{tag}_switch",
        })
    return switch_only


def extract_good_switch_trajectories(
    trajectories: list,
    min_actions: int = 1,
    weight_scale: float = 1.0,
    matchup_delta_min: float = 0.3,
    skip_forced: bool = True,
    tag_allowlist: tuple | None = None,
) -> list:
    """Extract switch-only steps where matchup improves by a threshold."""
    switch_only = []
    for traj in trajectories:
        actions = traj.get("actions", [])
        if not actions:
            continue
        if tag_allowlist:
            tag = traj.get("tag", "")
            if not any(t in tag for t in tag_allowlist):
                continue
        force_flags = traj.get("force_switch", [])
        switch_indices = []
        for i, action in enumerate(actions):
            if 4 <= action < 9:
                if skip_forced and i < len(force_flags) and force_flags[i]:
                    continue
                prev_m = traj.get("matchup_prev", [None] * len(actions))[i]
                new_m = traj.get("matchup_new", [None] * len(actions))[i]
                if prev_m is None or new_m is None:
                    continue
                if (new_m - prev_m) >= matchup_delta_min:
                    switch_indices.append(i)
        if len(switch_indices) < min_actions:
            continue
        features = [traj["features"][i] for i in switch_indices]
        masks = [traj["masks"][i] for i in switch_indices]
        action_seq = [actions[i] for i in switch_indices]
        rewards = [0.0] * len(action_seq)
        if traj.get("rewards"):
            rewards[-1] = traj["rewards"][-1]
        dones = [False] * len(action_seq)
        dones[-1] = True
        weight = float(traj.get("weight", 1.0)) * weight_scale
        tag = traj.get("tag", "") or "switch_focus"
        switch_only.append({
            "features": features,
            "masks": masks,
            "actions": action_seq,
            "rewards": rewards,
            "dones": dones,
            "weight": weight,
            "tag": f"{tag}_good_switch",
        })
    return switch_only


def extract_bad_switch_trajectories(
    trajectories: list,
    min_actions: int = 1,
    weight_scale: float = 1.0,
    matchup_delta_max: float = -0.2,
    skip_forced: bool = True,
    tag_allowlist: tuple | None = None,
) -> list:
    """Extract switch-only steps where matchup worsens beyond a threshold."""
    switch_only = []
    for traj in trajectories:
        actions = traj.get("actions", [])
        if not actions:
            continue
        if tag_allowlist:
            tag = traj.get("tag", "")
            if not any(t in tag for t in tag_allowlist):
                continue
        force_flags = traj.get("force_switch", [])
        switch_indices = []
        for i, action in enumerate(actions):
            if 4 <= action < 9:
                if skip_forced and i < len(force_flags) and force_flags[i]:
                    continue
                prev_m = traj.get("matchup_prev", [None] * len(actions))[i]
                new_m = traj.get("matchup_new", [None] * len(actions))[i]
                if prev_m is None or new_m is None:
                    continue
                if (new_m - prev_m) <= matchup_delta_max:
                    switch_indices.append(i)
        if len(switch_indices) < min_actions:
            continue
        features = [traj["features"][i] for i in switch_indices]
        masks = [traj["masks"][i] for i in switch_indices]
        action_seq = [actions[i] for i in switch_indices]
        rewards = [0.0] * len(action_seq)
        if traj.get("rewards"):
            rewards[-1] = traj["rewards"][-1]
        dones = [False] * len(action_seq)
        dones[-1] = True
        weight = float(traj.get("weight", 1.0)) * weight_scale
        tag = traj.get("tag", "") or "switch_focus"
        switch_only.append({
            "features": features,
            "masks": masks,
            "actions": action_seq,
            "rewards": rewards,
            "dones": dones,
            "weight": weight,
            "tag": f"{tag}_bad_switch",
        })
    return switch_only


def rollout_to_trajectory(rollout: dict, weight: float, tag: str) -> dict | None:
    if not rollout or not rollout.get('features'):
        return None
    return {
        'features': rollout['features'],
        'masks': rollout['masks'],
        'actions': rollout['actions'],
        'rewards': rollout['rewards'],
        'dones': rollout['dones'],
        'weight': weight,
        'tag': tag,
    }


def load_model_from_checkpoint(path: Path, config: RLConfig, device: str) -> RecurrentActorCritic:
    """Load a recurrent policy from a checkpoint path."""
    model = RecurrentActorCritic(
        config.feature_dim,
        config.d_model,
        config.n_actions,
        rnn_hidden=config.rnn_hidden,
        rnn_layers=config.rnn_layers,
    ).to(device)
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model


def build_selfplay_opponents(config: RLConfig, device: str, kwargs: dict,
                             current_model: RecurrentActorCritic) -> list[tuple[str, object]]:
    """Build an opponent pool for league-style self-play."""
    pool: list[tuple[str, object]] = []

    for name in config.offline_selfplay_checkpoints:
        path = Path(config.checkpoint_dir) / name
        if not path.exists():
            continue
        opponent_model = load_model_from_checkpoint(path, config, device)
        pool.append((
            path.stem,
            RLPlayer(
                model=opponent_model,
                config=config,
                device=device,
                training=False,
                **player_kwargs(kwargs, f"self_{path.stem}"),
            ),
        ))

    if config.offline_selfplay_include_current:
        pool.append((
            "current",
            RLPlayer(
                model=current_model,
                config=config,
                device=device,
                training=False,
                **player_kwargs(kwargs, "self_current"),
            ),
        ))

    if config.offline_selfplay_include_rulebot:
        pool.append(("rule_bot", RuleBotPlayer(**player_kwargs(kwargs, "self_rulebot"))))

    if config.offline_selfplay_include_heuristics:
        pool.append(("heuristics", SimpleHeuristicsPlayer(**player_kwargs(kwargs, "self_heuristics"))))

    if config.offline_selfplay_include_maxpower:
        pool.append(("max_power", MaxBasePowerPlayer(**player_kwargs(kwargs, "self_maxpower"))))

    return pool


async def quick_eval(model, config: RLConfig, device: str, label: str):
    """Small diagnostic eval to detect regressions between phases."""
    if not config.debug_eval_enabled:
        return {}

    opponents = []
    for name in config.debug_eval_opponents:
        if name == "random":
            opponents.append(("random", RandomPlayer))
        elif name == "max_power":
            opponents.append(("max_power", MaxBasePowerPlayer))
        elif name == "heuristics":
            opponents.append(("heuristics", SimpleHeuristicsPlayer))

    if not opponents:
        return {}

    kwargs = {
        'battle_format': 'gen9randombattle',
        'max_concurrent_battles': 1,
        'server_configuration': CustomServerConfig,
    }
    agent = RLPlayer(
        model=model,
        config=config,
        device=device,
        training=False,
        track_illegal=config.debug_eval_track_illegal,
        **player_kwargs(kwargs, f"eval_{label}"),
    )

    print(f"\n  DEBUG EVAL: {label}")
    results = {}
    for name, cls in opponents:
        opponent = cls(**player_kwargs(kwargs, f"eval_{name}"))
        wins = 0
        turns = 0
        illegal_start = agent.illegal_picks
        total_start = agent.total_picks

        for _ in range(config.debug_eval_battles):
            prev_wins = agent.n_won_battles
            await agent.battle_against(opponent, n_battles=1)
            if agent.n_won_battles > prev_wins:
                wins += 1
            if agent._last_battle:
                turns += getattr(agent._last_battle, "turn", 0)

        illegal = agent.illegal_picks - illegal_start
        total = agent.total_picks - total_start
        illegal_rate = (illegal / total) if total else 0.0
        avg_turns = turns / config.debug_eval_battles if config.debug_eval_battles else 0.0
        wr = wins / config.debug_eval_battles if config.debug_eval_battles else 0.0
        print(f"    {name}: win_rate={wr:.1%} avg_turns={avg_turns:.1f} illegal_rate={illegal_rate:.1%}")
        results[name] = {
            'win_rate': wr,
            'avg_turns': avg_turns,
            'illegal_rate': illegal_rate,
        }

        try:
            opponent.reset_battles()
        except Exception:
            pass

    try:
        agent.reset_battles()
    except Exception:
        pass
    return results


async def collect_selfplay_trajectories(model, config: RLConfig, device: str,
                                        n_battles: int = 1000) -> tuple[list, float]:
    """Collect trajectories using a league-style self-play pool."""
    kwargs = {
        'battle_format': 'gen9randombattle',
        'max_concurrent_battles': 1,
        'server_configuration': CustomServerConfig,
    }

    model.eval()
    agent = RLPlayer(
        model=model,
        config=config,
        device=device,
        training=True,
        **player_kwargs(kwargs, "selfplay_agent"),
    )

    if config.offline_selfplay_mode == "heuristics":
        pool = [("heuristics", SimpleHeuristicsPlayer(**player_kwargs(kwargs, "selfplay_heuristics")))]
    else:
        pool = build_selfplay_opponents(config, device, kwargs, current_model=model)
    if not pool:
        return [], 0.0
    pool_names = ", ".join(name for name, _ in pool)
    print(f"    Self-play pool: {pool_names}")

    trajectories = []
    wins = 0
    total_weight = 0.0
    weight_count = 0
    stats = {}
    start_time = time.time()
    for i in range(n_battles):
        if config.offline_selfplay_round_robin:
            opp_name, opponent = pool[i % len(pool)]
        else:
            opp_name, opponent = random.choice(pool)

        if opp_name not in stats:
            stats[opp_name] = {
                'battles': 0, 'wins': 0, 'turns': 0,
                'traj': 0, 'skipped_short': 0, 'skipped_return': 0, 'return': 0.0
            }

        agent.clear_rollout()
        prev_wins = agent.n_won_battles
        await agent.battle_against(opponent, n_battles=1)
        won = agent.n_won_battles > prev_wins
        stats[opp_name]['battles'] += 1
        if won:
            wins += 1
            stats[opp_name]['wins'] += 1

        battle_turns = getattr(agent._last_battle, "turn", 0) if agent._last_battle else 0
        stats[opp_name]['turns'] += battle_turns
        if battle_turns < config.offline_selfplay_min_turns:
            stats[opp_name]['skipped_short'] += 1
            continue

        rollout = agent.finalize_rollout(won)
        total_return = sum(rollout.get('rewards', [])) if rollout else 0.0
        if total_return < config.offline_selfplay_min_return:
            stats[opp_name]['skipped_return'] += 1
            continue
        if not config.offline_selfplay_wins_only or won:
            weight = 1.0
            if won:
                weight *= config.offline_selfplay_weight_win
                if agent._last_battle:
                    remaining = len([p for p in agent._last_battle.team.values() if not p.fainted])
                    if remaining >= config.offline_focus_min_remaining:
                        weight *= config.offline_selfplay_weight_dom
            else:
                weight *= config.offline_selfplay_weight_loss
            traj = rollout_to_trajectory(rollout, weight, tag=f"selfplay_vs_{opp_name}")
            if traj:
                trajectories.append(traj)
                stats[opp_name]['traj'] += 1
                stats[opp_name]['return'] += total_return
                total_weight += weight
                weight_count += 1

        if (i + 1) % 100 == 0:
            wr = wins / (i + 1)
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed if elapsed > 0 else 0.0
            eta = (n_battles - (i + 1)) / rate if rate > 0 else 0.0
            print(
                f"    Self-play {i+1}/{n_battles}: {wr:.1%} win rate, "
                f"{len(trajectories)} trajectories | ETA {eta/60:.1f}m"
            )

    del agent
    for _, opponent in pool:
        try:
            opponent.reset_battles()
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if config.debug_selfplay_stats and stats:
        print("    Self-play opponent stats:")
        for name, s in sorted(stats.items()):
            battles = s['battles'] or 1
            wr = s['wins'] / battles
            avg_turns = s['turns'] / battles
            avg_return = s['return'] / max(s['traj'], 1)
            print(
                f"      {name}: wr={wr:.1%} battles={s['battles']} traj={s['traj']} "
                f"avg_turns={avg_turns:.1f} avg_return={avg_return:.2f} "
                f"skipped_short={s['skipped_short']} skipped_return={s['skipped_return']}"
            )
        if weight_count:
            print(f"    Self-play avg trajectory weight: {total_weight / weight_count:.3f}")

    win_rate = wins / n_battles if n_battles else 0.0
    return trajectories, win_rate


def train_sequence_bc(
    model,
    trajectories,
    config: RLConfig,
    device: str,
    epochs: int | None = None,
    lr: float | None = None,
    chunk_paths: list[Path] | None = None,
    tag_filter: list[str] | None = None,
    value_coef: float | None = None,
):
    """Behavior cloning over full trajectories for sequence models."""
    train_epochs = epochs if epochs is not None else config.offline_bc_epochs
    train_lr = lr if lr is not None else 3e-4
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, train_epochs, eta_min=1e-5
    )
    num_workers, pin_memory = dataloader_settings(config, device)

    if chunk_paths:
        paths = list(chunk_paths)
    else:
        if trajectories is None:
            print("  No trajectories available for BC.")
            return model
        dataset = TrajectoryDataset(trajectories, config.gamma, include_returns=True)
        if len(dataset) == 0:
            print("  No trajectories available for BC.")
            return model
        loader = DataLoader(
            dataset,
            batch_size=config.offline_batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_trajectories,
        )

    best_loss = float("inf")
    start_time = time.time()
    for epoch in range(1, train_epochs + 1):
        wait_for_memory(config, f"BC epoch {epoch}")
        model.train()
        total_loss = 0.0
        total_steps = 0
        total_bc = 0.0
        total_value = 0.0
        total_illegal = 0.0
        total_switch = 0.0
        total_matchup = 0.0

        if chunk_paths:
            random.shuffle(paths)
            total_chunks = len(paths)
            for idx, path in enumerate(paths, start=1):
                if idx == 1 or idx == total_chunks or idx % config.bc_chunk_log_interval == 0:
                    pct = (idx / total_chunks) * 100 if total_chunks else 0
                    print(f"    Epoch {epoch}: chunk {idx}/{total_chunks} ({pct:.0f}%)")
                wait_for_memory(config, "BC chunk load")
                chunk = load_chunk(path)
                if tag_filter:
                    chunk = filter_trajectories(chunk, tag_filter)
                if not chunk:
                    continue
                dataset = TrajectoryDataset(chunk, config.gamma, include_returns=True)
                if len(dataset) == 0:
                    continue
                loader = DataLoader(
                    dataset,
                    batch_size=config.offline_batch_size,
                    shuffle=True,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                    collate_fn=collate_trajectories,
                )
                for feat, mask, act, returns, valid, weights, matchups in loader:
                    feat = feat.to(device)
                    mask = mask.to(device)
                    act = act.to(device)
                    returns = returns.to(device)
                    valid = valid.to(device)
                    weights = weights.to(device)
                    matchups = matchups.to(device)

                    optimizer.zero_grad()
                    logits, values, _ = model.forward_sequence(feat)
                    masked_logits = logits.masked_fill(~mask, -1e9)
                    log_probs = F.log_softmax(masked_logits, dim=-1)
                    nll = -log_probs.gather(-1, act.unsqueeze(-1)).squeeze(-1)
                    action_weight = torch.ones_like(nll)
                    if config.bc_switch_action_weight > 1.0:
                        switch_mask = (act >= 4) & (act < 9)
                        action_weight = action_weight + switch_mask.float() * (config.bc_switch_action_weight - 1.0)

                    valid_f = valid.float()
                    bc_loss = (nll * weights * action_weight * valid_f).sum() / valid_f.sum().clamp(min=1)
                    value_loss = (
                        (values - returns) ** 2 * valid_f
                    ).sum() / valid_f.sum().clamp(min=1)
                    probs = log_probs.exp()
                    switch_penalty = torch.tensor(0.0, device=device)
                    matchup_penalty = torch.tensor(0.0, device=device)
                    if config.switch_mass_coef > 0:
                        switch_mask = mask[..., 4:9]
                        switch_available = switch_mask.any(dim=-1).float()
                        if config.switch_mass_bad_matchup_only:
                            matchup_valid = torch.isfinite(matchups)
                            bad_matchup = matchups < config.switch_mass_bad_matchup_threshold
                            switch_available = switch_available * bad_matchup.float() * matchup_valid.float()
                        switch_mass = probs[..., 4:9].sum(dim=-1)
                        switch_shortfall = F.relu(config.switch_mass_target - switch_mass)
                        switch_penalty = (
                            switch_shortfall * switch_available * valid_f
                        ).sum() / valid_f.sum().clamp(min=1)
                    if config.bc_matchup_switch_penalty:
                        switch_available = mask[..., 4:9].any(dim=-1)
                        move_available = mask[..., :4].any(dim=-1)
                        matchup_valid = torch.isfinite(matchups)
                        bad_matchup = matchups < config.bc_matchup_threshold
                        penalty_mask = matchup_valid & bad_matchup & switch_available & move_available & valid
                        switch_mass = probs[..., 4:9].sum(dim=-1)
                        stay_penalty = (1.0 - switch_mass) * penalty_mask.float()
                        matchup_penalty = stay_penalty.sum() / valid_f.sum().clamp(min=1)
                    probs_raw = F.softmax(logits, dim=-1)
                    illegal_mask = (~mask).float()
                    illegal_mass = (probs_raw * illegal_mask).sum(-1)
                    illegal_penalty = (
                        illegal_mass * valid_f
                    ).sum() / valid_f.sum().clamp(min=1)

                    value_weight = config.offline_value_coef if value_coef is None else value_coef
                    loss = (
                        bc_loss
                        + value_weight * value_loss
                        + config.illegal_action_coef * illegal_penalty
                        + config.switch_mass_coef * switch_penalty
                        + config.bc_matchup_penalty_coef * matchup_penalty
                    )

                    if torch.isnan(loss):
                        continue

                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                    total_loss += loss.item() * valid_f.sum().item()
                    total_bc += bc_loss.item() * valid_f.sum().item()
                    total_value += value_loss.item() * valid_f.sum().item()
                    total_illegal += illegal_penalty.item() * valid_f.sum().item()
                    total_switch += switch_penalty.item() * valid_f.sum().item()
                    total_matchup += matchup_penalty.item() * valid_f.sum().item()
                    total_steps += valid_f.sum().item()

                del loader, dataset, chunk
                gc.collect()
        else:
            for feat, mask, act, returns, valid, weights, matchups in loader:
                feat = feat.to(device)
                mask = mask.to(device)
                act = act.to(device)
                returns = returns.to(device)
                valid = valid.to(device)
                weights = weights.to(device)
                matchups = matchups.to(device)

                optimizer.zero_grad()
                logits, values, _ = model.forward_sequence(feat)
                masked_logits = logits.masked_fill(~mask, -1e9)
                log_probs = F.log_softmax(masked_logits, dim=-1)
                nll = -log_probs.gather(-1, act.unsqueeze(-1)).squeeze(-1)
                action_weight = torch.ones_like(nll)
                if config.bc_switch_action_weight > 1.0:
                    switch_mask = (act >= 4) & (act < 9)
                    action_weight = action_weight + switch_mask.float() * (config.bc_switch_action_weight - 1.0)

                valid_f = valid.float()
                bc_loss = (nll * weights * action_weight * valid_f).sum() / valid_f.sum().clamp(min=1)
                value_loss = ((values - returns) ** 2 * valid_f).sum() / valid_f.sum().clamp(min=1)
                probs = log_probs.exp()
                switch_penalty = torch.tensor(0.0, device=device)
                matchup_penalty = torch.tensor(0.0, device=device)
                if config.switch_mass_coef > 0:
                    switch_mask = mask[..., 4:9]
                    switch_available = switch_mask.any(dim=-1).float()
                    if config.switch_mass_bad_matchup_only:
                        matchup_valid = torch.isfinite(matchups)
                        bad_matchup = matchups < config.switch_mass_bad_matchup_threshold
                        switch_available = switch_available * bad_matchup.float() * matchup_valid.float()
                    switch_mass = probs[..., 4:9].sum(dim=-1)
                    switch_shortfall = F.relu(config.switch_mass_target - switch_mass)
                    switch_penalty = (
                        switch_shortfall * switch_available * valid_f
                    ).sum() / valid_f.sum().clamp(min=1)
                if config.bc_matchup_switch_penalty:
                    switch_available = mask[..., 4:9].any(dim=-1)
                    move_available = mask[..., :4].any(dim=-1)
                    matchup_valid = torch.isfinite(matchups)
                    bad_matchup = matchups < config.bc_matchup_threshold
                    penalty_mask = matchup_valid & bad_matchup & switch_available & move_available & valid
                    switch_mass = probs[..., 4:9].sum(dim=-1)
                    stay_penalty = (1.0 - switch_mass) * penalty_mask.float()
                    matchup_penalty = stay_penalty.sum() / valid_f.sum().clamp(min=1)
                probs_raw = F.softmax(logits, dim=-1)
                illegal_mask = (~mask).float()
                illegal_mass = (probs_raw * illegal_mask).sum(-1)
                illegal_penalty = (illegal_mass * valid_f).sum() / valid_f.sum().clamp(min=1)

                value_weight = config.offline_value_coef if value_coef is None else value_coef
                loss = (
                    bc_loss
                    + value_weight * value_loss
                    + config.illegal_action_coef * illegal_penalty
                    + config.switch_mass_coef * switch_penalty
                    + config.bc_matchup_penalty_coef * matchup_penalty
                )

                if torch.isnan(loss):
                    continue

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item() * valid_f.sum().item()
                total_bc += bc_loss.item() * valid_f.sum().item()
                total_value += value_loss.item() * valid_f.sum().item()
                total_illegal += illegal_penalty.item() * valid_f.sum().item()
                total_switch += switch_penalty.item() * valid_f.sum().item()
                total_matchup += matchup_penalty.item() * valid_f.sum().item()
                total_steps += valid_f.sum().item()

        if total_steps == 0:
            print("  No trajectories available for BC after filtering.")
            return model

        scheduler.step()
        if epoch % config.bc_log_interval == 0 or epoch == 1 or epoch == train_epochs:
            avg_loss = total_loss / total_steps
            if avg_loss < best_loss:
                best_loss = avg_loss
            elapsed = time.time() - start_time
            avg_epoch = elapsed / epoch
            eta = avg_epoch * (train_epochs - epoch)
            if config.debug_log_losses:
                avg_bc = total_bc / total_steps
                avg_value = total_value / total_steps
                avg_illegal = total_illegal / total_steps
                avg_switch = total_switch / total_steps
                avg_matchup = total_matchup / total_steps
                extra = ""
                if config.bc_matchup_switch_penalty:
                    extra = f", matchup={avg_matchup:.4f}"
                print(
                    f"    Epoch {epoch}: avg_loss={avg_loss:.4f}, "
                    f"bc={avg_bc:.4f}, value={avg_value:.4f}, illegal={avg_illegal:.4f}, "
                    f"switch={avg_switch:.4f}{extra} "
                    f"| ETA {eta/60:.1f}m"
                )
            else:
                print(f"    Epoch {epoch}: avg_loss={avg_loss:.4f} | ETA {eta/60:.1f}m")

    print(f"  Best avg loss: {best_loss:.4f}")
    return model


# ============================================================================
# PHASE 2: Offline RL with BC Regularization
# ============================================================================

def train_offline_rl(
    model,
    trajectories,
    config: RLConfig,
    device: str,
    mode: str = "awbc",
    epochs: int | None = None,
    chunk_paths: list[Path] | None = None,
    tag_filter: list[str] | None = None,
):
    """Offline RL update using BC-regularized actor loss."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.01)
    num_workers, pin_memory = dataloader_settings(config, device)

    if chunk_paths:
        paths = list(chunk_paths)
    else:
        if trajectories is None:
            print("  No trajectories available for offline RL.")
            return model
        dataset = TrajectoryDataset(trajectories, config.gamma, include_returns=True)
        if len(dataset) == 0:
            print("  No trajectories available for offline RL.")
            return model
        loader = DataLoader(
            dataset,
            batch_size=config.offline_batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_trajectories,
        )

    total_epochs = epochs if epochs is not None else config.offline_rl_epochs
    start_time = time.time()
    for epoch in range(1, total_epochs + 1):
        wait_for_memory(config, f"offline RL epoch {epoch}")
        model.train()
        total_loss = 0.0
        total_steps = 0
        total_entropy = 0.0
        total_bc = 0.0
        total_value = 0.0
        total_aw = 0.0
        total_illegal = 0.0
        total_switch = 0.0

        if chunk_paths:
            random.shuffle(paths)
            for path in paths:
                wait_for_memory(config, "offline RL chunk load")
                chunk = load_chunk(path)
                if tag_filter:
                    chunk = filter_trajectories(chunk, tag_filter)
                if not chunk:
                    continue
                dataset = TrajectoryDataset(chunk, config.gamma, include_returns=True)
                if len(dataset) == 0:
                    continue
                loader = DataLoader(
                    dataset,
                    batch_size=config.offline_batch_size,
                    shuffle=True,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                    collate_fn=collate_trajectories,
                )
                for feat, mask, act, returns, valid, weights, matchups in loader:
                    feat = feat.to(device)
                    mask = mask.to(device)
                    act = act.to(device)
                    returns = returns.to(device)
                    valid = valid.to(device)
                    weights = weights.to(device)

                    optimizer.zero_grad()
                    logits, values, _ = model.forward_sequence(feat)
                    masked_logits = logits.masked_fill(~mask, -1e9)
                    log_probs = F.log_softmax(masked_logits, dim=-1)
                    nll = -log_probs.gather(-1, act.unsqueeze(-1)).squeeze(-1)

                    valid_f = valid.float()
                    advantages = returns - values.detach()

                    if mode == "binary":
                        aw = (advantages > 0).float()
                    else:
                        aw = torch.exp(config.offline_awbc_beta * advantages)
                        aw = torch.clamp(aw, max=config.offline_awbc_max_weight)

                    bc_loss = (nll * aw * weights * valid_f).sum() / valid_f.sum().clamp(min=1)
                    value_loss = (
                        (values - returns) ** 2 * valid_f
                    ).sum() / valid_f.sum().clamp(min=1)

                    probs = F.softmax(masked_logits, dim=-1)
                    log_probs_all = F.log_softmax(masked_logits, dim=-1)
                    entropy = -(probs * log_probs_all).sum(-1)
                    entropy = (entropy * valid_f).sum() / valid_f.sum().clamp(min=1)
                    switch_penalty = torch.tensor(0.0, device=device)
                    if config.switch_mass_coef > 0:
                        switch_mask = mask[..., 4:9]
                        switch_available = switch_mask.any(dim=-1).float()
                        switch_mass = probs[..., 4:9].sum(dim=-1)
                        switch_shortfall = F.relu(config.switch_mass_target - switch_mass)
                        switch_penalty = (
                            switch_shortfall * switch_available * valid_f
                        ).sum() / valid_f.sum().clamp(min=1)

                    probs_raw = F.softmax(logits, dim=-1)
                    illegal_mask = (~mask).float()
                    illegal_mass = (probs_raw * illegal_mask).sum(-1)
                    illegal_penalty = (
                        illegal_mass * valid_f
                    ).sum() / valid_f.sum().clamp(min=1)

                    loss = (
                        bc_loss
                        + config.offline_value_coef * value_loss
                        - config.offline_entropy_coef * entropy
                        + config.illegal_action_coef * illegal_penalty
                        + config.switch_mass_coef * switch_penalty
                    )

                    if torch.isnan(loss):
                        continue

                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                    optimizer.step()

                    total_loss += loss.item() * valid_f.sum().item()
                    total_entropy += entropy.item() * valid_f.sum().item()
                    total_bc += bc_loss.item() * valid_f.sum().item()
                    total_value += value_loss.item() * valid_f.sum().item()
                    total_aw += (aw * valid_f).sum().item()
                    total_illegal += illegal_penalty.item() * valid_f.sum().item()
                    total_switch += switch_penalty.item() * valid_f.sum().item()
                    total_steps += valid_f.sum().item()

                del loader, dataset, chunk
                gc.collect()
        else:
            for feat, mask, act, returns, valid, weights, matchups in loader:
                feat = feat.to(device)
                mask = mask.to(device)
                act = act.to(device)
                returns = returns.to(device)
                valid = valid.to(device)
                weights = weights.to(device)

                optimizer.zero_grad()
                logits, values, _ = model.forward_sequence(feat)
                masked_logits = logits.masked_fill(~mask, -1e9)
                log_probs = F.log_softmax(masked_logits, dim=-1)
                nll = -log_probs.gather(-1, act.unsqueeze(-1)).squeeze(-1)

                valid_f = valid.float()
                advantages = returns - values.detach()

                if mode == "binary":
                    aw = (advantages > 0).float()
                else:
                    aw = torch.exp(config.offline_awbc_beta * advantages)
                    aw = torch.clamp(aw, max=config.offline_awbc_max_weight)

                bc_loss = (nll * aw * weights * valid_f).sum() / valid_f.sum().clamp(min=1)
                value_loss = ((values - returns) ** 2 * valid_f).sum() / valid_f.sum().clamp(min=1)

                probs = F.softmax(masked_logits, dim=-1)
                log_probs_all = F.log_softmax(masked_logits, dim=-1)
                entropy = -(probs * log_probs_all).sum(-1)
                entropy = (entropy * valid_f).sum() / valid_f.sum().clamp(min=1)
                switch_penalty = torch.tensor(0.0, device=device)
                if config.switch_mass_coef > 0:
                    switch_mask = mask[..., 4:9]
                    switch_available = switch_mask.any(dim=-1).float()
                    switch_mass = probs[..., 4:9].sum(dim=-1)
                    switch_shortfall = F.relu(config.switch_mass_target - switch_mass)
                    switch_penalty = (
                        switch_shortfall * switch_available * valid_f
                    ).sum() / valid_f.sum().clamp(min=1)

                probs_raw = F.softmax(logits, dim=-1)
                illegal_mask = (~mask).float()
                illegal_mass = (probs_raw * illegal_mask).sum(-1)
                illegal_penalty = (illegal_mass * valid_f).sum() / valid_f.sum().clamp(min=1)

                loss = (
                    bc_loss
                    + config.offline_value_coef * value_loss
                    - config.offline_entropy_coef * entropy
                    + config.illegal_action_coef * illegal_penalty
                    + config.switch_mass_coef * switch_penalty
                )

                if torch.isnan(loss):
                    continue

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()

                total_loss += loss.item() * valid_f.sum().item()
                total_entropy += entropy.item() * valid_f.sum().item()
                total_bc += bc_loss.item() * valid_f.sum().item()
                total_value += value_loss.item() * valid_f.sum().item()
                total_aw += (aw * valid_f).sum().item()
                total_illegal += illegal_penalty.item() * valid_f.sum().item()
                total_switch += switch_penalty.item() * valid_f.sum().item()
                total_steps += valid_f.sum().item()

        if total_steps == 0:
            print("  No trajectories available for offline RL after filtering.")
            return model

        if epoch % 10 == 0:
            avg_loss = total_loss / total_steps
            avg_entropy = total_entropy / total_steps
            elapsed = time.time() - start_time
            avg_epoch = elapsed / epoch
            eta = avg_epoch * (total_epochs - epoch)
            if config.debug_log_losses:
                avg_bc = total_bc / total_steps
                avg_value = total_value / total_steps
                avg_aw = total_aw / total_steps
                avg_illegal = total_illegal / total_steps
                avg_switch = total_switch / total_steps
                print(
                    f"    Epoch {epoch}: avg_loss={avg_loss:.4f}, bc={avg_bc:.4f}, "
                    f"value={avg_value:.4f}, entropy={avg_entropy:.4f}, aw={avg_aw:.3f}, "
                    f"illegal={avg_illegal:.4f}, switch={avg_switch:.4f} | ETA {eta/60:.1f}m"
                )
            else:
                print(
                    f"    Epoch {epoch}: avg_loss={avg_loss:.4f}, "
                    f"entropy={avg_entropy:.4f} | ETA {eta/60:.1f}m"
                )

    return model


# ============================================================================
# Main Pipeline
# ============================================================================

async def main():
    print("=" * 70)
    print("  ORANGURU RL - Full Training Pipeline")
    print("  Hardware: 32GB RAM + RTX 3080 10GB")
    print("=" * 70)
    pipeline_start = time.time()

    config = RLConfig()
    apply_quick_overrides(config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    eval_scores: dict[str, dict] = {}
    checkpoint_map: dict[str, str] = {}
    model = None
    resume_start = None
    baseline_best_path = Path(config.best_checkpoint_path)
    baseline_best_score = None

    if config.rollback_if_worse and baseline_best_path.exists():
        try:
            baseline_model = load_model_from_checkpoint(baseline_best_path, config, device)
            baseline_metrics = await quick_eval(baseline_model, config, device, label="baseline-best")
            baseline_best_score = baseline_metrics.get(
                config.best_eval_target, {}
            ).get("win_rate")
            if baseline_best_score is not None:
                print(
                    f"\n  Baseline best ({baseline_best_path}) {config.best_eval_target}: "
                    f"{baseline_best_score:.1%}"
                )
        except Exception as exc:
            print(f"\n  Failed to load baseline best checkpoint: {exc}")

    if config.resume_checkpoint:
        resume_path = Path(config.resume_checkpoint)
        if not resume_path.exists():
            resume_path = Path(config.checkpoint_dir) / config.resume_checkpoint
        if resume_path.exists():
            checkpoint = torch.load(resume_path, map_location=device)
            model = RecurrentActorCritic(
                config.feature_dim,
                config.d_model,
                config.n_actions,
                rnn_hidden=config.rnn_hidden,
                rnn_layers=config.rnn_layers,
            ).to(device)
            model.load_state_dict(checkpoint['model'])
            model.eval()
            ckpt_phase = checkpoint.get("phase", "")
            resume_start = config.resume_phase_override or phase_after(ckpt_phase)
            print(f"\n  Resuming from {resume_path} (phase={ckpt_phase or 'unknown'}; next={resume_start})")
            checkpoint_map["resume"] = str(resume_path)
            if config.debug_eval_enabled:
                eval_scores["resume"] = await quick_eval(model, config, device, label="resume")
        else:
            print(f"\n  Resume checkpoint not found: {resume_path}. Starting fresh.")

    # Phase 1: Collect trajectories
    print("\n" + "=" * 70)
    print("  PHASE 1: Collecting Expert Trajectories")
    print("=" * 70)
    phase_start = time.time()

    traj_path = Path("data/rulebot_trajectories.pkl")
    existing_chunks = None
    if config.stream_trajectories and not config.stream_rebuild:
        chunk_dir = Path(config.stream_cache_dir)
        existing_chunks = sorted(chunk_dir.glob("full_chunk_*.pkl"))
        if existing_chunks:
            print(f"  Using cached trajectory chunks from {chunk_dir} ({len(existing_chunks)} files)")

    stream_stats = None
    streaming_mode = config.stream_trajectories and not config.stream_keep_in_memory

    if existing_chunks:
        trajectories = None
        full_chunk_paths = existing_chunks
    else:
        wait_for_memory(config, "trajectory load")
        if streaming_mode:
            trajectories, stream_stats = await collect_trajectories(
                config, n_battles_per_opponent=config.rulebot_battles_per_opponent
            )
            if stream_stats:
                print(
                    f"  Streamed {stream_stats['total_trajectories']} trajectories "
                    f"({stream_stats['total_steps']} steps) to {stream_stats['chunk_dir']}"
                )
            trajectories = None
        else:
            if traj_path.exists():
                print(f"  Loading cached trajectories from {traj_path}...")
                with open(traj_path, 'rb') as f:
                    trajectories = pickle.load(f)
                if not trajectories or len(trajectories) < config.rulebot_min_trajectories:
                    print("  Cache too small; re-collecting trajectories...")
                    trajectories, _ = await collect_trajectories(
                        config, n_battles_per_opponent=config.rulebot_battles_per_opponent
                    )
                    with open(traj_path, 'wb') as f:
                        pickle.dump(trajectories, f)
                total_steps = sum(len(t['actions']) for t in trajectories)
                print(f"  Loaded {len(trajectories)} trajectories ({total_steps} steps)")
            else:
                trajectories, _ = await collect_trajectories(
                    config, n_battles_per_opponent=config.rulebot_battles_per_opponent
                )
                traj_path.parent.mkdir(parents=True, exist_ok=True)
                with open(traj_path, 'wb') as f:
                    pickle.dump(trajectories, f)
                total_steps = sum(len(t['actions']) for t in trajectories)
                print(f"  Saved {len(trajectories)} trajectories ({total_steps} steps) to {traj_path}")
            if trajectories:
                trajectories = enforce_memory_budget(trajectories, config, label="rulebot_cache")

        # Optional: merge replay trajectories (human data)
        if config.use_replay_trajectories:
            replay_path = Path(config.replay_trajectories_path)
            if replay_path.exists():
                with open(replay_path, 'rb') as f:
                    replay_traj = pickle.load(f)
                if replay_traj:
                    if config.replay_max_trajectories:
                        replay_traj = replay_traj[:config.replay_max_trajectories]
                    if config.replay_weight != 1.0:
                        for t in replay_traj:
                            t['weight'] = float(t.get('weight', 1.0)) * config.replay_weight
                    if streaming_mode:
                        writer = TrajectoryStreamWriter(config, prefix="full", append_existing=True)
                        writer.append(replay_traj)
                        writer.finalize()
                        print(f"  Streamed replay trajectories: {len(replay_traj)}")
                    else:
                        trajectories.extend(replay_traj)
                        replay_steps = sum(len(t.get('actions', [])) for t in replay_traj)
                        print(f"  Added replay trajectories: {len(replay_traj)} ({replay_steps} steps)")
            else:
                print(f"  Replay dataset not found at {replay_path}, skipping.")
        # Optional: merge ladder trajectories (online data)
        if config.use_ladder_trajectories:
            ladder_path = Path(config.ladder_trajectories_path)
            if ladder_path.exists():
                with open(ladder_path, 'rb') as f:
                    ladder_traj = pickle.load(f)
                if ladder_traj:
                    if config.ladder_max_trajectories:
                        ladder_traj = ladder_traj[:config.ladder_max_trajectories]
                    if config.ladder_weight != 1.0:
                        for t in ladder_traj:
                            t['weight'] = float(t.get('weight', 1.0)) * config.ladder_weight
                    if streaming_mode:
                        writer = TrajectoryStreamWriter(config, prefix="full", append_existing=True)
                        writer.append(ladder_traj)
                        writer.finalize()
                        print(f"  Streamed ladder trajectories: {len(ladder_traj)}")
                    else:
                        trajectories.extend(ladder_traj)
                        ladder_steps = sum(len(t.get('actions', [])) for t in ladder_traj)
                        print(f"  Added ladder trajectories: {len(ladder_traj)} ({ladder_steps} steps)")
            else:
                print(f"  Ladder dataset not found at {ladder_path}, skipping.")
        # Optional: merge switch corrections (eval-derived)
        if config.use_switch_corrections:
            corr_path = Path(config.switch_corrections_path)
            if corr_path.exists():
                try:
                    with open(corr_path, "rb") as f:
                        corrections = pickle.load(f)
                except Exception as exc:
                    corrections = None
                    print(f"  Failed to load switch corrections: {exc}")
                if corrections:
                    if config.switch_corrections_weight != 1.0:
                        for t in corrections:
                            t["weight"] = float(t.get("weight", 1.0)) * config.switch_corrections_weight
                    if streaming_mode:
                        writer = TrajectoryStreamWriter(config, prefix="full", append_existing=True)
                        writer.append(corrections)
                        writer.finalize()
                        print(f"  Streamed switch corrections: {len(corrections)}")
                    else:
                        trajectories.extend(corrections)
                        correction_steps = sum(len(t.get("actions", [])) for t in corrections)
                        print(f"  Added switch corrections: {len(corrections)} ({correction_steps} steps)")
            else:
                print(f"  Switch corrections not found at {corr_path}, skipping.")
        # Optional: merge move corrections (eval-derived)
        if config.use_move_corrections:
            move_path = Path(config.move_corrections_path)
            if move_path.exists():
                try:
                    with open(move_path, "rb") as f:
                        corrections = pickle.load(f)
                except Exception as exc:
                    corrections = None
                    print(f"  Failed to load move corrections: {exc}")
                if corrections:
                    if config.move_corrections_weight != 1.0:
                        for t in corrections:
                            t["weight"] = float(t.get("weight", 1.0)) * config.move_corrections_weight
                    if streaming_mode:
                        writer = TrajectoryStreamWriter(config, prefix="full", append_existing=True)
                        writer.append(corrections)
                        writer.finalize()
                        print(f"  Streamed move corrections: {len(corrections)}")
                    else:
                        trajectories.extend(corrections)
                        correction_steps = sum(len(t.get("actions", [])) for t in corrections)
                        print(f"  Added move corrections: {len(corrections)} ({correction_steps} steps)")
            else:
                print(f"  Move corrections not found at {move_path}, skipping.")
        if trajectories:
            trajectories = enforce_memory_budget(trajectories, config, label="after_replay_merge")
        wait_for_memory(config, "trajectory chunking")

        full_chunk_paths = None
        if config.stream_trajectories:
            if streaming_mode:
                chunk_dir = Path(config.stream_cache_dir)
                full_chunk_paths = sorted(chunk_dir.glob("full_chunk_*.pkl"))
            else:
                full_chunk_paths = write_trajectory_chunks(trajectories, config, prefix="full")
                if not config.stream_keep_in_memory:
                    trajectories = None
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
    print(
        f"\n  Phase 1 duration: {format_duration(time.time() - phase_start)} "
        f"| Total elapsed: {format_duration(time.time() - pipeline_start)}"
    )

    phase_order = {
        "bc": 1,
        "bc_focus": 2,
        "bc_switch_focus": 3,
        "offline_rl": 4,
        "offline_focus": 5,
        "selfplay": 6,
        "done": 7,
    }
    start_phase = resume_start or "bc"
    start_rank = phase_order.get(start_phase, 1)

    # Phase 2: Sequence BC pretraining
    print("\n" + "=" * 70)
    print("  PHASE 2: Sequence Behavior Cloning")
    print("=" * 70)
    phase_start = time.time()

    if model is None:
        model = RecurrentActorCritic(
            config.feature_dim,
            config.d_model,
            config.n_actions,
            rnn_hidden=config.rnn_hidden,
            rnn_layers=config.rnn_layers,
        ).to(device)

    if start_rank <= phase_order["bc"]:
        model = train_sequence_bc(
            model,
            trajectories,
            config,
            device,
            chunk_paths=full_chunk_paths,
        )
        eval_scores["post-BC"] = await quick_eval(model, config, device, label="post-BC")

        # Save imitation checkpoint
        imitation_path = f"{config.checkpoint_dir}/imitation.pt"
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        torch.save({
            'model': model.state_dict(),
            'phase': 'imitation',
            'model_type': 'recurrent',
            'config': asdict(config),
        }, imitation_path)
        print(f"  Saved imitation model to {imitation_path}")
        checkpoint_map["post-BC"] = imitation_path
    else:
        print("  Skipping BC phase (resume).")
    print(
        f"\n  Phase 2 duration: {format_duration(time.time() - phase_start)} "
        f"| Total elapsed: {format_duration(time.time() - pipeline_start)}"
    )

    # Optional: BC fine-tune on heuristics-only trajectories
    if config.bc_focus_epochs > 0 and start_rank <= phase_order["bc_focus"]:
        heuristics_bc = None
        if trajectories is not None:
            primary = filter_trajectories(trajectories, list(config.bc_focus_tags))
            secondary = filter_trajectories(trajectories, list(config.bc_focus_mix_tags))
            heuristics_bc = mix_trajectories(primary, secondary, config.bc_focus_mix_ratio)
        if heuristics_bc or full_chunk_paths:
            print("\n" + "=" * 70)
            print("  PHASE 2B: Heuristics BC Fine-tune")
            print("=" * 70)
            phase_start = time.time()
            model = train_sequence_bc(
                model,
                heuristics_bc,
                config,
                device,
                epochs=config.bc_focus_epochs,
                lr=config.bc_focus_lr,
                chunk_paths=full_chunk_paths,
                tag_filter=list(config.bc_focus_mix_tags),
            )
            best_focus_score = -1.0
            best_focus_state = None
            if config.bc_focus_eval_interval and config.bc_focus_eval_interval > 0:
                for epoch in range(config.bc_focus_eval_interval, config.bc_focus_epochs + 1, config.bc_focus_eval_interval):
                    eval_scores["post-bc-focus"] = await quick_eval(
                        model, config, device, label=f"post-bc-focus-epoch-{epoch}"
                    )
                    score = eval_scores["post-bc-focus"].get(config.best_eval_target, {}).get("win_rate")
                    if score is not None and score > best_focus_score:
                        best_focus_score = score
                        best_focus_state = {k: v.cpu() for k, v in model.state_dict().items()}
            if best_focus_state:
                model.load_state_dict(best_focus_state)
            eval_scores["post-bc-focus"] = await quick_eval(model, config, device, label="post-bc-focus")
            bc_focus_path = f"{config.checkpoint_dir}/bc_focus.pt"
            torch.save({
                'model': model.state_dict(),
                'phase': 'bc_focus',
                'model_type': 'recurrent',
                'config': asdict(config),
            }, bc_focus_path)
            print(f"  Saved BC focus model to {bc_focus_path}")
            checkpoint_map["post-bc-focus"] = bc_focus_path
            print(
                f"\n  Phase 2B duration: {format_duration(time.time() - phase_start)} "
                f"| Total elapsed: {format_duration(time.time() - pipeline_start)}"
            )
    elif config.bc_focus_epochs > 0:
        print("  Skipping BC focus phase (resume).")

    # Optional: switch-focused BC pass
    if config.bc_switch_focus_epochs > 0 and start_rank <= phase_order["bc_switch_focus"]:
        print("\n" + "=" * 70)
        print("  PHASE 2C: Switch-Focused BC")
        print("=" * 70)
        phase_start = time.time()
        switch_traj = None
        if trajectories is not None:
            switch_traj = extract_switch_only_trajectories(
                trajectories,
                min_actions=config.bc_switch_focus_min_actions,
                weight_scale=config.bc_switch_focus_weight,
                skip_forced=config.bc_switch_focus_skip_forced,
                tag_allowlist=config.bc_switch_focus_tags,
            )
            good_switch_traj = extract_good_switch_trajectories(
                trajectories,
                min_actions=config.bc_switch_focus_min_actions,
                weight_scale=config.bc_switch_focus_weight,
                matchup_delta_min=config.bc_switch_focus_matchup_delta_min,
                skip_forced=config.bc_switch_focus_skip_forced,
                tag_allowlist=config.bc_switch_focus_tags,
            )
            bad_switch_traj = extract_bad_switch_trajectories(
                trajectories,
                min_actions=config.bc_switch_focus_min_actions,
                weight_scale=config.bc_switch_focus_weight * config.bc_switch_focus_bad_weight,
                matchup_delta_max=config.bc_switch_focus_bad_delta_max,
                skip_forced=config.bc_switch_focus_skip_forced,
                tag_allowlist=config.bc_switch_focus_tags,
            )
            if good_switch_traj:
                switch_traj = (switch_traj or []) + good_switch_traj
            if bad_switch_traj:
                switch_traj = (switch_traj or []) + bad_switch_traj
        elif full_chunk_paths:
            switch_traj = []
            for path in full_chunk_paths:
                wait_for_memory(config, "switch focus chunk load")
                chunk = load_chunk(path)
                switch_chunk = extract_switch_only_trajectories(
                    chunk,
                    min_actions=config.bc_switch_focus_min_actions,
                    weight_scale=config.bc_switch_focus_weight,
                    skip_forced=config.bc_switch_focus_skip_forced,
                    tag_allowlist=config.bc_switch_focus_tags,
                )
                good_switch_chunk = extract_good_switch_trajectories(
                    chunk,
                    min_actions=config.bc_switch_focus_min_actions,
                    weight_scale=config.bc_switch_focus_weight,
                    matchup_delta_min=config.bc_switch_focus_matchup_delta_min,
                    skip_forced=config.bc_switch_focus_skip_forced,
                    tag_allowlist=config.bc_switch_focus_tags,
                )
                bad_switch_chunk = extract_bad_switch_trajectories(
                    chunk,
                    min_actions=config.bc_switch_focus_min_actions,
                    weight_scale=config.bc_switch_focus_weight * config.bc_switch_focus_bad_weight,
                    matchup_delta_max=config.bc_switch_focus_bad_delta_max,
                    skip_forced=config.bc_switch_focus_skip_forced,
                    tag_allowlist=config.bc_switch_focus_tags,
                )
                if switch_chunk:
                    switch_traj.extend(switch_chunk)
                if good_switch_chunk:
                    switch_traj.extend(good_switch_chunk)
                if bad_switch_chunk:
                    switch_traj.extend(bad_switch_chunk)
                if switch_traj:
                    switch_traj = enforce_memory_budget(
                        switch_traj, config, label="switch_focus"
                    )
                del chunk
                gc.collect()
        if switch_traj:
            switch_traj = enforce_memory_budget(switch_traj, config, label="switch_focus")
        if switch_traj:
            print(f"  Switch focus trajectories: {len(switch_traj)}")
            model = train_sequence_bc(
                model,
                switch_traj,
                config,
                device,
                epochs=config.bc_switch_focus_epochs,
                lr=config.bc_switch_focus_lr,
                value_coef=config.bc_switch_focus_value_coef,
            )
            eval_scores["post-bc-switch"] = await quick_eval(
                model, config, device, label="post-bc-switch"
            )
            switch_path = f"{config.checkpoint_dir}/bc_switch.pt"
            torch.save({
                'model': model.state_dict(),
                'phase': 'bc_switch_focus',
                'model_type': 'recurrent',
                'config': asdict(config),
            }, switch_path)
            print(f"  Saved switch-focus model to {switch_path}")
            checkpoint_map["post-bc-switch"] = switch_path
            print(
                f"\n  Phase 2C duration: {format_duration(time.time() - phase_start)} "
                f"| Total elapsed: {format_duration(time.time() - pipeline_start)}"
            )
        else:
            print("  No switch trajectories found; skipping switch focus.")
    elif config.bc_switch_focus_epochs > 0:
        print("  Skipping switch focus phase (resume).")

    # Phase 3: Offline RL fine-tuning (BC-regularized)
    print("\n" + "=" * 70)
    print("  PHASE 3: Offline RL Fine-tuning")
    print("=" * 70)
    phase_start = time.time()

    if not config.skip_offline_rl and start_rank <= phase_order["offline_rl"]:
        model = train_offline_rl(
            model,
            trajectories,
            config,
            device,
            mode="binary",
            chunk_paths=full_chunk_paths,
        )
        eval_scores["post-offline-rl"] = await quick_eval(model, config, device, label="post-offline-rl")
        offline_rl_path = f"{config.checkpoint_dir}/offline_rl.pt"
        torch.save({
            'model': model.state_dict(),
            'phase': 'offline_rl',
            'model_type': 'recurrent',
            'config': asdict(config),
        }, offline_rl_path)
        print(f"  Saved offline RL model to {offline_rl_path}")
        checkpoint_map["post-offline-rl"] = offline_rl_path
    elif not config.skip_offline_rl:
        print("  Skipping offline RL phase (resume).")
    print(
        f"\n  Phase 3 duration: {format_duration(time.time() - phase_start)} "
        f"| Total elapsed: {format_duration(time.time() - pipeline_start)}"
    )

    # Phase 4: Heuristics-focused offline fine-tuning
    heuristics_traj = None
    if trajectories is not None:
        heuristics_traj = filter_trajectories(trajectories, ["heuristics"])
    if (heuristics_traj or full_chunk_paths) and not config.skip_focus and start_rank <= phase_order["offline_focus"]:
        print("\n" + "=" * 70)
        print("  PHASE 4: Heuristics Focused Offline RL")
        print("=" * 70)
        phase_start = time.time()
        model = train_offline_rl(
            model,
            heuristics_traj,
            config,
            device,
            mode="binary",
            epochs=config.offline_focus_epochs,
            chunk_paths=full_chunk_paths,
            tag_filter=["heuristics"],
        )
        eval_scores["post-focus"] = await quick_eval(model, config, device, label="post-focus")
        offline_focus_path = f"{config.checkpoint_dir}/offline_focus.pt"
        torch.save({
            'model': model.state_dict(),
            'phase': 'offline_focus',
            'model_type': 'recurrent',
            'config': asdict(config),
        }, offline_focus_path)
        print(f"  Saved offline focus model to {offline_focus_path}")
        checkpoint_map["post-focus"] = offline_focus_path
        print(
            f"\n  Phase 4 duration: {format_duration(time.time() - phase_start)} "
            f"| Total elapsed: {format_duration(time.time() - pipeline_start)}"
        )
    elif not config.skip_focus and start_rank > phase_order["offline_focus"]:
        print("  Skipping offline focus phase (resume).")

    # Phase 5: Offline self-play fine-tuning
    print("\n" + "=" * 70)
    print("  PHASE 5: Offline Self-Play Fine-tuning")
    print("=" * 70)
    if config.skip_selfplay:
        print("  Skipping self-play fine-tune (skip_selfplay=True)")
    elif start_rank > phase_order["selfplay"]:
        print("  Skipping self-play fine-tune (resume).")
    else:
        selfplay_traj, selfplay_wr = await collect_selfplay_trajectories(
            model, config, device, n_battles=config.offline_selfplay_battles
        )
        if config.offline_selfplay_mode == "heuristics" and selfplay_wr < config.offline_selfplay_min_winrate:
            print(
                f"  Skipping self-play fine-tune (win rate {selfplay_wr:.1%} < "
                f"{config.offline_selfplay_min_winrate:.0%})"
            )
        elif selfplay_traj:
            model = train_offline_rl(
                model,
                selfplay_traj,
                config,
                device,
                mode="binary",
                epochs=config.offline_selfplay_epochs,
            )
            eval_scores["post-selfplay"] = await quick_eval(model, config, device, label="post-selfplay")
            checkpoint_map["post-selfplay"] = f"{config.checkpoint_dir}/selfplay.pt"
            torch.save({
                'model': model.state_dict(),
                'phase': 'offline_selfplay',
                'model_type': 'recurrent',
                'config': asdict(config),
            }, checkpoint_map["post-selfplay"])

    if config.select_best_checkpoint and eval_scores:
        target = config.best_eval_target
        best_label = None
        best_score = -1.0
        for label, metrics in eval_scores.items():
            score = metrics.get(target, {}).get('win_rate')
            if score is not None and score > best_score and label in checkpoint_map:
                best_score = score
                best_label = label
        if best_label:
            print(f"\n  Selecting best checkpoint: {best_label} ({best_score:.1%} vs {target})")
            best_path = Path(checkpoint_map[best_label])
            model = load_model_from_checkpoint(best_path, config, device)
            if (
                baseline_best_score is not None
                and best_score < baseline_best_score - config.rollback_min_delta
            ):
                print(
                    f"  Rolling back to previous best ({baseline_best_score:.1%}) "
                    f"over current ({best_score:.1%})."
                )
                model = load_model_from_checkpoint(baseline_best_path, config, device)
            else:
                best_dir = Path(config.best_checkpoint_path).parent
                best_dir.mkdir(parents=True, exist_ok=True)
                if Path(config.best_checkpoint_path).exists():
                    prev_path = Path(config.best_checkpoint_path).with_suffix(".prev.pt")
                    try:
                        shutil.copy2(config.best_checkpoint_path, prev_path)
                        print(f"  Saved previous best to {prev_path}")
                    except Exception as exc:
                        print(f"  Failed to backup previous best: {exc}")
                torch.save({
                    'model': model.state_dict(),
                    'phase': 'best_overall',
                    'model_type': 'recurrent',
                    'config': asdict(config),
                    'source': str(best_path),
                    'best_label': best_label,
                    'best_score': best_score,
                }, config.best_checkpoint_path)
                print(f"  Backed up best checkpoint to {config.best_checkpoint_path}")

    # Save final model
    final_path = f"{config.checkpoint_dir}/full_pipeline.pt"
    torch.save({
        'model': model.state_dict(),
        'phase': 'offline_rl',
        'model_type': 'recurrent',
        'config': asdict(config),
    }, final_path)
    print(f"\n  Saved final model to {final_path}")

    print("\n" + "=" * 70)
    print("  TRAINING COMPLETE")
    print("=" * 70)
    print(f"\nEvaluate with:")
    print(f"  python evaluation/evaluate.py --checkpoint {final_path}")


if __name__ == "__main__":
    print("\nEnsure server is running: docker start pokemon-showdown\n")
    asyncio.run(main())
