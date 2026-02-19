#!/usr/bin/env python3
"""
Ladder RuleBot on the public Pokemon Showdown server.

Set credentials via environment variables:
  PS_USERNAME="your_username"
  PS_PASSWORD="your_password"
"""

import asyncio
import json
import logging
import os
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import fields
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from poke_env import AccountConfiguration
from poke_env.data.normalize import to_id_str
from poke_env.ps_client.server_configuration import ShowdownServerConfiguration

from src.models.actor_critic import ActorCritic, RecurrentActorCritic
from src.players.rl_player import RLPlayer
from src.players.rule_bot import RuleBotPlayer
from src.players.oranguru_engine import OranguruEnginePlayer
from poke_env.battle import MoveCategory
from evaluation.test_rulebot import test_rulebot
from training.config import RLConfig
from src.utils.features import load_moves, EnhancedFeatureBuilder
from src.utils.damage_calc import normalize_name


class LadderDataCollector:
    """Collects full trajectories from ladder games for training."""

    def __init__(
        self,
        min_turns: int = 0,
        min_actions: int = 0,
        min_rating: int | None = None,
        min_player_rating: int | None = None,
        min_opponent_rating: int | None = None,
        skip_forfeit: bool = False,
        max_illegal_rate: float | None = None,
        min_win_remaining: int | None = None,
        max_opp_remaining: int | None = None,
        win_weight: float = 1.0,
        loss_weight: float = 1.0,
        tag_prefix: str = "ladder",
    ):
        self.feature_builder = EnhancedFeatureBuilder()
        self.trajectories = []
        self._current_battles = {}  # battle_tag -> trajectory dict
        self.min_turns = min_turns
        self.min_actions = min_actions
        self.min_rating = min_rating
        self.min_player_rating = min_player_rating
        self.min_opponent_rating = min_opponent_rating
        self.skip_forfeit = skip_forfeit
        self.max_illegal_rate = max_illegal_rate
        self.min_win_remaining = min_win_remaining
        self.max_opp_remaining = max_opp_remaining
        self.win_weight = win_weight
        self.loss_weight = loss_weight
        self.tag_prefix = tag_prefix
        self.skipped = Counter()

    def record(self, battle, order):
        """Record a state-action pair from a battle into a trajectory."""
        tag = getattr(battle, "battle_tag", "unknown")
        if tag not in self._current_battles:
            self._current_battles[tag] = {
                'features': [],
                'masks': [],
                'actions': [],
                'illegal': 0,
                'attempts': 0,
            }

        features = self.feature_builder.build(battle)
        mask = self._build_mask(battle)
        action = self._order_to_action_idx(battle, order)

        current = self._current_battles[tag]
        current['attempts'] += 1
        if action is not None:
            current['features'].append(features)
            current['masks'].append(mask)
            current['actions'].append(action)
        else:
            current['illegal'] += 1

    def commit_battle(
        self,
        battle,
        won: bool,
        wins_only: bool = False,
    ):
        """Commit battle trajectories if conditions met."""
        tag = getattr(battle, "battle_tag", "unknown")
        if tag not in self._current_battles:
            return

        traj = self._current_battles.pop(tag)
        actions = traj.get('actions', [])
        total_actions = len(actions)
        if total_actions == 0:
            self.skipped["empty"] += 1
            return

        turns = getattr(battle, "turn", 0) or 0
        if self.min_turns and turns < self.min_turns:
            self.skipped["short"] += 1
            return
        if self.min_actions and total_actions < self.min_actions:
            self.skipped["short_actions"] += 1
            return

        rating, opponent_rating = self._extract_battle_ratings(battle)
        rating_value = rating if rating is not None else opponent_rating
        if self.min_rating and (rating_value is None or rating_value < self.min_rating):
            if rating_value is None:
                self.skipped["missing_rating"] += 1
            self.skipped["low_rating"] += 1
            return
        if self.min_player_rating and (rating is None or rating < self.min_player_rating):
            if rating is None:
                self.skipped["missing_player_rating"] += 1
            self.skipped["low_player_rating"] += 1
            return
        if self.min_opponent_rating and (opponent_rating is None or opponent_rating < self.min_opponent_rating):
            if opponent_rating is None:
                self.skipped["missing_opponent_rating"] += 1
            self.skipped["low_opponent_rating"] += 1
            return

        if self.skip_forfeit and self._battle_has_forfeit(battle):
            self.skipped["forfeit"] += 1
            return

        attempts = max(1, traj.get("attempts", total_actions))
        illegal_rate = traj.get("illegal", 0) / attempts
        if self.max_illegal_rate is not None and illegal_rate > self.max_illegal_rate:
            self.skipped["illegal"] += 1
            return

        if wins_only and not won:
            self.skipped["loss"] += 1
            return

        if won and (self.min_win_remaining or self.max_opp_remaining is not None):
            my_remaining = self._remaining_pokemon(battle, is_player=True)
            opp_remaining = self._remaining_pokemon(battle, is_player=False)
            if self.min_win_remaining and (my_remaining is None or my_remaining < self.min_win_remaining):
                self.skipped["win_remaining"] += 1
                return
            if self.max_opp_remaining is not None and opp_remaining is not None:
                if opp_remaining > self.max_opp_remaining:
                    self.skipped["opp_remaining"] += 1
                    return

        rewards = [0.0] * total_actions
        rewards[-1] = 1.0 if won else -1.0
        dones = [False] * (total_actions - 1) + [True]
        weight = self.win_weight if won else self.loss_weight
        tag_suffix = getattr(battle, "format", "") or getattr(battle, "battle_tag", "ladder")
        if tag_suffix.startswith("battle-"):
            tag_suffix = tag_suffix[7:]

        traj.update({
            'rewards': rewards,
            'dones': dones,
            'weight': weight,
            'tag': f"{self.tag_prefix}_{tag_suffix}",
            'rating': rating,
            'opponent_rating': opponent_rating,
            'turns': turns,
            'forfeit': self._battle_has_forfeit(battle),
            'illegal_rate': illegal_rate,
        })

        self.trajectories.append(traj)

    def save(self, path: str):
        """Save collected trajectories."""
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self.trajectories, f)
        print(f"Saved {len(self.trajectories)} trajectories to {path}")

    def _build_mask(self, battle) -> list:
        """Build action mask for battle state."""
        mask = [False] * 13  # 4 moves + 5 switches + 4 tera moves
        if getattr(battle, "force_switch", False):
            for i in range(min(5, len(battle.available_switches))):
                mask[4 + i] = True
        else:
            for i in range(min(4, len(battle.available_moves))):
                mask[i] = True
            for i in range(min(5, len(battle.available_switches))):
                mask[4 + i] = True
            if battle.can_tera:
                for i in range(min(4, len(battle.available_moves))):
                    mask[9 + i] = True
        return mask

    def _order_to_action_idx(self, battle, order):
        """Convert order to action index."""
        if order is None or not hasattr(order, 'order'):
            return None

        move_or_pokemon = order.order
        if move_or_pokemon is None:
            return None

        # Check for move
        if hasattr(move_or_pokemon, 'base_power') or hasattr(move_or_pokemon, 'category'):
            try:
                idx = battle.available_moves.index(move_or_pokemon)
                if hasattr(order, 'terastallize') and order.terastallize:
                    return 9 + idx
                return idx
            except ValueError:
                pass
        else:
            # Switch
            try:
                idx = battle.available_switches.index(move_or_pokemon)
                return 4 + idx
            except ValueError:
                pass
        return None

    def _battle_has_forfeit(self, battle) -> bool:
        observations = getattr(battle, "_observations", {})
        for obs in observations.values():
            for event in getattr(obs, "events", []):
                if len(event) > 1 and event[1] == "forfeit":
                    return True
        return False

    def _remaining_pokemon(self, battle, is_player: bool) -> int | None:
        team = battle.team if is_player else battle.opponent_team
        if not team:
            return None
        return sum(1 for mon in team.values() if not getattr(mon, "fainted", False))

    def _extract_battle_ratings(self, battle) -> tuple[int | None, int | None]:
        """Get player/opponent ratings with fallback parsing from observation events.

        Some Showdown flows don't populate `battle.rating` / `battle.opponent_rating`
        even though rating updates are present in raw battle messages.
        """
        rating = getattr(battle, "rating", None)
        opponent_rating = getattr(battle, "opponent_rating", None)
        if rating is not None and opponent_rating is not None:
            return rating, opponent_rating

        player_name = to_id_str(getattr(battle, "player_username", "") or "")
        opp_name = to_id_str(getattr(battle, "opponent_username", "") or "")

        # Example source strings:
        # "A_Jar_Of_Water's rating: 1242 -> 1266"
        # "A_Jar_Of_Water's rating: 1242 &rarr; <strong>1266</strong>"
        pattern = re.compile(r"(.+?)'s rating:\s*(\d+)\s*(?:->|→)?\s*(\d+)")
        observations = getattr(battle, "_observations", {})

        for obs in observations.values():
            for event in getattr(obs, "events", []):
                if not event:
                    continue
                for chunk in event:
                    if not isinstance(chunk, str) or "rating:" not in chunk:
                        continue
                    text = chunk.replace("&rarr;", "->")
                    text = re.sub(r"<[^>]+>", " ", text)
                    for line in text.splitlines():
                        m = pattern.search(line)
                        if not m:
                            continue
                        name = to_id_str(m.group(1).strip())
                        try:
                            new_rating = int(m.group(3))
                        except Exception:
                            continue
                        if player_name and name == player_name and rating is None:
                            rating = new_rating
                        elif opp_name and name == opp_name and opponent_rating is None:
                            opponent_rating = new_rating
                        elif rating is None:
                            rating = new_rating
                        elif opponent_rating is None:
                            opponent_rating = new_rating
                        if rating is not None and opponent_rating is not None:
                            return rating, opponent_rating

        return rating, opponent_rating


def load_checkpoint_for_ladder(path: str, device: str):
    """Load a checkpoint and return (model, config)."""
    import torch

    checkpoint = torch.load(path, map_location=device)
    cfg_data = checkpoint.get("config", {}) or {}
    cfg_fields = {f.name for f in fields(RLConfig)}
    filtered = {k: v for k, v in cfg_data.items() if k in cfg_fields}
    config = RLConfig(**filtered) if filtered else RLConfig()

    model_type = checkpoint.get("model_type", "recurrent")
    if model_type == "recurrent":
        model = RecurrentActorCritic(
            config.feature_dim,
            config.d_model,
            config.n_actions,
            rnn_hidden=config.rnn_hidden,
            rnn_layers=config.rnn_layers,
        )
    else:
        model = ActorCritic(config.feature_dim, config.d_model, config.n_actions)
    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    elif "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        raise ValueError("Checkpoint missing model weights.")
    model.to(device)
    model.eval()
    return model, config


def battle_has_forfeit(battle) -> bool:
    observations = getattr(battle, "_observations", {})
    for obs in observations.values():
        for event in getattr(obs, "events", []):
            if len(event) > 1 and event[1] == "forfeit":
                return True
    return False


def load_accounts_from_file(path: str) -> list[tuple[str, str]]:
    """Load account credentials from a file (one per line)."""
    accounts: list[tuple[str, str]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw or raw.startswith("#"):
                continue
            if ":" in raw:
                username, password = raw.split(":", 1)
            elif "," in raw:
                username, password = raw.split(",", 1)
            else:
                parts = raw.split()
                if len(parts) < 2:
                    continue
                username, password = parts[0], parts[1]
            username = username.strip()
            password = password.strip()
            if not username or not password:
                continue
            accounts.append((username, password))
    return accounts


def filter_accounts_by_name(
    accounts: list[tuple[str, str]], names: list[str]
) -> list[tuple[str, str]]:
    if not names:
        return accounts
    wanted = {to_id_str(name) for name in names}
    filtered = [(user, password) for user, password in accounts if to_id_str(user) in wanted]
    missing = wanted - {to_id_str(user) for user, _ in filtered}
    if missing:
        missing_str = ", ".join(sorted(missing))
        print(f"⚠️  Account(s) not found in accounts file: {missing_str}")
    return filtered


async def ladder_rulebot(
    n_battles: int,
    battle_format: str,
    max_concurrent: int,
    verbose: bool,
    pre_eval: bool,
    pre_eval_battles: int,
    log_decisions: bool,
    decision_slow_ms: int,
    log_level: str,
    open_timeout: float | None,
    ping_interval: float | None,
    ping_timeout: float | None,
    start_timer: bool,
    player_type: str,
    checkpoint_path: str | None,
    device: str,
    decision_log: bool,
    decision_log_every: int,
    decision_log_topk: int,
    decision_log_max: int,
    policy_temperature: float,
    sample_actions: bool,
    switch_mass_min: float,
    switch_mass_warn: float,
    collect_data: bool = False,
    wins_only: bool = False,
    data_output: str = "ladder_demos.pkl",
    min_turns: int = 0,
    min_actions: int = 0,
    min_rating: int | None = None,
    min_player_rating: int | None = None,
    min_opponent_rating: int | None = None,
    skip_forfeit: bool = False,
    max_illegal_rate: float | None = None,
    min_win_remaining: int | None = None,
    max_opp_remaining: int | None = None,
    win_weight: float = 1.0,
    loss_weight: float = 1.0,
    tag_prefix: str = "ladder",
    progress_every: int = 0,
    username: str | None = None,
    password: str | None = None,
    auto_reconnect: bool = False,
    max_retries: int = 2,
    retry_wait_s: float = 5.0,
    rejoin_active: bool = False,
    login_timeout_s: float | None = None,
    chat_message: str = "",
    challenge_opponents: list[str] | None = None,
    challenge_battles: int | None = None,
    snapshot_log: str | None = None,
    snapshot_every: int = 1,
):
    if username is None:
        username = os.environ.get("PS_USERNAME")
    if password is None:
        password = os.environ.get("PS_PASSWORD")

    if not username or not password:
        print("❌ Missing credentials. Set PS_USERNAME and PS_PASSWORD in your environment.")
        return 1

    if pre_eval:
        print(f"🔎 Pre-eval: {pre_eval_battles} battles per opponent (local server required)")
        try:
            await test_rulebot(pre_eval_battles, progress_every=pre_eval_battles, timeout_s=90)
        except Exception as exc:
            print(f"⚠️  Pre-eval failed: {exc}")

    account_config = AccountConfiguration(username, password)

    # Data collector for training
    data_collector = (
        LadderDataCollector(
            min_turns=min_turns,
            min_actions=min_actions,
            min_rating=min_rating,
            min_player_rating=min_player_rating,
            min_opponent_rating=min_opponent_rating,
            skip_forfeit=skip_forfeit,
            max_illegal_rate=max_illegal_rate,
            min_win_remaining=min_win_remaining,
            max_opp_remaining=max_opp_remaining,
            win_weight=win_weight,
            loss_weight=loss_weight,
            tag_prefix=tag_prefix,
        )
        if collect_data
        else None
    )

    action_stats = {
        "actions": 0,
        "switches": 0,
        "attacks": 0,
        "status": 0,
        "setup": 0,
        "hazards": 0,
        "moves": Counter(),
        "switch_targets": Counter(),
        "per_battle": defaultdict(lambda: Counter()),
        "status_available": 0,
        "switch_available": 0,
        "force_switch": 0,
        "tera_available": 0,
        "tera": 0,
        "decision_ms_sum": 0.0,
        "decision_ms_max": 0.0,
        "decision_ms_count": 0,
        "decision_slow": 0,
        "guard_errors": 0,
    }
    battle_stats = {
        "turns": [],
        "remaining": [],
        "opp_remaining": [],
        "forfeits": 0,
    }
    opponent_stats = defaultdict(lambda: Counter())

    class TrackedBase:
        def __init__(
            self,
            *args,
            data_collector=None,
            total_battles: int = 0,
            progress_every: int = 0,
            **kwargs,
        ):
            super().__init__(*args, **kwargs)
            self.data_collector = data_collector
            self._total_battles = int(total_battles) if total_battles else 0
            self._progress_every = max(0, int(progress_every))
            self._decision_count = 0
            self._decision_logged = 0
            self._summary_counts = {
                "finished": 0,
                "won": 0,
                "lost": 0,
                "tied": 0,
            }
            self._chat_message = chat_message or ""
            self._chat_sent = set()
            self._guard_errors: int = 0
            self._snapshot_log_path = snapshot_log
            self._snapshot_every = max(0, int(snapshot_every))
            self._switch_mass_stats = {
                "count": 0,
                "sum": 0.0,
                "min": 1.0,
                "max": 0.0,
                "low": 0,
                "raw_sum": 0.0,
                "raw_min": 1.0,
                "raw_max": 0.0,
                "raw_low": 0,
                "boosted": 0,
            }

        def _build_snapshot_payload(self, reason: str, battle=None) -> dict:
            counts = self._summary_counts
            finished = counts["finished"]
            wins = counts["won"]
            losses = counts["lost"]
            ties = counts["tied"]
            turns = battle_stats["turns"]
            remaining = battle_stats["remaining"]
            opp_remaining = battle_stats["opp_remaining"]
            payload = {
                "ts": time.time(),
                "reason": reason,
                "account": username,
                "finished": finished,
                "target": self._total_battles,
                "wins": wins,
                "losses": losses,
                "ties": ties,
                "win_rate": (wins / finished) if finished else 0.0,
                "avg_turns": (sum(turns) / len(turns)) if turns else None,
                "avg_remaining": (sum(remaining) / len(remaining)) if remaining else None,
                "avg_opp_remaining": (sum(opp_remaining) / len(opp_remaining)) if opp_remaining else None,
                "forfeits": battle_stats["forfeits"],
                "guard_errors": action_stats["guard_errors"],
            }
            if battle is not None:
                try:
                    my_remaining = sum(1 for p in battle.team.values() if not p.fainted)
                except Exception:
                    my_remaining = None
                try:
                    opp_rem = sum(1 for p in battle.opponent_team.values() if not p.fainted)
                except Exception:
                    opp_rem = None
                payload.update(
                    {
                        "battle_tag": getattr(battle, "battle_tag", None),
                        "turn": getattr(battle, "turn", None),
                        "won": bool(getattr(battle, "won", False)),
                        "lost": bool(getattr(battle, "lost", False)),
                        "tied": bool(
                            not getattr(battle, "won", False)
                            and not getattr(battle, "lost", False)
                        ),
                        "remaining": my_remaining,
                        "opp_remaining": opp_rem,
                    }
                )
            return payload

        def _emit_snapshot(self, reason: str, battle=None):
            payload = self._build_snapshot_payload(reason, battle=battle)
            finished = payload["finished"]
            target = payload["target"]
            turns = payload["avg_turns"]
            turns_s = f"{turns:.1f}" if isinstance(turns, (float, int)) else "n/a"
            print(
                "📍 Snapshot: "
                f"{finished}/{target} W/L/T {payload['wins']}/{payload['losses']}/{payload['ties']} "
                f"({payload['win_rate']:.1%}) avg_turns={turns_s}"
            )
            if self._snapshot_log_path:
                path = Path(self._snapshot_log_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                with open(path, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(payload, sort_keys=True) + "\n")

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

        def _maybe_send_chat(self, battle):
            if not self._chat_message:
                return
            tag = getattr(battle, "battle_tag", None)
            if not tag or tag in self._chat_sent:
                return
            self._chat_sent.add(tag)
            if hasattr(self, "ps_client") and hasattr(self.ps_client, "send_message"):
                asyncio.create_task(self.ps_client.send_message(self._chat_message, room=tag))

        def _battle_finished_callback(self, battle):
            if self.data_collector and battle.finished:
                won = getattr(battle, "won", False)
                self.data_collector.commit_battle(battle, won, wins_only=wins_only)
            if battle.finished:
                self._summary_counts["finished"] += 1
                opp_name = getattr(battle, "opponent_username", None) or "unknown"
                opp_entry = opponent_stats[opp_name]
                opp_entry["played"] += 1
                if getattr(battle, "won", False):
                    self._summary_counts["won"] += 1
                    opp_entry["won"] += 1
                elif getattr(battle, "lost", False):
                    self._summary_counts["lost"] += 1
                    opp_entry["lost"] += 1
                else:
                    self._summary_counts["tied"] += 1
                    opp_entry["tied"] += 1
                if self._progress_every:
                    finished = self._summary_counts["finished"]
                    if finished % self._progress_every == 0 or finished == self._total_battles:
                        remaining = max(self._total_battles - finished, 0)
                        wins = self._summary_counts["won"]
                        losses = self._summary_counts["lost"]
                        ties = self._summary_counts["tied"]
                        print(
                            f"📊 Progress: {finished}/{self._total_battles} battles "
                            f"({remaining} left) | W/L/T {wins}/{losses}/{ties}"
                        )
                turns = getattr(battle, "turn", 0) or 0
                battle_stats["turns"].append(turns)
                try:
                    battle_stats["remaining"].append(
                        sum(1 for p in battle.team.values() if not p.fainted)
                    )
                    battle_stats["opp_remaining"].append(
                        sum(1 for p in battle.opponent_team.values() if not p.fainted)
                    )
                except Exception:
                    pass
                if battle_has_forfeit(battle):
                    battle_stats["forfeits"] += 1
                if self._snapshot_every and (self._summary_counts["finished"] % self._snapshot_every == 0):
                    self._emit_snapshot("battle_end", battle=battle)
            self._battles.pop(battle.battle_tag, None)

        def _log_action(self, battle, order, duration_ms):
            tag = getattr(battle, "battle_tag", "unknown")
            active = getattr(battle, "active_pokemon", None)
            opponent = getattr(battle, "opponent_active_pokemon", None)
            active_name = getattr(active, "species", "unknown")
            opp_name = getattr(opponent, "species", "unknown")
            turn = getattr(battle, "turn", "?")
            if hasattr(order, "order"):
                move_or_switch = order.order
                if hasattr(move_or_switch, "species"):
                    action = f"switch->{move_or_switch.species}"
                elif hasattr(move_or_switch, "id"):
                    action = f"move->{move_or_switch.id}"
                else:
                    action = "order"
            else:
                action = "order"
            print(
                f"[{tag}] turn={turn} {active_name} vs {opp_name} "
                f"{action} ({duration_ms:.1f}ms)"
            )

        async def _handle_battle_message(self, split_messages):
            try:
                await super()._handle_battle_message(split_messages)
            except Exception as exc:
                battle_tag = split_messages[0][0] if split_messages else "unknown"
                self._guard_errors += 1
                action_stats["guard_errors"] += 1
                self.logger.warning(
                    "Guard: battle message error in %s (%s). Leaving battle.",
                    battle_tag,
                    exc,
                )
                self._emit_snapshot("guard_error")
                try:
                    if hasattr(self.ps_client, "send_message"):
                        await self.ps_client.send_message(f"/leave {battle_tag}")
                except Exception:
                    pass
                self._battles.pop(battle_tag, None)

    class TrackedRuleBot(TrackedBase, RuleBotPlayer):
        def __init__(self, *args, data_collector=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.data_collector = data_collector

        def choose_move(self, battle):
            start = time.perf_counter()
            self._maybe_send_chat(battle)
            try:
                order = super().choose_move(battle)
            except Exception as exc:
                print(f"⚠️ Decision error: {exc}")
                return self.choose_random_move(battle)
            duration_ms = (time.perf_counter() - start) * 1000
            tag = getattr(battle, "battle_tag", "unknown")

            # Record for training data
            if self.data_collector:
                self.data_collector.record(battle, order)

            action_stats["actions"] += 1
            action_stats["decision_ms_sum"] += duration_ms
            action_stats["decision_ms_max"] = max(action_stats["decision_ms_max"], duration_ms)
            action_stats["decision_ms_count"] += 1
            if duration_ms >= decision_slow_ms:
                action_stats["decision_slow"] += 1
            if self._status_available_in_moves(getattr(battle, "available_moves", [])):
                action_stats["status_available"] += 1
            if getattr(battle, "available_switches", []):
                action_stats["switch_available"] += 1
            if getattr(battle, "force_switch", False):
                action_stats["force_switch"] += 1
            if getattr(battle, "can_tera", False) and getattr(battle.active_pokemon, "tera_type", None):
                action_stats["tera_available"] += 1

            if hasattr(order, "order"):
                move_or_switch = order.order
                if hasattr(move_or_switch, "species"):
                    action_stats["switches"] += 1
                    action_stats["per_battle"][tag]["switches"] += 1
                    action_stats["switch_targets"][move_or_switch.species] += 1
                elif hasattr(move_or_switch, "id"):
                    move_id = move_or_switch.id
                    action_stats["moves"][move_id] += 1
                    action_stats["per_battle"][tag]["moves"] += 1
                    if move_id in self.ENTRY_HAZARDS or move_id in self.ANTI_HAZARDS_MOVES:
                        action_stats["hazards"] += 1
                        action_stats["per_battle"][tag]["hazards"] += 1
                    if self._is_status_move(move_or_switch):
                        action_stats["status"] += 1
                        action_stats["per_battle"][tag]["status"] += 1
                    if getattr(move_or_switch, "boosts", None):
                        action_stats["setup"] += 1
                        action_stats["per_battle"][tag]["setup"] += 1
                    if (not self._is_status_move(move_or_switch) and
                        not getattr(move_or_switch, "boosts", None) and
                        move_id not in self.ENTRY_HAZARDS and
                        move_id not in self.ANTI_HAZARDS_MOVES):
                        action_stats["attacks"] += 1
                        action_stats["per_battle"][tag]["attacks"] += 1

            if log_decisions or duration_ms >= decision_slow_ms:
                self._log_action(battle, order, duration_ms)

            if hasattr(order, "terastallize") and order.terastallize:
                action_stats["tera"] += 1
                action_stats["per_battle"][tag]["tera"] += 1

            return order

    class TrackedOranguruEngine(TrackedBase, OranguruEnginePlayer):
        def __init__(self, *args, data_collector=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.data_collector = data_collector

        def choose_move(self, battle):
            start = time.perf_counter()
            self._maybe_send_chat(battle)
            try:
                order = super().choose_move(battle)
            except Exception as exc:
                print(f"⚠️ Decision error: {exc}")
                return self.choose_random_move(battle)
            duration_ms = (time.perf_counter() - start) * 1000
            tag = getattr(battle, "battle_tag", "unknown")

            if self.data_collector:
                self.data_collector.record(battle, order)

            action_stats["actions"] += 1
            action_stats["decision_ms_sum"] += duration_ms
            action_stats["decision_ms_max"] = max(action_stats["decision_ms_max"], duration_ms)
            action_stats["decision_ms_count"] += 1
            if duration_ms >= decision_slow_ms:
                action_stats["decision_slow"] += 1
            if self._status_available_in_moves(getattr(battle, "available_moves", [])):
                action_stats["status_available"] += 1
            if getattr(battle, "available_switches", []):
                action_stats["switch_available"] += 1
            if getattr(battle, "force_switch", False):
                action_stats["force_switch"] += 1
            if getattr(battle, "can_tera", False) and getattr(battle.active_pokemon, "tera_type", None):
                action_stats["tera_available"] += 1

            if hasattr(order, "order"):
                move_or_switch = order.order
                if hasattr(move_or_switch, "species"):
                    action_stats["switches"] += 1
                    action_stats["per_battle"][tag]["switches"] += 1
                    action_stats["switch_targets"][move_or_switch.species] += 1
                elif hasattr(move_or_switch, "id"):
                    move_id = move_or_switch.id
                    action_stats["moves"][move_id] += 1
                    action_stats["per_battle"][tag]["moves"] += 1
                    if move_id in self.ENTRY_HAZARDS or move_id in self.ANTI_HAZARDS_MOVES:
                        action_stats["hazards"] += 1
                        action_stats["per_battle"][tag]["hazards"] += 1
                    if self._is_status_move(move_or_switch):
                        action_stats["status"] += 1
                        action_stats["per_battle"][tag]["status"] += 1
                    if getattr(move_or_switch, "boosts", None):
                        action_stats["setup"] += 1
                        action_stats["per_battle"][tag]["setup"] += 1
                    if (not self._is_status_move(move_or_switch) and
                        not getattr(move_or_switch, "boosts", None) and
                        move_id not in self.ENTRY_HAZARDS and
                        move_id not in self.ANTI_HAZARDS_MOVES):
                        action_stats["attacks"] += 1
                        action_stats["per_battle"][tag]["attacks"] += 1

            if log_decisions or duration_ms >= decision_slow_ms:
                self._log_action(battle, order, duration_ms)

            if hasattr(order, "terastallize") and order.terastallize:
                action_stats["tera"] += 1
                action_stats["per_battle"][tag]["tera"] += 1

            return order

    class TrackedRLPlayer(TrackedBase, RLPlayer):
        def __init__(self, *args, data_collector=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.data_collector = data_collector

        def choose_move(self, battle):
            start = time.perf_counter()
            self._maybe_send_chat(battle)
            self._last_battle = battle
            if battle.battle_tag != self._battle_tag:
                self._battle_tag = battle.battle_tag
                self._reset_hidden()

            features = self.feature_builder.build(battle)
            mask = self._build_mask(battle)
            features_t = torch.tensor([features], dtype=torch.float, device=self.device)
            mask_t = torch.tensor([mask], dtype=torch.bool, device=self.device)

            with torch.no_grad():
                if getattr(self.model, "is_recurrent", False):
                    logits, value, next_hidden = self.model.forward_step(features_t, self._hidden)
                    masked_logits = logits.masked_fill(~mask_t, -1e9)
                    masked_logits = torch.clamp(masked_logits, min=-1e8, max=1e8)
                    temp = max(1e-3, float(policy_temperature))
                    if temp != 1.0:
                        masked_logits = masked_logits / temp
                    probs = F.softmax(masked_logits, dim=-1)
                    probs = torch.clamp(probs, min=1e-8)
                    probs = probs / probs.sum(dim=-1, keepdim=True)
                    action = masked_logits.argmax(dim=-1)
                    self._hidden = next_hidden
                else:
                    logits, value = self.model.forward(features_t, mask_t)
                    logits = torch.clamp(logits, min=-1e8, max=1e8)
                    temp = max(1e-3, float(policy_temperature))
                    if temp != 1.0:
                        logits = logits / temp
                    probs = F.softmax(logits, dim=-1)
                    probs = torch.clamp(probs, min=1e-8)
                    probs = probs / probs.sum(dim=-1, keepdim=True)
                    action = logits.argmax(dim=-1)

            probs_1d = probs.squeeze(0)
            mask_1d = mask_t.squeeze(0)
            legal_probs = torch.where(mask_1d, probs_1d, torch.zeros_like(probs_1d))
            switch_probs = legal_probs[4:9]
            raw_switch_mass = float(switch_probs.sum().item())
            non_switch_probs = legal_probs.clone()
            non_switch_probs[4:9] = 0.0
            non_switch_mass = float(non_switch_probs.sum().item())
            switch_actions = int(mask_1d[4:9].sum().item())
            force_switch = bool(getattr(battle, "force_switch", False))
            switch_available = switch_actions > 0

            adjusted = False
            target_switch_mass = min(max(0.0, float(switch_mass_min)), 0.95)
            if (switch_available and not force_switch and target_switch_mass > 0.0
                    and raw_switch_mass < target_switch_mass):
                adjusted = True
                if raw_switch_mass > 0:
                    switch_probs = switch_probs * (target_switch_mass / raw_switch_mass)
                else:
                    switch_probs = torch.zeros_like(switch_probs)
                    if switch_actions > 0:
                        switch_probs[mask_1d[4:9]] = target_switch_mass / switch_actions

                if non_switch_mass > 0:
                    non_switch_probs = non_switch_probs * ((1.0 - target_switch_mass) / non_switch_mass)
                else:
                    non_switch_probs = torch.zeros_like(non_switch_probs)

                legal_probs = non_switch_probs
                legal_probs[4:9] = switch_probs

            adj_switch_mass = float(legal_probs[4:9].sum().item())
            action_dist = legal_probs / legal_probs.sum().clamp(min=1e-8)

            if switch_available:
                stats = self._switch_mass_stats
                stats["count"] += 1
                stats["raw_sum"] += raw_switch_mass
                stats["raw_min"] = min(stats["raw_min"], raw_switch_mass)
                stats["raw_max"] = max(stats["raw_max"], raw_switch_mass)
                if raw_switch_mass < switch_mass_warn:
                    stats["raw_low"] += 1
                stats["sum"] += adj_switch_mass
                stats["min"] = min(stats["min"], adj_switch_mass)
                stats["max"] = max(stats["max"], adj_switch_mass)
                if adj_switch_mass < switch_mass_warn:
                    stats["low"] += 1
                if adjusted:
                    stats["boosted"] += 1
            if sample_actions:
                action = torch.distributions.Categorical(action_dist).sample()
            else:
                action = action_dist.argmax(dim=-1)

            action_idx = action.item()
            order = self._action_to_order(battle, action_idx)
            duration_ms = (time.perf_counter() - start) * 1000

            # Record for training data
            if self.data_collector:
                self.data_collector.record(battle, order)

            action_stats["actions"] += 1
            action_stats["decision_ms_sum"] += duration_ms
            action_stats["decision_ms_max"] = max(action_stats["decision_ms_max"], duration_ms)
            action_stats["decision_ms_count"] += 1
            if duration_ms >= decision_slow_ms:
                action_stats["decision_slow"] += 1
            if self._status_available_in_moves(getattr(battle, "available_moves", [])):
                action_stats["status_available"] += 1
            if getattr(battle, "available_switches", []):
                action_stats["switch_available"] += 1
            if getattr(battle, "force_switch", False):
                action_stats["force_switch"] += 1
            if getattr(battle, "can_tera", False) and getattr(battle.active_pokemon, "tera_type", None):
                action_stats["tera_available"] += 1

            if hasattr(order, "order"):
                move_or_switch = order.order
                tag = getattr(battle, "battle_tag", "unknown")
                if hasattr(move_or_switch, "species"):
                    action_stats["switches"] += 1
                    action_stats["per_battle"][tag]["switches"] += 1
                    action_stats["switch_targets"][move_or_switch.species] += 1
                elif hasattr(move_or_switch, "id"):
                    move_id = move_or_switch.id
                    action_stats["moves"][move_id] += 1
                    action_stats["per_battle"][tag]["moves"] += 1
                    if move_id in RuleBotPlayer.ENTRY_HAZARDS or move_id in RuleBotPlayer.ANTI_HAZARDS_MOVES:
                        action_stats["hazards"] += 1
                        action_stats["per_battle"][tag]["hazards"] += 1
                    if self._is_status_move(move_or_switch):
                        action_stats["status"] += 1
                        action_stats["per_battle"][tag]["status"] += 1
                    if getattr(move_or_switch, "boosts", None):
                        action_stats["setup"] += 1
                        action_stats["per_battle"][tag]["setup"] += 1
                    if (not self._is_status_move(move_or_switch) and
                        not getattr(move_or_switch, "boosts", None) and
                        move_id not in RuleBotPlayer.ENTRY_HAZARDS and
                        move_id not in RuleBotPlayer.ANTI_HAZARDS_MOVES):
                        action_stats["attacks"] += 1
                        action_stats["per_battle"][tag]["attacks"] += 1

            if decision_log:
                self._decision_count += 1
                if decision_log_max <= 0 or self._decision_logged < decision_log_max:
                    if self._decision_count % max(1, decision_log_every) == 0:
                        self._decision_logged += 1
                        self._log_decision(
                            battle,
                            action_idx,
                            action_dist,
                            mask,
                            duration_ms,
                            topk=decision_log_topk,
                            value=value,
                            adjusted=adjusted,
                        )
            if log_decisions or duration_ms >= decision_slow_ms:
                self._log_action(battle, order, duration_ms)

            if hasattr(order, "terastallize") and order.terastallize:
                action_stats["tera"] += 1
                action_stats["per_battle"][tag]["tera"] += 1

            return order

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

        def _log_decision(self, battle, action_idx, probs, mask, duration_ms, topk=3,
                          value=None, adjusted: bool = False):
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

            topk = max(1, min(topk, len(legal_probs)))
            top_vals, top_idxs = torch.topk(legal_probs, topk)
            top_parts = []
            for idx, val in zip(top_idxs.tolist(), top_vals.tolist()):
                if val <= 0:
                    continue
                top_parts.append(f"{self._action_label(battle, idx)}@{val:.2f}")

            chosen_label = self._action_label(battle, action_idx)
            avail_moves = len(getattr(battle, "available_moves", []))
            avail_switches = len(getattr(battle, "available_switches", []))
            force_switch = bool(getattr(battle, "force_switch", False))
            tera_available = bool(getattr(battle, "can_tera", False))
            value_s = f"{float(value.item()):.3f}" if value is not None else "?"
            adj_flag = 1 if adjusted else 0

            print(
                "DECISION "
                f"tag={tag} turn={turn} {active_name}({hp_s}) vs {opp_name}({opp_hp_s}) "
                f"avail_m={avail_moves} avail_s={avail_switches} force={int(force_switch)} "
                f"tera={int(tera_available)} "
                f"chosen={chosen_label} "
                f"mass_m={move_mass:.2f} mass_s={switch_mass:.2f} mass_t={tera_mass:.2f} "
                f"adj={adj_flag} v={value_s} "
                f"top=[{', '.join(top_parts)}] "
                f"t={duration_ms:.0f}ms"
            )

    log_level_value = getattr(logging, log_level.upper(), None)
    if log_level_value is None:
        log_level_value = logging.INFO

    challenge_opponents = challenge_opponents or []
    if challenge_opponents:
        per_opp = int(challenge_battles) if challenge_battles is not None else int(n_battles)
        per_opp = max(1, per_opp)
        total_target = per_opp * len(challenge_opponents)
    else:
        total_target = int(n_battles)
    total = 0
    wins = 0
    losses = 0
    ties = 0
    retries = 0

    if challenge_opponents:
        opponents_str = ", ".join(challenge_opponents)
        print(
            f"🤖 Challenging as {username} in {battle_format} "
            f"({total_target} battles vs {opponents_str})..."
        )
    else:
        print(f"🤖 Laddering as {username} in {battle_format} ({n_battles} battles)...")
    if collect_data:
        print(
            f"   📊 Collecting training data (wins_only={wins_only}, "
            f"min_turns={min_turns}, min_actions={min_actions}, "
            f"min_rating={min_rating}, min_player_rating={min_player_rating}, "
            f"min_opponent_rating={min_opponent_rating}, skip_forfeit={skip_forfeit}, "
            f"min_win_remaining={min_win_remaining}, max_opp_remaining={max_opp_remaining})"
        )

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

    async def _wait_logged_in(player) -> bool:
        ps_client = getattr(player, "ps_client", None)
        if ps_client is None:
            return False
        wait_coro = ps_client.logged_in.wait()
        if login_timeout_s is None or login_timeout_s <= 0:
            await wait_coro
            return True
        try:
            await asyncio.wait_for(wait_coro, timeout=login_timeout_s)
            return True
        except asyncio.TimeoutError:
            return False

    pending_rejoins: list[str] = []

    async def _rejoin_pending(bot, tags: list[str]):
        if not tags:
            return
        try:
            await bot.ps_client.logged_in.wait()
        except Exception:
            return
        for tag in tags:
            try:
                await bot.ps_client.send_message(f"/rejoin {tag}")
            except Exception:
                pass

    bot = None
    interrupted = False
    try:
        while total < total_target:
            remaining = total_target - total
            if remaining <= 0:
                break
            if player_type == "rl":
                if not checkpoint_path:
                    print("❌ --checkpoint is required when --player rl")
                    return 1
                model, config = load_checkpoint_for_ladder(checkpoint_path, device)
                bot = TrackedRLPlayer(
                    model=model,
                    config=config,
                    device=device,
                    training=False,
                    track_illegal=False,
                    battle_format=battle_format,
                    max_concurrent_battles=max_concurrent,
                    account_configuration=account_config,
                    server_configuration=ShowdownServerConfiguration,
                    log_level=log_level_value,
                    open_timeout=open_timeout,
                    ping_interval=ping_interval,
                    ping_timeout=ping_timeout,
                    start_timer_on_battle_start=start_timer,
                    data_collector=data_collector,
                    total_battles=remaining,
                    progress_every=progress_every,
                )
            elif player_type == "oranguru_engine":
                bot = TrackedOranguruEngine(
                    battle_format=battle_format,
                    max_concurrent_battles=max_concurrent,
                    account_configuration=account_config,
                    server_configuration=ShowdownServerConfiguration,
                    log_level=log_level_value,
                    open_timeout=open_timeout,
                    ping_interval=ping_interval,
                    ping_timeout=ping_timeout,
                    start_timer_on_battle_start=start_timer,
                    data_collector=data_collector,
                    total_battles=remaining,
                    progress_every=progress_every,
                )
            else:
                bot = TrackedRuleBot(
                    battle_format=battle_format,
                    max_concurrent_battles=max_concurrent,
                    account_configuration=account_config,
                    server_configuration=ShowdownServerConfiguration,
                    log_level=log_level_value,
                    open_timeout=open_timeout,
                    ping_interval=ping_interval,
                    ping_timeout=ping_timeout,
                    start_timer_on_battle_start=start_timer,
                    data_collector=data_collector,
                    total_battles=remaining,
                    progress_every=progress_every,
                )

            try:
                logged_in = await _wait_logged_in(bot)
                if not logged_in:
                    raise TimeoutError("Login timeout waiting for Showdown websocket")
                if rejoin_active and pending_rejoins:
                    asyncio.create_task(_rejoin_pending(bot, pending_rejoins))
                    pending_rejoins = []
                if challenge_opponents:
                    battles_per_opponent = int(challenge_battles) if challenge_battles is not None else int(n_battles)
                    battles_per_opponent = max(1, battles_per_opponent)
                    planned = 0
                    for opponent in challenge_opponents:
                        if planned >= remaining:
                            break
                        to_play = min(battles_per_opponent, max(0, remaining - planned))
                        if to_play <= 0:
                            break
                        opponent_id = to_id_str(opponent)
                        print(f"   📌 Sending {to_play} challenge(s) to {opponent_id}...")
                        await bot.send_challenges(opponent_id, to_play)
                        planned += to_play
                else:
                    await bot.ladder(remaining)
            except Exception as exc:
                retries += 1
                print(f"⚠️ Ladder error ({username}): {exc}")
                if rejoin_active:
                    pending_rejoins = [
                        tag for tag, battle in bot.battles.items()
                        if not getattr(battle, "finished", False)
                    ]
                await _safe_stop(bot)
                if not auto_reconnect or retries > max_retries:
                    raise
                if retry_wait_s > 0:
                    await asyncio.sleep(retry_wait_s)
            finally:
                counts = getattr(bot, "_summary_counts", None)
                if counts:
                    wins += counts["won"]
                    losses += counts["lost"]
                    ties += counts["tied"]
                    total += counts["finished"]
                else:
                    wins += bot.n_won_battles
                    losses += bot.n_lost_battles
                    ties += bot.n_tied_battles
                    total += bot.n_finished_battles
    except (KeyboardInterrupt, asyncio.CancelledError):
        interrupted = True
        if bot is not None:
            await _safe_stop(bot)
    if interrupted:
        print("⚠️ Ladder interrupted; printing partial summary.")
    if bot is not None:
        try:
            bot._emit_snapshot("final")
        except Exception:
            pass

    # Commit finished battles to data collector
    if data_collector:
        for tag, battle in bot.battles.items():
            if battle.finished:
                won = battle.won if hasattr(battle, 'won') else False
                data_collector.commit_battle(battle, won, wins_only=wins_only)
        data_collector.save(data_output)
        print(f"   💾 Saved {len(data_collector.trajectories)} trajectories")
        if data_collector.skipped:
            print(f"   🧹 Skipped: {dict(data_collector.skipped)}")

    total_actions = max(action_stats["actions"], 1)
    print("\n📌 Action Summary")
    print(f"   Switch rate: {action_stats['switches'] / total_actions:.1%}")
    print(f"   Attack rate: {action_stats['attacks'] / total_actions:.1%}")
    print(f"   Status rate: {action_stats['status'] / total_actions:.1%}")
    print(f"   Setup rate:  {action_stats['setup'] / total_actions:.1%}")
    print(f"   Hazard rate: {action_stats['hazards'] / total_actions:.1%}")
    print(f"   Status avail: {action_stats['status_available'] / total_actions:.1%}")
    print(f"   Switch avail: {action_stats['switch_available'] / total_actions:.1%}")
    print(f"   Force switch: {action_stats['force_switch'] / total_actions:.1%}")
    print(f"   Tera avail:   {action_stats['tera_available'] / total_actions:.1%}")
    print(f"   Tera used/avail: {action_stats['tera']}/{action_stats['tera_available']}")

    if action_stats["moves"]:
        top_moves = action_stats["moves"].most_common(10)
        print("\n📌 Top Moves")
        for move_id, count in top_moves:
            print(f"   {move_id}: {count}")
    if action_stats["switch_targets"]:
        top_switches = action_stats["switch_targets"].most_common(10)
        print("\n📌 Top Switches")
        for species, count in top_switches:
            print(f"   {species}: {count}")

    switch_stats = getattr(bot, "_switch_mass_stats", None)
    if switch_stats and switch_stats["count"] > 0:
        avg_raw = switch_stats["raw_sum"] / switch_stats["count"]
        avg_adj = switch_stats["sum"] / switch_stats["count"]
        print("\n📌 Policy Switch Mass (RL)")
        print(
            f"   decisions w/ switches: {switch_stats['count']} | "
            f"raw avg={avg_raw:.2f} (min {switch_stats['raw_min']:.2f} max {switch_stats['raw_max']:.2f}) "
            f"low<{switch_mass_warn:.2f}: {switch_stats['raw_low']}"
        )
        print(
            f"   adjusted avg={avg_adj:.2f} (min {switch_stats['min']:.2f} max {switch_stats['max']:.2f}) "
            f"low<{switch_mass_warn:.2f}: {switch_stats['low']} | boosted={switch_stats['boosted']}"
        )

    if verbose and bot and bot.battles:
        print("\n📌 Battle Details")
        for tag, battle in sorted(bot.battles.items()):
            if not battle.finished:
                continue
            remaining = sum(1 for p in battle.team.values() if not p.fainted)
            opp_remaining = sum(1 for p in battle.opponent_team.values() if not p.fainted)
            result = "W" if battle.won else "L" if battle.lost else "T"
            per = action_stats["per_battle"].get(tag, Counter())
            print(
                f"   {tag}: {result} | turns={battle.turn} | remaining={remaining} | "
                f"opp_remaining={opp_remaining} | switches={per.get('switches', 0)} | "
                f"status={per.get('status', 0)} | setup={per.get('setup', 0)} | "
                f"hazards={per.get('hazards', 0)} | tera={per.get('tera', 0)}"
            )

    repeat_opponents = [
        (name, stats)
        for name, stats in opponent_stats.items()
        if stats.get("played", 0) >= 2
    ]
    if repeat_opponents:
        print("\n📌 Opponent Summary (>=2 games)")
        for name, stats in sorted(
            repeat_opponents, key=lambda x: x[1].get("played", 0), reverse=True
        ):
            played = stats.get("played", 0)
            won = stats.get("won", 0)
            lost = stats.get("lost", 0)
            tied = stats.get("tied", 0)
            win_rate = (won / played) if played else 0.0
            print(f"   {name}: {won}/{lost}/{tied} ({win_rate:.1%}) in {played}")

    if total:
        win_rate = wins / total if total else 0.0
        turns = battle_stats["turns"]
        remaining = battle_stats["remaining"]
        opp_remaining = battle_stats["opp_remaining"]
        forfeits = battle_stats["forfeits"]
        guard_errors = action_stats["guard_errors"]
        if turns:
            avg_turns = sum(turns) / len(turns)
            print(f"   Avg turns: {avg_turns:.1f}")
        if remaining:
            avg_remaining = sum(remaining) / len(remaining)
            print(f"   Avg remaining mons: {avg_remaining:.2f}")
        if opp_remaining:
            avg_opp_remaining = sum(opp_remaining) / len(opp_remaining)
            print(f"   Avg opp remaining: {avg_opp_remaining:.2f}")
        if forfeits:
            print(f"   Forfeits: {forfeits}")
        if guard_errors:
            print(f"   Guard errors: {guard_errors}")
        if action_stats["decision_ms_count"]:
            avg_ms = action_stats["decision_ms_sum"] / action_stats["decision_ms_count"]
            print(
                f"   Decision time: avg {avg_ms:.1f}ms | "
                f"max {action_stats['decision_ms_max']:.1f}ms | "
                f"slow {action_stats['decision_slow']}"
            )
        print(f"✅ Finished {total} battles: {wins}W/{losses}L/{ties}T ({win_rate:.1%} win rate)")
    else:
        print("⚠️  No finished battles recorded.")

    return 0


async def ladder_rulebot_multi(
    accounts: list[tuple[str, str]],
    n_battles: int,
    battle_format: str,
    max_concurrent: int,
    verbose: bool,
    pre_eval: bool,
    pre_eval_battles: int,
    log_decisions: bool,
    decision_slow_ms: int,
    log_level: str,
    open_timeout: float | None,
    ping_interval: float | None,
    ping_timeout: float | None,
    start_timer: bool,
    player_type: str,
    checkpoint_path: str | None,
    device: str,
    decision_log: bool,
    decision_log_every: int,
    decision_log_topk: int,
    decision_log_max: int,
    policy_temperature: float,
    sample_actions: bool,
    switch_mass_min: float,
    switch_mass_warn: float,
    collect_data: bool,
    wins_only: bool,
    data_output: str,
    min_turns: int,
    min_actions: int,
    min_rating: int | None,
    min_player_rating: int | None,
    min_opponent_rating: int | None,
    skip_forfeit: bool,
    max_illegal_rate: float | None,
    min_win_remaining: int | None,
    max_opp_remaining: int | None,
    win_weight: float,
    loss_weight: float,
    tag_prefix: str,
    progress_every: int,
    auto_reconnect: bool = False,
    max_retries: int = 2,
    retry_wait_s: float = 5.0,
    rejoin_active: bool = False,
    login_timeout_s: float | None = None,
    chat_message: str = "",
    challenge_opponents: list[str] | None = None,
    challenge_battles: int | None = None,
    snapshot_log: str | None = None,
    snapshot_every: int = 1,
):
    if not accounts:
        print("❌ No accounts provided.")
        return 1

    tasks = []
    for idx, (acct_user, acct_pass) in enumerate(accounts):
        suffix = normalize_name(acct_user) or f"acct{idx + 1}"
        output_path = data_output
        tag_prefix_i = tag_prefix
        if collect_data and data_output:
            base, ext = os.path.splitext(data_output)
            output_path = f"{base}_{suffix}{ext}" if ext else f"{data_output}_{suffix}"
            tag_prefix_i = f"{tag_prefix}_{suffix}"
        tasks.append(
            ladder_rulebot(
                n_battles,
                battle_format,
                max_concurrent,
                verbose,
                pre_eval,
                pre_eval_battles,
                log_decisions,
                decision_slow_ms,
                log_level,
                open_timeout,
                ping_interval,
                ping_timeout,
                start_timer,
                player_type,
                checkpoint_path,
                device,
                decision_log,
                decision_log_every,
                decision_log_topk,
                decision_log_max,
                policy_temperature,
                sample_actions,
                switch_mass_min,
                switch_mass_warn,
                collect_data,
                wins_only,
                output_path,
                min_turns,
                min_actions,
                min_rating,
                min_player_rating,
                min_opponent_rating,
                skip_forfeit,
                max_illegal_rate,
                min_win_remaining,
                max_opp_remaining,
                win_weight,
                loss_weight,
                tag_prefix_i,
                progress_every,
                username=acct_user,
                password=acct_pass,
                auto_reconnect=auto_reconnect,
                max_retries=max_retries,
                retry_wait_s=retry_wait_s,
                rejoin_active=rejoin_active,
                login_timeout_s=login_timeout_s,
                chat_message=chat_message,
                challenge_opponents=challenge_opponents,
                challenge_battles=challenge_battles,
                snapshot_log=snapshot_log,
                snapshot_every=snapshot_every,
            )
        )

    results = await asyncio.gather(*tasks)
    return 0 if all(code == 0 for code in results) else 1


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--battles", type=int, default=20, help="Number of ladder battles")
    parser.add_argument("--format", type=str, default="gen9randombattle",
                        help="Showdown battle format")
    parser.add_argument("--max-concurrent", type=int, default=1,
                        help="Max concurrent ladder battles")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-battle summary after laddering")
    parser.add_argument("--pre-eval", action="store_true",
                        help="Run local evaluation before laddering")
    parser.add_argument("--pre-eval-battles", type=int, default=200,
                        help="Battles per opponent for pre-eval")
    parser.add_argument("--log-decisions", action="store_true",
                        help="Print a line for every decision")
    parser.add_argument("--decision-slow-ms", type=int, default=1500,
                        help="Warn when a decision exceeds this time (ms)")
    parser.add_argument("--log-level", type=str, default="INFO",
                        help="Set Showdown client log level (DEBUG/INFO/WARNING/ERROR)")
    parser.add_argument("--open-timeout", type=float, default=10.0,
                        help="Websocket open timeout (seconds)")
    parser.add_argument("--ping-interval", type=float, default=20.0,
                        help="Websocket ping interval (seconds)")
    parser.add_argument("--ping-timeout", type=float, default=20.0,
                        help="Websocket ping timeout (seconds)")
    parser.add_argument("--player", type=str, choices=["rulebot", "rl", "oranguru_engine"], default="rulebot",
                        help="Which player to ladder with")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint path for --player rl")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device for RL model (cpu/cuda)")
    parser.add_argument("--decision-log", action="store_true",
                        help="Log compact decision traces for the RL model")
    parser.add_argument("--decision-log-every", type=int, default=1,
                        help="Log every N decisions when --decision-log is set")
    parser.add_argument("--decision-log-topk", type=int, default=3,
                        help="Top-k actions to print in decision logs")
    parser.add_argument("--decision-log-max", type=int, default=0,
                        help="Max decision logs to print (0 = no limit)")
    parser.add_argument("--policy-temperature", type=float, default=1.0,
                        help="Temperature for RL policy logits (>=0)")
    parser.add_argument("--sample-actions", action="store_true",
                        help="Sample actions instead of argmax for RL policy")
    parser.add_argument("--switch-mass-min", type=float, default=0.0,
                        help="Minimum probability mass to allocate to switches")
    parser.add_argument("--switch-mass-warn", type=float, default=0.05,
                        help="Threshold for low switch-mass warning stats")
    parser.add_argument("--start-timer", action="store_true", default=True,
                        help="Start Showdown battle timer automatically (default: on)")
    parser.add_argument("--no-start-timer", action="store_true",
                        help="Disable automatic battle timer")
    parser.add_argument("--collect-data", action="store_true",
                        help="Collect training data from ladder games")
    parser.add_argument("--wins-only", action="store_true",
                        help="Only save winning games for training")
    parser.add_argument("--data-output", type=str, default="data/ladder_demos.pkl",
                        help="Output path for collected training data")
    parser.add_argument("--min-turns", type=int, default=0,
                        help="Minimum turns required to keep a battle")
    parser.add_argument("--min-actions", type=int, default=0,
                        help="Minimum actions required to keep a battle")
    parser.add_argument("--min-rating", type=int, default=0,
                        help="Minimum rating required (uses player or opponent rating)")
    parser.add_argument("--min-player-rating", type=int, default=0,
                        help="Minimum player rating required (0 disables)")
    parser.add_argument("--min-opponent-rating", type=int, default=0,
                        help="Minimum opponent rating required (0 disables)")
    parser.add_argument("--skip-forfeit", action="store_true",
                        help="Skip battles that end in forfeit")
    parser.add_argument("--max-illegal-rate", type=float, default=-1.0,
                        help="Skip battles above this illegal rate (<=0 disables)")
    parser.add_argument("--min-win-remaining", type=int, default=0,
                        help="Minimum remaining Pokemon in wins (0 disables)")
    parser.add_argument("--max-opp-remaining", type=int, default=-1,
                        help="Maximum opponent remaining Pokemon in wins (<=0 disables)")
    parser.add_argument("--win-weight", type=float, default=1.0,
                        help="Weight for winning trajectories")
    parser.add_argument("--loss-weight", type=float, default=1.0,
                        help="Weight for losing trajectories")
    parser.add_argument("--tag-prefix", type=str, default="ladder",
                        help="Tag prefix for saved trajectories")
    parser.add_argument("--progress-every", type=int, default=25,
                        help="Print progress every N finished battles (0 disables)")
    parser.add_argument("--accounts-file", type=str, default=None,
                        help="Optional accounts file (one per line: user:pass or user,pass)")
    parser.add_argument("--accounts-limit", type=int, default=0,
                        help="Use only the first N accounts from --accounts-file (0 = all)")
    parser.add_argument("--account-name", action="append", default=[],
                        help="Account username(s) to use from --accounts-file (repeatable, comma-separated)")
    parser.add_argument("--auto-reconnect", action="store_true",
                        help="Retry laddering on websocket disconnects")
    parser.add_argument("--max-retries", type=int, default=2,
                        help="Max reconnect attempts per account")
    parser.add_argument("--retry-wait", type=float, default=5.0,
                        help="Seconds to wait before reconnecting")
    parser.add_argument("--rejoin-active", action="store_true",
                        help="Attempt to rejoin active battles after reconnecting")
    parser.add_argument("--login-timeout", type=float, default=30.0,
                        help="Seconds to wait for Showdown login (<=0 disables)")
    parser.add_argument("--chat-message", type=str, default="",
                        help="Optional battle chat message sent once per battle")
    parser.add_argument("--challenge", action="append", default=[],
                        help="Challenge specific players (repeatable, comma-separated)")
    parser.add_argument("--challenge-battles", type=int, default=0,
                        help="Battles per challenge opponent (0 uses --battles)")
    parser.add_argument("--snapshot-log", type=str, default="",
                        help="Optional JSONL file for per-battle progress snapshots")
    parser.add_argument("--snapshot-every", type=int, default=1,
                        help="Emit snapshot every N finished battles (0 disables)")
    args = parser.parse_args()

    open_timeout = None if args.open_timeout <= 0 else args.open_timeout
    ping_interval = None if args.ping_interval <= 0 else args.ping_interval
    ping_timeout = None if args.ping_timeout <= 0 else args.ping_timeout
    min_rating = None if args.min_rating <= 0 else args.min_rating
    min_player_rating = None if args.min_player_rating <= 0 else args.min_player_rating
    min_opponent_rating = None if args.min_opponent_rating <= 0 else args.min_opponent_rating
    max_illegal_rate = None if args.max_illegal_rate <= 0 else args.max_illegal_rate
    min_win_remaining = None if args.min_win_remaining <= 0 else args.min_win_remaining
    max_opp_remaining = None if args.max_opp_remaining <= 0 else args.max_opp_remaining
    progress_every = args.progress_every if args.progress_every > 0 else 0
    login_timeout_s = None if args.login_timeout <= 0 else args.login_timeout
    challenge_opponents = []
    for entry in args.challenge or []:
        for name in entry.split(","):
            name = name.strip()
            if name:
                challenge_opponents.append(name)
    challenge_battles = args.challenge_battles if args.challenge_battles > 0 else None
    snapshot_log = args.snapshot_log.strip() if args.snapshot_log else ""
    snapshot_log = snapshot_log or None
    snapshot_every = max(0, int(args.snapshot_every))
    account_names = []
    for entry in args.account_name or []:
        for name in entry.split(","):
            name = name.strip()
            if name:
                account_names.append(name)
    if account_names and not args.accounts_file:
        print("⚠️  --account-name is ignored without --accounts-file")

    async def _run():
        if args.accounts_file:
            accounts = load_accounts_from_file(args.accounts_file)
            if account_names:
                accounts = filter_accounts_by_name(accounts, account_names)
            if args.accounts_limit and args.accounts_limit > 0:
                accounts = accounts[: args.accounts_limit]
            if not accounts:
                print("❌ No matching accounts found for --account-name.")
                return 1
            return await ladder_rulebot_multi(
                accounts,
                args.battles,
                args.format,
                args.max_concurrent,
                args.verbose,
                args.pre_eval,
                args.pre_eval_battles,
                args.log_decisions,
                args.decision_slow_ms,
                args.log_level,
                open_timeout,
                ping_interval,
                ping_timeout,
                (args.start_timer and not args.no_start_timer),
                args.player,
                args.checkpoint,
                args.device,
                args.decision_log,
                args.decision_log_every,
                args.decision_log_topk,
                args.decision_log_max,
                args.policy_temperature,
                args.sample_actions,
                args.switch_mass_min,
                args.switch_mass_warn,
                args.collect_data,
                args.wins_only,
                args.data_output,
                args.min_turns,
                args.min_actions,
                min_rating,
                min_player_rating,
                min_opponent_rating,
                args.skip_forfeit,
                max_illegal_rate,
                min_win_remaining,
                max_opp_remaining,
                args.win_weight,
                args.loss_weight,
                args.tag_prefix,
                progress_every,
                auto_reconnect=args.auto_reconnect,
                max_retries=args.max_retries,
                retry_wait_s=args.retry_wait,
                rejoin_active=args.rejoin_active,
                login_timeout_s=login_timeout_s,
                chat_message=args.chat_message,
                challenge_opponents=challenge_opponents,
                challenge_battles=challenge_battles,
                snapshot_log=snapshot_log,
                snapshot_every=snapshot_every,
            )
        return await ladder_rulebot(
            args.battles,
            args.format,
            args.max_concurrent,
            args.verbose,
            args.pre_eval,
            args.pre_eval_battles,
            args.log_decisions,
            args.decision_slow_ms,
            args.log_level,
            open_timeout,
            ping_interval,
            ping_timeout,
            (args.start_timer and not args.no_start_timer),
            args.player,
            args.checkpoint,
            args.device,
            args.decision_log,
            args.decision_log_every,
            args.decision_log_topk,
            args.decision_log_max,
            args.policy_temperature,
            args.sample_actions,
            args.switch_mass_min,
            args.switch_mass_warn,
            args.collect_data,
            args.wins_only,
            args.data_output,
            args.min_turns,
            args.min_actions,
            min_rating,
            min_player_rating,
            min_opponent_rating,
            args.skip_forfeit,
            max_illegal_rate,
            min_win_remaining,
            max_opp_remaining,
            args.win_weight,
            args.loss_weight,
            args.tag_prefix,
            progress_every,
            auto_reconnect=args.auto_reconnect,
            max_retries=args.max_retries,
            retry_wait_s=args.retry_wait,
            rejoin_active=args.rejoin_active,
            login_timeout_s=login_timeout_s,
            chat_message=args.chat_message,
            challenge_opponents=challenge_opponents,
            challenge_battles=challenge_battles,
            snapshot_log=snapshot_log,
            snapshot_every=snapshot_every,
        )

    raise SystemExit(asyncio.run(_run()))
