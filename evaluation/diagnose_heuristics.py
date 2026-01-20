#!/usr/bin/env python3
"""
Diagnostic: Compare SmartHeuristicsPlayer vs SimpleHeuristicsPlayer
and log action statistics.
"""

import asyncio
import sys
from pathlib import Path
from dataclasses import dataclass, field

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from poke_env.player.baselines import SimpleHeuristicsPlayer
from src.utils.server_config import get_server_configuration

CustomServerConfig = get_server_configuration(default_port=8080)


@dataclass
class ActionStats:
    """Track action types."""
    total_actions: int = 0
    switches: int = 0
    attacks: int = 0
    setup_moves: int = 0
    status_moves: int = 0
    hazard_moves: int = 0
    tera_moves: int = 0
    turns_to_first_faint: list = field(default_factory=list)

    def switch_rate(self):
        return self.switches / max(self.total_actions, 1)

    def setup_rate(self):
        return self.setup_moves / max(self.total_actions, 1)

    def status_rate(self):
        return self.status_moves / max(self.total_actions, 1)

    def hazard_rate(self):
        return self.hazard_moves / max(self.total_actions, 1)


class TrackedSmartHeuristics(SimpleHeuristicsPlayer):
    """SmartHeuristicsPlayer with action tracking."""

    SETUP_MOVES = {'swordsdance', 'nastyplot', 'dragondance', 'calmmind',
                   'bulkup', 'irondefense', 'agility', 'shellsmash',
                   'quiverdance', 'coil', 'curse', 'workup', 'growth'}
    STATUS_MOVES = {'thunderwave', 'toxic', 'willowisp', 'spore', 'sleeppowder',
                    'stunspore', 'glare', 'nuzzle', 'yawn'}
    HAZARD_MOVES = {'stealthrock', 'spikes', 'toxicspikes', 'stickyweb',
                    'defog', 'rapidspin', 'tidyup', 'mortalspin'}

    def __init__(self, stats: ActionStats, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stats = stats
        self._battle_turns = {}
        self._first_faint_recorded = {}

    def choose_move(self, battle):
        order = super().choose_move(battle)

        self.stats.total_actions += 1

        # Track first faint
        battle_id = battle.battle_tag
        if battle_id not in self._battle_turns:
            self._battle_turns[battle_id] = 0
            self._first_faint_recorded[battle_id] = False
        self._battle_turns[battle_id] += 1

        # Check for first faint
        if not self._first_faint_recorded[battle_id]:
            my_fainted = sum(1 for p in battle.team.values() if p.fainted)
            opp_fainted = sum(1 for p in battle.opponent_team.values() if p.fainted)
            if my_fainted > 0 or opp_fainted > 0:
                self.stats.turns_to_first_faint.append(self._battle_turns[battle_id])
                self._first_faint_recorded[battle_id] = True

        # Analyze the order
        if hasattr(order, 'order'):
            move_or_switch = order.order
            if hasattr(move_or_switch, 'species'):  # It's a Pokemon (switch)
                self.stats.switches += 1
            elif hasattr(move_or_switch, 'id'):  # It's a move
                move_id = move_or_switch.id
                if move_id in self.SETUP_MOVES:
                    self.stats.setup_moves += 1
                elif move_id in self.STATUS_MOVES:
                    self.stats.status_moves += 1
                elif move_id in self.HAZARD_MOVES:
                    self.stats.hazard_moves += 1
                else:
                    self.stats.attacks += 1

                if hasattr(order, 'terastallize') and order.terastallize:
                    self.stats.tera_moves += 1

        return order


class TrackedSimpleHeuristics(SimpleHeuristicsPlayer):
    """SimpleHeuristicsPlayer with action tracking."""

    SETUP_MOVES = {'swordsdance', 'nastyplot', 'dragondance', 'calmmind',
                   'bulkup', 'irondefense', 'agility', 'shellsmash',
                   'quiverdance', 'coil', 'curse', 'workup', 'growth'}
    STATUS_MOVES = {'thunderwave', 'toxic', 'willowisp', 'spore', 'sleeppowder',
                    'stunspore', 'glare', 'nuzzle', 'yawn'}
    HAZARD_MOVES = {'stealthrock', 'spikes', 'toxicspikes', 'stickyweb',
                    'defog', 'rapidspin', 'tidyup', 'mortalspin'}

    def __init__(self, stats: ActionStats, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stats = stats
        self._battle_turns = {}
        self._first_faint_recorded = {}

    def choose_move(self, battle):
        order = super().choose_move(battle)

        self.stats.total_actions += 1

        # Track first faint
        battle_id = battle.battle_tag
        if battle_id not in self._battle_turns:
            self._battle_turns[battle_id] = 0
            self._first_faint_recorded[battle_id] = False
        self._battle_turns[battle_id] += 1

        # Check for first faint
        if not self._first_faint_recorded[battle_id]:
            my_fainted = sum(1 for p in battle.team.values() if p.fainted)
            opp_fainted = sum(1 for p in battle.opponent_team.values() if p.fainted)
            if my_fainted > 0 or opp_fainted > 0:
                self.stats.turns_to_first_faint.append(self._battle_turns[battle_id])
                self._first_faint_recorded[battle_id] = True

        # Analyze the order
        if hasattr(order, 'order'):
            move_or_switch = order.order
            if hasattr(move_or_switch, 'species'):  # It's a Pokemon (switch)
                self.stats.switches += 1
            elif hasattr(move_or_switch, 'id'):  # It's a move
                move_id = move_or_switch.id
                if move_id in self.SETUP_MOVES:
                    self.stats.setup_moves += 1
                elif move_id in self.STATUS_MOVES:
                    self.stats.status_moves += 1
                elif move_id in self.HAZARD_MOVES:
                    self.stats.hazard_moves += 1
                else:
                    self.stats.attacks += 1

                if hasattr(order, 'terastallize') and order.terastallize:
                    self.stats.tera_moves += 1

        return order


async def run_diagnostic(n_battles: int = 200):
    """Run SmartHeuristics vs SimpleHeuristics with stats tracking."""
    from src.players.smart_heuristics import SmartHeuristicsPlayer

    kwargs = {
        'battle_format': 'gen9randombattle',
        'max_concurrent_battles': 1,
        'server_configuration': CustomServerConfig,
    }

    print("=" * 60)
    print("🔍 DIAGNOSTIC: SmartHeuristics vs SimpleHeuristics")
    print("=" * 60)

    # Test SmartHeuristics as player
    smart_stats = ActionStats()
    simple_stats = ActionStats()

    print(f"\n📊 Running {n_battles} battles...")

    wins = 0
    for i in range(n_battles):
        # Fresh players each battle to avoid state issues
        smart_player = SmartHeuristicsPlayer(**kwargs)
        simple_opponent = SimpleHeuristicsPlayer(**kwargs)

        # Wrap with tracking
        class TrackedSmart(SmartHeuristicsPlayer):
            def choose_move(self, battle):
                order = super().choose_move(battle)
                smart_stats.total_actions += 1

                if hasattr(order, 'order'):
                    move_or_switch = order.order
                    if hasattr(move_or_switch, 'species'):
                        smart_stats.switches += 1
                    elif hasattr(move_or_switch, 'id'):
                        move_id = move_or_switch.id
                        if move_id in TrackedSmartHeuristics.SETUP_MOVES:
                            smart_stats.setup_moves += 1
                        elif move_id in TrackedSmartHeuristics.STATUS_MOVES:
                            smart_stats.status_moves += 1
                        elif move_id in TrackedSmartHeuristics.HAZARD_MOVES:
                            smart_stats.hazard_moves += 1
                        else:
                            smart_stats.attacks += 1
                return order

        smart_player = TrackedSmart(**kwargs)
        simple_opponent = TrackedSimpleHeuristics(simple_stats, **kwargs)

        prev_wins = smart_player.n_won_battles
        try:
            await asyncio.wait_for(
                smart_player.battle_against(simple_opponent, n_battles=1),
                timeout=60
            )
            if smart_player.n_won_battles > prev_wins:
                wins += 1
        except:
            pass

        if (i + 1) % 50 == 0:
            print(f"   {i+1}/{n_battles} battles, SmartHeuristics wins: {wins}")

    win_rate = wins / n_battles

    print("\n" + "=" * 60)
    print("📊 RESULTS")
    print("=" * 60)

    print(f"\n🎯 SmartHeuristics vs SimpleHeuristics: {wins}/{n_battles} = {win_rate:.1%}")

    print(f"\n📈 SmartHeuristics Action Stats:")
    print(f"   Total actions: {smart_stats.total_actions}")
    print(f"   Switch rate: {smart_stats.switch_rate():.1%}")
    print(f"   Setup rate: {smart_stats.setup_rate():.1%}")
    print(f"   Status rate: {smart_stats.status_rate():.1%}")
    print(f"   Hazard rate: {smart_stats.hazard_rate():.1%}")
    print(f"   Attack rate: {smart_stats.attacks / max(smart_stats.total_actions, 1):.1%}")

    print(f"\n📈 SimpleHeuristics Action Stats (opponent):")
    print(f"   Total actions: {simple_stats.total_actions}")
    print(f"   Switch rate: {simple_stats.switch_rate():.1%}")
    print(f"   Setup rate: {simple_stats.setup_rate():.1%}")
    print(f"   Status rate: {simple_stats.status_rate():.1%}")
    print(f"   Hazard rate: {simple_stats.hazard_rate():.1%}")
    print(f"   Attack rate: {simple_stats.attacks / max(simple_stats.total_actions, 1):.1%}")
    if simple_stats.turns_to_first_faint:
        avg_turns = sum(simple_stats.turns_to_first_faint) / len(simple_stats.turns_to_first_faint)
        print(f"   Avg turns to first faint: {avg_turns:.1f}")

    print("\n" + "=" * 60)

    if win_rate < 0.40:
        print("❌ SmartHeuristics is WORSE than SimpleHeuristics!")
        print("   This is a BAD teacher for imitation learning.")
    elif win_rate < 0.55:
        print("⚠️ SmartHeuristics is about EQUAL to SimpleHeuristics.")
        print("   Not a useful teacher - can't teach how to beat it.")
    else:
        print("✅ SmartHeuristics BEATS SimpleHeuristics.")
        print("   This could be a useful teacher.")

    return win_rate, smart_stats, simple_stats


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--battles", type=int, default=200)
    args = parser.parse_args()

    asyncio.run(run_diagnostic(args.battles))
