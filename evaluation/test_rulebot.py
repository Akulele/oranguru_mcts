#!/usr/bin/env python3
"""
Test RuleBot vs SimpleHeuristics to validate it as a teacher.

Target: RuleBot should beat SimpleHeuristics >55% to be useful as teacher.
"""

import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from poke_env.player.baselines import SimpleHeuristicsPlayer, MaxBasePowerPlayer, RandomPlayer
from src.utils.server_config import get_server_configuration
from poke_env.battle import MoveCategory

from src.players.rule_bot import RuleBotPlayer
from src.utils.features import load_moves
from src.utils.damage_calc import normalize_name

# Use local showdown server (override via env vars).
CustomServerConfig = get_server_configuration(default_port=8000)


async def test_rulebot(n_battles: int = 100, progress_every: int = 25, timeout_s: int = 60):
    """Test RuleBot against various opponents."""
    kwargs = {
        'battle_format': 'gen9randombattle',
        'max_concurrent_battles': 1,
        'server_configuration': CustomServerConfig,
    }

    print("=" * 60)
    print("🤖 RULEBOT EVALUATION")
    print("=" * 60)

    results = {}
    result_details = {}
    heuristics_stats = {
        "actions": 0,
        "switches": 0,
        "switch_available": 0,
        "attacks": 0,
        "status": 0,
        "setup": 0,
        "setup_available": 0,
        "hazards": 0,
        "hazard_available": 0,
        "tera": 0,
        "status_available": 0,
        "priority_available": 0,
        "priority": 0,
        "force_switch": 0,
        "endgame_turns": 0,
        "tera_available": 0,
        "can_tera_true": 0,
        "tera_type_present": 0,
        "turns": [],
        "remaining_mons": [],
        "move_counts": {},
        "switch_counts": {},
    }

    def wilson_interval(wins: int, n: int, z: float = 1.96):
        if n <= 0:
            return 0.0, 1.0
        phat = wins / n
        denom = 1 + (z * z) / n
        center = (phat + (z * z) / (2 * n)) / denom
        margin = (z * ((phat * (1 - phat) / n) + (z * z) / (4 * n * n)) ** 0.5) / denom
        return max(0.0, center - margin), min(1.0, center + margin)

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

    # Test 1: vs Random
    print("\n📊 Test 1: RuleBot vs Random...")
    wins = 0
    for i in range(n_battles):
        rulebot = RuleBotPlayer(**kwargs)
        random_opp = RandomPlayer(**kwargs)

        prev_wins = rulebot.n_won_battles
        try:
            await asyncio.wait_for(
                rulebot.battle_against(random_opp, n_battles=1),
                timeout=timeout_s
            )
            if rulebot.n_won_battles > prev_wins:
                wins += 1
        except Exception as e:
            pass
        finally:
            # Ensure proper cleanup of connections
            await _safe_stop(rulebot)
            await _safe_stop(random_opp)
            await asyncio.sleep(0.1)  # Brief delay to release resources

        if (i + 1) % progress_every == 0:
            print(f"   {i+1}/{n_battles}: {wins}/{i+1} wins ({wins/(i+1)*100:.1f}%)")

    results['random'] = wins / n_battles
    low, high = wilson_interval(wins, n_battles)
    result_details["random"] = {"wins": wins, "n": n_battles, "low": low, "high": high}
    print(f"   ✅ RuleBot vs Random: {wins}/{n_battles} = {results['random']:.1%} (95% CI {low:.1%}-{high:.1%})")

    # Test 2: vs MaxPower
    print("\n📊 Test 2: RuleBot vs MaxPower...")
    wins = 0
    for i in range(n_battles):
        rulebot = RuleBotPlayer(**kwargs)
        maxpower = MaxBasePowerPlayer(**kwargs)

        prev_wins = rulebot.n_won_battles
        try:
            await asyncio.wait_for(
                rulebot.battle_against(maxpower, n_battles=1),
                timeout=timeout_s
            )
            if rulebot.n_won_battles > prev_wins:
                wins += 1
        except Exception as e:
            pass
        finally:
            # Ensure proper cleanup of connections
            await _safe_stop(rulebot)
            await _safe_stop(maxpower)
            await asyncio.sleep(0.1)  # Brief delay to release resources

        if (i + 1) % progress_every == 0:
            print(f"   {i+1}/{n_battles}: {wins}/{i+1} wins ({wins/(i+1)*100:.1f}%)")

    results['maxpower'] = wins / n_battles
    low, high = wilson_interval(wins, n_battles)
    result_details["maxpower"] = {"wins": wins, "n": n_battles, "low": low, "high": high}
    print(f"   ✅ RuleBot vs MaxPower: {wins}/{n_battles} = {results['maxpower']:.1%} (95% CI {low:.1%}-{high:.1%})")

    # Test 3: vs SimpleHeuristics (THE KEY TEST)
    print("\n📊 Test 3: RuleBot vs SimpleHeuristics...")
    wins = 0

    class TrackedRuleBot(RuleBotPlayer):
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

        def _is_setup_move(self, move):
            if getattr(move, "boosts", None) and "self" in str(getattr(move, "target", "")).lower():
                return True
            return False

        def _is_hazard_move(self, move):
            return move.id in self.ENTRY_HAZARDS or move.id in self.ANTI_HAZARDS_MOVES

        def _move_priority(self, move):
            moves_data = load_moves()
            entry = moves_data.get(move.id, {})
            return entry.get("priority", 0)

        def choose_move(self, battle):
            order = super().choose_move(battle)

            heuristics_stats["actions"] += 1
            if getattr(battle, "force_switch", False):
                heuristics_stats["force_switch"] += 1
            if self._status_available_in_moves(getattr(battle, "available_moves", [])):
                heuristics_stats["status_available"] += 1
            if any(self._is_setup_move(m) for m in getattr(battle, "available_moves", [])):
                heuristics_stats["setup_available"] += 1
            if any(self._is_hazard_move(m) for m in getattr(battle, "available_moves", [])):
                heuristics_stats["hazard_available"] += 1
            if any(self._move_priority(m) > 0 for m in getattr(battle, "available_moves", [])):
                heuristics_stats["priority_available"] += 1
            if getattr(battle, "available_switches", []):
                heuristics_stats["switch_available"] += 1
            if getattr(battle, "can_tera", False):
                heuristics_stats["can_tera_true"] += 1
            if getattr(battle.active_pokemon, "tera_type", None):
                heuristics_stats["tera_type_present"] += 1
            if getattr(battle, "can_tera", False) and getattr(battle.active_pokemon, "tera_type", None):
                heuristics_stats["tera_available"] += 1
            my_remaining = sum(1 for p in battle.team.values() if not p.fainted)
            opp_remaining = 6 - sum(1 for p in battle.opponent_team.values() if p.fainted)
            if my_remaining <= 2 or opp_remaining <= 2:
                heuristics_stats["endgame_turns"] += 1

            if hasattr(order, "order"):
                move_or_switch = order.order
                if hasattr(move_or_switch, "species"):
                    heuristics_stats["switches"] += 1
                    name = normalize_name(str(move_or_switch.species))
                    heuristics_stats["switch_counts"][name] = (
                        heuristics_stats["switch_counts"].get(name, 0) + 1
                    )
                elif hasattr(move_or_switch, "id"):
                    move_id = move_or_switch.id
                    heuristics_stats["move_counts"][move_id] = (
                        heuristics_stats["move_counts"].get(move_id, 0) + 1
                    )
                    if move_id in self.ENTRY_HAZARDS or move_id in self.ANTI_HAZARDS_MOVES:
                        heuristics_stats["hazards"] += 1
                    if self._is_status_move(move_or_switch):
                        heuristics_stats["status"] += 1
                    if self._is_setup_move(move_or_switch):
                        heuristics_stats["setup"] += 1
                    if self._move_priority(move_or_switch) > 0:
                        heuristics_stats["priority"] += 1
                    if (not self._is_status_move(move_or_switch) and
                        not self._is_setup_move(move_or_switch) and
                        move_id not in self.ENTRY_HAZARDS and
                        move_id not in self.ANTI_HAZARDS_MOVES):
                        heuristics_stats["attacks"] += 1
                    else:
                        pass

                if hasattr(order, "terastallize") and order.terastallize:
                    heuristics_stats["tera"] += 1
            return order

    for i in range(n_battles):
        rulebot = TrackedRuleBot(**kwargs)
        heuristics = SimpleHeuristicsPlayer(**kwargs)

        prev_wins = rulebot.n_won_battles
        try:
            await asyncio.wait_for(
                rulebot.battle_against(heuristics, n_battles=1),
                timeout=timeout_s
            )
            if rulebot.n_won_battles > prev_wins:
                wins += 1
        except Exception as e:
            pass
        finally:
            battle = next(iter(rulebot.battles.values()), None)
            if battle is not None:
                heuristics_stats["turns"].append(battle.turn)
                remaining = sum(1 for p in battle.team.values() if not p.fainted)
                heuristics_stats["remaining_mons"].append(remaining)
            # Ensure proper cleanup of connections
            await _safe_stop(rulebot)
            await _safe_stop(heuristics)
            await asyncio.sleep(0.1)  # Brief delay to release resources

        if (i + 1) % progress_every == 0:
            print(f"   {i+1}/{n_battles}: {wins}/{i+1} wins ({wins/(i+1)*100:.1f}%)")

    results['heuristics'] = wins / n_battles
    low, high = wilson_interval(wins, n_battles)
    result_details["heuristics"] = {"wins": wins, "n": n_battles, "low": low, "high": high}
    print(f"   ✅ RuleBot vs SimpleHeuristics: {wins}/{n_battles} = {results['heuristics']:.1%} (95% CI {low:.1%}-{high:.1%})")

    # Summary
    print("\n" + "=" * 60)
    print("📈 SUMMARY")
    print("=" * 60)
    print(
        f"   vs Random:      {results['random']:.1%} "
        f"(95% CI {result_details['random']['low']:.1%}-{result_details['random']['high']:.1%})"
    )
    print(
        f"   vs MaxPower:    {results['maxpower']:.1%} "
        f"(95% CI {result_details['maxpower']['low']:.1%}-{result_details['maxpower']['high']:.1%})"
    )
    print(
        f"   vs Heuristics:  {results['heuristics']:.1%} "
        f"(95% CI {result_details['heuristics']['low']:.1%}-{result_details['heuristics']['high']:.1%})"
    )
    print("=" * 60)

    if heuristics_stats["turns"]:
        turns_sorted = sorted(heuristics_stats["turns"])
        remaining_sorted = sorted(heuristics_stats["remaining_mons"])
        avg_turns = sum(heuristics_stats["turns"]) / len(heuristics_stats["turns"])
        avg_remaining = sum(heuristics_stats["remaining_mons"]) / len(heuristics_stats["remaining_mons"])
        median_turns = turns_sorted[len(turns_sorted) // 2]
        median_remaining = remaining_sorted[len(remaining_sorted) // 2]
        p10_turns = turns_sorted[max(0, int(0.1 * len(turns_sorted)) - 1)]
        p90_turns = turns_sorted[min(len(turns_sorted) - 1, int(0.9 * len(turns_sorted)))]
        total_actions = max(heuristics_stats["actions"], 1)
        switch_available = max(heuristics_stats["switch_available"], 1)
        status_available = max(heuristics_stats["status_available"], 1)
        setup_available = max(heuristics_stats["setup_available"], 1)
        hazard_available = max(heuristics_stats["hazard_available"], 1)
        priority_available = max(heuristics_stats["priority_available"], 1)

        print("\n📌 Heuristics Action Summary (RuleBot)")
        print(f"   Avg turns:           {avg_turns:.1f} (median {median_turns}, p10 {p10_turns}, p90 {p90_turns})")
        print(f"   Avg remaining mons:  {avg_remaining:.2f} (median {median_remaining})")
        print(f"   Switch rate:         {heuristics_stats['switches'] / total_actions:.1%}")
        print(f"   Switch when avail:   {heuristics_stats['switches'] / switch_available:.1%}")
        print(f"   Attack rate:         {heuristics_stats['attacks'] / total_actions:.1%}")
        print(f"   Status rate:         {heuristics_stats['status'] / total_actions:.1%}")
        print(f"   Status when avail:   {heuristics_stats['status'] / status_available:.1%}")
        print(f"   Setup rate:          {heuristics_stats['setup'] / total_actions:.1%}")
        print(f"   Setup when avail:    {heuristics_stats['setup'] / setup_available:.1%}")
        print(f"   Hazard rate:         {heuristics_stats['hazards'] / total_actions:.1%}")
        print(f"   Hazard when avail:   {heuristics_stats['hazards'] / hazard_available:.1%}")
        print(f"   Priority rate:       {heuristics_stats['priority'] / total_actions:.1%}")
        print(f"   Priority when avail: {heuristics_stats['priority'] / priority_available:.1%}")
        print(f"   Tera rate:           {heuristics_stats['tera'] / total_actions:.1%}")
        print(f"   Status available:    {heuristics_stats['status_available'] / total_actions:.1%}")
        print(f"   Tera available:      {heuristics_stats['tera_available'] / total_actions:.1%}")
        print(f"   Tera used/avail:     {heuristics_stats['tera']}/{heuristics_stats['tera_available']}")
        print(f"   can_tera true:       {heuristics_stats['can_tera_true'] / total_actions:.1%}")
        print(f"   tera_type present:   {heuristics_stats['tera_type_present'] / total_actions:.1%}")
        print(f"   Force switch rate:   {heuristics_stats['force_switch'] / total_actions:.1%}")
        print(f"   Endgame share:       {heuristics_stats['endgame_turns'] / total_actions:.1%}")

        if hasattr(RuleBotPlayer, "STATUS_SKIP_COUNTS"):
            skipped = RuleBotPlayer.STATUS_SKIP_COUNTS.get("skipped", 0)
            available = RuleBotPlayer.STATUS_SKIP_COUNTS.get("available", 0)
            if available:
                print(f"   Status skips:        {skipped}/{available} ({skipped/available:.1%})")

        if heuristics_stats["move_counts"]:
            top_moves = sorted(
                heuristics_stats["move_counts"].items(), key=lambda x: x[1], reverse=True
            )[:10]
            print("\n📌 Top Moves (RuleBot vs Heuristics)")
            for move_id, count in top_moves:
                print(f"   {move_id}: {count}")
        if heuristics_stats["switch_counts"]:
            top_switches = sorted(
                heuristics_stats["switch_counts"].items(), key=lambda x: x[1], reverse=True
            )[:10]
            print("\n📌 Top Switches (RuleBot vs Heuristics)")
            for species, count in top_switches:
                print(f"   {species}: {count}")

    if results['heuristics'] >= 0.55:
        print("✅ RuleBot BEATS SimpleHeuristics! Good teacher candidate.")
        return True, results
    elif results['heuristics'] >= 0.45:
        print("⚠️ RuleBot is EQUAL to SimpleHeuristics. Marginal teacher.")
        return False, results
    else:
        print("❌ RuleBot LOSES to SimpleHeuristics. Bad teacher.")
        return False, results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--battles", type=int, default=100, help="Battles per opponent")
    parser.add_argument("--progress-every", type=int, default=25,
                        help="Print progress every N battles")
    parser.add_argument("--timeout", type=int, default=60,
                        help="Timeout per battle in seconds")
    args = parser.parse_args()

    asyncio.run(test_rulebot(args.battles, args.progress_every, args.timeout))
