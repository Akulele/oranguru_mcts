#!/usr/bin/env python3
"""
🦧 ORANGURU RL - Feature Diagnostics

Debug why model isn't learning by inspecting features during battle.
"""

import asyncio
import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from poke_env.player import RandomPlayer
from poke_env import LocalhostServerConfiguration
from poke_env.battle import AbstractBattle

from training.config import RLConfig
from src.utils.type_chart import get_type_effectiveness, parse_type


class DiagnosticPlayer(RandomPlayer):
    """Player that logs feature values for debugging."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.feature_log = []
        self.decision_log = []
    
    def choose_move(self, battle: AbstractBattle):
        features = self._analyze_features(battle)
        self.feature_log.append(features)
        return super().choose_move(battle)
    
    def _analyze_features(self, battle):
        """Analyze what features look like."""
        my_poke = battle.active_pokemon
        opp_poke = battle.opponent_active_pokemon
        
        info = {
            'turn': battle.turn,
            'my_pokemon': str(my_poke) if my_poke else None,
            'opp_pokemon': str(opp_poke) if opp_poke else None,
            'my_hp': my_poke.current_hp_fraction if my_poke else 0,
            'opp_hp': opp_poke.current_hp_fraction if opp_poke else 0,
            'available_moves': len(battle.available_moves),
            'available_switches': len(battle.available_switches),
            'can_tera': battle.can_tera,
            'force_switch': battle.force_switch,
        }
        
        # Analyze type effectiveness for each move
        if my_poke and opp_poke:
            opp_types = [parse_type(t) for t in (opp_poke.types or []) if t]
            my_types = [parse_type(t) for t in (my_poke.types or []) if t]
            
            info['opp_types'] = opp_types
            info['my_types'] = my_types
            
            move_info = []
            for move in battle.available_moves[:4]:
                move_type = parse_type(move.type) if move.type else 'Normal'
                eff = get_type_effectiveness(move_type, opp_types)
                move_info.append({
                    'name': move.id,
                    'type': move_type,
                    'power': move.base_power,
                    'effectiveness': eff,
                    'stab': move_type in my_types,
                })
            info['moves'] = move_info
        
        return info


async def run_diagnostic():
    """Run diagnostic battles and analyze features."""
    
    print("\n🔍 FEATURE DIAGNOSTICS")
    print("=" * 60)
    
    kwargs = {
        'battle_format': 'gen9randombattle',
        'max_concurrent_battles': 1,
        'server_configuration': LocalhostServerConfiguration,
    }
    
    diag = DiagnosticPlayer(**kwargs)
    opp = RandomPlayer(**kwargs)
    
    print("\nRunning 3 diagnostic battles...")
    await diag.battle_against(opp, n_battles=3)
    
    print(f"\nCollected {len(diag.feature_log)} decision points")
    
    # Analyze type effectiveness distribution
    all_eff = []
    for log in diag.feature_log:
        if 'moves' in log:
            for m in log['moves']:
                all_eff.append(m['effectiveness'])
    
    if all_eff:
        print(f"\n📊 Type Effectiveness Distribution:")
        print(f"   Min: {min(all_eff)}")
        print(f"   Max: {max(all_eff)}")
        print(f"   Mean: {np.mean(all_eff):.3f}")
        print(f"   Values: {sorted(set(all_eff))}")
        
        # Count super effective opportunities
        super_eff = sum(1 for e in all_eff if e >= 2)
        not_very = sum(1 for e in all_eff if e < 1 and e > 0)
        immune = sum(1 for e in all_eff if e == 0)
        
        print(f"\n   Super effective (2x+): {super_eff}/{len(all_eff)} = {super_eff/len(all_eff):.1%}")
        print(f"   Not very effective: {not_very}/{len(all_eff)} = {not_very/len(all_eff):.1%}")
        print(f"   Immune: {immune}/{len(all_eff)} = {immune/len(all_eff):.1%}")
    
    # Sample some decisions
    print(f"\n📝 Sample Decisions:")
    for i, log in enumerate(diag.feature_log[:5]):
        print(f"\n   Turn {log['turn']}:")
        print(f"   {log['my_pokemon']} vs {log['opp_pokemon']}")
        print(f"   HP: {log['my_hp']:.0%} vs {log['opp_hp']:.0%}")
        if 'moves' in log:
            for m in log['moves']:
                eff_str = f"x{m['effectiveness']}" if m['effectiveness'] != 1 else ""
                stab_str = " STAB" if m['stab'] else ""
                print(f"      {m['name']}: {m['type']} pow={m['power']} {eff_str}{stab_str}")
    
    print("\n" + "=" * 60)
    
    # Check if effectiveness is always 1.0 (would indicate bug)
    unique_eff = set(all_eff) if all_eff else set()
    if unique_eff == {1.0}:
        print("⚠️ WARNING: All effectiveness values are 1.0!")
        print("   This suggests type chart lookup is broken.")
    elif len(unique_eff) < 3:
        print("⚠️ WARNING: Very few unique effectiveness values.")
        print("   Type chart may not be working correctly.")
    else:
        print("✅ Type effectiveness appears to be working.")


if __name__ == "__main__":
    print("\n⚠️ Ensure server is running: docker start pokemon-showdown\n")
    asyncio.run(run_diagnostic())