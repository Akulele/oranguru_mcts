#!/usr/bin/env python3
"""
🔍 ENHANCED FEATURE DIAGNOSTICS

Test the 256-dim feature vector before training.
"""

import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from poke_env.player import RandomPlayer
from poke_env import LocalhostServerConfiguration

from src.utils.features import EnhancedFeatureBuilder, get_speed_tier, get_base_stats


class DiagnosticPlayer(RandomPlayer):
    """Player that logs feature diagnostics."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.feature_builder = EnhancedFeatureBuilder()
        self.decision_log = []
    
    def choose_move(self, battle):
        """Log features then pick random move."""
        features = self.feature_builder.build(battle)
        
        my_poke = battle.active_pokemon
        opp_poke = battle.opponent_active_pokemon
        
        # Log decision point
        info = {
            'turn': battle.turn,
            'my_poke': str(my_poke.species) if my_poke else 'None',
            'opp_poke': str(opp_poke.species) if opp_poke else 'None',
            'my_hp': my_poke.current_hp_fraction if my_poke else 0,
            'opp_hp': opp_poke.current_hp_fraction if opp_poke else 0,
            'my_speed': get_speed_tier(my_poke),
            'opp_speed': get_speed_tier(opp_poke),
            'n_features': len(features),
            'feature_sample': features[:16],  # First 16 features
            'moves': [],
        }
        
        # Log moves with details
        for i, move in enumerate(battle.available_moves[:4]):
            move_type = str(move.type).split('.')[-1].split('(')[0].strip() if move.type else 'Normal'
            opp_types = [str(t).split('.')[-1].split('(')[0].strip().lower() 
                        for t in (opp_poke.types if opp_poke else []) if t]
            
            from src.utils.features import get_type_effectiveness, get_move_priority, get_move_effect
            eff = get_type_effectiveness(move_type.lower(), opp_types) if move.base_power else 0
            priority = get_move_priority(move)
            effect_type, effect_val = get_move_effect(move)
            
            info['moves'].append({
                'name': str(move.id),
                'type': move_type,
                'power': move.base_power or 0,
                'eff': eff,
                'priority': priority,
                'effect': effect_type,
            })
        
        self.decision_log.append(info)
        
        return super().choose_move(battle)


async def run_diagnostics(n_battles: int = 3):
    """Run diagnostic battles."""
    print("=" * 60)
    print("🔍 ENHANCED FEATURE DIAGNOSTICS (256-dim)")
    print("=" * 60)
    
    kwargs = {
        'battle_format': 'gen9randombattle',
        'server_configuration': LocalhostServerConfiguration,
    }
    
    player = DiagnosticPlayer(**kwargs)
    opponent = RandomPlayer(**kwargs)
    
    print(f"\nRunning {n_battles} diagnostic battles...")
    await player.battle_against(opponent, n_battles=n_battles)
    
    print(f"Collected {len(player.decision_log)} decision points\n")
    
    # === Feature Statistics ===
    all_features = [d['feature_sample'] for d in player.decision_log]
    if all_features:
        import numpy as np
        arr = np.array(all_features)
        print("📊 Feature Statistics (first 16 features):")
        print(f"   Shape: {arr.shape}")
        print(f"   Range: [{arr.min():.3f}, {arr.max():.3f}]")
        print(f"   Mean: {arr.mean():.3f}")
        print(f"   Non-zero: {(arr != 0).sum() / arr.size * 100:.1f}%")
    
    # === Speed Tier Analysis ===
    speeds = [(d['my_poke'], d['my_speed'], d['opp_poke'], d['opp_speed']) 
              for d in player.decision_log if d['my_speed'] > 0]
    
    if speeds:
        print("\n📊 Speed Tier Analysis:")
        my_speeds = [s[1] for s in speeds]
        opp_speeds = [s[3] for s in speeds]
        faster_count = sum(1 for s in speeds if s[1] > s[3])
        print(f"   My avg speed: {sum(my_speeds)/len(my_speeds):.3f}")
        print(f"   Opp avg speed: {sum(opp_speeds)/len(opp_speeds):.3f}")
        print(f"   Times faster: {faster_count}/{len(speeds)} ({faster_count/len(speeds)*100:.1f}%)")
    
    # === Type Effectiveness Distribution ===
    all_eff = []
    for d in player.decision_log:
        for move in d['moves']:
            if move['power'] > 0:
                all_eff.append(move['eff'])
    
    if all_eff:
        print("\n📊 Type Effectiveness Distribution:")
        print(f"   Min: {min(all_eff)}")
        print(f"   Max: {max(all_eff)}")
        print(f"   Mean: {sum(all_eff)/len(all_eff):.3f}")
        print(f"   Unique values: {sorted(set(all_eff))}")
        super_eff = sum(1 for e in all_eff if e >= 2)
        resist = sum(1 for e in all_eff if e < 1 and e > 0)
        immune = sum(1 for e in all_eff if e == 0)
        print(f"   Super effective (2x+): {super_eff}/{len(all_eff)} = {super_eff/len(all_eff)*100:.1f}%")
        print(f"   Resisted: {resist}/{len(all_eff)} = {resist/len(all_eff)*100:.1f}%")
        print(f"   Immune: {immune}/{len(all_eff)} = {immune/len(all_eff)*100:.1f}%")
    
    # === Priority Move Analysis ===
    priority_moves = []
    for d in player.decision_log:
        for move in d['moves']:
            if move['priority'] != 0:
                priority_moves.append((move['name'], move['priority']))
    
    if priority_moves:
        print(f"\n📊 Priority Moves Found: {len(priority_moves)}")
        unique_priority = set(priority_moves)
        for name, pri in list(unique_priority)[:10]:
            print(f"   {name}: priority {pri:+d}")
    
    # === Status Move Analysis ===
    status_moves = []
    for d in player.decision_log:
        for move in d['moves']:
            if move['effect'] != 'none':
                status_moves.append((move['name'], move['effect']))
    
    if status_moves:
        print(f"\n📊 Status Move Effects Found: {len(status_moves)}")
        # Count by effect type
        effect_counts = {}
        for name, effect in status_moves:
            effect_counts[effect] = effect_counts.get(effect, 0) + 1
        for effect, count in sorted(effect_counts.items(), key=lambda x: -x[1])[:10]:
            print(f"   {effect}: {count}")
    
    # === Sample Decisions ===
    print("\n📝 Sample Decisions:")
    for i, d in enumerate(player.decision_log[:5]):
        print(f"   Turn {d['turn']}:")
        print(f"   {d['my_poke']} (Spd:{d['my_speed']:.2f}) vs {d['opp_poke']} (Spd:{d['opp_speed']:.2f})")
        print(f"   HP: {d['my_hp']*100:.0f}% vs {d['opp_hp']*100:.0f}%")
        for move in d['moves']:
            flags = []
            if move['eff'] >= 2:
                flags.append(f"x{move['eff']}")
            elif move['eff'] == 0:
                flags.append("IMMUNE")
            elif move['eff'] < 1:
                flags.append(f"x{move['eff']}")
            if move['priority'] != 0:
                flags.append(f"pri:{move['priority']:+d}")
            if move['effect'] != 'none':
                flags.append(move['effect'])
            flag_str = " ".join(flags) if flags else ""
            print(f"      {move['name']}: {move['type']} pow={move['power']} {flag_str}")
        print()
    
    # === Validation ===
    print("=" * 60)
    issues = []
    
    if all_eff and max(all_eff) <= 1.0:
        issues.append("⚠️ No super-effective moves found - type chart may be broken")
    
    if len(priority_moves) == 0:
        issues.append("⚠️ No priority moves detected - priority data may be missing")
    
    if len(status_moves) == 0:
        issues.append("⚠️ No status move effects detected - move effect data may be missing")
    
    if speeds and sum(1 for s in speeds if s[1] == s[3]) == len(speeds):
        issues.append("⚠️ All speeds identical - speed calculation may be broken")
    
    if issues:
        for issue in issues:
            print(issue)
    else:
        print("✅ Enhanced features appear to be working correctly!")
    
    print("=" * 60)


if __name__ == "__main__":
    print("\n⚠️ Ensure server is running: docker start pokemon-showdown\n")
    asyncio.run(run_diagnostics(n_battles=3))