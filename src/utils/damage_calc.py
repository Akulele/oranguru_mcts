#!/usr/bin/env python3
"""
Pokemon Damage Calculator

Implements the Generation V+ damage formula:
Damage = ((2*Level/5 + 2) * Power * A/D / 50 + 2) * Modifiers

Modifiers include:
- STAB (Same Type Attack Bonus)
- Type effectiveness
- Weather
- Critical hits
- Burn (physical moves)
- Abilities
- Items
"""

from typing import Optional, Tuple
from functools import lru_cache
import json
from pathlib import Path

# Type chart for effectiveness calculation
TYPE_CHART = {
    'normal': {'rock': 0.5, 'ghost': 0, 'steel': 0.5},
    'fire': {'fire': 0.5, 'water': 0.5, 'grass': 2, 'ice': 2, 'bug': 2, 'rock': 0.5, 'dragon': 0.5, 'steel': 2},
    'water': {'fire': 2, 'water': 0.5, 'grass': 0.5, 'ground': 2, 'rock': 2, 'dragon': 0.5},
    'electric': {'water': 2, 'electric': 0.5, 'grass': 0.5, 'ground': 0, 'flying': 2, 'dragon': 0.5},
    'grass': {'fire': 0.5, 'water': 2, 'grass': 0.5, 'poison': 0.5, 'ground': 2, 'flying': 0.5, 'bug': 0.5, 'rock': 2, 'dragon': 0.5, 'steel': 0.5},
    'ice': {'fire': 0.5, 'water': 0.5, 'grass': 2, 'ice': 0.5, 'ground': 2, 'flying': 2, 'dragon': 2, 'steel': 0.5},
    'fighting': {'normal': 2, 'ice': 2, 'poison': 0.5, 'flying': 0.5, 'psychic': 0.5, 'bug': 0.5, 'rock': 2, 'ghost': 0, 'dark': 2, 'steel': 2, 'fairy': 0.5},
    'poison': {'grass': 2, 'poison': 0.5, 'ground': 0.5, 'rock': 0.5, 'ghost': 0.5, 'steel': 0, 'fairy': 2},
    'ground': {'fire': 2, 'electric': 2, 'grass': 0.5, 'poison': 2, 'flying': 0, 'bug': 0.5, 'rock': 2, 'steel': 2},
    'flying': {'electric': 0.5, 'grass': 2, 'fighting': 2, 'bug': 2, 'rock': 0.5, 'steel': 0.5},
    'psychic': {'fighting': 2, 'poison': 2, 'psychic': 0.5, 'dark': 0, 'steel': 0.5},
    'bug': {'fire': 0.5, 'grass': 2, 'fighting': 0.5, 'poison': 0.5, 'flying': 0.5, 'psychic': 2, 'ghost': 0.5, 'dark': 2, 'steel': 0.5, 'fairy': 0.5},
    'rock': {'fire': 2, 'ice': 2, 'fighting': 0.5, 'ground': 0.5, 'flying': 2, 'bug': 2, 'steel': 0.5},
    'ghost': {'normal': 0, 'psychic': 2, 'ghost': 2, 'dark': 0.5},
    'dragon': {'dragon': 2, 'steel': 0.5, 'fairy': 0},
    'dark': {'fighting': 0.5, 'psychic': 2, 'ghost': 2, 'dark': 0.5, 'fairy': 0.5},
    'steel': {'fire': 0.5, 'water': 0.5, 'electric': 0.5, 'ice': 2, 'rock': 2, 'steel': 0.5, 'fairy': 2},
    'fairy': {'fire': 0.5, 'fighting': 2, 'poison': 0.5, 'dragon': 2, 'dark': 2, 'steel': 0.5},
}

# Important items and their effects
ITEM_EFFECTS = {
    # Damage boosters
    'lifeorb': {'damage_mult': 1.3, 'recoil': 0.1},
    'choiceband': {'atk_mult': 1.5, 'locked': True},
    'choicespecs': {'spa_mult': 1.5, 'locked': True},
    'choicescarf': {'spe_mult': 1.5, 'locked': True},
    'expertbelt': {'super_eff_mult': 1.2},

    # Type boosters
    'charcoal': {'type_boost': 'fire', 'mult': 1.2},
    'mysticwater': {'type_boost': 'water', 'mult': 1.2},
    'miracleseed': {'type_boost': 'grass', 'mult': 1.2},
    'magnet': {'type_boost': 'electric', 'mult': 1.2},
    'nevermeltice': {'type_boost': 'ice', 'mult': 1.2},
    'blackbelt': {'type_boost': 'fighting', 'mult': 1.2},
    'poisonbarb': {'type_boost': 'poison', 'mult': 1.2},
    'softsand': {'type_boost': 'ground', 'mult': 1.2},
    'sharpbeak': {'type_boost': 'flying', 'mult': 1.2},
    'twistedspoon': {'type_boost': 'psychic', 'mult': 1.2},
    'silverpowder': {'type_boost': 'bug', 'mult': 1.2},
    'hardstone': {'type_boost': 'rock', 'mult': 1.2},
    'spelltag': {'type_boost': 'ghost', 'mult': 1.2},
    'dragonfang': {'type_boost': 'dragon', 'mult': 1.2},
    'blackglasses': {'type_boost': 'dark', 'mult': 1.2},
    'metalcoat': {'type_boost': 'steel', 'mult': 1.2},

    # Defensive items
    'assaultvest': {'spd_mult': 1.5, 'no_status': True},
    'eviolite': {'def_mult': 1.5, 'spd_mult': 1.5, 'nfe_only': True},
    'rockyhelmet': {'contact_damage': 1/6},

    # Berry/recovery
    'sitrusberry': {'heal_at': 0.5, 'heal_amount': 0.25},
    'leftovers': {'heal_per_turn': 1/16},
    'blacksludge': {'heal_per_turn': 1/16, 'poison_only': True},

    # Setup support
    'whiteherb': {'clears_stat_drops': True, 'one_time': True},
    'weaknesspolicy': {'triggers_on_super_eff': True, 'boost': 2},
    'throatspray': {'triggers_on_sound': True, 'spa_boost': 1},

    # Priority/speed
    'quickclaw': {'priority_chance': 0.2},
    'custapberry': {'priority_at_low_hp': True},

    # Survival
    'focussash': {'survives_ohko_at_full': True, 'one_time': True},
    'airballoon': {'immune_ground': True, 'pops_on_hit': True},

    # Hazard boots
    'heavydutyboots': {'immune_hazards': True},
}

# Ability damage modifiers
ABILITY_DAMAGE_MODS = {
    # Attack doublers
    'hugepower': {'atk_mult': 2.0},
    'purepower': {'atk_mult': 2.0},

    # STAB boosters
    'adaptability': {'stab_mult': 2.0},  # Changes STAB from 1.5 to 2.0

    # Conditional boosters
    'technician': {'weak_move_mult': 1.5, 'threshold': 60},
    'toughclaws': {'contact_mult': 1.3},
    'sheerforce': {'secondary_mult': 1.3, 'no_secondary': True},
    'strongjaw': {'bite_mult': 1.5},
    'ironfist': {'punch_mult': 1.2},
    'megalaunch': {'pulse_mult': 1.5},

    # Type immunities that boost
    'flashfire': {'fire_immune': True, 'fire_boost': 1.5},
    'waterabsorb': {'water_immune': True},
    'voltabsorb': {'electric_immune': True},
    'lightningrod': {'electric_immune': True, 'spa_boost': 1},
    'stormdrain': {'water_immune': True, 'spa_boost': 1},
    'sapsipper': {'grass_immune': True, 'atk_boost': 1},
    'motordrive': {'electric_immune': True, 'spe_boost': 1},

    # Defensive
    'multiscale': {'damage_reduction': 0.5, 'at_full_hp': True},
    'shadowshield': {'damage_reduction': 0.5, 'at_full_hp': True},
    'sturdy': {'survives_ohko_at_full': True},
    'disguise': {'blocks_first_hit': True},
    'iceface': {'blocks_first_physical': True},
    'fluffy': {'contact_reduction': 0.5, 'fire_weakness': 2.0},
    'furcoat': {'physical_reduction': 0.5},
    'icescales': {'special_reduction': 0.5},

    # Weather
    'chlorophyll': {'sun_speed': 2.0},
    'swiftswim': {'rain_speed': 2.0},
    'sandrush': {'sand_speed': 2.0},
    'slushrush': {'snow_speed': 2.0},
    'solarpower': {'sun_spa': 1.5, 'sun_hp_loss': 1/8},
}


def normalize_name(name: str) -> str:
    """Normalize name for lookup."""
    if not name:
        return ""
    return name.lower().replace(' ', '').replace('-', '').replace("'", "").replace('.', '')


def get_type_effectiveness(atk_type: str, def_types: list) -> float:
    """Calculate type effectiveness multiplier."""
    atk_type = normalize_name(atk_type)
    mult = 1.0

    for def_type in def_types:
        def_type = normalize_name(def_type)
        if atk_type in TYPE_CHART:
            mult *= TYPE_CHART[atk_type].get(def_type, 1.0)

    return mult


def calc_stat(base: int, level: int, iv: int = 31, ev: int = 85, nature: float = 1.0, is_hp: bool = False) -> int:
    """Calculate actual stat value."""
    if is_hp:
        return int((2 * base + iv + ev // 4) * level / 100 + level + 10)
    else:
        return int(((2 * base + iv + ev // 4) * level / 100 + 5) * nature)


def apply_boost(stat: int, stages: int) -> int:
    """Apply stat stage modifier."""
    if stages >= 0:
        return int(stat * (2 + stages) / 2)
    else:
        return int(stat * 2 / (2 - stages))


def calc_damage(
    move_power: int,
    move_type: str,
    move_category: str,  # 'physical' or 'special'
    attacker_level: int,
    attacker_stats: Tuple[int, int, int, int, int, int],  # HP, Atk, Def, SpA, SpD, Spe
    attacker_types: list,
    attacker_boosts: dict,
    attacker_status: str,
    attacker_ability: str,
    attacker_item: str,
    attacker_hp_fraction: float,
    defender_stats: Tuple[int, int, int, int, int, int],
    defender_types: list,
    defender_boosts: dict,
    defender_ability: str,
    defender_item: str,
    defender_hp_fraction: float,
    weather: str = None,
    terrain: str = None,
    is_critical: bool = False,
) -> Tuple[int, int, float]:
    """
    Calculate damage range using Pokemon damage formula.

    Returns: (min_damage, max_damage, ko_chance)
    """
    if move_power == 0:
        return (0, 0, 0.0)

    # Normalize names
    move_type = normalize_name(move_type)
    attacker_ability = normalize_name(attacker_ability)
    attacker_item = normalize_name(attacker_item)
    defender_ability = normalize_name(defender_ability)
    defender_item = normalize_name(defender_item)
    attacker_types = [normalize_name(t) for t in attacker_types]
    defender_types = [normalize_name(t) for t in defender_types]

    # Get attack and defense stats
    if move_category.lower() == 'physical':
        atk_base = attacker_stats[1]  # Attack
        def_base = defender_stats[2]  # Defense
        atk_boost = attacker_boosts.get('atk', 0)
        def_boost = defender_boosts.get('def', 0)
    else:  # Special
        atk_base = attacker_stats[3]  # SpA
        def_base = defender_stats[4]  # SpD
        atk_boost = attacker_boosts.get('spa', 0)
        def_boost = defender_boosts.get('spd', 0)

    # Apply boosts (crits ignore negative atk boosts and positive def boosts)
    if is_critical:
        atk_boost = max(atk_boost, 0)
        def_boost = min(def_boost, 0)

    atk_stat = apply_boost(calc_stat(atk_base, attacker_level), atk_boost)
    def_stat = apply_boost(calc_stat(def_base, attacker_level), def_boost)

    # === MODIFIERS ===
    modifiers = 1.0

    # STAB
    stab = 1.5 if move_type in attacker_types else 1.0
    if attacker_ability == 'adaptability' and stab > 1.0:
        stab = 2.0
    modifiers *= stab

    # Type effectiveness
    type_eff = get_type_effectiveness(move_type, defender_types)

    # Check for immunities from abilities
    if defender_ability in ['levitate'] and move_type == 'ground':
        type_eff = 0
    if defender_ability in ['flashfire'] and move_type == 'fire':
        type_eff = 0
    if defender_ability in ['waterabsorb', 'stormdrain', 'dryskin'] and move_type == 'water':
        type_eff = 0
    if defender_ability in ['voltabsorb', 'lightningrod', 'motordrive'] and move_type == 'electric':
        type_eff = 0
    if defender_ability in ['sapsipper'] and move_type == 'grass':
        type_eff = 0

    modifiers *= type_eff

    if type_eff == 0:
        return (0, 0, 0.0)

    # Weather
    if weather:
        weather = normalize_name(weather)
        if 'sun' in weather:
            if move_type == 'fire':
                modifiers *= 1.5
            elif move_type == 'water':
                modifiers *= 0.5
        elif 'rain' in weather:
            if move_type == 'water':
                modifiers *= 1.5
            elif move_type == 'fire':
                modifiers *= 0.5

    # Burn (physical moves)
    if move_category.lower() == 'physical' and attacker_status and 'brn' in attacker_status.lower():
        if attacker_ability != 'guts':
            modifiers *= 0.5

    # Ability modifiers
    if attacker_ability == 'hugepower' or attacker_ability == 'purepower':
        if move_category.lower() == 'physical':
            atk_stat *= 2

    if attacker_ability == 'technician' and move_power <= 60:
        modifiers *= 1.5

    # Item modifiers
    if attacker_item == 'lifeorb':
        modifiers *= 1.3
    elif attacker_item == 'choiceband' and move_category.lower() == 'physical':
        atk_stat = int(atk_stat * 1.5)
    elif attacker_item == 'choicespecs' and move_category.lower() == 'special':
        atk_stat = int(atk_stat * 1.5)
    elif attacker_item == 'expertbelt' and type_eff > 1.0:
        modifiers *= 1.2

    # Type-boosting items
    item_data = ITEM_EFFECTS.get(attacker_item, {})
    if item_data.get('type_boost') == move_type:
        modifiers *= item_data.get('mult', 1.2)

    # Defender items
    if defender_item == 'assaultvest' and move_category.lower() == 'special':
        def_stat = int(def_stat * 1.5)

    # Defensive abilities
    if defender_ability in ['multiscale', 'shadowshield'] and defender_hp_fraction >= 0.99:
        modifiers *= 0.5
    if defender_ability == 'fluffy':
        modifiers *= 0.5  # Contact moves (simplified)
    if defender_ability == 'furcoat' and move_category.lower() == 'physical':
        def_stat *= 2
    if defender_ability == 'icescales' and move_category.lower() == 'special':
        modifiers *= 0.5

    # Critical hit
    if is_critical:
        modifiers *= 1.5

    # === DAMAGE FORMULA ===
    # Damage = ((2*Level/5 + 2) * Power * A/D / 50 + 2) * Modifiers
    base_damage = ((2 * attacker_level / 5 + 2) * move_power * atk_stat / def_stat) / 50 + 2

    # Random factor (0.85 - 1.00)
    min_damage = int(base_damage * modifiers * 0.85)
    max_damage = int(base_damage * modifiers * 1.00)

    # Calculate KO chance
    defender_hp = calc_stat(defender_stats[0], attacker_level, is_hp=True)
    current_hp = int(defender_hp * defender_hp_fraction)

    if max_damage == 0:
        ko_chance = 0.0
    elif min_damage >= current_hp:
        ko_chance = 1.0
    elif max_damage < current_hp:
        ko_chance = 0.0
    else:
        # Linear interpolation for KO chance
        damage_range = max_damage - min_damage + 1
        ko_rolls = max(0, max_damage - current_hp + 1)
        ko_chance = ko_rolls / damage_range

    return (min_damage, max_damage, ko_chance)


def is_safe_to_setup(
    my_pokemon,
    opp_pokemon,
    my_ability: str,
    my_item: str,
    my_hp_fraction: float,
    estimated_incoming_damage_pct: float,
) -> Tuple[bool, str]:
    """
    Determine if it's safe to use a setup move.

    Returns: (is_safe, reason)

    Safe conditions:
    - Sturdy + White Herb (Shell Smash)
    - HP > 70% and incoming damage < 50%
    - Opponent is passive (status move predicted)
    - Substitute up
    - Disguise intact
    """
    my_ability = normalize_name(my_ability)
    my_item = normalize_name(my_item)

    # Sturdy + White Herb is the classic Shell Smash setup
    if my_ability == 'sturdy' and my_hp_fraction >= 0.99:
        return (True, "sturdy_full_hp")

    # Focus Sash at full HP
    if my_item == 'focussash' and my_hp_fraction >= 0.99:
        return (True, "sash_full_hp")

    # Disguise intact
    if my_ability == 'disguise':
        # Would need to track if disguise is busted
        return (True, "disguise_intact")

    # General HP-based safety
    if my_hp_fraction >= 0.7 and estimated_incoming_damage_pct < 0.5:
        return (True, "safe_hp_buffer")

    # Risky if low HP or facing big damage
    if my_hp_fraction < 0.4:
        return (False, "low_hp")

    if estimated_incoming_damage_pct > 0.6:
        return (False, "high_incoming_damage")

    # Default: moderately safe
    if my_hp_fraction >= 0.5:
        return (True, "moderate_hp")

    return (False, "unsafe_general")


def get_item_effect(item_name: str) -> dict:
    """Get item effect data."""
    item_name = normalize_name(item_name)
    return ITEM_EFFECTS.get(item_name, {})


def get_ability_damage_mod(ability_name: str) -> dict:
    """Get ability damage modifier data."""
    ability_name = normalize_name(ability_name)
    return ABILITY_DAMAGE_MODS.get(ability_name, {})
