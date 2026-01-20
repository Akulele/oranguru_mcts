"""
🦧 ORANGURU RL - Type Chart

Uses typechart.json for accuracy.
Properly handles poke-env type objects.
"""

import json
from pathlib import Path

# Load type chart from JSON
_TYPE_CHART = None

def _load_type_chart():
    """Load type chart from JSON file."""
    global _TYPE_CHART
    if _TYPE_CHART is not None:
        return _TYPE_CHART
    
    # Try multiple locations
    possible_paths = [
        Path(__file__).parent.parent.parent / "data" / "typechart.json",
        Path("data/typechart.json"),
        Path("typechart.json"),
    ]
    
    for path in possible_paths:
        if path.exists():
            with open(path) as f:
                _TYPE_CHART = json.load(f)
            return _TYPE_CHART
    
    raise FileNotFoundError(f"typechart.json not found. Tried: {possible_paths}")


# Damage taken values in the JSON:
# 0 = normal (1x)
# 1 = super effective (2x) - this type is WEAK to attacking type
# 2 = not very effective (0.5x) - this type RESISTS attacking type  
# 3 = immune (0x)
DAMAGE_MULTIPLIERS = {
    0: 1.0,
    1: 2.0,   # Weak
    2: 0.5,   # Resist
    3: 0.0,   # Immune
}


STATUS_TO_IDX = {
    '': 0, 'brn': 1, 'par': 2, 'slp': 3, 'psn': 4, 'tox': 5, 'frz': 6
}


def parse_type(type_obj) -> str:
    """
    Extract type name from poke-env type object.
    
    Handles:
    - PokemonType.FIRE -> "Fire"
    - "FIRE" -> "Fire"  
    - "Fire" -> "Fire"
    - "fire" -> "Fire"
    - "FIRE (pokemon type) object" -> "Fire"
    """
    if type_obj is None:
        return None
    
    # Convert to string and extract the type name
    type_str = str(type_obj)
    
    # Handle "FIRE (pokemon type) object" format
    if "(" in type_str:
        type_str = type_str.split("(")[0].strip()
    
    # Handle "PokemonType.FIRE" format
    if "." in type_str:
        type_str = type_str.split(".")[-1]
    
    # Clean up and capitalize properly
    type_str = type_str.strip().upper()
    
    # Map to proper capitalization
    type_map = {
        'NORMAL': 'Normal', 'FIRE': 'Fire', 'WATER': 'Water',
        'ELECTRIC': 'Electric', 'GRASS': 'Grass', 'ICE': 'Ice',
        'FIGHTING': 'Fighting', 'POISON': 'Poison', 'GROUND': 'Ground',
        'FLYING': 'Flying', 'PSYCHIC': 'Psychic', 'BUG': 'Bug',
        'ROCK': 'Rock', 'GHOST': 'Ghost', 'DRAGON': 'Dragon',
        'DARK': 'Dark', 'STEEL': 'Steel', 'FAIRY': 'Fairy',
        'STELLAR': 'Stellar',
    }
    
    return type_map.get(type_str, type_str.capitalize())


def get_type_effectiveness(atk_type, def_types) -> float:
    """
    Calculate type effectiveness multiplier.
    
    Args:
        atk_type: Attacking type (string or poke-env type object)
        def_types: List of defending types (strings or poke-env type objects)
        
    Returns:
        Multiplier (0, 0.25, 0.5, 1, 2, or 4)
    """
    type_chart = _load_type_chart()
    
    # Parse the attacking type
    atk_type_name = parse_type(atk_type)
    if not atk_type_name:
        return 1.0
    
    multiplier = 1.0
    
    for def_type in def_types:
        def_type_name = parse_type(def_type)
        if not def_type_name:
            continue
        
        # Look up in type chart
        # The chart is keyed by defending type (lowercase)
        def_key = def_type_name.lower()
        
        if def_key in type_chart:
            damage_taken = type_chart[def_key].get("damageTaken", {})
            
            # Look up attacking type in the damage taken table
            if atk_type_name in damage_taken:
                damage_value = damage_taken[atk_type_name]
                multiplier *= DAMAGE_MULTIPLIERS.get(damage_value, 1.0)
    
    return multiplier


def get_all_type_matchups(atk_type) -> dict:
    """Get effectiveness against all types for a given attacking type."""
    type_chart = _load_type_chart()
    atk_type_name = parse_type(atk_type)
    
    matchups = {}
    for def_type in type_chart.keys():
        eff = get_type_effectiveness(atk_type_name, [def_type.capitalize()])
        matchups[def_type.capitalize()] = eff
    
    return matchups


# Quick test
if __name__ == "__main__":
    print("Testing type effectiveness...")
    
    # Fire vs Grass should be 2x
    eff = get_type_effectiveness("Fire", ["Grass"])
    print(f"Fire vs Grass: {eff}x (expected: 2x)")
    
    # Fire vs Water should be 0.5x
    eff = get_type_effectiveness("Fire", ["Water"])
    print(f"Fire vs Water: {eff}x (expected: 0.5x)")
    
    # Electric vs Ground should be 0x
    eff = get_type_effectiveness("Electric", ["Ground"])
    print(f"Electric vs Ground: {eff}x (expected: 0x)")
    
    # Fire vs Grass/Steel should be 4x
    eff = get_type_effectiveness("Fire", ["Grass", "Steel"])
    print(f"Fire vs Grass/Steel: {eff}x (expected: 4x)")
    
    # Test with poke-env style type strings
    eff = get_type_effectiveness("FIRE (pokemon type) object", ["GRASS (pokemon type) object"])
    print(f"FIRE (pokemon type) vs GRASS (pokemon type): {eff}x (expected: 2x)")
    
    # Dragon vs Fairy (immune)
    eff = get_type_effectiveness("Dragon", ["Fairy"])
    print(f"Dragon vs Fairy: {eff}x (expected: 0x)")
    
    print("\n✅ Type chart loaded successfully!")