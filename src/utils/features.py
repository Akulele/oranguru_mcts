#!/usr/bin/env python3
"""
🦧 ENHANCED FEATURE BUILDER

Comprehensive feature vector with:
- Speed comparison (who goes first)
- Move priority and effects
- Ability interactions
- Status move value
- Better switch evaluation

Target: 256 features for richer representation
"""

import json
from pathlib import Path
from functools import lru_cache
from typing import List, Dict, Optional, Tuple

from src.utils.moveset_priors import load_moveset_priors

# Load data files
DATA_DIR = Path(__file__).parent.parent.parent / "data"

@lru_cache(maxsize=1)
def load_moves() -> Dict:
    """Load moves.json data."""
    try:
        with open(DATA_DIR / "moves.json") as f:
            return json.load(f)
    except:
        return {}

@lru_cache(maxsize=1)
def load_abilities() -> Dict:
    """Load abilities.json data."""
    try:
        with open(DATA_DIR / "abilities.json") as f:
            return json.load(f)
    except:
        return {}


@lru_cache(maxsize=1)
def load_items() -> Dict:
    """Load items.json data."""
    try:
        with open(DATA_DIR / "items.json") as f:
            return json.load(f)
    except:
        return {}

@lru_cache(maxsize=1)
def load_typechart() -> Dict:
    """Load typechart.json data."""
    try:
        with open(DATA_DIR / "typechart.json") as f:
            return json.load(f)
    except:
        return {}


# Pokemon base stats (common Pokemon in random battles)
# Format: species -> (hp, atk, def, spa, spd, spe)
POKEMON_STATS = {
    # Starters and common
    'pikachu': (35, 55, 40, 50, 50, 90),
    'charizard': (78, 84, 78, 109, 85, 100),
    'blastoise': (79, 83, 100, 85, 105, 78),
    'venusaur': (80, 82, 83, 100, 100, 80),
    
    # Gen 9 competitive
    'dragapult': (88, 120, 75, 100, 75, 142),
    'gholdengo': (87, 60, 95, 133, 91, 84),
    'kingambit': (100, 135, 120, 60, 85, 50),
    'ironvaliant': (74, 130, 90, 120, 60, 116),
    'greattusk': (115, 131, 131, 53, 53, 87),
    'fluttermane': (55, 55, 55, 135, 135, 135),
    'ironhands': (154, 140, 108, 50, 68, 50),
    'chienpaot': (80, 120, 80, 90, 65, 135),
    'chiyu': (55, 80, 80, 135, 120, 100),
    'tinglu': (155, 110, 125, 55, 80, 45),
    'wochien': (85, 85, 100, 95, 135, 70),
    'roaringmoon': (105, 139, 71, 55, 101, 119),
    'ironmoth': (80, 70, 60, 140, 110, 110),
    'irontreads': (90, 112, 120, 72, 70, 106),
    'walkingwake': (99, 83, 91, 125, 83, 109),
    'ragingbolt': (125, 73, 91, 137, 89, 75),
    'gougingfire': (105, 115, 121, 65, 93, 91),
    'ironcrown': (90, 72, 100, 122, 108, 98),
    'ironboulder': (90, 120, 80, 68, 108, 124),
    
    # OU staples
    'garchomp': (108, 130, 95, 80, 85, 102),
    'tyranitar': (100, 134, 110, 95, 100, 61),
    'dragonite': (91, 134, 95, 100, 100, 80),
    'excadrill': (110, 135, 60, 50, 65, 88),
    'ferrothorn': (74, 94, 131, 54, 116, 20),
    'toxapex': (50, 63, 152, 53, 142, 35),
    'clefable': (95, 70, 73, 95, 90, 60),
    'corviknight': (98, 87, 105, 53, 85, 67),
    'heatran': (91, 90, 106, 130, 106, 77),
    'landorus': (89, 125, 90, 115, 80, 101),
    'landorustherian': (89, 145, 90, 105, 80, 91),
    'volcarona': (85, 60, 65, 135, 105, 100),
    'weavile': (70, 120, 65, 45, 85, 125),
    'scizor': (70, 130, 100, 55, 80, 65),
    'azumarill': (100, 50, 80, 60, 80, 50),
    'slowking': (95, 75, 80, 100, 110, 30),
    'slowkinggalar': (95, 65, 80, 110, 110, 30),
    'gliscor': (75, 95, 125, 45, 75, 95),
    'garganacl': (100, 100, 130, 45, 90, 35),
    'skeledirge': (104, 75, 100, 110, 75, 66),
    'annihilape': (110, 115, 80, 50, 90, 90),
    'meowscarada': (76, 110, 70, 81, 70, 123),
    'palafin': (100, 160, 97, 106, 87, 100),
    'tinkaton': (85, 75, 77, 70, 105, 94),
    'ceruledge': (75, 125, 80, 60, 100, 85),
    'armarouge': (85, 60, 100, 125, 80, 75),
    
    # Random battle commons
    'gyarados': (95, 125, 79, 60, 100, 81),
    'salamence': (95, 135, 80, 110, 80, 100),
    'metagross': (80, 135, 130, 95, 90, 70),
    'hydreigon': (92, 105, 90, 125, 90, 98),
    'goodra': (90, 100, 70, 110, 150, 80),
    'kommo': (75, 110, 125, 100, 105, 85),
    'mimikyu': (55, 90, 80, 50, 105, 96),
    'cinderace': (80, 116, 75, 65, 75, 119),
    'inteleon': (70, 85, 65, 125, 65, 120),
    'rillaboom': (100, 125, 90, 60, 70, 85),
    'urshifu': (100, 130, 100, 63, 60, 97),
    'calyrex': (100, 80, 80, 80, 80, 80),
    'calyrexice': (100, 165, 150, 85, 130, 50),
    'calyrexshadow': (100, 85, 80, 165, 100, 150),
    'zacian': (92, 130, 115, 80, 115, 138),
    'zamazenta': (92, 130, 145, 80, 145, 128),
    
    # More randbats Pokemon
    'gengar': (60, 65, 60, 130, 75, 110),
    'alakazam': (55, 50, 45, 135, 95, 120),
    'machamp': (90, 130, 80, 65, 85, 55),
    'slowbro': (95, 75, 110, 100, 80, 30),
    'magnezone': (70, 70, 115, 130, 90, 60),
    'espeon': (65, 65, 60, 130, 95, 110),
    'umbreon': (95, 65, 110, 60, 130, 65),
    'blissey': (255, 10, 10, 75, 135, 55),
    'milotic': (95, 60, 79, 100, 125, 81),
    'lucario': (70, 110, 70, 115, 70, 90),
    'togekiss': (85, 50, 95, 120, 115, 80),
    'rotom': (50, 50, 77, 95, 77, 91),
    'rotomwash': (50, 65, 107, 105, 107, 86),
    'rotomheat': (50, 65, 107, 105, 107, 86),
    'chandelure': (60, 55, 90, 145, 90, 80),
    'aegislash': (60, 50, 140, 50, 140, 60),  # Shield form
    'hawlucha': (78, 92, 75, 74, 63, 118),
    'talonflame': (78, 81, 71, 74, 69, 126),
    'primarina': (80, 74, 74, 126, 116, 60),
    'decidueye': (78, 107, 75, 100, 100, 70),
    'incineroar': (95, 115, 90, 80, 90, 60),
    'toxtricity': (75, 98, 70, 114, 70, 75),
    'grimmsnarl': (95, 120, 65, 95, 75, 60),
    'hatterene': (57, 90, 95, 136, 103, 29),
    'dragapult': (88, 120, 75, 100, 75, 142),
    'dracovish': (90, 90, 100, 70, 80, 75),
    'pelipper': (60, 50, 100, 95, 70, 65),
    'torkoal': (70, 85, 140, 85, 70, 20),
    'ninetales': (73, 76, 75, 81, 100, 100),
    'ninetalesalola': (73, 67, 75, 81, 100, 109),
    'hippowdon': (108, 112, 118, 68, 72, 47),
    'abomasnow': (90, 92, 75, 92, 85, 60),
    
    # More coverage
    'breloom': (60, 130, 80, 60, 60, 70),
    'conkeldurr': (105, 140, 95, 55, 65, 45),
    'bisharp': (65, 125, 100, 60, 70, 70),
    'hydreigon': (92, 105, 90, 125, 90, 98),
    'volcanion': (80, 110, 120, 130, 90, 70),
    'tapu koko': (70, 115, 85, 95, 75, 130),
    'tapu lele': (70, 85, 75, 130, 115, 95),
    'tapu bulu': (70, 130, 115, 85, 95, 75),
    'tapu fini': (70, 75, 115, 95, 130, 85),
    'magearna': (80, 95, 115, 130, 115, 65),
    'zeraora': (88, 112, 75, 102, 80, 143),
    'spectrier': (100, 65, 60, 145, 80, 130),
    'glastrier': (100, 145, 130, 65, 110, 30),
}


# Important abilities to track
ABILITY_EFFECTS = {
    # Speed modifiers
    'swiftswim': 'speed_rain',
    'chlorophyll': 'speed_sun',
    'sandrush': 'speed_sand',
    'slushrush': 'speed_snow',
    'unburden': 'speed_double',
    'speedboost': 'speed_boost',
    'quickfeet': 'speed_status',
    
    # Priority blockers
    'queenlymajesty': 'block_priority',
    'dazzling': 'block_priority', 
    'armortail': 'block_priority',
    'psychicsurge': 'psychic_terrain',
    
    # Damage modifiers
    'hugepower': 'double_atk',
    'purepower': 'double_atk',
    'adaptability': 'stab_boost',
    'technician': 'weak_boost',
    'toughclaws': 'contact_boost',
    'sheerforce': 'secondary_boost',
    'strongjaw': 'bite_boost',
    'ironfist': 'punch_boost',
    
    # Immunities
    'levitate': 'immune_ground',
    'flashfire': 'immune_fire',
    'waterabsorb': 'immune_water',
    'voltabsorb': 'immune_electric',
    'lightningrod': 'immune_electric',
    'stormdrain': 'immune_water',
    'sapsipper': 'immune_grass',
    'motordrive': 'immune_electric',
    'dryskin': 'immune_water',
    
    # Type changers
    'protean': 'type_change',
    'libero': 'type_change',
    
    # Defensive
    'multiscale': 'half_damage_full',
    'shadowshield': 'half_damage_full',
    'sturdy': 'survives_ohko',
    'disguise': 'free_hit',
    'iceface': 'free_physical',
    'magicbounce': 'reflect_status',
    'magicguard': 'no_indirect',
    
    # Weather setters
    'drought': 'set_sun',
    'drizzle': 'set_rain',
    'sandstream': 'set_sand',
    'snowwarning': 'set_snow',
    
    # Terrain setters
    'electricsurge': 'set_electric',
    'grassysurge': 'set_grassy',
    'psychicsurge': 'set_psychic',
    'mistysurge': 'set_misty',
    
    # Other important
    'intimidate': 'lower_atk',
    'regenerator': 'heal_switch',
    'naturalcure': 'cure_switch',
    'beastboost': 'stat_on_ko',
    'moxie': 'atk_on_ko',
    'soulheart': 'spa_on_ko',
}


# Item effects - combines data from items.json with known competitive effects
# Effect categories for feature encoding
ITEM_EFFECT_CATEGORIES = {
    # Choice items (locked to one move)
    'choiceband': ('choice', 'atk_boost', 1.5),
    'choicespecs': ('choice', 'spa_boost', 1.5),
    'choicescarf': ('choice', 'spe_boost', 1.5),

    # Damage boosters
    'lifeorb': ('damage', 'all_boost', 1.3),
    'expertbelt': ('damage', 'super_eff', 1.2),
    'metronome': ('damage', 'repeat_boost', 1.2),

    # Survival/defensive
    'focussash': ('survival', 'ohko_survive', 1.0),
    'assaultvest': ('defensive', 'spd_boost', 1.5),
    'eviolite': ('defensive', 'bulk_boost', 1.5),
    'rockyhelmet': ('defensive', 'contact_punish', 1/6),
    'airballoon': ('defensive', 'ground_immune', 1.0),
    'heavydutyboots': ('defensive', 'hazard_immune', 1.0),

    # Recovery
    'leftovers': ('recovery', 'passive', 1/16),
    'blacksludge': ('recovery', 'passive', 1/16),
    'shellbell': ('recovery', 'on_damage', 1/8),
    'sitrusberry': ('recovery', 'berry', 0.25),

    # Setup support
    'whiteherb': ('setup', 'clear_drops', 1.0),
    'weaknesspolicy': ('setup', 'boost_on_hit', 2.0),
    'throatspray': ('setup', 'spa_on_sound', 1.0),
    'powerherb': ('setup', 'skip_charge', 1.0),

    # Status berries
    'lumberry': ('status', 'cure_all', 1.0),
    'chestoberry': ('status', 'cure_sleep', 1.0),
    'rawstberry': ('status', 'cure_burn', 1.0),

    # Orbs (self-status for abilities)
    'flameorb': ('orb', 'self_burn', 1.0),
    'toxicorb': ('orb', 'self_poison', 1.0),

    # Speed control
    'ironball': ('speed', 'halve_speed', 0.5),
    'quickclaw': ('speed', 'priority_chance', 0.2),
    'custapberry': ('speed', 'priority_low_hp', 1.0),

    # Terrain/weather extenders
    'terrainextender': ('field', 'terrain_extend', 1.0),
    'heatrock': ('field', 'sun_extend', 1.0),
    'damprock': ('field', 'rain_extend', 1.0),
    'smoothrock': ('field', 'sand_extend', 1.0),
    'icyrock': ('field', 'snow_extend', 1.0),

    # Z-crystals and Tera (simplified)
    'terashard': ('tera', 'tera_support', 1.0),
}


def get_item_effect(item: str) -> tuple:
    """Get item effect category and value."""
    if not item:
        return ('none', 'none', 0)

    item_name = normalize_name(item)

    # Direct lookup
    if item_name in ITEM_EFFECT_CATEGORIES:
        return ITEM_EFFECT_CATEGORIES[item_name]

    # Check items.json for properties
    items_data = load_items()
    if item_name in items_data:
        item_data = items_data[item_name]

        # Choice items
        if item_data.get('isChoice'):
            return ('choice', 'locked', 1.0)

        # Items with boosts
        if 'boosts' in item_data:
            boosts = item_data['boosts']
            if 'atk' in boosts:
                return ('boost', 'atk', boosts['atk'])
            if 'spa' in boosts:
                return ('boost', 'spa', boosts['spa'])
            if 'spe' in boosts:
                return ('boost', 'spe', boosts['spe'])
            if 'def' in boosts:
                return ('boost', 'def', boosts['def'])

        # Berries
        if item_data.get('isBerry'):
            return ('berry', 'consumable', 1.0)

        # Gems
        if item_data.get('isGem'):
            return ('gem', 'type_boost', 1.3)

    return ('none', 'none', 0)


# Move categories for status moves
STATUS_MOVE_EFFECTS = {
    # Setup
    'swordsdance': ('setup', 2.0),
    'nastyplot': ('setup', 2.0),
    'dragondance': ('setup', 1.5),
    'quiverdance': ('setup', 2.0),
    'calmmind': ('setup', 1.5),
    'bulkup': ('setup', 1.3),
    'irondefense': ('setup', 1.3),
    'shellsmash': ('setup', 2.5),
    'coil': ('setup', 1.5),
    'shiftgear': ('setup', 1.8),
    'agility': ('speed', 2.0),
    'rockpolish': ('speed', 2.0),
    'autotomize': ('speed', 2.0),
    'tailwind': ('speed', 2.0),
    
    # Recovery
    'recover': ('heal', 0.5),
    'roost': ('heal', 0.5),
    'softboiled': ('heal', 0.5),
    'slackoff': ('heal', 0.5),
    'moonlight': ('heal', 0.5),
    'synthesis': ('heal', 0.5),
    'morningsun': ('heal', 0.5),
    'shoreup': ('heal', 0.5),
    'wish': ('heal', 0.5),
    'rest': ('heal', 1.0),
    'strengthsap': ('heal', 0.5),
    
    # Hazards
    'stealthrock': ('hazard', 1.5),
    'spikes': ('hazard', 1.2),
    'toxicspikes': ('hazard', 1.0),
    'stickyweb': ('hazard', 1.3),
    
    # Hazard removal
    'defog': ('remove_hazard', 1.0),
    'rapidspin': ('remove_hazard', 1.0),
    'courtchange': ('remove_hazard', 0.8),
    'tidyup': ('remove_hazard', 1.2),
    'mortalspin': ('remove_hazard', 1.0),
    
    # Status
    'willowisp': ('burn', 1.0),
    'thunderwave': ('paralyze', 1.0),
    'toxic': ('poison', 1.2),
    'spore': ('sleep', 1.5),
    'sleeppowder': ('sleep', 1.2),
    'hypnosis': ('sleep', 0.8),
    'yawn': ('sleep', 0.9),
    
    # Disruption
    'taunt': ('disrupt', 1.0),
    'encore': ('disrupt', 1.2),
    'disable': ('disrupt', 0.8),
    'trick': ('disrupt', 1.0),
    'switcheroo': ('disrupt', 1.0),
    'knockoff': ('disrupt', 1.0),
    
    # Phazing
    'whirlwind': ('phaze', 1.0),
    'roar': ('phaze', 1.0),
    'dragontail': ('phaze', 0.8),
    'circlethrow': ('phaze', 0.8),
    
    # Protection
    'protect': ('protect', 1.0),
    'detect': ('protect', 1.0),
    'kingsshield': ('protect', 1.0),
    'spikyshield': ('protect', 1.0),
    'banefulbunker': ('protect', 1.0),
    'silktrap': ('protect', 1.0),
    'substitute': ('protect', 1.5),
    
    # Pivoting
    'uturn': ('pivot', 1.0),
    'voltswitch': ('pivot', 1.0),
    'flipturn': ('pivot', 1.0),
    'partingshot': ('pivot', 1.2),
    'teleport': ('pivot', 1.0),
    'batonpass': ('pivot', 1.0),
    'shedtail': ('pivot', 1.5),
}


def normalize_name(name: str) -> str:
    """Normalize Pokemon/move name for lookup."""
    if not name:
        return ""
    return name.lower().replace(' ', '').replace('-', '').replace("'", "").replace('.', '')


def get_base_stats(species: str) -> Tuple[int, int, int, int, int, int]:
    """Get base stats for a Pokemon species."""
    name = normalize_name(species)
    
    # Direct lookup
    if name in POKEMON_STATS:
        return POKEMON_STATS[name]
    
    # Partial match
    for poke, stats in POKEMON_STATS.items():
        if name in poke or poke in name:
            return stats
    
    # Default average stats
    return (80, 80, 80, 80, 80, 80)


def estimate_stat(base: int, level: int = 84, is_hp: bool = False) -> int:
    """
    Estimate actual stat from base stat.
    
    For random battles:
    - Level varies by BST (typically 70-100)
    - IVs are 31
    - EVs are distributed (assume ~85 avg per stat)
    
    Simplified formula:
    HP: floor((2*Base + 31 + 21) * Level/100 + Level + 10)
    Other: floor((floor((2*Base + 31 + 21) * Level/100) + 5) * 1.0)  # Neutral nature
    """
    iv = 31
    ev_contrib = 21  # ~85 EVs / 4
    
    if is_hp:
        return int((2 * base + iv + ev_contrib) * level / 100 + level + 10)
    else:
        return int((2 * base + iv + ev_contrib) * level / 100 + 5)


def get_speed_tier(pokemon, boosts: dict = None) -> float:
    """
    Estimate effective speed for a Pokemon.
    Returns normalized speed (0-1 range).
    """
    if not pokemon:
        return 0.5
    
    # Get species name
    species = str(pokemon.species) if hasattr(pokemon, 'species') else str(pokemon)
    base_stats = get_base_stats(species)
    base_speed = base_stats[5]  # Speed is index 5
    
    # Estimate level (random battles use variable levels)
    # Higher BST Pokemon get lower levels
    bst = sum(base_stats)
    if bst >= 600:
        level = 72
    elif bst >= 500:
        level = 80
    elif bst >= 400:
        level = 88
    else:
        level = 92
    
    # Calculate base stat value
    speed = estimate_stat(base_speed, level)
    
    # Apply boosts
    if boosts:
        spe_boost = boosts.get('spe', 0)
        if spe_boost > 0:
            speed = speed * (2 + spe_boost) / 2
        elif spe_boost < 0:
            speed = speed * 2 / (2 - spe_boost)
    
    # Check for status (paralysis halves speed)
    if hasattr(pokemon, 'status') and pokemon.status:
        status = str(pokemon.status).lower()
        if 'par' in status:
            speed *= 0.5
    
    # Normalize to 0-1 range (max reasonable speed ~400)
    return min(speed / 400, 1.0)


def get_move_priority(move) -> int:
    """Get priority of a move."""
    if not move:
        return 0
    
    # Try to get priority from move object
    try:
        return move.priority
    except (KeyError, AttributeError):
        pass
    
    # Lookup from moves.json
    moves_data = load_moves()
    move_name = normalize_name(str(move.id) if hasattr(move, 'id') else str(move))
    
    if move_name in moves_data:
        return moves_data[move_name].get('priority', 0)
    
    return 0


def get_move_effect(move) -> Tuple[str, float]:
    """Get status move effect category and value."""
    if not move:
        return ('none', 0)
    
    move_name = normalize_name(str(move.id) if hasattr(move, 'id') else str(move))
    
    if move_name in STATUS_MOVE_EFFECTS:
        return STATUS_MOVE_EFFECTS[move_name]
    
    return ('none', 0)


def get_ability_effect(ability: str) -> str:
    """Get ability effect category."""
    if not ability:
        return 'none'
    
    ability_name = normalize_name(ability)
    return ABILITY_EFFECTS.get(ability_name, 'none')


def has_priority_immunity(pokemon) -> bool:
    """Check if Pokemon blocks priority moves."""
    if not pokemon:
        return False
    
    # Check ability
    if hasattr(pokemon, 'ability'):
        ability = normalize_name(str(pokemon.ability))
        if ability in ['queenlymajesty', 'dazzling', 'armortail']:
            return True
    
    return False


def get_type_effectiveness(atk_type: str, def_types: List[str]) -> float:
    """Calculate type effectiveness multiplier."""
    typechart = load_typechart()
    
    atk_type = atk_type.lower()
    multiplier = 1.0
    
    for def_type in def_types:
        def_type = def_type.lower()
        if def_type in typechart:
            damage_taken = typechart[def_type].get('damageTaken', {})
            # Type chart format: 0=normal, 1=weak, 2=resist, 3=immune
            effect = damage_taken.get(atk_type.capitalize(), 0)
            if effect == 1:
                multiplier *= 2.0
            elif effect == 2:
                multiplier *= 0.5
            elif effect == 3:
                multiplier *= 0.0
    
    return multiplier


class EnhancedFeatureBuilder:
    """
    Build 272-dimensional feature vector with:
    - Basic state (16)
    - Moves (4 × 12 = 48)
    - Switches (5 × 12 = 60)
    - Speed/priority (16)
    - Abilities (16)
    - Items (16)  <-- NEW
    - Field conditions (16)
    - Matchup analysis (32)
    - Boosts (24)
    - Status (16)
    - Hazards (12)
    """

    FEATURE_DIM = 272
    
    def __init__(self, enable_prediction_features: bool = False):
        self.moves_data = load_moves()
        self.abilities_data = load_abilities()
        self.typechart = load_typechart()
        self.moveset_priors = load_moveset_priors()
        self.enable_prediction_features = enable_prediction_features
    
    def build(self, battle) -> List[float]:
        """Build feature vector from battle state."""
        features = []
        
        my_poke = battle.active_pokemon
        opp_poke = battle.opponent_active_pokemon
        
        # Get types
        my_types = self._get_types(my_poke)
        opp_types = self._get_types(opp_poke)
        
        # === Basic State (16) ===
        features.extend(self._build_basic_state(battle, my_poke, opp_poke))
        
        # === Moves (48) ===
        features.extend(self._build_move_features(battle, my_poke, opp_poke, my_types, opp_types))
        
        # === Switches (60) ===
        features.extend(self._build_switch_features(battle, opp_poke, opp_types))
        
        # === Speed/Priority (16) ===
        features.extend(self._build_speed_features(battle, my_poke, opp_poke))
        
        # === Abilities (16) ===
        features.extend(self._build_ability_features(my_poke, opp_poke))

        # === Items (16) ===
        features.extend(self._build_item_features(my_poke, opp_poke))

        # === Field Conditions (16) ===
        features.extend(self._build_field_features(battle))
        
        # === Matchup Analysis (32) ===
        features.extend(self._build_matchup_features(my_poke, opp_poke, my_types, opp_types))
        
        # === Boosts (24) ===
        features.extend(self._build_boost_features(my_poke, opp_poke))
        
        # === Status (16) ===
        features.extend(self._build_status_features(battle, my_poke, opp_poke))
        
        # === Hazards (12) ===
        features.extend(self._build_hazard_features(battle))
        
        # Pad to FEATURE_DIM
        while len(features) < self.FEATURE_DIM:
            features.append(0.0)
        
        return features[:self.FEATURE_DIM]
    
    def _get_types(self, pokemon) -> List[str]:
        """Get Pokemon types as lowercase strings."""
        if not pokemon or not hasattr(pokemon, 'types'):
            return []
        
        types = []
        for t in pokemon.types:
            if t:
                type_str = str(t)
                # Handle "FIRE (pokemon type) object" format
                if '(' in type_str:
                    type_str = type_str.split('(')[0].strip()
                if '.' in type_str:
                    type_str = type_str.split('.')[-1]
                types.append(type_str.lower())
        
        return types
    
    def _build_basic_state(self, battle, my_poke, opp_poke) -> List[float]:
        """Basic battle state (16 features)."""
        features = []
        
        # Turn info
        features.append(min(battle.turn / 50, 1.0))
        
        # HP
        features.append(my_poke.current_hp_fraction if my_poke else 0)
        features.append(opp_poke.current_hp_fraction if opp_poke else 0)
        
        # Tera
        features.append(1 if battle.can_tera else 0)
        
        # Team count
        my_alive = len([p for p in battle.team.values() if not p.fainted])
        opp_alive = 6 - len([p for p in battle.opponent_team.values() if p.fainted])
        features.append(my_alive / 6)
        features.append(opp_alive / 6)
        features.append((my_alive - opp_alive + 6) / 12)
        
        # Team HP total
        my_total_hp = sum(p.current_hp_fraction for p in battle.team.values() if not p.fainted)
        features.append(my_total_hp / 6)
        
        # Force switch
        features.append(1 if battle.force_switch else 0)
        
        # Padding
        features.extend([0] * 7)
        
        return features
    
    def _build_move_features(self, battle, my_poke, opp_poke, my_types, opp_types) -> List[float]:
        """Move features (4 moves × 12 features = 48)."""
        features = []

        for i in range(4):
            if i < len(battle.available_moves):
                move = battle.available_moves[i]
                features.extend(self._get_single_move_features(move, my_types, opp_types, opp_poke, my_poke))
            else:
                features.extend([0] * 12)

        return features
    
    def _get_single_move_features(self, move, my_types, opp_types, opp_poke, my_poke=None) -> List[float]:
        """Features for a single move (12 features)."""
        features = []

        # Basic info
        features.append(1.0)  # Available

        move_type = self._parse_type(move.type) if hasattr(move, 'type') and move.type else 'normal'
        power = move.base_power or 0

        # Type effectiveness (only for damaging moves)
        if power > 0:
            eff = self._calc_effectiveness(move_type, opp_types)
        else:
            eff = 0
        features.append(eff / 4)

        # Power normalized
        features.append(power / 150)

        # STAB
        stab = 1 if move_type.lower() in [t.lower() for t in my_types] else 0
        features.append(stab)

        # Category
        cat = str(move.category).split('.')[-1].upper() if hasattr(move, 'category') else ''
        is_physical = 1 if cat == 'PHYSICAL' else 0
        is_special = 1 if cat == 'SPECIAL' else 0
        is_status = 1 if cat == 'STATUS' else 0
        features.append(is_physical)
        features.append(is_special)
        features.append(is_status)

        # Priority
        priority = get_move_priority(move)
        features.append(priority / 5 if priority > 0 else 0)
        features.append(-priority / 7 if priority < 0 else 0)

        # Status move effect value
        effect_type, effect_value = get_move_effect(move)
        features.append(effect_value / 2.5)

        # Accuracy
        acc = move.accuracy if hasattr(move, 'accuracy') and move.accuracy else 100
        if acc is True:
            acc = 100
        elif isinstance(acc, (int, float)) and acc <= 1.0:
            acc = acc * 100
        features.append(acc / 100)

        # Expected damage WITH stat ratios (like SimpleHeuristicsPlayer)
        if power > 0 and my_poke and opp_poke:
            # Get base stats
            my_stats = get_base_stats(str(my_poke.species) if hasattr(my_poke, 'species') else '')
            opp_stats = get_base_stats(str(opp_poke.species) if hasattr(opp_poke, 'species') else '')

            # Calculate stat ratio (attacker stat / defender stat)
            if is_physical:
                # Apply boosts
                my_atk_boost = my_poke.boosts.get('atk', 0) if hasattr(my_poke, 'boosts') else 0
                opp_def_boost = opp_poke.boosts.get('def', 0) if hasattr(opp_poke, 'boosts') else 0
                my_atk_mult = (2 + max(my_atk_boost, 0)) / (2 + max(-my_atk_boost, 0))
                opp_def_mult = (2 + max(opp_def_boost, 0)) / (2 + max(-opp_def_boost, 0))

                stat_ratio = (my_stats[1] * my_atk_mult) / max(opp_stats[2] * opp_def_mult, 1)
            elif is_special:
                # Apply boosts
                my_spa_boost = my_poke.boosts.get('spa', 0) if hasattr(my_poke, 'boosts') else 0
                opp_spd_boost = opp_poke.boosts.get('spd', 0) if hasattr(opp_poke, 'boosts') else 0
                my_spa_mult = (2 + max(my_spa_boost, 0)) / (2 + max(-my_spa_boost, 0))
                opp_spd_mult = (2 + max(opp_spd_boost, 0)) / (2 + max(-opp_spd_boost, 0))

                stat_ratio = (my_stats[3] * my_spa_mult) / max(opp_stats[4] * opp_spd_mult, 1)
            else:
                stat_ratio = 1.0

            # Expected damage = power * STAB * type_eff * accuracy * stat_ratio
            expected = power * (1.5 if stab else 1.0) * eff * (acc / 100) * stat_ratio
        else:
            expected = power * (1.5 if stab else 1.0) * (eff if eff > 0 else 1.0) * (acc / 100)

        # Normalize by a larger value to keep signal stronger
        features.append(min(expected / 200, 2.0))  # Cap at 2.0 for super effective moves

        return features
    
    def _build_switch_features(self, battle, opp_poke, opp_types) -> List[float]:
        """Switch features (5 Pokemon × 12 features = 60)."""
        features = []
        
        switches = battle.available_switches
        for i in range(5):
            if i < len(switches):
                poke = switches[i]
                features.extend(self._get_single_switch_features(poke, opp_poke, opp_types))
            else:
                features.extend([0] * 12)
        
        return features
    
    def _get_single_switch_features(self, poke, opp_poke, opp_types) -> List[float]:
        """Features for a single switch option (12 features)."""
        features = []
        
        poke_types = self._get_types(poke)
        
        # Available and HP
        features.append(1.0)
        features.append(poke.current_hp_fraction)
        
        # Offensive potential (best STAB vs opponent)
        best_off = max((self._calc_effectiveness(t, opp_types) for t in poke_types), default=1.0)
        features.append(best_off / 4)
        
        # Defensive matchup (opponent's best vs us)
        best_def = max((self._calc_effectiveness(t, poke_types) for t in opp_types), default=1.0)
        features.append(1 / max(best_def, 0.25) / 4)  # Invert - lower is better defense
        
        # Speed tier
        speed = get_speed_tier(poke)
        features.append(speed)
        
        # Is faster than opponent?
        opp_speed = get_speed_tier(opp_poke) if opp_poke else 0.5
        features.append(1 if speed > opp_speed else 0)
        
        # Base stats estimate
        stats = get_base_stats(str(poke.species) if hasattr(poke, 'species') else str(poke))
        features.append(stats[1] / 150)  # Attack
        features.append(stats[3] / 150)  # SpA
        features.append((stats[2] + stats[4]) / 300)  # Def + SpD
        
        # Status
        has_status = 1 if hasattr(poke, 'status') and poke.status else 0
        features.append(has_status)
        
        # Padding
        features.extend([0, 0])
        
        return features
    
    def _build_speed_features(self, battle, my_poke, opp_poke) -> List[float]:
        """Speed and priority features (16)."""
        features = []
        
        my_speed = get_speed_tier(my_poke, my_poke.boosts if my_poke else None)
        opp_speed = get_speed_tier(opp_poke, opp_poke.boosts if opp_poke else None)
        
        # Raw speeds
        features.append(my_speed)
        features.append(opp_speed)
        
        # Speed comparison
        features.append(1 if my_speed > opp_speed else 0)
        features.append(my_speed - opp_speed)
        
        # Has priority moves?
        has_priority = 0
        for move in battle.available_moves:
            if get_move_priority(move) > 0:
                has_priority = 1
                break
        features.append(has_priority)
        
        # Opponent blocks priority?
        features.append(1 if has_priority_immunity(opp_poke) else 0)
        
        # Speed control active (Tailwind, Trick Room)
        features.append(1 if 'tailwind' in str(battle.side_conditions).lower() else 0)
        features.append(1 if 'trickroom' in str(battle.fields).lower() else 0)
        
        # Paralysis speed drop
        my_para = 1 if my_poke and my_poke.status and 'par' in str(my_poke.status).lower() else 0
        opp_para = 1 if opp_poke and opp_poke.status and 'par' in str(opp_poke.status).lower() else 0
        features.append(my_para)
        features.append(opp_para)
        
        # Prediction features (optional)
        if self.enable_prediction_features:
            features.extend(self._build_prediction_features(battle, my_poke, opp_poke))
        else:
            features.extend([0] * 6)
        
        return features

    def _build_prediction_features(self, battle, my_poke, opp_poke) -> List[float]:
        """Prediction features (6): damage, status, setup, recovery, priority, hazard."""
        if not opp_poke:
            return [0] * 6

        my_types = self._get_types(my_poke)
        candidates = []
        best_damage = 0.0

        known_ids = self._get_known_move_ids(opp_poke)
        for move_id in known_ids:
            entry = self.moves_data.get(move_id, {})
            kind = self._classify_move_entry(entry)
            if kind == "damage":
                score = self._estimate_entry_damage(entry, my_types)
                best_damage = max(best_damage, score)
            else:
                score = self._status_entry_score(kind, opp_poke)
            priority = 1 if entry.get("priority", 0) > 0 else 0
            candidates.append((score, kind, priority))

        species_key = normalize_name(str(getattr(opp_poke, "species", "")))
        prior_moves = self.moveset_priors.get(species_key, {})
        if isinstance(prior_moves, dict) and prior_moves:
            max_count = max(prior_moves.values())
            for move_id, count in sorted(prior_moves.items(), key=lambda x: x[1], reverse=True)[:6]:
                if move_id in known_ids:
                    continue
                entry = self.moves_data.get(move_id, {})
                kind = self._classify_move_entry(entry)
                if kind == "damage":
                    score = self._estimate_entry_damage(entry, my_types)
                    best_damage = max(best_damage, score)
                else:
                    score = self._status_entry_score(kind, opp_poke)
                weight = 0.5 + 0.5 * (count / max_count)
                priority = 1 if entry.get("priority", 0) > 0 else 0
                candidates.append((score * weight, kind, priority))

        if candidates:
            best = max(candidates, key=lambda x: x[0])
            best_kind = best[1]
            best_priority = best[2]
        else:
            best_kind = "unknown"
            best_priority = 0

        predicted_damage = min(best_damage / 200.0, 1.0)
        predicted_status = 1 if best_kind == "status" else 0
        predicted_setup = 1 if best_kind == "setup" else 0
        predicted_recovery = 1 if best_kind == "recovery" else 0
        predicted_priority = best_priority
        predicted_hazard = 1 if best_kind == "hazard" else 0

        return [
            predicted_damage,
            predicted_status,
            predicted_setup,
            predicted_recovery,
            predicted_priority,
            predicted_hazard,
        ]

    def _get_known_move_ids(self, pokemon) -> List[str]:
        if not pokemon:
            return []
        moves = getattr(pokemon, "moves", None)
        move_objs = []
        if isinstance(moves, dict):
            move_objs = list(moves.values())
        elif moves:
            move_objs = list(moves)
        move_ids = []
        for move in move_objs:
            move_id = getattr(move, "id", None)
            if move_id:
                move_ids.append(normalize_name(str(move_id)))
        return move_ids

    def _classify_move_entry(self, entry: dict) -> str:
        if not entry:
            return "status"
        category = str(entry.get("category", "")).lower()
        if category != "status":
            return "damage"
        if entry.get("heal"):
            return "recovery"
        if entry.get("sideCondition") or (entry.get("self") or {}).get("sideCondition"):
            return "hazard"
        boosts = entry.get("boosts") or (entry.get("self") or {}).get("boosts")
        if boosts and any(delta > 0 for delta in boosts.values()):
            return "setup"
        return "status"

    def _status_entry_score(self, kind: str, opponent) -> float:
        if kind == "setup":
            return 130.0
        if kind == "recovery":
            if getattr(opponent, "current_hp_fraction", 1.0) < 0.6:
                return 120.0
            return 70.0
        if kind == "hazard":
            return 90.0
        return 80.0

    def _estimate_entry_damage(self, entry: dict, target_types: List[str]) -> float:
        if not entry:
            return 0.0
        category = str(entry.get("category", "")).lower()
        if category == "status":
            return 0.0
        base_power = entry.get("basePower") or 0
        if base_power <= 0:
            return 0.0
        move_type = str(entry.get("type", "")).lower()
        type_mult = self._calc_effectiveness(move_type, target_types) if move_type else 1.0
        accuracy = entry.get("accuracy")
        if accuracy is True or accuracy is None:
            acc_mult = 1.0
        else:
            acc_mult = max(min(float(accuracy) / 100.0, 1.0), 0.0)
        return base_power * type_mult * acc_mult
    
    def _build_ability_features(self, my_poke, opp_poke) -> List[float]:
        """Ability features (16)."""
        features = []
        
        # My ability effects
        my_ability = str(my_poke.ability) if my_poke and hasattr(my_poke, 'ability') else ''
        my_effect = get_ability_effect(my_ability)
        
        # Key ability categories
        features.append(1 if 'speed' in my_effect else 0)
        features.append(1 if 'immune' in my_effect else 0)
        features.append(1 if 'boost' in my_effect else 0)
        features.append(1 if my_effect in ['double_atk', 'stab_boost'] else 0)
        
        # Opponent ability effects
        opp_ability = str(opp_poke.ability) if opp_poke and hasattr(opp_poke, 'ability') else ''
        opp_effect = get_ability_effect(opp_ability)
        
        features.append(1 if 'immune' in opp_effect else 0)
        features.append(1 if opp_effect == 'block_priority' else 0)
        features.append(1 if opp_effect in ['half_damage_full', 'sturdy', 'disguise'] else 0)
        features.append(1 if opp_effect == 'reflect_status' else 0)
        
        # Weather/terrain setting abilities
        features.append(1 if 'set_' in my_effect else 0)
        features.append(1 if 'set_' in opp_effect else 0)
        
        # Intimidate
        features.append(1 if my_effect == 'lower_atk' else 0)
        features.append(1 if opp_effect == 'lower_atk' else 0)
        
        # Regenerator/pivoting abilities
        features.append(1 if my_effect in ['heal_switch', 'cure_switch'] else 0)
        features.append(1 if opp_effect in ['heal_switch', 'cure_switch'] else 0)

        # Padding
        features.extend([0, 0])

        return features

    def _build_item_features(self, my_poke, opp_poke) -> List[float]:
        """Item features (16)."""
        features = []

        # My item
        my_item = str(my_poke.item) if my_poke and hasattr(my_poke, 'item') and my_poke.item else ''
        my_category, my_sub, my_value = get_item_effect(my_item)

        # Key item categories for my pokemon
        features.append(1 if my_category == 'choice' else 0)
        features.append(1 if my_category == 'damage' else 0)
        features.append(1 if my_category in ['survival', 'defensive'] else 0)
        features.append(1 if my_category == 'recovery' else 0)
        features.append(1 if my_category == 'setup' else 0)
        features.append(my_value if my_category in ['choice', 'damage'] else 0)

        # Opponent item
        opp_item = str(opp_poke.item) if opp_poke and hasattr(opp_poke, 'item') and opp_poke.item else ''
        opp_category, opp_sub, _ = get_item_effect(opp_item)

        features.append(1 if opp_category == 'choice' else 0)
        features.append(1 if opp_category in ['survival', 'defensive'] else 0)
        features.append(1 if opp_sub == 'ohko_survive' else 0)  # Focus Sash
        features.append(1 if opp_sub == 'hazard_immune' else 0)  # Heavy Duty Boots

        # Setup synergy - my item supports setup?
        # Sturdy + White Herb, Focus Sash, etc.
        my_ability = str(my_poke.ability) if my_poke and hasattr(my_poke, 'ability') else ''
        my_ability_effect = get_ability_effect(my_ability)

        # Shell Smash synergy (Sturdy/Sash + White Herb or just Sash)
        setup_synergy = 0
        if my_ability_effect == 'survives_ohko' and my_sub == 'clear_drops':
            setup_synergy = 1.0  # Sturdy + White Herb
        elif my_sub == 'ohko_survive':
            setup_synergy = 0.8  # Focus Sash for setup
        elif my_sub == 'boost_on_hit':
            setup_synergy = 0.7  # Weakness Policy
        features.append(setup_synergy)

        # Padding
        features.extend([0] * 5)

        return features

    def _build_field_features(self, battle) -> List[float]:
        """Field condition features (16)."""
        features = []
        
        # Weather
        weather = str(battle.weather).lower() if battle.weather else ''
        features.append(1 if 'sun' in weather else 0)
        features.append(1 if 'rain' in weather else 0)
        features.append(1 if 'sand' in weather else 0)
        features.append(1 if 'snow' in weather or 'hail' in weather else 0)
        
        # Terrain
        terrain = str(battle.fields).lower() if battle.fields else ''
        features.append(1 if 'electric' in terrain else 0)
        features.append(1 if 'grassy' in terrain else 0)
        features.append(1 if 'psychic' in terrain else 0)
        features.append(1 if 'misty' in terrain else 0)
        
        # Trick room
        features.append(1 if 'trickroom' in terrain else 0)
        
        # Screens (my side)
        side = str(battle.side_conditions).lower()
        features.append(1 if 'reflect' in side else 0)
        features.append(1 if 'lightscreen' in side else 0)
        features.append(1 if 'auroraveil' in side else 0)
        
        # Screens (opponent side)
        opp_side = str(battle.opponent_side_conditions).lower()
        features.append(1 if 'reflect' in opp_side else 0)
        features.append(1 if 'lightscreen' in opp_side else 0)
        features.append(1 if 'auroraveil' in opp_side else 0)
        
        # Padding
        features.append(0)
        
        return features
    
    def _build_matchup_features(self, my_poke, opp_poke, my_types, opp_types) -> List[float]:
        """Matchup analysis features (32)."""
        features = []
        
        # Type matchups (all 18 types vs opponent)
        ALL_TYPES = ['normal', 'fire', 'water', 'electric', 'grass', 'ice', 
                     'fighting', 'poison', 'ground', 'flying', 'psychic', 'bug',
                     'rock', 'ghost', 'dragon', 'dark', 'steel', 'fairy']
        
        # Best offensive types vs opponent
        best_off = 0
        worst_off = 4
        for t in ALL_TYPES:
            eff = self._calc_effectiveness(t, opp_types)
            best_off = max(best_off, eff)
            worst_off = min(worst_off, eff)
        
        features.append(best_off / 4)
        features.append(worst_off / 4)
        
        # My types vs opponent
        for t in my_types[:2]:  # Max 2 types
            features.append(self._calc_effectiveness(t, opp_types) / 4)
        while len(features) < 4:
            features.append(0.25)
        
        # Opponent's best vs my types
        best_def = 0
        for t in ALL_TYPES:
            eff = self._calc_effectiveness(t, my_types)
            best_def = max(best_def, eff)
        features.append(best_def / 4)
        
        # Am I resistant to opponent's STAB?
        if opp_types:
            opp_stab_eff = max(self._calc_effectiveness(t, my_types) for t in opp_types)
            features.append(1 if opp_stab_eff <= 0.5 else 0)
            features.append(1 if opp_stab_eff == 0 else 0)
        else:
            features.extend([0, 0])
        
        # Base stat comparison
        my_stats = get_base_stats(str(my_poke.species) if my_poke else '')
        opp_stats = get_base_stats(str(opp_poke.species) if opp_poke else '')
        
        # Attack vs Def
        features.append((my_stats[1] - opp_stats[2]) / 100)  # Atk vs Def
        features.append((my_stats[3] - opp_stats[4]) / 100)  # SpA vs SpD
        features.append((my_stats[5] - opp_stats[5]) / 100)  # Speed comparison
        
        # Offensive vs Defensive
        my_offense = max(my_stats[1], my_stats[3])
        opp_defense = (opp_stats[2] + opp_stats[4]) / 2
        features.append((my_offense - opp_defense) / 100)
        
        # HP comparison
        features.append((my_stats[0] - opp_stats[0]) / 100)
        
        # BST comparison
        my_bst = sum(my_stats)
        opp_bst = sum(opp_stats)
        features.append((my_bst - opp_bst) / 200)
        
        # Is my best attack physical or special?
        features.append(1 if my_stats[1] > my_stats[3] else 0)
        
        # Padding to 32
        while len(features) < 32:
            features.append(0)
        
        return features[:32]
    
    def _build_boost_features(self, my_poke, opp_poke) -> List[float]:
        """Boost/stat stage features (24)."""
        features = []
        
        # My boosts
        if my_poke and hasattr(my_poke, 'boosts'):
            for stat in ['atk', 'def', 'spa', 'spd', 'spe', 'accuracy', 'evasion']:
                features.append(my_poke.boosts.get(stat, 0) / 6)
        else:
            features.extend([0] * 7)
        
        # Opponent boosts
        if opp_poke and hasattr(opp_poke, 'boosts'):
            for stat in ['atk', 'def', 'spa', 'spd', 'spe', 'accuracy', 'evasion']:
                features.append(opp_poke.boosts.get(stat, 0) / 6)
        else:
            features.extend([0] * 7)
        
        # Total boost advantage
        my_total = sum(my_poke.boosts.values()) if my_poke and hasattr(my_poke, 'boosts') else 0
        opp_total = sum(opp_poke.boosts.values()) if opp_poke and hasattr(opp_poke, 'boosts') else 0
        features.append((my_total - opp_total) / 12)
        
        # Offensive boost advantage
        my_off = 0
        opp_off = 0
        if my_poke and hasattr(my_poke, 'boosts'):
            my_off = max(my_poke.boosts.get('atk', 0), my_poke.boosts.get('spa', 0))
        if opp_poke and hasattr(opp_poke, 'boosts'):
            opp_off = max(opp_poke.boosts.get('atk', 0), opp_poke.boosts.get('spa', 0))
        features.append((my_off - opp_off) / 6)
        
        # Padding
        features.extend([0] * 8)
        
        return features[:24]
    
    def _build_status_features(self, battle, my_poke, opp_poke) -> List[float]:
        """Status condition features (16)."""
        features = []
        
        STATUS_MAP = {'brn': 0, 'par': 1, 'slp': 2, 'psn': 3, 'tox': 4, 'frz': 5}
        
        # My status
        my_status = str(my_poke.status).lower().split('.')[-1] if my_poke and my_poke.status else ''
        for status in ['brn', 'par', 'slp', 'psn', 'tox', 'frz']:
            features.append(1 if status in my_status else 0)
        
        # Opponent status
        opp_status = str(opp_poke.status).lower().split('.')[-1] if opp_poke and opp_poke.status else ''
        for status in ['brn', 'par', 'slp', 'psn', 'tox', 'frz']:
            features.append(1 if status in opp_status else 0)
        
        # Any status
        features.append(1 if my_status else 0)
        features.append(1 if opp_status else 0)
        
        # Padding
        features.extend([0, 0])
        
        return features
    
    def _build_hazard_features(self, battle) -> List[float]:
        """Entry hazard features (12)."""
        features = []
        
        def count_hazard(side, name):
            for k, v in side.items():
                if name in str(k).lower():
                    if isinstance(v, int):
                        return v
                    return 1
            return 0
        
        # My side hazards
        features.append(count_hazard(battle.side_conditions, 'stealthrock'))
        features.append(count_hazard(battle.side_conditions, 'spikes') / 3)
        features.append(count_hazard(battle.side_conditions, 'toxicspikes') / 2)
        features.append(count_hazard(battle.side_conditions, 'stickyweb'))
        
        # Opponent hazards
        features.append(count_hazard(battle.opponent_side_conditions, 'stealthrock'))
        features.append(count_hazard(battle.opponent_side_conditions, 'spikes') / 3)
        features.append(count_hazard(battle.opponent_side_conditions, 'toxicspikes') / 2)
        features.append(count_hazard(battle.opponent_side_conditions, 'stickyweb'))
        
        # Net hazard advantage
        my_hazards = sum(features[:4])
        opp_hazards = sum(features[4:8])
        features.append(opp_hazards - my_hazards)  # Positive = good for us
        
        # Padding
        features.extend([0, 0, 0])
        
        return features
    
    def _parse_type(self, type_obj) -> str:
        """Parse type from poke-env type object."""
        if not type_obj:
            return 'normal'
        
        type_str = str(type_obj)
        if '(' in type_str:
            type_str = type_str.split('(')[0].strip()
        if '.' in type_str:
            type_str = type_str.split('.')[-1]
        
        return type_str.lower()
    
    def _calc_effectiveness(self, atk_type: str, def_types: List[str]) -> float:
        """Calculate type effectiveness."""
        return get_type_effectiveness(atk_type, def_types)


# Convenience function
def build_features(battle) -> List[float]:
    """Build 256-dim feature vector from battle."""
    builder = EnhancedFeatureBuilder()
    return builder.build(battle)
