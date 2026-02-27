#!/usr/bin/env python3
"""
Damage-based belief refinement for MCTS opponent set estimation.

Pure-function module — no poke_env / FoulPlay dependencies.
Given a damage observation (we hit opponent for X% HP), scores how
consistent each candidate set (ability + item) is with that observation.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Type chart (inline, same as damage_calc.py)
# ---------------------------------------------------------------------------
_TYPE_CHART = {
    "normal": {"rock": 0.5, "ghost": 0, "steel": 0.5},
    "fire": {"fire": 0.5, "water": 0.5, "grass": 2, "ice": 2, "bug": 2, "rock": 0.5, "dragon": 0.5, "steel": 2},
    "water": {"fire": 2, "water": 0.5, "grass": 0.5, "ground": 2, "rock": 2, "dragon": 0.5},
    "electric": {"water": 2, "electric": 0.5, "grass": 0.5, "ground": 0, "flying": 2, "dragon": 0.5},
    "grass": {"fire": 0.5, "water": 2, "grass": 0.5, "poison": 0.5, "ground": 2, "flying": 0.5, "bug": 0.5, "rock": 2, "dragon": 0.5, "steel": 0.5},
    "ice": {"fire": 0.5, "water": 0.5, "grass": 2, "ice": 0.5, "ground": 2, "flying": 2, "dragon": 2, "steel": 0.5},
    "fighting": {"normal": 2, "ice": 2, "poison": 0.5, "flying": 0.5, "psychic": 0.5, "bug": 0.5, "rock": 2, "ghost": 0, "dark": 2, "steel": 2, "fairy": 0.5},
    "poison": {"grass": 2, "poison": 0.5, "ground": 0.5, "rock": 0.5, "ghost": 0.5, "steel": 0, "fairy": 2},
    "ground": {"fire": 2, "electric": 2, "grass": 0.5, "poison": 2, "flying": 0, "bug": 0.5, "rock": 2, "steel": 2},
    "flying": {"electric": 0.5, "grass": 2, "fighting": 2, "bug": 2, "rock": 0.5, "steel": 0.5},
    "psychic": {"fighting": 2, "poison": 2, "psychic": 0.5, "dark": 0, "steel": 0.5},
    "bug": {"fire": 0.5, "grass": 2, "fighting": 0.5, "poison": 0.5, "flying": 0.5, "psychic": 2, "ghost": 0.5, "dark": 2, "steel": 0.5, "fairy": 0.5},
    "rock": {"fire": 2, "ice": 2, "fighting": 0.5, "ground": 0.5, "flying": 2, "bug": 2, "steel": 0.5},
    "ghost": {"normal": 0, "psychic": 2, "ghost": 2, "dark": 0.5},
    "dragon": {"dragon": 2, "steel": 0.5, "fairy": 0},
    "dark": {"fighting": 0.5, "psychic": 2, "ghost": 2, "dark": 0.5, "fairy": 0.5},
    "steel": {"fire": 0.5, "water": 0.5, "electric": 0.5, "ice": 2, "rock": 2, "steel": 0.5, "fairy": 2},
    "fairy": {"fire": 0.5, "fighting": 2, "poison": 0.5, "dragon": 2, "dark": 2, "steel": 0.5},
}

# Type-immunity abilities
_IMMUNE_ABILITIES: Dict[str, str] = {
    "levitate": "ground",
    "flashfire": "fire",
    "wellbakedbody": "fire",
    "waterabsorb": "water",
    "stormdrain": "water",
    "dryskin": "water",
    "voltabsorb": "electric",
    "lightningrod": "electric",
    "motordrive": "electric",
    "sapsipper": "grass",
    "eartheater": "ground",
}

# Type-boosting items (normalized name → boosted type)
_TYPE_BOOST_ITEMS: Dict[str, str] = {
    "charcoal": "fire",
    "mysticwater": "water",
    "miracleseed": "grass",
    "magnet": "electric",
    "nevermeltice": "ice",
    "blackbelt": "fighting",
    "poisonbarb": "poison",
    "softsand": "ground",
    "sharpbeak": "flying",
    "twistedspoon": "psychic",
    "silverpowder": "bug",
    "hardstone": "rock",
    "spelltag": "ghost",
    "dragonfang": "dragon",
    "blackglasses": "dark",
    "metalcoat": "steel",
}


def _norm(name: str) -> str:
    if not name:
        return ""
    return name.lower().replace(" ", "").replace("-", "").replace("'", "").replace(".", "")


def _type_eff(atk_type: str, def_types: List[str]) -> float:
    atk = _norm(atk_type)
    mult = 1.0
    chart = _TYPE_CHART.get(atk, {})
    for dt in def_types:
        mult *= chart.get(_norm(dt), 1.0)
    return mult


# ---------------------------------------------------------------------------
# Randbats stat computation
# ---------------------------------------------------------------------------
def calc_randbat_stat(base: int, level: int, is_hp: bool = False) -> int:
    """Compute stat for a randbats Pokemon (31 IVs, 85 EVs, neutral nature)."""
    iv = 31
    ev = 85
    if is_hp:
        return int((2 * base + iv + ev // 4) * level / 100 + level + 10)
    else:
        return int(((2 * base + iv + ev // 4) * level / 100 + 5) * 1.0)


# ---------------------------------------------------------------------------
# Core damage range computation
# ---------------------------------------------------------------------------
def compute_expected_damage_frac(
    move_bp: int,
    move_type: str,
    move_category: str,            # "physical" or "special"
    attacker_stat: int,            # actual atk or spa integer (already boosted)
    attacker_level: int,
    attacker_types: List[str],
    attacker_boosts: Dict[str, int],
    attacker_status: Optional[str],
    attacker_ability: str,
    attacker_item: str,
    defender_base_stats: Dict[str, int],  # {"hp","atk","def","spa","spd","spe"}
    defender_level: int,
    defender_types: List[str],
    defender_boosts: Dict[str, int],
    defender_ability: str,
    defender_item: str,
    defender_hp_frac: float,
    weather: Optional[str] = None,
    terrain: Optional[str] = None,
    is_crit: bool = False,
) -> Tuple[float, float]:
    """Return (min_frac, max_frac) of defender max HP dealt by the attack.

    Returns (0.0, 0.0) if the attack would be fully blocked (immunity).
    """
    if move_bp <= 0:
        return (0.0, 0.0)

    move_type_n = _norm(move_type)
    move_cat = (move_category or "physical").lower()
    atk_ability = _norm(attacker_ability)
    atk_item = _norm(attacker_item)
    def_ability = _norm(defender_ability)
    def_item = _norm(defender_item)
    atk_types = [_norm(t) for t in attacker_types]
    def_types = [_norm(t) for t in defender_types]

    # --- Attacker stat ---
    # attacker_stat is provided pre-boosted (the active pokemon's actual stat
    # value after stage multipliers). We use it directly.
    atk_val = float(attacker_stat)

    # Attacker ability mods on atk_val
    if atk_ability in ("hugepower", "purepower") and move_cat == "physical":
        atk_val *= 2.0

    # Attacker item mods on atk_val
    if atk_item == "choiceband" and move_cat == "physical":
        atk_val *= 1.5
    elif atk_item == "choicespecs" and move_cat == "special":
        atk_val *= 1.5

    # --- Defender stat ---
    if move_cat == "physical":
        def_base = defender_base_stats.get("def", 80)
    else:
        def_base = defender_base_stats.get("spd", 80)

    def_stat_raw = calc_randbat_stat(def_base, defender_level)

    # Apply boosts (crits ignore positive def boosts)
    if move_cat == "physical":
        def_boost = defender_boosts.get("def", 0)
    else:
        def_boost = defender_boosts.get("spd", 0)

    if is_crit:
        def_boost = min(def_boost, 0)

    if def_boost > 0:
        def_stat = int(def_stat_raw * (2 + def_boost) / 2)
    elif def_boost < 0:
        def_stat = int(def_stat_raw * 2 / (2 - def_boost))
    else:
        def_stat = def_stat_raw

    # --- Defender ability mods on def_stat ---
    if def_ability == "furcoat" and move_cat == "physical":
        def_stat *= 2
    if def_ability == "icescales" and move_cat == "special":
        def_stat *= 2  # effectively doubles SpD in the formula

    # --- Defender item mods on def_stat ---
    if def_item == "assaultvest" and move_cat == "special":
        def_stat = int(def_stat * 1.5)
    if def_item == "eviolite":
        def_stat = int(def_stat * 1.5)

    # --- Modifiers ---
    modifiers = 1.0

    # STAB
    stab = 1.0
    if move_type_n in atk_types:
        stab = 2.0 if atk_ability == "adaptability" else 1.5
    modifiers *= stab

    # Type effectiveness
    type_eff = _type_eff(move_type_n, def_types)

    # Ability immunities
    immune_type = _IMMUNE_ABILITIES.get(def_ability)
    if immune_type and immune_type == move_type_n:
        return (0.0, 0.0)

    if type_eff == 0:
        return (0.0, 0.0)
    modifiers *= type_eff

    # Weather
    if weather:
        w = _norm(weather)
        if "sun" in w or "sunnyday" in w:
            if move_type_n == "fire":
                modifiers *= 1.5
            elif move_type_n == "water":
                modifiers *= 0.5
        elif "rain" in w or "raindance" in w:
            if move_type_n == "water":
                modifiers *= 1.5
            elif move_type_n == "fire":
                modifiers *= 0.5

    # Terrain
    if terrain:
        t = _norm(terrain)
        grounded_attacker = "flying" not in atk_types and atk_ability != "levitate"
        grounded_defender = "flying" not in def_types and def_ability != "levitate"
        if "electric" in t and move_type_n == "electric" and grounded_attacker:
            modifiers *= 1.3
        elif "grassy" in t and move_type_n == "grass" and grounded_attacker:
            modifiers *= 1.3
        elif "psychic" in t and move_type_n == "psychic" and grounded_attacker:
            modifiers *= 1.3
        if "misty" in t and move_type_n == "dragon" and grounded_defender:
            modifiers *= 0.5

    # Burn
    if move_cat == "physical" and attacker_status:
        s = _norm(str(attacker_status))
        if "brn" in s or "burn" in s:
            if atk_ability != "guts":
                modifiers *= 0.5

    # Critical hit
    if is_crit:
        modifiers *= 1.5

    # --- Defender abilities (damage-modifying) ---
    if def_ability in ("multiscale", "shadowshield") and defender_hp_frac >= 0.99:
        modifiers *= 0.5
    if def_ability in ("filter", "solidrock", "prismarmor") and type_eff > 1.0:
        modifiers *= 0.75
    if def_ability == "fluffy":
        # Fluffy halves contact moves, doubles fire. We approximate contact as
        # physical (good enough for belief scoring).
        if move_cat == "physical" and move_type_n != "fire":
            modifiers *= 0.5
        if move_type_n == "fire":
            modifiers *= 2.0
    if def_ability == "thickfat" and move_type_n in ("fire", "ice"):
        modifiers *= 0.5
    if def_ability == "waterbubble" and move_type_n == "fire":
        modifiers *= 0.5
    if def_ability == "heatproof" and move_type_n == "fire":
        modifiers *= 0.5
    if def_ability == "dryskin" and move_type_n == "fire":
        modifiers *= 1.25

    # --- Attacker abilities (damage-modifying) ---
    if atk_ability == "technician" and move_bp <= 60:
        modifiers *= 1.5

    # --- Attacker items (damage-modifying) ---
    if atk_item == "lifeorb":
        modifiers *= 1.3
    elif atk_item == "expertbelt" and type_eff > 1.0:
        modifiers *= 1.2
    else:
        boosted_type = _TYPE_BOOST_ITEMS.get(atk_item)
        if boosted_type and boosted_type == move_type_n:
            modifiers *= 1.2

    # --- Damage formula ---
    # ((2*Level/5 + 2) * Power * A / D / 50 + 2) * Modifiers
    base_damage = ((2.0 * attacker_level / 5.0 + 2.0) * move_bp * atk_val / max(def_stat, 1)) / 50.0 + 2.0

    # Defender max HP
    hp_base = defender_base_stats.get("hp", 80)
    max_hp = calc_randbat_stat(hp_base, defender_level, is_hp=True)
    if max_hp <= 0:
        max_hp = 1

    min_dmg = base_damage * modifiers * 0.85
    max_dmg = base_damage * modifiers * 1.00

    min_frac = min_dmg / max_hp
    max_frac = max_dmg / max_hp

    return (min_frac, max_frac)


# ---------------------------------------------------------------------------
# Score a set against damage observations
# ---------------------------------------------------------------------------
def score_set_damage_consistency(
    observations: List[dict],
    set_ability: str,
    set_item: str,
    species_base_stats: Dict[str, int],
    species_level: int,
    species_types: List[str],
    tolerance: float = 0.02,
    mode: str = "hard",
    per_obs_min: float = 0.35,
    per_obs_max: float = 1.10,
    final_min: float = 0.10,
    final_max: float = 10.0,
) -> float:
    """Score how consistent a candidate set is with observed damage.

    Each observation dict must contain:
      - move_bp, move_type, move_category
      - attacker_stat, attacker_level, attacker_types, attacker_boosts
      - attacker_status, attacker_ability, attacker_item
      - defender_boosts, defender_hp_frac
      - observed_frac   (fraction of defender max HP lost)
      - weather (optional), terrain (optional), is_crit (optional)

    Returns a weight multiplier.
    - hard mode: original tiered penalties (strong effect)
    - soft mode: gentle per-observation factors for reranking
    """
    if not observations:
        return 1.0

    mode_norm = (mode or "hard").strip().lower()
    combined = 1.0
    for obs in observations:
        min_frac, max_frac = compute_expected_damage_frac(
            move_bp=obs["move_bp"],
            move_type=obs["move_type"],
            move_category=obs["move_category"],
            attacker_stat=obs["attacker_stat"],
            attacker_level=obs["attacker_level"],
            attacker_types=obs["attacker_types"],
            attacker_boosts=obs.get("attacker_boosts", {}),
            attacker_status=obs.get("attacker_status"),
            attacker_ability=obs.get("attacker_ability", ""),
            attacker_item=obs.get("attacker_item", ""),
            defender_base_stats=species_base_stats,
            defender_level=species_level,
            defender_types=species_types,
            defender_boosts=obs.get("defender_boosts", {}),
            defender_ability=set_ability,
            defender_item=set_item,
            defender_hp_frac=obs.get("defender_hp_frac", 1.0),
            weather=obs.get("weather"),
            terrain=obs.get("terrain"),
            is_crit=obs.get("is_crit", False),
        )

        # If the set says the move should be immune but we observed damage,
        # that's a strong mismatch.
        if max_frac <= 0.0:
            observed = obs.get("observed_frac", 0.0)
            if observed > 0.01:
                combined *= 0.2
            continue

        observed = obs.get("observed_frac", 0.0)
        mid = (min_frac + max_frac) / 2.0
        spread = (max_frac - min_frac) / 2.0 + tolerance
        diff = abs(observed - mid)

        if mode_norm == "soft":
            if diff <= spread:
                score = 1.03
            elif diff <= 2 * spread:
                score = 0.99
            elif diff <= 3 * spread:
                score = 0.95
            else:
                score = 0.90
        else:
            if diff <= spread:
                score = 1.1
            elif diff <= 2 * spread:
                score = 0.85
            elif diff <= 3 * spread:
                score = 0.6
            else:
                score = 0.35

        score = max(per_obs_min, min(per_obs_max, score))

        combined *= score

    return max(final_min, min(final_max, combined))
