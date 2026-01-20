#!/usr/bin/env python3
"""
Build move-correction trajectories from switch opportunity logs.

Targets bad-matchup stays by relabeling to the best available damaging move
based on move feature effectiveness/expected damage.
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.features import STATUS_MOVE_EFFECTS, normalize_name

MOVE_OFFSET = 16
MOVE_STRIDE = 12
MOVE_AVAIL_IDX = 0
MOVE_EFF_IDX = 1
MOVE_POWER_IDX = 2
MOVE_PHYSICAL_IDX = 4
MOVE_SPECIAL_IDX = 5
MOVE_STATUS_IDX = 6
MOVE_PRIORITY_POS_IDX = 7
MOVE_EFFECT_VALUE_IDX = 9
MOVE_EXPECTED_IDX = 11

BASE_OPP_HP_IDX = 2
FEATURE_SPEED_START = 124
FEATURE_ITEM_START = 156
FEATURE_FIELD_START = 172
FEATURE_BOOST_START = 220
FEATURE_STATUS_START = 244
FEATURE_HAZARD_START = 260

OPP_STATUS_ANY_OFFSET = 13
OPP_DEF_BOOST_OFFSET = 8
OPP_SPD_BOOST_OFFSET = 10
OPP_PRIORITY_BLOCK_OFFSET = 5
PSYCHIC_TERRAIN_OFFSET = 6
OPP_ITEM_START = 6
OPP_ITEM_END = 10

MY_ATK_BOOST_OFFSET = 0
MY_DEF_BOOST_OFFSET = 1
MY_SPA_BOOST_OFFSET = 2
MY_SPD_BOOST_OFFSET = 3
MY_SPE_BOOST_OFFSET = 4

SETUP_MOVE_STATS = {
    "swordsdance": ("atk",),
    "nastyplot": ("spa",),
    "dragondance": ("atk", "spe"),
    "calmmind": ("spa", "spd"),
    "bulkup": ("atk", "def"),
    "irondefense": ("def",),
    "shellsmash": ("atk", "spa", "spe"),
    "coil": ("atk", "def"),
    "shiftgear": ("atk", "spe"),
    "agility": ("spe",),
    "rockpolish": ("spe",),
    "autotomize": ("spe",),
    "tailwind": ("spe",),
    "quiverdance": ("spa", "spd", "spe"),
    "workup": ("atk", "spa"),
}

BOOST_OFFSETS = {
    "atk": MY_ATK_BOOST_OFFSET,
    "def": MY_DEF_BOOST_OFFSET,
    "spa": MY_SPA_BOOST_OFFSET,
    "spd": MY_SPD_BOOST_OFFSET,
    "spe": MY_SPE_BOOST_OFFSET,
}


def infer_tag(path: Path) -> str:
    stem = path.stem
    prefix = "switch_opportunities_"
    if stem.startswith(prefix):
        return stem[len(prefix):]
    return stem


def iter_log_paths(log_path: str | None, log_dir: str | None) -> list[Path]:
    paths: list[Path] = []
    if log_dir:
        base = Path(log_dir)
        paths.extend(sorted(base.glob("switch_opportunities_*.jsonl")))
    if log_path:
        paths.append(Path(log_path))
    return paths


def _safe_feature(features: list[float], idx: int, default: float = 0.0) -> float:
    if idx < 0 or idx >= len(features):
        return default
    return features[idx]


def _boost_stage(features: list[float], offset: int) -> float:
    return _safe_feature(features, FEATURE_BOOST_START + offset) * 6.0


def _chosen_move_id(data: dict) -> str | None:
    chosen = data.get("chosen")
    if not isinstance(chosen, str) or ":" not in chosen:
        return None
    prefix, raw = chosen.split(":", 1)
    if prefix not in {"move", "tera"}:
        return None
    return normalize_name(raw)


def _opponent_item_known(features: list[float]) -> bool:
    base = FEATURE_ITEM_START + OPP_ITEM_START
    for idx in range(base, base + (OPP_ITEM_END - OPP_ITEM_START)):
        if _safe_feature(features, idx) > 0.0:
            return True
    return False


def _priority_blocked(features: list[float]) -> bool:
    blocked = _safe_feature(features, FEATURE_SPEED_START + OPP_PRIORITY_BLOCK_OFFSET) >= 0.5
    psychic = _safe_feature(features, FEATURE_FIELD_START + PSYCHIC_TERRAIN_OFFSET) >= 0.5
    return blocked or psychic


def _collect_moves(features: list[float], mask: list[bool]) -> list[dict]:
    moves = []
    for i in range(4):
        if i >= len(mask) or not mask[i]:
            continue
        base = MOVE_OFFSET + i * MOVE_STRIDE
        if base + MOVE_EXPECTED_IDX >= len(features):
            continue
        available = features[base + MOVE_AVAIL_IDX]
        if available <= 0:
            continue
        eff = features[base + MOVE_EFF_IDX] * 4.0
        power = features[base + MOVE_POWER_IDX] * 150.0
        expected = features[base + MOVE_EXPECTED_IDX]
        is_physical = features[base + MOVE_PHYSICAL_IDX] >= 0.5
        is_special = features[base + MOVE_SPECIAL_IDX] >= 0.5
        status_flag = features[base + MOVE_STATUS_IDX] >= 0.5
        # Some logs have status flag unset; treat power<=0 as status-like.
        is_status = status_flag or power <= 0.0
        priority = features[base + MOVE_PRIORITY_POS_IDX] > 0
        effect_value = features[base + MOVE_EFFECT_VALUE_IDX]
        moves.append({
            "idx": i,
            "eff": eff,
            "power": power,
            "expected": expected,
            "is_physical": is_physical,
            "is_special": is_special,
            "is_status": is_status,
            "priority": priority,
            "effect_value": effect_value,
        })
    return moves


def _best_by_expected(
    moves: list[dict],
    eff_min: float,
    expected_min: float,
    predicate,
) -> dict | None:
    best = None
    for move in moves:
        if not predicate(move):
            continue
        if eff_min and move["eff"] < eff_min:
            continue
        if move["expected"] < expected_min:
            continue
        if best is None or move["expected"] > best["expected"]:
            best = move
    return best


def _best_status_move(moves: list[dict], min_value: float) -> dict | None:
    best = None
    for move in moves:
        if not move["is_status"]:
            continue
        if move["effect_value"] < min_value:
            continue
        if best is None or move["effect_value"] > best["effect_value"]:
            best = move
    return best


def pick_correction_action(
    features: list[float],
    mask: list[bool],
    eff_min: float,
    expected_min: float,
    rule_priority: bool,
    priority_hp_threshold: float,
    priority_expected_ratio: float,
    rule_knockoff: bool,
    knockoff_effect_min: float,
    knockoff_expected_ratio: float,
    rule_defensive: bool,
    defensive_max_damaging: int,
    status_value_min: float,
    rule_boost: bool,
    opp_boost_threshold: float,
    boost_mismatch_ratio: float,
) -> tuple[int | None, dict | None, str]:
    moves = _collect_moves(features, mask)
    if not moves:
        return None, None, "no_moves"

    def damaging(move: dict) -> bool:
        return not move["is_status"] and move["power"] > 0

    best_damage = _best_by_expected(moves, eff_min, expected_min, damaging)
    if best_damage is None:
        return None, None, "no_damage"

    opp_hp = _safe_feature(features, BASE_OPP_HP_IDX)
    if rule_priority and opp_hp <= priority_hp_threshold and not _priority_blocked(features):
        best_priority = _best_by_expected(
            moves,
            eff_min,
            expected_min,
            lambda m: damaging(m) and m["priority"],
        )
        if best_priority and best_priority["expected"] >= priority_expected_ratio * best_damage["expected"]:
            return best_priority["idx"], best_priority, "priority_finish"

    if rule_knockoff and _opponent_item_known(features):
        best_disrupt = _best_by_expected(
            moves,
            eff_min,
            expected_min,
            lambda m: damaging(m) and m["effect_value"] >= knockoff_effect_min,
        )
        if best_disrupt and best_disrupt["expected"] >= knockoff_expected_ratio * best_damage["expected"]:
            return best_disrupt["idx"], best_disrupt, "knockoff_value"

    if rule_boost:
        opp_def_boost = _safe_feature(features, FEATURE_BOOST_START + OPP_DEF_BOOST_OFFSET) * 6.0
        opp_spd_boost = _safe_feature(features, FEATURE_BOOST_START + OPP_SPD_BOOST_OFFSET) * 6.0
        best_physical = _best_by_expected(
            moves, eff_min, expected_min, lambda m: damaging(m) and m["is_physical"]
        )
        best_special = _best_by_expected(
            moves, eff_min, expected_min, lambda m: damaging(m) and m["is_special"]
        )
        if opp_def_boost >= opp_boost_threshold and best_special:
            if best_physical is None or best_special["expected"] >= boost_mismatch_ratio * best_physical["expected"]:
                return best_special["idx"], best_special, "boost_mismatch_special"
        if opp_spd_boost >= opp_boost_threshold and best_physical:
            if best_special is None or best_physical["expected"] >= boost_mismatch_ratio * best_special["expected"]:
                return best_physical["idx"], best_physical, "boost_mismatch_physical"

    if rule_defensive:
        damaging_count = sum(1 for move in moves if damaging(move))
        has_status = any(move["is_status"] for move in moves)
        opp_statused = _safe_feature(features, FEATURE_STATUS_START + OPP_STATUS_ANY_OFFSET) >= 0.5
        if damaging_count <= defensive_max_damaging and has_status and not opp_statused:
            best_status = _best_status_move(moves, status_value_min)
            if best_status:
                return best_status["idx"], best_status, "defensive_status"

    return best_damage["idx"], best_damage, "best_damage"


def action_to_move_index(action_idx: int | None) -> int | None:
    if action_idx is None:
        return None
    if 0 <= action_idx < 4:
        return action_idx
    if 9 <= action_idx < 13:
        return action_idx - 9
    return None


def get_move_features(features: list[float], move_idx: int) -> tuple[float, float, float]:
    base = MOVE_OFFSET + move_idx * MOVE_STRIDE
    available = features[base + MOVE_AVAIL_IDX]
    eff = features[base + MOVE_EFF_IDX] * 4.0
    expected = features[base + MOVE_EXPECTED_IDX]
    is_status = features[base + MOVE_STATUS_IDX]
    return available, eff, expected, is_status


def build_trajectories(
    paths: list[Path],
    weight: float,
    matchup_threshold: float,
    eff_min: float,
    expected_min: float,
    skip_if_good_switch: bool,
    min_good_switch_delta: float,
    filter_chosen: bool,
    chosen_eff_max: float,
    chosen_expected_max: float,
    require_chosen_damaging: bool,
    require_chosen_action: bool,
    max_best_delta: float | None,
    rule_setup_cap: bool,
    setup_atk_cap: float,
    setup_spa_cap: float,
    setup_def_cap: float,
    setup_spd_cap: float,
    setup_spe_cap: float,
    rule_priority: bool,
    priority_hp_threshold: float,
    priority_expected_ratio: float,
    rule_knockoff: bool,
    knockoff_effect_min: float,
    knockoff_expected_ratio: float,
    rule_defensive: bool,
    defensive_max_damaging: int,
    status_value_min: float,
    rule_boost: bool,
    opp_boost_threshold: float,
    boost_mismatch_ratio: float,
    limit: int,
) -> tuple[list[dict], dict]:
    trajectories: list[dict] = []
    stats = {
        "lines": 0,
        "kept": 0,
        "skipped_no_features": 0,
        "skipped_not_stay": 0,
        "skipped_not_bad_matchup": 0,
        "skipped_good_switch": 0,
        "skipped_no_candidate": 0,
        "skipped_illegal_action": 0,
        "skipped_no_chosen": 0,
        "skipped_chosen_ok": 0,
        "skipped_best_delta": 0,
        "skipped_same_action": 0,
        "picked_setup_cap": 0,
        "picked_priority": 0,
        "picked_knockoff": 0,
        "picked_defensive": 0,
        "picked_boost": 0,
        "picked_best": 0,
    }

    for path in paths:
        tag = infer_tag(path)
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if limit and stats["kept"] >= limit:
                    return trajectories, stats
                line = line.strip()
                if not line:
                    continue
                stats["lines"] += 1
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if data.get("decision") != "stay":
                    stats["skipped_not_stay"] += 1
                    continue
                matchup = data.get("matchup")
                if matchup is None or matchup > matchup_threshold:
                    stats["skipped_not_bad_matchup"] += 1
                    continue
                if skip_if_good_switch:
                    best_delta = data.get("best_delta")
                    if best_delta is not None and best_delta >= min_good_switch_delta:
                        stats["skipped_good_switch"] += 1
                        continue
                if max_best_delta is not None:
                    best_delta = data.get("best_delta")
                    if best_delta is None or best_delta > max_best_delta:
                        stats["skipped_best_delta"] += 1
                        continue
                features = data.get("features")
                mask = data.get("mask")
                if features is None or mask is None:
                    stats["skipped_no_features"] += 1
                    continue
                needs_chosen = filter_chosen or require_chosen_action or rule_setup_cap
                if needs_chosen:
                    chosen_action = data.get("chosen_action")
                    move_idx = action_to_move_index(chosen_action)
                    if move_idx is None:
                        stats["skipped_no_chosen"] += 1
                        continue
                    if MOVE_OFFSET + move_idx * MOVE_STRIDE + MOVE_EXPECTED_IDX >= len(features):
                        stats["skipped_no_features"] += 1
                        continue
                    available, eff, expected, is_status = get_move_features(features, move_idx)
                    if available <= 0:
                        stats["skipped_no_chosen"] += 1
                        continue
                    if require_chosen_damaging and is_status >= 0.5:
                        stats["skipped_no_chosen"] += 1
                        continue
                    if filter_chosen:
                        if eff > chosen_eff_max and expected > chosen_expected_max:
                            stats["skipped_chosen_ok"] += 1
                            continue
                if rule_setup_cap:
                    move_id = _chosen_move_id(data)
                    setup_stats = SETUP_MOVE_STATS.get(move_id)
                    if setup_stats:
                        caps = {
                            "atk": setup_atk_cap,
                            "spa": setup_spa_cap,
                            "def": setup_def_cap,
                            "spd": setup_spd_cap,
                            "spe": setup_spe_cap,
                        }
                        over_cap = True
                        for stat in setup_stats:
                            offset = BOOST_OFFSETS.get(stat)
                            if offset is None:
                                continue
                            if _boost_stage(features, offset) < caps[stat]:
                                over_cap = False
                                break
                        if over_cap:
                            moves = _collect_moves(features, mask)
                            best_damage = _best_by_expected(
                                moves,
                                eff_min,
                                expected_min,
                                lambda m: not m["is_status"] and m["power"] > 0,
                            )
                            if best_damage is not None:
                                best_action = best_damage["idx"]
                                reason = "setup_overcap"
                            else:
                                stats["skipped_no_candidate"] += 1
                                continue
                        else:
                            best_action = None
                            reason = ""
                    else:
                        best_action = None
                        reason = ""
                else:
                    best_action = None
                    reason = ""
                if best_action is None:
                    best_action, _meta, reason = pick_correction_action(
                        features,
                        mask,
                        eff_min=eff_min,
                        expected_min=expected_min,
                        rule_priority=rule_priority,
                        priority_hp_threshold=priority_hp_threshold,
                        priority_expected_ratio=priority_expected_ratio,
                        rule_knockoff=rule_knockoff,
                        knockoff_effect_min=knockoff_effect_min,
                        knockoff_expected_ratio=knockoff_expected_ratio,
                        rule_defensive=rule_defensive,
                        defensive_max_damaging=defensive_max_damaging,
                        status_value_min=status_value_min,
                        rule_boost=rule_boost,
                        opp_boost_threshold=opp_boost_threshold,
                        boost_mismatch_ratio=boost_mismatch_ratio,
                    )
                if best_action is None:
                    stats["skipped_no_candidate"] += 1
                    continue
                if best_action >= len(mask) or not mask[best_action]:
                    stats["skipped_illegal_action"] += 1
                    continue
                if filter_chosen or require_chosen_action:
                    if move_idx == best_action:
                        stats["skipped_same_action"] += 1
                        continue

                if reason == "setup_overcap":
                    stats["picked_setup_cap"] += 1
                elif reason == "priority_finish":
                    stats["picked_priority"] += 1
                elif reason == "knockoff_value":
                    stats["picked_knockoff"] += 1
                elif reason == "defensive_status":
                    stats["picked_defensive"] += 1
                elif reason.startswith("boost_mismatch"):
                    stats["picked_boost"] += 1
                else:
                    stats["picked_best"] += 1

                trajectories.append({
                    "features": [features],
                    "masks": [mask],
                    "actions": [best_action],
                    "rewards": [0.0],
                    "dones": [True],
                    "weight": weight,
                    "tag": f"move_correction_{tag}",
                })
                stats["kept"] += 1

    return trajectories, stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Build move-correction trajectories")
    parser.add_argument("--log", type=str, default=None,
                        help="Single opportunity log JSONL")
    parser.add_argument("--log-dir", type=str, default=None,
                        help="Directory containing switch_opportunities_*.jsonl")
    parser.add_argument("--output", type=str, required=True,
                        help="Output pickle path for trajectories")
    parser.add_argument("--weight", type=float, default=1.5,
                        help="Trajectory weight for move corrections")
    parser.add_argument("--matchup-threshold", type=float, default=-0.2,
                        help="Matchup threshold to consider bad")
    parser.add_argument("--eff-min", type=float, default=0.5,
                        help="Minimum effectiveness multiplier to keep a move")
    parser.add_argument("--expected-min", type=float, default=0.05,
                        help="Minimum expected damage feature to keep a move")
    parser.add_argument("--skip-if-good-switch", action="store_true",
                        help="Skip corrections when a good switch exists")
    parser.add_argument("--min-good-switch-delta", type=float, default=0.3,
                        help="Best switch delta threshold")
    parser.add_argument("--filter-chosen", action="store_true",
                        help="Only correct when chosen move looks weak")
    parser.add_argument("--chosen-eff-max", type=float, default=0.5,
                        help="Chosen move effectiveness max to trigger correction")
    parser.add_argument("--chosen-expected-max", type=float, default=0.08,
                        help="Chosen expected damage max to trigger correction")
    parser.add_argument("--require-chosen-damaging", action="store_true",
                        help="Require chosen move to be damaging (not status)")
    parser.add_argument("--require-chosen-action", action="store_true",
                        help="Skip lines without chosen_action")
    parser.add_argument("--max-best-delta", type=float, default=None,
                        help="Only keep entries with best_delta <= this")
    parser.add_argument("--rule-setup-cap", action="store_true",
                        help="Relabel setup/speed moves if boosts already high")
    parser.add_argument("--setup-atk-cap", type=float, default=2.0,
                        help="Atk boost cap for setup spam correction")
    parser.add_argument("--setup-spa-cap", type=float, default=2.0,
                        help="SpA boost cap for setup spam correction")
    parser.add_argument("--setup-def-cap", type=float, default=3.0,
                        help="Def boost cap for setup spam correction")
    parser.add_argument("--setup-spd-cap", type=float, default=3.0,
                        help="SpD boost cap for setup spam correction")
    parser.add_argument("--setup-spe-cap", type=float, default=2.0,
                        help="Spe boost cap for setup spam correction")
    parser.add_argument("--rule-priority", action="store_true",
                        help="Prefer priority finishers when opp is low HP")
    parser.add_argument("--priority-hp-threshold", type=float, default=0.25,
                        help="Opponent HP threshold for priority finishes")
    parser.add_argument("--priority-expected-ratio", type=float, default=0.8,
                        help="Priority expected ratio vs best damage")
    parser.add_argument("--rule-knockoff", action="store_true",
                        help="Prefer disrupt moves when opp item is known")
    parser.add_argument("--knockoff-effect-min", type=float, default=0.35,
                        help="Min effect value (normalized) for knockoff-like moves")
    parser.add_argument("--knockoff-expected-ratio", type=float, default=0.8,
                        help="Disrupt expected ratio vs best damage")
    parser.add_argument("--rule-defensive", action="store_true",
                        help="Prefer status/hazard on defensive roles")
    parser.add_argument("--defensive-max-damaging", type=int, default=2,
                        help="Max damaging moves to consider a defensive role")
    parser.add_argument("--status-value-min", type=float, default=0.35,
                        help="Min status effect value (normalized)")
    parser.add_argument("--rule-boost", action="store_true",
                        help="Correct move category vs opponent boosts")
    parser.add_argument("--opp-boost-threshold", type=float, default=1.0,
                        help="Opponent boost stages to trigger boost mismatch")
    parser.add_argument("--boost-mismatch-ratio", type=float, default=1.1,
                        help="Expected ratio needed to switch category")
    parser.add_argument("--limit", type=int, default=0,
                        help="Optional max number of trajectories to keep (0 = no limit)")
    args = parser.parse_args()

    if not args.log and not args.log_dir:
        parser.error("--log or --log-dir is required")

    paths = iter_log_paths(args.log, args.log_dir)
    if not paths:
        print("No log files found.")
        return 1

    limit = args.limit if args.limit and args.limit > 0 else 0
    trajectories, stats = build_trajectories(
        paths,
        weight=args.weight,
        matchup_threshold=args.matchup_threshold,
        eff_min=args.eff_min,
        expected_min=args.expected_min,
        skip_if_good_switch=args.skip_if_good_switch,
        min_good_switch_delta=args.min_good_switch_delta,
        filter_chosen=args.filter_chosen,
        chosen_eff_max=args.chosen_eff_max,
        chosen_expected_max=args.chosen_expected_max,
        require_chosen_damaging=args.require_chosen_damaging,
        require_chosen_action=args.require_chosen_action,
        max_best_delta=args.max_best_delta,
        rule_setup_cap=args.rule_setup_cap,
        setup_atk_cap=args.setup_atk_cap,
        setup_spa_cap=args.setup_spa_cap,
        setup_def_cap=args.setup_def_cap,
        setup_spd_cap=args.setup_spd_cap,
        setup_spe_cap=args.setup_spe_cap,
        rule_priority=args.rule_priority,
        priority_hp_threshold=args.priority_hp_threshold,
        priority_expected_ratio=args.priority_expected_ratio,
        rule_knockoff=args.rule_knockoff,
        knockoff_effect_min=args.knockoff_effect_min,
        knockoff_expected_ratio=args.knockoff_expected_ratio,
        rule_defensive=args.rule_defensive,
        defensive_max_damaging=args.defensive_max_damaging,
        status_value_min=args.status_value_min,
        rule_boost=args.rule_boost,
        opp_boost_threshold=args.opp_boost_threshold,
        boost_mismatch_ratio=args.boost_mismatch_ratio,
        limit=limit,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        pickle.dump(trajectories, handle)

    print("Saved trajectories:", len(trajectories))
    print("Output:", output_path)
    for key, value in stats.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
