#!/usr/bin/env python3
"""
Relabel Oranguru search traces with a Foul Play-style oracle.

This expects traces that include:
- `action_labels`
- `fp_oracle_battle`

The teacher keeps Oranguru's on-policy state distribution, but replaces the
runtime labels with a stronger Foul Play sampling/search solve.
"""

from __future__ import annotations

import argparse
import concurrent.futures
from collections import Counter, defaultdict
import glob
import hashlib
import json
import math
import pickle
from pathlib import Path
import random
import sys
from typing import Iterable, Optional

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

FP_ROOT = PROJECT_ROOT / "third_party" / "foul-play"
if str(FP_ROOT) not in sys.path:
    sys.path.insert(0, str(FP_ROOT))

import constants  # noqa: E402
from constants import BattleType  # noqa: E402
from fp.battle import Battle as FPBattle, Battler, LastUsedMove, Move, Pokemon as FPPokemon, StatRange  # noqa: E402
from fp.search.poke_engine_helpers import battle_to_poke_engine_state  # noqa: E402
from fp.search.random_battles import prepare_random_battles  # noqa: E402
from fp.search.standard_battles import prepare_battles  # noqa: E402
from poke_engine import State as PokeEngineState, monte_carlo_tree_search  # noqa: E402

from training.search_assist_utils import safe_float


def _resolve_inputs(patterns: list[str]) -> list[str]:
    paths: list[str] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            paths.extend(matches)
        elif Path(pattern).exists():
            paths.append(pattern)
    return paths


def _iter_examples(path: str) -> Iterable[dict]:
    file_path = Path(path)
    if file_path.suffix == ".pkl":
        with file_path.open("rb") as handle:
            obj = pickle.load(handle)
        if isinstance(obj, list):
            for item in obj:
                if isinstance(item, dict):
                    yield item
        return

    with file_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except Exception:
                continue
            if isinstance(item, dict):
                yield item


def _stable_seed(base_seed: int, battle_id: str, turn: int) -> int:
    payload = f"{base_seed}|{battle_id}|{turn}".encode("utf-8", "ignore")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _restore_last_used_move(payload: object) -> LastUsedMove:
    if not isinstance(payload, dict):
        return LastUsedMove("", "", 0)
    return LastUsedMove(
        str(payload.get("pokemon_name", "") or ""),
        str(payload.get("move", "") or ""),
        int(payload.get("turn", 0) or 0),
    )


def _restore_move(payload: object) -> Optional[Move]:
    if not isinstance(payload, dict):
        return None
    name = str(payload.get("name", "") or "")
    if not name:
        return None
    try:
        move = Move(name)
    except Exception:
        return None
    move.disabled = bool(payload.get("disabled", False))
    move.can_z = bool(payload.get("can_z", False))
    move.current_pp = int(payload.get("current_pp", move.current_pp) or move.current_pp)
    move.max_pp = int(payload.get("max_pp", move.max_pp) or move.max_pp)
    return move


def _restore_pokemon(payload: object) -> Optional[FPPokemon]:
    if not isinstance(payload, dict):
        return None
    name = str(payload.get("name", "") or "")
    level = int(payload.get("level", 100) or 100)
    if not name:
        return None
    mon = FPPokemon(
        name=name,
        level=level,
        nature=str(payload.get("nature", "serious") or "serious"),
        evs=tuple(int(v) for v in list(payload.get("evs", (85,) * 6) or (85,) * 6)),
    )
    mon.nickname = payload.get("nickname", None)
    mon.base_name = str(payload.get("base_name", mon.base_name) or mon.base_name)
    mon.base_stats = dict(payload.get("base_stats", mon.base_stats) or mon.base_stats)
    mon.stats = dict(payload.get("stats", mon.stats) or mon.stats)
    mon.max_hp = int(payload.get("max_hp", mon.max_hp) or mon.max_hp)
    mon.hp = int(payload.get("hp", mon.hp) or mon.hp)
    mon.substitute_hit = bool(payload.get("substitute_hit", False))
    mon.ability = payload.get("ability", mon.ability)
    mon.types = tuple(str(v) for v in list(payload.get("types", mon.types) or mon.types))
    mon.item = payload.get("item", mon.item)
    mon.removed_item = payload.get("removed_item", None)
    mon.unknown_forme = bool(payload.get("unknown_forme", False))
    mon.moves_used_since_switch_in = set(
        str(v) for v in list(payload.get("moves_used_since_switch_in", []) or [])
    )
    mon.zoroark_disguised_as = payload.get("zoroark_disguised_as", None)
    mon.hp_at_switch_in = int(payload.get("hp_at_switch_in", mon.hp_at_switch_in) or mon.hp_at_switch_in)
    mon.status_at_switch_in = payload.get("status_at_switch_in", None)
    mon.terastallized = bool(payload.get("terastallized", False))
    mon.tera_type = payload.get("tera_type", None)
    mon.forme_changed = bool(payload.get("forme_changed", False))
    mon.original_ability = payload.get("original_ability", None)
    mon.fainted = bool(payload.get("fainted", False))
    mon.reviving = bool(payload.get("reviving", False))
    mon.moves = [m for m in (_restore_move(m) for m in list(payload.get("moves", []) or [])) if m is not None]
    mon.status = payload.get("status", None)
    mon.volatile_statuses = [str(v) for v in list(payload.get("volatile_statuses", []) or [])]
    mon.volatile_status_durations = defaultdict(
        lambda: 0,
        {str(k): int(v) for k, v in dict(payload.get("volatile_status_durations", {}) or {}).items()},
    )
    mon.boosts = defaultdict(
        lambda: 0,
        {str(k): int(v) for k, v in dict(payload.get("boosts", {}) or {}).items()},
    )
    mon.rest_turns = int(payload.get("rest_turns", 0) or 0)
    mon.sleep_turns = int(payload.get("sleep_turns", 0) or 0)
    mon.knocked_off = bool(payload.get("knocked_off", False))
    mon.can_mega_evo = bool(payload.get("can_mega_evo", False))
    mon.can_ultra_burst = bool(payload.get("can_ultra_burst", False))
    mon.can_dynamax = bool(payload.get("can_dynamax", False))
    mon.can_terastallize = bool(payload.get("can_terastallize", False))
    mon.is_mega = bool(payload.get("is_mega", False))
    mon.mega_name = payload.get("mega_name", None)
    mon.can_have_choice_item = bool(payload.get("can_have_choice_item", True))
    mon.item_inferred = bool(payload.get("item_inferred", False))
    mon.gen_3_consecutive_sleep_talks = int(
        payload.get("gen_3_consecutive_sleep_talks", 0) or 0
    )
    mon.impossible_items = set(str(v) for v in list(payload.get("impossible_items", []) or []))
    mon.impossible_abilities = set(
        str(v) for v in list(payload.get("impossible_abilities", []) or [])
    )
    speed_range = payload.get("speed_range", {}) or {}
    mon.speed_range = StatRange(
        min=int(speed_range.get("min", 0) or 0),
        max=float(speed_range.get("max", math.inf) or math.inf),
    )
    mon.hidden_power_possibilities = set(
        str(v) for v in list(payload.get("hidden_power_possibilities", []) or [])
    )
    return mon


def _restore_battler(payload: object) -> Battler:
    battler = Battler()
    if not isinstance(payload, dict):
        return battler
    battler.active = _restore_pokemon(payload.get("active"))
    battler.reserve = [
        mon
        for mon in (_restore_pokemon(p) for p in list(payload.get("reserve", []) or []))
        if mon is not None
    ]
    battler.side_conditions = defaultdict(
        lambda: 0,
        {str(k): int(v) for k, v in dict(payload.get("side_conditions", {}) or {}).items()},
    )
    battler.name = payload.get("name", None)
    battler.trapped = bool(payload.get("trapped", False))
    battler.baton_passing = bool(payload.get("baton_passing", False))
    battler.shed_tailing = bool(payload.get("shed_tailing", False))
    wish = list(payload.get("wish", [0, 0]) or [0, 0])
    battler.wish = (int(wish[0] or 0), int(wish[1] or 0)) if len(wish) >= 2 else (0, 0)
    future_sight = list(payload.get("future_sight", [0, ""]) or [0, ""])
    battler.future_sight = (
        int(future_sight[0] or 0),
        str(future_sight[1] or ""),
    ) if len(future_sight) >= 2 else (0, "")
    battler.account_name = payload.get("account_name", None)
    battler.team_dict = payload.get("team_dict", None)
    battler.last_selected_move = _restore_last_used_move(payload.get("last_selected_move"))
    battler.last_used_move = _restore_last_used_move(payload.get("last_used_move"))
    return battler


def _restore_battle(payload: object) -> Optional[FPBattle]:
    if not isinstance(payload, dict):
        return None
    battle_tag = str(payload.get("battle_tag", "teacher") or "teacher")
    battle = FPBattle(battle_tag)
    battle.turn = int(payload.get("turn", 0) or 0)
    battle.weather = payload.get("weather", None)
    battle.weather_turns_remaining = int(
        payload.get("weather_turns_remaining", -1) or -1
    )
    battle.weather_source = str(payload.get("weather_source", "") or "")
    battle.field = payload.get("field", None)
    battle.field_turns_remaining = int(payload.get("field_turns_remaining", 0) or 0)
    battle.trick_room = bool(payload.get("trick_room", False))
    battle.trick_room_turns_remaining = int(
        payload.get("trick_room_turns_remaining", 0) or 0
    )
    battle.gravity = bool(payload.get("gravity", False))
    battle.team_preview = bool(payload.get("team_preview", False))
    battle.started = bool(payload.get("started", False))
    battle.rqid = payload.get("rqid", None)
    battle.force_switch = bool(payload.get("force_switch", False))
    battle.wait = bool(payload.get("wait", False))
    battle_type = payload.get("battle_type", None)
    if battle_type:
        battle.battle_type = BattleType(str(battle_type))
    battle.pokemon_format = payload.get("pokemon_format", None)
    battle.generation = payload.get("generation", None)
    battle.time_remaining = payload.get("time_remaining", None)
    battle.user = _restore_battler(payload.get("user"))
    battle.opponent = _restore_battler(payload.get("opponent"))
    return battle


def _mcts_from_state(task: tuple[str, int]):
    state_str, search_ms = task
    state = PokeEngineState.from_string(state_str)
    return monte_carlo_tree_search(state, search_ms)


def _prepare_teacher_battles(battle: FPBattle, num_battles: int) -> list[tuple[str, float]]:
    if battle.battle_type in {BattleType.RANDOM_BATTLE, BattleType.BATTLE_FACTORY}:
        prepared = prepare_random_battles(battle, num_battles)
    elif battle.battle_type == BattleType.STANDARD_BATTLE:
        prepared = prepare_battles(battle, num_battles)
    else:
        raise ValueError(f"Unsupported battle type for FP oracle: {battle.battle_type}")
    return [(battle_to_poke_engine_state(b).to_string(), float(chance)) for b, chance in prepared]


def _relabel_example(
    ex: dict,
    search_ms: int,
    num_battles: int,
    seed: int,
    value_mode: str,
    executor,
) -> dict | None:
    action_mask = ex.get("action_mask")
    action_features = ex.get("action_features")
    action_labels = ex.get("action_labels")
    board_features = ex.get("board_features")
    fp_oracle_battle = ex.get("fp_oracle_battle")

    if not isinstance(action_mask, list) or not action_mask:
        return None
    if not isinstance(action_features, list) or len(action_features) != len(action_mask):
        return None
    if not isinstance(action_labels, list) or len(action_labels) != len(action_mask):
        return None
    if not isinstance(board_features, list) or not board_features:
        return None

    battle = _restore_battle(fp_oracle_battle)
    if battle is None:
        return None
    battle_id = str(ex.get("battle_id", "") or "")
    turn = int(ex.get("turn", 0) or 0)
    random.seed(_stable_seed(seed, battle_id, turn))
    try:
        states_and_weights = _prepare_teacher_battles(battle, max(1, num_battles))
    except Exception:
        return None
    if not states_and_weights:
        return None

    tasks = [(state_str, search_ms) for state_str, _ in states_and_weights]
    try:
        if executor is not None and len(tasks) > 1:
            results = list(executor.map(_mcts_from_state, tasks))
        else:
            results = [_mcts_from_state(task) for task in tasks]
    except Exception:
        return None

    choice_to_idx = {}
    for idx, label in enumerate(action_labels):
        if action_mask[idx] and isinstance(label, str) and label:
            choice_to_idx[label] = idx

    policy = [0.0] * len(action_mask)
    teacher_value01 = 0.0
    teacher_visits = 0.0
    samples_used = 0

    for (_state_str, sample_weight), res in zip(states_and_weights, results):
        total_visits = max(0.0, float(getattr(res, "total_visits", 0) or 0.0))
        side_one = list(getattr(res, "side_one", []) or [])
        if total_visits <= 0 or not side_one:
            continue
        samples_used += 1
        teacher_visits += total_visits
        world_value01 = 0.0
        for opt in side_one:
            choice = str(getattr(opt, "move_choice", "") or "")
            visits = max(0.0, float(getattr(opt, "visits", 0) or 0.0))
            if visits <= 0:
                continue
            prob = visits / total_visits
            idx = choice_to_idx.get(choice)
            if idx is not None:
                policy[idx] += sample_weight * prob
            total_score = float(getattr(opt, "total_score", 0.0) or 0.0)
            avg_score = total_score / visits if visits > 0 else 0.5
            avg_score = max(0.0, min(1.0, avg_score))
            world_value01 += prob * avg_score
        teacher_value01 += sample_weight * world_value01

    policy_total = sum(policy[i] for i, ok in enumerate(action_mask) if ok)
    if policy_total <= 0:
        legal = [i for i, ok in enumerate(action_mask) if ok]
        if not legal:
            return None
        uniform = 1.0 / float(len(legal))
        policy = [uniform if ok else 0.0 for ok in action_mask]
    else:
        policy = [(v / policy_total) if action_mask[i] else 0.0 for i, v in enumerate(policy)]

    fp_value_target = max(-1.0, min(1.0, 2.0 * teacher_value01 - 1.0))
    legal_probs = [policy[i] for i, ok in enumerate(action_mask) if ok and policy[i] > 0.0]
    teacher_top1_prob = max(legal_probs) if legal_probs else 0.0
    teacher_entropy = -sum(p * math.log(p) for p in legal_probs)
    orig_value_target = safe_float(ex.get("value_target", 0.0), 0.0)
    if value_mode == "orig":
        value_target = orig_value_target
    else:
        value_target = fp_value_target
    return {
        "battle_id": battle_id,
        "turn": turn,
        "rating": ex.get("rating"),
        "board_features": [safe_float(v, 0.0) for v in board_features],
        "action_features": action_features,
        "action_labels": action_labels,
        "action_mask": [bool(v) for v in action_mask],
        "policy_target": policy,
        "value_target": value_target,
        "weight": float(ex.get("weight", 1.0) or 1.0),
        "source": str(ex.get("source", "")),
        "tag": str(ex.get("tag", "")),
        "chosen_action": ex.get("chosen_action"),
        "teacher_source": "fp_oracle",
        "teacher_search_ms": int(search_ms),
        "teacher_num_battles": int(num_battles),
        "teacher_samples_used": int(samples_used),
        "teacher_worlds_used": int(samples_used),
        "teacher_total_visits": float(teacher_visits),
        "teacher_top1_prob": float(teacher_top1_prob),
        "teacher_entropy": float(teacher_entropy),
        "teacher_value01": float(teacher_value01),
        "fp_value_target": fp_value_target,
        "orig_value_target": orig_value_target,
        "value_target_source": value_mode,
        "orig_policy_confidence": safe_float(ex.get("policy_confidence", 0.0), 0.0),
        "orig_policy_threshold": safe_float(ex.get("policy_threshold", 0.0), 0.0),
        "selection_path": str(ex.get("selection_path", "")),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Relabel search traces with a Foul Play oracle.")
    parser.add_argument("--input", action="append", default=[], help="JSONL/PKL path or glob. Repeatable.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-out", default="")
    parser.add_argument("--search-ms", type=int, default=1200)
    parser.add_argument("--num-battles", type=int, default=12)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--value-mode", choices=["fp", "orig"], default="fp")
    parser.add_argument("--min-turn", type=int, default=1)
    parser.add_argument("--keep-lowconf-only", action="store_true")
    parser.add_argument("--keep-nonfallback-only", action="store_true")
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=100)
    args = parser.parse_args()

    paths = _resolve_inputs(args.input)
    if not paths:
        raise SystemExit("No input files matched.")

    counters = Counter()
    rows: list[dict] = []
    executor = None
    if args.workers > 1:
        executor = concurrent.futures.ProcessPoolExecutor(max_workers=args.workers)

    try:
        for path in paths:
            for ex in _iter_examples(path):
                counters["rows_seen"] += 1
                turn = int(ex.get("turn", 0) or 0)
                if turn < args.min_turn:
                    counters["drop_min_turn"] += 1
                    continue
                if args.keep_nonfallback_only and str(ex.get("selection_path", "") or "").startswith("fallback"):
                    counters["drop_fallback"] += 1
                    continue
                confidence = safe_float(ex.get("policy_confidence", 0.0), 0.0)
                threshold = safe_float(ex.get("policy_threshold", 0.0), 0.0)
                if args.keep_lowconf_only and confidence >= threshold:
                    counters["drop_not_lowconf"] += 1
                    continue
                if not ex.get("fp_oracle_battle"):
                    counters["drop_missing_fp_oracle"] += 1
                    continue
                relabeled = _relabel_example(
                    ex,
                    search_ms=max(1, args.search_ms),
                    num_battles=max(1, args.num_battles),
                    seed=args.seed,
                    value_mode=args.value_mode,
                    executor=executor,
                )
                if relabeled is None:
                    counters["drop_unrelabelable"] += 1
                    continue
                rows.append(relabeled)
                counters["kept"] += 1
                if args.progress_every > 0 and counters["kept"] % args.progress_every == 0:
                    print(f"Relabeled {counters['kept']} rows", flush=True)
                if args.max_rows > 0 and counters["kept"] >= args.max_rows:
                    break
            if args.max_rows > 0 and counters["kept"] >= args.max_rows:
                break
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as handle:
        pickle.dump(rows, handle)

    print(f"Wrote {len(rows)} FP-oracle teacher rows -> {out_path}")
    print(f"Stats: {dict(counters)}")

    if args.summary_out:
        summary = {
            "input_files": paths,
            "rows_out": len(rows),
            "stats": dict(counters),
            "search_ms": int(args.search_ms),
            "num_battles": int(args.num_battles),
            "workers": int(args.workers),
            "seed": int(args.seed),
            "value_mode": args.value_mode,
        }
        summary_path = Path(args.summary_out)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        print(f"Summary -> {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
