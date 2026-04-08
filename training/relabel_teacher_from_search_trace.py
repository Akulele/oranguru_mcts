#!/usr/bin/env python3
"""
Relabel Oranguru search-trace states with a stronger search teacher.

This expects traces that include:
- `action_labels`
- `world_candidates[*].state_str`

It outputs search-assist-style examples with teacher `policy_target` and
teacher `value_target`, so existing policy/value trainers can consume them.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import glob
import json
import math
import pickle
from collections import Counter
from pathlib import Path
import sys
from typing import Iterable

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from poke_engine import State as PokeEngineState, monte_carlo_tree_search

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


def _teacher_search(task: tuple[str, int]):
    state_str, search_ms = task
    state = PokeEngineState.from_string(state_str)
    return monte_carlo_tree_search(state, search_ms)


def _relabel_example(ex: dict, search_ms: int, max_worlds: int, executor) -> dict | None:
    action_mask = ex.get("action_mask")
    action_features = ex.get("action_features")
    action_labels = ex.get("action_labels")
    board_features = ex.get("board_features")
    world_candidates = ex.get("world_candidates") or []

    if not isinstance(action_mask, list) or not action_mask:
        return None
    if not isinstance(action_features, list) or len(action_features) != len(action_mask):
        return None
    if not isinstance(action_labels, list) or len(action_labels) != len(action_mask):
        return None
    if not isinstance(board_features, list) or not board_features:
        return None
    if not isinstance(world_candidates, list) or not world_candidates:
        return None

    valid_worlds = []
    for world in world_candidates:
        if not isinstance(world, dict):
            continue
        state_str = str(world.get("state_str", "") or "")
        if not state_str:
            continue
        weight = max(0.0, safe_float(world.get("sample_weight", 0.0), 0.0))
        valid_worlds.append((weight, state_str))
    if not valid_worlds:
        return None

    valid_worlds.sort(key=lambda item: item[0], reverse=True)
    if max_worlds > 0:
        valid_worlds = valid_worlds[:max_worlds]
    total_weight = sum(weight for weight, _ in valid_worlds)
    if total_weight <= 0:
        return None
    valid_worlds = [(weight / total_weight, state_str) for weight, state_str in valid_worlds]

    tasks = [(state_str, search_ms) for _, state_str in valid_worlds]
    try:
        if executor is not None and len(tasks) > 1:
            results = list(executor.map(_teacher_search, tasks))
        else:
            results = [_teacher_search(task) for task in tasks]
    except Exception:
        return None

    choice_to_idx = {}
    for idx, label in enumerate(action_labels):
        if action_mask[idx] and isinstance(label, str) and label:
            choice_to_idx[label] = idx

    policy = [0.0] * len(action_mask)
    teacher_value01 = 0.0
    teacher_visits = 0.0
    worlds_used = 0

    for (weight, _state_str), res in zip(valid_worlds, results):
        total_visits = max(0.0, float(getattr(res, "total_visits", 0) or 0.0))
        side_one = list(getattr(res, "side_one", []) or [])
        if total_visits <= 0 or not side_one:
            continue
        worlds_used += 1
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
                policy[idx] += weight * prob
            total_score = float(getattr(opt, "total_score", 0.0) or 0.0)
            avg_score = total_score / visits if visits > 0 else 0.5
            avg_score = max(0.0, min(1.0, avg_score))
            world_value01 += prob * avg_score
        teacher_value01 += weight * world_value01

    policy_total = sum(policy[i] for i, ok in enumerate(action_mask) if ok)
    if policy_total <= 0:
        legal = [i for i, ok in enumerate(action_mask) if ok]
        if not legal:
            return None
        uni = 1.0 / float(len(legal))
        policy = [uni if ok else 0.0 for ok in action_mask]
    else:
        policy = [(v / policy_total) if action_mask[i] else 0.0 for i, v in enumerate(policy)]

    teacher_value = max(-1.0, min(1.0, 2.0 * teacher_value01 - 1.0))
    legal_probs = [policy[i] for i, ok in enumerate(action_mask) if ok and policy[i] > 0.0]
    teacher_top1_prob = max(legal_probs) if legal_probs else 0.0
    teacher_entropy = -sum(p * math.log(p) for p in legal_probs)

    return {
        "battle_id": str(ex.get("battle_id", "")),
        "turn": int(ex.get("turn", 0) or 0),
        "rating": ex.get("rating"),
        "board_features": [safe_float(v, 0.0) for v in board_features],
        "action_features": action_features,
        "action_labels": action_labels,
        "action_mask": [bool(v) for v in action_mask],
        "policy_target": policy,
        "value_target": teacher_value,
        "weight": float(ex.get("weight", 1.0) or 1.0),
        "source": str(ex.get("source", "")),
        "tag": str(ex.get("tag", "")),
        "chosen_action": ex.get("chosen_action"),
        "teacher_source": "oranguru_search_trace",
        "teacher_search_ms": int(search_ms),
        "teacher_worlds_used": int(worlds_used),
        "teacher_total_visits": float(teacher_visits),
        "teacher_top1_prob": float(teacher_top1_prob),
        "teacher_entropy": float(teacher_entropy),
        "teacher_value01": float(teacher_value01),
        "orig_value_target": safe_float(ex.get("value_target", 0.0), 0.0),
        "orig_policy_confidence": safe_float(ex.get("policy_confidence", 0.0), 0.0),
        "orig_policy_threshold": safe_float(ex.get("policy_threshold", 0.0), 0.0),
        "selection_path": str(ex.get("selection_path", "")),
    }


def _precheck_example(ex: dict) -> str | None:
    action_mask = ex.get("action_mask")
    action_features = ex.get("action_features")
    action_labels = ex.get("action_labels")
    board_features = ex.get("board_features")
    world_candidates = ex.get("world_candidates") or []

    if not isinstance(action_mask, list) or not action_mask:
        return "missing_action_mask"
    if not isinstance(action_features, list) or len(action_features) != len(action_mask):
        return "bad_action_features"
    if not isinstance(action_labels, list) or len(action_labels) != len(action_mask):
        return "missing_action_labels"
    if not isinstance(board_features, list) or not board_features:
        return "missing_board_features"
    if not isinstance(world_candidates, list) or not world_candidates:
        return "missing_world_candidates"

    has_state_str = False
    for world in world_candidates:
        if not isinstance(world, dict):
            continue
        state_str = str(world.get("state_str", "") or "")
        if state_str:
            has_state_str = True
            break
    if not has_state_str:
        return "missing_world_state_str"
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Relabel search traces with a stronger search teacher.")
    parser.add_argument("--input", action="append", default=[], help="JSONL/PKL path or glob. Repeatable.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-out", default="")
    parser.add_argument("--search-ms", type=int, default=1200)
    parser.add_argument("--max-worlds", type=int, default=12)
    parser.add_argument("--workers", type=int, default=1)
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
                reason = _precheck_example(ex)
                if reason is not None:
                    counters[f"drop_{reason}"] += 1
                    continue
                relabeled = _relabel_example(
                    ex,
                    search_ms=max(1, args.search_ms),
                    max_worlds=max(1, args.max_worlds),
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

    print(f"Wrote {len(rows)} teacher-relabeled rows -> {out_path}")
    print(f"Stats: {dict(counters)}")

    if args.summary_out:
        summary = {
            "inputs": paths,
            "output": str(out_path),
            "rows": len(rows),
            "counters": dict(counters),
            "search_ms": args.search_ms,
            "max_worlds": args.max_worlds,
            "workers": args.workers,
            "min_turn": args.min_turn,
            "keep_lowconf_only": args.keep_lowconf_only,
            "keep_nonfallback_only": args.keep_nonfallback_only,
            "max_rows": args.max_rows,
        }
        summary_path = Path(args.summary_out)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Summary -> {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
