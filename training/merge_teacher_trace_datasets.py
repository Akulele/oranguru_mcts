#!/usr/bin/env python3
"""
Merge teacher-trace datasets into one training-ready pickle with quality filters.

Inputs can be PKL or JSONL and may be specified as PATH, GLOB, or PATH::WEIGHT.
The output stays in the search-assist schema consumed by
`training/train_search_prior_value.py`.
"""

from __future__ import annotations

import argparse
from collections import Counter
import glob
import json
import math
import pickle
from pathlib import Path
import random
from typing import Iterable


def _parse_input_spec(spec: str) -> tuple[str, float]:
    if "::" not in spec:
        return spec, 1.0
    path, scale = spec.rsplit("::", 1)
    return path, float(scale)


def _resolve_input_specs(specs: list[str]) -> list[tuple[str, float]]:
    resolved: list[tuple[str, float]] = []
    for spec in specs:
        pattern, scale = _parse_input_spec(spec)
        matches = sorted(glob.glob(pattern))
        if matches:
            for match in matches:
                resolved.append((match, scale))
        elif Path(pattern).exists():
            resolved.append((pattern, scale))
    return resolved


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


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _legal_probs(row: dict) -> list[float]:
    policy = row.get("policy_target") or []
    mask = row.get("action_mask") or []
    if not isinstance(policy, list) or not isinstance(mask, list) or len(policy) != len(mask):
        return []
    probs: list[float] = []
    for prob, ok in zip(policy, mask):
        if not ok:
            continue
        p = _safe_float(prob, 0.0)
        if p > 0.0:
            probs.append(p)
    return probs


def _teacher_top1_prob(row: dict) -> float:
    if "teacher_top1_prob" in row:
        return _safe_float(row.get("teacher_top1_prob"), 0.0)
    probs = _legal_probs(row)
    return max(probs) if probs else 0.0


def _teacher_entropy(row: dict) -> float:
    if "teacher_entropy" in row:
        return _safe_float(row.get("teacher_entropy"), 0.0)
    probs = _legal_probs(row)
    return -sum(p * math.log(p) for p in probs)


def _teacher_worlds_used(row: dict) -> int:
    return int(
        row.get(
            "teacher_worlds_used",
            row.get("teacher_samples_used", 0),
        )
        or 0
    )


def _best_label(row: dict) -> str:
    policy = row.get("policy_target") or []
    labels = row.get("action_labels") or []
    mask = row.get("action_mask") or []
    if (
        not isinstance(policy, list)
        or not isinstance(labels, list)
        or not isinstance(mask, list)
        or len(policy) != len(labels)
        or len(policy) != len(mask)
        or not policy
    ):
        return ""
    legal = [i for i, ok in enumerate(mask) if ok]
    if not legal:
        return ""
    best_i = max(legal, key=lambda i: _safe_float(policy[i], 0.0))
    return str(labels[best_i] or "")


def _action_kind(label: str) -> str:
    if label.startswith("switch "):
        return "switch"
    if label.endswith("-tera"):
        return "tera_move"
    if label:
        return "move"
    return "unknown"


def _is_concrete_label(label: str) -> bool:
    return bool(label) and not label.startswith("move_slot_")


def _onehot_top1(policy: list[float], mask: list[bool]) -> list[float]:
    legal = [i for i, ok in enumerate(mask) if ok]
    if not legal:
        return policy
    best_i = max(legal, key=lambda i: _safe_float(policy[i], 0.0))
    out = [0.0] * len(policy)
    out[best_i] = 1.0
    return out


def _row_dims_ok(row: dict, board_dim: int | None, action_dim: int | None, n_actions: int | None) -> bool:
    board = row.get("board_features")
    action = row.get("action_features")
    mask = row.get("action_mask")
    labels = row.get("action_labels")
    policy = row.get("policy_target")
    if not isinstance(board, list) or not board:
        return False
    if not isinstance(action, list) or not action:
        return False
    if not isinstance(mask, list) or not isinstance(labels, list) or not isinstance(policy, list):
        return False
    if len(action) != len(mask) or len(labels) != len(mask) or len(policy) != len(mask):
        return False
    first_action = action[0]
    if not isinstance(first_action, list) or not first_action:
        return False
    if board_dim is not None and len(board) != board_dim:
        return False
    if action_dim is not None and len(first_action) != action_dim:
        return False
    if n_actions is not None and len(mask) != n_actions:
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge teacher-trace datasets with quality filters.")
    parser.add_argument("--input", action="append", required=True, help="PATH, GLOB, or PATH::WEIGHT")
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-out", default="")
    parser.add_argument("--teacher-source", action="append", default=[], help="Keep only matching teacher_source values.")
    parser.add_argument("--min-turn", type=int, default=1)
    parser.add_argument("--keep-nonfallback-only", action="store_true")
    parser.add_argument("--min-top1-prob", type=float, default=0.0)
    parser.add_argument("--max-entropy", type=float, default=0.0)
    parser.add_argument("--min-total-visits", type=float, default=0.0)
    parser.add_argument("--min-worlds-used", type=int, default=0)
    parser.add_argument("--only-concrete-top1", action="store_true")
    parser.add_argument("--only-move-top1", action="store_true")
    parser.add_argument("--only-switch-top1", action="store_true")
    parser.add_argument("--only-tera-top1", action="store_true")
    parser.add_argument("--onehot-top1", action="store_true")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rows", type=int, default=0)
    args = parser.parse_args()

    resolved_inputs = _resolve_input_specs(args.input)
    if not resolved_inputs:
        raise SystemExit("No input files matched.")

    allowed_sources = set(args.teacher_source)
    counters = Counter()
    teacher_source_counts = Counter()
    top_labels = Counter()
    kept_rows: list[dict] = []
    board_dim = None
    action_dim = None
    n_actions = None
    top1_probs: list[float] = []
    entropies: list[float] = []
    visits: list[float] = []
    worlds_used_values: list[int] = []

    for path, scale in resolved_inputs:
        for row in _iter_examples(path):
            counters["rows_seen"] += 1
            turn = int(row.get("turn", 0) or 0)
            if turn < args.min_turn:
                counters["drop_min_turn"] += 1
                continue
            if args.keep_nonfallback_only and str(row.get("selection_path", "") or "").startswith("fallback"):
                counters["drop_fallback"] += 1
                continue
            teacher_source = str(row.get("teacher_source", "") or "")
            if allowed_sources and teacher_source not in allowed_sources:
                counters["drop_teacher_source"] += 1
                continue
            if not _row_dims_ok(row, board_dim, action_dim, n_actions):
                if board_dim is None:
                    board = row.get("board_features") or []
                    action = row.get("action_features") or []
                    mask = row.get("action_mask") or []
                    board_dim = len(board)
                    action_dim = len(action[0]) if action and isinstance(action[0], list) else None
                    n_actions = len(mask)
                else:
                    counters["drop_dim_mismatch"] += 1
                    continue

            top1 = _teacher_top1_prob(row)
            entropy = _teacher_entropy(row)
            total_visits = _safe_float(row.get("teacher_total_visits", 0.0), 0.0)
            worlds_used = _teacher_worlds_used(row)
            label = _best_label(row)
            kind = _action_kind(label)

            if top1 < args.min_top1_prob:
                counters["drop_top1_prob"] += 1
                continue
            if args.max_entropy > 0.0 and entropy > args.max_entropy:
                counters["drop_entropy"] += 1
                continue
            if total_visits < args.min_total_visits:
                counters["drop_total_visits"] += 1
                continue
            if worlds_used < args.min_worlds_used:
                counters["drop_worlds_used"] += 1
                continue
            if args.only_concrete_top1 and not _is_concrete_label(label):
                counters["drop_not_concrete_top1"] += 1
                continue
            if args.only_move_top1 and kind != "move":
                counters["drop_not_move_top1"] += 1
                continue
            if args.only_switch_top1 and kind != "switch":
                counters["drop_not_switch_top1"] += 1
                continue
            if args.only_tera_top1 and kind != "tera_move":
                counters["drop_not_tera_top1"] += 1
                continue

            out = dict(row)
            out["weight"] = float(out.get("weight", 1.0) or 1.0) * float(scale)
            out["teacher_top1_prob"] = float(top1)
            out["teacher_entropy"] = float(entropy)
            out["teacher_worlds_used"] = int(worlds_used)
            if args.onehot_top1:
                out["policy_target"] = _onehot_top1(
                    [_safe_float(v, 0.0) for v in list(out.get("policy_target", []) or [])],
                    [bool(v) for v in list(out.get("action_mask", []) or [])],
                )
            kept_rows.append(out)
            counters["kept"] += 1
            teacher_source_counts[teacher_source or ""] += 1
            top_labels[label] += 1
            top1_probs.append(top1)
            entropies.append(entropy)
            visits.append(total_visits)
            worlds_used_values.append(worlds_used)
            if args.max_rows > 0 and counters["kept"] >= args.max_rows:
                break
        if args.max_rows > 0 and counters["kept"] >= args.max_rows:
            break

    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(kept_rows)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as handle:
        pickle.dump(kept_rows, handle)

    summary = {
        "inputs": [{"path": path, "weight_scale": scale} for path, scale in resolved_inputs],
        "output": str(out_path),
        "rows_out": len(kept_rows),
        "stats": dict(counters),
        "teacher_source_counts": dict(teacher_source_counts),
        "top_labels": top_labels.most_common(30),
        "teacher_top1_mean": (sum(top1_probs) / len(top1_probs)) if top1_probs else 0.0,
        "teacher_entropy_mean": (sum(entropies) / len(entropies)) if entropies else 0.0,
        "teacher_total_visits_mean": (sum(visits) / len(visits)) if visits else 0.0,
        "teacher_worlds_used_mean": (sum(worlds_used_values) / len(worlds_used_values)) if worlds_used_values else 0.0,
        "board_dim": board_dim or 0,
        "action_dim": action_dim or 0,
        "n_actions": n_actions or 0,
        "filters": {
            "teacher_source": sorted(allowed_sources),
            "min_turn": int(args.min_turn),
            "keep_nonfallback_only": bool(args.keep_nonfallback_only),
            "min_top1_prob": float(args.min_top1_prob),
            "max_entropy": float(args.max_entropy),
            "min_total_visits": float(args.min_total_visits),
            "min_worlds_used": int(args.min_worlds_used),
            "only_concrete_top1": bool(args.only_concrete_top1),
            "only_move_top1": bool(args.only_move_top1),
            "only_switch_top1": bool(args.only_switch_top1),
            "only_tera_top1": bool(args.only_tera_top1),
            "onehot_top1": bool(args.onehot_top1),
        },
    }

    print(f"Merged {len(kept_rows)} teacher-trace rows -> {out_path}")
    print(f"Stats: {dict(counters)}")
    print(f"Teacher sources: {dict(teacher_source_counts)}")
    print(f"Top labels: {top_labels.most_common(20)}")
    print(f"Teacher top1 mean: {summary['teacher_top1_mean']:.4f}")
    print(f"Teacher entropy mean: {summary['teacher_entropy_mean']:.4f}")

    if args.summary_out:
        summary_path = Path(args.summary_out)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        print(f"Summary -> {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
