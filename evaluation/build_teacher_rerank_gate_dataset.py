#!/usr/bin/env python3
"""Build rerank-gate examples labeled by a stronger search teacher."""

from __future__ import annotations

import argparse
from collections import Counter
import glob
import json
import pickle
from pathlib import Path
import sys
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.players.oranguru_rerank_gate import (
    FEATURE_NAMES,
    RERANK_WINDOWS,
    TAKE_TARGET_KEYS,
    action_for_choice,
    action_heuristic,
    action_kind,
    action_risk,
    action_score,
    build_feature_dict,
    feature_vector,
    safe_float,
    top_actions,
)


def _resolve_inputs(patterns: list[str]) -> list[str]:
    paths: list[str] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            paths.extend(matches)
        elif Path(pattern).exists():
            paths.append(pattern)
    return paths


def _iter_trace_rows(paths: Iterable[str]):
    for path in paths:
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                if isinstance(row, dict):
                    yield row


def _row_key(row: dict) -> tuple[str, int]:
    return str(row.get("battle_id", "") or ""), int(row.get("turn", 0) or 0)


def _load_teacher_rows(paths: list[str]) -> dict[tuple[str, int], dict]:
    teacher: dict[tuple[str, int], dict] = {}
    for path in paths:
        with Path(path).open("rb") as handle:
            rows = pickle.load(handle)
        if not isinstance(rows, list):
            continue
        for row in rows:
            if isinstance(row, dict):
                teacher[_row_key(row)] = row
    return teacher


def _teacher_prob(teacher_row: dict, choice: str) -> float:
    labels = teacher_row.get("action_labels") or []
    probs = teacher_row.get("policy_target") or []
    if not isinstance(labels, list) or not isinstance(probs, list):
        return 0.0
    for idx, label in enumerate(labels):
        if str(label or "") == choice and idx < len(probs):
            return max(0.0, safe_float(probs[idx], 0.0))
    return 0.0


def _examples_for_row(row: dict, teacher_row: dict, *, min_delta: float, min_candidate_prob: float) -> list[dict]:
    actions = top_actions(row)
    if not actions:
        return []
    top1 = actions[0]
    top1_choice = str(top1.get("choice", "") or "")
    if not top1_choice:
        return []
    top1_prob = _teacher_prob(teacher_row, top1_choice)
    examples: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for source, key in RERANK_WINDOWS:
        payload = row.get(key)
        if not isinstance(payload, dict):
            continue
        reason = str(payload.get("reason", "") or "")
        if not reason.startswith("take_"):
            continue
        for target_key in TAKE_TARGET_KEYS:
            candidate_choice = str(payload.get(target_key, "") or "")
            if not candidate_choice or candidate_choice == top1_choice:
                continue
            source_reason = f"{source}:{reason}"
            dedupe_key = (source_reason, candidate_choice)
            if dedupe_key in seen:
                continue
            candidate = action_for_choice(row, candidate_choice)
            if candidate is None:
                continue
            seen.add(dedupe_key)
            candidate_prob = _teacher_prob(teacher_row, candidate_choice)
            teacher_delta = candidate_prob - top1_prob
            label = int(candidate_prob >= min_candidate_prob and teacher_delta >= min_delta)
            candidate_kind = action_kind(candidate_choice, candidate)
            top1_kind = action_kind(top1_choice, top1)
            features = build_feature_dict(
                source_reason=source_reason,
                candidate_choice=candidate_choice,
                top1_choice=top1_choice,
                candidate_kind=candidate_kind,
                top1_kind=top1_kind,
                candidate_score=action_score(candidate),
                top1_score=action_score(top1),
                candidate_heuristic=action_heuristic(candidate),
                top1_heuristic=action_heuristic(top1),
                candidate_risk=action_risk(candidate),
                top1_risk=action_risk(top1),
                policy_confidence=safe_float(row.get("policy_confidence", 0.0), 0.0),
                policy_threshold=safe_float(row.get("policy_threshold", 0.0), 0.0),
                matchup_score=safe_float(row.get("matchup_score", 0.0), 0.0),
                best_reply_score=safe_float(row.get("best_reply_score", 0.0), 0.0),
                hazard_load=safe_float(row.get("hazard_load", 0.0), 0.0),
                payload=payload,
            )
            features["teacher_candidate_prob"] = float(candidate_prob)
            features["teacher_top1_prob"] = float(top1_prob)
            features["teacher_delta_candidate_minus_top1"] = float(teacher_delta)
            examples.append(
                {
                    "battle_id": str(row.get("battle_id", "") or ""),
                    "turn": int(row.get("turn", 0) or 0),
                    "source": source_reason,
                    "label": label,
                    "candidate_choice": candidate_choice,
                    "top1_choice": top1_choice,
                    "candidate_kind": candidate_kind,
                    "top1_kind": top1_kind,
                    "teacher_candidate_prob": float(candidate_prob),
                    "teacher_top1_prob": float(top1_prob),
                    "teacher_delta_candidate_minus_top1": float(teacher_delta),
                    "teacher_top1_prob_overall": safe_float(teacher_row.get("teacher_top1_prob", 0.0), 0.0),
                    "teacher_entropy": safe_float(teacher_row.get("teacher_entropy", 0.0), 0.0),
                    "teacher_worlds_used": int(teacher_row.get("teacher_worlds_used", 0) or 0),
                    "teacher_total_visits": safe_float(teacher_row.get("teacher_total_visits", 0.0), 0.0),
                    "features": features,
                    "feature_vector": feature_vector(features),
                    "feature_names": list(FEATURE_NAMES),
                    "example_type": "teacher_shadow_take",
                    "selection_path": str(row.get("selection_path", "") or ""),
                }
            )
    return examples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", action="append", required=True, help="Filtered trace JSONL path/glob. Repeatable.")
    parser.add_argument("--teacher", action="append", required=True, help="Teacher relabel PKL path/glob. Repeatable.")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--summary-out", default="", help="Optional JSON summary path")
    parser.add_argument("--min-delta", type=float, default=0.02, help="Candidate teacher prob advantage required for label=1")
    parser.add_argument("--min-candidate-prob", type=float, default=0.05, help="Minimum teacher prob for candidate label=1")
    parser.add_argument("--source", action="append", default=[], help="Optional source filter; repeatable")
    args = parser.parse_args()

    trace_paths = _resolve_inputs(args.trace)
    teacher_paths = _resolve_inputs(args.teacher)
    if not trace_paths:
        raise SystemExit("No trace inputs matched.")
    if not teacher_paths:
        raise SystemExit("No teacher inputs matched.")

    teacher_rows = _load_teacher_rows(teacher_paths)
    source_filter = set(args.source)
    counters = Counter()
    labels = Counter()
    by_source = Counter()
    by_source_label = Counter()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in _iter_trace_rows(trace_paths):
            counters["rows_seen"] += 1
            teacher_row = teacher_rows.get(_row_key(row))
            if teacher_row is None:
                counters["drop_no_teacher"] += 1
                continue
            examples = _examples_for_row(
                row,
                teacher_row,
                min_delta=max(0.0, float(args.min_delta)),
                min_candidate_prob=max(0.0, float(args.min_candidate_prob)),
            )
            if not examples:
                counters["drop_no_examples"] += 1
                continue
            for example in examples:
                if source_filter and example["source"] not in source_filter:
                    counters["drop_source_filter"] += 1
                    continue
                handle.write(json.dumps(example, sort_keys=True) + "\n")
                counters["examples"] += 1
                label = str(example["label"])
                labels[label] += 1
                by_source[example["source"]] += 1
                by_source_label[f"{example['source']}:{label}"] += 1

    summary = {
        "trace_inputs": trace_paths,
        "teacher_inputs": teacher_paths,
        "output": str(output),
        "counters": dict(counters),
        "labels": dict(labels),
        "by_source": dict(by_source),
        "by_source_label": dict(by_source_label),
        "min_delta": float(args.min_delta),
        "min_candidate_prob": float(args.min_candidate_prob),
    }
    print(f"Trace inputs: {len(trace_paths)}")
    print(f"Teacher inputs: {len(teacher_paths)}")
    print(f"Rows seen: {counters['rows_seen']}")
    print(f"Examples: {counters['examples']}")
    print(f"Labels: {dict(labels)}")
    print(f"By source: {dict(by_source)}")
    print(f"Output -> {output}")
    if args.summary_out:
        summary_path = Path(args.summary_out)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Summary -> {summary_path}")


if __name__ == "__main__":
    main()
