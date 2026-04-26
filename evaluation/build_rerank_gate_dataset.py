#!/usr/bin/env python3
"""Build a weakly supervised dataset for tactical rerank gate experiments.

The dataset is intentionally about *supporting* MCTS: rows are accepted reranks
where the candidate displaced the MCTS top action.  The label is battle outcome,
so it is noisy, but useful for calibrating which rerank sources and score-drop
profiles are worth trusting.
"""

from __future__ import annotations

import argparse
from collections import Counter
import glob
import json
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
    build_trace_rerank_gate_example,
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


def _iter_rows(paths: Iterable[str]):
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


def _shadow_take_examples(row: dict) -> list[dict]:
    value_target = safe_float(row.get("value_target", 0.0), 0.0)
    if value_target == 0.0:
        return []
    actions = top_actions(row)
    if not actions:
        return []
    top1 = actions[0]
    top1_choice = str(top1.get("choice", "") or "")
    if not top1_choice:
        return []

    examples: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for source, key in RERANK_WINDOWS:
        payload = row.get(key)
        if not isinstance(payload, dict):
            continue
        reason = str(payload.get("reason", "") or "")
        if not reason.startswith("take_"):
            continue
        targets = [
            str(payload.get(target_key, "") or "")
            for target_key in TAKE_TARGET_KEYS
            if payload.get(target_key)
        ]
        for candidate_choice in targets:
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
            examples.append(
                {
                    "battle_id": str(row.get("battle_id", "") or ""),
                    "turn": int(row.get("turn", 0) or 0),
                    "source": source_reason,
                    "label": 1 if value_target > 0.0 else 0,
                    "value_target": value_target,
                    "candidate_choice": candidate_choice,
                    "top1_choice": top1_choice,
                    "candidate_kind": candidate_kind,
                    "top1_kind": top1_kind,
                    "features": features,
                    "feature_vector": feature_vector(features),
                    "feature_names": list(FEATURE_NAMES),
                    "example_type": "shadow_take",
                    "selection_path": str(row.get("selection_path", "") or ""),
                }
            )
    return examples


def build_dataset(
    paths: list[str],
    *,
    sources: set[str] | None = None,
    limit: int = 0,
    include_shadow: bool = False,
) -> tuple[list[dict], dict]:
    examples: list[dict] = []
    seen = 0
    skipped = Counter()
    labels = Counter()
    by_source = Counter()
    by_type = Counter()
    for row in _iter_rows(paths):
        seen += 1
        row_examples: list[dict] = []
        accepted = build_trace_rerank_gate_example(row)
        if accepted is not None:
            accepted.setdefault("example_type", "accepted_rerank")
            row_examples.append(accepted)
        if include_shadow:
            row_examples.extend(_shadow_take_examples(row))
        if not row_examples:
            skipped["not_training_example"] += 1
            continue
        for example in row_examples:
            if sources and example["source"] not in sources:
                skipped["source_filter"] += 1
                continue
            examples.append(example)
            labels[str(example["label"])] += 1
            by_source[str(example["source"])] += 1
            by_type[str(example.get("example_type", "accepted_rerank"))] += 1
            if limit and len(examples) >= limit:
                break
        if limit and len(examples) >= limit:
            break
    summary = {
        "rows_seen": seen,
        "examples": len(examples),
        "labels": dict(labels),
        "by_source": dict(by_source),
        "by_type": dict(by_type),
        "skipped": dict(skipped),
    }
    return examples, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build rerank gate JSONL examples from search traces.")
    parser.add_argument("inputs", nargs="+", help="Trace JSONL paths or glob patterns")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        help="Optional rerank source filter, e.g. setup_window:take_setup. Repeatable.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Maximum examples to write; 0 means no limit")
    parser.add_argument(
        "--include-shadow",
        action="store_true",
        help="Also emit shadow take_* window candidates from MCTS rows without applying the rerank.",
    )
    parser.add_argument("--summary-out", default="", help="Optional JSON summary path")
    args = parser.parse_args()

    paths = _resolve_inputs(args.inputs)
    if not paths:
        raise SystemExit("No input traces matched.")

    examples, summary = build_dataset(
        paths,
        sources=set(args.source) if args.source else None,
        limit=args.limit,
        include_shadow=bool(args.include_shadow),
    )
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example, sort_keys=True) + "\n")

    if args.summary_out:
        summary_path = Path(args.summary_out)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"Inputs: {len(paths)}")
    print(f"Rows seen: {summary['rows_seen']}")
    print(f"Examples: {summary['examples']}")
    print(f"Labels: {summary['labels']}")
    print(f"By source: {summary['by_source']}")
    print(f"By type: {summary['by_type']}")
    print(f"Output -> {out_path}")
    if args.summary_out:
        print(f"Summary -> {args.summary_out}")


if __name__ == "__main__":
    main()
