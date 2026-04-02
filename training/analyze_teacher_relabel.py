#!/usr/bin/env python3
"""
Analyze a teacher-relabeled search dataset before training.
"""

from __future__ import annotations

import argparse
import math
import pickle
import statistics
from collections import Counter
from pathlib import Path


def _safe_mean(values):
    return statistics.mean(values) if values else 0.0


def _safe_median(values):
    return statistics.median(values) if values else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze a teacher-relabeled dataset.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--show-samples", type=int, default=0)
    args = parser.parse_args()

    path = Path(args.input)
    with path.open("rb") as handle:
        rows = pickle.load(handle)
    if not isinstance(rows, list):
        raise SystemExit(f"Expected list dataset in {path}")
    rows = [row for row in rows if isinstance(row, dict)]
    if not rows:
        raise SystemExit("Dataset is empty.")

    turns = [int(r.get("turn", 0) or 0) for r in rows]
    values = [float(r.get("value_target", 0.0) or 0.0) for r in rows]
    orig_values = [float(r.get("orig_value_target", 0.0) or 0.0) for r in rows]
    worlds = [
        int(
            r.get(
                "teacher_worlds_used",
                r.get("teacher_samples_used", 0),
            )
            or 0
        )
        for r in rows
    ]
    visits = [float(r.get("teacher_total_visits", 0.0) or 0.0) for r in rows]
    paths = Counter(str(r.get("selection_path", "")) for r in rows)

    entropies = []
    top1s = []
    legal_counts = []
    top_actions = Counter()
    action_kind_counts = Counter()
    value_diffs = []

    for row in rows:
        mask = [bool(x) for x in row.get("action_mask", [])]
        probs = [float(x) for x in row.get("policy_target", [])]
        labels = [str(x) for x in row.get("action_labels", [])]
        if mask and probs and len(mask) == len(probs):
            legal = sum(mask)
            legal_counts.append(legal)
            legal_probs = [p for p, ok in zip(probs, mask) if ok and p > 0]
            entropies.append(-sum(p * math.log(p) for p in legal_probs))
            top1s.append(max(legal_probs) if legal_probs else 0.0)
            best_i = max(range(len(probs)), key=lambda i: probs[i])
            label = labels[best_i] if best_i < len(labels) else ""
            top_actions[label] += 1
            if label.startswith("switch "):
                action_kind_counts["switch"] += 1
            elif label.endswith("-tera"):
                action_kind_counts["tera_move"] += 1
            elif label:
                action_kind_counts["move"] += 1
            else:
                action_kind_counts["unknown"] += 1
        value_diffs.append(float(row.get("value_target", 0.0) or 0.0) - float(row.get("orig_value_target", 0.0) or 0.0))

    absdiffs = [abs(x) for x in value_diffs]
    sign_agreement = sum((tv >= 0) == (ov >= 0) for tv, ov in zip(values, orig_values)) / len(rows)

    print(f"Dataset: {path}")
    print(f"Rows: {len(rows)}")
    print()
    print("Basic")
    print(f"  turn mean/median: {_safe_mean(turns):.2f} / {_safe_median(turns):.2f}")
    print(f"  teacher value mean/min/max: {_safe_mean(values):.4f} / {min(values):.4f} / {max(values):.4f}")
    print(f"  orig value mean/min/max: {_safe_mean(orig_values):.4f} / {min(orig_values):.4f} / {max(orig_values):.4f}")
    print(f"  worlds used mean/median: {_safe_mean(worlds):.2f} / {_safe_median(worlds):.2f}")
    print(f"  teacher visits mean/median: {_safe_mean(visits):.2f} / {_safe_median(visits):.2f}")
    print(f"  selection paths: {paths.most_common(10)}")
    print()
    print("Policy")
    print(f"  legal action count mean/median: {_safe_mean(legal_counts):.2f} / {_safe_median(legal_counts):.2f}")
    print(f"  entropy mean/median: {_safe_mean(entropies):.4f} / {_safe_median(entropies):.4f}")
    print(f"  top1 prob mean/median: {_safe_mean(top1s):.4f} / {_safe_median(top1s):.4f}")
    print(f"  top action kinds: {action_kind_counts.most_common()}")
    print(f"  top labels: {top_actions.most_common(30)}")
    print()
    print("Teacher vs Original Value")
    print(f"  mean diff: {_safe_mean(value_diffs):.4f}")
    print(f"  mean abs diff: {_safe_mean(absdiffs):.4f}")
    print(f"  median abs diff: {_safe_median(absdiffs):.4f}")
    print(f"  sign agreement: {sign_agreement:.4f}")
    print(f"  share abs diff >= 0.25: {sum(x >= 0.25 for x in absdiffs) / len(absdiffs):.4f}")
    print(f"  share abs diff >= 0.50: {sum(x >= 0.50 for x in absdiffs) / len(absdiffs):.4f}")
    if args.show_samples > 0:
        print()
        print("Samples")
        for i, row in enumerate(rows[: args.show_samples]):
            probs = [float(x) for x in row.get("policy_target", [])]
            labels = [str(x) for x in row.get("action_labels", [])]
            best_i = max(range(len(probs)), key=lambda j: probs[j]) if probs else 0
            best_label = labels[best_i] if best_i < len(labels) else ""
            best_prob = probs[best_i] if probs and best_i < len(probs) else 0.0
            print(
                {
                    "i": i,
                    "turn": int(row.get("turn", 0) or 0),
                    "orig_value": round(float(row.get("orig_value_target", 0.0) or 0.0), 4),
                    "teacher_value": round(float(row.get("value_target", 0.0) or 0.0), 4),
                    "teacher_value01": round(float(row.get("teacher_value01", 0.0) or 0.0), 4),
                    "best_label": best_label,
                    "best_prob": round(float(best_prob), 4),
                    "samples_used": int(
                        row.get(
                            "teacher_worlds_used",
                            row.get("teacher_samples_used", 0),
                        )
                        or 0
                    ),
                }
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
