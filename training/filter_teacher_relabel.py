#!/usr/bin/env python3
"""
Filter teacher-relabeled datasets down to higher-quality policy targets.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
from collections import Counter
from pathlib import Path


def _top1_prob_and_label(row: dict) -> tuple[float, str]:
    probs = [float(x) for x in row.get("policy_target", [])]
    labels = [str(x) for x in row.get("action_labels", [])]
    if not probs:
        return 0.0, ""
    best_i = max(range(len(probs)), key=lambda i: probs[i])
    label = labels[best_i] if best_i < len(labels) else ""
    return float(probs[best_i]), label


def _entropy(row: dict) -> float:
    probs = [float(x) for x in row.get("policy_target", [])]
    mask = [bool(x) for x in row.get("action_mask", [])]
    legal_probs = [p for p, ok in zip(probs, mask) if ok and p > 0]
    if not legal_probs:
        return 0.0
    return -sum(p * math.log(p) for p in legal_probs)


def _label_kind(label: str) -> str:
    if not label:
        return "unknown"
    if label.startswith("switch "):
        return "switch"
    if label.endswith("-tera"):
        return "tera_move"
    return "move"


def main() -> int:
    parser = argparse.ArgumentParser(description="Filter a teacher-relabeled dataset.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-out", default="")
    parser.add_argument("--min-top1-prob", type=float, default=0.0)
    parser.add_argument("--max-entropy", type=float, default=-1.0)
    parser.add_argument("--min-samples-used", type=int, default=0)
    parser.add_argument("--only-move-top1", action="store_true")
    parser.add_argument("--only-switch-top1", action="store_true")
    parser.add_argument("--max-rows", type=int, default=0)
    args = parser.parse_args()

    with open(args.input, "rb") as handle:
        rows = pickle.load(handle)
    if not isinstance(rows, list):
        raise SystemExit(f"Expected list pickle: {args.input}")

    counters = Counter()
    kept: list[dict] = []
    kept_top_labels = Counter()
    kept_kinds = Counter()

    for row in rows:
        if not isinstance(row, dict):
            counters["drop_not_dict"] += 1
            continue
        counters["rows_seen"] += 1
        top1_prob, top_label = _top1_prob_and_label(row)
        ent = _entropy(row)
        samples_used = int(
            row.get("teacher_worlds_used", row.get("teacher_samples_used", 0)) or 0
        )
        kind = _label_kind(top_label)

        if top1_prob < args.min_top1_prob:
            counters["drop_top1_prob"] += 1
            continue
        if args.max_entropy >= 0.0 and ent > args.max_entropy:
            counters["drop_entropy"] += 1
            continue
        if samples_used < args.min_samples_used:
            counters["drop_samples_used"] += 1
            continue
        if args.only_move_top1 and kind != "move":
            counters["drop_not_move_top1"] += 1
            continue
        if args.only_switch_top1 and kind != "switch":
            counters["drop_not_switch_top1"] += 1
            continue

        row = dict(row)
        row["filter_top1_prob"] = top1_prob
        row["filter_entropy"] = ent
        row["filter_top1_kind"] = kind
        kept.append(row)
        kept_top_labels[top_label] += 1
        kept_kinds[kind] += 1
        counters["kept"] += 1
        if args.max_rows > 0 and len(kept) >= args.max_rows:
            break

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as handle:
        pickle.dump(kept, handle)

    print(f"Wrote {len(kept)} filtered rows -> {out_path}")
    print(f"Stats: {dict(counters)}")
    print(f"Top kinds: {kept_kinds.most_common()}")
    print(f"Top labels: {kept_top_labels.most_common(20)}")

    if args.summary_out:
        payload = {
            "input": args.input,
            "output": args.output,
            "rows_out": len(kept),
            "stats": dict(counters),
            "top_kinds": kept_kinds.most_common(),
            "top_labels": kept_top_labels.most_common(20),
            "filters": {
                "min_top1_prob": args.min_top1_prob,
                "max_entropy": args.max_entropy,
                "min_samples_used": args.min_samples_used,
                "only_move_top1": bool(args.only_move_top1),
                "only_switch_top1": bool(args.only_switch_top1),
                "max_rows": args.max_rows,
            },
        }
        summary_path = Path(args.summary_out)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Summary -> {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
