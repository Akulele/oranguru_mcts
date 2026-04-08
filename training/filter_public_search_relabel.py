#!/usr/bin/env python3
"""Filter public-search-relabeled datasets down to sharper teacher targets."""

from __future__ import annotations

import argparse
import json
import math
import pickle
from collections import Counter
from pathlib import Path


def _teacher_probs(row: dict) -> list[float]:
    probs = row.get("teacher_policy_target")
    if isinstance(probs, list) and probs:
        return [float(x) for x in probs]
    probs = row.get("policy_target")
    if isinstance(probs, list) and probs:
        return [float(x) for x in probs]
    return []


def _top1_info(row: dict) -> tuple[int, float, str]:
    probs = _teacher_probs(row)
    labels = [str(x) for x in row.get("action_labels", [])]
    if not probs:
        return -1, 0.0, ""
    best_i = max(range(len(probs)), key=lambda i: probs[i])
    label = labels[best_i] if best_i < len(labels) else ""
    return best_i, float(probs[best_i]), label


def _entropy(row: dict) -> float:
    probs = _teacher_probs(row)
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


def _is_concrete_label(label: str) -> bool:
    if not label:
        return False
    if "move_slot_" in label:
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Filter a public-search-relabeled dataset.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-out", default="")
    parser.add_argument("--min-top1-prob", type=float, default=0.0)
    parser.add_argument("--max-entropy", type=float, default=-1.0)
    parser.add_argument("--min-worlds-used", type=int, default=0)
    parser.add_argument("--min-total-visits", type=float, default=0.0)
    parser.add_argument("--only-concrete-top1", action="store_true")
    parser.add_argument("--only-move-top1", action="store_true")
    parser.add_argument("--only-switch-top1", action="store_true")
    parser.add_argument("--only-tera-top1", action="store_true")
    parser.add_argument("--min-turn", type=int, default=0)
    parser.add_argument("--phase", action="append", default=[])
    parser.add_argument("--onehot-top1", action="store_true")
    parser.add_argument("--max-rows", type=int, default=0)
    args = parser.parse_args()

    with open(args.input, "rb") as handle:
        rows = pickle.load(handle)
    if not isinstance(rows, list):
        raise SystemExit(f"Expected list pickle: {args.input}")

    phases = {str(x) for x in args.phase if str(x)}
    counters = Counter()
    kept: list[dict] = []
    kept_labels = Counter()
    kept_kinds = Counter()

    for row in rows:
        if not isinstance(row, dict):
            counters["drop_not_dict"] += 1
            continue
        counters["rows_seen"] += 1

        turn = int(row.get("turn", 0) or 0)
        if turn < args.min_turn:
            counters["drop_min_turn"] += 1
            continue
        phase = str(row.get("phase", "") or "")
        if phases and phase not in phases:
            counters["drop_phase"] += 1
            continue

        best_i, top1_prob, top_label = _top1_info(row)
        ent = _entropy(row)
        kind = _label_kind(top_label)
        worlds_used = int(row.get("teacher_worlds_used", row.get("teacher_samples_used", 0)) or 0)
        total_visits = float(row.get("teacher_total_visits", 0.0) or 0.0)

        if top1_prob < args.min_top1_prob:
            counters["drop_top1_prob"] += 1
            continue
        if args.max_entropy >= 0.0 and ent > args.max_entropy:
            counters["drop_entropy"] += 1
            continue
        if worlds_used < args.min_worlds_used:
            counters["drop_worlds_used"] += 1
            continue
        if total_visits < args.min_total_visits:
            counters["drop_total_visits"] += 1
            continue
        if args.only_concrete_top1 and not _is_concrete_label(top_label):
            counters["drop_non_concrete_top1"] += 1
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

        new_row = dict(row)
        new_row["filter_top1_prob"] = top1_prob
        new_row["filter_entropy"] = ent
        new_row["filter_top1_kind"] = kind
        new_row["filter_top1_label"] = top_label
        if args.onehot_top1:
            probs = [0.0] * len(_teacher_probs(row))
            if 0 <= best_i < len(probs):
                probs[best_i] = 1.0
            new_row["policy_target"] = probs
            new_row["teacher_policy_target"] = list(probs)
        kept.append(new_row)
        kept_labels[top_label] += 1
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
    print(f"Top labels: {kept_labels.most_common(20)}")

    if args.summary_out:
        payload = {
            "input": args.input,
            "output": args.output,
            "rows_out": len(kept),
            "stats": dict(counters),
            "top_kinds": kept_kinds.most_common(),
            "top_labels": kept_labels.most_common(20),
            "filters": {
                "min_top1_prob": args.min_top1_prob,
                "max_entropy": args.max_entropy,
                "min_worlds_used": args.min_worlds_used,
                "min_total_visits": args.min_total_visits,
                "only_concrete_top1": bool(args.only_concrete_top1),
                "only_move_top1": bool(args.only_move_top1),
                "only_switch_top1": bool(args.only_switch_top1),
                "only_tera_top1": bool(args.only_tera_top1),
                "min_turn": args.min_turn,
                "phase": list(phases),
                "onehot_top1": bool(args.onehot_top1),
                "max_rows": args.max_rows,
            },
        }
        summary_path = Path(args.summary_out)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
        print(f"Summary -> {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
