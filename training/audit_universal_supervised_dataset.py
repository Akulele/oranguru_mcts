#!/usr/bin/env python3
"""Audit a universal supervised dataset pickle."""

from __future__ import annotations

import argparse
import math
import pickle
import statistics
from collections import Counter
from pathlib import Path


def _safe_float(value, default=0.0):
    try:
        out = float(value)
        if math.isfinite(out):
            return out
    except Exception:
        pass
    return default


def _quantiles(values: list[float]) -> tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    vals = sorted(values)
    def pick(q: float) -> float:
        idx = min(len(vals) - 1, max(0, int(round((len(vals) - 1) * q))))
        return float(vals[idx])
    return pick(0.1), pick(0.5), pick(0.9)


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit universal supervised dataset pickle.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--summary-out", default="")
    parser.add_argument("--top-k", type=int, default=25)
    args = parser.parse_args()

    with open(args.input, "rb") as handle:
        rows = pickle.load(handle)
    if not isinstance(rows, list):
        raise SystemExit(f"Expected list pickle: {args.input}")

    battle_ids = Counter()
    action_kind_counts = Counter()
    top_labels = Counter()
    duplicate_turn_keys = Counter()
    duplicate_decision_keys = Counter()
    legal_counts = []
    turns = []
    decision_indexes = []
    ratings = []
    values = []
    sources = Counter()
    empty_masks = 0
    bad_policy_len = 0
    bad_action_features = 0

    for row in rows:
        battle_id = str(row.get("battle_id", ""))
        if battle_id:
            battle_ids[battle_id] += 1

        turn = int(row.get("turn", 0) or 0)
        turns.append(turn)
        decision_index = int(row.get("decision_index", 0) or 0)
        decision_indexes.append(decision_index)

        rating = row.get("rating")
        if isinstance(rating, (int, float)):
            ratings.append(float(rating))

        value = row.get("value_target")
        if isinstance(value, (int, float)):
            values.append(float(value))

        action_kind = str(row.get("chosen_action_kind", ""))
        if action_kind:
            action_kind_counts[action_kind] += 1

        chosen_label = str(row.get("chosen_action_label", ""))
        if chosen_label:
            top_labels[chosen_label] += 1

        source = str(row.get("source", ""))
        if source:
            sources[source] += 1

        mask = row.get("action_mask") or []
        if not isinstance(mask, list) or not any(bool(x) for x in mask):
            empty_masks += 1
        else:
            legal_counts.append(sum(1 for x in mask if bool(x)))

        policy = row.get("policy_target") or []
        if not isinstance(policy, list) or (mask and len(policy) != len(mask)):
            bad_policy_len += 1

        action_features = row.get("action_features") or []
        if not isinstance(action_features, list) or (mask and len(action_features) != len(mask)):
            bad_action_features += 1

        dup_turn_key = (battle_id, str(row.get("player", "")), turn)
        duplicate_turn_keys[dup_turn_key] += 1
        dup_decision_key = (battle_id, str(row.get("player", "")), turn, decision_index)
        duplicate_decision_keys[dup_decision_key] += 1

    duplicate_turn_collisions = sum(1 for _, count in duplicate_turn_keys.items() if count > 1)
    duplicate_decision_collisions = sum(1 for _, count in duplicate_decision_keys.items() if count > 1)
    unique_battles = len(battle_ids)
    rows_per_battle = [count for count in battle_ids.values()]
    max_decision_index = max(decision_indexes) if decision_indexes else 0
    multi_decision_turns = sum(1 for _, count in duplicate_turn_keys.items() if count > 1)

    turn_p10, turn_p50, turn_p90 = _quantiles([float(x) for x in turns]) if turns else (0.0, 0.0, 0.0)
    rating_p10, rating_p50, rating_p90 = _quantiles(ratings) if ratings else (0.0, 0.0, 0.0)
    legal_p10, legal_p50, legal_p90 = _quantiles([float(x) for x in legal_counts]) if legal_counts else (0.0, 0.0, 0.0)

    value_counter = Counter()
    for v in values:
        if v > 0:
            value_counter["positive"] += 1
        elif v < 0:
            value_counter["negative"] += 1
        else:
            value_counter["zero"] += 1

    summary = {
        "rows": len(rows),
        "unique_battles": unique_battles,
        "rows_per_battle_mean": statistics.mean(rows_per_battle) if rows_per_battle else 0.0,
        "rows_per_battle_median": statistics.median(rows_per_battle) if rows_per_battle else 0.0,
        "turn_mean": statistics.mean(turns) if turns else 0.0,
        "turn_p10": turn_p10,
        "turn_p50": turn_p50,
        "turn_p90": turn_p90,
        "rating_count": len(ratings),
        "rating_mean": statistics.mean(ratings) if ratings else 0.0,
        "rating_p10": rating_p10,
        "rating_p50": rating_p50,
        "rating_p90": rating_p90,
        "value_counts": dict(value_counter),
        "legal_count_mean": statistics.mean(legal_counts) if legal_counts else 0.0,
        "legal_count_p10": legal_p10,
        "legal_count_p50": legal_p50,
        "legal_count_p90": legal_p90,
        "action_kind_counts": dict(action_kind_counts),
        "top_labels": top_labels.most_common(args.top_k),
        "source_counts": dict(sources),
        "duplicate_battle_player_turn_keys": duplicate_turn_collisions,
        "duplicate_battle_player_turn_decision_keys": duplicate_decision_collisions,
        "multi_decision_turns": multi_decision_turns,
        "max_decision_index": max_decision_index,
        "empty_masks": empty_masks,
        "bad_policy_len": bad_policy_len,
        "bad_action_features": bad_action_features,
    }

    print(f"Dataset: {args.input}")
    print(f"Rows: {summary['rows']}")
    print()
    print("Basic")
    print(f"  unique battles: {summary['unique_battles']}")
    print(f"  rows/battle mean/median: {summary['rows_per_battle_mean']:.2f} / {summary['rows_per_battle_median']:.2f}")
    print(f"  turn mean p10/p50/p90: {summary['turn_mean']:.2f} / {summary['turn_p10']:.1f} / {summary['turn_p50']:.1f} / {summary['turn_p90']:.1f}")
    print(f"  rating count/mean p10/p50/p90: {summary['rating_count']} / {summary['rating_mean']:.2f} / {summary['rating_p10']:.1f} / {summary['rating_p50']:.1f} / {summary['rating_p90']:.1f}")
    print()
    print("Actions")
    print(f"  action kinds: {dict(action_kind_counts)}")
    print(f"  legal count mean p10/p50/p90: {summary['legal_count_mean']:.2f} / {summary['legal_count_p10']:.1f} / {summary['legal_count_p50']:.1f} / {summary['legal_count_p90']:.1f}")
    print(f"  top labels: {top_labels.most_common(args.top_k)}")
    print()
    print("Values")
    print(f"  value counts: {dict(value_counter)}")
    print()
    print("Integrity")
    print(f"  duplicate battle/player/turn keys: {duplicate_turn_collisions}")
    print(f"  duplicate battle/player/turn/decision keys: {duplicate_decision_collisions}")
    print(f"  multi-decision turns: {multi_decision_turns}")
    print(f"  max decision index: {max_decision_index}")
    print(f"  empty masks: {empty_masks}")
    print(f"  bad policy len: {bad_policy_len}")
    print(f"  bad action features: {bad_action_features}")
    print(f"  source counts: {dict(sources)}")

    if args.summary_out:
        out_path = Path(args.summary_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        import json
        out_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding='utf-8')
        print(f"Summary -> {args.summary_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
