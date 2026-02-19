#!/usr/bin/env python3
"""
Quick quality audit for gen9 random replay JSON dumps.

Example:
  venv/bin/python training/audit_gen9random_replays.py \
    --input-dir data/gen9random \
    --summary-out logs/replay_audit/gen9random_audit.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
from collections import Counter
from pathlib import Path
from statistics import mean, median


def _avg_pre_rating(obj: dict) -> float | None:
    players = obj.get("players", {}) or {}
    p1 = (players.get("p1", {}) or {}).get("ladder_rating_pre")
    p2 = (players.get("p2", {}) or {}).get("ladder_rating_pre")
    if isinstance(p1, (int, float)) and isinstance(p2, (int, float)):
        return (float(p1) + float(p2)) / 2.0
    return None


def _battle_text(obj: dict) -> str:
    chunks: list[str] = []
    for turn in obj.get("turns", []):
        for event in turn.get("events", []):
            raw = event.get("raw_parts") or []
            if raw:
                chunks.append(" ".join(str(x) for x in raw).lower())
    return "\n".join(chunks)


def _pct(sorted_vals: list[int], q: float) -> int:
    if not sorted_vals:
        return 0
    idx = int((len(sorted_vals) - 1) * q)
    idx = max(0, min(len(sorted_vals) - 1, idx))
    return sorted_vals[idx]


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit gen9 random replay JSON quality.")
    parser.add_argument("--input-dir", default="data/gen9random")
    parser.add_argument("--ladder-raw-glob", default="data/ladder_raw/*.pkl")
    parser.add_argument("--summary-out", default="")
    parser.add_argument("--min-turns", type=int, default=8)
    parser.add_argument("--min-avg-rating", type=int, default=1400)
    args = parser.parse_args()

    replay_paths = sorted(glob.glob(str(Path(args.input_dir) / "*.json")))
    stats = Counter()
    reasons = Counter()
    turn_counts: list[int] = []
    ratings: list[float] = []
    battle_avg_ratings: list[float] = []
    battle_ids: list[str] = []

    for path in replay_paths:
        stats["files_seen"] += 1
        try:
            obj = json.load(open(path, "r", encoding="utf-8"))
        except Exception:
            stats["parse_errors"] += 1
            continue

        battle_id = obj.get("battle_id")
        if battle_id:
            battle_ids.append(str(battle_id))

        turns = len(obj.get("turns", []) or [])
        turn_counts.append(turns)
        if turns == 0:
            stats["turns_0"] += 1
        if turns == 1:
            stats["turns_1"] += 1
        if turns < args.min_turns:
            stats["turns_below_min"] += 1

        players = obj.get("players", {}) or {}
        for side in ("p1", "p2"):
            r = (players.get(side, {}) or {}).get("ladder_rating_pre")
            if isinstance(r, (int, float)):
                ratings.append(float(r))

        avg_rating = _avg_pre_rating(obj)
        if avg_rating is not None:
            battle_avg_ratings.append(avg_rating)
            if avg_rating >= args.min_avg_rating:
                stats["avg_rating_meets_threshold"] += 1
            if avg_rating >= 1500:
                stats["avg_rating_ge_1500"] += 1

        text = _battle_text(obj)
        has_forfeit = "forfeit" in text
        has_inactivity = "lost due to inactivity" in text
        if has_forfeit:
            stats["has_forfeit"] += 1
        if has_inactivity:
            stats["has_inactivity"] += 1

        outcome = (obj.get("metadata", {}) or {}).get("outcome", {}) or {}
        if outcome:
            stats[f"outcome_{outcome.get('result', 'unknown')}"] += 1
            reason = str(outcome.get("reason", "unknown"))
            reasons[reason] += 1

        good = True
        if turns < args.min_turns:
            good = False
        if avg_rating is None or avg_rating < args.min_avg_rating:
            good = False
        if has_forfeit or has_inactivity:
            good = False
        if good:
            stats["clean_candidates"] += 1

    uniq = len(set(battle_ids))
    stats["unique_battle_ids"] = uniq
    stats["duplicate_battle_ids"] = max(0, len(battle_ids) - uniq)

    ladder_rows = 0
    ladder_nonempty = 0
    ladder_files = sorted(glob.glob(args.ladder_raw_glob))
    for path in ladder_files:
        try:
            obj = pickle.load(open(path, "rb"))
        except Exception:
            continue
        if isinstance(obj, list):
            n = len(obj)
            ladder_rows += n
            if n > 0:
                ladder_nonempty += 1

    print("Replay audit")
    print(f"- files: {stats['files_seen']} parse_errors: {stats['parse_errors']}")
    if turn_counts:
        tc = sorted(turn_counts)
        print(
            "- turns min/p10/p50/p90/max: "
            f"{tc[0]}/{_pct(tc,0.1)}/{_pct(tc,0.5)}/{_pct(tc,0.9)}/{tc[-1]}"
        )
    print(
        f"- short(<{args.min_turns}): {stats['turns_below_min']} "
        f"({(stats['turns_below_min']/max(1,stats['files_seen'])):.1%})"
    )
    print(
        f"- forfeit: {stats['has_forfeit']} inactivity: {stats['has_inactivity']}"
    )
    if ratings:
        print(
            "- player pre-rating mean/median/min/max: "
            f"{mean(ratings):.1f}/{median(ratings):.1f}/{int(min(ratings))}/{int(max(ratings))}"
        )
    if battle_avg_ratings:
        print(
            f"- avg battle rating >= {args.min_avg_rating}: {stats['avg_rating_meets_threshold']} "
            f"/ {len(battle_avg_ratings)} "
            f"({stats['avg_rating_meets_threshold']/len(battle_avg_ratings):.1%})"
        )
        print(
            f"- clean candidates (turns>={args.min_turns}, avg>={args.min_avg_rating}, no forfeit/inactivity): "
            f"{stats['clean_candidates']}"
        )
    print(f"- unique battle IDs: {stats['unique_battle_ids']}")
    print(f"- top outcome reasons: {reasons.most_common(8)}")
    print(
        f"- ladder_raw files: {len(ladder_files)} nonempty: {ladder_nonempty} "
        f"rows: {ladder_rows}"
    )

    if args.summary_out:
        out = Path(args.summary_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "input_dir": args.input_dir,
            "min_turns": args.min_turns,
            "min_avg_rating": args.min_avg_rating,
            "stats": dict(stats),
            "top_outcome_reasons": reasons.most_common(20),
            "ladder_raw": {
                "glob": args.ladder_raw_glob,
                "files": len(ladder_files),
                "nonempty_files": ladder_nonempty,
                "rows": ladder_rows,
            },
        }
        with out.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        print(f"- wrote summary: {out}")


if __name__ == "__main__":
    main()
