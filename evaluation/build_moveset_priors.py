#!/usr/bin/env python3
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Tuple

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.damage_calc import normalize_name


def _battle_text(obj: dict) -> str:
    chunks = []
    for turn in obj.get("turns", []):
        for event in turn.get("events", []):
            raw = event.get("raw_parts") or []
            if raw:
                chunks.append(" ".join(str(x) for x in raw).lower())
    return "\n".join(chunks)


def _avg_pre_rating(obj: dict) -> float | None:
    players = obj.get("players", {}) or {}
    p1 = (players.get("p1", {}) or {}).get("ladder_rating_pre")
    p2 = (players.get("p2", {}) or {}).get("ladder_rating_pre")
    if isinstance(p1, (int, float)) and isinstance(p2, (int, float)):
        return (float(p1) + float(p2)) / 2.0
    return None


def _passes_filters(
    obj: dict,
    *,
    min_turns: int,
    min_avg_rating: int,
    skip_forfeit: bool,
    skip_inactivity: bool,
) -> Tuple[bool, str]:
    turns = len(obj.get("turns", []) or [])
    if min_turns > 0 and turns < min_turns:
        return False, "short"

    avg_rating = _avg_pre_rating(obj)
    if min_avg_rating > 0 and (avg_rating is None or avg_rating < min_avg_rating):
        return False, "low_rating"

    text = _battle_text(obj)
    if skip_forfeit and "forfeit" in text:
        return False, "forfeit"
    if skip_inactivity and "lost due to inactivity" in text:
        return False, "inactivity"
    return True, "ok"


def build_priors(
    json_dir: Path,
    top_k: int,
    *,
    min_turns: int,
    min_avg_rating: int,
    skip_forfeit: bool,
    skip_inactivity: bool,
) -> tuple[dict, Dict[str, int]]:
    counts = defaultdict(Counter)
    stats = Counter()
    for path in json_dir.glob("*.json"):
        stats["files_seen"] += 1
        try:
            with path.open("r", encoding="utf-8") as handle:
                obj = json.load(handle)
        except Exception:
            stats["parse_error"] += 1
            continue

        ok, reason = _passes_filters(
            obj,
            min_turns=min_turns,
            min_avg_rating=min_avg_rating,
            skip_forfeit=skip_forfeit,
            skip_inactivity=skip_inactivity,
        )
        if not ok:
            stats[f"skip_{reason}"] += 1
            continue
        stats["files_used"] += 1

        teams = obj.get("team_revelation", {}).get("teams", {})
        uid_to_species = {}
        for _, mons in teams.items():
            for mon in mons:
                uid = mon.get("pokemon_uid")
                species = normalize_name(mon.get("species", ""))
                if uid and species:
                    uid_to_species[uid] = species

        for turn in obj.get("turns", []):
            for event in turn.get("events", []):
                if event.get("type") != "move":
                    continue
                uid = event.get("pokemon_uid")
                move_id = normalize_name(event.get("move_id", ""))
                species = uid_to_species.get(uid)
                if species and move_id:
                    counts[species][move_id] += 1
                    stats["move_events"] += 1

    priors = {}
    for species, counter in counts.items():
        priors[species] = {
            move_id: int(count) for move_id, count in counter.most_common(top_k)
        }
    stats["species"] = len(priors)
    stats["moves_kept"] = sum(len(v) for v in priors.values())
    return priors, dict(stats)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build random battle moveset priors.")
    parser.add_argument("--json-dir", type=str, required=True,
                        help="Directory with replay JSONs (gen9randombattle)")
    parser.add_argument("--output", type=str, default="data/moveset_priors.json",
                        help="Output JSON path")
    parser.add_argument("--top-k", type=int, default=8,
                        help="Top moves to keep per species")
    parser.add_argument("--min-turns", type=int, default=0,
                        help="Drop battles below this turn count")
    parser.add_argument("--min-avg-rating", type=int, default=0,
                        help="Drop battles with avg pre-rating below this threshold")
    parser.add_argument("--skip-forfeit", action="store_true",
                        help="Drop battles containing forfeit text")
    parser.add_argument("--skip-inactivity", action="store_true",
                        help="Drop battles containing inactivity losses")
    parser.add_argument("--summary-output", type=str, default="",
                        help="Optional JSON summary output")
    args = parser.parse_args()

    json_dir = Path(args.json_dir)
    if not json_dir.exists():
        raise SystemExit(f"Missing json dir: {json_dir}")

    priors, stats = build_priors(
        json_dir,
        args.top_k,
        min_turns=args.min_turns,
        min_avg_rating=args.min_avg_rating,
        skip_forfeit=args.skip_forfeit,
        skip_inactivity=args.skip_inactivity,
    )
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(priors, handle, indent=2, sort_keys=True)

    species_count = len(priors)
    move_count = sum(len(v) for v in priors.values())
    print(f"Wrote {out_path} ({species_count} species, {move_count} moves)")
    print("Stats:")
    for k in sorted(stats):
        print(f"- {k}: {stats[k]}")

    if args.summary_output:
        summary_path = Path(args.summary_output)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "input_dir": str(json_dir),
            "output": str(out_path),
            "top_k": args.top_k,
            "min_turns": args.min_turns,
            "min_avg_rating": args.min_avg_rating,
            "skip_forfeit": bool(args.skip_forfeit),
            "skip_inactivity": bool(args.skip_inactivity),
            "stats": stats,
        }
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        print(f"Wrote summary {summary_path}")


if __name__ == "__main__":
    main()
