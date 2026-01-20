#!/usr/bin/env python3
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.damage_calc import normalize_name


def build_priors(json_dir: Path, top_k: int) -> dict:
    counts = defaultdict(Counter)
    for path in json_dir.glob("*.json"):
        try:
            with path.open("r", encoding="utf-8") as handle:
                obj = json.load(handle)
        except Exception:
            continue

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

    priors = {}
    for species, counter in counts.items():
        priors[species] = {
            move_id: int(count) for move_id, count in counter.most_common(top_k)
        }
    return priors


def main() -> None:
    parser = argparse.ArgumentParser(description="Build random battle moveset priors.")
    parser.add_argument("--json-dir", type=str, required=True,
                        help="Directory with replay JSONs (gen9randombattle)")
    parser.add_argument("--output", type=str, default="data/moveset_priors.json",
                        help="Output JSON path")
    parser.add_argument("--top-k", type=int, default=8,
                        help="Top moves to keep per species")
    args = parser.parse_args()

    json_dir = Path(args.json_dir)
    if not json_dir.exists():
        raise SystemExit(f"Missing json dir: {json_dir}")

    priors = build_priors(json_dir, args.top_k)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(priors, handle, indent=2, sort_keys=True)

    species_count = len(priors)
    move_count = sum(len(v) for v in priors.values())
    print(f"Wrote {out_path} ({species_count} species, {move_count} moves)")


if __name__ == "__main__":
    main()
