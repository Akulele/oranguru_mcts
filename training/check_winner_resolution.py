#!/usr/bin/env python3
"""Check whether parsed replay files can resolve a winner side with the current helper."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.prepare_search_assist_bootstrap import _resolve_winner_side


def _build_min_state(obj: dict) -> dict:
    teams = (obj.get("team_revelation", {}) or {}).get("teams", {}) or {}
    team_order = {
        side: [mon.get("pokemon_uid") for mon in teams.get(side, []) if mon.get("pokemon_uid")]
        for side in ("p1", "p2")
    }
    state = {
        "team_order": team_order,
        "uid_side": {uid: side for side in ("p1", "p2") for uid in team_order.get(side, [])},
        "active": {"p1": None, "p2": None},
        "hp": {},
        "max_hp": {},
        "status": {},
        "species": {},
        "move_slots": {},
        "hazards": {
            "p1": {"spikes": 0, "toxicspikes": 0, "stealthrock": 0, "stickyweb": 0},
            "p2": {"spikes": 0, "toxicspikes": 0, "stealthrock": 0, "stickyweb": 0},
        },
        "tera_used": {"p1": False, "p2": False},
    }
    for side in ("p1", "p2"):
        for mon in teams.get(side, []):
            uid = mon.get("pokemon_uid")
            if not uid:
                continue
            hp = ((mon.get("base_stats") or {}).get("hp"))
            if isinstance(hp, (int, float)):
                state["hp"][uid] = int(hp)
                state["max_hp"][uid] = int(hp)
    return state


def main() -> int:
    parser = argparse.ArgumentParser(description="Check winner resolution on parsed replay JSON files.")
    parser.add_argument("--input-dir", default="data/showdown/parsed")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--summary-out", default="")
    args = parser.parse_args()

    paths = sorted(Path(args.input_dir).glob("*.json"))
    bad: list[str] = []
    json_failures = 0
    for path in paths:
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            json_failures += 1
            continue
        state = _build_min_state(obj)
        if _resolve_winner_side(obj, state) is None:
            bad.append(str(path))

    print(f"files_seen {len(paths)}")
    print(f"json_failures {json_failures}")
    print(f"resolve_bad_count {len(bad)}")
    for path in bad[: max(0, args.limit)]:
        print(path)

    if args.summary_out:
        summary = {
            "files_seen": len(paths),
            "json_failures": json_failures,
            "resolve_bad_count": len(bad),
            "sample": bad[: max(0, args.limit)],
        }
        out_path = Path(args.summary_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")
        print(f"Summary -> {args.summary_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
