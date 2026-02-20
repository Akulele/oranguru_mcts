#!/usr/bin/env python3
"""
Parse FoulPlay debug logs into imitation demos.

Expected logs come from `evaluation/run_foulplay_selfplay_collect.py`, where each
bot logs websocket receive/send lines at DEBUG level.
"""

from __future__ import annotations

import argparse
import glob
import json
import pickle
import re
from collections import Counter
from pathlib import Path

FEATURE_DIM = 272
N_ACTIONS = 13

REQUEST_TOKEN = "|request|{"
CHOOSE_RE = re.compile(
    r"Sending message to websocket:\s*([^\s|]+)\|/choose\s+([^|\n]+)\|(\d+)",
    re.IGNORECASE,
)
BATTLE_TAG_RE = re.compile(r">(battle-[a-z0-9-]+)", re.IGNORECASE)


def _norm(s: str | None) -> str:
    if not s:
        return ""
    return "".join(ch.lower() for ch in str(s) if ch.isalnum())


def _parse_condition(cond: str | None) -> tuple[int | None, int | None, str]:
    if not cond:
        return None, None, ""
    raw = str(cond).strip()
    if "fnt" in raw:
        return 0, None, "fnt"
    first = raw.split(" ", 1)[0]
    parts = first.split("/")
    if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
        hp = int(parts[0])
        max_hp = int(parts[1])
    else:
        hp = None
        max_hp = None
    status = ""
    if " " in raw:
        status = raw.split(" ", 1)[1].strip()
    return hp, max_hp, status


def _build_mask_and_maps(request: dict) -> tuple[list[bool], list[str], dict[int, int], bool]:
    mask = [False] * N_ACTIONS
    active = (request.get("active") or [{}])[0] or {}
    side = (request.get("side") or {})
    team = side.get("pokemon") or []
    force_switch = bool(request.get("forceSwitch"))

    move_ids: list[str] = []
    moves = active.get("moves") or []
    if not force_switch:
        for i, m in enumerate(moves[:4]):
            move_id = _norm(m.get("id") or m.get("move"))
            move_ids.append(move_id)
            disabled = bool(m.get("disabled", False))
            pp = m.get("pp", 1)
            if not disabled and (pp is None or pp > 0):
                mask[i] = True

        can_tera = bool(active.get("canTerastallize"))
        if can_tera:
            for i in range(min(4, len(moves))):
                if mask[i]:
                    mask[9 + i] = True
    else:
        for i, m in enumerate(moves[:4]):
            move_ids.append(_norm(m.get("id") or m.get("move")))

    switch_slot_to_action: dict[int, int] = {}
    legal_slots: list[int] = []
    for slot, mon in enumerate(team, start=1):
        if mon.get("active"):
            continue
        hp, _, _ = _parse_condition(mon.get("condition"))
        if hp == 0:
            continue
        legal_slots.append(slot)
    for idx, slot in enumerate(legal_slots[:5]):
        mask[4 + idx] = True
        switch_slot_to_action[slot] = 4 + idx
    return mask, move_ids, switch_slot_to_action, force_switch


def _action_from_choice(choice: str, mask: list[bool], move_ids: list[str], switch_map: dict[int, int]) -> int | None:
    low = choice.strip().lower()
    parts = low.split()
    if not parts:
        return None

    if parts[0] == "move":
        if len(parts) < 2:
            return None
        token = _norm(parts[1])
        idx = None
        if token.isdigit():
            j = int(token) - 1
            if 0 <= j < 4:
                idx = j
        else:
            for j, mid in enumerate(move_ids[:4]):
                if token == mid:
                    idx = j
                    break
        if idx is None:
            return None
        tera = any(p in {"terastallize", "tera"} for p in parts[2:])
        if tera and (9 + idx) < N_ACTIONS and mask[9 + idx]:
            return 9 + idx
        return idx if mask[idx] else None

    if parts[0] == "switch":
        if len(parts) < 2:
            return None
        tok = parts[1]
        if tok.isdigit():
            slot = int(tok)
            return switch_map.get(slot)
        return None

    return None


def _features_from_request(request: dict, force_switch: bool, mask: list[bool]) -> list[float]:
    feat = [0.0] * FEATURE_DIM
    side = request.get("side") or {}
    team = side.get("pokemon") or []
    active_mon = None
    for mon in team:
        if mon.get("active"):
            active_mon = mon
            break
    if active_mon is None and team:
        active_mon = team[0]

    hp_frac = 1.0
    status_flag = 0.0
    if active_mon:
        hp, max_hp, status = _parse_condition(active_mon.get("condition"))
        if isinstance(hp, int) and isinstance(max_hp, int) and max_hp > 0:
            hp_frac = hp / max_hp
        if status and status != "fnt":
            status_flag = 1.0

    alive = 0
    for mon in team:
        hp, _, _ = _parse_condition(mon.get("condition"))
        if hp is None or hp > 0:
            alive += 1
    fainted = max(0, len(team) - alive)
    can_tera = bool((request.get("active") or [{}])[0].get("canTerastallize"))

    move_count = sum(1 for i in range(4) if mask[i])
    switch_count = sum(1 for i in range(4, 9) if mask[i])
    tera_count = sum(1 for i in range(9, 13) if mask[i])

    feat[0] = float(hp_frac)
    feat[1] = float(alive) / 6.0
    feat[2] = float(fainted) / 6.0
    feat[3] = status_flag
    feat[4] = 1.0 if can_tera else 0.0
    feat[5] = 1.0 if force_switch else 0.0
    feat[6] = float(move_count) / 4.0
    feat[7] = float(switch_count) / 5.0
    feat[8] = float(tera_count) / 4.0

    stats = (active_mon or {}).get("stats") or {}
    for i, key in enumerate(["atk", "def", "spa", "spd", "spe"], start=9):
        value = stats.get(key)
        if isinstance(value, (int, float)):
            feat[i] = min(500.0, float(value)) / 500.0

    moves = ((request.get("active") or [{}])[0] or {}).get("moves") or []
    offset = 14
    for i, move in enumerate(moves[:4]):
        pp = move.get("pp")
        max_pp = move.get("maxpp")
        if isinstance(pp, (int, float)) and isinstance(max_pp, (int, float)) and max_pp > 0:
            feat[offset + i] = max(0.0, min(1.0, float(pp) / float(max_pp)))
        feat[offset + 4 + i] = 1.0 if move.get("disabled") else 0.0

    return feat


def parse_log(path: Path, counters: Counter) -> list[dict]:
    pending: dict[tuple[str, int], dict] = {}
    latest_by_battle: dict[str, tuple[int, dict]] = {}
    demos: list[dict] = []
    last_battle_tag = ""

    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line_no, line in enumerate(handle, start=1):
            if line_no % 200000 == 0:
                print(
                    f"{path.name}: lines={line_no} demos={counters['demos']}",
                    flush=True,
                )

            battle_match = BATTLE_TAG_RE.search(line)
            if battle_match:
                last_battle_tag = battle_match.group(1).strip()

            if REQUEST_TOKEN in line:
                json_start = line.find("|request|") + len("|request|")
                payload_str = line[json_start:].strip()
                try:
                    req = json.loads(payload_str)
                except Exception:
                    counters["skip_request_parse_error"] += 1
                    req = None
                if isinstance(req, dict):
                    rqid = req.get("rqid")
                    if not isinstance(rqid, int):
                        counters["skip_request_no_rqid"] += 1
                    elif last_battle_tag:
                        pending[(last_battle_tag, rqid)] = req
                        latest_by_battle[last_battle_tag] = (rqid, req)
                        counters["requests"] += 1
                    else:
                        counters["skip_request_no_battle_tag"] += 1

            choose_match = CHOOSE_RE.search(line)
            if not choose_match:
                continue
            battle_tag = choose_match.group(1).strip()
            choice = choose_match.group(2).strip()
            rqid = int(choose_match.group(3))

            request = pending.pop((battle_tag, rqid), None)
            if request is None:
                latest = latest_by_battle.get(battle_tag)
                if latest and latest[0] == rqid:
                    request = latest[1]
            if request is None:
                counters["skip_choose_missing_request"] += 1
                continue

            mask, move_ids, switch_map, force_switch = _build_mask_and_maps(request)
            action = _action_from_choice(choice, mask, move_ids, switch_map)
            if action is None:
                counters["skip_unmapped_action"] += 1
                continue
            if not mask[action]:
                counters["skip_illegal_action"] += 1
                continue

            features = _features_from_request(request, force_switch, mask)
            if len(features) != FEATURE_DIM:
                counters["skip_feature_dim"] += 1
                continue

            demos.append(
                {
                    "features": features,
                    "action": int(action),
                    "mask": mask,
                    "rating": None,
                    "weight": 1.0,
                    "battle_tag": battle_tag,
                    "rqid": rqid,
                    "source_log": str(path),
                }
            )
            counters["demos"] += 1
    return demos


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log",
        action="append",
        default=[],
        help="FoulPlay debug log path or glob (repeatable).",
    )
    parser.add_argument("--output", default="data/highelo_demos_foulplay_selfplay.pkl")
    parser.add_argument("--summary-out", default="")
    parser.add_argument("--min-demos", type=int, default=1)
    args = parser.parse_args()

    paths: list[Path] = []
    for entry in args.log:
        if any(ch in entry for ch in "*?[]"):
            paths.extend(Path(p) for p in sorted(glob.glob(entry)))
        else:
            p = Path(entry)
            if p.exists():
                paths.append(p)
    # dedup preserve order
    seen = set()
    uniq_paths = []
    for p in paths:
        rp = str(p.resolve())
        if rp not in seen:
            seen.add(rp)
            uniq_paths.append(Path(rp))
    paths = uniq_paths

    if not paths:
        raise SystemExit("No logs found. Pass --log <path|glob>.")

    counters = Counter()
    demos: list[dict] = []
    for path in paths:
        counters["logs_seen"] += 1
        try:
            parsed = parse_log(path, counters)
        except Exception:
            counters["log_parse_error"] += 1
            continue
        demos.extend(parsed)

    if len(demos) < args.min_demos:
        raise SystemExit(
            f"Collected {len(demos)} demos, below min {args.min_demos}. Check logs/filters."
        )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as handle:
        pickle.dump(demos, handle)

    summary = {
        "logs": [str(p) for p in paths],
        "output": str(out_path),
        "counts": dict(counters),
        "demos": len(demos),
    }
    summary_path = Path(args.summary_out) if args.summary_out else out_path.with_suffix(
        out_path.suffix + ".summary.json"
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Wrote {len(demos)} demos -> {out_path}")
    print(f"Summary -> {summary_path}")
    print(f"Counts: {dict(counters)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
