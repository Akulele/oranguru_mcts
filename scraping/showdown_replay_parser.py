#!/usr/bin/env python3
"""Parse raw public Showdown replay JSON into the local structured replay schema.

The output intentionally mirrors the existing `data/gen9random/*.json` shape closely
so current training scripts can consume scraped data with minimal changes.
"""

from __future__ import annotations

import argparse
import copy
import glob
import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "1.1.0"


def _norm(value: object) -> str:
    if value is None:
        return ""
    return "".join(ch.lower() for ch in str(value) if ch.isalnum())


def _side_from_ident(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    low = value.strip().lower()
    if low.startswith("p1"):
        return "p1"
    if low.startswith("p2"):
        return "p2"
    return None


def _split_protocol(line: str) -> list[str]:
    if not line.startswith("|"):
        return []
    return line.split("|")[1:]


def _extract_log_text(payload: dict[str, Any]) -> str:
    for key in ("log", "inputlog", "inputLog"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _raw_player_name(payload: dict[str, Any], side: str) -> str | None:
    for key in (side, side.upper()):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    players = payload.get("players")
    if isinstance(players, list):
        idx = 0 if side == "p1" else 1
        if idx < len(players) and isinstance(players[idx], str) and players[idx].strip():
            return players[idx].strip()
    if isinstance(players, dict):
        value = players.get(side)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _parse_details(details: str) -> dict[str, Any]:
    out = {
        "species": None,
        "gender": None,
        "level": None,
    }
    if not isinstance(details, str):
        return out
    bits = [bit.strip() for bit in details.split(",") if bit.strip()]
    if not bits:
        return out
    out["species"] = bits[0]
    for bit in bits[1:]:
        if bit in {"M", "F", "N"}:
            out["gender"] = bit
        elif bit.startswith("L") and bit[1:].isdigit():
            out["level"] = int(bit[1:])
    return out


def _parse_hp_status(token: object) -> tuple[int | None, int | None, str | None, bool]:
    if not isinstance(token, str):
        return None, None, None, False
    text = token.strip()
    if not text:
        return None, None, None, False
    if text == "0 fnt" or text.endswith(" fnt"):
        return 0, None, None, True
    bits = text.split()
    hp_bits = bits[0]
    status = bits[1] if len(bits) > 1 else None
    if "/" in hp_bits:
        left, right = hp_bits.split("/", 1)
        try:
            hp = int(float(left))
        except Exception:
            hp = None
        try:
            max_hp = int(float(right))
        except Exception:
            max_hp = None
        return hp, max_hp, status, False
    try:
        hp = int(float(hp_bits))
    except Exception:
        hp = None
    return hp, None, status, False


def _hash_log(log_text: str) -> str:
    return hashlib.sha256(log_text.encode("utf-8", "ignore")).hexdigest()


def _initial_state() -> dict[str, Any]:
    return {
        "team_order": {"p1": [], "p2": []},
        "uid_side": {},
        "active": {"p1": None, "p2": None},
        "hp": {},
        "max_hp": {},
        "status": defaultdict(bool),
        "species": {},
        "move_slots": defaultdict(dict),
        "hazards": {
            "p1": {"spikes": 0, "toxicspikes": 0, "stealthrock": 0, "stickyweb": 0},
            "p2": {"spikes": 0, "toxicspikes": 0, "stealthrock": 0, "stickyweb": 0},
        },
        "tera_used": {"p1": False, "p2": False},
    }


def _ensure_team_entry(
    team_revelation: dict[str, Any],
    state: dict[str, Any],
    *,
    side: str,
    species: str | None,
    turn_number: int,
    details: str | None = None,
) -> str | None:
    parsed_input = _parse_details(details or str(species or ""))
    species_name = str(parsed_input.get("species") or species or "").strip()
    species_norm = _norm(species_name)
    if side not in ("p1", "p2") or not species_norm:
        return None

    teams = team_revelation.setdefault("teams", {"p1": [], "p2": []})
    side_team = teams.setdefault(side, [])
    for mon in side_team:
        if _norm(mon.get("species")) == species_norm:
            return mon.get("pokemon_uid")

    uid = f"{side}-{len(side_team)}"
    mon = {
        "pokemon_uid": uid,
        "species": species_name,
        "nickname": None,
        "level": parsed_input.get("level"),
        "gender": parsed_input.get("gender"),
        "first_seen_turn": int(turn_number),
        "revelation_status": "partially_revealed",
        "known_ability": None,
        "known_item": None,
        "known_tera_type": None,
        "base_stats": {
            "hp": None,
            "attack": None,
            "defense": None,
            "sp_attack": None,
            "sp_defense": None,
            "speed": None,
        },
        "known_moves": [],
        "unknown_move_slots": 4,
    }
    side_team.append(mon)
    state["team_order"][side].append(uid)
    state["uid_side"][uid] = side
    state["species"][uid] = species_norm
    return uid


def _event_from_parts(
    parts: list[str],
    *,
    state: dict[str, Any],
    team_revelation: dict[str, Any],
    active_slots: dict[str, str],
    turn_number: int,
    seq: int,
    fainted_order: dict[str, list[str]],
) -> dict[str, Any] | None:
    if not parts:
        return None
    kind = parts[0]
    raw_parts = parts

    if kind == "poke" and len(parts) >= 3:
        side = _side_from_ident(parts[1])
        if side:
            _ensure_team_entry(team_revelation, state, side=side, species=parts[2], details=parts[2], turn_number=0)
        return None

    if kind in {"switch", "drag", "replace"} and len(parts) >= 4:
        ident, details, hp_token = parts[1], parts[2], parts[3]
        side = _side_from_ident(ident)
        parsed = _parse_details(details)
        uid = _ensure_team_entry(
            team_revelation,
            state,
            side=side or "",
            species=parsed.get("species"),
            details=details,
            turn_number=turn_number,
        )
        if side and uid:
            active_slots[f"{side}a"] = uid
            state["active"][side] = uid
            hp_after, max_hp, status, fainted = _parse_hp_status(hp_token)
            if hp_after is not None:
                state["hp"][uid] = hp_after
            if max_hp is not None:
                state["max_hp"][uid] = max_hp
                teams = team_revelation.get("teams", {}) or {}
                for mon in teams.get(side, []):
                    if mon.get("pokemon_uid") == uid:
                        mon["base_stats"]["hp"] = max_hp
                        break
            if status:
                state["status"][uid] = True
            elif not fainted:
                state["status"][uid] = False
        return {
            "seq": seq,
            "type": "switch",
            "player": side,
            "pokemon_uid": uid,
            "into_uid": uid,
            "raw_parts": raw_parts,
        }

    if kind == "move" and len(parts) >= 3:
        ident = parts[1]
        side = _side_from_ident(ident)
        uid = active_slots.get(f"{side}a") if side else None
        move_name = parts[2]
        target_uid = None
        if len(parts) >= 4:
            target_side = _side_from_ident(parts[3])
            if target_side:
                target_uid = active_slots.get(f"{target_side}a")
        if side and uid and move_name:
            move_id = _norm(move_name)
            slots = state["move_slots"][uid]
            if move_id and move_id not in slots and len(slots) < 4:
                slots[move_id] = len(slots)
            teams = team_revelation.get("teams", {}) or {}
            for mon in teams.get(side, []):
                if mon.get("pokemon_uid") == uid and move_id and move_id not in mon["known_moves"]:
                    mon["known_moves"].append(move_id)
                    mon["unknown_move_slots"] = max(0, 4 - len(mon["known_moves"]))
                    break
        return {
            "seq": seq,
            "type": "move",
            "player": side,
            "pokemon_uid": uid,
            "move_id": _norm(move_name),
            "target_uid": target_uid,
            "raw_parts": raw_parts,
        }

    if kind in {"-damage", "-heal", "-sethp"} and len(parts) >= 3:
        ident = parts[1]
        side = _side_from_ident(ident)
        uid = active_slots.get(f"{side}a") if side else None
        hp_after, max_hp, status, fainted = _parse_hp_status(parts[2])
        if uid and hp_after is not None:
            state["hp"][uid] = hp_after
        if uid and max_hp is not None:
            state["max_hp"][uid] = max_hp
        if uid and status:
            state["status"][uid] = True
        if uid and fainted:
            state["hp"][uid] = 0
        return {
            "seq": seq,
            "type": "damage" if kind != "-heal" else "heal",
            "target_uid": uid,
            "amount": None,
            "hp_after": hp_after,
            "max_hp": max_hp,
            "effectiveness": None,
            "crit": False,
            "source": "effect" if kind != "-sethp" else "sethp",
            "raw_parts": raw_parts,
        }

    if kind == "faint" and len(parts) >= 2:
        ident = parts[1]
        side = _side_from_ident(ident)
        uid = active_slots.get(f"{side}a") if side else None
        if uid and side:
            state["hp"][uid] = 0
            fainted_order[side].append(uid)
        return {
            "seq": seq,
            "type": "faint",
            "target_uid": uid,
            "raw_parts": raw_parts,
        }

    if kind in {"-status", "status"} and len(parts) >= 3:
        ident = parts[1]
        side = _side_from_ident(ident)
        uid = active_slots.get(f"{side}a") if side else None
        if uid:
            state["status"][uid] = True
        return {
            "seq": seq,
            "type": "status_start",
            "target_uid": uid,
            "status": _norm(parts[2]),
            "raw_parts": raw_parts,
        }

    if kind in {"-curestatus", "curestatus"} and len(parts) >= 3:
        ident = parts[1]
        side = _side_from_ident(ident)
        uid = active_slots.get(f"{side}a") if side else None
        if uid:
            state["status"][uid] = False
        return {
            "seq": seq,
            "type": "status_end",
            "target_uid": uid,
            "status": _norm(parts[2]),
            "raw_parts": raw_parts,
        }

    if kind in {"detailschange", "-formechange"} and len(parts) >= 3:
        ident = parts[1]
        side = _side_from_ident(ident)
        uid = active_slots.get(f"{side}a") if side else None
        parsed = _parse_details(parts[2])
        if uid and parsed.get("species"):
            state["species"][uid] = _norm(parsed["species"])
        return {
            "seq": seq,
            "type": "effect",
            "effect_type": _norm(kind.lstrip("-")),
            "raw_parts": raw_parts,
        }

    if kind.startswith("-"):
        effect_type = _norm(kind.lstrip("-"))
        if effect_type == "terastallize" and len(parts) >= 2:
            side = _side_from_ident(parts[1])
            if side:
                state["tera_used"][side] = True
                uid = active_slots.get(f"{side}a")
                teams = team_revelation.get("teams", {}) or {}
                for mon in teams.get(side, []):
                    if mon.get("pokemon_uid") == uid and len(parts) >= 3:
                        mon["known_tera_type"] = str(parts[2])
                        break
        return {
            "seq": seq,
            "type": "effect",
            "effect_type": effect_type,
            "raw_parts": raw_parts,
        }

    return None


def _parse_replay(payload: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    start_ms = int(time.time() * 1000)
    log_text = _extract_log_text(payload)
    if not log_text.strip():
        return None, "missing_log"

    battle_id = str(payload.get("id") or payload.get("battle_id") or "").strip()
    if not battle_id:
        return None, "missing_battle_id"
    format_id = str(payload.get("format") or payload.get("formatid") or payload.get("_scrape", {}).get("format_id") or "").strip()
    if not format_id:
        format_id = battle_id.split("-", 1)[0]

    state = _initial_state()
    active_slots: dict[str, str] = {}
    team_revelation: dict[str, Any] = {"teams": {"p1": [], "p2": []}}
    fainted_order: dict[str, list[str]] = {"p1": [], "p2": []}

    players = {
        "p1": {
            "name": _raw_player_name(payload, "p1"),
            "ladder_rating_pre": None,
        },
        "p2": {
            "name": _raw_player_name(payload, "p2"),
            "ladder_rating_pre": None,
        },
    }
    replay_rating = payload.get("rating")
    scrape_meta = payload.get("_scrape") if isinstance(payload.get("_scrape"), dict) else {}
    if replay_rating is None:
        replay_rating = scrape_meta.get("rating")

    lines = [line.strip() for line in log_text.splitlines() if line.strip()]
    turns: list[dict[str, Any]] = []
    current_turn: dict[str, Any] | None = None
    current_turn_number = 0
    winner_name = None
    turn_event_seq = 0

    for line in lines:
        parts = _split_protocol(line)
        if not parts:
            continue
        kind = parts[0]
        if kind == "player" and len(parts) >= 3:
            side = _side_from_ident(parts[1])
            if side:
                players[side]["name"] = parts[2]
            continue
        if kind == "win" and len(parts) >= 2:
            winner_name = parts[1]
            continue
        if kind == "turn" and len(parts) >= 2:
            if current_turn is not None:
                current_turn["state_after"] = copy.deepcopy(state)
                turns.append(current_turn)
            try:
                current_turn_number = int(parts[1])
            except Exception:
                current_turn_number = len(turns) + 1
            current_turn = {
                "turn_number": current_turn_number,
                "events": [],
                "state_after": None,
            }
            turn_event_seq = 0
            continue

        event = _event_from_parts(
            parts,
            state=state,
            team_revelation=team_revelation,
            active_slots=active_slots,
            turn_number=current_turn_number,
            seq=turn_event_seq,
            fainted_order=fainted_order,
        )
        if event is None:
            continue
        if current_turn is None:
            continue
        current_turn["events"].append(event)
        turn_event_seq += 1

    if current_turn is not None:
        current_turn["state_after"] = copy.deepcopy(state)
        turns.append(current_turn)

    winner_side = None
    if isinstance(winner_name, str) and winner_name.strip():
        winner_norm = _norm(winner_name)
        for side in ("p1", "p2"):
            if winner_norm == _norm(players[side].get("name")):
                winner_side = side
                break

    total_turns = len(turns)
    out = {
        "schema_version": SCHEMA_VERSION,
        "battle_id": battle_id,
        "format_id": format_id,
        "metadata": {
            "timestamp_unix": payload.get("uploadtime") or scrape_meta.get("uploadtime"),
            "total_turns": total_turns,
            "replay_rating": replay_rating,
            "outcome": {
                "winner": winner_name,
                "winner_name": winner_name,
                "winner_side": winner_side,
                "winner_player": winner_side,
            },
            "source": {
                "site": "replay.pokemonshowdown.com",
                "search_page": scrape_meta.get("search_page"),
            },
        },
        "players": players,
        "team_revelation": team_revelation,
        "turns": turns,
        "summary": {
            "pokemon_used": {
                side: list(state["team_order"].get(side, []))
                for side in ("p1", "p2")
            },
            "fainted_order": fainted_order,
            "major_events": None,
        },
        "integrity": {
            "validated": total_turns > 0,
            "issues": [] if total_turns > 0 else ["no_turns_parsed"],
            "sha256_raw_log": _hash_log(log_text),
            "generation_time_ms": int(time.time() * 1000) - start_ms,
        },
    }
    return out, None


def main() -> int:
    parser = argparse.ArgumentParser(description="Parse raw Showdown replay JSON into local structured JSON.")
    parser.add_argument("--input", nargs="+", required=True, help="Files, dirs, or globs of raw replay JSON")
    parser.add_argument("--output-dir", default="data/showdown/parsed")
    parser.add_argument("--summary-out", default="logs/replay_audit/showdown_parse.summary.json")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    Path(args.summary_out).parent.mkdir(parents=True, exist_ok=True)

    inputs: list[Path] = []
    for spec in args.input:
        path = Path(spec)
        if path.is_dir():
            inputs.extend(sorted(path.glob("*.json")))
        elif path.exists():
            inputs.append(path)
        else:
            inputs.extend(Path(p) for p in sorted(glob.glob(spec)))
    seen: set[str] = set()
    unique_inputs: list[Path] = []
    for path in inputs:
        key = str(path.resolve())
        if key not in seen:
            seen.add(key)
            unique_inputs.append(path)

    counters = Counter()
    for raw_path in unique_inputs:
        counters["files_seen"] += 1
        try:
            payload = json.loads(raw_path.read_text(encoding="utf-8"))
        except Exception:
            counters["json_load_failed"] += 1
            continue

        parsed, error = _parse_replay(payload)
        if parsed is None:
            counters[f"parse_failed_{error or 'unknown'}"] += 1
            continue
        battle_id = str(parsed.get("battle_id") or raw_path.stem)
        out_path = output_dir / f"{battle_id}.json"
        if out_path.exists() and not args.overwrite:
            counters["skip_existing"] += 1
            continue
        out_path.write_text(json.dumps(parsed, ensure_ascii=True, indent=2), encoding="utf-8")
        counters["parsed_ok"] += 1

    summary = {
        "input_count": len(unique_inputs),
        "output_dir": str(output_dir),
        "stats": dict(counters),
        "generated_at_unix": int(time.time()),
    }
    Path(args.summary_out).write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")
    print(f"Summary -> {args.summary_out}")
    print(json.dumps(summary["stats"], ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
