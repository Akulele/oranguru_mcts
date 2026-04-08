#!/usr/bin/env python3
"""
Audit engine decision traces for suspicious behavior patterns.

Supports raw search-trace JSONL files and prepared PKL trace datasets.
The main use is to flag behavior that is syntactically legal but tactically
abnormal, such as:
- repeated passive moves in the same matchup
- status moves into already-statused or immune targets
- Protect/recovery spam
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import glob
import json
import pickle
from pathlib import Path
from typing import Iterable, Optional

PROJECT_ROOT = Path(__file__).parent.parent
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.players.rule_bot import RuleBotPlayer
from src.utils.damage_calc import normalize_name
from src.utils.features import load_moves


def _resolve_inputs(patterns: list[str]) -> list[str]:
    paths: list[str] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            paths.extend(matches)
        elif Path(pattern).exists():
            paths.append(pattern)
    return paths


def _iter_examples(path: str) -> Iterable[dict]:
    file_path = Path(path)
    if file_path.suffix == ".pkl":
        with file_path.open("rb") as handle:
            obj = pickle.load(handle)
        if isinstance(obj, list):
            for item in obj:
                if isinstance(item, dict):
                    yield item
        return

    with file_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except Exception:
                continue
            if isinstance(item, dict):
                yield item


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _normalize_status(value: object) -> str:
    return normalize_name(str(value or ""))


def _choice_label(row: dict) -> str:
    choice = str(row.get("chosen_choice", "") or "")
    if choice:
        return choice
    labels = row.get("action_labels") or []
    idx = row.get("chosen_action")
    if isinstance(labels, list) and isinstance(idx, int) and 0 <= idx < len(labels):
        return str(labels[idx] or "")
    return ""


def _move_id_from_choice(choice: str) -> str:
    if not choice:
        return ""
    if choice.startswith("switch "):
        return ""
    return normalize_name(choice.replace("-tera", ""))


def _choice_kind(choice: str) -> str:
    if choice.startswith("switch "):
        return "switch"
    if choice.endswith("-tera"):
        return "tera_move"
    if choice:
        return "move"
    return ""


def _status_from_move_entry(entry: dict) -> Optional[str]:
    if not entry:
        return None
    status = normalize_name(entry.get("status", ""))
    if status in {"psn", "tox"}:
        return "poison"
    if status == "brn":
        return "burn"
    if status == "par":
        return "para"
    if status == "slp":
        return "sleep"
    vol = normalize_name(entry.get("volatileStatus", ""))
    if vol == "yawn":
        return "yawn"
    if vol in {"taunt", "encore"}:
        return vol
    if normalize_name(entry.get("name", "")) == "strengthsap":
        return "sap"
    return None


def _passive_kind(move_id: str, moves_data: dict) -> str:
    if not move_id:
        return ""
    if move_id in RuleBotPlayer.PROTECT_MOVES:
        return "protect"
    if move_id in RuleBotPlayer.RECOVERY_MOVES:
        return "recovery"
    if move_id in RuleBotPlayer.ANTI_SETUP_MOVES:
        return "status"
    entry = moves_data.get(move_id, {}) or {}
    status_type = RuleBotPlayer.STATUS_MOVES.get(move_id) or _status_from_move_entry(entry)
    if status_type:
        return "status"
    boosts = entry.get("boosts") or {}
    self_boosts = (entry.get("self") or {}).get("boosts") or {}
    self_boost = entry.get("selfBoost") or {}
    target = normalize_name(str(entry.get("target", "") or ""))
    if (boosts or self_boosts or self_boost) and target in {"self", "allyside", "selfside"}:
        return "setup"
    category = normalize_name(entry.get("category", ""))
    if category == "status":
        return "status"
    return ""


def _oracle_side_active(fp_oracle_battle: dict, side: str) -> Optional[dict]:
    if not isinstance(fp_oracle_battle, dict):
        return None
    return ((fp_oracle_battle.get(side) or {}).get("active")) or None


def _oracle_side_reserve(fp_oracle_battle: dict, side: str) -> list[dict]:
    if not isinstance(fp_oracle_battle, dict):
        return []
    reserve = ((fp_oracle_battle.get(side) or {}).get("reserve")) or []
    return [r for r in reserve if isinstance(r, dict)]


def _species(mon: Optional[dict]) -> str:
    if not isinstance(mon, dict):
        return ""
    return normalize_name(mon.get("name") or mon.get("base_name") or "")


def _types(mon: Optional[dict]) -> set[str]:
    if not isinstance(mon, dict):
        return set()
    return {normalize_name(t) for t in list(mon.get("types", []) or []) if t}


def _status(mon: Optional[dict]) -> str:
    if not isinstance(mon, dict):
        return ""
    return _normalize_status(mon.get("status"))


def _hp_frac(mon: Optional[dict]) -> float:
    if not isinstance(mon, dict):
        return 1.0
    hp = _safe_float(mon.get("hp", 0.0), 0.0)
    max_hp = _safe_float(mon.get("max_hp", 0.0), 0.0)
    if max_hp <= 0.0:
        return 1.0
    return max(0.0, min(1.0, hp / max_hp))


def _boost(mon: Optional[dict], stat: str) -> int:
    if not isinstance(mon, dict):
        return 0
    boosts = mon.get("boosts") or {}
    try:
        return int(boosts.get(stat, 0) or 0)
    except Exception:
        return 0


def _sleep_clause_active(fp_oracle_battle: dict) -> bool:
    for mon in [_oracle_side_active(fp_oracle_battle, "opponent"), *_oracle_side_reserve(fp_oracle_battle, "opponent")]:
        if _status(mon) in {"slp", "sleep"}:
            return True
    return False


def _issue_base(row: dict, choice: str, fp_oracle_battle: Optional[dict]) -> dict:
    user_active = _oracle_side_active(fp_oracle_battle or {}, "user")
    opp_active = _oracle_side_active(fp_oracle_battle or {}, "opponent")
    return {
        "battle_id": str(row.get("battle_id", "") or ""),
        "turn": int(row.get("turn", 0) or 0),
        "choice": choice,
        "selection_path": str(row.get("selection_path", "") or ""),
        "source": str(row.get("source", "") or ""),
        "tag": str(row.get("tag", "") or ""),
        "active_species": _species(user_active),
        "opponent_species": _species(opp_active),
    }


def analyze_examples(
    examples: Iterable[dict],
    *,
    moves_data: dict,
    repeat_streak: int = 3,
    passive_repeat_streak: int = 2,
    high_hp_recovery: float = 0.75,
    setup_reply_threshold: float = 160.0,
    setup_low_hp: float = 0.55,
    sample_limit: int = 20,
) -> dict:
    rows = [row for row in examples if isinstance(row, dict)]
    rows.sort(key=lambda r: (str(r.get("battle_id", "") or ""), int(r.get("turn", 0) or 0)))

    issue_counts = Counter()
    choice_counts = Counter()
    issue_choice_counts = defaultdict(Counter)
    samples_by_issue: dict[str, list[dict]] = defaultdict(list)
    battle_repeat_state: dict[str, dict] = {}
    battles_seen = set()

    for row in rows:
        battle_id = str(row.get("battle_id", "") or "")
        if not battle_id:
            continue
        battles_seen.add(battle_id)
        choice = _choice_label(row)
        if not choice:
            continue
        choice_counts[choice] += 1
        fp_oracle_battle = row.get("fp_oracle_battle")
        base = _issue_base(row, choice, fp_oracle_battle if isinstance(fp_oracle_battle, dict) else None)
        move_id = _move_id_from_choice(choice)
        kind = _choice_kind(choice)
        entry = moves_data.get(move_id, {}) if move_id else {}
        status_type = RuleBotPlayer.STATUS_MOVES.get(move_id) or _status_from_move_entry(entry)
        passive = _passive_kind(move_id, moves_data)

        def add_issue(category: str, **extra) -> None:
            issue_counts[category] += 1
            issue_choice_counts[category][choice] += 1
            if len(samples_by_issue[category]) < sample_limit:
                sample = dict(base)
                sample.update(extra)
                samples_by_issue[category].append(sample)

        if isinstance(fp_oracle_battle, dict):
            user_active = _oracle_side_active(fp_oracle_battle, "user")
            opp_active = _oracle_side_active(fp_oracle_battle, "opponent")
            opp_status = _status(opp_active)
            opp_types = _types(opp_active)
            active_hp = _hp_frac(user_active)
            opp_atk = _boost(opp_active, "attack")
            opp_spa = _boost(opp_active, "special-attack")

            if status_type in {"poison", "burn", "para", "sleep", "yawn"} and opp_status:
                add_issue("status_into_statused_target", opponent_status=opp_status)
            if status_type == "sleep" and _sleep_clause_active(fp_oracle_battle):
                add_issue("sleep_clause_risk")
            if status_type == "poison" and (("steel" in opp_types) or ("poison" in opp_types)):
                add_issue("status_into_type_immunity", opponent_types=sorted(opp_types))
            if status_type == "burn" and "fire" in opp_types:
                add_issue("status_into_type_immunity", opponent_types=sorted(opp_types))
            if status_type == "para" and move_id == "thunderwave" and (("ground" in opp_types) or ("electric" in opp_types)):
                add_issue("status_into_type_immunity", opponent_types=sorted(opp_types))
            if status_type == "sap" and active_hp > 0.45 and opp_atk <= opp_spa:
                add_issue("strength_sap_low_value", active_hp=round(active_hp, 3), opp_atk_boost=opp_atk, opp_spa_boost=opp_spa)
            if passive == "setup":
                reply_score = _safe_float(row.get("best_reply_score", 0.0), 0.0)
                if active_hp <= setup_low_hp and reply_score >= setup_reply_threshold:
                    add_issue(
                        "setup_under_pressure",
                        active_hp=round(active_hp, 3),
                        best_reply_score=round(reply_score, 3),
                    )

        state = battle_repeat_state.get(battle_id)
        active_species = base["active_species"]
        opp_species = base["opponent_species"]
        same_context = bool(
            state
            and state.get("choice") == choice
            and state.get("active_species") == active_species
            and state.get("opponent_species") == opp_species
            and int(row.get("turn", 0) or 0) == int(state.get("turn", 0)) + 1
        )
        if same_context:
            streak = int(state.get("streak", 1) or 1) + 1
        else:
            streak = 1
        battle_repeat_state[battle_id] = {
            "choice": choice,
            "active_species": active_species,
            "opponent_species": opp_species,
            "turn": int(row.get("turn", 0) or 0),
            "streak": streak,
            "passive": passive,
            "kind": kind,
        }

        if kind in {"move", "tera_move"} and streak >= repeat_streak:
            add_issue("repeat_move_streak", streak=streak, passive_kind=passive)
        if passive and streak >= passive_repeat_streak:
            add_issue("repeat_passive_move", streak=streak, passive_kind=passive)
        if passive == "protect" and streak >= 2:
            add_issue("protect_spam", streak=streak)
        if passive == "recovery" and streak >= 2:
            active_hp = _hp_frac(_oracle_side_active(fp_oracle_battle or {}, "user"))
            if active_hp >= high_hp_recovery:
                add_issue("recovery_spam_high_hp", streak=streak, active_hp=round(active_hp, 3))

    summary = {
        "rows_seen": len(rows),
        "battles_seen": len(battles_seen),
        "issue_counts": dict(issue_counts),
        "top_choices": choice_counts.most_common(30),
        "issue_top_choices": {
            category: counts.most_common(15) for category, counts in issue_choice_counts.items()
        },
        "samples": dict(samples_by_issue),
        "config": {
            "repeat_streak": int(repeat_streak),
            "passive_repeat_streak": int(passive_repeat_streak),
            "high_hp_recovery": float(high_hp_recovery),
            "setup_reply_threshold": float(setup_reply_threshold),
            "setup_low_hp": float(setup_low_hp),
        },
    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit engine traces for suspicious behavior.")
    parser.add_argument("--input", action="append", required=True, help="JSONL/PKL path or glob. Repeatable.")
    parser.add_argument("--summary-out", default="")
    parser.add_argument("--repeat-streak", type=int, default=3)
    parser.add_argument("--passive-repeat-streak", type=int, default=2)
    parser.add_argument("--high-hp-recovery", type=float, default=0.75)
    parser.add_argument("--setup-reply-threshold", type=float, default=160.0)
    parser.add_argument("--setup-low-hp", type=float, default=0.55)
    parser.add_argument("--sample-limit", type=int, default=20)
    args = parser.parse_args()

    paths = _resolve_inputs(args.input)
    if not paths:
        raise SystemExit("No input files matched.")

    rows: list[dict] = []
    for path in paths:
        rows.extend(_iter_examples(path))

    summary = analyze_examples(
        rows,
        moves_data=load_moves(),
        repeat_streak=max(2, args.repeat_streak),
        passive_repeat_streak=max(2, args.passive_repeat_streak),
        high_hp_recovery=max(0.0, min(1.0, args.high_hp_recovery)),
        setup_reply_threshold=max(0.0, args.setup_reply_threshold),
        setup_low_hp=max(0.0, min(1.0, args.setup_low_hp)),
        sample_limit=max(1, args.sample_limit),
    )

    print(f"Rows seen: {summary['rows_seen']}")
    print(f"Battles seen: {summary['battles_seen']}")
    print("Issue counts:")
    for category, count in sorted(summary["issue_counts"].items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {category}: {count}")

    if summary["issue_counts"]:
        print("Top issue choices:")
        for category, items in sorted(summary["issue_top_choices"].items()):
            if not items:
                continue
            head = ", ".join(f"{choice}:{count}" for choice, count in items[:8])
            print(f"  {category}: {head}")

    if args.summary_out:
        out_path = Path(args.summary_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        print(f"Summary -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
