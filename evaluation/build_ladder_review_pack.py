#!/usr/bin/env python3
"""Build a ranked human-review queue from ladder metrics and search traces.

The goal is to turn a ladder canary into a small set of high-leverage decisions
for human labeling.  Rating residual remains the scoreboard metric; this pack is
the "why did it win/lose?" metric.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import pickle
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).parent.parent
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.audit_engine_behavior import (  # noqa: E402
    _hp_frac,
    _iter_examples,
    _oracle_side_active,
    _oracle_side_reserve,
    _resolve_inputs,
    _safe_float,
    _species,
)
from evaluation.build_decision_review_pack import build_review_pack  # noqa: E402
from evaluation.mine_bad_decisions import mine_examples  # noqa: E402
from src.utils.features import load_moves  # noqa: E402


LABEL_SCHEMA = {
    "quality": ["good", "minor_mistake", "clear_blunder", "unclear"],
    "category": [
        "missed_ko",
        "bad_switch",
        "overpassive",
        "missed_recovery",
        "bad_setup",
        "tera_error",
        "sack_error",
        "endgame_plan",
        "other",
    ],
}


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _metric_rows(path: str | Path) -> list[dict[str, Any]]:
    return [
        row
        for row in _load_jsonl(path)
        if row.get("schema_version") == 1 and row.get("result") in {"win", "loss", "tie"}
    ]


def _load_teacher_rows(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with Path(path).open("rb") as handle:
            obj = pickle.load(handle)
        if isinstance(obj, list):
            rows.extend(item for item in obj if isinstance(item, dict))
    return rows


def _choice(row: dict[str, Any]) -> str:
    choice = str(row.get("chosen_choice", "") or "")
    if choice:
        return choice
    labels = row.get("action_labels") or []
    idx = row.get("chosen_action")
    if isinstance(labels, list) and isinstance(idx, int) and 0 <= idx < len(labels):
        return str(labels[idx] or "")
    return ""


def _top_actions(row: dict[str, Any]) -> list[dict[str, Any]]:
    actions = row.get("top_actions") or []
    return [a for a in actions if isinstance(a, dict)]


def _score(action: dict[str, Any] | None) -> float:
    if not action:
        return 0.0
    return _safe_float(action.get("score", action.get("weight")), 0.0)


def _heuristic(action: dict[str, Any] | None) -> float:
    if not action:
        return 0.0
    return _safe_float(action.get("heuristic_score"), 0.0)


def _choice_action(row: dict[str, Any], choice: str) -> dict[str, Any] | None:
    for action in _top_actions(row):
        if str(action.get("choice", "") or "") == choice:
            return action
    return None


def _teacher_lookup(teacher_rows: list[dict[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    lookup: dict[tuple[str, int], dict[str, Any]] = {}
    for row in teacher_rows:
        key = (str(row.get("battle_id", "") or ""), int(row.get("turn", 0) or 0))
        if not key[0]:
            continue
        lookup[key] = row
    return lookup


def _teacher_decision(row: dict[str, Any] | None, chosen_choice: str) -> dict[str, Any] | None:
    if not isinstance(row, dict):
        return None
    labels = row.get("action_labels") or []
    probs = row.get("policy_target") or []
    mask = row.get("action_mask") or []
    if not isinstance(labels, list) or not isinstance(probs, list):
        return None
    best_choice = ""
    best_prob = -1.0
    chosen_prob = 0.0
    distribution: list[dict[str, Any]] = []
    for idx, label in enumerate(labels):
        if idx >= len(probs):
            continue
        if isinstance(mask, list) and idx < len(mask) and not mask[idx]:
            continue
        choice = str(label or "")
        if not choice:
            continue
        prob = max(0.0, _safe_float(probs[idx], 0.0))
        distribution.append({"choice": choice, "prob": round(prob, 4)})
        if choice == chosen_choice:
            chosen_prob = prob
        if prob > best_prob:
            best_prob = prob
            best_choice = choice
    if not best_choice:
        return None
    distribution.sort(key=lambda item: -float(item.get("prob", 0.0) or 0.0))
    return {
        "source": str(row.get("teacher_source", "") or "teacher"),
        "top_choice": best_choice,
        "top_prob": round(best_prob, 4),
        "chosen_prob": round(chosen_prob, 4),
        "delta_top_minus_chosen": round(max(0.0, best_prob - chosen_prob), 4),
        "entropy": round(_safe_float(row.get("teacher_entropy"), 0.0), 4),
        "samples_used": int(row.get("teacher_samples_used", row.get("teacher_worlds_used", 0)) or 0),
        "total_visits": round(_safe_float(row.get("teacher_total_visits"), 0.0), 1),
        "top_distribution": distribution[:5],
    }


def _alive_count(fp_oracle_battle: dict[str, Any], side: str) -> int | None:
    if not isinstance(fp_oracle_battle, dict):
        return None
    mons = [_oracle_side_active(fp_oracle_battle, side), *_oracle_side_reserve(fp_oracle_battle, side)]
    count = 0
    seen = False
    for mon in mons:
        if not isinstance(mon, dict):
            continue
        seen = True
        if _hp_frac(mon) > 0.0:
            count += 1
    return count if seen else None


def _mon_snapshot(mon: Any) -> dict[str, Any] | None:
    if not isinstance(mon, dict):
        return None
    hp = _hp_frac(mon)
    moves = mon.get("moves") or []
    move_names: list[str] = []
    if isinstance(moves, list):
        for move in moves[:6]:
            if isinstance(move, dict):
                name = str(move.get("name", "") or "")
            else:
                name = str(move or "")
            if name:
                move_names.append(name)
    return {
        "species": _species(mon),
        "hp": round(hp, 3),
        "status": str(mon.get("status", "") or ""),
        "boosts": dict(mon.get("boosts") or {}),
        "item": mon.get("item"),
        "ability": mon.get("ability"),
        "tera_type": mon.get("tera_type"),
        "fainted": bool(mon.get("fainted", False) or hp <= 0.0),
        "moves": move_names,
    }


def _side_context(fp_oracle_battle: dict[str, Any], side: str) -> dict[str, Any]:
    side_obj = fp_oracle_battle.get(side) if isinstance(fp_oracle_battle, dict) else None
    if not isinstance(side_obj, dict):
        side_obj = {}
    active = _mon_snapshot(_oracle_side_active(fp_oracle_battle, side))
    bench = [
        snap
        for snap in (_mon_snapshot(mon) for mon in _oracle_side_reserve(fp_oracle_battle, side))
        if snap is not None
    ]
    alive = 0
    if active and not active.get("fainted"):
        alive += 1
    alive += sum(1 for mon in bench if not mon.get("fainted"))
    visible = int(bool(active)) + len(bench)
    hidden_estimate = 0
    if side == "opponent":
        # Random battles only expose revealed opposing Pokemon.  Make that
        # uncertainty explicit so reviewers do not read known-alive as full
        # team state.
        hidden_estimate = max(0, 6 - visible)
    return {
        "active": active,
        "bench": bench,
        "alive": alive,
        "visible": visible,
        "hidden_estimate": hidden_estimate,
        "side_conditions": dict(side_obj.get("side_conditions") or {}),
        "trapped": bool(side_obj.get("trapped", False)),
        "wish": list(side_obj.get("wish") or []),
        "future_sight": list(side_obj.get("future_sight") or []),
    }


def _field_context(fp_oracle_battle: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(fp_oracle_battle, dict):
        return {}
    return {
        "turn": fp_oracle_battle.get("turn"),
        "weather": fp_oracle_battle.get("weather"),
        "weather_turns_remaining": fp_oracle_battle.get("weather_turns_remaining"),
        "field": fp_oracle_battle.get("field"),
        "field_turns_remaining": fp_oracle_battle.get("field_turns_remaining"),
        "trick_room": fp_oracle_battle.get("trick_room"),
        "trick_room_turns_remaining": fp_oracle_battle.get("trick_room_turns_remaining"),
        "gravity": fp_oracle_battle.get("gravity"),
        "force_switch": fp_oracle_battle.get("force_switch"),
    }


def _board_context(row: dict[str, Any]) -> dict[str, Any]:
    fp = row.get("fp_oracle_battle")
    if not isinstance(fp, dict):
        return {}
    return {
        "field": _field_context(fp),
        "user": _side_context(fp, "user"),
        "opponent": _side_context(fp, "opponent"),
    }


def _board_summary(row: dict[str, Any]) -> dict[str, Any]:
    fp = row.get("fp_oracle_battle")
    if not isinstance(fp, dict):
        return {
            "active_species": "",
            "opponent_species": "",
            "active_hp": None,
            "opponent_hp": None,
            "user_alive": None,
            "opp_alive": None,
            "opp_hidden_estimate": None,
        }
    user_active = _oracle_side_active(fp, "user")
    opp_active = _oracle_side_active(fp, "opponent")
    context = _board_context(row)
    opponent = context.get("opponent") if isinstance(context, dict) else {}
    return {
        "active_species": _species(user_active),
        "opponent_species": _species(opp_active),
        "active_hp": None if not isinstance(user_active, dict) else round(_hp_frac(user_active), 3),
        "opponent_hp": None if not isinstance(opp_active, dict) else round(_hp_frac(opp_active), 3),
        "user_alive": _alive_count(fp, "user"),
        "opp_alive": _alive_count(fp, "opponent"),
        "opp_hidden_estimate": opponent.get("hidden_estimate") if isinstance(opponent, dict) else None,
    }


def _nearby_turns(
    battle_rows: list[dict[str, Any]],
    *,
    turn: int,
    radius: int = 3,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in battle_rows:
        row_turn = int(row.get("turn", 0) or 0)
        if abs(row_turn - turn) > radius:
            continue
        choice = _choice(row)
        actions = _top_actions(row)
        top_choice = str(actions[0].get("choice", "") or "") if actions else ""
        board = _board_summary(row)
        out.append(
            {
                "turn": row_turn,
                "choice": choice,
                "top_choice": top_choice,
                "active_species": board.get("active_species"),
                "opponent_species": board.get("opponent_species"),
                "active_hp": board.get("active_hp"),
                "opponent_hp": board.get("opponent_hp"),
                "policy_confidence": round(_safe_float(row.get("policy_confidence"), 0.0), 4),
                "selection_path": str(row.get("selection_path", "") or ""),
            }
        )
    out.sort(key=lambda item: int(item.get("turn", 0) or 0))
    return out


def _score_drop(row: dict[str, Any], choice: str) -> float:
    actions = _top_actions(row)
    if not actions:
        return 0.0
    chosen = _choice_action(row, choice)
    return max(0.0, _score(actions[0]) - _score(chosen))


def _margin(row: dict[str, Any]) -> float:
    actions = _top_actions(row)
    if len(actions) >= 2:
        return _score(actions[0]) - _score(actions[1])
    if actions:
        return _score(actions[0])
    return 0.0


def _top_action_payload(row: dict[str, Any], limit: int = 5) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for action in _top_actions(row)[:limit]:
        out.append(
            {
                "choice": str(action.get("choice", "") or ""),
                "kind": str(action.get("kind", "") or ""),
                "score": round(_score(action), 4),
                "heuristic_score": round(_heuristic(action), 3),
                "risk_penalty": None
                if action.get("risk_penalty") is None
                else round(_safe_float(action.get("risk_penalty"), 0.0), 3),
            }
        )
    return out


def _issue_lookup(trace_rows: list[dict[str, Any]], *, sample_limit: int) -> dict[tuple[str, int, str], dict[str, Any]]:
    summary = mine_examples(trace_rows, moves_data=load_moves(), sample_limit=max(sample_limit, 100))
    pack = build_review_pack(summary, limit=max(sample_limit * 4, 200), losses_only=False)
    lookup: dict[tuple[str, int, str], dict[str, Any]] = {}
    for item in pack:
        key = (
            str(item.get("battle_id", "") or ""),
            int(item.get("turn", 0) or 0),
            str(item.get("choice", "") or ""),
        )
        existing = lookup.get(key)
        if existing is None or float(item.get("priority", 0.0) or 0.0) > float(
            existing.get("priority", 0.0) or 0.0
        ):
            lookup[key] = item
    return lookup


def _priority(
    row: dict[str, Any],
    ladder: dict[str, Any] | None,
    issue: dict[str, Any] | None,
    teacher: dict[str, Any] | None = None,
) -> tuple[float, list[str]]:
    choice = _choice(row)
    actions = _top_actions(row)
    top_choice = str(actions[0].get("choice", "") or "") if actions else ""
    confidence = _safe_float(row.get("policy_confidence"), 0.0)
    margin = _margin(row)
    score_drop = _score_drop(row, choice)
    turn = int(row.get("turn", 0) or 0)
    phase = str(row.get("phase", "") or "")
    path = str(row.get("selection_path", "") or "")
    board = _board_summary(row)
    user_alive = board.get("user_alive")
    opp_alive = board.get("opp_alive")
    opp_hidden_estimate = int(board.get("opp_hidden_estimate") or 0)
    user_low_remaining = isinstance(user_alive, int) and user_alive <= 2
    opp_total_remaining = None
    if isinstance(opp_alive, int):
        opp_total_remaining = opp_alive + max(0, opp_hidden_estimate)
    opp_low_remaining = (
        isinstance(opp_total_remaining, int)
        and opp_total_remaining <= 2
    )
    result = str((ladder or {}).get("result", "") or "")
    residual = _safe_float((ladder or {}).get("rating_residual"), 0.0)
    expected = _safe_float((ladder or {}).get("expected_score"), 0.0)
    opp_remaining = (ladder or {}).get("opp_remaining")

    score = 0.0
    reasons: list[str] = []

    if result == "loss":
        score += 24.0
        reasons.append("lost battle")
        if expected >= 0.55:
            score += 12.0
            reasons.append("expected to score well")
    if residual < 0.0:
        score += min(24.0, abs(residual) * 28.0)
        reasons.append(f"negative residual {residual:+.3f}")
    if isinstance(opp_remaining, (int, float)) and float(opp_remaining) <= 2.0 and result == "loss":
        score += 8.0
        reasons.append("close loss")

    if turn >= 20:
        score += min(14.0, (turn - 19) * 0.7)
        reasons.append("late-game turn")
    mutual_low_remaining = user_low_remaining and opp_low_remaining
    if phase == "end" and mutual_low_remaining:
        score += 8.0
        reasons.append("endgame phase")

    if mutual_low_remaining:
        score += 8.0
        reasons.append("low remaining mons")
    elif user_low_remaining:
        score += 4.0
        reasons.append("our low remaining mons")
    elif opp_low_remaining:
        score += 4.0
        reasons.append("opponent low remaining mons")
    if board["active_hp"] is not None and float(board["active_hp"]) <= 0.35:
        score += 5.0
        reasons.append("active low HP")
    if board["opponent_hp"] is not None and float(board["opponent_hp"]) <= 0.35:
        score += 5.0
        reasons.append("opponent low HP")

    if score_drop > 0.0:
        score += min(25.0, score_drop * 80.0)
        reasons.append(f"non-top1 score drop {score_drop:.3f}")
    if top_choice and choice and top_choice != choice:
        score += 7.0
        reasons.append(f"chose {choice} over top {top_choice}")
    if confidence and confidence < 0.35:
        score += min(12.0, (0.35 - confidence) * 35.0)
        reasons.append(f"low confidence {confidence:.3f}")
    if margin and margin < 0.08:
        score += min(8.0, (0.08 - margin) * 70.0)
        reasons.append(f"thin margin {margin:.3f}")
    if path == "rerank":
        score += 8.0
        reasons.append("rerank decision")

    if teacher:
        teacher_top = str(teacher.get("top_choice", "") or "")
        teacher_delta = _safe_float(teacher.get("delta_top_minus_chosen"), 0.0)
        teacher_top_prob = _safe_float(teacher.get("top_prob"), 0.0)
        if teacher_top and teacher_top != choice and teacher_delta >= 0.03:
            score += min(24.0, 8.0 + teacher_delta * 80.0)
            reasons.append(
                f"FP teacher prefers {teacher_top} by {teacher_delta:.3f}"
            )
        elif teacher_top == choice and teacher_top_prob >= 0.45:
            score -= 8.0
            reasons.append("FP teacher agrees")

    chosen_action = _choice_action(row, choice)
    chosen_kind = str((chosen_action or {}).get("kind", "") or "")
    if result == "loss" and chosen_kind in {"switch", "status", "recovery", "protect"}:
        score += 5.0
        reasons.append(f"loss with {chosen_kind} choice")

    if issue:
        issue_priority = float(issue.get("priority", 0.0) or 0.0)
        score += min(50.0, issue_priority * 0.6)
        category = str(issue.get("category", "") or "")
        if category:
            reasons.append(f"mined issue: {category}")

    return round(score, 3), reasons


def _review_prompt(item: dict[str, Any]) -> str:
    active = item.get("active_species") or "active"
    opp = item.get("opponent_species") or "opponent"
    choice = item.get("choice") or "unknown"
    top = item.get("top_choice") or ""
    if item.get("issue_blurb"):
        return str(item["issue_blurb"])
    if top and top != choice:
        return f"{active} into {opp}: reviewed choice {choice}; compare against top line {top}."
    return f"{active} into {opp}: review whether {choice} was strategically correct."


def _dedupe_and_cap(rows: list[dict[str, Any]], *, limit: int, max_per_battle: int) -> list[dict[str, Any]]:
    rows.sort(
        key=lambda item: (
            -float(item.get("priority", 0.0) or 0.0),
            str(item.get("battle_id", "")),
            int(item.get("turn", 0) or 0),
        )
    )
    seen: set[tuple[str, int, str]] = set()
    per_battle: Counter[str] = Counter()
    out: list[dict[str, Any]] = []
    for row in rows:
        key = (
            str(row.get("battle_id", "") or ""),
            int(row.get("turn", 0) or 0),
            str(row.get("choice", "") or ""),
        )
        battle_id = key[0]
        if key in seen:
            continue
        if max_per_battle > 0 and per_battle[battle_id] >= max_per_battle:
            continue
        seen.add(key)
        per_battle[battle_id] += 1
        out.append(row)
        if len(out) >= limit:
            break
    return out


def build_ladder_review_pack(
    *,
    ladder_rows: list[dict[str, Any]],
    trace_rows: list[dict[str, Any]],
    teacher_rows: list[dict[str, Any]] | None = None,
    limit: int = 25,
    max_per_battle: int = 3,
) -> dict[str, Any]:
    ladder_by_battle = {
        str(row.get("battle_tag", "") or ""): row
        for row in ladder_rows
        if str(row.get("battle_tag", "") or "")
    }
    trace_rows = [
        row
        for row in trace_rows
        if str(row.get("battle_id", "") or "") in ladder_by_battle
    ]
    rows_by_battle: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in trace_rows:
        rows_by_battle[str(row.get("battle_id", "") or "")].append(row)
    for rows in rows_by_battle.values():
        rows.sort(key=lambda item: int(item.get("turn", 0) or 0))
    issues = _issue_lookup(trace_rows, sample_limit=max(limit, 25)) if trace_rows else {}
    teacher_by_turn = _teacher_lookup(teacher_rows or [])
    candidates: list[dict[str, Any]] = []
    for row in trace_rows:
        battle_id = str(row.get("battle_id", "") or "")
        ladder = ladder_by_battle.get(battle_id)
        choice = _choice(row)
        key = (battle_id, int(row.get("turn", 0) or 0), choice)
        issue = issues.get(key)
        teacher = _teacher_decision(
            teacher_by_turn.get((battle_id, int(row.get("turn", 0) or 0))),
            choice,
        )
        priority, reasons = _priority(row, ladder, issue, teacher)
        if priority <= 0.0:
            continue
        actions = _top_actions(row)
        top = actions[0] if actions else None
        chosen_action = _choice_action(row, choice)
        board = _board_summary(row)
        item: dict[str, Any] = {
            "priority": priority,
            "battle_id": battle_id,
            "turn": int(row.get("turn", 0) or 0),
            "result": (ladder or {}).get("result"),
            "expected_score": (ladder or {}).get("expected_score"),
            "rating_residual": (ladder or {}).get("rating_residual"),
            "opponent_rating_pre": (ladder or {}).get("opponent_rating_pre"),
            "decision_ms_avg_battle": (ladder or {}).get("decision_ms_avg"),
            "choice": choice,
            "choice_kind": str((chosen_action or {}).get("kind", "") or ""),
            "top_choice": str((top or {}).get("choice", "") or ""),
            "top_kind": str((top or {}).get("kind", "") or ""),
            "score_drop": round(_score_drop(row, choice), 4),
            "margin": round(_margin(row), 4),
            "policy_confidence": round(_safe_float(row.get("policy_confidence"), 0.0), 4),
            "selection_path": str(row.get("selection_path", "") or ""),
            "phase": str(row.get("phase", "") or ""),
            "best_reply_score": round(_safe_float(row.get("best_reply_score"), 0.0), 3),
            "top_actions": _top_action_payload(row),
            "reasons": reasons[:8],
            "issue_category": None if issue is None else issue.get("category"),
            "issue_blurb": None if issue is None else issue.get("review_blurb"),
            "issue_priority": None if issue is None else issue.get("priority"),
            "teacher": teacher,
            "human_label": "",
            "human_category": "",
            "human_notes": "",
        }
        item.update(board)
        item["board_context"] = _board_context(row)
        item["nearby_turns"] = _nearby_turns(
            rows_by_battle.get(battle_id, []),
            turn=int(row.get("turn", 0) or 0),
        )
        item["review_prompt"] = _review_prompt(item)
        candidates.append(item)

    rows = _dedupe_and_cap(candidates, limit=max(1, limit), max_per_battle=max_per_battle)
    result_counts = Counter(str(row.get("result", "") or "") for row in ladder_rows)
    residuals = [
        _safe_float(row.get("rating_residual"), 0.0)
        for row in ladder_rows
        if row.get("rating_residual") is not None
    ]
    return {
        "rows": rows,
        "count": len(rows),
        "label_schema": LABEL_SCHEMA,
        "summary": {
            "ladder_battles": len(ladder_rows),
            "trace_rows": len(trace_rows),
            "teacher_rows": len(teacher_rows or []),
            "teacher_labeled_rows": sum(1 for row in rows if row.get("teacher")),
            "teacher_disagreements": sum(
                1
                for row in rows
                if (row.get("teacher") or {}).get("top_choice")
                and (row.get("teacher") or {}).get("top_choice") != row.get("choice")
            ),
            "rated_battles": len(residuals),
            "wins": result_counts.get("win", 0),
            "losses": result_counts.get("loss", 0),
            "ties": result_counts.get("tie", 0),
            "win_rate": result_counts.get("win", 0) / max(1, len(ladder_rows)),
            "avg_residual": mean(residuals) if residuals else None,
            "residual_sum": sum(residuals) if residuals else None,
            "avg_decision_ms": mean(
                [
                    _safe_float(row.get("decision_ms_avg"), 0.0)
                    for row in ladder_rows
                    if row.get("decision_ms_avg") is not None
                ]
            )
            if any(row.get("decision_ms_avg") is not None for row in ladder_rows)
            else None,
            "issue_category_counts": dict(
                Counter(str(row.get("issue_category") or "unmined") for row in rows)
            ),
        },
    }


def write_markdown(pack: dict[str, Any], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    summary = pack.get("summary") or {}
    lines: list[str] = []
    lines.append("# Ladder Review Pack")
    lines.append("")
    lines.append(
        "Battles: {ladder_battles} | W/L/T {wins}/{losses}/{ties} | residual_sum={residual_sum} | avg_decision_ms={avg_decision_ms}".format(
            **{
                **summary,
                "residual_sum": "n/a"
                if summary.get("residual_sum") is None
                else f"{float(summary['residual_sum']):+.2f}",
                "avg_decision_ms": "n/a"
                if summary.get("avg_decision_ms") is None
                else f"{float(summary['avg_decision_ms']):.1f}",
            }
        )
    )
    lines.append("")
    lines.append("Labels: good | minor_mistake | clear_blunder | unclear")
    lines.append("")
    for idx, row in enumerate(pack.get("rows") or [], start=1):
        lines.append(
            "## {idx}. priority={priority:.1f} {battle_id} t{turn} {result}".format(
                idx=idx,
                priority=float(row.get("priority", 0.0) or 0.0),
                battle_id=row.get("battle_id", ""),
                turn=row.get("turn", ""),
                result=row.get("result", ""),
            )
        )
        lines.append("")
        lines.append(f"- Board: {row.get('active_species') or '?'} into {row.get('opponent_species') or '?'}")
        lines.append(
            f"- Choice: {row.get('choice')} ({row.get('choice_kind')}) | top: {row.get('top_choice')} ({row.get('top_kind')})"
        )
        lines.append(
            f"- Search: conf={row.get('policy_confidence')} margin={row.get('margin')} score_drop={row.get('score_drop')} path={row.get('selection_path')}"
        )
        lines.append(
            f"- Ladder: expected={row.get('expected_score')} residual={row.get('rating_residual')} opp_pre={row.get('opponent_rating_pre')}"
        )
        teacher = row.get("teacher")
        if isinstance(teacher, dict):
            lines.append(
                f"- FP teacher: top={teacher.get('top_choice')} p={teacher.get('top_prob')} "
                f"chosen_p={teacher.get('chosen_prob')} delta={teacher.get('delta_top_minus_chosen')} "
                f"samples={teacher.get('samples_used')} visits={teacher.get('total_visits')}"
            )
            distribution = teacher.get("top_distribution") or []
            if distribution:
                dist_text = ", ".join(
                    f"{item.get('choice')}={item.get('prob')}" for item in distribution if isinstance(item, dict)
                )
                lines.append(f"- FP teacher top dist: {dist_text}")
        context = row.get("board_context") or {}
        user = context.get("user") if isinstance(context, dict) else {}
        opponent = context.get("opponent") if isinstance(context, dict) else {}
        field = context.get("field") if isinstance(context, dict) else {}
        if isinstance(user, dict) and isinstance(opponent, dict):
            lines.append(
                f"- Known remaining: us {user.get('alive', '?')} / opp {opponent.get('alive', '?')} "
                f"(opp hidden estimate {opponent.get('hidden_estimate', 0)})"
            )
            lines.append(
                f"- Field: weather={field.get('weather') if isinstance(field, dict) else None} "
                f"field={field.get('field') if isinstance(field, dict) else None} "
                f"trick_room={field.get('trick_room') if isinstance(field, dict) else None}"
            )
            lines.append(
                f"- Hazards: us [{_format_conditions(user.get('side_conditions'))}] "
                f"opp [{_format_conditions(opponent.get('side_conditions'))}]"
            )
            lines.append(f"- Our active: {_format_mon(user.get('active'))}")
            lines.append("- Our bench:")
            for mon in user.get("bench") or []:
                lines.append(f"  - {_format_mon(mon)}")
            lines.append(f"- Opp active: {_format_mon(opponent.get('active'))}")
            lines.append("- Opp revealed:")
            for mon in opponent.get("bench") or []:
                lines.append(f"  - {_format_mon(mon)}")
        lines.append(f"- Why selected: {', '.join(row.get('reasons') or [])}")
        lines.append(f"- Prompt: {row.get('review_prompt')}")
        nearby = row.get("nearby_turns") or []
        if nearby:
            lines.append("- Nearby decisions:")
            for near in nearby:
                marker = " <= review" if int(near.get("turn", -1) or -1) == int(row.get("turn", -2) or -2) else ""
                lines.append(
                    f"  - t{near.get('turn')}: {near.get('active_species')} vs {near.get('opponent_species')} "
                    f"choice={near.get('choice')} top={near.get('top_choice')} "
                    f"hp={near.get('active_hp')}/{near.get('opponent_hp')} conf={near.get('policy_confidence')}{marker}"
                )
        lines.append("- Top actions:")
        for action in row.get("top_actions") or []:
            lines.append(
                f"  - {action.get('choice')} kind={action.get('kind')} score={action.get('score')} heur={action.get('heuristic_score')} risk={action.get('risk_penalty')}"
            )
        lines.append("- Human label: ")
        lines.append("- Human category: ")
        lines.append("- Human notes: ")
        lines.append("")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format_mon(mon: dict[str, Any] | None) -> str:
    if not mon:
        return "?"
    bits = [
        str(mon.get("species") or "?"),
        f"hp={mon.get('hp')}",
    ]
    if mon.get("status"):
        bits.append(f"status={mon.get('status')}")
    boosts = mon.get("boosts") or {}
    if isinstance(boosts, dict):
        useful_boosts = {k: v for k, v in boosts.items() if v}
        if useful_boosts:
            bits.append(f"boosts={useful_boosts}")
    if mon.get("item"):
        bits.append(f"item={mon.get('item')}")
    if mon.get("ability"):
        bits.append(f"ability={mon.get('ability')}")
    moves = mon.get("moves") or []
    if moves:
        bits.append("moves=" + ",".join(str(m) for m in moves[:4]))
    if mon.get("fainted"):
        bits.append("fainted")
    return " ".join(bits)


def _format_conditions(conditions: Any) -> str:
    if not isinstance(conditions, dict) or not conditions:
        return "none"
    return ", ".join(f"{k}:{v}" for k, v in sorted(conditions.items()))


def print_pack(pack: dict[str, Any], *, limit: int) -> None:
    summary = pack.get("summary") or {}
    print(
        "Review pack: battles={ladder_battles} trace_rows={trace_rows} rated={rated_battles} residual_sum={residual_sum}".format(
            **{
                **summary,
                "residual_sum": "n/a"
                if summary.get("residual_sum") is None
                else f"{float(summary['residual_sum']):+.2f}",
            }
        )
    )
    print(f"Rows: {pack.get('count', 0)}")
    for row in (pack.get("rows") or [])[: max(0, limit)]:
        reasons = ", ".join(row.get("reasons") or [])
        print(
            f"{float(row.get('priority', 0.0) or 0.0):6.1f} "
            f"{row.get('battle_id')} t{row.get('turn')} {row.get('result')} "
            f"{row.get('choice')} -> top {row.get('top_choice')} | {reasons}"
        )
        print(f"       {row.get('review_prompt')}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ladder-log", required=True, help="Ladder metrics JSONL for this run")
    parser.add_argument("--trace", action="append", required=True, help="Search trace JSONL path or glob")
    parser.add_argument(
        "--teacher",
        action="append",
        default=[],
        help="Optional FP-oracle teacher PKL from training/relabel_teacher_from_fp_trace.py",
    )
    parser.add_argument("--output-json", required=True, help="Output review JSON path")
    parser.add_argument("--output-md", default="", help="Optional Markdown review worksheet")
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--max-per-battle", type=int, default=3)
    parser.add_argument("--print-limit", type=int, default=20)
    args = parser.parse_args()

    trace_paths = _resolve_inputs(args.trace)
    if not trace_paths:
        raise SystemExit("No trace files matched --trace.")
    teacher_paths = _resolve_inputs(args.teacher) if args.teacher else []
    ladder_rows = _metric_rows(args.ladder_log)
    if not ladder_rows:
        raise SystemExit("No metric rows found in --ladder-log.")
    trace_rows: list[dict[str, Any]] = []
    for path in trace_paths:
        trace_rows.extend(_iter_examples(path))
    teacher_rows = _load_teacher_rows(teacher_paths) if teacher_paths else []
    pack = build_ladder_review_pack(
        ladder_rows=ladder_rows,
        trace_rows=trace_rows,
        teacher_rows=teacher_rows,
        limit=max(1, args.limit),
        max_per_battle=max(0, args.max_per_battle),
    )
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(pack, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.output_md:
        write_markdown(pack, args.output_md)
    print_pack(pack, limit=args.print_limit)
    print(f"Review JSON -> {out_json}")
    if args.output_md:
        print(f"Review MD -> {args.output_md}")
    return 0


if __name__ == "__main__":
    # Some imported battle-analysis dependencies can leave background runtime
    # state alive in CLI use.  Flush outputs and exit directly so review-pack
    # generation is safe in nohup/server pipelines.
    import os

    exit_code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
