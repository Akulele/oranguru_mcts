#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).parent.parent
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.audit_engine_behavior import (
    _boost,
    _choice_kind,
    _choice_label,
    _hp_frac,
    _issue_base,
    _iter_examples,
    _move_id_from_choice,
    _move_type,
    _oracle_side_active,
    _oracle_side_reserve,
    _passive_kind,
    _resolve_inputs,
    _safe_float,
    _side_conditions,
    _status,
    _status_from_move_entry,
    _types,
)
from src.players.rule_bot import RuleBotPlayer
from src.utils.damage_calc import get_type_effectiveness, normalize_name
from src.utils.features import load_moves


PRIORITY_WEIGHTS = {
    "missed_ko": 100,
    "ignored_safe_recovery": 80,
    "underused_setup_window": 70,
    "underused_status_window": 65,
    "over_attacked_into_bad_trade": 60,
    "over_switched_negative_matchup": 55,
    "failed_to_progress_when_behind": 50,
}


def _top_actions(row: dict) -> list[dict]:
    actions = row.get("top_actions") or []
    if not isinstance(actions, list):
        return []
    return [a for a in actions if isinstance(a, dict) and a.get("choice")]


def _score_for_choice(row: dict, choice: str) -> Optional[float]:
    for action in _top_actions(row):
        if str(action.get("choice", "") or "") == choice:
            return _safe_float(action.get("score", action.get("weight")), 0.0)
    return None


def _heuristic_for_choice(row: dict, choice: str) -> Optional[float]:
    for action in _top_actions(row):
        if str(action.get("choice", "") or "") == choice and action.get("heuristic_score") is not None:
            return _safe_float(action.get("heuristic_score"), 0.0)
    return None


def _risk_for_choice(row: dict, choice: str) -> Optional[float]:
    for action in _top_actions(row):
        if str(action.get("choice", "") or "") == choice and action.get("risk_penalty") is not None:
            return _safe_float(action.get("risk_penalty"), 0.0)
    return None


def _find_alternative(row: dict, predicate) -> Optional[dict]:
    for action in _top_actions(row):
        choice = str(action.get("choice", "") or "")
        if predicate(choice, action):
            return action
    return None


def _alive_count(fp_oracle_battle: dict, side: str) -> int:
    mons = [_oracle_side_active(fp_oracle_battle, side), *_oracle_side_reserve(fp_oracle_battle, side)]
    count = 0
    for mon in mons:
        if not isinstance(mon, dict):
            continue
        if _hp_frac(mon) > 0.0:
            count += 1
    return count


def _available_move_ids(row: dict) -> set[str]:
    out: set[str] = set()
    for label in row.get("action_labels") or []:
        if not isinstance(label, str):
            continue
        move_id = _move_id_from_choice(label)
        if move_id:
            out.add(move_id)
    for action in _top_actions(row):
        move_id = _move_id_from_choice(str(action.get("choice", "") or ""))
        if move_id:
            out.add(move_id)
    return out


def _has_damaging_option(row: dict, moves_data: dict, opp_types: set[str]) -> bool:
    return _best_damaging_alternative(row, moves_data, opp_types) is not None


def _best_damaging_alternative(row: dict, moves_data: dict, opp_types: set[str], exclude_choice: str = "") -> dict | None:
    top_actions = _top_actions(row)
    candidates = top_actions if top_actions else [{"choice": label, "score": 0.0} for label in row.get("action_labels") or [] if isinstance(label, str)]
    best = None
    best_score = float("-inf")
    for action in candidates:
        choice = str(action.get("choice", "") or "")
        if not choice or choice == exclude_choice or choice.startswith("switch "):
            continue
        move_id = _move_id_from_choice(choice)
        if not move_id:
            continue
        entry = moves_data.get(move_id, {}) or {}
        if normalize_name(entry.get("category", "")) == "status":
            continue
        move_type = _move_type(entry)
        try:
            if move_type and get_type_effectiveness(move_type, sorted(opp_types)) <= 0.0:
                continue
        except Exception:
            pass
        score = _safe_float(action.get("score", action.get("weight")), 0.0)
        if best is None or score > best_score:
            best = {"choice": choice, "score": score}
            best_score = score
    return best


def _has_setup_option(row: dict, moves_data: dict) -> bool:
    for move_id in _available_move_ids(row):
        if _passive_kind(move_id, moves_data) == "setup":
            return True
    return False


def _best_setup_alternative(row: dict, moves_data: dict, exclude_choice: str = "") -> dict | None:
    top_actions = _top_actions(row)
    candidates = top_actions if top_actions else [{"choice": label, "score": 0.0} for label in row.get("action_labels") or [] if isinstance(label, str)]
    best = None
    best_score = float("-inf")
    for action in candidates:
        choice = str(action.get("choice", "") or "")
        if not choice or choice == exclude_choice or choice.startswith("switch "):
            continue
        move_id = _move_id_from_choice(choice)
        if not move_id or _passive_kind(move_id, moves_data) != "setup":
            continue
        score = _safe_float(action.get("score", action.get("weight")), 0.0)
        if best is None or score > best_score:
            best = {"choice": choice, "score": score}
            best_score = score
    return best


def _is_status_candidate(move_id: str, moves_data: dict, opp_types: set[str], opp_status: str) -> bool:
    if not move_id or opp_status:
        return False
    entry = moves_data.get(move_id, {}) or {}
    status_type = RuleBotPlayer.STATUS_MOVES.get(move_id) or _status_from_move_entry(entry)
    if not status_type or status_type in {"sap", "taunt", "encore"}:
        return False
    if status_type == "poison" and (("steel" in opp_types) or ("poison" in opp_types)):
        return False
    if status_type == "burn" and "fire" in opp_types:
        return False
    if status_type == "para" and move_id == "thunderwave" and (("ground" in opp_types) or ("electric" in opp_types)):
        return False
    return True


def _has_status_option(row: dict, moves_data: dict, opp_types: set[str], opp_status: str, fp_oracle_battle: dict) -> bool:
    if opp_status:
        return False
    for move_id in _available_move_ids(row):
        if _is_status_candidate(move_id, moves_data, opp_types, opp_status):
            return True
    return False


def _best_status_alternative(row: dict, moves_data: dict, opp_types: set[str], opp_status: str, exclude_choice: str = "") -> dict | None:
    top_actions = _top_actions(row)
    candidates = top_actions if top_actions else [{"choice": label, "score": 0.0} for label in row.get("action_labels") or [] if isinstance(label, str)]
    best = None
    best_score = float("-inf")
    for action in candidates:
        choice = str(action.get("choice", "") or "")
        if not choice or choice == exclude_choice or choice.startswith("switch "):
            continue
        move_id = _move_id_from_choice(choice)
        if not _is_status_candidate(move_id, moves_data, opp_types, opp_status):
            continue
        score = _safe_float(action.get("score", action.get("weight")), 0.0)
        if best is None or score > best_score:
            best = {"choice": choice, "score": score}
            best_score = score
    return best


def _has_recovery_option(row: dict, moves_data: dict) -> bool:
    for move_id in _available_move_ids(row):
        if _passive_kind(move_id, moves_data) == "recovery":
            return True
    return False


def _best_recovery_alternative(row: dict, moves_data: dict, exclude_choice: str = "") -> dict | None:
    top_actions = _top_actions(row)
    candidates = top_actions if top_actions else [{"choice": label, "score": 0.0} for label in row.get("action_labels") or [] if isinstance(label, str)]
    best = None
    best_score = float("-inf")
    for action in candidates:
        choice = str(action.get("choice", "") or "")
        if not choice or choice == exclude_choice or choice.startswith("switch "):
            continue
        move_id = _move_id_from_choice(choice)
        if _passive_kind(move_id, moves_data) != "recovery":
            continue
        score = _safe_float(action.get("score", action.get("weight")), 0.0)
        if best is None or score > best_score:
            best = {"choice": choice, "score": score}
            best_score = score
    return best


def _best_progress_alternative(
    row: dict,
    moves_data: dict,
    opp_types: set[str],
    opp_status: str,
    opp_hazard_free: bool,
    exclude_choice: str = "",
) -> dict | None:
    top_actions = _top_actions(row)
    candidates = top_actions if top_actions else [{"choice": label, "score": 0.0} for label in row.get("action_labels") or [] if isinstance(label, str)]
    best = None
    best_score = float("-inf")
    for action in candidates:
        choice = str(action.get("choice", "") or "")
        if not choice or choice == exclude_choice or choice.startswith("switch "):
            continue
        move_id = _move_id_from_choice(choice)
        if _is_status_candidate(move_id, moves_data, opp_types, opp_status):
            kind = "status"
        elif _passive_kind(move_id, moves_data) == "setup":
            kind = "setup"
        elif opp_hazard_free and move_id in RuleBotPlayer.ENTRY_HAZARDS:
            kind = "hazard"
        else:
            continue
        score = _safe_float(action.get("score", action.get("weight")), 0.0)
        if best is None or score > best_score:
            best = {"choice": choice, "score": score, "progress_kind": kind}
            best_score = score
    return best


def _is_damaging_move(choice: str, moves_data: dict) -> bool:
    move_id = _move_id_from_choice(choice)
    if not move_id:
        return False
    entry = moves_data.get(move_id, {}) or {}
    return normalize_name(entry.get("category", "")) != "status"


def _issue_priority(issue: dict) -> float:
    base = float(PRIORITY_WEIGHTS.get(issue.get("category", ""), 10))
    base += float(issue.get("score_gap", 0.0) or 0.0)
    base += 25.0 * float(issue.get("regret", 0.0) or 0.0)
    if issue.get("lost_battle"):
        base += 20.0
    return round(base, 3)


def _ratio_stats(values: list[float]) -> dict:
    cleaned = sorted(v for v in values if isinstance(v, (int, float)) and v >= 0.0)
    if not cleaned:
        return {}

    def q(frac: float) -> float:
        idx = min(len(cleaned) - 1, max(0, int(round((len(cleaned) - 1) * frac))))
        return round(float(cleaned[idx]), 3)

    return {
        "n": len(cleaned),
        "p50": q(0.50),
        "p75": q(0.75),
        "p90": q(0.90),
        "max": round(float(cleaned[-1]), 3),
        "near_075": sum(1 for v in cleaned if v >= 0.75),
        "near_090": sum(1 for v in cleaned if v >= 0.90),
    }


def mine_examples(
    examples: Iterable[dict],
    *,
    moves_data: dict,
    ko_hp_threshold: float = 0.35,
    missed_ko_min_finish_ratio: float = 0.75,
    safe_reply_threshold: float = 110.0,
    punish_reply_threshold: float = 180.0,
    low_hp_recovery: float = 0.4,
    min_score_gap: float = 15.0,
    sample_limit: int = 30,
) -> dict:
    rows = [row for row in examples if isinstance(row, dict)]
    rows.sort(key=lambda r: (str(r.get("battle_id", "") or ""), int(r.get("turn", 0) or 0)))

    issue_counts = Counter()
    issue_choice_counts = defaultdict(Counter)
    finish_blow_reasons = Counter()
    finish_blow_rows = 0
    finish_blow_stale_reasons = Counter()
    missed_ko_finish_reasons = Counter()
    missed_ko_finish_stale_reasons = Counter()
    missed_ko_finish_ratios: list[float] = []
    recovery_window_reasons = Counter()
    recovery_window_rows = 0
    switch_guard_reasons = Counter()
    switch_guard_rows = 0
    progress_window_reasons = Counter()
    progress_window_rows = 0
    setup_window_reasons = Counter()
    setup_window_rows = 0
    issue_window_reasons = defaultdict(Counter)
    samples_by_issue: dict[str, list[dict]] = defaultdict(list)
    battles_seen = set()

    for row in rows:
        finish_blow = row.get("finish_blow")
        if isinstance(finish_blow, dict):
            finish_blow_rows += 1
            reason = str(finish_blow.get("reason", "") or "")
            if reason:
                finish_blow_reasons[reason] += 1
            diag_choice = str(finish_blow.get("chosen_choice", "") or "")
            final_choice = str(row.get("chosen_choice", "") or "")
            if reason and diag_choice and final_choice and diag_choice != final_choice:
                finish_blow_stale_reasons[reason] += 1
        setup_window = row.get("setup_window")
        if isinstance(setup_window, dict):
            setup_window_rows += 1
            reason = str(setup_window.get("reason", "") or "")
            if reason:
                setup_window_reasons[reason] += 1
        recovery_window = row.get("recovery_window")
        if isinstance(recovery_window, dict):
            recovery_window_rows += 1
            reason = str(recovery_window.get("reason", "") or "")
            if reason:
                recovery_window_reasons[reason] += 1
        switch_guard = row.get("switch_guard")
        if isinstance(switch_guard, dict):
            switch_guard_rows += 1
            reason = str(switch_guard.get("reason", "") or "")
            if reason:
                switch_guard_reasons[reason] += 1
        progress_window = row.get("progress_window")
        if isinstance(progress_window, dict):
            progress_window_rows += 1
            reason = str(progress_window.get("reason", "") or "")
            if reason:
                progress_window_reasons[reason] += 1
        battle_id = str(row.get("battle_id", "") or "")
        if not battle_id:
            continue
        battles_seen.add(battle_id)
        fp_oracle_battle = row.get("fp_oracle_battle")
        if not isinstance(fp_oracle_battle, dict):
            continue
        choice = _choice_label(row)
        if not choice:
            continue
        base = _issue_base(row, choice, fp_oracle_battle)
        user_active = _oracle_side_active(fp_oracle_battle, "user")
        opp_active = _oracle_side_active(fp_oracle_battle, "opponent")
        if not isinstance(user_active, dict) or not isinstance(opp_active, dict):
            continue
        active_hp = _hp_frac(user_active)
        opp_hp = _hp_frac(opp_active)
        opp_status = _status(opp_active)
        opp_types = _types(opp_active)
        user_alive = _alive_count(fp_oracle_battle, "user")
        opp_alive = _alive_count(fp_oracle_battle, "opponent")
        behind = user_alive < opp_alive
        chosen_kind = _choice_kind(choice)
        chosen_score = _score_for_choice(row, choice)
        best_action = _top_actions(row)[0] if _top_actions(row) else None
        best_choice = str((best_action or {}).get("choice", "") or "")
        best_score = _safe_float((best_action or {}).get("score", (best_action or {}).get("weight")), 0.0) if best_action else 0.0
        score_gap = max(0.0, best_score - (chosen_score if chosen_score is not None else best_score))
        reply_score = _safe_float(row.get("best_reply_score"), 0.0)
        chosen_move_id = _move_id_from_choice(choice)
        chosen_passive = _passive_kind(chosen_move_id, moves_data)
        chosen_damaging = _is_damaging_move(choice, moves_data)
        active_boost_max = max((_boost(user_active, stat) for stat in ["attack", "special-attack", "speed"]), default=0)
        opp_hazards = _side_conditions(fp_oracle_battle, "opponent")

        def issue_window_diag(category: str) -> tuple[str, str, str]:
            window_by_category = {
                "missed_ko": "finish_blow",
                "underused_setup_window": "setup_window",
                "ignored_safe_recovery": "recovery_window",
                "over_switched_negative_matchup": "switch_guard",
                "failed_to_progress_when_behind": "progress_window",
            }
            window_name = window_by_category.get(category, "")
            if not window_name:
                return "", "", ""
            diagnostic = row.get(window_name)
            if not isinstance(diagnostic, dict):
                return window_name, "missing", ""
            reason = str(diagnostic.get("reason", "") or "missing")
            diag_choice = str(diagnostic.get("chosen_choice", "") or "")
            return window_name, reason, diag_choice

        def add_issue(category: str, **extra) -> None:
            issue_counts[category] += 1
            issue_choice_counts[category][choice] += 1
            runtime_window, runtime_reason, runtime_choice = issue_window_diag(category)
            if runtime_window:
                issue_window_reasons[category][runtime_reason] += 1
            if len(samples_by_issue[category]) >= sample_limit:
                return
            sample = dict(base)
            sample.update(
                {
                    "category": category,
                    "active_hp": round(active_hp, 3),
                    "opp_hp": round(opp_hp, 3),
                    "user_alive": user_alive,
                    "opp_alive": opp_alive,
                    "behind": behind,
                    "best_reply_score": round(reply_score, 3),
                    "best_choice": best_choice,
                    "best_score": round(best_score, 3),
                    "chosen_score": None if chosen_score is None else round(chosen_score, 3),
                    "score_gap": round(score_gap, 3),
                    "phase": str(row.get("phase", "") or ""),
                    "lost_battle": bool(row.get("winner") and str(row.get("winner")) != str(row.get("bot_id", row.get("player_name", "")))),
                }
            )
            if runtime_window:
                sample.update(
                    {
                        "runtime_window": runtime_window,
                        "runtime_window_reason": runtime_reason,
                        "runtime_window_choice": runtime_choice,
                    }
                )
            sample.update(extra)
            sample["priority"] = _issue_priority(sample)
            samples_by_issue[category].append(sample)

        if opp_hp <= ko_hp_threshold and not chosen_damaging:
            ko_alt = _best_damaging_alternative(row, moves_data, opp_types, exclude_choice=choice)
            if ko_alt is not None:
                alt_choice = str(ko_alt.get("choice", "") or "")
                alt_score = _safe_float(ko_alt.get("score", ko_alt.get("weight")), 0.0)
                alt_heur = _heuristic_for_choice(row, alt_choice)
                choice_heur = _heuristic_for_choice(row, choice)
                if chosen_score is not None and chosen_score > 0.0:
                    policy_ratio = alt_score / max(chosen_score, 1e-6)
                else:
                    policy_ratio = 1.0 if alt_score > 0.0 else 0.0
                if alt_heur is not None and choice_heur is not None:
                    heur_delta = alt_heur - choice_heur
                    should_flag_ko = (policy_ratio >= 0.70 and heur_delta >= 1.0) or (
                        policy_ratio >= 0.40 and heur_delta >= 3.0
                    )
                else:
                    should_flag_ko = True
                if should_flag_ko:
                    finish_reason = ""
                    finish_best_damage = None
                    finish_ko_threshold = None
                    finish_ko_ratio = None
                    finish_blow = row.get("finish_blow")
                    if isinstance(finish_blow, dict):
                        finish_reason = str(finish_blow.get("reason", "") or "")
                        if finish_reason:
                            missed_ko_finish_reasons[finish_reason] += 1
                        diag_choice = str(finish_blow.get("chosen_choice", "") or "")
                        final_choice = str(row.get("chosen_choice", "") or "")
                        if finish_reason and diag_choice and final_choice and diag_choice != final_choice:
                            missed_ko_finish_stale_reasons[finish_reason] += 1
                        finish_best_damage = _safe_float(finish_blow.get("best_damage_score"), 0.0)
                        finish_ko_threshold = _safe_float(finish_blow.get("ko_threshold"), 0.0)
                        if finish_ko_threshold > 0.0:
                            finish_ko_ratio = finish_best_damage / finish_ko_threshold
                            missed_ko_finish_ratios.append(finish_ko_ratio)
                    if (
                        finish_reason == "no_ko_window"
                        and finish_ko_ratio is not None
                        and finish_ko_ratio < missed_ko_min_finish_ratio
                    ):
                        continue
                    add_issue(
                        "missed_ko",
                        regret=ko_hp_threshold - opp_hp,
                        alternative=alt_choice,
                        alternative_score=round(alt_score, 3),
                        alternative_heuristic_score=None if alt_heur is None else round(alt_heur, 3),
                        chosen_heuristic_score=None if choice_heur is None else round(choice_heur, 3),
                        policy_ratio=round(policy_ratio, 3),
                        best_choice=alt_choice,
                        best_score=round(alt_score, 3),
                        score_gap=round(max(0.0, alt_score - (chosen_score if chosen_score is not None else alt_score)), 3),
                        finish_blow_reason=finish_reason,
                        finish_best_damage_score=None if finish_best_damage is None else round(finish_best_damage, 3),
                        finish_ko_threshold=None if finish_ko_threshold is None else round(finish_ko_threshold, 3),
                        finish_ko_ratio=None if finish_ko_ratio is None else round(finish_ko_ratio, 3),
                    )

        if chosen_kind == "switch" and not bool(row.get("force_switch")) and active_hp >= 0.45:
            alt_attack = _find_alternative(
                row,
                lambda alt_choice, _a: alt_choice != choice and _is_damaging_move(alt_choice, moves_data),
            )
            if alt_attack:
                alt_score = _safe_float(alt_attack.get("score", alt_attack.get("weight")), 0.0)
                gap = alt_score - (chosen_score if chosen_score is not None else alt_score)
                alt_choice = str(alt_attack.get("choice", "") or "")
                alt_heur = _heuristic_for_choice(row, alt_choice)
                choice_heur = _heuristic_for_choice(row, choice)
                choice_risk = _risk_for_choice(row, choice)
                if chosen_score is not None and chosen_score > 0.0:
                    policy_ratio = alt_score / max(chosen_score, 1e-6)
                else:
                    policy_ratio = 1.0 if alt_score > 0.0 else 0.0
                if alt_heur is not None and choice_heur is not None:
                    heur_delta = alt_heur - choice_heur
                    risk = choice_risk if choice_risk is not None else 0.0
                    should_flag_switch = (policy_ratio >= 0.70 and heur_delta >= 1.0) or (
                        policy_ratio >= 0.60 and risk >= 20.0 and heur_delta >= -0.5
                    )
                else:
                    should_flag_switch = gap >= -5.0
                if should_flag_switch:
                    add_issue(
                        "over_switched_negative_matchup",
                        alternative=alt_choice,
                        alternative_score=round(alt_score, 3),
                        alternative_heuristic_score=None if alt_heur is None else round(alt_heur, 3),
                        chosen_heuristic_score=None if choice_heur is None else round(choice_heur, 3),
                        chosen_risk_penalty=None if choice_risk is None else round(choice_risk, 3),
                        policy_ratio=round(policy_ratio, 3),
                        best_choice=alt_choice,
                        best_score=round(alt_score, 3),
                        score_gap=round(max(0.0, gap), 3),
                    )

        if chosen_damaging and _has_status_option(row, moves_data, opp_types, opp_status, fp_oracle_battle) and opp_hp > 0.45 and reply_score <= safe_reply_threshold:
            status_alt = _best_status_alternative(row, moves_data, opp_types, opp_status, exclude_choice=choice)
            if status_alt is not None:
                alt_choice = str(status_alt.get("choice", "") or "")
                alt_score = _safe_float(status_alt.get("score", status_alt.get("weight")), 0.0)
                alt_heur = _heuristic_for_choice(row, alt_choice)
                choice_heur = _heuristic_for_choice(row, choice)
                if chosen_score is not None and chosen_score > 0.0:
                    policy_ratio = alt_score / max(chosen_score, 1e-6)
                else:
                    policy_ratio = 1.0 if alt_score > 0.0 else 0.0
                if alt_heur is not None and choice_heur is not None:
                    heur_delta = alt_heur - choice_heur
                    should_flag_status = (policy_ratio >= 0.65 and heur_delta >= 1.0) or (
                        policy_ratio >= 0.30 and heur_delta >= 10.0
                    )
                else:
                    should_flag_status = True
                if should_flag_status:
                    add_issue(
                        "underused_status_window",
                        alternative=alt_choice,
                        alternative_score=round(alt_score, 3),
                        alternative_heuristic_score=None if alt_heur is None else round(alt_heur, 3),
                        chosen_heuristic_score=None if choice_heur is None else round(choice_heur, 3),
                        policy_ratio=round(policy_ratio, 3),
                        best_choice=alt_choice,
                        best_score=round(alt_score, 3),
                        score_gap=round(max(0.0, alt_score - (chosen_score if chosen_score is not None else alt_score)), 3),
                    )

        if chosen_damaging and _has_setup_option(row, moves_data) and active_hp >= 0.65 and opp_hp >= 0.55 and reply_score <= safe_reply_threshold and active_boost_max < 2:
            setup_alt = _best_setup_alternative(row, moves_data, exclude_choice=choice)
            if setup_alt is not None:
                alt_choice = str(setup_alt.get("choice", "") or "")
                alt_score = _safe_float(setup_alt.get("score", setup_alt.get("weight")), 0.0)
                alt_heur = _heuristic_for_choice(row, alt_choice)
                choice_heur = _heuristic_for_choice(row, choice)
                alt_move_id = _move_id_from_choice(alt_choice)
                if chosen_score is not None and chosen_score > 0.0:
                    policy_ratio = alt_score / max(chosen_score, 1e-6)
                else:
                    policy_ratio = 1.0 if alt_score > 0.0 else 0.0
                if alt_move_id and alt_move_id != chosen_move_id and alt_heur is not None and choice_heur is not None:
                    heur_delta = alt_heur - choice_heur
                    should_flag_setup = (policy_ratio >= 0.65 and heur_delta >= 1.0) or (
                        policy_ratio >= 0.20 and heur_delta >= 15.0
                    )
                else:
                    should_flag_setup = bool(alt_move_id and alt_move_id != chosen_move_id)
                if should_flag_setup:
                    add_issue(
                        "underused_setup_window",
                        alternative=alt_choice,
                        alternative_score=round(alt_score, 3),
                        alternative_heuristic_score=None if alt_heur is None else round(alt_heur, 3),
                        chosen_heuristic_score=None if choice_heur is None else round(choice_heur, 3),
                        policy_ratio=round(policy_ratio, 3),
                        best_choice=alt_choice,
                        best_score=round(alt_score, 3),
                        score_gap=round(max(0.0, alt_score - (chosen_score if chosen_score is not None else alt_score)), 3),
                    )

        if chosen_passive != "recovery" and _has_recovery_option(row, moves_data) and active_hp <= low_hp_recovery and reply_score <= safe_reply_threshold and opp_hp > 0.25:
            recovery_alt = _best_recovery_alternative(row, moves_data, exclude_choice=choice)
            if recovery_alt is not None:
                alt_choice = str(recovery_alt.get("choice", "") or "")
                alt_score = _safe_float(recovery_alt.get("score", recovery_alt.get("weight")), 0.0)
                alt_heur = _heuristic_for_choice(row, alt_choice)
                choice_heur = _heuristic_for_choice(row, choice)
                if chosen_score is not None and chosen_score > 0.0:
                    policy_ratio = alt_score / max(chosen_score, 1e-6)
                else:
                    policy_ratio = 1.0 if alt_score > 0.0 else 0.0
                if alt_heur is not None and choice_heur is not None:
                    heur_delta = alt_heur - choice_heur
                    should_flag_recovery = (policy_ratio >= 0.65 and heur_delta >= 1.0) or (
                        policy_ratio >= 0.30 and heur_delta >= 10.0
                    )
                else:
                    should_flag_recovery = True
                if should_flag_recovery:
                    add_issue(
                        "ignored_safe_recovery",
                        regret=low_hp_recovery - active_hp,
                        alternative=alt_choice,
                        alternative_score=round(alt_score, 3),
                        alternative_heuristic_score=None if alt_heur is None else round(alt_heur, 3),
                        chosen_heuristic_score=None if choice_heur is None else round(choice_heur, 3),
                        policy_ratio=round(policy_ratio, 3),
                        best_choice=alt_choice,
                        best_score=round(alt_score, 3),
                        score_gap=round(max(0.0, alt_score - (chosen_score if chosen_score is not None else alt_score)), 3),
                    )

        if chosen_damaging and active_hp <= low_hp_recovery and reply_score >= punish_reply_threshold and (_has_recovery_option(row, moves_data) or chosen_kind == "switch"):
            add_issue("over_attacked_into_bad_trade")

        hazard_available = any(move_id in RuleBotPlayer.ENTRY_HAZARDS for move_id in _available_move_ids(row))
        opp_hazard_free = not bool(opp_hazards)
        if behind and chosen_damaging and opp_hp > 0.55 and reply_score <= safe_reply_threshold and (
            _has_status_option(row, moves_data, opp_types, opp_status, fp_oracle_battle)
            or _has_setup_option(row, moves_data)
            or (hazard_available and opp_hazard_free)
        ):
            progress_alt = _best_progress_alternative(row, moves_data, opp_types, opp_status, opp_hazard_free, exclude_choice=choice)
            if progress_alt is not None:
                alt_choice = str(progress_alt.get("choice", "") or "")
                alt_score = _safe_float(progress_alt.get("score", progress_alt.get("weight")), 0.0)
                alt_heur = _heuristic_for_choice(row, alt_choice)
                choice_heur = _heuristic_for_choice(row, choice)
                if chosen_score is not None and chosen_score > 0.0:
                    policy_ratio = alt_score / max(chosen_score, 1e-6)
                else:
                    policy_ratio = 1.0 if alt_score > 0.0 else 0.0
                if alt_heur is not None and choice_heur is not None:
                    heur_delta = alt_heur - choice_heur
                    should_flag_progress = (policy_ratio >= 0.65 and heur_delta >= 1.0) or (
                        policy_ratio >= 0.30 and heur_delta >= 10.0
                    )
                else:
                    should_flag_progress = True
                if should_flag_progress:
                    add_issue(
                        "failed_to_progress_when_behind",
                        alternative=alt_choice,
                        alternative_score=round(alt_score, 3),
                        alternative_heuristic_score=None if alt_heur is None else round(alt_heur, 3),
                        chosen_heuristic_score=None if choice_heur is None else round(choice_heur, 3),
                        policy_ratio=round(policy_ratio, 3),
                        progress_kind=str(progress_alt.get("progress_kind", "") or ""),
                        best_choice=alt_choice,
                        best_score=round(alt_score, 3),
                        score_gap=round(max(0.0, alt_score - (chosen_score if chosen_score is not None else alt_score)), 3),
                    )

    for items in samples_by_issue.values():
        items.sort(key=_issue_priority, reverse=True)

    summary = {
        "rows_seen": len(rows),
        "battles_seen": len(battles_seen),
        "issue_counts": dict(issue_counts),
        "issue_top_choices": {category: counts.most_common(15) for category, counts in issue_choice_counts.items()},
        "finish_blow_reasons": dict(finish_blow_reasons),
        "finish_blow_rows": finish_blow_rows,
        "finish_blow_stale_reasons": dict(finish_blow_stale_reasons),
        "missed_ko_finish_reasons": dict(missed_ko_finish_reasons),
        "missed_ko_finish_stale_reasons": dict(missed_ko_finish_stale_reasons),
        "missed_ko_finish_ratio_stats": _ratio_stats(missed_ko_finish_ratios),
        "recovery_window_reasons": dict(recovery_window_reasons),
        "recovery_window_rows": recovery_window_rows,
        "switch_guard_reasons": dict(switch_guard_reasons),
        "switch_guard_rows": switch_guard_rows,
        "progress_window_reasons": dict(progress_window_reasons),
        "progress_window_rows": progress_window_rows,
        "setup_window_reasons": dict(setup_window_reasons),
        "setup_window_rows": setup_window_rows,
        "issue_window_reasons": {category: dict(counts) for category, counts in issue_window_reasons.items()},
        "samples": dict(samples_by_issue),
        "config": {
            "ko_hp_threshold": ko_hp_threshold,
            "missed_ko_min_finish_ratio": missed_ko_min_finish_ratio,
            "safe_reply_threshold": safe_reply_threshold,
            "punish_reply_threshold": punish_reply_threshold,
            "low_hp_recovery": low_hp_recovery,
            "min_score_gap": min_score_gap,
        },
    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Mine decision traces for strategic regret buckets.")
    parser.add_argument("--input", action="append", required=True, help="JSONL/PKL path or glob. Repeatable.")
    parser.add_argument("--summary-out", default="")
    parser.add_argument("--ko-hp-threshold", type=float, default=0.35)
    parser.add_argument("--missed-ko-min-finish-ratio", type=float, default=0.75)
    parser.add_argument("--safe-reply-threshold", type=float, default=110.0)
    parser.add_argument("--punish-reply-threshold", type=float, default=180.0)
    parser.add_argument("--low-hp-recovery", type=float, default=0.4)
    parser.add_argument("--sample-limit", type=int, default=30)
    args = parser.parse_args()

    paths = _resolve_inputs(args.input)
    if not paths:
        raise SystemExit("No input files matched.")
    rows: list[dict] = []
    for path in paths:
        rows.extend(_iter_examples(path))
    summary = mine_examples(
        rows,
        moves_data=load_moves(),
        ko_hp_threshold=max(0.05, min(1.0, args.ko_hp_threshold)),
        missed_ko_min_finish_ratio=max(0.0, args.missed_ko_min_finish_ratio),
        safe_reply_threshold=max(0.0, args.safe_reply_threshold),
        punish_reply_threshold=max(0.0, args.punish_reply_threshold),
        low_hp_recovery=max(0.05, min(1.0, args.low_hp_recovery)),
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
    if summary.get("issue_window_reasons"):
        print("Issue window reasons:")
        for category, counts in sorted(summary["issue_window_reasons"].items()):
            if not counts:
                continue
            head = ", ".join(
                f"{reason}:{count}"
                for reason, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:8]
            )
            print(f"  {category}: {head}")
    if summary.get("setup_window_reasons"):
        head = ", ".join(
            f"{reason}:{count}"
            for reason, count in sorted(
                summary["setup_window_reasons"].items(),
                key=lambda kv: (-kv[1], kv[0]),
            )[:12]
        )
        print(f"Setup window reasons: {head}")
    else:
        print(f"Setup window reasons: none ({int(summary.get('setup_window_rows', 0) or 0)} diagnostic rows)")
    if summary.get("recovery_window_reasons"):
        head = ", ".join(
            f"{reason}:{count}"
            for reason, count in sorted(
                summary["recovery_window_reasons"].items(),
                key=lambda kv: (-kv[1], kv[0]),
            )[:12]
        )
        print(f"Recovery window reasons: {head}")
    else:
        print(f"Recovery window reasons: none ({int(summary.get('recovery_window_rows', 0) or 0)} diagnostic rows)")
    if summary.get("switch_guard_reasons"):
        head = ", ".join(
            f"{reason}:{count}"
            for reason, count in sorted(
                summary["switch_guard_reasons"].items(),
                key=lambda kv: (-kv[1], kv[0]),
            )[:12]
        )
        print(f"Switch guard reasons: {head}")
    else:
        print(f"Switch guard reasons: none ({int(summary.get('switch_guard_rows', 0) or 0)} diagnostic rows)")
    if summary.get("progress_window_reasons"):
        head = ", ".join(
            f"{reason}:{count}"
            for reason, count in sorted(
                summary["progress_window_reasons"].items(),
                key=lambda kv: (-kv[1], kv[0]),
            )[:12]
        )
        print(f"Progress window reasons: {head}")
    else:
        print(f"Progress window reasons: none ({int(summary.get('progress_window_rows', 0) or 0)} diagnostic rows)")
    if summary.get("finish_blow_reasons"):
        head = ", ".join(
            f"{reason}:{count}"
            for reason, count in sorted(
                summary["finish_blow_reasons"].items(),
                key=lambda kv: (-kv[1], kv[0]),
            )[:12]
        )
        print(f"Finish blow reasons: {head}")
    else:
        print(f"Finish blow reasons: none ({int(summary.get('finish_blow_rows', 0) or 0)} diagnostic rows)")
    if summary.get("missed_ko_finish_reasons"):
        head = ", ".join(
            f"{reason}:{count}"
            for reason, count in sorted(
                summary["missed_ko_finish_reasons"].items(),
                key=lambda kv: (-kv[1], kv[0]),
            )[:12]
        )
        print(f"Missed KO finish reasons: {head}")
    else:
        print("Missed KO finish reasons: none")
    if summary.get("finish_blow_stale_reasons"):
        head = ", ".join(
            f"{reason}:{count}"
            for reason, count in sorted(
                summary["finish_blow_stale_reasons"].items(),
                key=lambda kv: (-kv[1], kv[0]),
            )[:12]
        )
        print(f"Stale finish reasons: {head}")
    else:
        print("Stale finish reasons: none")
    if summary.get("missed_ko_finish_stale_reasons"):
        head = ", ".join(
            f"{reason}:{count}"
            for reason, count in sorted(
                summary["missed_ko_finish_stale_reasons"].items(),
                key=lambda kv: (-kv[1], kv[0]),
            )[:12]
        )
        print(f"Missed KO stale finish reasons: {head}")
    else:
        print("Missed KO stale finish reasons: none")
    if summary.get("missed_ko_finish_ratio_stats"):
        stats = summary["missed_ko_finish_ratio_stats"]
        print(
            "Missed KO finish ratio best/threshold: "
            f"n={stats.get('n', 0)} "
            f"p50={stats.get('p50', 0.0)} "
            f"p75={stats.get('p75', 0.0)} "
            f"p90={stats.get('p90', 0.0)} "
            f"max={stats.get('max', 0.0)} "
            f">=0.75={stats.get('near_075', 0)} "
            f">=0.90={stats.get('near_090', 0)}"
        )
    else:
        print("Missed KO finish ratio best/threshold: none")
    if args.summary_out:
        out_path = Path(args.summary_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        print(f"Summary -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
