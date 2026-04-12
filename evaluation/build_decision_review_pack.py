#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.mine_bad_decisions import _issue_priority, mine_examples
from evaluation.audit_engine_behavior import _iter_examples, _resolve_inputs
from src.utils.features import load_moves


def build_review_pack(summary: dict, *, limit: int = 50, losses_only: bool = False) -> list[dict]:
    rows: list[dict] = []
    for category, samples in (summary.get("samples") or {}).items():
        if not isinstance(samples, list):
            continue
        for sample in samples:
            if not isinstance(sample, dict):
                continue
            if losses_only and not sample.get("lost_battle"):
                continue
            row = dict(sample)
            row["category"] = category
            row["priority"] = _issue_priority(row)
            row["review_blurb"] = _review_blurb(row)
            rows.append(row)
    rows.sort(key=lambda row: (-float(row.get("priority", 0.0) or 0.0), str(row.get("battle_id", "")), int(row.get("turn", 0) or 0)))
    return rows[: max(1, limit)]


def _review_blurb(row: dict) -> str:
    category = str(row.get("category", "") or "")
    choice = str(row.get("choice", "") or "")
    best = str(row.get("best_choice", "") or "")
    active = str(row.get("active_species", "") or "")
    opp = str(row.get("opponent_species", "") or "")
    gap = float(row.get("score_gap", 0.0) or 0.0)
    if category == "missed_ko":
        alternative = str(row.get("alternative", "") or best)
        detail = _detail_suffix(row)
        return f"{active} into {opp}: chose {choice} with opponent low; review KO line{_alt(alternative)}{detail}."
    if category == "ignored_safe_recovery":
        alternative = str(row.get("alternative", "") or best or "recovery")
        detail = _detail_suffix(row)
        return f"{active} into {opp}: skipped recovery at low HP; review recovery line like {alternative}{detail}."
    if category == "underused_setup_window":
        alternative = str(row.get("alternative", "") or best or "setup")
        detail = _detail_suffix(row)
        return f"{active} into {opp}: attacked instead of converting a safe setup window; review setup line like {alternative}{detail}."
    if category == "underused_status_window":
        alternative = str(row.get("alternative", "") or best or "status")
        detail = _detail_suffix(row)
        return f"{active} into {opp}: used {choice} despite an open status window; review status line like {alternative}{detail}."
    if category == "over_switched_negative_matchup":
        alternative = str(row.get("alternative", "") or best or "best attack")
        heur = _heuristic_suffix(row)
        policy = _policy_suffix(row)
        return f"{active} into {opp}: switched out with live board presence; compare against {alternative} (gap {gap:.1f}{policy}{heur})."
    if category == "over_attacked_into_bad_trade":
        return f"{active} into {opp}: attacked from a fragile position under strong reply pressure; review safer line."
    if category == "failed_to_progress_when_behind":
        return f"{active} into {opp}: behind on board but chose {choice}; review if status/setup/hazards created better comeback equity."
    return f"{active} into {opp}: review {choice}."


def _alt(best: str) -> str:
    return f" like {best}" if best else ""


def _detail_suffix(row: dict) -> str:
    parts = []
    policy = _policy_suffix(row).lstrip(", ")
    heur = _heuristic_suffix(row).lstrip(", ")
    if policy:
        parts.append(policy)
    if heur:
        parts.append(heur)
    return f" ({', '.join(parts)})" if parts else ""


def _heuristic_suffix(row: dict) -> str:
    chosen = row.get("chosen_heuristic_score")
    alt = row.get("alternative_heuristic_score")
    if chosen is None or alt is None:
        return ""
    try:
        return f", heur {float(chosen):.1f}->{float(alt):.1f}"
    except Exception:
        return ""


def _policy_suffix(row: dict) -> str:
    chosen = row.get("chosen_score")
    alt = row.get("alternative_score")
    if chosen is None or alt is None:
        return ""
    try:
        return f", policy {float(chosen):.3f}->{float(alt):.3f}"
    except Exception:
        return ""


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a ranked review pack from mined decision issues.")
    parser.add_argument("--issues-json", default="", help="Existing JSON summary from mine_bad_decisions.py")
    parser.add_argument("--input", action="append", default=[], help="Trace JSONL/PKL path or glob. Used when --issues-json is omitted.")
    parser.add_argument("--output", required=True, help="Output JSON path for review pack")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--losses-only", action="store_true")
    args = parser.parse_args()

    if args.issues_json:
        with Path(args.issues_json).open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
    else:
        paths = _resolve_inputs(args.input)
        if not paths:
            raise SystemExit("No input files matched.")
        rows = []
        for path in paths:
            rows.extend(_iter_examples(path))
        summary = mine_examples(rows, moves_data=load_moves())

    review_rows = build_review_pack(summary, limit=max(1, args.limit), losses_only=bool(args.losses_only))
    output = {
        "rows": review_rows,
        "count": len(review_rows),
        "losses_only": bool(args.losses_only),
        "source_issue_counts": summary.get("issue_counts") or {},
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)

    print(f"Review rows: {len(review_rows)}")
    for row in review_rows[:10]:
        print(f"  {row.get('category')}: {row.get('battle_id')} t{row.get('turn')} {row.get('choice')} | {row.get('review_blurb')}")
    print(f"Review pack -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
