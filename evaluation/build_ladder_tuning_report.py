#!/usr/bin/env python3
"""Build an outcome-calibrated tuning report from ladder metrics and search traces.

Unlike a review pack, this aggregates every traced decision by engine path,
action kind, and concrete failure signature.  Use it to decide whether a guard
is helping before changing ladder envs.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import glob
import json
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

PIVOT_MOVES = {"uturn", "voltswitch", "flipturn", "partingshot", "teleport", "chillyreception", "batonpass", "shedtail"}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _iter_jsonl(path: str | Path):
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
                yield row


def _resolve_inputs(patterns: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if not matches and Path(pattern).exists():
            matches = [pattern]
        for path in matches:
            key = str(Path(path).resolve())
            if key not in seen:
                seen.add(key)
                out.append(path)
    return out


def _choice(row: dict[str, Any]) -> str:
    choice = str(row.get("chosen_choice", "") or "")
    if choice:
        return choice
    labels = row.get("action_labels") or []
    idx = row.get("chosen_action")
    if isinstance(labels, list) and isinstance(idx, int) and 0 <= idx < len(labels):
        return str(labels[idx] or "")
    return ""


def _base_choice(choice: str) -> str:
    return choice.replace("-tera", "").replace("move ", "").strip()


def _kind(choice: str, action: dict[str, Any] | None = None) -> str:
    if action and action.get("kind"):
        return str(action.get("kind") or "")
    if choice.startswith("switch "):
        return "switch"
    if choice.endswith("-tera"):
        return "tera_attack"
    return "move" if choice else "unknown"


def _top_actions(row: dict[str, Any]) -> list[dict[str, Any]]:
    return [a for a in (row.get("top_actions") or []) if isinstance(a, dict)]


def _action_for(row: dict[str, Any], choice: str) -> dict[str, Any] | None:
    for action in _top_actions(row):
        if str(action.get("choice", "") or "") == choice:
            return action
    return None


def _score(action: dict[str, Any] | None) -> float:
    if not action:
        return 0.0
    return _safe_float(action.get("score", action.get("weight", 0.0)), 0.0)


def _heur(action: dict[str, Any] | None) -> float:
    if not action:
        return 0.0
    return _safe_float(action.get("heuristic_score", 0.0), 0.0)


class Bucket:
    def __init__(self) -> None:
        self.rows = 0
        self.wins = 0
        self.losses = 0
        self.rating_residuals: list[float] = []
        self.score_drops: list[float] = []
        self.heur_deltas: list[float] = []

    def add(self, result: str, residual: float, score_drop: float, heur_delta: float) -> None:
        self.rows += 1
        if result == "win":
            self.wins += 1
        elif result == "loss":
            self.losses += 1
        self.rating_residuals.append(residual)
        self.score_drops.append(score_drop)
        self.heur_deltas.append(heur_delta)

    def payload(self) -> dict[str, Any]:
        decided = self.wins + self.losses
        return {
            "rows": self.rows,
            "wins": self.wins,
            "losses": self.losses,
            "loss_rate": self.losses / max(1, decided),
            "avg_rating_residual": mean(self.rating_residuals) if self.rating_residuals else 0.0,
            "avg_score_drop_top1_minus_chosen": mean(self.score_drops) if self.score_drops else 0.0,
            "avg_heur_delta_chosen_minus_top1": mean(self.heur_deltas) if self.heur_deltas else 0.0,
        }


def _load_ladder(path: str) -> dict[str, dict[str, Any]]:
    rows = {}
    for row in _iter_jsonl(path):
        tag = str(row.get("battle_tag") or row.get("battle_id") or "")
        result = str(row.get("result", "") or "").lower()
        if tag and result in {"win", "loss", "tie"}:
            rows[tag] = row
    return rows


def _issue_tags(row: dict[str, Any], choice: str, chosen_action: dict[str, Any] | None, top1: dict[str, Any] | None) -> list[str]:
    tags: list[str] = []
    base = _base_choice(choice)
    top_choice = str((top1 or {}).get("choice", "") or "")
    chosen_kind = _kind(choice, chosen_action)
    top_kind = _kind(top_choice, top1)
    if choice != top_choice and _score(top1) - _score(chosen_action) >= 0.20:
        tags.append("large_policy_drop")
    if choice.endswith("-tera") and top_choice == choice.replace("-tera", ""):
        tags.append("tera_over_base")
    if choice.endswith("-tera") and chosen_kind in {"tera_attack", "attack"}:
        tags.append("attack_tera")
    if base in PIVOT_MOVES:
        tags.append("pivot_move")
    if base == "rapidspin":
        hazard_load = _safe_float(row.get("hazard_load", 0.0), 0.0)
        tags.append("rapid_spin")
        if hazard_load <= 0.01:
            tags.append("rapid_spin_no_hazards")
    if top_kind == "switch" and chosen_kind not in {"switch", "protect", "recovery"}:
        tags.append("ignored_switch_top")
    if top_kind == "recovery" and chosen_kind not in {"recovery", "protect"}:
        tags.append("ignored_recovery_top")
    if chosen_kind in {"recovery", "protect", "status"} and top_kind in {"attack", "tera_attack"}:
        tags.append("passive_over_attack_top")
    for key in (
        "finish_blow",
        "critical_recovery",
        "recovery_window",
        "setup_window",
        "progress_window",
        "switch_guard",
        "low_hp_defensive_top",
        "tera_sanity",
        "rapid_spin_guard",
        "pivot_churn_guard",
    ):
        payload = row.get(key)
        if isinstance(payload, dict):
            reason = str(payload.get("reason", "") or "")
            if reason.startswith("take_"):
                tags.append(f"{key}:{reason}")
    return tags or ["uncategorized"]


def build_report(ladder_log: str, traces: list[str], *, min_rows: int) -> dict[str, Any]:
    ladder = _load_ladder(ladder_log)
    total = Bucket()
    by_path: dict[str, Bucket] = defaultdict(Bucket)
    by_kind: dict[str, Bucket] = defaultdict(Bucket)
    by_top_chosen: dict[str, Bucket] = defaultdict(Bucket)
    by_issue: dict[str, Bucket] = defaultdict(Bucket)
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    counters = Counter()

    for path in traces:
        for row in _iter_jsonl(path):
            battle_id = str(row.get("battle_id", "") or "")
            meta = ladder.get(battle_id)
            if not meta:
                counters["drop_no_ladder"] += 1
                continue
            result = str(meta.get("result", "") or "").lower()
            if result not in {"win", "loss"}:
                counters["drop_no_result"] += 1
                continue
            choice = _choice(row)
            actions = _top_actions(row)
            top1 = actions[0] if actions else None
            top_choice = str((top1 or {}).get("choice", "") or "")
            chosen_action = _action_for(row, choice)
            chosen_kind = _kind(choice, chosen_action)
            top_kind = _kind(top_choice, top1)
            residual = _safe_float(meta.get("rating_residual", 0.0), 0.0)
            score_drop = max(0.0, _score(top1) - _score(chosen_action))
            heur_delta = _heur(chosen_action) - _heur(top1)

            total.add(result, residual, score_drop, heur_delta)
            by_path[str(row.get("selection_path", "unknown") or "unknown")].add(result, residual, score_drop, heur_delta)
            by_kind[chosen_kind].add(result, residual, score_drop, heur_delta)
            by_top_chosen[f"{top_kind}->{chosen_kind}"].add(result, residual, score_drop, heur_delta)
            for tag in _issue_tags(row, choice, chosen_action, top1):
                by_issue[tag].add(result, residual, score_drop, heur_delta)
                if result == "loss" and len(examples[tag]) < 5:
                    examples[tag].append({
                        "battle_id": battle_id,
                        "turn": int(row.get("turn", 0) or 0),
                        "choice": choice,
                        "top_choice": top_choice,
                        "selection_path": str(row.get("selection_path", "") or ""),
                        "score_drop": round(score_drop, 4),
                        "rating_residual": round(residual, 4),
                    })
            counters["rows"] += 1

    def emit(mapping: dict[str, Bucket]) -> dict[str, Any]:
        rows = {key: bucket.payload() for key, bucket in mapping.items() if bucket.rows >= min_rows}
        return dict(sorted(rows.items(), key=lambda item: (-item[1]["loss_rate"], -item[1]["rows"], item[0])))

    return {
        "ladder_log": ladder_log,
        "trace_inputs": traces,
        "counters": dict(counters),
        "overall": total.payload(),
        "by_selection_path": emit(by_path),
        "by_chosen_kind": emit(by_kind),
        "by_top_to_chosen_kind": emit(by_top_chosen),
        "by_issue": emit(by_issue),
        "loss_examples_by_issue": {k: v for k, v in examples.items() if k in by_issue and by_issue[k].rows >= min_rows},
    }


def _write_md(report: dict[str, Any], path: str, *, limit: int) -> None:
    lines = ["# Ladder Tuning Report", ""]
    overall = report["overall"]
    lines.append(
        f"Rows: {overall['rows']} | W/L {overall['wins']}/{overall['losses']} | "
        f"loss_rate={overall['loss_rate']:.1%} | avg_residual={overall['avg_rating_residual']:+.3f}"
    )
    for title, key in (
        ("Selection Path", "by_selection_path"),
        ("Chosen Kind", "by_chosen_kind"),
        ("Top Kind -> Chosen Kind", "by_top_to_chosen_kind"),
        ("Issue Tags", "by_issue"),
    ):
        lines.extend(["", f"## {title}", "", "| bucket | rows | W/L | loss_rate | residual | score_drop | heur_delta |", "|---|---:|---:|---:|---:|---:|---:|"])
        for bucket, payload in list(report.get(key, {}).items())[:limit]:
            lines.append(
                f"| {bucket} | {payload['rows']} | {payload['wins']}/{payload['losses']} | "
                f"{payload['loss_rate']:.1%} | {payload['avg_rating_residual']:+.3f} | "
                f"{payload['avg_score_drop_top1_minus_chosen']:.3f} | {payload['avg_heur_delta_chosen_minus_top1']:+.1f} |"
            )
    lines.extend(["", "## Loss Examples", ""])
    for issue, rows in list(report.get("loss_examples_by_issue", {}).items())[:limit]:
        lines.append(f"### {issue}")
        for row in rows:
            lines.append(
                f"- {row['battle_id']} t{row['turn']} {row['choice']} -> top {row['top_choice']} "
                f"path={row['selection_path']} drop={row['score_drop']} residual={row['rating_residual']}"
            )
        lines.append("")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ladder-log", required=True)
    parser.add_argument("--trace", action="append", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", default="")
    parser.add_argument("--min-rows", type=int, default=3)
    parser.add_argument("--limit", type=int, default=30)
    args = parser.parse_args()

    traces = _resolve_inputs(args.trace)
    if not traces:
        raise SystemExit("No trace inputs matched.")
    report = build_report(args.ladder_log, traces, min_rows=max(1, args.min_rows))
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.output_md:
        _write_md(report, args.output_md, limit=max(1, args.limit))
    print(
        f"Tuning report: rows={report['overall']['rows']} "
        f"W/L={report['overall']['wins']}/{report['overall']['losses']} -> {out}"
    )
    if args.output_md:
        print(f"Markdown -> {args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
