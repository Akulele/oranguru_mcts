#!/usr/bin/env python3
"""Summarize search-trace decision behavior by path, action kind, and outcome.

This complements the tactical bad-decision miner.  The miner is good at finding
specific hand-authored failure classes; this script is for detecting broader
engine-path problems such as rerank decisions being loss-skewed or non-top1
choices consistently dropping policy mass.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import glob
import json
from pathlib import Path
from statistics import mean
from typing import Iterable


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _choice(row: dict) -> str:
    choice = str(row.get("chosen_choice", "") or "")
    if choice:
        return choice
    labels = row.get("action_labels") or []
    idx = row.get("chosen_action")
    if isinstance(labels, list) and isinstance(idx, int) and 0 <= idx < len(labels):
        return str(labels[idx] or "")
    return ""


def _kind(choice: str, top_action: dict | None = None) -> str:
    if top_action and top_action.get("kind"):
        return str(top_action.get("kind") or "")
    if choice.startswith("switch "):
        return "switch"
    if choice.endswith("-tera"):
        return "tera"
    if choice:
        return "move"
    return "unknown"


def _top_actions(row: dict) -> list[dict]:
    actions = row.get("top_actions") or []
    return [a for a in actions if isinstance(a, dict)]


def _choice_action(row: dict, choice: str) -> dict | None:
    for action in _top_actions(row):
        if str(action.get("choice", "") or "") == choice:
            return action
    return None


def _score(action: dict | None) -> float:
    if not action:
        return 0.0
    return _safe_float(action.get("score", action.get("weight", 0.0)), 0.0)


def _heuristic(action: dict | None) -> float:
    if not action:
        return 0.0
    return _safe_float(action.get("heuristic_score", 0.0), 0.0)


def _loss_bucket(value_target: float) -> str:
    if value_target < 0.0:
        return "loss"
    if value_target > 0.0:
        return "win"
    return "unknown"


def _iter_rows(paths: Iterable[str]) -> Iterable[dict]:
    for path in paths:
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


def _resolve_inputs(patterns: list[str]) -> list[str]:
    paths: list[str] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            paths.extend(matches)
        elif Path(pattern).exists():
            paths.append(pattern)
    return paths


class Bucket:
    def __init__(self) -> None:
        self.rows = 0
        self.wins = 0
        self.losses = 0
        self.non_top1 = 0
        self.confidences: list[float] = []
        self.margins: list[float] = []
        self.score_deltas: list[float] = []
        self.heuristic_deltas: list[float] = []

    def add(self, row: dict, *, non_top1: bool, score_delta: float, heuristic_delta: float) -> None:
        self.rows += 1
        value_target = _safe_float(row.get("value_target", 0.0), 0.0)
        if value_target > 0.0:
            self.wins += 1
        elif value_target < 0.0:
            self.losses += 1
        if non_top1:
            self.non_top1 += 1
        self.confidences.append(_safe_float(row.get("policy_confidence", 0.0), 0.0))
        actions = _top_actions(row)
        if len(actions) >= 2:
            self.margins.append(_score(actions[0]) - _score(actions[1]))
        elif actions:
            self.margins.append(_score(actions[0]))
        self.score_deltas.append(score_delta)
        self.heuristic_deltas.append(heuristic_delta)

    def payload(self) -> dict:
        return {
            "rows": self.rows,
            "wins": self.wins,
            "losses": self.losses,
            "loss_row_rate": self.losses / max(1, self.wins + self.losses),
            "non_top1_rate": self.non_top1 / max(1, self.rows),
            "avg_confidence": mean(self.confidences) if self.confidences else 0.0,
            "avg_margin": mean(self.margins) if self.margins else 0.0,
            "avg_score_delta_top1_minus_chosen": mean(self.score_deltas) if self.score_deltas else 0.0,
            "avg_heur_delta_chosen_minus_top1": mean(self.heuristic_deltas) if self.heuristic_deltas else 0.0,
        }


def summarize(paths: list[str], *, sample_limit: int = 20) -> dict:
    total = Bucket()
    by_path: dict[str, Bucket] = defaultdict(Bucket)
    by_choice_kind: dict[str, Bucket] = defaultdict(Bucket)
    by_path_kind: dict[str, Bucket] = defaultdict(Bucket)
    by_window_reason: dict[str, Counter] = defaultdict(Counter)
    outcome_counts = Counter()
    battle_ids: set[str] = set()
    path_counts = Counter()
    chosen_kind_counts = Counter()
    top1_kind_counts = Counter()
    non_top1_loss_samples: list[dict] = []
    rerank_loss_samples: list[dict] = []

    for row in _iter_rows(paths):
        battle_id = str(row.get("battle_id", "") or "")
        if battle_id:
            battle_ids.add(battle_id)
        choice = _choice(row)
        actions = _top_actions(row)
        top1 = actions[0] if actions else None
        top1_choice = str(top1.get("choice", "") or "") if top1 else ""
        chosen_action = _choice_action(row, choice)
        path = str(row.get("selection_path", "") or "unknown")
        chosen_kind = _kind(choice, chosen_action)
        top1_kind = _kind(top1_choice, top1)
        non_top1 = bool(top1_choice and choice and choice != top1_choice)
        score_delta = max(0.0, _score(top1) - _score(chosen_action))
        heuristic_delta = _heuristic(chosen_action) - _heuristic(top1)
        value_target = _safe_float(row.get("value_target", 0.0), 0.0)
        outcome = _loss_bucket(value_target)

        path_counts[path] += 1
        chosen_kind_counts[chosen_kind] += 1
        top1_kind_counts[top1_kind] += 1
        outcome_counts[outcome] += 1

        total.add(row, non_top1=non_top1, score_delta=score_delta, heuristic_delta=heuristic_delta)
        by_path[path].add(row, non_top1=non_top1, score_delta=score_delta, heuristic_delta=heuristic_delta)
        by_choice_kind[chosen_kind].add(row, non_top1=non_top1, score_delta=score_delta, heuristic_delta=heuristic_delta)
        by_path_kind[f"{path}/{chosen_kind}"].add(
            row,
            non_top1=non_top1,
            score_delta=score_delta,
            heuristic_delta=heuristic_delta,
        )

        for window in ("finish_blow", "setup_window", "recovery_window", "switch_guard", "progress_window"):
            payload = row.get(window)
            if isinstance(payload, dict):
                reason = str(payload.get("reason", "") or "")
                if reason:
                    by_window_reason[window][reason] += 1

        if value_target < 0.0 and non_top1 and len(non_top1_loss_samples) < sample_limit:
            non_top1_loss_samples.append(
                {
                    "battle_id": battle_id,
                    "turn": row.get("turn"),
                    "path": path,
                    "chosen": choice,
                    "top1": top1_choice,
                    "chosen_kind": chosen_kind,
                    "top1_kind": top1_kind,
                    "score_delta": round(score_delta, 4),
                    "heur_delta_chosen_minus_top1": round(heuristic_delta, 3),
                    "confidence": round(_safe_float(row.get("policy_confidence", 0.0), 0.0), 4),
                }
            )
        if value_target < 0.0 and path == "rerank" and len(rerank_loss_samples) < sample_limit:
            rerank_loss_samples.append(
                {
                    "battle_id": battle_id,
                    "turn": row.get("turn"),
                    "chosen": choice,
                    "top1": top1_choice,
                    "chosen_kind": chosen_kind,
                    "top1_kind": top1_kind,
                    "score_delta": round(score_delta, 4),
                    "heur_delta_chosen_minus_top1": round(heuristic_delta, 3),
                    "confidence": round(_safe_float(row.get("policy_confidence", 0.0), 0.0), 4),
                    "top_actions": [
                        {
                            "choice": str(action.get("choice", "") or ""),
                            "kind": str(action.get("kind", "") or ""),
                            "score": round(_score(action), 4),
                            "heuristic": round(_heuristic(action), 3),
                            "risk": round(_safe_float(action.get("risk_penalty", 0.0), 0.0), 3),
                        }
                        for action in actions[:5]
                    ],
                }
            )

    return {
        "inputs": paths,
        "rows": total.rows,
        "battles": len(battle_ids),
        "outcomes": dict(outcome_counts),
        "overall": total.payload(),
        "path_counts": dict(path_counts),
        "chosen_kind_counts": dict(chosen_kind_counts),
        "top1_kind_counts": dict(top1_kind_counts),
        "by_path": {key: bucket.payload() for key, bucket in sorted(by_path.items())},
        "by_choice_kind": {key: bucket.payload() for key, bucket in sorted(by_choice_kind.items())},
        "by_path_kind": {key: bucket.payload() for key, bucket in sorted(by_path_kind.items())},
        "window_reasons": {key: dict(counter) for key, counter in by_window_reason.items()},
        "non_top1_loss_samples": non_top1_loss_samples,
        "rerank_loss_samples": rerank_loss_samples,
    }


def _print_bucket_table(title: str, payload: dict[str, dict], *, limit: int) -> None:
    print(title)
    rows = sorted(payload.items(), key=lambda item: item[1].get("rows", 0), reverse=True)
    if not rows:
        print("  none")
        return
    print("  key                       rows  loss%  nonTop1%  conf   margin  scoreDrop  heurDelta")
    for key, data in rows[:limit]:
        print(
            f"  {key[:25]:25s} "
            f"{int(data['rows']):5d} "
            f"{data['loss_row_rate'] * 100:6.1f} "
            f"{data['non_top1_rate'] * 100:8.1f} "
            f"{data['avg_confidence']:6.3f} "
            f"{data['avg_margin']:7.3f} "
            f"{data['avg_score_delta_top1_minus_chosen']:9.3f} "
            f"{data['avg_heur_delta_chosen_minus_top1']:9.2f}"
        )


def print_summary(summary: dict, *, limit: int) -> None:
    print(f"Rows: {summary['rows']}")
    print(f"Battles: {summary['battles']}")
    print(f"Outcomes: {summary['outcomes']}")
    print(f"Path counts: {summary['path_counts']}")
    print(f"Chosen kind counts: {summary['chosen_kind_counts']}")
    print(f"Top1 kind counts: {summary['top1_kind_counts']}")
    _print_bucket_table("By selection path:", summary["by_path"], limit=limit)
    _print_bucket_table("By chosen kind:", summary["by_choice_kind"], limit=limit)
    _print_bucket_table("By path/chosen kind:", summary["by_path_kind"], limit=limit)
    print("Window reasons:")
    for window, reasons in sorted(summary["window_reasons"].items()):
        top = ", ".join(f"{k}:{v}" for k, v in Counter(reasons).most_common(8))
        print(f"  {window}: {top or 'none'}")
    print("Non-top1 loss samples:")
    for sample in summary["non_top1_loss_samples"][:limit]:
        print(
            "  {battle_id} t{turn} {path}: {chosen} over {top1} "
            "scoreDrop={score_delta} heurDelta={heur_delta_chosen_minus_top1} conf={confidence}".format(**sample)
        )
    print("Rerank loss samples:")
    for sample in summary["rerank_loss_samples"][:limit]:
        print(
            "  {battle_id} t{turn}: {chosen} over {top1} "
            "scoreDrop={score_delta} heurDelta={heur_delta_chosen_minus_top1} conf={confidence}".format(**sample)
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="Trace JSONL paths or glob patterns.")
    parser.add_argument("--limit", type=int, default=20, help="Rows to print per table/sample section.")
    parser.add_argument("--json-out", help="Optional JSON output path.")
    args = parser.parse_args()

    paths = _resolve_inputs(args.inputs)
    if not paths:
        print("No input traces matched.")
        return 1
    summary = summarize(paths, sample_limit=max(0, args.limit))
    print_summary(summary, limit=max(0, args.limit))
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        with Path(args.json_out).open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
            handle.write("\n")
        print(f"Summary -> {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
