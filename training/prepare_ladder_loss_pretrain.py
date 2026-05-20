#!/usr/bin/env python3
"""Build RL pretrain examples from ladder losses plus search traces.

The live search trace already stores policy/value tensors. This script joins
those rows with ladder metrics, keeps loss battles by default, and enriches
each example with rating/result metadata so SearchPriorValueNet/LeafValueNet
can train directly on the failed trajectories.
"""

from __future__ import annotations

import argparse
import glob
import json
import pickle
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.search_assist_utils import safe_float, validate_search_assist_example


def _resolve_inputs(patterns: Iterable[str]) -> list[str]:
    paths: list[str] = []
    seen: set[str] = set()
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if not matches and Path(pattern).exists():
            matches = [pattern]
        for path in matches:
            key = str(Path(path).resolve())
            if key not in seen:
                seen.add(key)
                paths.append(path)
    return paths


def _iter_jsonl(path: str):
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                yield {"_parse_error": True, "_path": path, "_line": line_no}
                continue
            if isinstance(row, dict):
                yield row


def _load_ladder_rows(path: str) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for row in _iter_jsonl(path):
        tag = str(row.get("battle_tag") or row.get("battle_id") or "")
        if tag:
            rows[tag] = row
    return rows


def _total_visits(row: dict) -> float:
    return sum(max(0.0, safe_float(v, 0.0)) for v in (row.get("visit_counts") or []))


def _result_value(result: str) -> float:
    result = result.strip().lower()
    if result == "win":
        return 1.0
    if result == "loss":
        return -1.0
    return 0.0


def _weight_for(row: dict, ladder: dict, args: argparse.Namespace) -> float:
    weight = safe_float(row.get("weight", 1.0), 1.0) * max(0.0, args.base_weight)
    residual = safe_float(ladder.get("rating_residual", 0.0), 0.0)
    if args.residual_weight > 0.0:
        weight *= 1.0 + args.residual_weight * min(1.0, abs(residual))
    if args.endgame_weight > 0.0 and str(row.get("phase", "")).lower() == "end":
        weight *= 1.0 + args.endgame_weight
    if args.close_loss_weight > 0.0 and str(ladder.get("result", "")).lower() == "loss":
        remaining = ladder.get("remaining")
        opp_remaining = ladder.get("opp_remaining")
        if isinstance(remaining, int) and isinstance(opp_remaining, int) and opp_remaining <= 2:
            weight *= 1.0 + args.close_loss_weight
    return weight


def _enrich_trace_row(row: dict, ladder: dict, args: argparse.Namespace) -> dict:
    out = dict(row)
    result = str(ladder.get("result", "") or "").lower()
    out["source"] = "ladder_loss_pretrain"
    out["source_trace"] = str(row.get("source", "") or "")
    out["battle_result"] = result
    out["value_target"] = _result_value(result)
    out["weight"] = _weight_for(row, ladder, args)
    out["rating"] = ladder.get("player_rating_pre")
    out["ladder"] = {
        "account": ladder.get("account"),
        "bot_version": ladder.get("bot_version"),
        "opponent_username": ladder.get("opponent_username"),
        "expected_score": ladder.get("expected_score"),
        "rating_residual": ladder.get("rating_residual"),
        "player_rating_pre": ladder.get("player_rating_pre"),
        "player_rating_post": ladder.get("player_rating_post"),
        "player_rating_delta": ladder.get("player_rating_delta"),
        "opponent_rating_pre": ladder.get("opponent_rating_pre"),
        "turns": ladder.get("turns"),
        "remaining": ladder.get("remaining"),
        "opp_remaining": ladder.get("opp_remaining"),
        "forfeit": ladder.get("forfeit"),
    }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare loss-only RL pretrain rows from ladder traces.")
    parser.add_argument("--ladder-log", required=True, help="Ladder metrics JSONL")
    parser.add_argument("--trace", action="append", required=True, help="Search trace JSONL path/glob. Repeatable.")
    parser.add_argument("--output", required=True, help="Output .pkl or .jsonl")
    parser.add_argument("--jsonl-out", default="", help="Optional additional JSONL output")
    parser.add_argument("--summary-out", default="", help="Optional summary JSON output")
    parser.add_argument("--result", choices=["loss", "win", "all"], default="loss")
    parser.add_argument("--min-turn", type=int, default=1)
    parser.add_argument("--min-visits", type=float, default=1.0)
    parser.add_argument("--drop-forfeit", action="store_true", default=True)
    parser.add_argument("--keep-forfeit", action="store_true", help="Disable forfeit filtering.")
    parser.add_argument("--base-weight", type=float, default=1.0)
    parser.add_argument("--residual-weight", type=float, default=1.0)
    parser.add_argument("--endgame-weight", type=float, default=0.25)
    parser.add_argument("--close-loss-weight", type=float, default=0.25)
    args = parser.parse_args()

    if args.keep_forfeit:
        args.drop_forfeit = False

    trace_paths = _resolve_inputs(args.trace)
    if not trace_paths:
        raise SystemExit("No trace inputs matched.")

    ladder_rows = _load_ladder_rows(args.ladder_log)
    if not ladder_rows:
        raise SystemExit("No ladder metric rows found.")

    counters = Counter()
    examples: list[dict] = []
    kept_battles: set[str] = set()

    for path in trace_paths:
        for row in _iter_jsonl(path):
            if row.get("_parse_error"):
                counters["drop_trace_parse"] += 1
                continue
            counters["trace_rows_seen"] += 1
            battle_id = str(row.get("battle_id") or "")
            ladder = ladder_rows.get(battle_id)
            if ladder is None:
                counters["drop_no_ladder_match"] += 1
                continue
            result = str(ladder.get("result", "") or "").lower()
            if args.result != "all" and result != args.result:
                counters[f"drop_result_{result or 'unknown'}"] += 1
                continue
            if args.drop_forfeit and bool(ladder.get("forfeit", False)):
                counters["drop_forfeit"] += 1
                continue
            if int(row.get("turn", 0) or 0) < args.min_turn:
                counters["drop_min_turn"] += 1
                continue
            if _total_visits(row) < args.min_visits:
                counters["drop_min_visits"] += 1
                continue
            enriched = _enrich_trace_row(row, ladder, args)
            ok, reason = validate_search_assist_example(enriched)
            if not ok:
                counters[f"drop_invalid_{reason}"] += 1
                continue
            examples.append(enriched)
            kept_battles.add(battle_id)
            counters["kept"] += 1

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix == ".jsonl":
        with out_path.open("w", encoding="utf-8") as handle:
            for ex in examples:
                handle.write(json.dumps(ex, separators=(",", ":")) + "\n")
    else:
        with out_path.open("wb") as handle:
            pickle.dump(examples, handle)

    if args.jsonl_out:
        jsonl_path = Path(args.jsonl_out)
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with jsonl_path.open("w", encoding="utf-8") as handle:
            for ex in examples:
                handle.write(json.dumps(ex, separators=(",", ":")) + "\n")

    summary = {
        "ladder_log": args.ladder_log,
        "trace_inputs": trace_paths,
        "output": str(out_path),
        "jsonl_out": args.jsonl_out,
        "result": args.result,
        "examples": len(examples),
        "battles": len(kept_battles),
        "counters": dict(counters),
        "min_turn": args.min_turn,
        "min_visits": args.min_visits,
        "drop_forfeit": args.drop_forfeit,
        "base_weight": args.base_weight,
        "residual_weight": args.residual_weight,
        "endgame_weight": args.endgame_weight,
        "close_loss_weight": args.close_loss_weight,
    }
    print(f"Wrote {len(examples)} examples from {len(kept_battles)} battles -> {out_path}")
    print(f"Stats: {dict(counters)}")

    if args.summary_out:
        summary_path = Path(args.summary_out)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        print(f"Summary -> {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
