#!/usr/bin/env python3
"""
Prepare a compact value-learning dataset from live search traces.

Inputs may be either raw search-trace JSONL files or pickled search-assist
examples produced by `training/prepare_search_assist_from_jsonl.py`.
"""

from __future__ import annotations

import argparse
import glob
import json
import pickle
from collections import Counter
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.search_assist_utils import safe_float


def _iter_examples(path: str):
    file_path = Path(path)
    if file_path.suffix == ".pkl":
        with file_path.open("rb") as handle:
            obj = pickle.load(handle)
        if not isinstance(obj, list):
            return
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


def _resolve_inputs(patterns: list[str]) -> list[str]:
    paths: list[str] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            paths.extend(matches)
        elif Path(pattern).exists():
            paths.append(pattern)
    return paths


def _build_extra_features(example: dict) -> list[float]:
    raw = example.get("state_value_features")
    if isinstance(raw, list) and raw:
        return [safe_float(v, 0.0) for v in raw]

    phase = str(example.get("phase", "") or "").strip().lower()
    switch_candidate_count = int(example.get("switch_candidate_count", 0) or 0)
    tera_candidate_count = int(example.get("tera_candidate_count", 0) or 0)
    return [
        min(1.0, safe_float(example.get("turn", 0), 0.0) / 30.0),
        1.0 if phase == "opening" else 0.0,
        1.0 if phase == "mid" else 0.0,
        1.0 if phase == "end" else 0.0,
        1.0 if bool(example.get("can_tera", False)) else 0.0,
        min(1.0, float(max(0, switch_candidate_count)) / 5.0),
        min(1.0, float(max(0, tera_candidate_count)) / 4.0),
        max(-1.0, min(1.0, safe_float(example.get("hazard_load", 0.0), 0.0))),
        max(-1.0, min(1.0, safe_float(example.get("matchup_score", 0.0), 0.0))),
        max(-1.0, min(1.0, safe_float(example.get("best_reply_score", 0.0), 0.0))),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare a leaf-value dataset from search traces.")
    parser.add_argument("--input", action="append", default=[], help="JSONL/PKL path or glob. Repeatable.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-out", default="")
    parser.add_argument("--min-turn", type=int, default=1)
    parser.add_argument("--phase", action="append", default=[], choices=["opening", "mid", "end"])
    parser.add_argument("--keep-lowconf-only", action="store_true")
    parser.add_argument("--keep-endgame-only", action="store_true")
    parser.add_argument("--keep-nonfallback-only", action="store_true")
    parser.add_argument("--weight-lowconf", type=float, default=0.0)
    parser.add_argument("--weight-endgame", type=float, default=0.0)
    args = parser.parse_args()

    paths = _resolve_inputs(args.input)
    if not paths:
        raise SystemExit("No input files matched.")

    allowed_phases = {str(p).strip().lower() for p in args.phase if str(p).strip()}
    counters = Counter()
    rows: list[dict] = []

    for path in paths:
        for ex in _iter_examples(path):
            counters["rows_seen"] += 1
            board = ex.get("board_features")
            if not isinstance(board, list) or not board:
                counters["drop_missing_board"] += 1
                continue
            value_target = safe_float(ex.get("value_target", 0.0), 0.0)
            if abs(value_target) < 1e-9:
                counters["drop_zero_value"] += 1
                continue
            turn = int(ex.get("turn", 0) or 0)
            if turn < args.min_turn:
                counters["drop_min_turn"] += 1
                continue
            phase = str(ex.get("phase", "") or "").strip().lower()
            if allowed_phases and phase not in allowed_phases:
                counters["drop_phase"] += 1
                continue
            selection_path = str(ex.get("selection_path", "") or "")
            if args.keep_nonfallback_only and selection_path.startswith("fallback"):
                counters["drop_fallback"] += 1
                continue

            confidence = safe_float(ex.get("policy_confidence", 0.0), 0.0)
            threshold = safe_float(ex.get("policy_threshold", 0.0), 0.0)
            lowconf = confidence < threshold
            if args.keep_lowconf_only and not lowconf:
                counters["drop_not_lowconf"] += 1
                continue
            endgame = phase == "end"
            if args.keep_endgame_only and not endgame:
                counters["drop_not_endgame"] += 1
                continue

            weight = 1.0
            if lowconf and args.weight_lowconf > 0.0:
                weight *= 1.0 + args.weight_lowconf
            if endgame and args.weight_endgame > 0.0:
                weight *= 1.0 + args.weight_endgame

            rows.append(
                {
                    "battle_id": str(ex.get("battle_id", "")),
                    "turn": turn,
                    "board_features": [safe_float(v, 0.0) for v in board],
                    "extra_features": _build_extra_features(ex),
                    "value_target": value_target,
                    "weight": weight,
                    "phase": phase,
                    "selection_path": selection_path,
                    "low_confidence": bool(lowconf),
                    "source": str(ex.get("source", "")),
                    "tag": str(ex.get("tag", "")),
                }
            )
            counters["kept"] += 1

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as handle:
        pickle.dump(rows, handle)

    print(f"Wrote {len(rows)} leaf-value rows -> {out_path}")
    print(f"Stats: {dict(counters)}")

    if args.summary_out:
        summary = {
            "inputs": paths,
            "output": str(out_path),
            "rows": len(rows),
            "counters": dict(counters),
            "min_turn": args.min_turn,
            "phase": sorted(allowed_phases),
            "keep_lowconf_only": args.keep_lowconf_only,
            "keep_endgame_only": args.keep_endgame_only,
            "keep_nonfallback_only": args.keep_nonfallback_only,
            "weight_lowconf": args.weight_lowconf,
            "weight_endgame": args.weight_endgame,
        }
        summary_path = Path(args.summary_out)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Summary -> {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
