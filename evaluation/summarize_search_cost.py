#!/usr/bin/env python3
"""
Summarize search-cost tradeoffs from block eval summaries and stdout logs.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


CALLS_STATES_RE = re.compile(r"Calls/states:\s+(\d+)/(\d+)")
AVG_WORLDS_RE = re.compile(
    r"Avg worlds req/budget/gen/searched:\s+([0-9.]+)/([0-9.]+)/([0-9.]+)/([0-9.]+)"
)
WORLD_KEEP_RE = re.compile(
    r"World keep rate:\s+([0-9.]+)% \| Low-unc turns/saved:\s+(\d+)/(\d+) \| Endgame turns/saved:\s+(\d+)/(\d+)"
)


def _load_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _parse_stdout_metrics(path: str) -> dict:
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    metrics = {
        "calls": 0,
        "states": 0,
        "req_worlds_total": 0.0,
        "budget_worlds_total": 0.0,
        "generated_worlds_total": 0.0,
        "kept_worlds_total": 0.0,
        "low_unc_turns": 0,
        "low_unc_saved": 0,
        "endgame_turns": 0,
        "endgame_saved": 0,
    }

    match = CALLS_STATES_RE.search(text)
    if match:
        metrics["calls"] = int(match.group(1))
        metrics["states"] = int(match.group(2))

    match = AVG_WORLDS_RE.search(text)
    if match and metrics["calls"] > 0:
        req = float(match.group(1))
        budget = float(match.group(2))
        generated = float(match.group(3))
        kept = float(match.group(4))
        metrics["req_worlds_total"] = req * metrics["calls"]
        metrics["budget_worlds_total"] = budget * metrics["calls"]
        metrics["generated_worlds_total"] = generated * metrics["calls"]
        metrics["kept_worlds_total"] = kept * metrics["calls"]

    match = WORLD_KEEP_RE.search(text)
    if match:
        metrics["low_unc_turns"] = int(match.group(2))
        metrics["low_unc_saved"] = int(match.group(3))
        metrics["endgame_turns"] = int(match.group(4))
        metrics["endgame_saved"] = int(match.group(5))

    return metrics


def _summarize(summary_path: str) -> dict:
    payload = _load_json(summary_path)
    blocks = payload.get("blocks", []) or []
    agg = {
        "name": str(payload.get("name", Path(summary_path).stem)),
        "summary_path": summary_path,
        "battles": int(payload.get("battles", 0) or 0),
        "wins": int(payload.get("wins", 0) or 0),
        "losses": int(payload.get("losses", 0) or 0),
        "ties": int(payload.get("ties", 0) or 0),
        "win_rate": float(payload.get("pooled_win_rate", 0.0) or 0.0),
        "calls": 0,
        "states": 0,
        "req_worlds_total": 0.0,
        "budget_worlds_total": 0.0,
        "generated_worlds_total": 0.0,
        "kept_worlds_total": 0.0,
        "low_unc_turns": 0,
        "low_unc_saved": 0,
        "endgame_turns": 0,
        "endgame_saved": 0,
        "logs_parsed": 0,
    }

    for block in blocks:
        stdout_log = str(block.get("stdout_log", "") or "")
        if not stdout_log:
            continue
        path = Path(stdout_log)
        if not path.exists():
            continue
        parsed = _parse_stdout_metrics(stdout_log)
        agg["logs_parsed"] += 1
        for key in (
            "calls",
            "states",
            "req_worlds_total",
            "budget_worlds_total",
            "generated_worlds_total",
            "kept_worlds_total",
            "low_unc_turns",
            "low_unc_saved",
            "endgame_turns",
            "endgame_saved",
        ):
            agg[key] += parsed[key]

    calls = max(1, agg["calls"])
    battles = max(1, agg["battles"])
    agg["states_per_battle"] = agg["states"] / battles
    agg["calls_per_battle"] = agg["calls"] / battles
    agg["states_per_call"] = agg["states"] / calls
    agg["avg_req_worlds_per_call"] = agg["req_worlds_total"] / calls
    agg["avg_budget_worlds_per_call"] = agg["budget_worlds_total"] / calls
    agg["avg_generated_worlds_per_call"] = agg["generated_worlds_total"] / calls
    agg["avg_kept_worlds_per_call"] = agg["kept_worlds_total"] / calls
    return agg


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize search cost from block eval outputs.")
    parser.add_argument("--summary", action="append", required=True, help="Block eval JSON summary path. Repeatable.")
    parser.add_argument("--summary-out", default="")
    args = parser.parse_args()

    rows = [_summarize(path) for path in args.summary]

    print(
        "name".ljust(24),
        "win%".rjust(7),
        "states/b".rjust(10),
        "calls/b".rjust(9),
        "states/c".rjust(10),
        "req/bud/gen/kept".rjust(22),
        "lowunc_saved".rjust(13),
        "endg_saved".rjust(11),
    )
    for row in rows:
        print(
            row["name"][:24].ljust(24),
            f"{100.0 * row['win_rate']:.2f}".rjust(7),
            f"{row['states_per_battle']:.1f}".rjust(10),
            f"{row['calls_per_battle']:.2f}".rjust(9),
            f"{row['states_per_call']:.1f}".rjust(10),
            (
                f"{row['avg_req_worlds_per_call']:.2f}/"
                f"{row['avg_budget_worlds_per_call']:.2f}/"
                f"{row['avg_generated_worlds_per_call']:.2f}/"
                f"{row['avg_kept_worlds_per_call']:.2f}"
            ).rjust(22),
            str(int(row["low_unc_saved"])).rjust(13),
            str(int(row["endgame_saved"])).rjust(11),
        )

    if args.summary_out:
        out = Path(args.summary_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"runs": rows}, indent=2), encoding="utf-8")
        print(f"Summary -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
