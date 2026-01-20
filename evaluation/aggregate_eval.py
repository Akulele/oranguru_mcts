#!/usr/bin/env python3
import argparse
import glob
import json
import os
import re
import sys
from dataclasses import dataclass
from typing import Iterable, List, Optional


@dataclass
class RunSummary:
    kind: str
    battles: int
    wins: int
    losses: int
    ties: int
    win_rate: float
    source: str


FOULPLAY_RE = re.compile(
    r"FOUL PLAY EVAL SUMMARY.*?"
    r"Battles:\s+(\d+)/(\d+).*?"
    r"W/L/T:\s+(\d+)/(\d+)/(\d+).*?"
    r"Win rate:\s+([0-9.]+)%",
    re.DOTALL,
)
HEURISTICS_RE = re.compile(r"Final:\s+(\d+)/(\d+)\s+wins\s+\(([0-9.]+)%\)")
LADDER_RE = re.compile(
    r"Finished\s+(\d+)\s+battles:\s+(\d+)W/(\d+)L/(\d+)T\s+\(([0-9.]+)% win rate\)",
    re.IGNORECASE,
)


def _parse_foulplay(text: str, source: str) -> List[RunSummary]:
    runs = []
    for match in FOULPLAY_RE.finditer(text):
        finished = int(match.group(1))
        wins = int(match.group(3))
        losses = int(match.group(4))
        ties = int(match.group(5))
        win_rate = float(match.group(6)) / 100.0
        runs.append(
            RunSummary(
                kind="foulplay",
                battles=finished,
                wins=wins,
                losses=losses,
                ties=ties,
                win_rate=win_rate,
                source=source,
            )
        )
    return runs


def _parse_heuristics(text: str, source: str) -> List[RunSummary]:
    runs = []
    for match in HEURISTICS_RE.finditer(text):
        wins = int(match.group(1))
        battles = int(match.group(2))
        win_rate = float(match.group(3)) / 100.0
        losses = max(0, battles - wins)
        runs.append(
            RunSummary(
                kind="heuristics",
                battles=battles,
                wins=wins,
                losses=losses,
                ties=0,
                win_rate=win_rate,
                source=source,
            )
        )
    return runs


def _parse_ladder(text: str, source: str) -> List[RunSummary]:
    runs = []
    for match in LADDER_RE.finditer(text):
        battles = int(match.group(1))
        wins = int(match.group(2))
        losses = int(match.group(3))
        ties = int(match.group(4))
        win_rate = float(match.group(5)) / 100.0
        runs.append(
            RunSummary(
                kind="ladder",
                battles=battles,
                wins=wins,
                losses=losses,
                ties=ties,
                win_rate=win_rate,
                source=source,
            )
        )
    return runs


def _load_sources(paths: Iterable[str], globs: Iterable[str]) -> List[str]:
    sources = list(paths)
    for pattern in globs:
        sources.extend(sorted(glob.glob(pattern)))
    return sources


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def _summarize(runs: List[RunSummary]) -> None:
    if not runs:
        print("No summaries found.")
        return

    totals = {}
    for run in runs:
        bucket = totals.setdefault(run.kind, {"battles": 0, "wins": 0, "losses": 0, "ties": 0})
        bucket["battles"] += run.battles
        bucket["wins"] += run.wins
        bucket["losses"] += run.losses
        bucket["ties"] += run.ties

    print(f"Found {len(runs)} runs.")
    for run in runs:
        print(
            f"- {run.kind}: {run.wins}/{run.losses}/{run.ties} "
            f"({run.wins / max(1, run.battles):.1%}) battles={run.battles} "
            f"source={run.source}"
        )
    print("Totals:")
    for kind, bucket in totals.items():
        win_rate = bucket["wins"] / max(1, bucket["battles"])
        print(
            f"- {kind}: {bucket['wins']}/{bucket['losses']}/{bucket['ties']} "
            f"({win_rate:.1%}) battles={bucket['battles']}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate eval summaries from stdout logs.",
    )
    parser.add_argument("paths", nargs="*", help="Paths to log files to scan.")
    parser.add_argument(
        "--glob",
        action="append",
        default=[],
        help="Glob patterns to add (can be repeated).",
    )
    parser.add_argument(
        "--kind",
        choices=["foulplay", "heuristics", "ladder", "all"],
        default="all",
        help="Filter to a single summary type.",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON output.")
    args = parser.parse_args()

    sources = _load_sources(args.paths, args.glob)
    runs: List[RunSummary] = []

    if not sources:
        if sys.stdin.isatty():
            print("No log files matched and no stdin provided.")
            return 1
        text = sys.stdin.read()
        sources = ["<stdin>"]
        texts = [text]
    else:
        texts = [_read_text(path) for path in sources]

    for source, text in zip(sources, texts):
        runs.extend(_parse_foulplay(text, source))
        runs.extend(_parse_heuristics(text, source))
        runs.extend(_parse_ladder(text, source))

    if args.kind != "all":
        runs = [run for run in runs if run.kind == args.kind]

    if args.json:
        payload = [
            {
                "kind": run.kind,
                "battles": run.battles,
                "wins": run.wins,
                "losses": run.losses,
                "ties": run.ties,
                "win_rate": run.win_rate,
                "source": run.source,
            }
            for run in runs
        ]
        print(json.dumps(payload, indent=2))
        return 0

    _summarize(runs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
