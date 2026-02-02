#!/usr/bin/env python3
import argparse
import glob
import json
import os
import re
from dataclasses import dataclass
from typing import Iterable, List


SECTION_RE = re.compile(r"📌 MCTS Diagnostics(.*?)(?:\n📌 |\n=+|\Z)", re.DOTALL)
CALLS_RE = re.compile(r"Calls/states:\s*(\d+)\s*/\s*(\d+)")
EMPTY_RE = re.compile(r"Empty results:\s*(\d+)\s*\(([\d.]+)%\)")
FAIL_RE = re.compile(r"State fails:\s*(\d+)\s*\(([\d.]+)%\)")
DET_RE = re.compile(r"Deterministic:\s*(\d+)\s*\|\s*Stochastic:\s*(\d+)")
FALLBACK_RE = re.compile(
    r"Fallback\(super/random\):\s*(\d+)\s*/\s*(\d+)\s*\(([\d.]+)%/([\d.]+)%\)"
)


@dataclass
class MCTSDiag:
    source: str
    calls: int
    states: int
    empty_results: int
    state_fails: int
    deterministic: int
    stochastic: int
    fallback_super: int
    fallback_random: int

    @property
    def empty_rate(self) -> float:
        return self.empty_results / max(1, self.calls)

    @property
    def state_fail_rate(self) -> float:
        return self.state_fails / max(1, self.states)

    @property
    def fallback_super_rate(self) -> float:
        return self.fallback_super / max(1, self.calls)

    @property
    def fallback_random_rate(self) -> float:
        return self.fallback_random / max(1, self.calls)


def _load_sources(paths: Iterable[str], globs: Iterable[str]) -> List[str]:
    sources = list(paths)
    for pattern in globs:
        sources.extend(sorted(glob.glob(pattern)))
    return sources


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        return handle.read()


def _parse_section(section: str, source: str) -> MCTSDiag | None:
    m_calls = CALLS_RE.search(section)
    m_empty = EMPTY_RE.search(section)
    m_fail = FAIL_RE.search(section)
    m_det = DET_RE.search(section)
    m_fb = FALLBACK_RE.search(section)
    if not all((m_calls, m_empty, m_fail, m_det, m_fb)):
        return None
    return MCTSDiag(
        source=source,
        calls=int(m_calls.group(1)),
        states=int(m_calls.group(2)),
        empty_results=int(m_empty.group(1)),
        state_fails=int(m_fail.group(1)),
        deterministic=int(m_det.group(1)),
        stochastic=int(m_det.group(2)),
        fallback_super=int(m_fb.group(1)),
        fallback_random=int(m_fb.group(2)),
    )


def _parse_text(text: str, source: str) -> List[MCTSDiag]:
    runs: List[MCTSDiag] = []
    for match in SECTION_RE.finditer(text):
        run = _parse_section(match.group(1), source)
        if run is not None:
            runs.append(run)
    return runs


def _print_summary(runs: List[MCTSDiag]) -> None:
    if not runs:
        print("No MCTS diagnostics found.")
        return

    print(f"Found {len(runs)} run(s).")
    for run in runs:
        print(
            "- {} calls={} states={} empty={:.1%} fail={:.1%} "
            "fallback(super/random)={:.1%}/{:.1%} det/stoch={}/{}".format(
                run.source,
                run.calls,
                run.states,
                run.empty_rate,
                run.state_fail_rate,
                run.fallback_super_rate,
                run.fallback_random_rate,
                run.deterministic,
                run.stochastic,
            )
        )

    calls = sum(r.calls for r in runs)
    states = sum(r.states for r in runs)
    empty = sum(r.empty_results for r in runs)
    fails = sum(r.state_fails for r in runs)
    fb_super = sum(r.fallback_super for r in runs)
    fb_random = sum(r.fallback_random for r in runs)
    det = sum(r.deterministic for r in runs)
    stoch = sum(r.stochastic for r in runs)
    print("Totals:")
    print(f"- calls/states: {calls}/{states}")
    print(f"- empty rate: {empty / max(1, calls):.1%}")
    print(f"- state fail rate: {fails / max(1, states):.1%}")
    print(
        "- fallback(super/random): "
        f"{fb_super / max(1, calls):.1%}/{fb_random / max(1, calls):.1%}"
    )
    print(f"- deterministic/stochastic: {det}/{stoch}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate MCTS diagnostics sections from eval logs.",
    )
    parser.add_argument("paths", nargs="*", help="Paths to log files to scan.")
    parser.add_argument(
        "--glob",
        action="append",
        default=[],
        help="Glob patterns to add (can be repeated).",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON output.")
    args = parser.parse_args()

    sources = _load_sources(args.paths, args.glob)
    if not sources:
        print("No log files matched.")
        return 1

    runs: List[MCTSDiag] = []
    for source in sources:
        if not os.path.isfile(source):
            continue
        text = _read_text(source)
        runs.extend(_parse_text(text, source))

    if args.json:
        payload = [
            {
                "source": run.source,
                "calls": run.calls,
                "states": run.states,
                "empty_results": run.empty_results,
                "empty_rate": run.empty_rate,
                "state_fails": run.state_fails,
                "state_fail_rate": run.state_fail_rate,
                "deterministic": run.deterministic,
                "stochastic": run.stochastic,
                "fallback_super": run.fallback_super,
                "fallback_super_rate": run.fallback_super_rate,
                "fallback_random": run.fallback_random,
                "fallback_random_rate": run.fallback_random_rate,
            }
            for run in runs
        ]
        print(json.dumps(payload, indent=2))
        return 0

    _print_summary(runs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
