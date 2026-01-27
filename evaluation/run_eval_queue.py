#!/usr/bin/env python3
"""
Run queued eval_vs_foulplay.py jobs with bounded parallelism.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_SCRIPT = PROJECT_ROOT / "evaluation" / "eval_vs_foulplay.py"

PROGRESS_RE = re.compile(r"\s*(\d+)/(\d+):")
BATTLES_RE = re.compile(r"Battles:\s+(\d+)/(\d+)")
SUMMARY_RE = re.compile(
    r"FOUL PLAY EVAL SUMMARY.*?"
    r"Battles:\s+(\d+)/(\d+).*?"
    r"W/L/T:\s+(\d+)/(\d+)/(\d+).*?"
    r"Win rate:\s+([0-9.]+)%",
    re.DOTALL,
)


def _format_eta(seconds: Optional[float]) -> str:
    if seconds is None or seconds <= 0:
        return "--:--:--"
    seconds_int = int(round(seconds))
    hours = seconds_int // 3600
    minutes = (seconds_int % 3600) // 60
    secs = seconds_int % 60
    return f"{hours}:{minutes:02d}:{secs:02d}"


@dataclass
class Job:
    name: str
    args: List[str]
    env: Dict[str, str]
    log_path: Path
    process: Optional[subprocess.Popen] = None
    log_handle: Optional[object] = None
    start_time: Optional[float] = None
    done: int = 0
    total: int = 0
    offset: int = 0
    exit_code: Optional[int] = None

    def is_running(self) -> bool:
        return self.process is not None and self.exit_code is None


def _load_queue(path: Path) -> List[dict]:
    text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        return [data]
    except json.JSONDecodeError:
        items = []
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            items.append(json.loads(line))
        return items


def _resolve_log_path(value: Optional[str], index: int) -> Path:
    if value:
        return (PROJECT_ROOT / value).resolve()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return (PROJECT_ROOT / "logs" / "evals" / "current" / f"run_{stamp}_{index}.log").resolve()


def _start_job(job: Job, python: Path, script: Path) -> None:
    job.log_path.parent.mkdir(parents=True, exist_ok=True)
    job.log_handle = job.log_path.open("w", encoding="utf-8")
    env = os.environ.copy()
    env.update(job.env)
    env.setdefault("PYTHONUNBUFFERED", "1")
    cmd = [str(python), str(script)] + job.args
    job.process = subprocess.Popen(
        cmd,
        cwd=PROJECT_ROOT,
        stdout=job.log_handle,
        stderr=subprocess.STDOUT,
        env=env,
    )
    job.start_time = time.time()


def _update_progress(job: Job) -> None:
    if not job.log_path.exists():
        return
    try:
        with job.log_path.open("r", encoding="utf-8", errors="ignore") as handle:
            handle.seek(job.offset)
            chunk = handle.read()
            job.offset = handle.tell()
    except OSError:
        return
    if not chunk:
        return
    for line in chunk.splitlines():
        match = PROGRESS_RE.search(line)
        if match:
            job.done = int(match.group(1))
            job.total = int(match.group(2))
        match = BATTLES_RE.search(line)
        if match:
            job.done = int(match.group(1))
            job.total = int(match.group(2))


def _eta_for_job(job: Job) -> Optional[float]:
    if job.start_time is None or job.done <= 0 or job.total <= 0:
        return None
    elapsed = time.time() - job.start_time
    remaining = max(0, job.total - job.done)
    return (elapsed / job.done) * remaining


def _extract_summary(log_path: Path) -> Optional[dict]:
    try:
        text = log_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None
    matches = list(SUMMARY_RE.finditer(text))
    if not matches:
        return None
    match = matches[-1]
    finished = int(match.group(1))
    wins = int(match.group(3))
    losses = int(match.group(4))
    ties = int(match.group(5))
    win_rate = float(match.group(6)) / 100.0
    return {
        "battles": finished,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "win_rate": win_rate,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run queued eval_vs_foulplay.py jobs with bounded parallelism.",
    )
    parser.add_argument("--queue", required=True, help="Path to JSON or JSONL queue file.")
    parser.add_argument("--max-parallel", type=int, default=2)
    parser.add_argument("--poll-interval", type=float, default=10.0)
    parser.add_argument("--python", default=sys.executable, help="Python to invoke eval script.")
    parser.add_argument("--script", default=str(DEFAULT_SCRIPT), help="Eval script path.")
    parser.add_argument("--aggregate", action="store_true", help="Run aggregate_eval.py after completion.")
    args = parser.parse_args()

    queue_path = Path(args.queue).expanduser()
    if not queue_path.exists():
        print(f"Queue file not found: {queue_path}")
        return 1

    items = _load_queue(queue_path)
    jobs: List[Job] = []
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            raise SystemExit(f"Queue item {idx} must be an object.")
        name = str(item.get("name") or f"run_{idx}")
        args_list = item.get("args")
        if not isinstance(args_list, list):
            raise SystemExit(f"Queue item {idx} missing args list.")
        env = item.get("env", {})
        if not isinstance(env, dict):
            raise SystemExit(f"Queue item {idx} env must be a dict.")
        log_path = _resolve_log_path(item.get("log"), idx)
        jobs.append(Job(name=name, args=[str(a) for a in args_list], env=env, log_path=log_path))

    python = Path(args.python).expanduser()
    script = Path(args.script).expanduser()
    if not python.exists():
        print(f"Python not found: {python}")
        return 1
    if not script.exists():
        print(f"Script not found: {script}")
        return 1

    pending = jobs[:]
    running: List[Job] = []
    completed: List[Job] = []
    failed: List[Job] = []

    status_len = 0

    def _print_event(message: str) -> None:
        nonlocal status_len
        if status_len:
            sys.stdout.write("\r" + (" " * status_len) + "\r")
            sys.stdout.flush()
            status_len = 0
        print(message)

    def _print_status(line: str) -> None:
        nonlocal status_len
        pad = " " * max(0, status_len - len(line))
        sys.stdout.write("\r" + line + pad)
        sys.stdout.flush()
        status_len = len(line)

    _print_event(f"Queued {len(pending)} runs. Max parallel: {args.max_parallel}")
    while pending or running:
        while pending and len(running) < args.max_parallel:
            job = pending.pop(0)
            _start_job(job, python, script)
            running.append(job)
            _print_event(f"Started {job.name} -> {job.log_path}")

        time.sleep(args.poll_interval)

        for job in list(running):
            _update_progress(job)
            if job.process is None:
                continue
            ret = job.process.poll()
            if ret is not None:
                job.exit_code = ret
                if job.log_handle is not None:
                    job.log_handle.close()
                running.remove(job)
                if ret == 0:
                    completed.append(job)
                    _print_event(f"Completed {job.name} (ok)")
                else:
                    failed.append(job)
                    _print_event(f"Completed {job.name} (exit {ret})")

        if running:
            parts = [
                f"running={len(running)} queued={len(pending)} "
                f"completed={len(completed)} failed={len(failed)}"
            ]
            for job in running:
                eta = _format_eta(_eta_for_job(job))
                progress = f"{job.done}/{job.total}" if job.total else f"{job.done}/?"
                parts.append(f"{job.name} {progress} ETA {eta}")
            _print_status(" | ".join(parts))
    if status_len:
        sys.stdout.write("\n")
        sys.stdout.flush()

    if args.aggregate:
        log_paths = [str(job.log_path) for job in completed + failed]
        if log_paths:
            agg_cmd = [
                str(python),
                str(PROJECT_ROOT / "evaluation" / "aggregate_eval.py"),
                *log_paths,
            ]
            subprocess.run(agg_cmd, cwd=PROJECT_ROOT, check=False)

    scored = []
    for job in completed + failed:
        summary = _extract_summary(job.log_path)
        if summary is None:
            continue
        summary["name"] = job.name
        summary["log"] = str(job.log_path)
        scored.append(summary)

    if scored:
        scored.sort(key=lambda s: s["win_rate"], reverse=True)
        best = scored[0]
        print(
            "Best run: {} {:.1%} ({}/{}/{}) battles={} log={}".format(
                best["name"],
                best["win_rate"],
                best["wins"],
                best["losses"],
                best["ties"],
                best["battles"],
                best["log"],
            )
        )
        if len(scored) > 1:
            print("All runs ranked:")
            for entry in scored:
                print(
                    "  {} {:.1%} ({}/{}/{}) battles={} log={}".format(
                        entry["name"],
                        entry["win_rate"],
                        entry["wins"],
                        entry["losses"],
                        entry["ties"],
                        entry["battles"],
                        entry["log"],
                    )
                )

    print(f"All done. Completed={len(completed)} Failed={len(failed)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
