#!/usr/bin/env python3
"""
Run repeated eval_vs_foulplay blocks and aggregate variance-aware statistics.

Usage pattern:
  venv/bin/python evaluation/run_block_eval.py \
    --name h1 \
    --blocks 3 \
    --battles-per-block 500 \
    --foulplay-username-base fpb17h \
    --env ORANGURU_EVAL_MODE=1 \
    -- \
    --player oranguru_engine \
    --format gen9randombattle \
    ...
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_SCRIPT = PROJECT_ROOT / "evaluation" / "eval_vs_foulplay.py"
DEFAULT_STDOUT_DIR = PROJECT_ROOT / "logs" / "evals" / "current"
DEFAULT_FP_DIR = PROJECT_ROOT / "logs" / "foulplay" / "current"

PROGRESS_RE = re.compile(r"\s*(\d+)/(\d+):")
BATTLES_RE = re.compile(r"Battles:\s+(\d+)/(\d+)")
SUMMARY_RE = re.compile(
    r"FOUL PLAY EVAL SUMMARY.*?"
    r"Battles:\s+(\d+)/(\d+).*?"
    r"W/L/T:\s+(\d+)/(\d+)/(\d+).*?"
    r"Win rate:\s+([0-9.]+)%",
    re.DOTALL,
)


@dataclass
class Job:
    name: str
    run_id: str
    block_index: int
    stdout_log: Path
    foulplay_log: Path
    foulplay_user_id_file: Path
    args: List[str]
    env: Dict[str, str]
    process: Optional[subprocess.Popen] = None
    handle: Optional[object] = None
    start_time: Optional[float] = None
    offset: int = 0
    done: int = 0
    total: int = 0
    exit_code: Optional[int] = None


def _safe_name(value: str) -> str:
    return "".join(ch for ch in value.lower() if ch.isalnum()) or "run"


def _resolve_path(value: str | None, default: Path, *, dereference: bool = True) -> Path:
    if not value:
        return default
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if dereference:
        return path.resolve()
    return path.absolute()


def _parse_env(items: List[str]) -> Dict[str, str]:
    env: Dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"Invalid --env entry (expected KEY=VALUE): {item}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise SystemExit(f"Invalid --env entry with empty key: {item}")
        env[key] = value
    return env


def _upsert_arg(args: List[str], flag: str, value: str) -> List[str]:
    out = list(args)
    try:
        idx = out.index(flag)
    except ValueError:
        out.extend([flag, value])
        return out
    if idx == len(out) - 1:
        out.append(value)
    else:
        out[idx + 1] = value
    return out


def _extract_arg(args: List[str], flag: str) -> Optional[str]:
    try:
        idx = args.index(flag)
    except ValueError:
        return None
    if idx >= len(args) - 1:
        return None
    return args[idx + 1]


def _build_jobs(
    name: str,
    blocks: int,
    battles_per_block: int,
    foulplay_username_base: str,
    base_args: List[str],
    env: Dict[str, str],
    stdout_dir: Path,
    foulplay_dir: Path,
) -> List[Job]:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    safe = _safe_name(name)
    jobs: List[Job] = []
    for block_idx in range(blocks):
        run_id = f"{stamp}_{safe}_b{block_idx + 1:02d}of{blocks:02d}"
        foulplay_user = f"{_safe_name(foulplay_username_base)}b{block_idx + 1}"
        stdout_log = stdout_dir / f"{run_id}.stdout.log"
        foulplay_log = foulplay_dir / f"eval_{run_id}.log"
        foulplay_user_id = foulplay_dir / f"user_{run_id}.txt"

        args = list(base_args)
        args = _upsert_arg(args, "--battles", str(battles_per_block))
        args = _upsert_arg(args, "--foulplay-username", foulplay_user)
        args = _upsert_arg(args, "--foulplay-log", str(foulplay_log))
        args = _upsert_arg(args, "--foulplay-user-id-file", str(foulplay_user_id))

        jobs.append(
            Job(
                name=name,
                run_id=run_id,
                block_index=block_idx,
                stdout_log=stdout_log,
                foulplay_log=foulplay_log,
                foulplay_user_id_file=foulplay_user_id,
                args=args,
                env=dict(env),
            )
        )
    return jobs


def _start_job(job: Job, python_path: Path, script_path: Path) -> None:
    job.stdout_log.parent.mkdir(parents=True, exist_ok=True)
    job.foulplay_log.parent.mkdir(parents=True, exist_ok=True)
    job.foulplay_user_id_file.parent.mkdir(parents=True, exist_ok=True)
    job.handle = job.stdout_log.open("w", encoding="utf-8")

    env = os.environ.copy()
    env.update(job.env)
    env.setdefault("PYTHONUNBUFFERED", "1")
    cmd = [str(python_path), str(script_path)] + job.args
    job.process = subprocess.Popen(
        cmd,
        cwd=PROJECT_ROOT,
        stdout=job.handle,
        stderr=subprocess.STDOUT,
        env=env,
    )
    job.start_time = time.time()


def _update_progress(job: Job) -> None:
    if not job.stdout_log.exists():
        return
    try:
        with job.stdout_log.open("r", encoding="utf-8", errors="ignore") as handle:
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


def _eta_seconds(job: Job) -> Optional[float]:
    if job.start_time is None or job.done <= 0 or job.total <= 0:
        return None
    elapsed = time.time() - job.start_time
    remaining = max(0, job.total - job.done)
    return (elapsed / max(1, job.done)) * remaining


def _format_eta(seconds: Optional[float]) -> str:
    if seconds is None or seconds <= 0:
        return "--:--:--"
    sec = int(round(seconds))
    hours = sec // 3600
    minutes = (sec % 3600) // 60
    secs = sec % 60
    return f"{hours}:{minutes:02d}:{secs:02d}"


def _extract_summary(log_path: Path) -> Optional[dict]:
    try:
        text = log_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None
    matches = list(SUMMARY_RE.finditer(text))
    if not matches:
        return None
    match = matches[-1]
    battles = int(match.group(1))
    wins = int(match.group(3))
    losses = int(match.group(4))
    ties = int(match.group(5))
    win_rate = float(match.group(6)) / 100.0
    return {
        "battles": battles,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "win_rate": win_rate,
    }


def _tail_text(path: Path, max_lines: int = 20) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return ""
    if not lines:
        return ""
    return "\n".join(lines[-max_lines:])


def _wilson_interval(wins: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    phat = wins / total
    denom = 1.0 + (z * z) / total
    centre = phat + (z * z) / (2.0 * total)
    margin = z * math.sqrt((phat * (1.0 - phat) + (z * z) / (4.0 * total)) / total)
    lo = (centre - margin) / denom
    hi = (centre + margin) / denom
    return max(0.0, lo), min(1.0, hi)


def _summarize_blocks(name: str, results: List[dict]) -> dict:
    completed = [r for r in results if r.get("summary") is not None]
    battles = sum(r["summary"]["battles"] for r in completed)
    wins = sum(r["summary"]["wins"] for r in completed)
    losses = sum(r["summary"]["losses"] for r in completed)
    ties = sum(r["summary"]["ties"] for r in completed)
    block_rates = [r["summary"]["win_rate"] for r in completed]
    pooled = (wins / battles) if battles > 0 else 0.0
    wilson_lo, wilson_hi = _wilson_interval(wins, battles)
    mean = statistics.mean(block_rates) if block_rates else 0.0
    std = statistics.stdev(block_rates) if len(block_rates) >= 2 else 0.0
    block_ci = 0.0
    if len(block_rates) >= 2:
        block_ci = 1.96 * std / math.sqrt(len(block_rates))
    return {
        "name": name,
        "blocks_requested": len(results),
        "blocks_completed": len(completed),
        "battles": battles,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "pooled_win_rate": pooled,
        "pooled_wilson_low": wilson_lo,
        "pooled_wilson_high": wilson_hi,
        "mean_block_win_rate": mean,
        "block_std": std,
        "block_ci95_halfwidth": block_ci,
        "blocks": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run repeated eval_vs_foulplay blocks and report variance-aware stats.",
    )
    parser.add_argument("--name", required=True, help="Logical name for this config.")
    parser.add_argument("--blocks", type=int, required=True, help="Number of repeated blocks.")
    parser.add_argument(
        "--battles-per-block", type=int, required=True, help="Battles to run in each block."
    )
    parser.add_argument("--max-parallel", type=int, default=1, help="How many blocks to run at once.")
    parser.add_argument("--poll-interval", type=float, default=10.0)
    parser.add_argument(
        "--foulplay-username-base",
        required=True,
        help="Base foulplay username; block index suffixes are appended automatically.",
    )
    parser.add_argument("--python", default=sys.executable, help="Python used to invoke eval script.")
    parser.add_argument("--script", default=str(DEFAULT_SCRIPT), help="Eval script path.")
    parser.add_argument("--stdout-dir", default=str(DEFAULT_STDOUT_DIR))
    parser.add_argument("--foulplay-log-dir", default=str(DEFAULT_FP_DIR))
    parser.add_argument("--summary-out", default=None, help="Optional JSON summary output path.")
    parser.add_argument("--env", action="append", default=[], help="Extra env as KEY=VALUE.")
    parser.add_argument(
        "eval_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to eval_vs_foulplay.py after `--`.",
    )
    args = parser.parse_args()

    forward_args = list(args.eval_args)
    if forward_args and forward_args[0] == "--":
        forward_args = forward_args[1:]
    if not forward_args:
        raise SystemExit("Missing forwarded eval args. Pass them after `--`.")
    if _extract_arg(forward_args, "--player") is None:
        raise SystemExit(
            "Missing forwarded `--player` arg. "
            "Pass eval_vs_foulplay.py args after `--`, for example `-- --player oranguru_engine ...`."
        )

    # Do not dereference the interpreter path. For virtualenvs, resolving
    # `venv/bin/python` often collapses back to the system interpreter and
    # drops the venv site-packages.
    python_path = _resolve_path(args.python, Path(args.python), dereference=False)
    script_path = _resolve_path(args.script, DEFAULT_SCRIPT)
    stdout_dir = _resolve_path(args.stdout_dir, DEFAULT_STDOUT_DIR)
    foulplay_dir = _resolve_path(args.foulplay_log_dir, DEFAULT_FP_DIR)
    summary_out = _resolve_path(args.summary_out, Path(args.summary_out)) if args.summary_out else None

    env = _parse_env(args.env)
    jobs = _build_jobs(
        name=args.name,
        blocks=args.blocks,
        battles_per_block=args.battles_per_block,
        foulplay_username_base=args.foulplay_username_base,
        base_args=forward_args,
        env=env,
        stdout_dir=stdout_dir,
        foulplay_dir=foulplay_dir,
    )

    pending = jobs[:]
    running: List[Job] = []
    finished: List[Job] = []
    status_len = 0

    def _clear_status() -> None:
        nonlocal status_len
        if status_len:
            sys.stdout.write("\r" + (" " * status_len) + "\r")
            sys.stdout.flush()
            status_len = 0

    def _event(text: str) -> None:
        _clear_status()
        print(text, flush=True)

    def _status(text: str) -> None:
        nonlocal status_len
        pad = " " * max(0, status_len - len(text))
        sys.stdout.write("\r" + text + pad)
        sys.stdout.flush()
        status_len = len(text)

    _event(f"Queued {len(pending)} blocks for {args.name}. Max parallel: {args.max_parallel}")

    while pending or running:
        while pending and len(running) < max(1, args.max_parallel):
            job = pending.pop(0)
            _start_job(job, python_path, script_path)
            running.append(job)
            _event(f"Started {job.run_id} -> {job.stdout_log}")

        time.sleep(max(1.0, args.poll_interval))

        for job in list(running):
            _update_progress(job)
            if job.process is None:
                continue
            ret = job.process.poll()
            if ret is None:
                continue
            job.exit_code = ret
            if job.handle is not None:
                job.handle.close()
            running.remove(job)
            finished.append(job)
            if ret == 0:
                _event(f"Completed {job.run_id} (ok)")
            else:
                _event(f"Completed {job.run_id} (exit {ret})")
                tail = _tail_text(job.stdout_log, max_lines=12)
                if tail:
                    _event(f"Last log lines for {job.run_id}:\n{tail}")

        if running:
            parts = [f"running={len(running)} queued={len(pending)} finished={len(finished)}"]
            for job in running:
                progress = f"{job.done}/{job.total}" if job.total else f"{job.done}/?"
                parts.append(f"{job.run_id} {progress} ETA {_format_eta(_eta_seconds(job))}")
            _status(" | ".join(parts))

    _clear_status()

    results = []
    failed = 0
    for job in sorted(finished, key=lambda item: item.block_index):
        summary = _extract_summary(job.stdout_log)
        if job.exit_code != 0:
            failed += 1
        results.append(
            {
                "block_index": job.block_index + 1,
                "run_id": job.run_id,
                "exit_code": job.exit_code,
                "stdout_log": str(job.stdout_log),
                "foulplay_log": str(job.foulplay_log),
                "summary": summary,
            }
        )

    report = _summarize_blocks(args.name, results)

    print("")
    print(f"Block summary for {args.name}")
    print(
        "Pooled win rate: {:.2f}% ({}/{}) 95% CI [{:.2f}%, {:.2f}%]".format(
            100.0 * report["pooled_win_rate"],
            report["wins"],
            report["battles"],
            100.0 * report["pooled_wilson_low"],
            100.0 * report["pooled_wilson_high"],
        )
    )
    print(
        "Block mean/std: {:.2f}% / {:.2f}% over {} blocks".format(
            100.0 * report["mean_block_win_rate"],
            100.0 * report["block_std"],
            report["blocks_completed"],
        )
    )
    if report["block_ci95_halfwidth"] > 0:
        print(
            "Block-mean 95% CI: {:.2f}% +/- {:.2f}%".format(
                100.0 * report["mean_block_win_rate"],
                100.0 * report["block_ci95_halfwidth"],
            )
        )
    print("Blocks:")
    for block in report["blocks"]:
        summary = block.get("summary")
        if summary is None:
            print(
                f"- block {block['block_index']}: exit={block['exit_code']} summary=missing "
                f"log={block['stdout_log']}"
            )
            continue
        print(
            "- block {}: {:.2f}% ({}/{}) exit={} log={}".format(
                block["block_index"],
                100.0 * summary["win_rate"],
                summary["wins"],
                summary["battles"],
                block["exit_code"],
                block["stdout_log"],
            )
        )

    if summary_out is not None:
        summary_out.parent.mkdir(parents=True, exist_ok=True)
        summary_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Summary JSON: {summary_out}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
