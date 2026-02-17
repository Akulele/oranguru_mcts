#!/usr/bin/env python3
"""
Run one ladder worker per account and sample per-process CPU/memory.

Default account selection prefers the first three accounts whose usernames start
with "orangurutestacc" from the provided accounts file.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import subprocess
import time
from pathlib import Path


def load_accounts(path: Path) -> list[tuple[str, str]]:
    accounts: list[tuple[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw or raw.startswith("#"):
                continue
            if ":" in raw:
                username, password = raw.split(":", 1)
            elif "," in raw:
                username, password = raw.split(",", 1)
            else:
                parts = raw.split()
                if len(parts) < 2:
                    continue
                username, password = parts[0], parts[1]
            username = username.strip()
            password = password.strip()
            if username and password:
                accounts.append((username, password))
    return accounts


def pick_accounts(
    all_accounts: list[tuple[str, str]], names: list[str], limit: int
) -> list[str]:
    if names:
        wanted = {name.strip().lower() for name in names if name.strip()}
        return [user for user, _ in all_accounts if user.lower() in wanted][:limit]

    test_accounts = [
        user for user, _ in all_accounts if user.lower().startswith("orangurutestacc")
    ]
    if len(test_accounts) >= limit:
        return test_accounts[:limit]
    return [user for user, _ in all_accounts][:limit]


def read_ps_usage(pid: int) -> tuple[float, int] | None:
    try:
        out = subprocess.check_output(
            ["ps", "-p", str(pid), "-o", "%cpu=,rss="],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None
    if not out:
        return None
    parts = out.split()
    if len(parts) < 2:
        return None
    try:
        return float(parts[0]), int(parts[1])
    except ValueError:
        return None


def load_trajectories(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
    except Exception:
        return []
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    return []


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--accounts-file", default="data/ladder_accounts.txt")
    parser.add_argument("--account-name", action="append", default=[])
    parser.add_argument("--accounts-limit", type=int, default=3)
    parser.add_argument("--python-bin", default="venv/bin/python")
    parser.add_argument("--battles", type=int, default=200)
    parser.add_argument("--format", default="gen9randombattle")
    parser.add_argument("--player", choices=["oranguru_engine", "rulebot", "rl"], default="oranguru_engine")
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--sample-sec", type=float, default=5.0)
    parser.add_argument("--max-retries", type=int, default=30)
    parser.add_argument("--retry-wait", type=float, default=5.0)
    parser.add_argument("--login-timeout", type=float, default=60.0)
    parser.add_argument("--run-tag", default="")
    parser.add_argument("--log-dir", default="logs/ladder/current")
    parser.add_argument("--cpu-log-dir", default="logs/ladder/cpu")
    parser.add_argument("--collect-data", action="store_true")
    parser.add_argument("--data-output-dir", default="data/ladder_raw")
    parser.add_argument("--merged-data-output", default="data/ladder_trajectories.pkl")
    parser.add_argument("--data-min-turns", type=int, default=0)
    parser.add_argument("--data-min-actions", type=int, default=0)
    parser.add_argument("--data-min-rating", type=int, default=0)
    parser.add_argument("--data-min-player-rating", type=int, default=0)
    parser.add_argument("--data-min-opponent-rating", type=int, default=0)
    parser.add_argument("--data-skip-forfeit", action="store_true")
    parser.add_argument("--data-max-illegal-rate", type=float, default=-1.0)
    parser.add_argument("--data-win-weight", type=float, default=1.0)
    parser.add_argument("--data-loss-weight", type=float, default=1.0)
    parser.add_argument("--data-tag-prefix", default="ladder_triplet")
    parser.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="Extra arg to pass to evaluation/ladder_rulebot.py (repeatable).",
    )
    args = parser.parse_args()

    run_ts = time.strftime("%Y%m%d_%H%M%S")
    run_tag = args.run_tag.strip() or "triplet"

    accounts_file = Path(args.accounts_file)
    all_accounts = load_accounts(accounts_file)
    selected_users = pick_accounts(all_accounts, args.account_name, args.accounts_limit)
    if not selected_users:
        print("No accounts selected.")
        return 1

    log_dir = Path(args.log_dir)
    cpu_log_dir = Path(args.cpu_log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    cpu_log_dir.mkdir(parents=True, exist_ok=True)
    data_output_dir = Path(args.data_output_dir)
    if args.collect_data:
        data_output_dir.mkdir(parents=True, exist_ok=True)

    procs: dict[str, subprocess.Popen] = {}
    stdout_files: dict[str, object] = {}
    cpu_rows: dict[str, list[tuple[float, float, int]]] = {u: [] for u in selected_users}
    cpu_writers: dict[str, csv.writer] = {}
    cpu_handles: dict[str, object] = {}
    data_paths: dict[str, Path] = {}

    try:
        for user in selected_users:
            user_slug = "".join(c if c.isalnum() else "_" for c in user.lower())
            stdout_path = log_dir / f"{run_ts}_{run_tag}_{user_slug}.stdout.log"
            snapshot_path = log_dir / f"{run_ts}_{run_tag}_{user_slug}.snapshots.jsonl"
            cpu_csv_path = cpu_log_dir / f"{run_ts}_{run_tag}_{user_slug}.cpu.csv"

            stdout_fh = stdout_path.open("w", encoding="utf-8")
            cpu_fh = cpu_csv_path.open("w", encoding="utf-8", newline="")
            writer = csv.writer(cpu_fh)
            writer.writerow(["epoch_s", "cpu_pct", "rss_kb"])

            cmd = [
                args.python_bin,
                "evaluation/ladder_rulebot.py",
                "--player",
                args.player,
                "--battles",
                str(args.battles),
                "--format",
                args.format,
                "--accounts-file",
                str(accounts_file),
                "--account-name",
                user,
                "--progress-every",
                str(args.progress_every),
                "--snapshot-every",
                "1",
                "--snapshot-log",
                str(snapshot_path),
                "--auto-reconnect",
                "--max-retries",
                str(args.max_retries),
                "--retry-wait",
                str(args.retry_wait),
                "--rejoin-active",
                "--login-timeout",
                str(args.login_timeout),
            ]
            if args.collect_data:
                data_path = data_output_dir / f"{run_ts}_{run_tag}_{user_slug}.pkl"
                data_paths[user] = data_path
                cmd.extend([
                    "--collect-data",
                    "--data-output",
                    str(data_path),
                    "--tag-prefix",
                    f"{args.data_tag_prefix}_{user_slug}",
                    "--win-weight",
                    str(args.data_win_weight),
                    "--loss-weight",
                    str(args.data_loss_weight),
                ])
                if args.data_min_turns > 0:
                    cmd.extend(["--min-turns", str(args.data_min_turns)])
                if args.data_min_actions > 0:
                    cmd.extend(["--min-actions", str(args.data_min_actions)])
                if args.data_min_rating > 0:
                    cmd.extend(["--min-rating", str(args.data_min_rating)])
                if args.data_min_player_rating > 0:
                    cmd.extend(["--min-player-rating", str(args.data_min_player_rating)])
                if args.data_min_opponent_rating > 0:
                    cmd.extend(["--min-opponent-rating", str(args.data_min_opponent_rating)])
                if args.data_skip_forfeit:
                    cmd.append("--skip-forfeit")
                if args.data_max_illegal_rate > 0:
                    cmd.extend(["--max-illegal-rate", str(args.data_max_illegal_rate)])
            cmd.extend(args.extra_arg)

            print(f"Launching {user}: {' '.join(cmd)}")
            proc = subprocess.Popen(
                cmd,
                stdout=stdout_fh,
                stderr=subprocess.STDOUT,
                text=True,
                env=os.environ.copy(),
            )
            procs[user] = proc
            stdout_files[user] = stdout_fh
            cpu_handles[user] = cpu_fh
            cpu_writers[user] = writer

        while True:
            running = 0
            now = time.time()
            for user, proc in procs.items():
                if proc.poll() is None:
                    running += 1
                    usage = read_ps_usage(proc.pid)
                    if usage is None:
                        continue
                    cpu_pct, rss_kb = usage
                    cpu_rows[user].append((now, cpu_pct, rss_kb))
                    cpu_writers[user].writerow([f"{now:.3f}", f"{cpu_pct:.2f}", rss_kb])
                    cpu_handles[user].flush()
            if running == 0:
                break
            time.sleep(max(0.25, args.sample_sec))
    finally:
        for fh in stdout_files.values():
            try:
                fh.close()
            except Exception:
                pass
        for fh in cpu_handles.values():
            try:
                fh.close()
            except Exception:
                pass

    summary: dict[str, dict[str, float | int]] = {}
    overall_ok = True
    for user, proc in procs.items():
        code = proc.returncode
        if code != 0:
            overall_ok = False
        rows = cpu_rows[user]
        if rows:
            avg_cpu = sum(r[1] for r in rows) / len(rows)
            max_cpu = max(r[1] for r in rows)
            avg_rss = sum(r[2] for r in rows) / len(rows)
            max_rss = max(r[2] for r in rows)
        else:
            avg_cpu = 0.0
            max_cpu = 0.0
            avg_rss = 0.0
            max_rss = 0.0
        summary[user] = {
            "return_code": int(code if code is not None else -1),
            "samples": len(rows),
            "avg_cpu_pct": round(avg_cpu, 2),
            "max_cpu_pct": round(max_cpu, 2),
            "avg_rss_kb": round(avg_rss, 1),
            "max_rss_kb": int(max_rss),
        }

    summary_path = cpu_log_dir / f"{run_ts}_{run_tag}_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    merged_count = 0
    merged_output = Path(args.merged_data_output) if args.merged_data_output else None
    if args.collect_data and merged_output:
        merged: list[dict] = []
        per_account: dict[str, int] = {}
        for user in selected_users:
            path = data_paths.get(user)
            traj = load_trajectories(path) if path else []
            per_account[user] = len(traj)
            merged.extend(traj)

        merged_count = len(merged)
        merged_output.parent.mkdir(parents=True, exist_ok=True)
        with merged_output.open("wb") as handle:
            pickle.dump(merged, handle)

        merged_meta = {
            "run_ts": run_ts,
            "run_tag": run_tag,
            "merged_count": merged_count,
            "per_account": per_account,
            "inputs": {user: str(path) for user, path in data_paths.items()},
        }
        meta_path = merged_output.with_suffix(merged_output.suffix + ".meta.json")
        with meta_path.open("w", encoding="utf-8") as handle:
            json.dump(merged_meta, handle, indent=2, sort_keys=True)

    print("\nCPU Summary")
    for user, stats in summary.items():
        print(
            f"- {user}: rc={stats['return_code']} samples={stats['samples']} "
            f"cpu(avg/max)={stats['avg_cpu_pct']}/{stats['max_cpu_pct']} "
            f"rss_kb(avg/max)={stats['avg_rss_kb']}/{stats['max_rss_kb']}"
        )
    print(f"Summary JSON: {summary_path}")
    if args.collect_data and merged_output:
        print(f"Merged trajectories: {merged_count} -> {merged_output}")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
