#!/usr/bin/env python3
"""
Run local FoulPlay-vs-FoulPlay self-play and capture debug logs for dataset prep.

This script launches:
- bot B in `accept_challenge` mode
- bot A in `challenge_user` mode (after bot B resolves a userid)

It writes:
- two bot logs (stdout+stderr)
- one JSON manifest describing the run and file paths
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path


def _read_user_id(path: Path, timeout_s: float, proc: subprocess.Popen | None = None) -> tuple[str | None, str | None]:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if path.exists() and path.stat().st_size > 0:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                return data.get("userid"), data.get("username")
            except Exception:
                pass
        if proc is not None and proc.poll() is not None:
            return None, None
        time.sleep(0.2)
    return None, None


def _build_cmd(
    foulplay_python: str,
    ws_uri: str,
    username: str,
    password: str,
    mode: str,
    fmt: str,
    run_count: int,
    search_ms: int,
    parallelism: int,
    user_id_file: Path,
    user_to_challenge: str | None = None,
    log_level: str = "DEBUG",
) -> list[str]:
    cmd = [
        foulplay_python,
        "third_party/foul-play/run.py",
        "--websocket-uri",
        ws_uri,
        "--no-login",
        "--ps-username",
        username,
        "--ps-password",
        password,
        "--bot-mode",
        mode,
        "--pokemon-format",
        fmt,
        "--run-count",
        str(run_count),
        "--search-time-ms",
        str(search_ms),
        "--search-parallelism",
        str(parallelism),
        "--user-id-file",
        str(user_id_file),
        "--log-level",
        log_level,
    ]
    if user_to_challenge:
        cmd.extend(["--user-to-challenge", user_to_challenge])
    return cmd


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--foulplay-python", default="venv/bin/python")
    parser.add_argument("--ws-uri", default="ws://127.0.0.1:8000/showdown/websocket")
    parser.add_argument("--format", default="gen9randombattle")
    parser.add_argument("--battles", type=int, default=500)
    parser.add_argument("--search-ms", type=int, default=120)
    parser.add_argument("--parallelism", type=int, default=1)
    parser.add_argument("--bot-a-username", default="fpselfplaya")
    parser.add_argument("--bot-b-username", default="fpselfplayb")
    parser.add_argument("--bot-password", default="foulplay_pass")
    parser.add_argument("--run-tag", default="")
    parser.add_argument("--log-dir", default="logs/foulplay/selfplay")
    parser.add_argument("--manifest-out", default="")
    parser.add_argument("--userid-timeout", type=float, default=30.0)
    parser.add_argument("--max-runtime-min", type=float, default=360.0)
    args = parser.parse_args()

    run_ts = time.strftime("%Y%m%d_%H%M%S")
    run_tag = args.run_tag.strip() or f"{run_ts}_fp_selfplay_{args.battles}"
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    b_log = log_dir / f"{run_tag}_botb.log"
    a_log = log_dir / f"{run_tag}_bota.log"
    b_user_file = log_dir / f"{run_tag}_botb_user.json"
    a_user_file = log_dir / f"{run_tag}_bota_user.json"

    cmd_b = _build_cmd(
        foulplay_python=args.foulplay_python,
        ws_uri=args.ws_uri,
        username=args.bot_b_username,
        password=args.bot_password,
        mode="accept_challenge",
        fmt=args.format,
        run_count=args.battles,
        search_ms=args.search_ms,
        parallelism=args.parallelism,
        user_id_file=b_user_file,
    )
    print("Launching bot B:")
    print(" ".join(cmd_b))
    with b_log.open("w", encoding="utf-8") as b_handle:
        b_proc = subprocess.Popen(cmd_b, stdout=b_handle, stderr=subprocess.STDOUT, text=True)

    try:
        b_userid, b_username = _read_user_id(
            b_user_file,
            timeout_s=args.userid_timeout,
            proc=b_proc,
        )
        if not b_userid:
            b_rc = b_proc.poll()
            b_proc.terminate()
            raise SystemExit(
                f"bot B userid not resolved within {args.userid_timeout}s (rc={b_rc}). "
                f"Check log: {b_log}"
            )

        cmd_a = _build_cmd(
            foulplay_python=args.foulplay_python,
            ws_uri=args.ws_uri,
            username=args.bot_a_username,
            password=args.bot_password,
            mode="challenge_user",
            fmt=args.format,
            run_count=args.battles,
            search_ms=args.search_ms,
            parallelism=args.parallelism,
            user_id_file=a_user_file,
            user_to_challenge=b_userid,
        )
        print("Launching bot A:")
        print(" ".join(cmd_a))
        with a_log.open("w", encoding="utf-8") as a_handle:
            a_proc = subprocess.Popen(cmd_a, stdout=a_handle, stderr=subprocess.STDOUT, text=True)

        max_runtime_s = args.max_runtime_min * 60.0
        start = time.time()
        while True:
            a_done = a_proc.poll() is not None
            b_done = b_proc.poll() is not None
            if a_done and b_done:
                break
            if (time.time() - start) > max_runtime_s:
                print("Timeout reached; terminating both bots.")
                a_proc.terminate()
                b_proc.terminate()
                break
            time.sleep(2.0)

        a_rc = a_proc.wait(timeout=15)
        b_rc = b_proc.wait(timeout=15)
    except Exception:
        try:
            b_proc.terminate()
        except Exception:
            pass
        raise

    manifest = {
        "run_tag": run_tag,
        "created_at": run_ts,
        "ws_uri": args.ws_uri,
        "format": args.format,
        "battles_requested": args.battles,
        "bot_a": {
            "username": args.bot_a_username,
            "log": str(a_log),
            "userid_file": str(a_user_file),
            "return_code": int(a_rc),
        },
        "bot_b": {
            "username": args.bot_b_username,
            "resolved_userid": b_userid,
            "resolved_username": b_username,
            "log": str(b_log),
            "userid_file": str(b_user_file),
            "return_code": int(b_rc),
        },
    }

    manifest_path = Path(args.manifest_out) if args.manifest_out else log_dir / f"{run_tag}.manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Bot A rc={a_rc}, log={a_log}")
    print(f"Bot B rc={b_rc}, log={b_log}")
    print(f"Manifest: {manifest_path}")
    return 0 if a_rc == 0 and b_rc == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
