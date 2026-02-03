#!/usr/bin/env python3
"""
Stress-test only the challenge handshake vs Foul Play.

This repeatedly challenges Foul Play and immediately forfeits, so failures in
challenge/cancel/login flow can be reproduced quickly without full battles.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.players.rule_bot import RuleBotPlayer
from src.utils.server_config import get_server_configuration
from poke_env.ps_client.account_configuration import AccountConfiguration

from evaluation.eval_vs_foulplay import (
    _challenge_with_fallback,
    _read_user_id,
    _resolve_foulplay_identity,
    _safe_id,
    _spawn_foulplay,
    _tail_text,
    _terminate_proc,
    _wait_for_finished,
    _wait_for_foulplay_waiting,
)

CustomServerConfig = get_server_configuration(default_port=8000)


def _account_with_name(base: str) -> AccountConfiguration:
    safe = "".join(ch for ch in base.lower() if ch.isalnum()) or "bot"
    return AccountConfiguration.generate(safe, rand=True)


async def _forfeit_new_battles(agent, before_tags: set[str], timeout_s: int) -> bool:
    new_tags = [tag for tag in agent.battles.keys() if tag not in before_tags]
    if not new_tags:
        new_tags = [tag for tag, battle in agent.battles.items() if not battle.finished]
    if not new_tags:
        return False
    before_finished = agent.n_finished_battles
    for tag in new_tags:
        try:
            await agent.ps_client.send_message("/forfeit", room=tag)
        except Exception:
            pass
        await asyncio.sleep(0.2)
        try:
            await agent.ps_client.send_message(f"/leave {tag}")
        except Exception:
            pass
    target_finished = before_finished + len(new_tags)
    return await _wait_for_finished(agent, target_finished, timeout_s)


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--loops", type=int, default=600)
    parser.add_argument("--format", default="gen9randombattle")
    parser.add_argument("--ws-uri", default="ws://127.0.0.1:8000/showdown/websocket")
    parser.add_argument("--foulplay-path", default="third_party/foul-play")
    parser.add_argument("--foulplay-python", default="third_party/foul-play/.venv/bin/python")
    parser.add_argument("--foulplay-username", default="foulplay_bot_probe")
    parser.add_argument("--foulplay-password", default="foulplay_pass")
    parser.add_argument("--foulplay-search-ms", type=int, default=120)
    parser.add_argument("--foulplay-parallelism", type=int, default=1)
    parser.add_argument("--foulplay-wait", type=int, default=90)
    parser.add_argument("--forfeit-timeout", type=int, default=45)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--no-login", action="store_true")
    parser.add_argument("--log-prefix", default="")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    foulplay_path = Path(args.foulplay_path).expanduser()
    foulplay_python = Path(args.foulplay_python).expanduser()
    if not foulplay_path.exists():
        raise SystemExit(f"Foul Play repo not found: {foulplay_path}")
    if not foulplay_python.exists():
        raise SystemExit(f"Foul Play python not found: {foulplay_python}")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    suffix = f"{args.log_prefix}_{stamp}" if args.log_prefix else stamp
    base_log_path = (PROJECT_ROOT / f"logs/foulplay/current/challenge_probe_{suffix}.log").resolve()
    base_user_id_path = (PROJECT_ROOT / f"logs/foulplay/current/challenge_probe_user_{suffix}.txt").resolve()
    base_log_path.parent.mkdir(parents=True, exist_ok=True)
    base_user_id_path.parent.mkdir(parents=True, exist_ok=True)

    agent = RuleBotPlayer(
        battle_format=args.format,
        max_concurrent_battles=1,
        server_configuration=CustomServerConfig,
        account_configuration=_account_with_name("challengeprobe"),
    )

    retries_left = max(0, args.retries)
    waiting_count = 0
    failures = 0
    started = 0
    finished = 0

    proc, log_path, user_id_path, foulplay_id, foulplay_name = await _spawn_foulplay(
        attempt=0,
        remaining_battles=args.loops,
        foulplay_path=foulplay_path,
        foulplay_python=foulplay_python,
        foulplay_username=args.foulplay_username,
        foulplay_password=args.foulplay_password,
        ws_uri=args.ws_uri,
        battle_format=args.format,
        search_time_ms=args.foulplay_search_ms,
        parallelism=args.foulplay_parallelism,
        no_login=args.no_login,
        log_path=base_log_path,
        user_id_path=base_user_id_path,
        wait_s=args.foulplay_wait,
        bot_mode="accept_challenge",
        user_to_challenge=None,
        debug=args.debug,
    )
    foulplay_id, foulplay_name, foulplay_username = _resolve_foulplay_identity(
        foulplay_id, foulplay_name, args.foulplay_username
    )

    for i in range(args.loops):
        if proc.poll() is not None:
            if retries_left <= 0:
                print(f"[fail] foulplay process exited before battle {i + 1}")
                break
            retries_left -= 1
            _terminate_proc(proc)
            proc, log_path, user_id_path, foulplay_id, foulplay_name = await _spawn_foulplay(
                attempt=args.retries - retries_left,
                remaining_battles=args.loops - i,
                foulplay_path=foulplay_path,
                foulplay_python=foulplay_python,
                foulplay_username=foulplay_username,
                foulplay_password=args.foulplay_password,
                ws_uri=args.ws_uri,
                battle_format=args.format,
                search_time_ms=args.foulplay_search_ms,
                parallelism=args.foulplay_parallelism,
                no_login=args.no_login,
                log_path=base_log_path,
                user_id_path=base_user_id_path,
                wait_s=args.foulplay_wait,
                bot_mode="accept_challenge",
                user_to_challenge=None,
                debug=args.debug,
            )
            foulplay_id, foulplay_name, foulplay_username = _resolve_foulplay_identity(
                foulplay_id, foulplay_name, foulplay_username
            )
            waiting_count = 0

        next_count = await _wait_for_foulplay_waiting(
            log_path, min_count=waiting_count + 1, timeout_s=args.foulplay_wait
        )
        if next_count is None:
            failures += 1
            print(f"[fail] wait-ready timeout loop={i + 1} id={foulplay_id} name={foulplay_name}")
            print(_tail_text(log_path))
            if retries_left <= 0:
                break
            retries_left -= 1
            _terminate_proc(proc)
            proc, log_path, user_id_path, foulplay_id, foulplay_name = await _spawn_foulplay(
                attempt=args.retries - retries_left,
                remaining_battles=args.loops - i,
                foulplay_path=foulplay_path,
                foulplay_python=foulplay_python,
                foulplay_username=foulplay_username,
                foulplay_password=args.foulplay_password,
                ws_uri=args.ws_uri,
                battle_format=args.format,
                search_time_ms=args.foulplay_search_ms,
                parallelism=args.foulplay_parallelism,
                no_login=args.no_login,
                log_path=base_log_path,
                user_id_path=base_user_id_path,
                wait_s=args.foulplay_wait,
                bot_mode="accept_challenge",
                user_to_challenge=None,
                debug=args.debug,
            )
            foulplay_id, foulplay_name, foulplay_username = _resolve_foulplay_identity(
                foulplay_id, foulplay_name, foulplay_username
            )
            waiting_count = 0
            continue

        waiting_count = next_count
        fresh_id, fresh_name = await _read_user_id(user_id_path, timeout_s=1, debug=args.debug)
        foulplay_id, foulplay_name, foulplay_username = _resolve_foulplay_identity(
            fresh_id or foulplay_id, fresh_name or foulplay_name, foulplay_username
        )

        before_tags = set(agent.battles.keys())
        accepted = await _challenge_with_fallback(
            agent,
            opponent_id=foulplay_id,
            opponent_name=foulplay_name or foulplay_username,
            timeout_s=args.foulplay_wait,
            debug=args.debug,
        )
        if not accepted:
            failures += 1
            print(
                f"[fail] challenge not accepted loop={i + 1} id={foulplay_id} "
                f"name={foulplay_name or foulplay_username}"
            )
            print(_tail_text(log_path))
            if retries_left <= 0:
                break
            retries_left -= 1
            _terminate_proc(proc)
            proc, log_path, user_id_path, foulplay_id, foulplay_name = await _spawn_foulplay(
                attempt=args.retries - retries_left,
                remaining_battles=args.loops - i,
                foulplay_path=foulplay_path,
                foulplay_python=foulplay_python,
                foulplay_username=foulplay_username,
                foulplay_password=args.foulplay_password,
                ws_uri=args.ws_uri,
                battle_format=args.format,
                search_time_ms=args.foulplay_search_ms,
                parallelism=args.foulplay_parallelism,
                no_login=args.no_login,
                log_path=base_log_path,
                user_id_path=base_user_id_path,
                wait_s=args.foulplay_wait,
                bot_mode="accept_challenge",
                user_to_challenge=None,
                debug=args.debug,
            )
            foulplay_id, foulplay_name, foulplay_username = _resolve_foulplay_identity(
                foulplay_id, foulplay_name, foulplay_username
            )
            waiting_count = 0
            continue

        started += 1
        ok = await _forfeit_new_battles(agent, before_tags, timeout_s=args.forfeit_timeout)
        if ok:
            finished += 1
        else:
            failures += 1
            print(
                f"[fail] forfeit completion timeout loop={i + 1} "
                f"battles={len(agent.battles)} finished={agent.n_finished_battles}"
            )
        if (i + 1) % max(1, args.progress_every) == 0:
            print(
                f"[progress] {i + 1}/{args.loops} started={started} "
                f"forfeit-finished={finished} failures={failures} foulplay={foulplay_id}"
            )

    _terminate_proc(proc)
    print(
        f"[done] loops={args.loops} started={started} forfeit-finished={finished} "
        f"failures={failures} retries_left={retries_left} log={log_path}"
    )
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
