#!/usr/bin/env python3
"""
Offline evaluation against the Foul Play bot.

This launches Foul Play in accept_challenge mode and sends challenges
from our bot (rulebot/rl) via poke_env.
"""

import argparse
import asyncio
import os
import signal
import subprocess
import sys
import time
import math
from dataclasses import fields
from collections import Counter
from pathlib import Path
import json

import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from poke_env.ps_client.account_configuration import AccountConfiguration

from src.players.rule_bot import RuleBotPlayer
from src.players.Oranguru import OranguruPlayer
from src.players.oranguru_engine import OranguruEnginePlayer
from src.players.rl_player import RLPlayer
from src.models.actor_critic import ActorCritic, RecurrentActorCritic
from src.utils.server_config import get_server_configuration
from src.utils.features import load_moves
from src.utils.damage_calc import normalize_name
from training.config import RLConfig

CustomServerConfig = get_server_configuration(default_port=8000)


def _debug_print(enabled: bool, message: str) -> None:
    if not enabled:
        return
    stamp = time.strftime("%H:%M:%S")
    print(f"[debug {stamp}] {message}", flush=True)


def _safe_id(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum()) or "bot"


def _agent_user_target(agent) -> str:
    try:
        live_name = getattr(getattr(agent, "ps_client", None), "username", "") or ""
    except Exception:
        live_name = ""
    if not live_name:
        live_name = getattr(agent, "username", "")
    return _safe_id(live_name)


def _canonical_user_id(name: str | None) -> str:
    if not isinstance(name, str):
        return ""
    return "".join(ch for ch in name.lower() if ch.isalnum())


def _resolve_foulplay_identity(
    foulplay_id: str | None,
    foulplay_name: str | None,
    foulplay_username: str,
) -> tuple[str, str | None, str]:
    resolved_name = (
        foulplay_name.strip()
        if isinstance(foulplay_name, str) and foulplay_name.strip()
        else None
    )
    resolved_username = (
        foulplay_username.strip()
        if isinstance(foulplay_username, str) and foulplay_username.strip()
        else "foulplaybot"
    )
    resolved_id = (
        _canonical_user_id(foulplay_id)
        or _canonical_user_id(resolved_name)
        or _canonical_user_id(resolved_username)
    )
    if resolved_name:
        resolved_username = resolved_name
    return resolved_id, resolved_name, resolved_username


def _account_with_name(base: str) -> AccountConfiguration:
    safe = "".join(ch for ch in base.lower() if ch.isalnum()) or "bot"
    return AccountConfiguration.generate(safe, rand=True)


def _load_model(checkpoint_path: str, device: str):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    ckpt_config = checkpoint.get("config", {})
    cfg_fields = {f.name for f in fields(RLConfig)}
    filtered = {k: v for k, v in ckpt_config.items() if k in cfg_fields}
    config = RLConfig(**filtered) if filtered else RLConfig()
    d_model = config.d_model
    n_actions = config.n_actions
    rnn_hidden = config.rnn_hidden
    rnn_layers = config.rnn_layers
    model_type = checkpoint.get("model_type", "feedforward")

    if model_type == "recurrent":
        model = RecurrentActorCritic(
            feature_dim=config.feature_dim,
            d_model=d_model,
            n_actions=n_actions,
            rnn_hidden=rnn_hidden,
            rnn_layers=rnn_layers,
        ).to(device)
    else:
        model = ActorCritic(
            feature_dim=config.feature_dim,
            d_model=d_model,
            n_actions=n_actions,
        ).to(device)

    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    elif "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        raise ValueError("Checkpoint missing model weights.")

    model.eval()
    return model, config


def _apply_rl_eval_overrides(config: RLConfig, disable_biases: bool) -> None:
    if not disable_biases:
        return
    # Sequence/BC checkpoints often don't encode these fields; default RLConfig
    # switch/attack penalties can dominate policy behavior at inference time.
    config.switch_bias_enabled = False
    config.switch_stay_penalty_strength = 0.0
    config.attack_eff_penalty_enabled = False


def _percentile(values: list[int], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    idx = (len(ordered) - 1) * pct
    low = math.floor(idx)
    high = math.ceil(idx)
    if low == high:
        return float(ordered[low])
    weight = idx - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def _count_actions(battle) -> tuple[int, int, int]:
    role = battle.player_role
    if not role:
        return 0, 0, 0
    moves = 0
    switches = 0
    forced = 0
    for obs in battle.observations.values():
        for event in obs.events:
            if len(event) < 2:
                continue
            kind = event[1]
            if kind == "move" and len(event) > 2 and event[2].startswith(role):
                moves += 1
            elif kind in {"switch", "drag", "replace"} and len(event) > 2 and event[2].startswith(role):
                switches += 1
                if kind == "drag":
                    forced += 1
    return moves, switches, forced


def _extract_move_id(event: list[str]) -> str | None:
    if len(event) < 4:
        return None
    move_name = event[3]
    if not move_name:
        return None
    return normalize_name(move_name)


def _extract_switch_species(battle, event: list[str]) -> str | None:
    if len(event) < 3:
        return None
    try:
        mon = battle.get_pokemon(event[2])
        if mon and mon.species:
            return normalize_name(mon.species)
    except Exception:
        pass
    if ": " in event[2]:
        return normalize_name(event[2].split(": ", 1)[1])
    return normalize_name(event[2])


def _move_flags(move_id: str, moves_data: dict) -> dict:
    entry = moves_data.get(move_id, {})
    category = entry.get("category") or ""
    is_status = category.lower() == "status"
    boosts = entry.get("boosts") or {}
    self_boosts = (entry.get("self") or {}).get("boosts") or {}
    self_boost = entry.get("selfBoost") or {}
    is_setup = bool(boosts) or bool(self_boosts) or bool(self_boost)
    is_hazard = move_id in RuleBotPlayer.ENTRY_HAZARDS or entry.get("sideCondition") in RuleBotPlayer.ENTRY_HAZARDS
    is_anti_hazard = move_id in RuleBotPlayer.ANTI_HAZARDS_MOVES
    is_recovery = move_id in RuleBotPlayer.RECOVERY_MOVES or bool(entry.get("heal"))
    is_protect = move_id in RuleBotPlayer.PROTECT_MOVES
    is_pivot = move_id in RuleBotPlayer.PIVOT_MOVES
    priority = entry.get("priority")
    is_priority = bool(priority and priority > 0) or move_id in RuleBotPlayer.PRIORITY_MOVES
    return {
        "status": is_status,
        "setup": is_setup,
        "hazard": is_hazard,
        "anti_hazard": is_anti_hazard,
        "recovery": is_recovery,
        "protect": is_protect,
        "pivot": is_pivot,
        "priority": is_priority,
    }


def _start_foulplay(
    foulplay_path: Path,
    foulplay_python: Path,
    username: str,
    password: str,
    ws_uri: str,
    battle_format: str,
    n_battles: int,
    search_time_ms: int,
    parallelism: int,
    log_path: Path,
    no_login: bool,
    user_id_file: Path,
    bot_mode: str,
    user_to_challenge: str | None,
):
    probe = subprocess.run(
        [str(foulplay_python), "-c", "import poke_engine, sys; print(sys.executable)"],
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.returncode != 0:
        raise SystemExit(
            "Foul Play python cannot import poke_engine:\n"
            f"{probe.stderr or probe.stdout}"
        )

    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    venv_root = foulplay_python.parent.parent
    env["VIRTUAL_ENV"] = str(venv_root)
    env["PATH"] = str(foulplay_python.parent) + os.pathsep + env.get("PATH", "")

    cmd = [
        str(foulplay_python),
        "run.py",
        "--websocket-uri",
        ws_uri,
        "--ps-username",
        username,
        "--ps-password",
        password,
        "--bot-mode",
        bot_mode,
        "--pokemon-format",
        battle_format,
        "--run-count",
        str(n_battles),
        "--search-time-ms",
        str(search_time_ms),
        "--search-parallelism",
        str(parallelism),
        "--log-level",
        "INFO",
        "--user-id-file",
        str(user_id_file),
    ]
    if bot_mode == "challenge_user":
        if not user_to_challenge:
            raise SystemExit("Foul Play requires --user-to-challenge in challenge_user mode.")
        cmd.extend(["--user-to-challenge", user_to_challenge])
    if no_login:
        cmd.append("--no-login")

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("w", encoding="utf-8")
    return subprocess.Popen(
        cmd,
        cwd=foulplay_path,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        env=env,
    )


def _suffix_path(path: Path, suffix: str) -> Path:
    if not suffix:
        return path
    return path.with_name(f"{path.stem}_{suffix}{path.suffix}")


def _terminate_proc(proc: subprocess.Popen | None) -> None:
    if proc is None:
        return
    if proc.poll() is None:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def _format_eta(seconds: float) -> str:
    if seconds <= 0:
        return "0:00:00"
    seconds_int = int(round(seconds))
    hours = seconds_int // 3600
    minutes = (seconds_int % 3600) // 60
    secs = seconds_int % 60
    return f"{hours}:{minutes:02d}:{secs:02d}"


async def _wait_for_battles(player, n_battles: int, progress_every: int, timeout_s: int):
    start = time.time()
    last_print = 0
    while player.n_finished_battles < n_battles:
        await asyncio.sleep(0.5)
        if player.n_finished_battles - last_print >= progress_every:
            last_print = player.n_finished_battles
            elapsed = time.time() - start
            finished = max(1, player.n_finished_battles)
            avg_per_battle = elapsed / finished
            remaining = max(0, n_battles - player.n_finished_battles)
            eta = _format_eta(remaining * avg_per_battle)
            print(
                f"   {player.n_finished_battles}/{n_battles}: "
                f"{player.n_won_battles}/{player.n_finished_battles} wins "
                f"({(player.n_won_battles / max(player.n_finished_battles, 1)) * 100:.1f}%) "
                f"ETA {eta}"
            )
        if timeout_s and (time.time() - start) > timeout_s:
            break


async def _wait_for_finished(player, target_finished: int, timeout_s: int) -> bool:
    start = time.time()
    while player.n_finished_battles < target_finished:
        await asyncio.sleep(0.5)
        if timeout_s and (time.time() - start) > timeout_s:
            return False
    return True


async def _wait_for_foulplay_ready(log_path: Path, timeout_s: int = 30) -> None:
    start = time.time()
    seen = ""
    while time.time() - start < timeout_s:
        if log_path.exists():
            try:
                seen = log_path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                seen = ""
        if "Waiting for a" in seen or "Searching for ranked" in seen:
            return
        await asyncio.sleep(1)


async def _wait_for_foulplay_waiting(log_path: Path, min_count: int, timeout_s: int) -> int | None:
    start = time.time()
    while time.time() - start < timeout_s:
        if log_path.exists():
            try:
                content = log_path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                content = ""
            count = content.count("Waiting for a")
            if count >= min_count:
                return count
        await asyncio.sleep(1)
    return None


async def _wait_for_battle_start(player, previous_count: int, timeout_s: int) -> bool:
    start = time.time()
    while time.time() - start < timeout_s:
        if len(player.battles) > previous_count:
            return True
        await asyncio.sleep(0.5)
    return False


async def _forfeit_unfinished_battles(player, debug: bool = False) -> int:
    to_forfeit = []
    for battle in list(player.battles.values()):
        try:
            if battle is not None and not battle.finished:
                to_forfeit.append(battle.battle_tag)
        except Exception:
            continue
    if not to_forfeit:
        return 0
    for tag in to_forfeit:
        try:
            await player.ps_client.send_message("/forfeit", tag)
            _debug_print(debug, f"sent /forfeit to {tag}")
        except Exception as exc:
            _debug_print(debug, f"failed /forfeit for {tag}: {exc}")
    return len(to_forfeit)


async def _wait_for_login(ps_client, timeout_s: int) -> bool:
    print(f"   Waiting for login: {ps_client.username}")
    try:
        await asyncio.wait_for(ps_client.logged_in.wait(), timeout=timeout_s)
        print(f"   Logged in: {ps_client.username}")
        return True
    except asyncio.TimeoutError:
        print(f"   Login timed out: {ps_client.username}")
        return False


async def _cancel_challenge(
    ps_client,
    delay_s: float = 0.5,
    debug: bool = False,
    attempts: int = 1,
    target: str | None = None,
) -> None:
    try:
        cmd = "/cancelchallenge"
        for idx in range(max(1, attempts)):
            await ps_client.send_message(cmd)
            _debug_print(
                debug,
                f"sent {cmd} ({idx + 1}/{attempts})",
            )
            if delay_s:
                await asyncio.sleep(delay_s)
    except Exception:
        _debug_print(debug, "failed to send /cancelchallenge")


async def _challenge_with_fallback(
    player,
    opponent_id: str,
    opponent_name: str | None,
    timeout_s: int,
    debug: bool = False,
) -> str | None:
    candidates = []
    for raw in (opponent_id, opponent_name):
        candidate = _canonical_user_id(raw)
        if candidate and candidate not in candidates:
            candidates.append(candidate)

    if not candidates:
        _debug_print(debug, "no valid challenge candidate id available")
        return None

    if not await _wait_for_login(player.ps_client, timeout_s):
        return None
    for candidate in candidates:
        await _cancel_challenge(
            player.ps_client,
            delay_s=0.3,
            debug=debug,
            attempts=1,
            target=candidate,
        )
    for candidate in candidates:
        before = len(player.battles)
        print(f"   Trying challenge target: {candidate}")
        await _cancel_challenge(
            player.ps_client,
            delay_s=0.4,
            debug=debug,
            attempts=2,
            target=candidate,
        )
        await player.ps_client.challenge(candidate, player._format, player.next_team)
        if await _wait_for_battle_start(player, before, timeout_s):
            print(f"   Battle started with: {candidate}")
            return candidate
        print(f"   No battle start for: {candidate}")
        await _cancel_challenge(
            player.ps_client,
            delay_s=1.0,
            debug=debug,
            attempts=3,
            target=candidate,
        )
    return None


async def _read_user_id(
    user_id_file: Path, timeout_s: int = 30, debug: bool = False
) -> tuple[str | None, str | None]:
    start = time.time()
    while time.time() - start < timeout_s:
        if user_id_file.exists():
            try:
                content = user_id_file.read_text(encoding="utf-8", errors="ignore").strip()
                if not content:
                    await asyncio.sleep(1)
                    continue
                if content.startswith("{"):
                    payload = json.loads(content)
                    userid = payload.get("userid")
                    username = payload.get("username")
                    userid = _canonical_user_id(userid)
                    if not userid:
                        userid = None
                    if isinstance(username, str):
                        username = username.strip() or None
                    else:
                        username = None
                    _debug_print(debug, f"user-id file parsed: userid={userid} username={username}")
                    return userid, username
                raw_id = _canonical_user_id(content)
                if raw_id:
                    _debug_print(debug, f"user-id file raw: {content}")
                    return raw_id, None
            except Exception:
                pass
        await asyncio.sleep(1)
    return None, None


async def _spawn_foulplay(
    *,
    attempt: int,
    remaining_battles: int,
    foulplay_path: Path,
    foulplay_python: Path,
    foulplay_username: str,
    foulplay_password: str,
    ws_uri: str,
    battle_format: str,
    search_time_ms: int,
    parallelism: int,
    no_login: bool,
    log_path: Path,
    user_id_path: Path,
    wait_s: int,
    bot_mode: str,
    user_to_challenge: str | None,
    debug: bool,
) -> tuple[subprocess.Popen, Path, Path, str, str | None]:
    attempt_suffix = f"retry{attempt}" if attempt else ""
    attempt_log = _suffix_path(log_path, attempt_suffix)
    attempt_user_id = _suffix_path(user_id_path, attempt_suffix)
    attempt_user_id.parent.mkdir(parents=True, exist_ok=True)
    try:
        attempt_user_id.unlink()
    except FileNotFoundError:
        pass
    _debug_print(
        debug,
        "spawning foulplay mode={} log={} user_id_file={}".format(
            bot_mode, attempt_log, attempt_user_id
        ),
    )
    proc = _start_foulplay(
        foulplay_path=foulplay_path,
        foulplay_python=foulplay_python,
        username=foulplay_username,
        password=foulplay_password,
        ws_uri=ws_uri,
        battle_format=battle_format,
        n_battles=remaining_battles,
        search_time_ms=search_time_ms,
        parallelism=parallelism,
        log_path=attempt_log,
        no_login=no_login,
        user_id_file=attempt_user_id,
        bot_mode=bot_mode,
        user_to_challenge=user_to_challenge,
    )
    foulplay_id, foulplay_name = await _read_user_id(
        attempt_user_id, timeout_s=wait_s, debug=debug
    )
    return proc, attempt_log, attempt_user_id, foulplay_id or "", foulplay_name


def _tail_text(path: Path, max_chars: int = 2000) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def _git_commit_snapshot(project_root: Path) -> tuple[str, bool]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=project_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        commit = "unknown"
    try:
        dirty_raw = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=project_root,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        dirty = bool(dirty_raw.strip())
    except Exception:
        dirty = False
    return commit, dirty


def _print_startup_context(args: argparse.Namespace) -> None:
    commit, dirty = _git_commit_snapshot(PROJECT_ROOT)
    print("Eval Context")
    print(f"   Commit: {commit} {'(dirty)' if dirty else '(clean)'}")
    print(f"   Player: {args.player}")
    print(f"   Format: {args.format}")
    print(f"   Battles: {args.battles}")
    env_items = sorted(
        (key, value)
        for key, value in os.environ.items()
        if key.startswith("ORANGURU_")
    )
    if env_items:
        print("   ORANGURU_* env:")
        for key, value in env_items:
            print(f"     {key}={value}")
    else:
        print("   ORANGURU_* env: <none>")
    print()


async def main():
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    parser = argparse.ArgumentParser()
    parser.add_argument("--battles", type=int, default=50)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=0, help="Total timeout (seconds); 0 = no timeout")
    parser.add_argument("--battle-timeout", type=int, default=240, help="Per-battle timeout (seconds)")
    parser.add_argument("--format", default="gen9randombattle")
    parser.add_argument("--player", choices=["rulebot", "rl", "oranguru", "oranguru_engine"], default="rulebot")
    parser.add_argument("--checkpoint", default="checkpoints/rl/best_overall.pt")
    parser.add_argument("--foulplay-path", default="third_party/foul-play")
    parser.add_argument("--foulplay-username", default="foulplay_bot")
    parser.add_argument("--foulplay-password", default="foulplay_pass")
    parser.add_argument("--foulplay-python", default=sys.executable)
    parser.add_argument("--foulplay-search-ms", type=int, default=120)
    parser.add_argument("--foulplay-parallelism", type=int, default=1)
    parser.add_argument("--foulplay-log", default=None)
    parser.add_argument("--foulplay-wait", type=int, default=30)
    parser.add_argument("--foulplay-user-id-file", default=None)
    parser.add_argument("--foulplay-retries", type=int, default=2)
    parser.add_argument(
        "--foulplay-auto-challenge",
        dest="foulplay_auto_challenge",
        action="store_true",
        help="Auto-switch to foulplay-challenges mode after repeated challenge failures (off by default).",
    )
    parser.add_argument(
        "--no-foulplay-auto-challenge",
        dest="foulplay_auto_challenge",
        action="store_false",
        help="Disable auto-switch to foulplay-challenges mode.",
    )
    parser.set_defaults(foulplay_auto_challenge=False)
    parser.add_argument(
        "--foulplay-auto-challenge-after",
        type=int,
        default=3,
        help="Consecutive challenge failures before switching to foulplay-challenges.",
    )
    parser.add_argument(
        "--challenge-soft-retries",
        type=int,
        default=12,
        help="Challenge failures to tolerate per battle before consuming a hard foulplay retry.",
    )
    parser.add_argument(
        "--challenge-backoff",
        type=float,
        default=1.5,
        help="Seconds to wait between repeated challenge attempts for the same battle index.",
    )
    parser.add_argument(
        "--foulplay-restart-every",
        type=int,
        default=0,
        help="Restart Foul Play every N battle indices to reduce long-run drift (0 disables).",
    )
    parser.add_argument(
        "--foulplay-challenges",
        action="store_true",
        help="Have Foul Play challenge our agent instead of the other way around.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose debug logging for eval orchestration.",
    )
    parser.add_argument("--ws-uri", default="ws://localhost:8000/showdown/websocket")
    parser.add_argument("--no-login", action="store_true", help="Use guest /trn for local servers")
    parser.add_argument(
        "--rl-disable-biases",
        action="store_true",
        default=True,
        help="Disable RLPlayer switch/attack heuristic logit biases (default: on for eval stability).",
    )
    parser.add_argument(
        "--no-rl-disable-biases",
        action="store_true",
        help="Keep RLPlayer heuristic biases enabled.",
    )
    args = parser.parse_args()
    if args.no_rl_disable_biases:
        args.rl_disable_biases = False

    foulplay_path = Path(args.foulplay_path)
    if not foulplay_path.exists():
        raise SystemExit(f"Foul Play repo not found: {foulplay_path}")
    foulplay_python = Path(args.foulplay_python).expanduser()
    if not foulplay_python.exists():
        raise SystemExit(f"Foul Play python not found: {foulplay_python}")

    log_path = Path(args.foulplay_log) if args.foulplay_log else PROJECT_ROOT / Path(
        f"logs/foulplay/eval_{time.strftime('%Y%m%d_%H%M%S')}.log"
    )
    user_id_path = (
        Path(args.foulplay_user_id_file)
        if args.foulplay_user_id_file
        else PROJECT_ROOT / Path(f"logs/foulplay/user_{time.strftime('%Y%m%d_%H%M%S')}.txt")
    )
    log_path = log_path.expanduser().resolve()
    user_id_path = user_id_path.expanduser().resolve()
    user_id_path.parent.mkdir(parents=True, exist_ok=True)
    base_log_path = log_path
    base_user_id_path = user_id_path
    # If caller passes a shell-substituted username, it might include spaces.
    foulplay_username = args.foulplay_username.strip()

    kwargs = {
        "battle_format": args.format,
        "max_concurrent_battles": 1,
        "server_configuration": CustomServerConfig,
        "account_configuration": _account_with_name(args.player),
        "start_timer_on_battle_start": True,
    }
    # Make eval context explicit so engine-side eval-only switches are reliable.
    os.environ["ORANGURU_EVAL_MODE"] = "1"
    _print_startup_context(args)

    if args.player == "rulebot":
        agent = RuleBotPlayer(**kwargs)
    elif args.player == "oranguru":
        agent = OranguruPlayer(**kwargs)
    elif args.player == "oranguru_engine":
        agent = OranguruEnginePlayer(**kwargs)
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, config = _load_model(args.checkpoint, device)
        _apply_rl_eval_overrides(config, args.rl_disable_biases)
        agent = RLPlayer(model=model, config=config, device=device, training=False, **kwargs)

    bot_mode = "challenge_user" if args.foulplay_challenges else "accept_challenge"
    user_to_challenge = None
    if args.foulplay_challenges:
        login_ok = await _wait_for_login(
            agent.ps_client,
            timeout_s=max(10, args.foulplay_wait),
        )
        user_to_challenge = _agent_user_target(agent)
        if not login_ok:
            _debug_print(
                args.debug,
                "agent login not confirmed before spawning challenge_user foulplay; proceeding with fallback target",
            )
        print(f"   Challenge target: {user_to_challenge}")

    _debug_print(
        args.debug,
        "agent user={} ws={} foulplay_user={} mode={}".format(
            _safe_id(agent.username),
            args.ws_uri,
            foulplay_username,
            "challenge_user" if args.foulplay_challenges else "accept_challenge",
        ),
    )
    proc, log_path, user_id_path, foulplay_id, foulplay_name = await _spawn_foulplay(
        attempt=0,
        remaining_battles=args.battles,
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
        bot_mode=bot_mode,
        user_to_challenge=user_to_challenge,
        debug=args.debug,
    )
    foulplay_id, foulplay_name, foulplay_username = _resolve_foulplay_identity(
        foulplay_id,
        foulplay_name,
        foulplay_username,
    )
    print(
        "   Foul Play user-id: {} name: {} file: {}".format(
            foulplay_id or "<empty>",
            foulplay_name or "<none>",
            user_id_path,
        )
    )

    print(f"🔁 Challenging Foul Play ({foulplay_username}) for {args.battles} battles...")
    print(f"   Agent username: {agent.username}")
    print(f"   Server: {CustomServerConfig.websocket_url}")
    waiting_count = 0
    retries_left = max(0, args.foulplay_retries)
    challenge_failures = 0
    auto_challenge_after = max(1, args.foulplay_auto_challenge_after)
    forced_restart_idx = -1
    abort = False
    accept_task: asyncio.Task | None = None
    if args.foulplay_challenges:
        accept_task = asyncio.create_task(
            agent.accept_challenges(opponent=None, n_challenges=args.battles)
        )
    for i in range(args.battles):
        refresh_attempts = 0
        max_refresh_attempts = 2
        if args.foulplay_challenges:
            refreshed_target = _agent_user_target(agent)
            if refreshed_target and refreshed_target != user_to_challenge:
                _debug_print(
                    args.debug,
                    f"updated challenge target {user_to_challenge} -> {refreshed_target}",
                )
                user_to_challenge = refreshed_target
        if (
            not args.foulplay_challenges
            and args.foulplay_restart_every > 0
            and i > 0
            and (i % args.foulplay_restart_every) == 0
            and forced_restart_idx != i
        ):
            _debug_print(
                args.debug,
                f"proactive foulplay restart at battle index {i + 1}",
            )
            _terminate_proc(proc)
            remaining = args.battles - i
            proc, log_path, user_id_path, foulplay_id, foulplay_name = await _spawn_foulplay(
                attempt=args.foulplay_retries - retries_left,
                remaining_battles=remaining,
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
                bot_mode=bot_mode,
                user_to_challenge=user_to_challenge,
                debug=args.debug,
            )
            foulplay_id, foulplay_name, foulplay_username = _resolve_foulplay_identity(
                foulplay_id,
                foulplay_name,
                foulplay_username,
            )
            waiting_count = 0
            challenge_failures = 0
            forced_restart_idx = i
        if args.foulplay_challenges:
            while True:
                _debug_print(args.debug, f"waiting for challenge {i + 1}/{args.battles}")
                if proc.poll() is not None:
                    if retries_left <= 0:
                        abort = True
                        break
                    retries_left -= 1
                    _terminate_proc(proc)
                    remaining = args.battles - i
                    proc, log_path, user_id_path, foulplay_id, foulplay_name = await _spawn_foulplay(
                        attempt=args.foulplay_retries - retries_left,
                        remaining_battles=remaining,
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
                        bot_mode=bot_mode,
                        user_to_challenge=user_to_challenge,
                        debug=args.debug,
                    )
                    foulplay_id, foulplay_name, foulplay_username = _resolve_foulplay_identity(
                        foulplay_id,
                        foulplay_name,
                        foulplay_username,
                    )
                    waiting_count = 0
                    continue

                if accept_task and accept_task.done():
                    try:
                        await accept_task
                    except Exception as exc:
                        _debug_print(args.debug, f"accept_challenges task failed: {exc}")
                    if agent.n_finished_battles < (i + 1):
                        abort = True
                        break

                ok = await _wait_for_finished(
                    agent,
                    target_finished=i + 1,
                    timeout_s=max(args.battle_timeout, args.foulplay_wait),
                )
                if ok:
                    break

                if args.debug:
                    tail = _tail_text(log_path)
                    _debug_print(args.debug, f"timeout waiting; foulplay log tail:\n{tail}")
                forfeits = await _forfeit_unfinished_battles(agent, debug=args.debug)
                if forfeits:
                    await asyncio.sleep(2.0)
                if retries_left <= 0:
                    print(f"   Timeout waiting for battle {i + 1}/{args.battles}")
                    abort = True
                    break
                retries_left -= 1
                _terminate_proc(proc)
                remaining = args.battles - i
                proc, log_path, user_id_path, foulplay_id, foulplay_name = await _spawn_foulplay(
                    attempt=args.foulplay_retries - retries_left,
                    remaining_battles=remaining,
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
                    bot_mode=bot_mode,
                    user_to_challenge=user_to_challenge,
                    debug=args.debug,
                )
                foulplay_id, foulplay_name, foulplay_username = _resolve_foulplay_identity(
                    foulplay_id,
                    foulplay_name,
                    foulplay_username,
                )
                waiting_count = 0

            if abort:
                break
            if (i + 1) % args.progress_every == 0:
                wins = agent.n_won_battles
                finished = agent.n_finished_battles
                print(
                    f"   {i+1}/{args.battles}: {wins}/{finished} wins "
                    f"({wins/max(finished,1)*100:.1f}%)"
                )
            continue
        while True:
            if proc.poll() is not None:
                if retries_left <= 0:
                    abort = True
                    break
                retries_left -= 1
                _terminate_proc(proc)
                remaining = args.battles - i
                proc, log_path, user_id_path, foulplay_id, foulplay_name = await _spawn_foulplay(
                    attempt=args.foulplay_retries - retries_left,
                    remaining_battles=remaining,
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
                    bot_mode=bot_mode,
                    user_to_challenge=user_to_challenge,
                    debug=args.debug,
                )
                foulplay_id, foulplay_name, foulplay_username = _resolve_foulplay_identity(
                    foulplay_id,
                    foulplay_name,
                    foulplay_username,
                )
                waiting_count = 0

            print(f"   Waiting for Foul Play to be ready (battle {i + 1}/{args.battles})...")
            next_count = await _wait_for_foulplay_waiting(
                log_path, min_count=waiting_count + 1, timeout_s=args.foulplay_wait
            )
            if next_count is None:
                if args.debug:
                    tail = _tail_text(log_path)
                    _debug_print(args.debug, f"foulplay wait timeout; log tail:\n{tail}")
                if retries_left <= 0:
                    print(f"   Foul Play not ready for battle {i + 1}/{args.battles}")
                    abort = True
                    break
                retries_left -= 1
                _terminate_proc(proc)
                remaining = args.battles - i
                proc, log_path, user_id_path, foulplay_id, foulplay_name = await _spawn_foulplay(
                    attempt=args.foulplay_retries - retries_left,
                    remaining_battles=remaining,
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
                    bot_mode=bot_mode,
                    user_to_challenge=user_to_challenge,
                    debug=args.debug,
                )
                foulplay_id, foulplay_name, foulplay_username = _resolve_foulplay_identity(
                    foulplay_id,
                    foulplay_name,
                    foulplay_username,
                )
                waiting_count = 0
                continue
            waiting_count = next_count

            print(
                f"   Sending challenge to {foulplay_name or foulplay_username} "
                f"(battle {i + 1}/{args.battles})..."
            )
            prev_id = foulplay_id
            prev_name = foulplay_name or foulplay_username
            fresh_id, fresh_name = await _read_user_id(
                user_id_path, timeout_s=1, debug=args.debug
            )
            if fresh_id:
                foulplay_id = fresh_id
            if fresh_name:
                foulplay_name = fresh_name
                foulplay_username = fresh_name
            if (fresh_id and fresh_id != prev_id) or (fresh_name and fresh_name != prev_name):
                _debug_print(
                    args.debug,
                    "foulplay user changed; waiting briefly before challenge",
                )
                await asyncio.sleep(2)
            _debug_print(
                args.debug,
                "challenge candidates id={} name={}".format(
                    foulplay_id or "<empty>", foulplay_name or foulplay_username
                ),
            )
            accepted_by = await _challenge_with_fallback(
                agent,
                opponent_id=foulplay_id,
                opponent_name=foulplay_name or foulplay_username,
                timeout_s=args.foulplay_wait,
                debug=args.debug,
            )
            if not accepted_by:
                challenge_failures += 1
                if challenge_failures <= max(0, args.challenge_soft_retries):
                    _debug_print(
                        args.debug,
                        "challenge soft retry {}/{} for battle {}/{}".format(
                            challenge_failures,
                            max(0, args.challenge_soft_retries),
                            i + 1,
                            args.battles,
                        ),
                    )
                    await _cancel_challenge(
                        agent.ps_client,
                        delay_s=0.7,
                        debug=args.debug,
                        attempts=2,
                        target=foulplay_id,
                    )
                    await asyncio.sleep(max(0.0, args.challenge_backoff))
                    continue
                if (
                    args.foulplay_auto_challenge
                    and not args.foulplay_challenges
                    and challenge_failures >= auto_challenge_after
                ):
                    print(
                        f"   Switching to Foul Play challenge mode after {challenge_failures} failures"
                    )
                    args.foulplay_challenges = True
                    bot_mode = "challenge_user"
                    user_to_challenge = _agent_user_target(agent)
                    _terminate_proc(proc)
                    remaining = args.battles - i
                    proc, log_path, user_id_path, foulplay_id, foulplay_name = await _spawn_foulplay(
                        attempt=args.foulplay_retries - retries_left,
                        remaining_battles=remaining,
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
                        bot_mode=bot_mode,
                        user_to_challenge=user_to_challenge,
                        debug=args.debug,
                    )
                    foulplay_id, foulplay_name, foulplay_username = _resolve_foulplay_identity(
                        foulplay_id,
                        foulplay_name,
                        foulplay_username,
                    )
                    waiting_count = 0
                    challenge_failures = 0
                    try:
                        await asyncio.wait_for(
                            agent.accept_challenges(opponent=None, n_challenges=1),
                            timeout=args.battle_timeout,
                        )
                        _debug_print(
                            args.debug,
                            "accepted challenge after fallback; battles={} finished={}".format(
                                len(agent.battles), agent.n_finished_battles
                            ),
                        )
                    except asyncio.TimeoutError:
                        if args.debug:
                            tail = _tail_text(log_path)
                            _debug_print(
                                args.debug,
                                f"timeout waiting after fallback; foulplay log tail:\n{tail}",
                            )
                        if retries_left <= 0:
                            print(f"   Timeout waiting for battle {i + 1}/{args.battles}")
                            abort = True
                            break
                        retries_left -= 1
                        _terminate_proc(proc)
                        remaining = args.battles - i
                        proc, log_path, user_id_path, foulplay_id, foulplay_name = await _spawn_foulplay(
                            attempt=args.foulplay_retries - retries_left,
                            remaining_battles=remaining,
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
                            bot_mode=bot_mode,
                            user_to_challenge=user_to_challenge,
                            debug=args.debug,
                        )
                        foulplay_id, foulplay_name, foulplay_username = _resolve_foulplay_identity(
                            foulplay_id,
                            foulplay_name,
                            foulplay_username,
                        )
                        waiting_count = 0
                    if (i + 1) % args.progress_every == 0:
                        wins = agent.n_won_battles
                        finished = agent.n_finished_battles
                        print(
                            f"   {i+1}/{args.battles}: {wins}/{finished} wins "
                            f"({wins/max(finished,1)*100:.1f}%)"
                        )
                    break
                refresh_attempts += 1
                fresh_id, fresh_name = await _read_user_id(
                    user_id_path, timeout_s=2, debug=args.debug
                )
                if (
                    refresh_attempts <= max_refresh_attempts
                    and (
                        (fresh_id and fresh_id != foulplay_id)
                        or (fresh_name and fresh_name != (foulplay_name or foulplay_username))
                    )
                ):
                    if fresh_id:
                        foulplay_id = fresh_id
                    if fresh_name:
                        foulplay_name = fresh_name
                        foulplay_username = fresh_name
                    _debug_print(args.debug, "retrying challenge after foulplay id update")
                    continue
                if args.debug:
                    tail = _tail_text(log_path)
                    _debug_print(args.debug, f"challenge not accepted; foulplay log tail:\n{tail}")
                if retries_left <= 0:
                    print(f"   Challenge not accepted for battle {i + 1}/{args.battles}")
                    abort = True
                    break
                retries_left -= 1
                _terminate_proc(proc)
                remaining = args.battles - i
                proc, log_path, user_id_path, foulplay_id, foulplay_name = await _spawn_foulplay(
                    attempt=args.foulplay_retries - retries_left,
                    remaining_battles=remaining,
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
                    bot_mode=bot_mode,
                    user_to_challenge=user_to_challenge,
                    debug=args.debug,
                )
                foulplay_id, foulplay_name, foulplay_username = _resolve_foulplay_identity(
                    foulplay_id,
                    foulplay_name,
                    foulplay_username,
                )
                waiting_count = 0
                continue
            challenge_failures = 0

            ok = await _wait_for_finished(agent, agent.n_finished_battles + 1, args.battle_timeout)
            if not ok:
                if retries_left <= 0:
                    print(f"   Timeout waiting for battle {i + 1}/{args.battles}")
                    abort = True
                    break
                retries_left -= 1
                _terminate_proc(proc)
                remaining = args.battles - i
                proc, log_path, user_id_path, foulplay_id, foulplay_name = await _spawn_foulplay(
                    attempt=args.foulplay_retries - retries_left,
                    remaining_battles=remaining,
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
                    bot_mode=bot_mode,
                    user_to_challenge=user_to_challenge,
                    debug=args.debug,
                )
                foulplay_id, foulplay_name, foulplay_username = _resolve_foulplay_identity(
                    foulplay_id,
                    foulplay_name,
                    foulplay_username,
                )
                waiting_count = 0
                continue

            if (i + 1) % args.progress_every == 0:
                wins = agent.n_won_battles
                finished = agent.n_finished_battles
                print(f"   {i+1}/{args.battles}: {wins}/{finished} wins ({wins/max(finished,1)*100:.1f}%)")
            break
        if abort:
            break

    wins = agent.n_won_battles
    losses = agent.n_lost_battles
    finished = agent.n_finished_battles
    ties = finished - wins - losses
    win_rate = wins / finished if finished else 0.0

    finished_battles = [b for b in agent.battles.values() if b.finished]
    moves_data = load_moves()
    move_counter = Counter()
    switch_counter = Counter()
    turns = [b.turn for b in finished_battles if b.turn]
    remaining = [
        sum(1 for m in b.team.values() if m is not None and not m.fainted)
        for b in finished_battles
    ]
    opp_remaining = [
        sum(1 for m in b.opponent_team.values() if m is not None and not m.fainted)
        for b in finished_battles
    ]
    win_turns = [b.turn for b in finished_battles if b.won and b.turn]
    loss_turns = [b.turn for b in finished_battles if b.lost and b.turn]
    win_remaining = [
        sum(1 for m in b.team.values() if m is not None and not m.fainted)
        for b in finished_battles
        if b.won
    ]
    loss_remaining = [
        sum(1 for m in b.team.values() if m is not None and not m.fainted)
        for b in finished_battles
        if b.lost
    ]
    move_count = 0
    switch_count = 0
    forced_switches = 0
    status_moves = 0
    setup_moves = 0
    hazard_moves = 0
    anti_hazard_moves = 0
    recovery_moves = 0
    protect_moves = 0
    pivot_moves = 0
    priority_moves = 0
    attack_moves = 0
    tera_used = 0
    opp_tera_used = 0
    for battle in finished_battles:
        moves, switches, forced = _count_actions(battle)
        move_count += moves
        switch_count += switches
        forced_switches += forced
        role = battle.player_role
        if role:
            for obs in battle.observations.values():
                for event in obs.events:
                    if len(event) < 2:
                        continue
                    kind = event[1]
                    if kind == "move" and len(event) > 2 and event[2].startswith(role):
                        move_id = _extract_move_id(event)
                        if not move_id:
                            continue
                        move_counter[move_id] += 1
                        flags = _move_flags(move_id, moves_data)
                        if flags["status"]:
                            status_moves += 1
                        else:
                            attack_moves += 1
                        if flags["setup"]:
                            setup_moves += 1
                        if flags["hazard"]:
                            hazard_moves += 1
                        if flags["anti_hazard"]:
                            anti_hazard_moves += 1
                        if flags["recovery"]:
                            recovery_moves += 1
                        if flags["protect"]:
                            protect_moves += 1
                        if flags["pivot"]:
                            pivot_moves += 1
                        if flags["priority"]:
                            priority_moves += 1
                    elif kind in {"switch", "drag", "replace"} and len(event) > 2 and event[2].startswith(role):
                        species = _extract_switch_species(battle, event)
                        if species:
                            switch_counter[species] += 1
        if getattr(battle, "used_tera", False):
            tera_used += 1
        if getattr(battle, "opponent_used_tera", False):
            opp_tera_used += 1
    total_actions = move_count + switch_count

    print("\n" + "=" * 60)
    print("📈 FOUL PLAY EVAL SUMMARY")
    print("=" * 60)
    print(f"   Battles:        {finished}/{args.battles}")
    print(f"   W/L/T:          {wins}/{losses}/{ties}")
    print(f"   Win rate:       {win_rate:.1%}")
    if turns:
        print(f"   Avg turns:      {sum(turns) / len(turns):.1f}")
        median_turns = _percentile(turns, 0.5)
        p10 = _percentile(turns, 0.1)
        p90 = _percentile(turns, 0.9)
        if median_turns is not None and p10 is not None and p90 is not None:
            print(f"   Turn p10/p50/p90: {p10:.0f}/{median_turns:.0f}/{p90:.0f}")
    if win_turns and loss_turns:
        print(f"   Avg turns W/L: {sum(win_turns)/len(win_turns):.1f}/{sum(loss_turns)/len(loss_turns):.1f}")
    if remaining:
        print(f"   Avg remaining: {sum(remaining) / len(remaining):.2f}")
    if win_remaining and loss_remaining:
        print(f"   Avg remaining W/L: {sum(win_remaining)/len(win_remaining):.2f}/{sum(loss_remaining)/len(loss_remaining):.2f}")
    if opp_remaining:
        print(f"   Avg opp left:  {sum(opp_remaining) / len(opp_remaining):.2f}")
    if total_actions:
        print(f"   Move rate:     {move_count / total_actions:.1%}")
        print(f"   Switch rate:   {switch_count / total_actions:.1%}")
    if switch_count:
        print(f"   Forced switch: {forced_switches / switch_count:.1%}")
    if finished:
        print(f"   Tera used:     {tera_used}/{finished}")
        print(f"   Opp Tera used: {opp_tera_used}/{finished}")
    print(f"   Foul Play log:  {log_path}")

    mcts_stats = None
    if hasattr(agent, "get_mcts_stats"):
        try:
            mcts_stats = agent.get_mcts_stats()
        except Exception:
            mcts_stats = None
    if isinstance(mcts_stats, dict):
        print("\n📌 MCTS Diagnostics")
        print(
            "   Calls/states:  {}/{}".format(
                int(mcts_stats.get("calls", 0)),
                int(mcts_stats.get("states_sampled", 0)),
            )
        )
        print(
            "   Empty results: {} ({:.1%})".format(
                int(mcts_stats.get("empty_results", 0)),
                float(mcts_stats.get("empty_results_rate", 0.0)),
            )
        )
        print(
            "   State fails:   {} ({:.1%})".format(
                int(mcts_stats.get("result_none", 0)) + int(mcts_stats.get("result_errors", 0)),
                float(mcts_stats.get("state_failure_rate", 0.0)),
            )
        )
        print(
            "   Deterministic: {} | Stochastic: {}".format(
                int(mcts_stats.get("deterministic_decisions", 0)),
                int(mcts_stats.get("stochastic_decisions", 0)),
            )
        )
        print(
            "   Fallback(super/random): {}/{} ({:.1%}/{:.1%})".format(
                int(mcts_stats.get("fallback_super", 0)),
                int(mcts_stats.get("fallback_random", 0)),
                float(mcts_stats.get("fallback_super_rate", 0.0)),
                float(mcts_stats.get("fallback_random_rate", 0.0)),
            )
        )
        print(
            "   Avg worlds req/budget/gen/searched: {:.2f}/{:.2f}/{:.2f}/{:.2f}".format(
                float(mcts_stats.get("avg_requested_worlds_per_call", 0.0)),
                float(mcts_stats.get("avg_budgeted_worlds_per_call", 0.0)),
                float(mcts_stats.get("avg_generated_worlds_per_call", 0.0)),
                float(mcts_stats.get("avg_kept_worlds_per_call", 0.0)),
            )
        )
        print(
            "   World keep rate: {:.1%} | Low-unc turns/saved: {}/{} | Endgame turns/saved: {}/{}".format(
                float(mcts_stats.get("world_keep_rate", 0.0)),
                int(mcts_stats.get("low_uncertainty_turns", 0)),
                int(mcts_stats.get("low_uncertainty_worlds_saved", 0)),
                int(mcts_stats.get("endgame_reduction_turns", 0)),
                int(mcts_stats.get("endgame_worlds_saved", 0)),
            )
        )
        adaptive_triggered = int(mcts_stats.get("adaptive_triggered", 0))
        adaptive_rerank = int(mcts_stats.get("adaptive_rerank_used", 0))
        adaptive_second_pass = int(mcts_stats.get("adaptive_second_pass_used", 0))
        search_prior_used = int(mcts_stats.get("search_prior_used", 0))
        search_prior_calls = int(mcts_stats.get("search_prior_calls", 0))
        search_prior_mask_empty = int(mcts_stats.get("search_prior_mask_empty", 0))
        search_prior_unmapped = int(mcts_stats.get("search_prior_unmapped_choices", 0))
        search_prior_zero_sum = int(mcts_stats.get("search_prior_zero_sum", 0))
        search_prior_init_failed = int(mcts_stats.get("search_prior_init_failed", 0))
        search_prior_apply_failed = int(mcts_stats.get("search_prior_apply_failed", 0))
        switch_prior_used = int(mcts_stats.get("switch_prior_used", 0))
        switch_prior_pruned = int(mcts_stats.get("switch_prior_pruned", 0))
        passive_breaker_used = int(mcts_stats.get("passive_breaker_used", 0))
        passive_breaker_calls = int(mcts_stats.get("passive_breaker_calls", 0))
        passive_breaker_lowconf_skip = int(mcts_stats.get("passive_breaker_lowconf_skip", 0))
        passive_breaker_nonpassive_top = int(mcts_stats.get("passive_breaker_nonpassive_top", 0))
        passive_breaker_no_priors = int(mcts_stats.get("passive_breaker_no_priors", 0))
        passive_breaker_no_candidate = int(mcts_stats.get("passive_breaker_no_candidate", 0))
        passive_breaker_min_prob = int(mcts_stats.get("passive_breaker_min_prob", 0))
        passive_breaker_min_margin = int(mcts_stats.get("passive_breaker_min_margin", 0))
        tera_pruner_used = int(mcts_stats.get("tera_pruner_used", 0))
        tera_pruner_pruned = int(mcts_stats.get("tera_pruner_pruned", 0))
        world_ranker_used = int(mcts_stats.get("world_ranker_used", 0))
        world_ranker_pruned = int(mcts_stats.get("world_ranker_pruned", 0))
        leaf_value_used = int(mcts_stats.get("leaf_value_used", 0))
        leaf_value_escalated = int(mcts_stats.get("leaf_value_escalated", 0))
        leaf_value_failed = int(mcts_stats.get("leaf_value_escalate_failed", 0))
        leaf_value_avg = float(mcts_stats.get("avg_leaf_value_pred", 0.0))
        if adaptive_triggered or adaptive_rerank or adaptive_second_pass:
            print(
                "   Adaptive(trigger/rerank/2nd): {}/{}/{}".format(
                    adaptive_triggered,
                    adaptive_rerank,
                    adaptive_second_pass,
                )
            )
        if (
            search_prior_calls
            or search_prior_used
            or search_prior_mask_empty
            or search_prior_unmapped
            or search_prior_zero_sum
            or search_prior_init_failed
            or search_prior_apply_failed
        ):
            print(
                "   Search prior calls/uses/mask0/unmapped/zero/initF/applyF: {}/{}/{}/{}/{}/{}/{}".format(
                    search_prior_calls,
                    search_prior_used,
                    search_prior_mask_empty,
                    search_prior_unmapped,
                    search_prior_zero_sum,
                    search_prior_init_failed,
                    search_prior_apply_failed,
                )
            )
        if switch_prior_used or switch_prior_pruned:
            print(f"   Switch prior uses/pruned: {switch_prior_used}/{switch_prior_pruned}")
        if (
            passive_breaker_calls
            or passive_breaker_used
            or passive_breaker_lowconf_skip
            or passive_breaker_nonpassive_top
            or passive_breaker_no_priors
            or passive_breaker_no_candidate
            or passive_breaker_min_prob
            or passive_breaker_min_margin
        ):
            print(
                "   Passive breaker calls/uses/conf/nonpass/noPrior/noCand/minP/minM: {}/{}/{}/{}/{}/{}/{}/{}".format(
                    passive_breaker_calls,
                    passive_breaker_used,
                    passive_breaker_lowconf_skip,
                    passive_breaker_nonpassive_top,
                    passive_breaker_no_priors,
                    passive_breaker_no_candidate,
                    passive_breaker_min_prob,
                    passive_breaker_min_margin,
                )
            )
        if tera_pruner_used or tera_pruner_pruned:
            print(f"   Tera pruner uses/pruned: {tera_pruner_used}/{tera_pruner_pruned}")
        if world_ranker_used or world_ranker_pruned:
            print(f"   World ranker uses/pruned: {world_ranker_used}/{world_ranker_pruned}")
        if leaf_value_used or leaf_value_escalated or leaf_value_failed:
            print(
                "   Leaf value uses/escalated/failed avg: {}/{}/{} {:.3f}".format(
                    leaf_value_used,
                    leaf_value_escalated,
                    leaf_value_failed,
                    leaf_value_avg,
                )
            )
        diag_turns = int(mcts_stats.get("diag_turns", 0))
        if diag_turns > 0:
            print("\n📌 Decision Diagnostics")
            print(
                "   Low conf/margin/non-top1: {:.1%}/{:.1%}/{:.1%}".format(
                    float(mcts_stats.get("diag_low_conf_rate", 0.0)),
                    float(mcts_stats.get("diag_low_margin_rate", 0.0)),
                    float(mcts_stats.get("diag_non_top1_rate", 0.0)),
                )
            )
            print(
                "   Choice mix move/switch/tera: {}/{}/{}".format(
                    int(mcts_stats.get("diag_move_choices", 0)),
                    int(mcts_stats.get("diag_switch_choices", 0)),
                    int(mcts_stats.get("diag_tera_choices", 0)),
                )
            )
            print(
                "   Avg delta(best-chosen): {:.3f}".format(
                    float(mcts_stats.get("diag_choice_delta_avg", 0.0))
                )
            )
            print(
                "   Paths mcts/policy/rerank/blend/adapt/fallbackS/fallbackR: {}/{}/{}/{}/{}/{}/{}".format(
                    int(mcts_stats.get("diag_path_mcts", 0)),
                    int(mcts_stats.get("diag_path_policy", 0)),
                    int(mcts_stats.get("diag_path_rerank", 0)),
                    int(mcts_stats.get("diag_path_blend", 0)),
                    int(mcts_stats.get("diag_path_adaptive_rerank", 0)),
                    int(mcts_stats.get("diag_path_fallback_super", 0)),
                    int(mcts_stats.get("diag_path_fallback_random", 0)),
                )
            )
            reason_keys = sorted(
                [k for k in mcts_stats.keys() if k.startswith("diag_adaptive_reason_")]
            )
            reason_counts = []
            for key in reason_keys:
                count = int(mcts_stats.get(key, 0))
                if count > 0:
                    reason_counts.append((key.replace("diag_adaptive_reason_", ""), count))
            if reason_counts:
                reason_counts.sort(key=lambda x: x[1], reverse=True)
                top_reasons = ", ".join(f"{name}:{count}" for name, count in reason_counts[:5])
                print(f"   Adaptive reasons(top): {top_reasons}")
            loss_total = int(mcts_stats.get("diag_battles_lost", 0))
            if loss_total > 0:
                print(
                    "   Loss clusters fast/lowconf/switch/passive/forced/hazard/tempo/adapt/churn/other: {}/{}/{}/{}/{}/{}/{}/{}/{}/{}".format(
                        int(mcts_stats.get("diag_loss_fast", 0)),
                        int(mcts_stats.get("diag_loss_low_conf", 0)),
                        int(mcts_stats.get("diag_loss_switch_heavy", 0)),
                        int(mcts_stats.get("diag_loss_passive", mcts_stats.get("diag_loss_status_loop", 0))),
                        int(mcts_stats.get("diag_loss_forced_switch", 0)),
                        int(mcts_stats.get("diag_loss_hazard_pivot", 0)),
                        int(mcts_stats.get("diag_loss_tempo", 0)),
                        int(mcts_stats.get("diag_loss_adaptive_used", 0)),
                        int(mcts_stats.get("diag_loss_churn_breaks", 0)),
                        int(mcts_stats.get("diag_loss_other", 0)),
                    )
                )

    if move_count:
        print("\n📌 Action Summary")
        print(f"   Total moves:   {move_count}")
        print(f"   Attack rate:   {attack_moves / move_count:.1%}")
        print(f"   Status rate:   {status_moves / move_count:.1%}")
        print(f"   Setup rate:    {setup_moves / move_count:.1%}")
        print(f"   Hazard rate:   {hazard_moves / move_count:.1%}")
        print(f"   Anti-hazard:   {anti_hazard_moves / move_count:.1%}")
        print(f"   Recovery rate: {recovery_moves / move_count:.1%}")
        print(f"   Protect rate:  {protect_moves / move_count:.1%}")
        print(f"   Pivot rate:    {pivot_moves / move_count:.1%}")
        print(f"   Priority rate: {priority_moves / move_count:.1%}")

    if move_counter:
        print("\n📌 Top Moves")
        for move_id, count in move_counter.most_common(10):
            print(f"   {move_id}: {count}")

    if switch_counter:
        print("\n📌 Top Switches")
        for species, count in switch_counter.most_common(10):
            print(f"   {species}: {count}")

    if accept_task and not accept_task.done():
        accept_task.cancel()
        try:
            await accept_task
        except asyncio.CancelledError:
            pass
        except Exception:
            pass

    if proc.poll() is None:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

    return 0 if finished == args.battles else 2


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
