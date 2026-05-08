#!/usr/bin/env python3
"""Utilities for logging public-ladder evaluation results.

The logger writes one JSON object per finished battle.  It intentionally keeps
its schema independent of poke-env internals so ladder experiments can be
compared across bot versions and analyzed outside this repository.
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

def to_id_str(value: str) -> str:
    """Normalize a Showdown username without importing poke-env.

    Keeping this module dependency-light avoids surprising side effects when it
    is used by small analysis scripts.
    """

    return re.sub(r"[^a-z0-9]", "", str(value).lower())


SCHEMA_VERSION = 1


@dataclass(frozen=True)
class RatingTransition:
    """A Showdown ladder rating transition parsed from battle messages."""

    pre: int | None = None
    post: int | None = None

    @property
    def delta(self) -> int | None:
        if self.pre is None or self.post is None:
            return None
        return self.post - self.pre


RATING_RE = re.compile(
    r"(.+?)'s rating:\s*(\d+)\s*(?:->|→|&rarr;)\s*(?:<[^>]+>\s*)?(\d+)",
    re.IGNORECASE,
)


def resolve_bot_version(default: str = "unknown") -> str:
    """Resolve a human-readable bot version for ladder logs."""

    env_version = os.environ.get("ORANGURU_BOT_VERSION") or os.environ.get("BOT_VERSION")
    if env_version:
        return env_version.strip()
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except Exception:
        return default
    version = proc.stdout.strip()
    return version or default


def expected_score(player_rating: int | float | None, opponent_rating: int | float | None) -> float | None:
    """Return the Elo expected score for player vs opponent ratings."""

    if player_rating is None or opponent_rating is None:
        return None
    try:
        player = float(player_rating)
        opponent = float(opponent_rating)
    except (TypeError, ValueError):
        return None
    return 1.0 / (1.0 + math.pow(10.0, (opponent - player) / 400.0))


def rating_residual(score: float, expected: float | None) -> float | None:
    if expected is None:
        return None
    return float(score) - float(expected)


def battle_has_forfeit(battle: Any) -> bool:
    observations = getattr(battle, "_observations", {}) or {}
    for obs in observations.values():
        for event in getattr(obs, "events", []) or []:
            if len(event) > 1 and event[1] == "forfeit":
                return True
    return False


def remaining_pokemon(battle: Any, *, player: bool) -> int | None:
    team = getattr(battle, "team", None) if player else getattr(battle, "opponent_team", None)
    if not team:
        return None
    try:
        return sum(1 for mon in team.values() if not getattr(mon, "fainted", False))
    except Exception:
        return None


def _clean_rating_text(text: str) -> str:
    text = text.replace("&rarr;", "->").replace("→", "->")
    text = re.sub(r"<[^>]+>", " ", text)
    return text


def parse_rating_transitions_from_text(text: str) -> dict[str, RatingTransition]:
    """Parse Showdown rating lines from arbitrary raw text."""

    transitions: dict[str, RatingTransition] = {}
    for line in _clean_rating_text(text).splitlines():
        match = RATING_RE.search(line)
        if not match:
            continue
        user_id = to_id_str(match.group(1).strip())
        try:
            pre = int(match.group(2))
            post = int(match.group(3))
        except (TypeError, ValueError):
            continue
        if user_id:
            transitions[user_id] = RatingTransition(pre=pre, post=post)
    return transitions


def parse_rating_transitions_from_battle(battle: Any) -> dict[str, RatingTransition]:
    """Parse rating transitions from poke-env battle observations."""

    transitions: dict[str, RatingTransition] = {}
    observations = getattr(battle, "_observations", {}) or {}
    for obs in observations.values():
        for event in getattr(obs, "events", []) or []:
            for chunk in event or []:
                if isinstance(chunk, str) and "rating:" in chunk:
                    transitions.update(parse_rating_transitions_from_text(chunk))
    return transitions


def _transition_for_user(
    transitions: Mapping[str, RatingTransition],
    username: str | None,
) -> RatingTransition:
    user_id = to_id_str(username or "")
    if user_id and user_id in transitions:
        return transitions[user_id]
    return RatingTransition()


def extract_ladder_ratings(battle: Any) -> dict[str, int | None]:
    """Extract pre/post ladder ratings for player and opponent.

    Pre-ratings are only available when Showdown rating transition messages are
    present.  Post-ratings may fall back to poke-env's battle.rating fields.
    """

    player_name = getattr(battle, "player_username", None)
    opponent_name = getattr(battle, "opponent_username", None)
    transitions = parse_rating_transitions_from_battle(battle)
    player = _transition_for_user(transitions, player_name)
    opponent = _transition_for_user(transitions, opponent_name)
    if player.post is None and transitions:
        # Some poke-env battle objects omit usernames even though the raw
        # rating lines are present.  Preserve a best-effort fallback so rated
        # battles are still useful for aggregate ladder metrics.
        player = next(iter(transitions.values()))
    if opponent.post is None and len(transitions) > 1:
        opponent = list(transitions.values())[1]

    player_post = player.post if player.post is not None else getattr(battle, "rating", None)
    opponent_post = (
        opponent.post if opponent.post is not None else getattr(battle, "opponent_rating", None)
    )
    return {
        "player_rating_pre": player.pre,
        "player_rating_post": player_post,
        "player_rating_delta": None if player.pre is None or player_post is None else player_post - player.pre,
        "opponent_rating_pre": opponent.pre,
        "opponent_rating_post": opponent_post,
        "opponent_rating_delta": None
        if opponent.pre is None or opponent_post is None
        else opponent_post - opponent.pre,
    }


def extract_ladder_ratings_from_transitions(
    transitions: Mapping[str, RatingTransition],
    *,
    player_username: str | None,
    opponent_username: str | None,
) -> dict[str, int | None]:
    """Extract rating fields from already-parsed transitions."""

    player = _transition_for_user(transitions, player_username)
    opponent = _transition_for_user(transitions, opponent_username)
    if player.post is None and transitions:
        player = next(iter(transitions.values()))
    if opponent.post is None and len(transitions) > 1:
        opponent = list(transitions.values())[1]
    return {
        "player_rating_pre": player.pre,
        "player_rating_post": player.post,
        "player_rating_delta": player.delta,
        "opponent_rating_pre": opponent.pre,
        "opponent_rating_post": opponent.post,
        "opponent_rating_delta": opponent.delta,
    }


def result_score(battle: Any) -> tuple[str, float]:
    if bool(getattr(battle, "won", False)):
        return "win", 1.0
    if bool(getattr(battle, "lost", False)):
        return "loss", 0.0
    return "tie", 0.5


class LadderMetricsLogger:
    """Append-only ladder metrics logger with rating-adjusted aggregates."""

    def __init__(self, path: str | Path | None, *, bot_version: str = "unknown"):
        self.path = Path(path) if path else None
        self.bot_version = bot_version or "unknown"
        self.counts: Counter[str] = Counter()
        self.residual_sum = 0.0
        self.residual_count = 0
        self.expected_sum = 0.0
        self.expected_count = 0
        self.rating_delta_sum = 0
        self.rating_delta_count = 0
        self.rows: list[dict[str, Any]] = []
        self._existing_lines: list[str] = []
        if self.path is not None and self.path.exists():
            self._existing_lines = self.path.read_text(encoding="utf-8").splitlines()

    @property
    def enabled(self) -> bool:
        return self.path is not None

    def log_battle(
        self,
        battle: Any,
        *,
        account: str,
        player_type: str,
        battle_format: str,
        action_counts: Mapping[str, int] | None = None,
        decision_count: int | None = None,
        decision_ms_avg: float | None = None,
        decision_ms_max: float | None = None,
    ) -> dict[str, Any]:
        result, score = result_score(battle)
        ratings = extract_ladder_ratings(battle)
        expected = expected_score(ratings["player_rating_pre"], ratings["opponent_rating_pre"])
        residual = rating_residual(score, expected)
        payload: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "ts": time.time(),
            "bot_version": self.bot_version,
            "account": account,
            "player_type": player_type,
            "format": battle_format,
            "battle_tag": getattr(battle, "battle_tag", None),
            "opponent_username": getattr(battle, "opponent_username", None),
            "result": result,
            "score": score,
            "expected_score": expected,
            "rating_residual": residual,
            "turns": getattr(battle, "turn", None),
            "remaining": remaining_pokemon(battle, player=True),
            "opp_remaining": remaining_pokemon(battle, player=False),
            "forfeit": battle_has_forfeit(battle),
            "decision_count": decision_count,
            "decision_ms_avg": decision_ms_avg,
            "decision_ms_max": decision_ms_max,
            "action_counts": dict(action_counts or {}),
        }
        payload.update(ratings)
        self.rows.append(payload)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, sort_keys=True) + "\n")
        return payload

    def update_battle_ratings_from_text(
        self,
        *,
        battle_tag: str | None,
        text: str,
    ) -> bool:
        """Patch a logged battle row when Showdown emits late rating lines.

        poke-env often marks the battle finished before the final raw ladder
        rating messages are attached to the battle object.  This method keeps
        the metrics file as one row per battle by rewriting the JSONL file after
        the matching row is enriched.
        """

        transitions = parse_rating_transitions_from_text(text)
        if not transitions:
            return False
        row = self._find_row_for_rating_update(battle_tag)
        if row is None:
            return False
        ratings = extract_ladder_ratings_from_transitions(
            transitions,
            player_username=str(row.get("account") or ""),
            opponent_username=str(row.get("opponent_username") or ""),
        )
        return self._update_row_ratings(row, ratings)

    def update_battle_ratings_from_battle(self, battle: Any) -> bool:
        """Patch a logged row from a battle object after late observations arrive."""

        battle_tag = getattr(battle, "battle_tag", None)
        row = self._find_row_for_rating_update(battle_tag)
        if row is None:
            return False
        ratings = extract_ladder_ratings(battle)
        return self._update_row_ratings(row, ratings)

    def _update_row_ratings(
        self,
        row: dict[str, Any],
        ratings: Mapping[str, int | None],
    ) -> bool:
        if ratings["player_rating_pre"] is None and ratings["opponent_rating_pre"] is None:
            return False
        row.update(ratings)
        expected = expected_score(row.get("player_rating_pre"), row.get("opponent_rating_pre"))
        row["expected_score"] = expected
        row["rating_residual"] = rating_residual(float(row.get("score", 0.0)), expected)
        if self.path is not None:
            self._rewrite_file()
        return True

    def _find_row_for_rating_update(self, battle_tag: str | None) -> dict[str, Any] | None:
        if battle_tag:
            for row in reversed(self.rows):
                if row.get("battle_tag") == battle_tag:
                    return row
        for row in reversed(self.rows):
            if row.get("player_rating_pre") is None and row.get("opponent_rating_pre") is None:
                return row
        return None

    def _rewrite_file(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            for line in self._existing_lines:
                handle.write(line.rstrip("\n") + "\n")
            for row in self.rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        tmp.replace(self.path)

    def _accumulate(self, payload: Mapping[str, Any]) -> None:
        self.counts["battles"] += 1
        self.counts[str(payload.get("result") or "unknown")] += 1
        expected = payload.get("expected_score")
        if isinstance(expected, (int, float)):
            self.expected_sum += float(expected)
            self.expected_count += 1
        residual = payload.get("rating_residual")
        if isinstance(residual, (int, float)):
            self.residual_sum += float(residual)
            self.residual_count += 1
        delta = payload.get("player_rating_delta")
        if isinstance(delta, int):
            self.rating_delta_sum += delta
            self.rating_delta_count += 1

    def summary(self) -> dict[str, Any]:
        self.counts.clear()
        self.residual_sum = 0.0
        self.residual_count = 0
        self.expected_sum = 0.0
        self.expected_count = 0
        self.rating_delta_sum = 0
        self.rating_delta_count = 0
        for payload in self.rows:
            self._accumulate(payload)
        battles = int(self.counts.get("battles", 0))
        wins = int(self.counts.get("win", 0))
        losses = int(self.counts.get("loss", 0))
        ties = int(self.counts.get("tie", 0))
        return {
            "battles": battles,
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "win_rate": wins / battles if battles else 0.0,
            "avg_expected_score": self.expected_sum / self.expected_count if self.expected_count else None,
            "avg_rating_residual": self.residual_sum / self.residual_count if self.residual_count else None,
            "rating_residual_sum": self.residual_sum if self.residual_count else None,
            "rating_residual_count": self.residual_count,
            "rating_delta_sum": self.rating_delta_sum if self.rating_delta_count else None,
            "rating_delta_count": self.rating_delta_count,
            "path": str(self.path) if self.path is not None else None,
            "bot_version": self.bot_version,
        }

    def print_summary(self) -> None:
        summary = self.summary()
        print("\n📌 Ladder Metrics")
        print(
            "   Battles: {battles} | W/L/T {wins}/{losses}/{ties} ({win_rate:.1%})".format(
                **summary
            )
        )
        avg_expected = summary["avg_expected_score"]
        avg_residual = summary["avg_rating_residual"]
        if avg_expected is not None:
            print(f"   Avg expected score: {avg_expected:.3f}")
        if avg_residual is not None:
            print(
                f"   Avg rating residual: {avg_residual:+.3f} "
                f"over {summary['rating_residual_count']} rated battles"
            )
        if summary["rating_delta_sum"] is not None:
            print(
                f"   Rating delta sum: {summary['rating_delta_sum']:+d} "
                f"over {summary['rating_delta_count']} rated battles"
            )
        if self.path is not None:
            print(f"   Log: {self.path}")


def load_ladder_metrics(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows
