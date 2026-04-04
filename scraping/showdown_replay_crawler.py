#!/usr/bin/env python3
"""Crawl public Pokemon Showdown replays into a resumable local corpus.

This script is intentionally conservative:
- public replay search only
- configurable rate limit
- resumable crawl state in SQLite
- raw replay JSON persisted before downstream parsing

Default assumptions target public Gen 9 Random Battle search pages.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

DEFAULT_SEARCH_URL = "https://replay.pokemonshowdown.com/search.json?format={format_id}&page={page}"
DEFAULT_REPLAY_URL = "https://replay.pokemonshowdown.com/{replay_id}.json"
USER_AGENT = "oranguru-mcts-showdown-crawler/1.0"


@dataclass
class ReplayCandidate:
    replay_id: str
    format_id: str
    rating: float | None
    uploadtime: int | None
    players: list[str]
    page: int
    raw_item: dict[str, Any]


def _norm(value: object) -> str:
    if value is None:
        return ""
    return "".join(ch.lower() for ch in str(value).strip() if ch.isalnum())


def _safe_json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=True, sort_keys=True)


def _fetch_json(url: str, timeout_s: float) -> Any:
    req = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    with urlopen(req, timeout=timeout_s) as resp:
        raw = resp.read()
    return json.loads(raw.decode("utf-8"))


def _fetch_json_with_retries(
    url: str,
    *,
    timeout_s: float,
    retries: int,
    retry_sleep_s: float,
    context: str,
) -> Any:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return _fetch_json(url, timeout_s=timeout_s)
        except (HTTPError, URLError, TimeoutError) as exc:
            last_error = exc
            if attempt >= retries:
                raise
            wait_s = retry_sleep_s * float(attempt + 1)
            print(f"{context}: attempt {attempt + 1}/{retries + 1} failed: {exc}; retrying in {wait_s:.1f}s")
            time.sleep(wait_s)
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"{context}: unexpected fetch failure")


def _load_search_items(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("results", "replays", "items", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    return []


def _parse_replay_candidate(item: dict[str, Any], default_format: str, page: int) -> ReplayCandidate | None:
    replay_id = str(
        item.get("id")
        or item.get("replayid")
        or item.get("battle_id")
        or item.get("name")
        or ""
    ).strip()
    if not replay_id:
        return None
    format_id = str(item.get("format") or item.get("formatid") or default_format).strip() or default_format

    rating = item.get("rating")
    if rating is None:
        rating = item.get("elo")
    if rating is None:
        rating = item.get("rating_pre")
    rating_value: float | None = None
    if isinstance(rating, (int, float)):
        rating_value = float(rating)
    else:
        try:
            rating_value = float(str(rating)) if rating is not None else None
        except Exception:
            rating_value = None

    uploadtime = item.get("uploadtime")
    if not isinstance(uploadtime, int):
        try:
            uploadtime = int(uploadtime) if uploadtime is not None else None
        except Exception:
            uploadtime = None

    players_obj = item.get("players")
    players: list[str] = []
    if isinstance(players_obj, list):
        players = [str(v) for v in players_obj[:2]]
    elif isinstance(players_obj, str):
        bits = [bit.strip() for bit in players_obj.split(",") if bit.strip()]
        players = bits[:2]

    return ReplayCandidate(
        replay_id=replay_id,
        format_id=format_id,
        rating=rating_value,
        uploadtime=uploadtime,
        players=players,
        page=page,
        raw_item=item,
    )


def _extract_log_text(payload: dict[str, Any]) -> str:
    for key in ("log", "inputlog", "inputLog"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _count_turns_from_log(log_text: str) -> int:
    return sum(1 for line in log_text.splitlines() if line.startswith("|turn|"))


def _winner_from_payload(payload: dict[str, Any], log_text: str) -> str | None:
    for key in ("winner", "winnerid"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    for line in log_text.splitlines():
        if line.startswith("|win|"):
            return line.split("|", 2)[2].strip() if "|" in line else None
    return None


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sleep_with_jitter(base_s: float, jitter_s: float) -> None:
    wait_s = max(0.0, base_s) + random.uniform(0.0, max(0.0, jitter_s))
    if wait_s > 0:
        time.sleep(wait_s)


def _ensure_db(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS crawl_pages (
            format_id TEXT NOT NULL,
            page INTEGER NOT NULL,
            fetched_at INTEGER NOT NULL,
            url TEXT NOT NULL,
            item_count INTEGER NOT NULL,
            payload_path TEXT NOT NULL,
            PRIMARY KEY (format_id, page)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS replays (
            replay_id TEXT PRIMARY KEY,
            format_id TEXT NOT NULL,
            page INTEGER,
            rating REAL,
            uploadtime INTEGER,
            players_json TEXT NOT NULL,
            search_item_json TEXT NOT NULL,
            raw_json_path TEXT,
            raw_sha256 TEXT,
            turn_count INTEGER,
            winner TEXT,
            accepted INTEGER NOT NULL DEFAULT 0,
            reject_reason TEXT,
            fetched_at INTEGER,
            parse_status TEXT,
            parse_error TEXT
        )
        """
    )
    conn.commit()


def _upsert_page(
    conn: sqlite3.Connection,
    *,
    format_id: str,
    page: int,
    fetched_at: int,
    url: str,
    item_count: int,
    payload_path: str,
) -> None:
    conn.execute(
        """
        INSERT INTO crawl_pages(format_id, page, fetched_at, url, item_count, payload_path)
        VALUES(?, ?, ?, ?, ?, ?)
        ON CONFLICT(format_id, page) DO UPDATE SET
            fetched_at=excluded.fetched_at,
            url=excluded.url,
            item_count=excluded.item_count,
            payload_path=excluded.payload_path
        """,
        (format_id, page, fetched_at, url, item_count, payload_path),
    )
    conn.commit()


def _upsert_replay(conn: sqlite3.Connection, candidate: ReplayCandidate) -> None:
    conn.execute(
        """
        INSERT INTO replays(
            replay_id, format_id, page, rating, uploadtime,
            players_json, search_item_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(replay_id) DO UPDATE SET
            format_id=excluded.format_id,
            page=excluded.page,
            rating=excluded.rating,
            uploadtime=excluded.uploadtime,
            players_json=excluded.players_json,
            search_item_json=excluded.search_item_json
        """,
        (
            candidate.replay_id,
            candidate.format_id,
            candidate.page,
            candidate.rating,
            candidate.uploadtime,
            _safe_json_dumps(candidate.players),
            _safe_json_dumps(candidate.raw_item),
        ),
    )
    conn.commit()


def _mark_replay_fetch(
    conn: sqlite3.Connection,
    *,
    replay_id: str,
    raw_json_path: str,
    raw_sha256: str,
    turn_count: int,
    winner: str | None,
    accepted: bool,
    reject_reason: str | None,
) -> None:
    conn.execute(
        """
        UPDATE replays
        SET raw_json_path=?, raw_sha256=?, turn_count=?, winner=?, accepted=?, reject_reason=?, fetched_at=?
        WHERE replay_id=?
        """,
        (
            raw_json_path,
            raw_sha256,
            turn_count,
            winner,
            1 if accepted else 0,
            reject_reason,
            int(time.time()),
            replay_id,
        ),
    )
    conn.commit()


def _mark_replay_error(
    conn: sqlite3.Connection,
    *,
    replay_id: str,
    reject_reason: str,
) -> None:
    conn.execute(
        """
        UPDATE replays
        SET accepted=0, reject_reason=?, fetched_at=?
        WHERE replay_id=?
        """,
        (reject_reason, int(time.time()), replay_id),
    )
    conn.commit()


def main() -> int:
    parser = argparse.ArgumentParser(description="Crawl public Pokemon Showdown replays.")
    parser.add_argument("--format-id", default="gen9randombattle")
    parser.add_argument("--min-rating", type=float, default=1500.0)
    parser.add_argument("--min-turns", type=int, default=10)
    parser.add_argument("--start-page", type=int, default=1)
    parser.add_argument("--max-pages", type=int, default=100)
    parser.add_argument("--limit-replays", type=int, default=0, help="0 means no explicit replay cap")
    parser.add_argument("--stop-after-empty-pages", type=int, default=3)
    parser.add_argument("--sleep-seconds", type=float, default=1.0)
    parser.add_argument("--jitter-seconds", type=float, default=0.25)
    parser.add_argument("--timeout-seconds", type=float, default=20.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-sleep-seconds", type=float, default=2.0)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--search-url-template", default=DEFAULT_SEARCH_URL)
    parser.add_argument("--replay-url-template", default=DEFAULT_REPLAY_URL)
    parser.add_argument("--raw-root", default="data/showdown/raw")
    parser.add_argument("--db-path", default="data/showdown/index.sqlite")
    parser.add_argument("--summary-out", default="logs/replay_audit/showdown_crawl.summary.json")
    parser.add_argument("--allow-missing-rating", action="store_true")
    parser.add_argument("--overwrite-raw", action="store_true")
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    search_root = raw_root / "search_pages" / args.format_id
    replay_root = raw_root / "replays"
    search_root.mkdir(parents=True, exist_ok=True)
    replay_root.mkdir(parents=True, exist_ok=True)
    Path(args.db_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_out).parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(args.db_path)
    try:
        _ensure_db(conn)
        stats: dict[str, int] = {
            "pages_fetched": 0,
            "search_items_seen": 0,
            "replays_considered": 0,
            "replays_downloaded": 0,
            "accepted": 0,
            "reject_low_rating": 0,
            "reject_missing_rating": 0,
            "reject_short_game": 0,
            "http_errors": 0,
            "url_errors": 0,
        }
        empty_pages = 0
        downloaded = 0

        for page in range(args.start_page, args.start_page + max(0, args.max_pages)):
            search_url = args.search_url_template.format(
                format_id=quote(args.format_id, safe=""),
                page=page,
            )
            try:
                payload = _fetch_json_with_retries(
                    search_url,
                    timeout_s=args.timeout_seconds,
                    retries=max(0, args.retries),
                    retry_sleep_s=max(0.0, args.retry_sleep_seconds),
                    context=f"search page {page}",
                )
            except HTTPError as exc:
                stats["http_errors"] += 1
                print(f"HTTP error on search page {page}: {exc}")
                break
            except URLError as exc:
                stats["url_errors"] += 1
                print(f"URL error on search page {page}: {exc}")
                break
            except TimeoutError as exc:
                stats["url_errors"] += 1
                print(f"Timeout on search page {page}: {exc}")
                break

            payload_path = search_root / f"page_{page:06d}.json"
            payload_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
            items = _load_search_items(payload)
            stats["pages_fetched"] += 1
            stats["search_items_seen"] += len(items)
            _upsert_page(
                conn,
                format_id=args.format_id,
                page=page,
                fetched_at=int(time.time()),
                url=search_url,
                item_count=len(items),
                payload_path=str(payload_path),
            )
            print(f"Page {page}: {len(items)} candidates")

            if not items:
                empty_pages += 1
                if empty_pages >= max(1, args.stop_after_empty_pages):
                    break
                _sleep_with_jitter(args.sleep_seconds, args.jitter_seconds)
                continue
            empty_pages = 0

            for item in items:
                candidate = _parse_replay_candidate(item, args.format_id, page)
                if candidate is None:
                    continue
                stats["replays_considered"] += 1
                _upsert_replay(conn, candidate)

                if candidate.rating is None and not args.allow_missing_rating:
                    stats["reject_missing_rating"] += 1
                    _mark_replay_fetch(
                        conn,
                        replay_id=candidate.replay_id,
                        raw_json_path="",
                        raw_sha256="",
                        turn_count=0,
                        winner=None,
                        accepted=False,
                        reject_reason="missing_rating",
                    )
                    continue
                if candidate.rating is not None and candidate.rating < args.min_rating:
                    stats["reject_low_rating"] += 1
                    _mark_replay_fetch(
                        conn,
                        replay_id=candidate.replay_id,
                        raw_json_path="",
                        raw_sha256="",
                        turn_count=0,
                        winner=None,
                        accepted=False,
                        reject_reason="low_rating",
                    )
                    continue

                raw_path = replay_root / f"{candidate.replay_id}.json"
                payload_obj: dict[str, Any] | None = None
                raw_bytes: bytes | None = None
                if raw_path.exists() and not args.overwrite_raw:
                    try:
                        raw_text = raw_path.read_text(encoding="utf-8")
                        payload_obj = json.loads(raw_text)
                        raw_bytes = raw_text.encode("utf-8")
                    except Exception:
                        payload_obj = None
                        raw_bytes = None

                if payload_obj is None or raw_bytes is None:
                    replay_url = args.replay_url_template.format(replay_id=quote(candidate.replay_id, safe=""))
                    try:
                        payload_obj = _fetch_json_with_retries(
                            replay_url,
                            timeout_s=args.timeout_seconds,
                            retries=max(0, args.retries),
                            retry_sleep_s=max(0.0, args.retry_sleep_seconds),
                            context=f"replay {candidate.replay_id}",
                        )
                        raw_bytes = json.dumps(payload_obj, ensure_ascii=True, indent=2).encode("utf-8")
                    except HTTPError as exc:
                        stats["http_errors"] += 1
                        print(f"HTTP error on replay {candidate.replay_id}: {exc}")
                        _mark_replay_error(conn, replay_id=candidate.replay_id, reject_reason="fetch_http_error")
                        continue
                    except URLError as exc:
                        stats["url_errors"] += 1
                        print(f"URL error on replay {candidate.replay_id}: {exc}")
                        _mark_replay_error(conn, replay_id=candidate.replay_id, reject_reason="fetch_url_error")
                        continue
                    except TimeoutError as exc:
                        stats["url_errors"] += 1
                        print(f"Timeout on replay {candidate.replay_id}: {exc}")
                        _mark_replay_error(conn, replay_id=candidate.replay_id, reject_reason="fetch_timeout")
                        continue

                    payload_obj["_scrape"] = {
                        "format_id": candidate.format_id,
                        "rating": candidate.rating,
                        "uploadtime": candidate.uploadtime,
                        "players": candidate.players,
                        "search_page": candidate.page,
                    }
                    raw_text = json.dumps(payload_obj, ensure_ascii=True, indent=2)
                    raw_path.write_text(raw_text, encoding="utf-8")
                    raw_bytes = raw_text.encode("utf-8")
                    stats["replays_downloaded"] += 1
                    downloaded += 1
                    if args.progress_every > 0 and downloaded % args.progress_every == 0:
                        print(
                            "Downloaded {} accepted={} low_rating={} missing_rating={} short={}".format(
                                downloaded,
                                stats["accepted"],
                                stats["reject_low_rating"],
                                stats["reject_missing_rating"],
                                stats["reject_short_game"],
                            )
                        )
                    _sleep_with_jitter(args.sleep_seconds, args.jitter_seconds)

                assert payload_obj is not None and raw_bytes is not None
                log_text = _extract_log_text(payload_obj)
                turn_count = _count_turns_from_log(log_text)
                winner = _winner_from_payload(payload_obj, log_text)
                accepted = turn_count >= args.min_turns
                reject_reason = None if accepted else "short_game"
                if not accepted:
                    stats["reject_short_game"] += 1
                else:
                    stats["accepted"] += 1
                _mark_replay_fetch(
                    conn,
                    replay_id=candidate.replay_id,
                    raw_json_path=str(raw_path),
                    raw_sha256=_sha256_bytes(raw_bytes),
                    turn_count=turn_count,
                    winner=winner,
                    accepted=accepted,
                    reject_reason=reject_reason,
                )

                if args.limit_replays > 0 and downloaded >= args.limit_replays:
                    break

            if args.limit_replays > 0 and downloaded >= args.limit_replays:
                break

        summary = {
            "format_id": args.format_id,
            "min_rating": args.min_rating,
            "min_turns": args.min_turns,
            "search_url_template": args.search_url_template,
            "replay_url_template": args.replay_url_template,
            "stats": stats,
            "db_path": args.db_path,
            "raw_root": str(raw_root),
            "generated_at_unix": int(time.time()),
        }
        Path(args.summary_out).write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")
        print(f"Summary -> {args.summary_out}")
        print(json.dumps(summary["stats"], ensure_ascii=True, sort_keys=True))
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
