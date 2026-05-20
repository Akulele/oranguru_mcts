#!/usr/bin/env python3
"""Refresh cached Pokemon Showdown random-battle set references.

The engine consumes the pkmn/randbats "full" aggregate because it is already
expanded into concrete 4-move/item/ability/tera sets with counts. The official
Showdown source file is also cached for audit/reference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.request import urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PKMN_FULL_URL = "https://pkmn.github.io/randbats/data/full/{format_id}.json"
SHOWDOWN_SOURCE_URL = (
    "https://raw.githubusercontent.com/smogon/pokemon-showdown/master/"
    "data/random-battles/{gen}/sets.json"
)


def _download_json(url: str) -> dict:
    with urlopen(url, timeout=30) as response:
        payload = response.read().decode("utf-8")
    data = json.loads(payload)
    if not isinstance(data, dict) or not data:
        raise ValueError(f"downloaded JSON was empty or non-object: {url}")
    return data


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    tmp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Refresh randbats set caches.")
    parser.add_argument("--format", default="gen9randombattle", help="Randbats format id.")
    parser.add_argument("--gen", default="gen9", help="Showdown random-battles source directory.")
    parser.add_argument(
        "--cache-out",
        default="third_party/foul-play/data/pkmn_sets_cache/gen9randombattle.json",
        help="Output path for expanded pkmn/randbats full-set counts.",
    )
    parser.add_argument(
        "--source-out",
        default="data/showdown_random_sets/gen9_sets.json",
        help="Output path for official Showdown source sets.json.",
    )
    parser.add_argument("--skip-source", action="store_true", help="Only refresh the expanded full-set cache.")
    args = parser.parse_args()

    full_url = PKMN_FULL_URL.format(format_id=args.format)
    full_data = _download_json(full_url)
    cache_out = PROJECT_ROOT / args.cache_out
    _write_json(cache_out, full_data)
    print(f"Wrote {cache_out} ({len(full_data)} species) from {full_url}")

    if not args.skip_source:
        source_url = SHOWDOWN_SOURCE_URL.format(gen=args.gen)
        source_data = _download_json(source_url)
        source_out = PROJECT_ROOT / args.source_out
        _write_json(source_out, source_data)
        print(f"Wrote {source_out} ({len(source_data)} species) from {source_url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
