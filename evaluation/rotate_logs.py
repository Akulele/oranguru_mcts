#!/usr/bin/env python3
"""Rotate logs into current/previous/old folders."""

from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


def _is_empty(path: Path) -> bool:
    if not path.exists():
        return True
    try:
        return not any(path.iterdir())
    except OSError:
        return True


def _rotate(base: Path) -> None:
    current = base / "current"
    previous = base / "previous"
    old = base / "old"
    current.mkdir(parents=True, exist_ok=True)
    previous.mkdir(parents=True, exist_ok=True)
    old.mkdir(parents=True, exist_ok=True)

    if _is_empty(current):
        return

    if not _is_empty(previous):
        stamp = time.strftime("%Y%m%d_%H%M%S")
        dest = old / f"previous_{stamp}"
        shutil.move(str(previous), str(dest))
        previous.mkdir(parents=True, exist_ok=True)

    shutil.move(str(current), str(previous))
    current.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="logs", help="Logs root directory")
    args = parser.parse_args()
    root = PROJECT_ROOT / args.root

    _rotate(root / "foulplay")
    _rotate(root / "evals")


if __name__ == "__main__":
    main()
