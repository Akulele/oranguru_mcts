#!/usr/bin/env python3
"""
Smoke-test initialization of Oranguru's search prior checkpoint.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.players.oranguru_engine import OranguruEnginePlayer


def main() -> int:
    parser = argparse.ArgumentParser(description="Check Oranguru search-prior initialization.")
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/rl/search_prior_value_fp_teacher_policy_only_sharp.pt",
        help="Checkpoint path to test.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device override for search prior init.",
    )
    args = parser.parse_args()

    p = OranguruEnginePlayer()
    p.SEARCH_PRIOR_ENABLED = True
    p.SEARCH_PRIOR_CHECKPOINT = args.checkpoint
    p.SEARCH_PRIOR_DEVICE = args.device

    ok = p._init_search_prior()
    print("init_ok:", ok)
    print("ready:", getattr(p, "_search_prior_ready", None))
    print("failed:", getattr(p, "_search_prior_failed", None))
    print("checkpoint:", getattr(p, "_search_prior_checkpoint", None))
    print("device:", getattr(p, "_search_prior_device", None))
    print("model_none:", getattr(p, "_search_prior_model", None) is None)
    print("builder_none:", getattr(p, "_search_prior_feature_builder", None) is None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
