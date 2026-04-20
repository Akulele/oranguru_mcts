#!/usr/bin/env python3
"""Write a conservative bucket-rule config for the rerank gate.

This is a deliberately small bridge between trace mining and a learned gate:
the runtime hook supports a generic JSON rules format, while this script writes
the first ruleset derived from the April 2026 conservative-window traces.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_RULES = [
    {
        "name": "block_disabled_switch_risk_branch",
        "source": "switch_guard:take_risk_attack",
        "action": "block",
        "notes": "Low label rate and already disabled by default.",
    },
    {
        "name": "block_legacy_near_tie_switch_attack",
        "source": "switch_guard:take_legacy_near_tie_attack",
        "action": "block",
        "notes": "Low sample count and low label rate; preserve MCTS top switch.",
    },
    {
        "name": "block_live_switch_attack_low_drop_low_heur",
        "source": "switch_guard:take_live_attack",
        "score_drop_max": 0.05,
        "heuristic_delta_max": 5.0,
        "action": "block",
        "notes": "Avoid tiny heuristic gains overriding MCTS switch choices.",
    },
    {
        "name": "block_setup_low_drop_moderate_heur",
        "source": "setup_window:take_setup",
        "score_drop_max": 0.05,
        "heuristic_delta_min": 60.0,
        "heuristic_delta_max": 100.0,
        "action": "block",
        "notes": "Weak historical bucket; require stronger evidence before setup reranks.",
    },
    {
        "name": "block_recovery_low_drop_high_heur",
        "source": "recovery_window:take_recovery",
        "score_drop_max": 0.02,
        "heuristic_delta_min": 100.0,
        "action": "block",
        "notes": "Very low label-rate recovery bucket; keep MCTS top action.",
    },
    {
        "name": "block_recovery_mid_drop_high_heur",
        "source": "recovery_window:take_recovery",
        "score_drop_min": 0.05,
        "score_drop_max": 0.10,
        "heuristic_delta_min": 100.0,
        "action": "block",
        "notes": "Observed weak recovery bucket despite large heuristic gain.",
    },
]


def build_config() -> dict:
    return {
        "mode": "bucket_rules",
        "version": 1,
        "default": "allow",
        "description": (
            "Conservative rerank-gate rules from trace bucket rates. "
            "Default allow preserves existing behavior outside known weak buckets."
        ),
        "rules": DEFAULT_RULES,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="checkpoints/rl/rerank_gate_bucket_rules_v1.json",
        help="Path to write the runtime rerank-gate config",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    config = build_config()
    output.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {len(config['rules'])} bucket rules -> {output}")


if __name__ == "__main__":
    main()
