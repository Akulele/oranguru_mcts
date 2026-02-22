#!/usr/bin/env python3
"""
End-to-end runner for the sequence BC -> offline RL pipeline.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class Profile:
    bc_epochs: int
    rl_epochs: int
    batch_size: int
    device: str
    min_delta_acc: float


PROFILES = {
    "server": Profile(
        bc_epochs=40,
        rl_epochs=20,
        batch_size=32,
        device="auto",
        min_delta_acc=0.002,
    ),
    "local": Profile(
        bc_epochs=40,
        rl_epochs=20,
        batch_size=32,
        device="auto",
        min_delta_acc=0.002,
    ),
    "mini": Profile(
        bc_epochs=8,
        rl_epochs=4,
        batch_size=8,
        device="cpu",
        min_delta_acc=0.0,
    ),
}


def run(cmd: list[str], env: dict[str, str]) -> None:
    print("$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run sequence RL pipeline end-to-end.")
    parser.add_argument("--profile", choices=sorted(PROFILES.keys()), default="server")
    parser.add_argument("--run-tag", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--input-dir", default="data/gen9random")
    parser.add_argument("--python", default=sys.executable)

    parser.add_argument("--min-turns", type=int, default=8)
    parser.add_argument("--min-actions", type=int, default=4)
    parser.add_argument("--min-avg-rating", type=int, default=1400)
    parser.add_argument("--holdout-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--bc-epochs", type=int, default=None)
    parser.add_argument("--rl-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default=None, help="auto|cpu|cuda")
    parser.add_argument("--cuda-visible-devices", default="", help="e.g. 0")

    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--skip-bc", action="store_true")
    parser.add_argument("--skip-rl", action="store_true")
    parser.add_argument("--skip-gate", action="store_true")

    parser.add_argument("--train-trajectories", default="")
    parser.add_argument("--holdout-trajectories", default="")
    parser.add_argument("--bc-checkpoint", default="")
    parser.add_argument("--rl-checkpoint", default="")
    parser.add_argument("--gate-min-delta-acc", type=float, default=None)
    args = parser.parse_args()

    profile = PROFILES[args.profile]
    bc_epochs = args.bc_epochs if args.bc_epochs is not None else profile.bc_epochs
    rl_epochs = args.rl_epochs if args.rl_epochs is not None else profile.rl_epochs
    batch_size = args.batch_size if args.batch_size is not None else profile.batch_size
    device = args.device if args.device is not None else profile.device
    min_delta_acc = (
        args.gate_min_delta_acc
        if args.gate_min_delta_acc is not None
        else profile.min_delta_acc
    )

    run_tag = args.run_tag
    Path("logs/replay_audit").mkdir(parents=True, exist_ok=True)
    Path("data").mkdir(parents=True, exist_ok=True)
    Path("checkpoints/rl").mkdir(parents=True, exist_ok=True)

    train_traj = args.train_trajectories or f"data/gen9random_trajectories_train_{run_tag}.pkl"
    holdout_traj = args.holdout_trajectories or f"data/gen9random_trajectories_holdout_{run_tag}.pkl"
    bc_ckpt = args.bc_checkpoint or f"checkpoints/rl/seq_bc_{run_tag}.pt"
    rl_ckpt = args.rl_checkpoint or f"checkpoints/rl/seq_offline_rl_{run_tag}.pt"

    env = dict(os.environ)
    if args.cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    py = args.python

    if not args.skip_prepare:
        run(
            [
                py,
                "training/prepare_gen9random_trajectories.py",
                "--input-dir",
                args.input_dir,
                "--output-train",
                train_traj,
                "--output-holdout",
                holdout_traj,
                "--summary-out",
                f"logs/replay_audit/{run_tag}.prepare.summary.json",
                "--min-turns",
                str(args.min_turns),
                "--min-actions",
                str(args.min_actions),
                "--min-avg-rating",
                str(args.min_avg_rating),
                "--skip-forfeit",
                "--skip-inactivity",
                "--holdout-ratio",
                str(args.holdout_ratio),
                "--seed",
                str(args.seed),
            ],
            env,
        )

    if not args.skip_bc:
        run(
            [
                py,
                "training/train_sequence_bc.py",
                "--input",
                train_traj,
                "--val-input",
                holdout_traj,
                "--output",
                bc_ckpt,
                "--summary-out",
                f"logs/replay_audit/{run_tag}.bc.summary.json",
                "--epochs",
                str(bc_epochs),
                "--batch-size",
                str(batch_size),
                "--lr",
                "3e-4",
                "--weight-decay",
                "1e-4",
                "--device",
                device,
                "--gamma",
                "0.99",
                "--min-actions",
                str(args.min_actions),
                "--value-coef",
                "0.10",
                "--weight-by-rating",
                "--rating-baseline",
                str(args.min_avg_rating),
                "--rating-scale",
                "1000",
                "--max-rating-weight",
                "2.5",
            ],
            env,
        )

    if not args.skip_rl:
        run(
            [
                py,
                "training/finetune_offline_rl.py",
                "--input",
                train_traj,
                "--val-input",
                holdout_traj,
                "--checkpoint-in",
                bc_ckpt,
                "--output",
                rl_ckpt,
                "--summary-out",
                f"logs/replay_audit/{run_tag}.offline_rl.summary.json",
                "--epochs",
                str(rl_epochs),
                "--batch-size",
                str(batch_size),
                "--lr",
                "1e-4",
                "--weight-decay",
                "1e-4",
                "--device",
                device,
                "--gamma",
                "0.99",
                "--min-actions",
                str(args.min_actions),
                "--mode",
                "binary",
                "--value-coef",
                "0.5",
                "--entropy-coef",
                "0.001",
                "--illegal-action-coef",
                "0.1",
            ],
            env,
        )

    if not args.skip_gate:
        run(
            [
                py,
                "training/eval_policy_gate.py",
                "--checkpoint",
                rl_ckpt,
                "--baseline-checkpoint",
                bc_ckpt,
                "--holdout",
                holdout_traj,
                "--summary-out",
                f"logs/replay_audit/{run_tag}.gate.json",
                "--device",
                device,
                "--min-delta-acc",
                str(min_delta_acc),
            ],
            env,
        )

    print("\nPipeline complete.")
    print(f"- train trajectories: {train_traj}")
    print(f"- holdout trajectories: {holdout_traj}")
    print(f"- BC checkpoint: {bc_ckpt}")
    print(f"- RL checkpoint: {rl_ckpt}")
    print(f"- gate summary: logs/replay_audit/{run_tag}.gate.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
