#!/usr/bin/env python3
"""
Gate evaluation for policy checkpoints on holdout trajectories.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.actor_critic import ActorCritic, RecurrentActorCritic
from training.sequence_utils import TrajectoryDataset, collate_trajectories


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _load_trajectories(path: str) -> list[dict]:
    with open(path, "rb") as handle:
        obj = pickle.load(handle)
    if not isinstance(obj, list):
        raise ValueError(f"Expected list trajectory pickle: {path}")
    return obj


def _load_model(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {}) or {}
    model_type = ckpt.get("model_type", "feedforward")
    feature_dim = int(cfg.get("feature_dim", 272))
    d_model = int(cfg.get("d_model", 512))
    n_actions = int(cfg.get("n_actions", 13))
    rnn_hidden = int(cfg.get("rnn_hidden", 512))
    rnn_layers = int(cfg.get("rnn_layers", 1))

    if model_type == "recurrent":
        model = RecurrentActorCritic(
            feature_dim=feature_dim,
            d_model=d_model,
            n_actions=n_actions,
            rnn_hidden=rnn_hidden,
            rnn_layers=rnn_layers,
        ).to(device)
    else:
        model = ActorCritic(
            feature_dim=feature_dim, d_model=d_model, n_actions=n_actions
        ).to(device)

    if "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    elif "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        raise ValueError(f"Checkpoint missing model weights: {checkpoint_path}")
    model.eval()
    return model


def _forward(model, features, masks):
    if getattr(model, "is_recurrent", False):
        logits, _, _ = model.forward_sequence(features)
    else:
        b, t, d = features.shape
        fa = features.reshape(b * t, d)
        ma = masks.reshape(b * t, masks.shape[-1])
        logits_flat, _ = model.forward(fa, ma)
        logits = logits_flat.reshape(b, t, -1)
    return logits


@torch.no_grad()
def _evaluate(model, loader: DataLoader, device: torch.device):
    totals = {
        "steps": 0.0,
        "nll_sum": 0.0,
        "acc_num": 0.0,
        "move_num": 0.0,
        "move_den": 0.0,
        "switch_num": 0.0,
        "switch_den": 0.0,
    }
    for batch in loader:
        features = batch.features.to(device)
        masks = batch.masks.to(device)
        actions = batch.actions.to(device)
        valid = batch.valid.to(device)

        logits = _forward(model, features, masks)
        masked_logits = logits.masked_fill(~masks, -1e9)
        log_probs = F.log_softmax(masked_logits, dim=-1)
        nll = -log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)

        preds = masked_logits.argmax(dim=-1)
        valid_f = valid.float()
        steps = valid_f.sum().item()
        totals["steps"] += steps
        totals["nll_sum"] += (nll * valid_f).sum().item()
        totals["acc_num"] += (((preds == actions) & valid).float()).sum().item()

        move_valid = valid & ((actions < 4) | (actions >= 9))
        switch_valid = valid & ((actions >= 4) & (actions < 9))
        totals["move_num"] += (((preds == actions) & move_valid).float()).sum().item()
        totals["move_den"] += move_valid.float().sum().item()
        totals["switch_num"] += (((preds == actions) & switch_valid).float()).sum().item()
        totals["switch_den"] += switch_valid.float().sum().item()

    if totals["steps"] <= 0:
        raise RuntimeError("No valid steps during gate evaluation.")
    return {
        "nll": totals["nll_sum"] / totals["steps"],
        "acc": totals["acc_num"] / totals["steps"],
        "move_acc": totals["move_num"] / max(1.0, totals["move_den"]),
        "switch_acc": totals["switch_num"] / max(1.0, totals["switch_den"]),
        "steps": int(totals["steps"]),
    }


def _threshold_ok(name: str, value: float, lower: float | None, upper: float | None, failures: list[str]):
    if lower is not None and value < lower:
        failures.append(f"{name}={value:.4f} < min {lower:.4f}")
    if upper is not None and value > upper:
        failures.append(f"{name}={value:.4f} > max {upper:.4f}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate checkpoint against holdout trajectory gate.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--holdout", required=True)
    parser.add_argument("--baseline-checkpoint", default="")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--min-actions", type=int, default=4)
    parser.add_argument("--summary-out", default="")

    parser.add_argument("--min-acc", type=float, default=None)
    parser.add_argument("--min-move-acc", type=float, default=None)
    parser.add_argument("--min-switch-acc", type=float, default=None)
    parser.add_argument("--max-nll", type=float, default=None)
    parser.add_argument("--min-delta-acc", type=float, default=None)
    args = parser.parse_args()

    device = _resolve_device(args.device)
    holdout_trajs = _load_trajectories(args.holdout)
    ds = TrajectoryDataset(
        holdout_trajs,
        gamma=args.gamma,
        min_actions=args.min_actions,
        weight_by_rating=False,
    )
    if len(ds) == 0:
        raise SystemExit("Holdout dataset empty after filtering.")
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_trajectories,
    )

    model = _load_model(args.checkpoint, device)
    metrics = _evaluate(model, loader, device)

    baseline_metrics = None
    if args.baseline_checkpoint:
        baseline = _load_model(args.baseline_checkpoint, device)
        baseline_metrics = _evaluate(baseline, loader, device)

    failures: list[str] = []
    _threshold_ok("acc", metrics["acc"], args.min_acc, None, failures)
    _threshold_ok("move_acc", metrics["move_acc"], args.min_move_acc, None, failures)
    _threshold_ok("switch_acc", metrics["switch_acc"], args.min_switch_acc, None, failures)
    _threshold_ok("nll", metrics["nll"], None, args.max_nll, failures)

    if args.min_delta_acc is not None:
        if baseline_metrics is None:
            failures.append("min-delta-acc requires --baseline-checkpoint")
        else:
            delta = metrics["acc"] - baseline_metrics["acc"]
            if delta < args.min_delta_acc:
                failures.append(
                    f"delta_acc={delta:.4f} < min_delta_acc {args.min_delta_acc:.4f}"
                )

    passed = len(failures) == 0
    payload = {
        "checkpoint": args.checkpoint,
        "holdout": args.holdout,
        "baseline_checkpoint": args.baseline_checkpoint or None,
        "metrics": metrics,
        "baseline_metrics": baseline_metrics,
        "thresholds": {
            "min_acc": args.min_acc,
            "min_move_acc": args.min_move_acc,
            "min_switch_acc": args.min_switch_acc,
            "max_nll": args.max_nll,
            "min_delta_acc": args.min_delta_acc,
        },
        "passed": passed,
        "failures": failures,
    }

    summary_path = (
        Path(args.summary_out)
        if args.summary_out
        else Path(args.checkpoint).with_suffix(".gate.json")
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {args.checkpoint}")
    print(
        "metrics: "
        f"acc={metrics['acc']:.4f} move={metrics['move_acc']:.4f} "
        f"switch={metrics['switch_acc']:.4f} nll={metrics['nll']:.4f} steps={metrics['steps']}"
    )
    if baseline_metrics:
        delta = metrics["acc"] - baseline_metrics["acc"]
        print(
            "baseline: "
            f"acc={baseline_metrics['acc']:.4f} nll={baseline_metrics['nll']:.4f} "
            f"delta_acc={delta:+.4f}"
        )
    if failures:
        print("failures:")
        for msg in failures:
            print(f"- {msg}")
    print(f"summary: {summary_path}")

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
