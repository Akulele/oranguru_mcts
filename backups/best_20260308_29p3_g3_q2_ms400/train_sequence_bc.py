#!/usr/bin/env python3
"""
Train a recurrent sequence policy with behavior cloning on trajectory data.
"""

from __future__ import annotations

import argparse
import random
import json
import pickle
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.actor_critic import RecurrentActorCritic
from training.sequence_utils import (
    TrajectoryDataset,
    collate_trajectories,
    set_seed,
    stable_split_by_battle,
)


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _move_batch(batch, device: torch.device):
    return (
        batch.features.to(device),
        batch.masks.to(device),
        batch.actions.to(device),
        batch.returns.to(device),
        batch.valid.to(device),
        batch.weights.to(device),
    )


def _sequence_metrics(logits, masks, actions, valid):
    masked_logits = logits.masked_fill(~masks, -1e9)
    preds = masked_logits.argmax(dim=-1)
    correct = (preds == actions) & valid
    total = valid.sum().item()
    total_correct = correct.sum().item()

    move_valid = valid & ((actions < 4) | (actions >= 9))
    switch_valid = valid & ((actions >= 4) & (actions < 9))
    move_total = move_valid.sum().item()
    switch_total = switch_valid.sum().item()
    move_correct = ((preds == actions) & move_valid).sum().item()
    switch_correct = ((preds == actions) & switch_valid).sum().item()
    return {
        "total": total,
        "correct": total_correct,
        "move_total": move_total,
        "move_correct": move_correct,
        "switch_total": switch_total,
        "switch_correct": switch_correct,
    }


def train_epoch(
    model: RecurrentActorCritic,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    value_coef: float,
    switch_action_weight: float,
    grad_clip: float,
):
    model.train()
    total_loss = 0.0
    total_bc = 0.0
    total_value = 0.0
    total_steps = 0.0
    metrics = {"total": 0, "correct": 0, "move_total": 0, "move_correct": 0, "switch_total": 0, "switch_correct": 0}

    for batch in loader:
        features, masks, actions, returns, valid, weights = _move_batch(batch, device)
        optimizer.zero_grad()

        logits, values, _ = model.forward_sequence(features)
        masked_logits = logits.masked_fill(~masks, -1e9)
        log_probs = F.log_softmax(masked_logits, dim=-1)
        nll = -log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)

        valid_f = valid.float()
        sample_w = weights
        if switch_action_weight > 1.0:
            switch_mask = ((actions >= 4) & (actions < 9)).float()
            sample_w = sample_w * (1.0 + (switch_action_weight - 1.0) * switch_mask)

        denom = valid_f.sum().clamp(min=1.0)
        bc_loss = (nll * sample_w * valid_f).sum() / denom
        value_loss = ((values - returns) ** 2 * valid_f).sum() / denom
        loss = bc_loss + value_coef * value_loss

        if torch.isnan(loss) or torch.isinf(loss):
            continue

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        steps = denom.item()
        total_steps += steps
        total_loss += loss.item() * steps
        total_bc += bc_loss.item() * steps
        total_value += value_loss.item() * steps

        m = _sequence_metrics(logits, masks, actions, valid)
        for key in metrics:
            metrics[key] += m[key]

    if total_steps <= 0:
        return None
    return {
        "loss": total_loss / total_steps,
        "bc_loss": total_bc / total_steps,
        "value_loss": total_value / total_steps,
        "acc": metrics["correct"] / max(1, metrics["total"]),
        "move_acc": metrics["move_correct"] / max(1, metrics["move_total"]),
        "switch_acc": metrics["switch_correct"] / max(1, metrics["switch_total"]),
    }


@torch.no_grad()
def eval_epoch(
    model: RecurrentActorCritic,
    loader: DataLoader,
    device: torch.device,
):
    model.eval()
    total_nll = 0.0
    total_steps = 0.0
    metrics = {"total": 0, "correct": 0, "move_total": 0, "move_correct": 0, "switch_total": 0, "switch_correct": 0}

    for batch in loader:
        features, masks, actions, _, valid, _ = _move_batch(batch, device)
        logits, _, _ = model.forward_sequence(features)
        masked_logits = logits.masked_fill(~masks, -1e9)
        log_probs = F.log_softmax(masked_logits, dim=-1)
        nll = -log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)

        valid_f = valid.float()
        denom = valid_f.sum().clamp(min=1.0)
        total_nll += (nll * valid_f).sum().item()
        total_steps += denom.item()

        m = _sequence_metrics(logits, masks, actions, valid)
        for key in metrics:
            metrics[key] += m[key]

    if total_steps <= 0:
        return None
    return {
        "nll": total_nll / total_steps,
        "acc": metrics["correct"] / max(1, metrics["total"]),
        "move_acc": metrics["move_correct"] / max(1, metrics["move_total"]),
        "switch_acc": metrics["switch_correct"] / max(1, metrics["switch_total"]),
    }


def _load_trajectories(path: str) -> list[dict]:
    with open(path, "rb") as handle:
        obj = pickle.load(handle)
    if not isinstance(obj, list):
        raise ValueError(f"Expected list trajectory pickle: {path}")
    return obj


def main() -> int:
    parser = argparse.ArgumentParser(description="Train sequence BC policy on trajectories.")
    parser.add_argument("--input", required=True, help="Training trajectories pickle")
    parser.add_argument("--val-input", default="", help="Validation trajectories pickle")
    parser.add_argument("--output", default="checkpoints/rl/sequence_bc.pt")
    parser.add_argument("--summary-out", default="")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto|cpu|cuda")
    parser.add_argument("--feature-dim", type=int, default=272)
    parser.add_argument("--n-actions", type=int, default=13)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--rnn-hidden", type=int, default=512)
    parser.add_argument("--rnn-layers", type=int, default=1)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--min-actions", type=int, default=4)
    parser.add_argument("--value-coef", type=float, default=0.10)
    parser.add_argument("--switch-action-weight", type=float, default=1.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--weight-by-rating", action="store_true")
    parser.add_argument("--rating-baseline", type=float, default=1400.0)
    parser.add_argument("--rating-scale", type=float, default=1000.0)
    parser.add_argument("--max-rating-weight", type=float, default=2.5)
    args = parser.parse_args()

    set_seed(args.seed)
    device = _resolve_device(args.device)

    train_trajs = _load_trajectories(args.input)
    if args.val_input:
        val_trajs = _load_trajectories(args.val_input)
    else:
        train_trajs, val_trajs = stable_split_by_battle(
            train_trajs, holdout_ratio=args.val_split, seed=args.seed
        )
        # Fallback for datasets missing/degenerate battle_id values that collapse
        # hash-based split into an empty holdout.
        if not val_trajs and train_trajs and args.val_split > 0:
            rng = random.Random(args.seed)
            idx = list(range(len(train_trajs)))
            rng.shuffle(idx)
            val_n = max(1, int(len(train_trajs) * args.val_split))
            val_ids = set(idx[:val_n])
            val_trajs = [t for i, t in enumerate(train_trajs) if i in val_ids]
            train_trajs = [t for i, t in enumerate(train_trajs) if i not in val_ids]
            print(
                f"[warn] battle-id split produced empty holdout; "
                f"fallback random split used (train={len(train_trajs)} val={len(val_trajs)})"
            )

    train_ds = TrajectoryDataset(
        train_trajs,
        gamma=args.gamma,
        min_actions=args.min_actions,
        weight_by_rating=args.weight_by_rating,
        rating_baseline=args.rating_baseline,
        rating_scale=args.rating_scale,
        max_rating_weight=args.max_rating_weight,
    )
    val_ds = TrajectoryDataset(
        val_trajs,
        gamma=args.gamma,
        min_actions=args.min_actions,
        weight_by_rating=False,
    )
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise SystemExit(
            f"Empty dataset after filtering (train={len(train_ds)} val={len(val_ds)})."
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_trajectories,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_trajectories,
    )

    model = RecurrentActorCritic(
        feature_dim=args.feature_dim,
        d_model=args.d_model,
        n_actions=args.n_actions,
        rnn_hidden=args.rnn_hidden,
        rnn_layers=args.rnn_layers,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = -1.0
    best_epoch = 0
    history: list[dict] = []

    print(
        f"Training sequence BC on {len(train_ds)} train / {len(val_ds)} val trajectories"
    )
    print(f"Device: {device}")
    for epoch in range(1, args.epochs + 1):
        train_stats = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            value_coef=args.value_coef,
            switch_action_weight=args.switch_action_weight,
            grad_clip=args.grad_clip,
        )
        if train_stats is None:
            raise SystemExit("No valid steps during training epoch.")

        val_stats = eval_epoch(model, val_loader, device)
        if val_stats is None:
            raise SystemExit("No valid steps during validation epoch.")

        scheduler.step()
        row = {"epoch": epoch, "train": train_stats, "val": val_stats}
        history.append(row)
        print(
            f"Epoch {epoch:03d} "
            f"train_loss={train_stats['loss']:.4f} "
            f"train_acc={train_stats['acc']:.3f} "
            f"val_nll={val_stats['nll']:.4f} "
            f"val_acc={val_stats['acc']:.3f} "
            f"val_move={val_stats['move_acc']:.3f} "
            f"val_switch={val_stats['switch_acc']:.3f}"
        )

        if val_stats["acc"] > best_val_acc:
            best_val_acc = val_stats["acc"]
            best_epoch = epoch
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "val_acc": float(val_stats["acc"]),
                    "model_type": "recurrent",
                    "config": {
                        "feature_dim": args.feature_dim,
                        "d_model": args.d_model,
                        "n_actions": args.n_actions,
                        "rnn_hidden": args.rnn_hidden,
                        "rnn_layers": args.rnn_layers,
                        "prediction_features_enabled": False,
                        # Keep sequence checkpoints "policy-only" at inference time.
                        "switch_bias_enabled": False,
                        "switch_stay_penalty_strength": 0.0,
                        "attack_eff_penalty_enabled": False,
                    },
                },
                out_path,
            )

    summary = {
        "input": args.input,
        "val_input": args.val_input or "<split>",
        "output": args.output,
        "device": str(device),
        "train_trajectories": len(train_ds),
        "val_trajectories": len(val_ds),
        "epochs": args.epochs,
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "history": history,
    }
    summary_path = (
        Path(args.summary_out)
        if args.summary_out
        else Path(args.output).with_suffix(".summary.json")
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Best val acc: {best_val_acc:.4f} @ epoch {best_epoch}")
    print(f"Model saved to {args.output}")
    print(f"Summary -> {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
