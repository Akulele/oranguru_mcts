#!/usr/bin/env python3
"""
Offline RL fine-tuning over trajectory data.

Uses conservative actor updates:
- binary mode: keep actions with positive estimated advantage
- exp mode: AWBC-style exponential weighting
"""

from __future__ import annotations

import argparse
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

from src.models.actor_critic import ActorCritic, RecurrentActorCritic
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
    return model, ckpt


def _flatten_valid(tensor, valid):
    return tensor[valid]


def _compute_forward(model, features, masks):
    if getattr(model, "is_recurrent", False):
        logits, values, _ = model.forward_sequence(features)
    else:
        b, t, d = features.shape
        fa = features.reshape(b * t, d)
        ma = masks.reshape(b * t, masks.shape[-1])
        logits_flat, values_flat = model.forward(fa, ma)
        logits = logits_flat.reshape(b, t, -1)
        values = values_flat.reshape(b, t)
    return logits, values


def _metrics_from_logits(logits, masks, actions, valid):
    masked_logits = logits.masked_fill(~masks, -1e9)
    preds = masked_logits.argmax(dim=-1)
    total = valid.sum().item()
    correct = ((preds == actions) & valid).sum().item()

    move_valid = valid & ((actions < 4) | (actions >= 9))
    switch_valid = valid & ((actions >= 4) & (actions < 9))
    move_total = move_valid.sum().item()
    switch_total = switch_valid.sum().item()
    move_correct = ((preds == actions) & move_valid).sum().item()
    switch_correct = ((preds == actions) & switch_valid).sum().item()

    return {
        "acc": correct / max(1, total),
        "move_acc": move_correct / max(1, move_total),
        "switch_acc": switch_correct / max(1, switch_total),
    }


def train_epoch(
    model,
    loader: DataLoader,
    optimizer,
    device: torch.device,
    mode: str,
    beta: float,
    max_aw: float,
    value_coef: float,
    entropy_coef: float,
    illegal_action_coef: float,
    grad_clip: float,
):
    model.train()
    total_steps = 0.0
    sums = {
        "loss": 0.0,
        "bc": 0.0,
        "value": 0.0,
        "entropy": 0.0,
        "illegal": 0.0,
    }
    metric_numer = {"acc": 0.0, "move_acc": 0.0, "switch_acc": 0.0}
    metric_count = 0

    for batch in loader:
        features = batch.features.to(device)
        masks = batch.masks.to(device)
        actions = batch.actions.to(device)
        returns = batch.returns.to(device)
        valid = batch.valid.to(device)
        weights = batch.weights.to(device)

        optimizer.zero_grad()
        logits, values = _compute_forward(model, features, masks)

        masked_logits = logits.masked_fill(~masks, -1e9)
        log_probs = F.log_softmax(masked_logits, dim=-1)
        nll = -log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)

        valid_f = valid.float()
        denom = valid_f.sum().clamp(min=1.0)

        with torch.no_grad():
            advantages = returns - values
            if mode == "binary":
                aw = (advantages > 0).float()
            else:
                aw = torch.exp(beta * advantages).clamp(max=max_aw)

        bc_loss = (nll * aw * weights * valid_f).sum() / denom
        value_loss = ((values - returns) ** 2 * valid_f).sum() / denom

        probs_masked = F.softmax(masked_logits, dim=-1)
        entropy = (-(probs_masked * log_probs).sum(dim=-1) * valid_f).sum() / denom

        probs_raw = F.softmax(logits, dim=-1)
        illegal_mass = (probs_raw * (~masks).float()).sum(dim=-1)
        illegal_penalty = (illegal_mass * valid_f).sum() / denom

        loss = (
            bc_loss
            + value_coef * value_loss
            - entropy_coef * entropy
            + illegal_action_coef * illegal_penalty
        )
        if torch.isnan(loss) or torch.isinf(loss):
            continue

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        steps = denom.item()
        total_steps += steps
        sums["loss"] += loss.item() * steps
        sums["bc"] += bc_loss.item() * steps
        sums["value"] += value_loss.item() * steps
        sums["entropy"] += entropy.item() * steps
        sums["illegal"] += illegal_penalty.item() * steps

        m = _metrics_from_logits(logits, masks, actions, valid)
        for k in metric_numer:
            metric_numer[k] += m[k]
        metric_count += 1

    if total_steps <= 0:
        return None
    out = {k: v / total_steps for k, v in sums.items()}
    if metric_count > 0:
        out["acc"] = metric_numer["acc"] / metric_count
        out["move_acc"] = metric_numer["move_acc"] / metric_count
        out["switch_acc"] = metric_numer["switch_acc"] / metric_count
    else:
        out["acc"] = out["move_acc"] = out["switch_acc"] = 0.0
    return out


@torch.no_grad()
def eval_epoch(model, loader: DataLoader, device: torch.device):
    model.eval()
    total_steps = 0.0
    total_nll = 0.0
    metric_numer = {"acc": 0.0, "move_acc": 0.0, "switch_acc": 0.0}
    metric_count = 0

    for batch in loader:
        features = batch.features.to(device)
        masks = batch.masks.to(device)
        actions = batch.actions.to(device)
        valid = batch.valid.to(device)

        logits, _ = _compute_forward(model, features, masks)
        masked_logits = logits.masked_fill(~masks, -1e9)
        log_probs = F.log_softmax(masked_logits, dim=-1)
        nll = -log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)

        valid_f = valid.float()
        denom = valid_f.sum().clamp(min=1.0)
        total_steps += denom.item()
        total_nll += (nll * valid_f).sum().item()

        m = _metrics_from_logits(logits, masks, actions, valid)
        for k in metric_numer:
            metric_numer[k] += m[k]
        metric_count += 1

    if total_steps <= 0:
        return None
    out = {"nll": total_nll / total_steps}
    if metric_count > 0:
        out["acc"] = metric_numer["acc"] / metric_count
        out["move_acc"] = metric_numer["move_acc"] / metric_count
        out["switch_acc"] = metric_numer["switch_acc"] / metric_count
    else:
        out["acc"] = out["move_acc"] = out["switch_acc"] = 0.0
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline RL fine-tune from trajectory data.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--val-input", default="")
    parser.add_argument("--checkpoint-in", required=True)
    parser.add_argument("--output", default="checkpoints/rl/offline_finetune.pt")
    parser.add_argument("--summary-out", default="")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--min-actions", type=int, default=4)
    parser.add_argument("--mode", choices=["binary", "exp"], default="binary")
    parser.add_argument("--beta", type=float, default=3.0)
    parser.add_argument("--max-aw", type=float, default=20.0)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.001)
    parser.add_argument("--illegal-action-coef", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--weight-by-rating", action="store_true")
    parser.add_argument("--rating-baseline", type=float, default=1400.0)
    parser.add_argument("--rating-scale", type=float, default=1000.0)
    parser.add_argument("--max-rating-weight", type=float, default=2.5)
    args = parser.parse_args()

    set_seed(args.seed)
    device = _resolve_device(args.device)
    model, checkpoint = _load_model(args.checkpoint_in, device)

    train_trajs = _load_trajectories(args.input)
    if args.val_input:
        val_trajs = _load_trajectories(args.val_input)
    else:
        train_trajs, val_trajs = stable_split_by_battle(
            train_trajs, holdout_ratio=args.val_split, seed=args.seed
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

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = -1.0
    best_epoch = 0
    history: list[dict] = []
    print(
        f"Offline RL finetune ({args.mode}) on {len(train_ds)} train / {len(val_ds)} val trajectories"
    )
    print(f"Checkpoint: {args.checkpoint_in}")
    print(f"Device: {device}")

    for epoch in range(1, args.epochs + 1):
        train_stats = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            mode=args.mode,
            beta=args.beta,
            max_aw=args.max_aw,
            value_coef=args.value_coef,
            entropy_coef=args.entropy_coef,
            illegal_action_coef=args.illegal_action_coef,
            grad_clip=args.grad_clip,
        )
        if train_stats is None:
            raise SystemExit("No valid steps during train epoch.")

        val_stats = eval_epoch(model, val_loader, device)
        if val_stats is None:
            raise SystemExit("No valid steps during validation epoch.")

        scheduler.step()
        history.append({"epoch": epoch, "train": train_stats, "val": val_stats})
        print(
            f"Epoch {epoch:03d} "
            f"train_loss={train_stats['loss']:.4f} "
            f"train_bc={train_stats['bc']:.4f} "
            f"train_val={train_stats['value']:.4f} "
            f"train_acc={train_stats['acc']:.3f} "
            f"val_nll={val_stats['nll']:.4f} "
            f"val_acc={val_stats['acc']:.3f}"
        )

        if val_stats["acc"] > best_val_acc:
            best_val_acc = val_stats["acc"]
            best_epoch = epoch
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_cfg = dict((checkpoint.get("config", {}) or {}))
            out_cfg.setdefault("switch_bias_enabled", False)
            out_cfg.setdefault("switch_stay_penalty_strength", 0.0)
            out_cfg.setdefault("attack_eff_penalty_enabled", False)
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "val_acc": float(val_stats["acc"]),
                    "model_type": checkpoint.get("model_type", "recurrent"),
                    "config": out_cfg,
                    "offline_rl": {
                        "mode": args.mode,
                        "beta": args.beta,
                        "max_aw": args.max_aw,
                        "value_coef": args.value_coef,
                        "entropy_coef": args.entropy_coef,
                        "illegal_action_coef": args.illegal_action_coef,
                    },
                },
                out_path,
            )

    summary = {
        "input": args.input,
        "val_input": args.val_input or "<split>",
        "checkpoint_in": args.checkpoint_in,
        "output": args.output,
        "device": str(device),
        "train_trajectories": len(train_ds),
        "val_trajectories": len(val_ds),
        "mode": args.mode,
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
