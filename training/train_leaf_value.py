#!/usr/bin/env python3
"""
Train a compact leaf-value model from traced search states.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
import random
import sys
from typing import Iterable

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.leaf_value import LeafValueNet
from training.search_assist_utils import safe_float
from training.sequence_utils import stable_split_by_battle


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_examples(path: str) -> list[dict]:
    with Path(path).open("rb") as handle:
        obj = pickle.load(handle)
    if not isinstance(obj, list):
        raise SystemExit(f"Expected list dataset in {path}")
    return [row for row in obj if isinstance(row, dict)]


def _load_split(train_path: str, val_path: str, val_split: float, seed: int) -> tuple[list[dict], list[dict]]:
    train_examples = _load_examples(train_path)
    if val_path:
        return train_examples, _load_examples(val_path)
    train_examples, val_examples = stable_split_by_battle(
        train_examples,
        holdout_ratio=val_split,
        seed=seed,
    )
    if not val_examples and train_examples and val_split > 0:
        rng = random.Random(seed)
        idx = list(range(len(train_examples)))
        rng.shuffle(idx)
        val_n = max(1, int(len(train_examples) * val_split))
        val_ids = set(idx[:val_n])
        val_examples = [t for i, t in enumerate(train_examples) if i in val_ids]
        train_examples = [t for i, t in enumerate(train_examples) if i not in val_ids]
    return train_examples, val_examples


@dataclass
class LeafValueBatch:
    board_features: torch.Tensor
    extra_features: torch.Tensor
    value_target: torch.Tensor
    weights: torch.Tensor


class LeafValueDataset(Dataset):
    def __init__(self, rows: Iterable[dict]):
        self.items: list[dict] = []
        for row in rows:
            board = row.get("board_features")
            extra = row.get("extra_features")
            if not isinstance(board, list) or not board:
                continue
            if not isinstance(extra, list) or not extra:
                continue

            value_target = safe_float(row.get("value_target", 0.0), 0.0)
            if not math.isfinite(value_target):
                continue
            value_target = max(-1.0, min(1.0, value_target))

            weight = safe_float(row.get("weight", 1.0), 1.0)
            if not math.isfinite(weight) or weight <= 0:
                weight = 1.0

            self.items.append(
                {
                    "board_features": [safe_float(v, 0.0) for v in board],
                    "extra_features": [safe_float(v, 0.0) for v in extra],
                    "value_target": value_target,
                    "weight": weight,
                }
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        return self.items[idx]


def collate_leaf_value(batch: list[dict]) -> LeafValueBatch:
    board = torch.tensor([item["board_features"] for item in batch], dtype=torch.float32)
    extra = torch.tensor([item["extra_features"] for item in batch], dtype=torch.float32)
    value = torch.tensor([item["value_target"] for item in batch], dtype=torch.float32)
    weights = torch.tensor([item["weight"] for item in batch], dtype=torch.float32)
    return LeafValueBatch(
        board_features=board,
        extra_features=extra,
        value_target=value,
        weights=weights,
    )


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_mae = 0.0
    total_mse = 0.0
    total_sign = 0.0
    total_weight = 0.0

    for batch in loader:
        board = batch.board_features.to(device)
        extra = batch.extra_features.to(device)
        target = batch.value_target.to(device)
        weights = batch.weights.to(device)

        optimizer.zero_grad()
        pred = model(board, extra)
        mse = (pred - target).pow(2)
        loss = (mse * weights).sum() / weights.sum().clamp(min=1.0)
        if torch.isnan(loss) or torch.isinf(loss):
            continue
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        batch_weight = weights.sum().item()
        total_weight += batch_weight
        total_loss += loss.item() * batch_weight
        total_mae += ((pred - target).abs() * weights).sum().item()
        total_mse += (mse * weights).sum().item()
        sign_ok = ((pred >= 0) == (target >= 0)).float()
        total_sign += (sign_ok * weights).sum().item()

    if total_weight <= 0:
        return None
    return {
        "loss": total_loss / total_weight,
        "mae": total_mae / total_weight,
        "mse": total_mse / total_weight,
        "sign_acc": total_sign / total_weight,
    }


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    total_mae = 0.0
    total_mse = 0.0
    total_sign = 0.0
    total_weight = 0.0
    pred_sum = 0.0
    target_sum = 0.0
    pred_sq_sum = 0.0
    target_sq_sum = 0.0
    cross_sum = 0.0

    for batch in loader:
        board = batch.board_features.to(device)
        extra = batch.extra_features.to(device)
        target = batch.value_target.to(device)
        weights = batch.weights.to(device)

        pred = model(board, extra)
        mse = (pred - target).pow(2)

        batch_weight = weights.sum().item()
        total_weight += batch_weight
        total_mae += ((pred - target).abs() * weights).sum().item()
        total_mse += (mse * weights).sum().item()
        sign_ok = ((pred >= 0) == (target >= 0)).float()
        total_sign += (sign_ok * weights).sum().item()

        pred_sum += (pred * weights).sum().item()
        target_sum += (target * weights).sum().item()
        pred_sq_sum += ((pred * pred) * weights).sum().item()
        target_sq_sum += ((target * target) * weights).sum().item()
        cross_sum += ((pred * target) * weights).sum().item()

    if total_weight <= 0:
        return None

    pred_mean = pred_sum / total_weight
    target_mean = target_sum / total_weight
    pred_var = max(0.0, pred_sq_sum / total_weight - pred_mean * pred_mean)
    target_var = max(0.0, target_sq_sum / total_weight - target_mean * target_mean)
    cov = cross_sum / total_weight - pred_mean * target_mean
    corr = cov / math.sqrt(max(1e-12, pred_var * target_var))

    return {
        "mae": total_mae / total_weight,
        "mse": total_mse / total_weight,
        "rmse": math.sqrt(max(0.0, total_mse / total_weight)),
        "sign_acc": total_sign / total_weight,
        "corr": corr,
        "pred_mean": pred_mean,
        "target_mean": target_mean,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a leaf-value model.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--val-input", default="")
    parser.add_argument("--checkpoint-in", default="")
    parser.add_argument("--output", default="checkpoints/rl/leaf_value.pt")
    parser.add_argument("--summary-out", default="")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    args = parser.parse_args()

    _set_seed(args.seed)
    device = _resolve_device(args.device)

    train_examples, val_examples = _load_split(args.input, args.val_input, args.val_split, args.seed)
    train_ds = LeafValueDataset(train_examples)
    val_ds = LeafValueDataset(val_examples)
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise SystemExit(f"Empty dataset after filtering (train={len(train_ds)} val={len(val_ds)})")

    sample = train_ds[0]
    board_dim = len(sample["board_features"])
    extra_dim = len(sample["extra_features"])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_leaf_value)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_leaf_value)

    model = LeafValueNet(
        board_dim=board_dim,
        extra_dim=extra_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)

    if args.checkpoint_in:
        ckpt = torch.load(args.checkpoint_in, map_location=device, weights_only=False)
        ckpt_config = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
        if ckpt_config:
            ckpt_board_dim = int(ckpt_config.get("board_dim", board_dim))
            ckpt_extra_dim = int(ckpt_config.get("extra_dim", extra_dim))
            if ckpt_board_dim != board_dim or ckpt_extra_dim != extra_dim:
                raise SystemExit(
                    "Checkpoint dims do not match dataset: "
                    f"ckpt=({ckpt_board_dim},{ckpt_extra_dim}) data=({board_dim},{extra_dim})"
                )
        state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        model.load_state_dict(state_dict)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print(f"Training leaf value model on {len(train_ds)} train / {len(val_ds)} val rows")
    print(f"Device: {device}")
    print(f"Board dim: {board_dim} | Extra dim: {extra_dim}")
    if args.checkpoint_in:
        print(f"Initialized from checkpoint: {args.checkpoint_in}")

    best = None
    history = []
    for epoch in range(1, args.epochs + 1):
        train_stats = train_epoch(model, train_loader, optimizer, device)
        val_stats = eval_epoch(model, val_loader, device)
        if train_stats is None or val_stats is None:
            continue
        history.append(
            {
                "epoch": epoch,
                "train": train_stats,
                "val": val_stats,
            }
        )
        print(
            "Epoch {:03d} train_loss={:.4f} train_mae={:.4f} train_sign={:.3f} "
            "val_mae={:.4f} val_rmse={:.4f} val_sign={:.3f} val_corr={:.4f}".format(
                epoch,
                train_stats["loss"],
                train_stats["mae"],
                train_stats["sign_acc"],
                val_stats["mae"],
                val_stats["rmse"],
                val_stats["sign_acc"],
                val_stats["corr"],
            )
        )
        if best is None or val_stats["corr"] > best["val"]["corr"]:
            best = {
                "epoch": epoch,
                "train": train_stats,
                "val": val_stats,
                "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            }

    if best is None:
        raise SystemExit("Training produced no usable epoch.")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": best["state_dict"],
            "config": {
                "board_dim": board_dim,
                "extra_dim": extra_dim,
                "hidden_dim": args.hidden_dim,
                "dropout": args.dropout,
            },
            "best_epoch": best["epoch"],
            "best_val": best["val"],
            "history": history,
        },
        out_path,
    )
    print(f"Best val corr: {best['val']['corr']:.4f} @ epoch {best['epoch']}")
    print(f"Model saved to {out_path}")

    if args.summary_out:
        payload = {
            "input": args.input,
            "val_input": args.val_input or None,
            "output": str(out_path),
            "rows": {"train": len(train_ds), "val": len(val_ds)},
            "config": {
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "device": str(device),
                "seed": args.seed,
                "val_split": args.val_split,
                "hidden_dim": args.hidden_dim,
                "dropout": args.dropout,
                "board_dim": board_dim,
                "extra_dim": extra_dim,
            },
            "best_epoch": best["epoch"],
            "best_val": best["val"],
            "history": history,
        }
        summary_path = Path(args.summary_out)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Summary -> {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
