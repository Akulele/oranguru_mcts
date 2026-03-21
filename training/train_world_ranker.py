#!/usr/bin/env python3
"""
Train a compact scorer that ranks sampled hidden worlds before MCTS.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
from collections import defaultdict
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

from src.models.world_ranker import WorldRankerNet
from training.sequence_utils import stable_split_by_battle
from training.search_assist_utils import safe_float


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
class WorldRankBatch:
    board_features: torch.Tensor
    world_features: torch.Tensor
    target_score: torch.Tensor
    agreement_label: torch.Tensor
    weights: torch.Tensor
    parent_keys: list[str]


class WorldRankDataset(Dataset):
    def __init__(self, rows: Iterable[dict]):
        self.items: list[dict] = []
        for row in rows:
            board = row.get("board_features")
            world = row.get("world_features")
            if not isinstance(board, list) or not board:
                continue
            if not isinstance(world, list) or not world:
                continue

            target_score = safe_float(row.get("target_score", 0.0), 0.0)
            if not math.isfinite(target_score):
                continue
            target_score = max(0.0, min(1.0, target_score))

            weight = safe_float(row.get("sample_weight", 1.0), 1.0)
            if not math.isfinite(weight) or weight <= 0:
                weight = 1.0

            battle_id = str(row.get("battle_id", "unknown"))
            turn = int(row.get("turn", 0) or 0)
            self.items.append(
                {
                    "board_features": [safe_float(v, 0.0) for v in board],
                    "world_features": [safe_float(v, 0.0) for v in world],
                    "target_score": target_score,
                    "agreement_label": int(row.get("agreement_label", 0) or 0),
                    "weight": weight,
                    "parent_key": f"{battle_id}|{turn}",
                }
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        return self.items[idx]


def collate_world_rank(batch: list[dict]) -> WorldRankBatch:
    board = torch.tensor([item["board_features"] for item in batch], dtype=torch.float32)
    world = torch.tensor([item["world_features"] for item in batch], dtype=torch.float32)
    target = torch.tensor([item["target_score"] for item in batch], dtype=torch.float32)
    agree = torch.tensor([item["agreement_label"] for item in batch], dtype=torch.float32)
    weights = torch.tensor([item["weight"] for item in batch], dtype=torch.float32)
    parent_keys = [str(item["parent_key"]) for item in batch]
    return WorldRankBatch(
        board_features=board,
        world_features=world,
        target_score=target,
        agreement_label=agree,
        weights=weights,
        parent_keys=parent_keys,
    )


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_mse = 0.0
    total_bce = 0.0
    total_mae = 0.0
    total_weight = 0.0

    for batch in loader:
        board = batch.board_features.to(device)
        world = batch.world_features.to(device)
        target = batch.target_score.to(device)
        agree = batch.agreement_label.to(device)
        weights = batch.weights.to(device)

        optimizer.zero_grad()
        pred = model(board, world)
        mse = (pred - target).pow(2)
        bce = F.binary_cross_entropy(pred, agree, reduction="none")
        loss = ((mse + 0.25 * bce) * weights).sum() / weights.sum().clamp(min=1.0)
        if torch.isnan(loss) or torch.isinf(loss):
            continue
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        batch_weight = weights.sum().item()
        total_weight += batch_weight
        total_loss += loss.item() * batch_weight
        total_mse += (mse * weights).sum().item()
        total_bce += (bce * weights).sum().item()
        total_mae += ((pred - target).abs() * weights).sum().item()

    if total_weight <= 0:
        return None
    return {
        "loss": total_loss / total_weight,
        "mse": total_mse / total_weight,
        "bce": total_bce / total_weight,
        "mae": total_mae / total_weight,
    }


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    total_mse = 0.0
    total_bce = 0.0
    total_mae = 0.0
    total_weight = 0.0
    grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)

    for batch in loader:
        board = batch.board_features.to(device)
        world = batch.world_features.to(device)
        target = batch.target_score.to(device)
        agree = batch.agreement_label.to(device)
        weights = batch.weights.to(device)

        pred = model(board, world)
        mse = (pred - target).pow(2)
        bce = F.binary_cross_entropy(pred, agree, reduction="none")

        batch_weight = weights.sum().item()
        total_weight += batch_weight
        total_mse += (mse * weights).sum().item()
        total_bce += (bce * weights).sum().item()
        total_mae += ((pred - target).abs() * weights).sum().item()

        pred_cpu = pred.detach().cpu().tolist()
        target_cpu = target.detach().cpu().tolist()
        for parent_key, pred_value, target_value in zip(batch.parent_keys, pred_cpu, target_cpu):
            grouped[parent_key].append((float(pred_value), float(target_value)))

    if total_weight <= 0:
        return None

    parent_top1 = 0
    parent_count = 0
    parent_pick_target_sum = 0.0
    parent_oracle_target_sum = 0.0
    for rows in grouped.values():
        if not rows:
            continue
        parent_count += 1
        pred_idx = max(range(len(rows)), key=lambda i: rows[i][0])
        pred_target = rows[pred_idx][1]
        best_target = max(target for _, target in rows)
        parent_pick_target_sum += pred_target
        parent_oracle_target_sum += best_target
        if pred_target >= best_target - 1e-8:
            parent_top1 += 1

    parent_top1_acc = parent_top1 / max(1, parent_count)
    parent_pick_target_mean = parent_pick_target_sum / max(1, parent_count)
    parent_oracle_target_mean = parent_oracle_target_sum / max(1, parent_count)

    return {
        "mse": total_mse / total_weight,
        "bce": total_bce / total_weight,
        "mae": total_mae / total_weight,
        "parent_top1_acc": parent_top1_acc,
        "parent_pick_target_mean": parent_pick_target_mean,
        "parent_oracle_target_mean": parent_oracle_target_mean,
        "parent_gap": parent_oracle_target_mean - parent_pick_target_mean,
        "parents": parent_count,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a candidate-world ranker.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--val-input", default="")
    parser.add_argument("--checkpoint-in", default="")
    parser.add_argument("--output", default="checkpoints/rl/world_ranker.pt")
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
    train_ds = WorldRankDataset(train_examples)
    val_ds = WorldRankDataset(val_examples)
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise SystemExit(f"Empty dataset after filtering (train={len(train_ds)} val={len(val_ds)})")

    sample = train_ds[0]
    board_dim = len(sample["board_features"])
    world_dim = len(sample["world_features"])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_world_rank)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_world_rank)

    model = WorldRankerNet(
        board_dim=board_dim,
        world_dim=world_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)

    if args.checkpoint_in:
        ckpt = torch.load(args.checkpoint_in, map_location=device, weights_only=False)
        ckpt_config = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
        if ckpt_config:
            ckpt_board_dim = int(ckpt_config.get("board_dim", board_dim))
            ckpt_world_dim = int(ckpt_config.get("world_dim", world_dim))
            if ckpt_board_dim != board_dim or ckpt_world_dim != world_dim:
                raise SystemExit(
                    "Checkpoint dims do not match dataset: "
                    f"ckpt=({ckpt_board_dim},{ckpt_world_dim}) data=({board_dim},{world_dim})"
                )
        state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        model.load_state_dict(state_dict)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print(f"Training world ranker on {len(train_ds)} train / {len(val_ds)} val rows")
    print(f"Device: {device}")
    print(f"Board dim: {board_dim} | World dim: {world_dim}")
    if args.checkpoint_in:
        print(f"Initialized from checkpoint: {args.checkpoint_in}")

    best = None
    history = []
    for epoch in range(1, args.epochs + 1):
        train_stats = train_epoch(model, train_loader, optimizer, device)
        val_stats = eval_epoch(model, val_loader, device)
        if train_stats is None or val_stats is None:
            continue

        row = {
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "train_mse": train_stats["mse"],
            "train_bce": train_stats["bce"],
            "train_mae": train_stats["mae"],
            "val_mse": val_stats["mse"],
            "val_bce": val_stats["bce"],
            "val_mae": val_stats["mae"],
            "val_parent_top1_acc": val_stats["parent_top1_acc"],
            "val_parent_pick_target_mean": val_stats["parent_pick_target_mean"],
            "val_parent_oracle_target_mean": val_stats["parent_oracle_target_mean"],
            "val_parent_gap": val_stats["parent_gap"],
            "val_parents": val_stats["parents"],
        }
        history.append(row)
        print(
            "Epoch {:03d} train_loss={:.4f} train_mae={:.4f} val_mae={:.4f} "
            "val_top1={:.3f} val_pick={:.4f} val_oracle={:.4f}".format(
                epoch,
                row["train_loss"],
                row["train_mae"],
                row["val_mae"],
                row["val_parent_top1_acc"],
                row["val_parent_pick_target_mean"],
                row["val_parent_oracle_target_mean"],
            )
        )

        if best is None or row["val_parent_top1_acc"] > best["val_parent_top1_acc"]:
            best = dict(row)
            ckpt = {
                "model": model.state_dict(),
                "config": {
                    "board_dim": board_dim,
                    "world_dim": world_dim,
                    "hidden_dim": args.hidden_dim,
                    "dropout": args.dropout,
                    "checkpoint_in": args.checkpoint_in or None,
                },
            }
            out = Path(args.output)
            out.parent.mkdir(parents=True, exist_ok=True)
            torch.save(ckpt, out)

    if best is None:
        raise SystemExit("Training produced no valid epochs.")

    print(
        "Best val top1: {:.4f} @ epoch {}".format(
            best["val_parent_top1_acc"],
            best["epoch"],
        )
    )
    print(f"Model saved to {args.output}")

    if args.summary_out:
        payload = {
            "best": best,
            "history": history,
            "train_rows": len(train_ds),
            "val_rows": len(val_ds),
            "input": args.input,
            "val_input": args.val_input or None,
            "checkpoint_in": args.checkpoint_in or None,
            "output": args.output,
        }
        path = Path(args.summary_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Summary -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
