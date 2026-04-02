#!/usr/bin/env python3
"""
Train an action-conditioned prior/value model for MCTS assistance.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.search_prior_value import SearchPriorValueNet
from training.search_assist_utils import (
    SearchAssistDataset,
    collate_search_assist,
    load_search_assist_examples,
)
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


def _load_split(train_path: str, val_path: str, val_split: float, seed: int) -> tuple[list[dict], list[dict]]:
    train_examples = load_search_assist_examples(train_path)
    if val_path:
        return train_examples, load_search_assist_examples(val_path)
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


def _policy_metrics(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> dict:
    masked_logits = logits.masked_fill(~mask, -1e9)
    pred = masked_logits.argmax(dim=-1)
    gold = target.argmax(dim=-1)
    correct = (pred == gold).sum().item()
    return {"correct": correct, "total": logits.shape[0]}


def train_epoch(model, loader, optimizer, device, value_coef: float):
    model.train()
    total_loss = 0.0
    total_policy = 0.0
    total_value = 0.0
    total_weight = 0.0
    metrics = {"correct": 0, "total": 0}

    for batch in loader:
        board = batch.board_features.to(device)
        action = batch.action_features.to(device)
        mask = batch.action_mask.to(device)
        target_policy = batch.policy_target.to(device)
        target_value = batch.value_target.to(device)
        weights = batch.weights.to(device)

        optimizer.zero_grad()
        logits, values = model(board, action, mask)
        log_probs = F.log_softmax(logits, dim=-1)
        policy_loss = -(target_policy * log_probs).sum(dim=-1)
        value_loss = (values - target_value).pow(2)
        loss = ((policy_loss + value_coef * value_loss) * weights).sum() / weights.sum().clamp(min=1.0)
        if torch.isnan(loss) or torch.isinf(loss):
            continue
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        batch_weight = weights.sum().item()
        total_weight += batch_weight
        total_loss += loss.item() * batch_weight
        total_policy += (policy_loss * weights).sum().item()
        total_value += (value_loss * weights).sum().item()
        m = _policy_metrics(logits, target_policy, mask)
        metrics["correct"] += m["correct"]
        metrics["total"] += m["total"]

    if total_weight <= 0:
        return None
    return {
        "loss": total_loss / total_weight,
        "policy_loss": total_policy / total_weight,
        "value_loss": total_value / total_weight,
        "acc": metrics["correct"] / max(1, metrics["total"]),
    }


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    total_policy = 0.0
    total_value = 0.0
    total_weight = 0.0
    metrics = {"correct": 0, "total": 0}

    for batch in loader:
        board = batch.board_features.to(device)
        action = batch.action_features.to(device)
        mask = batch.action_mask.to(device)
        target_policy = batch.policy_target.to(device)
        target_value = batch.value_target.to(device)
        weights = batch.weights.to(device)

        logits, values = model(board, action, mask)
        log_probs = F.log_softmax(logits, dim=-1)
        policy_loss = -(target_policy * log_probs).sum(dim=-1)
        value_loss = (values - target_value).pow(2)

        batch_weight = weights.sum().item()
        total_weight += batch_weight
        total_policy += (policy_loss * weights).sum().item()
        total_value += (value_loss * weights).sum().item()
        m = _policy_metrics(logits, target_policy, mask)
        metrics["correct"] += m["correct"]
        metrics["total"] += m["total"]

    if total_weight <= 0:
        return None
    return {
        "policy_loss": total_policy / total_weight,
        "value_loss": total_value / total_weight,
        "acc": metrics["correct"] / max(1, metrics["total"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Train prior/value model for MCTS assistance.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--val-input", default="")
    parser.add_argument("--checkpoint-in", default="")
    parser.add_argument("--output", default="checkpoints/rl/search_prior_value.pt")
    parser.add_argument("--summary-out", default="")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--value-coef", type=float, default=0.25)
    parser.add_argument("--policy-only", action="store_true")
    parser.add_argument("--value-target-field", default="value_target")
    parser.add_argument("--min-visits", type=float, default=1.0)
    parser.add_argument("--rating-weight", type=float, default=0.0)
    parser.add_argument("--rating-baseline", type=float, default=1500.0)
    parser.add_argument("--rating-scale", type=float, default=1000.0)
    args = parser.parse_args()

    _set_seed(args.seed)
    device = _resolve_device(args.device)
    if args.policy_only:
        args.value_coef = 0.0

    train_examples, val_examples = _load_split(args.input, args.val_input, args.val_split, args.seed)
    train_ds = SearchAssistDataset(
        train_examples,
        min_visits=args.min_visits,
        rating_weight=args.rating_weight,
        rating_baseline=args.rating_baseline,
        rating_scale=args.rating_scale,
        value_target_field=args.value_target_field,
    )
    val_ds = SearchAssistDataset(
        val_examples,
        min_visits=args.min_visits,
        rating_weight=0.0,
        value_target_field=args.value_target_field,
    )
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise SystemExit(f"Empty dataset after filtering (train={len(train_ds)} val={len(val_ds)})")

    sample = train_ds[0]
    board_dim = len(sample["board_features"])
    action_dim = len(sample["action_features"][0])
    n_actions = len(sample["action_mask"])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_search_assist)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_search_assist)

    model = SearchPriorValueNet(
        board_dim=board_dim,
        action_dim=action_dim,
        hidden_dim=args.hidden_dim,
        n_actions=n_actions,
        dropout=args.dropout,
    ).to(device)
    checkpoint_meta = None
    if args.checkpoint_in:
        ckpt = torch.load(args.checkpoint_in, map_location=device, weights_only=False)
        checkpoint_meta = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
        if checkpoint_meta:
            ckpt_board_dim = int(checkpoint_meta.get("board_dim", board_dim))
            ckpt_action_dim = int(checkpoint_meta.get("action_dim", action_dim))
            ckpt_n_actions = int(checkpoint_meta.get("n_actions", n_actions))
            if ckpt_board_dim != board_dim or ckpt_action_dim != action_dim or ckpt_n_actions != n_actions:
                raise SystemExit(
                    "Checkpoint dims do not match dataset: "
                    f"ckpt=({ckpt_board_dim},{ckpt_action_dim},{ckpt_n_actions}) "
                    f"data=({board_dim},{action_dim},{n_actions})"
                )
        state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        model.load_state_dict(state_dict)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print(f"Training search prior/value on {len(train_ds)} train / {len(val_ds)} val examples")
    print(f"Device: {device}")
    print(f"Board dim: {board_dim} | Action dim: {action_dim} | Actions: {n_actions}")
    if args.checkpoint_in:
        print(f"Initialized from checkpoint: {args.checkpoint_in}")

    best = None
    history = []
    for epoch in range(1, args.epochs + 1):
        train_stats = train_epoch(model, train_loader, optimizer, device, value_coef=args.value_coef)
        val_stats = eval_epoch(model, val_loader, device)
        if train_stats is None or val_stats is None:
            continue
        row = {
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "train_policy": train_stats["policy_loss"],
            "train_value": train_stats["value_loss"],
            "train_acc": train_stats["acc"],
            "val_policy": val_stats["policy_loss"],
            "val_value": val_stats["value_loss"],
            "val_acc": val_stats["acc"],
        }
        history.append(row)
        print(
            "Epoch {:03d} train_loss={:.4f} train_acc={:.3f} val_policy={:.4f} val_value={:.4f} val_acc={:.3f}".format(
                epoch,
                row["train_loss"],
                row["train_acc"],
                row["val_policy"],
                row["val_value"],
                row["val_acc"],
            )
        )
        if best is None or row["val_acc"] > best["val_acc"]:
            best = dict(row)
            ckpt = {
                "model": model.state_dict(),
                "config": {
                    "board_dim": board_dim,
                    "action_dim": action_dim,
                    "n_actions": n_actions,
                    "hidden_dim": args.hidden_dim,
                "dropout": args.dropout,
                "value_coef": args.value_coef,
                "value_target_field": args.value_target_field,
                "policy_only": bool(args.policy_only),
                "checkpoint_in": args.checkpoint_in or None,
            },
        }
            out = Path(args.output)
            out.parent.mkdir(parents=True, exist_ok=True)
            torch.save(ckpt, out)

    if best is None:
        raise SystemExit("Training produced no valid epochs.")

    print("Best val acc: {:.4f} @ epoch {}".format(best["val_acc"], best["epoch"]))
    print(f"Model saved to {args.output}")

    if args.summary_out:
        payload = {
            "best": best,
            "history": history,
            "train_examples": len(train_ds),
            "val_examples": len(val_ds),
            "input": args.input,
            "val_input": args.val_input or None,
            "checkpoint_in": args.checkpoint_in or None,
            "output": args.output,
            "value_target_field": args.value_target_field,
            "policy_only": bool(args.policy_only),
            "value_coef": args.value_coef,
        }
        path = Path(args.summary_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Summary -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
