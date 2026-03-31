#!/usr/bin/env python3
"""
Offline evaluation for leaf-value checkpoints.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.leaf_value import LeafValueNet
from training.train_leaf_value import (
    LeafValueDataset,
    _load_split,
    collate_leaf_value,
    eval_epoch,
)


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline evaluation for a leaf-value checkpoint.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--val-input", default="")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--summary-out", default="")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--split", choices=("train", "val", "all"), default="val")
    args = parser.parse_args()

    _set_seed(args.seed)
    device = _resolve_device(args.device)

    train_examples, val_examples = _load_split(args.input, args.val_input, args.val_split, args.seed)
    if args.split == "train":
        rows = train_examples
    elif args.split == "all":
        rows = train_examples + val_examples
    else:
        rows = val_examples

    dataset = LeafValueDataset(rows)
    if len(dataset) == 0:
        raise SystemExit("Empty evaluation dataset after filtering.")

    sample = dataset[0]
    board_dim = len(sample["board_features"])
    extra_dim = len(sample["extra_features"])

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    ckpt_board_dim = int(config.get("board_dim", board_dim))
    ckpt_extra_dim = int(config.get("extra_dim", extra_dim))
    if ckpt_board_dim != board_dim or ckpt_extra_dim != extra_dim:
        raise SystemExit(
            "Checkpoint dims do not match dataset: "
            f"ckpt=({ckpt_board_dim},{ckpt_extra_dim}) data=({board_dim},{extra_dim})"
        )

    model = LeafValueNet(
        board_dim=board_dim,
        extra_dim=extra_dim,
        hidden_dim=int(config.get("hidden_dim", 256)),
        dropout=float(config.get("dropout", 0.1)),
    ).to(device)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict)

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_leaf_value)
    metrics = eval_epoch(model, loader, device)
    if metrics is None:
        raise SystemExit("Evaluation produced no usable metrics.")

    print(f"Evaluated leaf value model on {len(dataset)} rows")
    print(f"Split: {args.split} | Device: {device}")
    print(
        "MAE={:.4f} RMSE={:.4f} sign={:.3f} corr={:.4f} pred_mean={:.4f} target_mean={:.4f}".format(
            metrics["mae"],
            metrics["rmse"],
            metrics["sign_acc"],
            metrics["corr"],
            metrics["pred_mean"],
            metrics["target_mean"],
        )
    )

    if args.summary_out:
        payload = {
            "input": args.input,
            "val_input": args.val_input or None,
            "checkpoint": args.checkpoint,
            "split": args.split,
            "rows": len(dataset),
            "metrics": metrics,
        }
        path = Path(args.summary_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Summary -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
