#!/usr/bin/env python3
"""
Offline evaluation for candidate-world ranking checkpoints.
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

from src.models.world_ranker import WorldRankerNet
from training.train_world_ranker import (
    WorldRankDataset,
    _load_split,
    collate_world_rank,
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


@torch.no_grad()
def _collect_rows(model, loader, device):
    model.eval()
    rows: list[tuple[str, float, float]] = []
    for batch in loader:
        board = batch.board_features.to(device)
        world = batch.world_features.to(device)
        pred = model(board, world).detach().cpu().tolist()
        target = batch.target_score.detach().cpu().tolist()
        for parent_key, pred_value, target_value in zip(batch.parent_keys, pred, target):
            rows.append((parent_key, float(pred_value), float(target_value)))
    return rows


def _summarize_parent_metrics(rows: list[tuple[str, float, float]], topk_values: list[int], seed: int) -> dict:
    grouped: dict[str, list[tuple[float, float]]] = {}
    for parent_key, pred, target in rows:
        grouped.setdefault(parent_key, []).append((pred, target))

    rng = random.Random(seed)
    parent_count = 0
    top1_hits = 0
    pick_target_sum = 0.0
    oracle_target_sum = 0.0
    random_pick_target_sum = 0.0
    topk_hits = {k: 0 for k in topk_values}
    topk_mass = {k: 0.0 for k in topk_values}
    random_topk_mass = {k: 0.0 for k in topk_values}

    for rows_for_parent in grouped.values():
        if not rows_for_parent:
            continue
        parent_count += 1
        ordered = sorted(rows_for_parent, key=lambda item: item[0], reverse=True)
        oracle_idx = max(range(len(rows_for_parent)), key=lambda i: rows_for_parent[i][1])
        oracle_target = rows_for_parent[oracle_idx][1]
        pick_target = ordered[0][1]

        oracle_target_sum += oracle_target
        pick_target_sum += pick_target

        random_pick = rng.choice(rows_for_parent)[1]
        random_pick_target_sum += random_pick

        if pick_target >= oracle_target - 1e-8:
            top1_hits += 1

        ordered_targets = [target for _, target in ordered]
        pool = list(rows_for_parent)
        rng.shuffle(pool)
        shuffled_targets = [target for _, target in pool]

        for k in topk_values:
            actual_k = min(k, len(ordered_targets))
            if actual_k <= 0:
                continue
            pred_top_targets = ordered_targets[:actual_k]
            rand_top_targets = shuffled_targets[:actual_k]
            topk_mass[k] += sum(pred_top_targets)
            random_topk_mass[k] += sum(rand_top_targets)
            if oracle_target <= max(pred_top_targets) + 1e-8:
                topk_hits[k] += 1

    metrics = {
        "parents": parent_count,
        "top1_acc": top1_hits / max(1, parent_count),
        "pick_target_mean": pick_target_sum / max(1, parent_count),
        "oracle_target_mean": oracle_target_sum / max(1, parent_count),
        "random_pick_target_mean": random_pick_target_sum / max(1, parent_count),
        "gap_to_oracle": (oracle_target_sum - pick_target_sum) / max(1, parent_count),
    }
    for k in topk_values:
        metrics[f"top{k}_oracle_recall"] = topk_hits[k] / max(1, parent_count)
        metrics[f"top{k}_retained_target_mass"] = topk_mass[k] / max(1, parent_count)
        metrics[f"top{k}_random_target_mass"] = random_topk_mass[k] / max(1, parent_count)
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline evaluation for a world-ranker checkpoint.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--val-input", default="")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--summary-out", default="")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--topk", type=int, action="append", default=[2, 4, 6])
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

    dataset = WorldRankDataset(rows)
    if len(dataset) == 0:
        raise SystemExit("Empty evaluation dataset after filtering.")

    sample = dataset[0]
    board_dim = len(sample["board_features"])
    world_dim = len(sample["world_features"])

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    ckpt_board_dim = int(config.get("board_dim", board_dim))
    ckpt_world_dim = int(config.get("world_dim", world_dim))
    if ckpt_board_dim != board_dim or ckpt_world_dim != world_dim:
        raise SystemExit(
            "Checkpoint dims do not match dataset: "
            f"ckpt=({ckpt_board_dim},{ckpt_world_dim}) data=({board_dim},{world_dim})"
        )

    model = WorldRankerNet(
        board_dim=board_dim,
        world_dim=world_dim,
        hidden_dim=int(config.get("hidden_dim", 256)),
        dropout=float(config.get("dropout", 0.1)),
    ).to(device)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict)

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_world_rank)
    rows_with_preds = _collect_rows(model, loader, device)
    topk_values = sorted({k for k in args.topk if k > 0})
    metrics = _summarize_parent_metrics(rows_with_preds, topk_values=topk_values, seed=args.seed)

    print(f"Evaluated world ranker on {len(dataset)} rows ({metrics['parents']} parents)")
    print(f"Split: {args.split} | Device: {device}")
    print(
        "Top1={:.4f} pick={:.4f} oracle={:.4f} random_pick={:.4f} gap={:.4f}".format(
            metrics["top1_acc"],
            metrics["pick_target_mean"],
            metrics["oracle_target_mean"],
            metrics["random_pick_target_mean"],
            metrics["gap_to_oracle"],
        )
    )
    for k in topk_values:
        print(
            "Top{} recall={:.4f} retained_mass={:.4f} random_mass={:.4f}".format(
                k,
                metrics[f"top{k}_oracle_recall"],
                metrics[f"top{k}_retained_target_mass"],
                metrics[f"top{k}_random_target_mass"],
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
            "topk": topk_values,
        }
        path = Path(args.summary_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Summary -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
