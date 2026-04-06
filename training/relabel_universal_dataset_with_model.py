#!/usr/bin/env python3
"""Relabel a universal supervised dataset with a trained prior/value model.

This is the current practical teacher path for the clean Showdown corpus:
- preserve the original replay policy/value targets for comparison
- replace `policy_target` with the teacher model distribution
- optionally replace `value_target` with the teacher model value

This is not search-state relabeling. The current universal dataset does not
store reconstructible search states, so the teacher operates on the stored
feature interface directly.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
from collections import Counter
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.search_prior_value import SearchPriorValueNet
from training.search_assist_utils import load_search_assist_examples, safe_float


def _entropy(probs: list[float]) -> float:
    total = 0.0
    for p in probs:
        if p > 0.0:
            total -= p * math.log(max(p, 1e-12))
    return total


def _load_model(checkpoint_path: str, device_name: str):
    ckpt = torch.load(checkpoint_path, map_location=device_name, weights_only=False)
    config = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    board_dim = int(config.get("board_dim", 272))
    action_dim = int(config.get("action_dim", 16))
    n_actions = int(config.get("n_actions", 13))
    hidden_dim = int(config.get("hidden_dim", 256))
    dropout = float(config.get("dropout", 0.1))
    model = SearchPriorValueNet(
        board_dim=board_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        n_actions=n_actions,
        dropout=dropout,
    ).to(device_name)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()
    return model, config


def _softmax_masked(logits: torch.Tensor, mask: list[bool]) -> list[float]:
    mask_t = torch.tensor(mask, dtype=torch.bool, device=logits.device).unsqueeze(0)
    masked = logits.masked_fill(~mask_t, -1e9)
    probs = torch.softmax(masked, dim=-1).squeeze(0).detach().cpu().tolist()
    return [float(p) if mask[i] else 0.0 for i, p in enumerate(probs)]


def main() -> int:
    parser = argparse.ArgumentParser(description="Relabel universal dataset rows with a trained prior/value model.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-out", default="")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--value-mode", choices=["orig", "teacher"], default="orig")
    parser.add_argument("--min-teacher-top1", type=float, default=0.0)
    parser.add_argument("--max-teacher-entropy", type=float, default=0.0)
    parser.add_argument("--max-rows", type=int, default=0)
    args = parser.parse_args()

    device = args.device.strip().lower()
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    rows = load_search_assist_examples(args.input)
    model, config = _load_model(args.checkpoint, device)

    out_rows: list[dict] = []
    counters = Counter()
    action_kind_counts = Counter()
    teacher_top1_values: list[float] = []
    teacher_entropy_values: list[float] = []
    teacher_value_values: list[float] = []

    for start in range(0, len(rows), max(1, args.batch_size)):
        batch = rows[start : start + max(1, args.batch_size)]
        if not batch:
            continue
        board = torch.tensor(
            [[safe_float(v, 0.0) for v in row["board_features"]] for row in batch],
            dtype=torch.float32,
            device=device,
        )
        action = torch.tensor(
            [[[safe_float(v, 0.0) for v in feat] for feat in row["action_features"]] for row in batch],
            dtype=torch.float32,
            device=device,
        )
        mask = torch.tensor(
            [[bool(v) for v in row["action_mask"]] for row in batch],
            dtype=torch.bool,
            device=device,
        )

        with torch.no_grad():
            logits, values = model(board, action, mask)

        for idx, row in enumerate(batch):
            counters["rows_seen"] += 1
            probs = _softmax_masked(logits[idx : idx + 1], [bool(v) for v in row["action_mask"]])
            top1 = max(probs) if probs else 0.0
            ent = _entropy(probs)
            teacher_value = float(values[idx].detach().cpu().item())

            if top1 < args.min_teacher_top1:
                counters["drop_teacher_top1"] += 1
                continue
            if args.max_teacher_entropy > 0.0 and ent > args.max_teacher_entropy:
                counters["drop_teacher_entropy"] += 1
                continue

            new_row = dict(row)
            new_row["orig_policy_target"] = list(row.get("policy_target", []) or [])
            new_row["orig_value_target"] = safe_float(row.get("value_target", 0.0), 0.0)
            new_row["teacher_policy_target"] = probs
            new_row["teacher_value_target"] = teacher_value
            new_row["teacher_top1_prob"] = top1
            new_row["teacher_entropy"] = ent
            new_row["teacher_checkpoint"] = str(args.checkpoint)
            new_row["teacher_config"] = {
                "board_dim": int(config.get("board_dim", 272)),
                "action_dim": int(config.get("action_dim", 16)),
                "n_actions": int(config.get("n_actions", 13)),
                "hidden_dim": int(config.get("hidden_dim", 256)),
                "dropout": float(config.get("dropout", 0.1)),
            }
            new_row["policy_target"] = probs
            if args.value_mode == "teacher":
                new_row["value_target"] = teacher_value
            else:
                new_row["value_target"] = new_row["orig_value_target"]
            new_row["teacher_source"] = "search_prior_value_model"
            out_rows.append(new_row)
            counters["kept"] += 1
            action_kind_counts[str(new_row.get("chosen_action_kind", ""))] += 1
            teacher_top1_values.append(top1)
            teacher_entropy_values.append(ent)
            teacher_value_values.append(teacher_value)
            if args.max_rows > 0 and counters["kept"] >= args.max_rows:
                break
        if args.max_rows > 0 and counters["kept"] >= args.max_rows:
            break

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as handle:
        pickle.dump(out_rows, handle)

    summary = {
        "input": args.input,
        "checkpoint": args.checkpoint,
        "output": args.output,
        "rows": len(out_rows),
        "counters": dict(counters),
        "value_mode": args.value_mode,
        "device": device,
        "batch_size": args.batch_size,
        "min_teacher_top1": args.min_teacher_top1,
        "max_teacher_entropy": args.max_teacher_entropy,
        "action_kind_counts": dict(action_kind_counts),
        "teacher_top1_mean": (sum(teacher_top1_values) / len(teacher_top1_values)) if teacher_top1_values else 0.0,
        "teacher_entropy_mean": (sum(teacher_entropy_values) / len(teacher_entropy_values)) if teacher_entropy_values else 0.0,
        "teacher_value_mean": (sum(teacher_value_values) / len(teacher_value_values)) if teacher_value_values else 0.0,
    }

    print(f"Wrote {len(out_rows)} relabeled rows -> {args.output}")
    print(f"Stats: {dict(counters)}")
    print(f"Action kinds: {dict(action_kind_counts)}")
    print(f"Teacher top1 mean: {summary['teacher_top1_mean']:.4f}")
    print(f"Teacher entropy mean: {summary['teacher_entropy_mean']:.4f}")
    print(f"Teacher value mean: {summary['teacher_value_mean']:.4f}")

    if args.summary_out:
        Path(args.summary_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary_out).write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")
        print(f"Summary -> {args.summary_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
