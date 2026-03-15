#!/usr/bin/env python3
"""
Shared helpers for search-assist prior/value training.

Expected example schema:
{
  "battle_id": str,
  "turn": int,
  "rating": int | float | None,
  "board_features": List[float],        # [board_dim]
  "action_features": List[List[float]], # [n_actions, action_dim]
  "action_mask": List[bool],            # [n_actions]
  "visit_counts": List[float],          # [n_actions], optional if policy_target provided
  "policy_target": List[float],         # [n_actions], optional if visit_counts provided
  "value_target": float,                # final outcome from current player's perspective
  "weight": float,                      # optional
  "source": str,                        # optional
  "tag": str,                           # optional
}
"""

from __future__ import annotations

import math
import pickle
from dataclasses import dataclass
from typing import Iterable

import torch
from torch.utils.data import Dataset


SCHEMA_VERSION = 1


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        out = float(value)
        if not math.isfinite(out):
            return default
        return out
    except Exception:
        return default


def _normalize_policy_target(
    counts: list[float] | None,
    probs: list[float] | None,
    mask: list[bool],
) -> list[float] | None:
    n = len(mask)
    if probs is not None:
        if len(probs) != n:
            return None
        out = [max(0.0, safe_float(v, 0.0)) if mask[i] else 0.0 for i, v in enumerate(probs)]
        total = sum(out)
        if total <= 0:
            return None
        return [v / total for v in out]
    if counts is not None:
        if len(counts) != n:
            return None
        out = [max(0.0, safe_float(v, 0.0)) if mask[i] else 0.0 for i, v in enumerate(counts)]
        total = sum(out)
        if total <= 0:
            legal = [i for i, ok in enumerate(mask) if ok]
            if not legal:
                return None
            uni = 1.0 / float(len(legal))
            return [uni if ok else 0.0 for ok in mask]
        return [v / total for v in out]
    return None


def validate_search_assist_example(example: dict) -> tuple[bool, str]:
    board = example.get("board_features")
    action_features = example.get("action_features")
    mask = example.get("action_mask")
    value = example.get("value_target")

    if not isinstance(board, list) or not board:
        return False, "missing_board_features"
    if not isinstance(action_features, list) or not action_features:
        return False, "missing_action_features"
    if not isinstance(mask, list) or not mask:
        return False, "missing_action_mask"
    if len(action_features) != len(mask):
        return False, "len_action_features"
    if not any(bool(v) for v in mask):
        return False, "empty_action_mask"
    row0 = action_features[0]
    if not isinstance(row0, list) or not row0:
        return False, "bad_action_feature_row"
    action_dim = len(row0)
    for row in action_features:
        if not isinstance(row, list) or len(row) != action_dim:
            return False, "ragged_action_features"
    policy = _normalize_policy_target(example.get("visit_counts"), example.get("policy_target"), mask)
    if policy is None:
        return False, "missing_policy_target"
    if not isinstance(value, (int, float)):
        return False, "missing_value_target"
    return True, "ok"


def load_search_assist_examples(path: str) -> list[dict]:
    with open(path, "rb") as handle:
        obj = pickle.load(handle)
    if not isinstance(obj, list):
        raise ValueError(f"Expected list pickle: {path}")
    return obj


@dataclass
class SearchAssistBatch:
    board_features: torch.Tensor
    action_features: torch.Tensor
    action_mask: torch.Tensor
    policy_target: torch.Tensor
    value_target: torch.Tensor
    weights: torch.Tensor


class SearchAssistDataset(Dataset):
    def __init__(
        self,
        examples: Iterable[dict],
        min_visits: float = 1.0,
        rating_weight: float = 0.0,
        rating_baseline: float = 1500.0,
        rating_scale: float = 1000.0,
    ):
        self.items: list[dict] = []
        for ex in examples:
            ok, reason = validate_search_assist_example(ex)
            if not ok:
                continue
            mask = [bool(v) for v in ex["action_mask"]]
            policy = _normalize_policy_target(ex.get("visit_counts"), ex.get("policy_target"), mask)
            assert policy is not None
            total_visits = sum(max(0.0, safe_float(v, 0.0)) for v in (ex.get("visit_counts") or []))
            if ex.get("visit_counts") is not None and total_visits < min_visits:
                continue

            weight = safe_float(ex.get("weight", 1.0), 1.0)
            rating = ex.get("rating")
            if rating_weight > 0.0 and isinstance(rating, (int, float)) and rating_scale > 0:
                bonus = max(-1.0, min(1.0, (float(rating) - rating_baseline) / rating_scale))
                weight *= 1.0 + rating_weight * bonus

            self.items.append(
                {
                    "battle_id": str(ex.get("battle_id", "unknown")),
                    "turn": int(ex.get("turn", 0)),
                    "board_features": [safe_float(v, 0.0) for v in ex["board_features"]],
                    "action_features": [
                        [safe_float(v, 0.0) for v in row] for row in ex["action_features"]
                    ],
                    "action_mask": mask,
                    "policy_target": policy,
                    "value_target": safe_float(ex.get("value_target", 0.0), 0.0),
                    "weight": weight,
                }
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        return self.items[idx]


def collate_search_assist(batch: list[dict]) -> SearchAssistBatch:
    board_dim = len(batch[0]["board_features"])
    n_actions = len(batch[0]["action_mask"])
    action_dim = len(batch[0]["action_features"][0])

    board = torch.zeros(len(batch), board_dim, dtype=torch.float32)
    action = torch.zeros(len(batch), n_actions, action_dim, dtype=torch.float32)
    mask = torch.zeros(len(batch), n_actions, dtype=torch.bool)
    policy = torch.zeros(len(batch), n_actions, dtype=torch.float32)
    value = torch.zeros(len(batch), dtype=torch.float32)
    weights = torch.ones(len(batch), dtype=torch.float32)

    for i, item in enumerate(batch):
        board[i] = torch.tensor(item["board_features"], dtype=torch.float32)
        action[i] = torch.tensor(item["action_features"], dtype=torch.float32)
        mask[i] = torch.tensor(item["action_mask"], dtype=torch.bool)
        policy[i] = torch.tensor(item["policy_target"], dtype=torch.float32)
        value[i] = float(item["value_target"])
        weights[i] = float(item.get("weight", 1.0))

    return SearchAssistBatch(
        board_features=board,
        action_features=action,
        action_mask=mask,
        policy_target=policy,
        value_target=value,
        weights=weights,
    )
