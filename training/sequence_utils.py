#!/usr/bin/env python3
"""
Shared helpers for sequence-policy training/evaluation on trajectory datasets.
"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from typing import Iterable

import torch
from torch.utils.data import Dataset


DEFAULT_FEATURE_DIM = 272
DEFAULT_N_ACTIONS = 13


def discounted_returns(rewards: list[float], gamma: float) -> list[float]:
    out: list[float] = [0.0] * len(rewards)
    running = 0.0
    for i in range(len(rewards) - 1, -1, -1):
        running = float(rewards[i]) + gamma * running
        out[i] = running
    return out


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        f = float(value)
        if not math.isfinite(f):
            return default
        return f
    except Exception:
        return default


def stable_split_by_battle(
    trajectories: Iterable[dict],
    holdout_ratio: float,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    train: list[dict] = []
    holdout: list[dict] = []
    ratio = max(0.0, min(1.0, holdout_ratio))
    threshold = int(ratio * 10000)
    salt = str(seed)
    for traj in trajectories:
        battle_id = str(traj.get("battle_id", "unknown"))
        key = f"{battle_id}|{salt}".encode("utf-8")
        bucket = int(hashlib.sha1(key).hexdigest()[:8], 16) % 10000
        if bucket < threshold:
            holdout.append(traj)
        else:
            train.append(traj)
    return train, holdout


@dataclass
class SequenceBatch:
    features: torch.Tensor
    masks: torch.Tensor
    actions: torch.Tensor
    returns: torch.Tensor
    valid: torch.Tensor
    weights: torch.Tensor
    ratings: torch.Tensor


class TrajectoryDataset(Dataset):
    """
    Sequence dataset for recurrent policy training.
    """

    def __init__(
        self,
        trajectories: list[dict],
        gamma: float = 0.99,
        min_actions: int = 1,
        weight_by_rating: bool = False,
        rating_baseline: float = 1400.0,
        rating_scale: float = 1000.0,
        max_rating_weight: float = 2.5,
    ):
        self.items: list[dict] = []
        for traj in trajectories:
            actions = list(traj.get("actions", []) or [])
            if len(actions) < min_actions:
                continue
            features = list(traj.get("features", []) or [])
            masks = list(traj.get("masks", []) or [])
            rewards = list(traj.get("rewards", []) or [])
            if not features or not masks:
                continue
            if len(features) != len(actions) or len(masks) != len(actions):
                continue

            if len(rewards) != len(actions):
                rewards = [0.0] * len(actions)
                if traj.get("winner") is not None:
                    rewards[-1] = safe_float(traj.get("winner_reward", 0.0), 0.0)

            returns = discounted_returns(rewards, gamma)

            base_weight = safe_float(traj.get("weight", 1.0), 1.0)
            rating = traj.get("rating")
            if weight_by_rating and isinstance(rating, (int, float)) and rating_scale > 0:
                rw = 1.0 + (float(rating) - float(rating_baseline)) / float(rating_scale)
                rw = max(0.25, min(float(max_rating_weight), rw))
                base_weight *= rw

            self.items.append(
                {
                    "features": features,
                    "masks": masks,
                    "actions": actions,
                    "returns": returns,
                    "weight": base_weight,
                    "rating": safe_float(rating, 0.0) if rating is not None else 0.0,
                }
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        return self.items[idx]


def collate_trajectories(batch: list[dict]) -> SequenceBatch:
    batch_size = len(batch)
    max_len = max(len(item["actions"]) for item in batch)
    feature_dim = len(batch[0]["features"][0])
    n_actions = len(batch[0]["masks"][0])

    features = torch.zeros(batch_size, max_len, feature_dim, dtype=torch.float32)
    masks = torch.zeros(batch_size, max_len, n_actions, dtype=torch.bool)
    actions = torch.zeros(batch_size, max_len, dtype=torch.long)
    returns = torch.zeros(batch_size, max_len, dtype=torch.float32)
    valid = torch.zeros(batch_size, max_len, dtype=torch.bool)
    weights = torch.ones(batch_size, max_len, dtype=torch.float32)
    ratings = torch.zeros(batch_size, dtype=torch.float32)

    for i, traj in enumerate(batch):
        length = len(traj["actions"])
        features[i, :length] = torch.tensor(traj["features"], dtype=torch.float32)
        masks[i, :length] = torch.tensor(traj["masks"], dtype=torch.bool)
        actions[i, :length] = torch.tensor(traj["actions"], dtype=torch.long)
        returns[i, :length] = torch.tensor(traj["returns"], dtype=torch.float32)
        valid[i, :length] = True
        weights[i, :length] = float(traj.get("weight", 1.0))
        ratings[i] = float(traj.get("rating", 0.0))

    return SequenceBatch(
        features=features,
        masks=masks,
        actions=actions,
        returns=returns,
        valid=valid,
        weights=weights,
        ratings=ratings,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
