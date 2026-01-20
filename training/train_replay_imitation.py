#!/usr/bin/env python3
"""
Train imitation learning model on high-ELO replay demonstrations.

This trains a model to imitate high-rated player decisions from
parsed Pokemon Showdown replays.

Usage:
    python training/train_replay_imitation.py --input data/highelo_demos.pkl --epochs 50
"""

import argparse
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.actor_critic import ActorCritic


class ReplayDemoDataset(Dataset):
    """Dataset of (state, action) pairs from replay demonstrations."""

    def __init__(self, demos: list, weight_by_rating: bool = True, min_rating: int = 1500):
        """
        Initialize dataset.

        Args:
            demos: List of demonstration dicts with 'features', 'action', 'mask', 'rating'
            weight_by_rating: Whether to weight samples by player rating
            min_rating: Minimum rating for weight calculation baseline
        """
        self.samples = []
        self.weights = []

        for demo in demos:
            features = demo['features']
            action = demo['action']
            mask = demo['mask']
            rating = demo.get('rating', min_rating)
            base_weight = demo.get('weight', 1.0)

            # Convert to tensors
            features = torch.tensor(features, dtype=torch.float32)
            action = torch.tensor(action, dtype=torch.long)
            mask = torch.tensor(mask, dtype=torch.bool)

            # Weight by rating (higher rating = more weight)
            if weight_by_rating and rating:
                # Scale: 1500 -> 1.0, 2000 -> 1.5, 2500 -> 2.0
                rating_weight = 1.0 + (rating - min_rating) / 1000.0
                weight = base_weight * rating_weight
            else:
                weight = base_weight

            self.samples.append((features, action, mask))
            self.weights.append(weight)

        self.weights = torch.tensor(self.weights, dtype=torch.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        features, action, mask = self.samples[idx]
        weight = self.weights[idx]
        return features, action, mask, weight


def train_epoch(
    model: ActorCritic,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    temperature: float = 1.0,
) -> tuple:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for features, actions, masks, weights in dataloader:
        features = features.to(device)
        actions = actions.to(device)
        masks = masks.to(device)
        weights = weights.to(device)

        # Ensure at least one action is valid per sample
        valid_samples = masks.any(dim=-1)
        if not valid_samples.all():
            features = features[valid_samples]
            actions = actions[valid_samples]
            masks = masks[valid_samples]
            weights = weights[valid_samples]
            if len(features) == 0:
                continue

        optimizer.zero_grad()

        # Forward pass (model applies mask internally)
        logits, _ = model(features, masks)

        # Clamp for numerical stability
        logits = torch.clamp(logits, min=-100, max=100)

        # Temperature scaling for training
        if temperature != 1.0:
            logits = logits / temperature

        # Weighted cross-entropy loss with label smoothing effect
        log_probs = F.log_softmax(logits, dim=-1)

        # Gather log probs for actions
        action_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Negative log likelihood (higher weight = more importance)
        nll = -action_log_probs
        loss = (nll * weights).mean()

        # Skip if loss is NaN
        if torch.isnan(loss) or torch.isinf(loss):
            continue

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * len(actions)

        # Accuracy
        preds = logits.argmax(dim=-1)
        total_correct += (preds == actions).sum().item()
        total_samples += len(actions)

    if total_samples == 0:
        return 0.0, 0.0
    return total_loss / total_samples, total_correct / total_samples


def evaluate(
    model: ActorCritic,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple:
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for features, actions, masks, weights in dataloader:
            features = features.to(device)
            actions = actions.to(device)
            masks = masks.to(device)

            # Ensure at least one action is valid per sample
            valid_samples = masks.any(dim=-1)
            if not valid_samples.all():
                features = features[valid_samples]
                actions = actions[valid_samples]
                masks = masks[valid_samples]
                if len(features) == 0:
                    continue

            # Forward pass (model applies mask internally)
            logits, _ = model(features, masks)
            logits = torch.clamp(logits, min=-100, max=100)

            log_probs = F.log_softmax(logits, dim=-1)
            action_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
            nll = -action_log_probs.mean()

            if not torch.isnan(nll) and not torch.isinf(nll):
                total_loss += nll.item() * len(actions)

            preds = logits.argmax(dim=-1)
            total_correct += (preds == actions).sum().item()
            total_samples += len(actions)

    if total_samples == 0:
        return 0.0, 0.0
    return total_loss / total_samples, total_correct / total_samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='data/highelo_demos.pkl',
                       help='Path to demonstration pickle file')
    parser.add_argument('--output', type=str, default='checkpoints/rl/replay_imitation.pt',
                       help='Output checkpoint path')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-4,
                       help='Learning rate')
    parser.add_argument('--val-split', type=float, default=0.1,
                       help='Validation split ratio')
    parser.add_argument('--weight-by-rating', action='store_true', default=True,
                       help='Weight samples by player rating')
    parser.add_argument('--no-weight-by-rating', action='store_true',
                       help='Disable rating-based weighting')
    parser.add_argument('--min-rating', type=int, default=1500,
                       help='Minimum rating for weight baseline')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='Temperature for softmax (lower = sharper)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    args = parser.parse_args()

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load demonstrations
    print(f"Loading demonstrations from {args.input}...")
    with open(args.input, 'rb') as f:
        demos = pickle.load(f)
    print(f"Loaded {len(demos)} demonstrations")

    # Filter invalid action-mask pairs
    valid_demos = []
    for demo in demos:
        action = demo['action']
        mask = demo['mask']
        # Check if action is within valid range and mask is True for that action
        if 0 <= action < len(mask) and mask[action]:
            valid_demos.append(demo)
        else:
            # Make action valid by setting mask to True for the action
            # This handles cases where we don't have full move info from replays
            demo['mask'][action] = True
            valid_demos.append(demo)
    demos = valid_demos
    print(f"After mask fixing: {len(demos)} demonstrations")

    # Print rating distribution
    ratings = [d.get('rating', 0) for d in demos if d.get('rating')]
    if ratings:
        print(f"Rating range: {min(ratings)} - {max(ratings)}")
        print(f"Average rating: {sum(ratings)/len(ratings):.0f}")

    # Train/val split
    random.shuffle(demos)
    val_size = int(len(demos) * args.val_split)
    val_demos = demos[:val_size]
    train_demos = demos[val_size:]
    print(f"Train: {len(train_demos)}, Val: {len(val_demos)}")

    # Create datasets
    weight_by_rating = args.weight_by_rating and not args.no_weight_by_rating
    train_dataset = ReplayDemoDataset(train_demos, weight_by_rating, args.min_rating)
    val_dataset = ReplayDemoDataset(val_demos, weight_by_rating=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    # Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = ActorCritic(
        feature_dim=272,
        d_model=512,
        n_actions=13,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    # Training loop
    best_val_acc = 0.0
    best_epoch = 0

    print("\nTraining...")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, device, args.temperature
        )
        val_loss, val_acc = evaluate(model, val_loader, device)

        scheduler.step()

        print(f"Epoch {epoch:3d}: train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}")

        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch

            # Save checkpoint
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_acc': val_acc,
                'config': {
                    'feature_dim': 272,
                    'd_model': 512,
                    'n_actions': 13,
                }
            }, output_path)

    print(f"\nBest validation accuracy: {best_val_acc:.3f} at epoch {best_epoch}")
    print(f"Model saved to {args.output}")


if __name__ == '__main__':
    main()
