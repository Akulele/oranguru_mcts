"""
Compact candidate-world ranking model for hidden-state pruning.
"""

from __future__ import annotations

import torch
from torch import nn


class WorldRankerNet(nn.Module):
    """
    Lightweight scorer over (board, world) feature pairs.
    """

    def __init__(
        self,
        board_dim: int,
        world_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.board_encoder = nn.Sequential(
            nn.Linear(board_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.world_encoder = nn.Sequential(
            nn.Linear(world_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        final = self.score_head[-1]
        if isinstance(final, nn.Linear):
            nn.init.xavier_uniform_(final.weight, gain=0.1)

    def forward(self, board_features: torch.Tensor, world_features: torch.Tensor) -> torch.Tensor:
        board_ctx = self.board_encoder(board_features)
        world_ctx = self.world_encoder(world_features)
        logits = self.score_head(torch.cat([board_ctx, world_ctx], dim=-1)).squeeze(-1)
        return torch.sigmoid(logits)
