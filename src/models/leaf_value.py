"""
Compact value model for cutoff-state evaluation.
"""

from __future__ import annotations

import torch
from torch import nn


class LeafValueNet(nn.Module):
    """
    Lightweight regressor over board features plus compact state extras.
    Outputs a value in [-1, 1].
    """

    def __init__(
        self,
        board_dim: int,
        extra_dim: int,
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
        self.extra_encoder = nn.Sequential(
            nn.Linear(extra_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        final = self.value_head[-2]
        if isinstance(final, nn.Linear):
            nn.init.xavier_uniform_(final.weight, gain=0.1)

    def forward(self, board_features: torch.Tensor, extra_features: torch.Tensor) -> torch.Tensor:
        board_ctx = self.board_encoder(board_features)
        extra_ctx = self.extra_encoder(extra_features)
        return self.value_head(torch.cat([board_ctx, extra_ctx], dim=-1)).squeeze(-1)
