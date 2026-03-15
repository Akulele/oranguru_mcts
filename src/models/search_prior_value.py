"""
Compact action-conditioned prior/value model for MCTS assistance.
"""

from __future__ import annotations

import torch
from torch import nn


class SearchPriorValueNet(nn.Module):
    """
    DeepSets-style model:
    - shared board encoder
    - shared action encoder
    - action-conditioned policy head
    - pooled action context for value head
    """

    def __init__(
        self,
        board_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        n_actions: int = 13,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_actions = n_actions
        self.board_encoder = nn.Sequential(
            nn.Linear(board_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
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
        final_policy = self.policy_head[-1]
        final_value = self.value_head[-2]
        if isinstance(final_policy, nn.Linear):
            nn.init.xavier_uniform_(final_policy.weight, gain=0.01)
        if isinstance(final_value, nn.Linear):
            nn.init.xavier_uniform_(final_value.weight, gain=1.0)

    def forward(
        self,
        board_features: torch.Tensor,
        action_features: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            board_features: [B, board_dim]
            action_features: [B, A, action_dim]
            action_mask: [B, A]
        Returns:
            logits: [B, A]
            value: [B]
        """
        board_ctx = self.board_encoder(board_features)
        action_ctx = self.action_encoder(action_features)
        board_rep = board_ctx.unsqueeze(1).expand(-1, action_ctx.shape[1], -1)

        valid = action_mask.float().unsqueeze(-1)
        pooled = (action_ctx * valid).sum(dim=1)
        pooled = pooled / valid.sum(dim=1).clamp(min=1.0)
        pooled_rep = pooled.unsqueeze(1).expand_as(action_ctx)

        policy_in = torch.cat([board_rep, action_ctx, pooled_rep], dim=-1)
        logits = self.policy_head(policy_in).squeeze(-1)
        logits = logits.masked_fill(~action_mask, -1e9)

        value_in = torch.cat([board_ctx, pooled], dim=-1)
        value = self.value_head(value_in).squeeze(-1)
        return logits, value
