"""
🦧 ORANGURU RL - Actor-Critic Model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    """Actor-Critic network for PPO with residual connections."""

    def __init__(self, feature_dim: int = 128, d_model: int = 128, n_actions: int = 13):
        super().__init__()

        # Input projection
        self.input_proj = nn.Linear(feature_dim, d_model)

        # Encoder with residual connections
        self.encoder_layers = nn.ModuleList([
            nn.ModuleDict({
                'linear': nn.Linear(d_model, d_model),
                'norm': nn.LayerNorm(d_model),
                'dropout': nn.Dropout(0.1)
            })
            for _ in range(3)  # 3 residual blocks
        ])

        # Actor head with larger capacity
        self.actor = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, n_actions),
        )

        # Critic head with larger capacity
        self.critic = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
        )

        # Initialize with better weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with better values for faster learning."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Use xavier for better gradient flow
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Smaller initialization for final layers
        nn.init.xavier_uniform_(self.actor[-1].weight, gain=0.01)
        nn.init.xavier_uniform_(self.critic[-1].weight, gain=1.0)
    
    def forward(self, features: torch.Tensor, action_mask: torch.Tensor):
        """
        Args:
            features: (batch, feature_dim)
            action_mask: (batch, n_actions) - True for legal actions

        Returns:
            logits: (batch, n_actions) - masked
            value: (batch,)
        """
        # Input projection
        x = self.input_proj(features)
        x = F.relu(x)

        # Encoder with residual connections
        for layer in self.encoder_layers:
            residual = x
            x = layer['linear'](x)
            x = layer['norm'](x)
            x = F.relu(x)
            x = layer['dropout'](x)
            x = x + residual  # Residual connection

        # Actor and critic heads
        logits = self.actor(x)
        logits = logits.masked_fill(~action_mask, float('-inf'))
        value = self.critic(x).squeeze(-1)
        return logits, value
    
    def get_action(self, features: torch.Tensor, action_mask: torch.Tensor, 
                   deterministic: bool = False):
        """Sample action from policy."""
        logits, value = self.forward(features, action_mask)
        
        # Clamp logits to prevent numerical instability
        logits = torch.clamp(logits, min=-1e8, max=1e8)
        
        # Softmax with numerical stability
        probs = F.softmax(logits, dim=-1)
        
        # Handle NaN values by replacing with uniform distribution over valid actions
        nan_mask = torch.isnan(probs)
        if nan_mask.any():
            # Use uniform distribution over masked actions
            uniform_probs = action_mask.float() / action_mask.float().sum(dim=-1, keepdim=True)
            probs = torch.where(nan_mask, uniform_probs, probs)
        
        # Ensure valid probability distribution
        probs = torch.clamp(probs, min=1e-8)
        probs = probs / probs.sum(dim=-1, keepdim=True)
        
        dist = Categorical(probs)
        
        if deterministic:
            action = logits.argmax(dim=-1)
        else:
            action = dist.sample()
        
        return action, dist.log_prob(action), value, dist.entropy()
    
    def evaluate(self, features: torch.Tensor, action_mask: torch.Tensor, 
                 actions: torch.Tensor):
        """Evaluate actions for PPO update."""
        logits, value = self.forward(features, action_mask)
        
        # Clamp logits to prevent numerical instability
        logits = torch.clamp(logits, min=-1e8, max=1e8)
        
        # Softmax with numerical stability
        probs = F.softmax(logits, dim=-1)
        
        # Handle NaN values by replacing with uniform distribution over valid actions
        nan_mask = torch.isnan(probs)
        if nan_mask.any():
            uniform_probs = action_mask.float() / action_mask.float().sum(dim=-1, keepdim=True)
            probs = torch.where(nan_mask, uniform_probs, probs)
        
        # Ensure valid probability distribution
        probs = torch.clamp(probs, min=1e-8)
        probs = probs / probs.sum(dim=-1, keepdim=True)
        
        dist = Categorical(probs)
        return dist.log_prob(actions), value, dist.entropy()


class RecurrentActorCritic(nn.Module):
    """Recurrent actor-critic with a GRU for sequence conditioning."""

    is_recurrent = True

    def __init__(self, feature_dim: int = 128, d_model: int = 128, n_actions: int = 13,
                 rnn_hidden: int = 256, rnn_layers: int = 1):
        super().__init__()

        self.input_proj = nn.Linear(feature_dim, d_model)
        self.rnn = nn.GRU(input_size=d_model, hidden_size=rnn_hidden,
                          num_layers=rnn_layers, batch_first=True)

        self.actor = nn.Sequential(
            nn.Linear(rnn_hidden, rnn_hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(rnn_hidden, n_actions),
        )
        self.critic = nn.Sequential(
            nn.Linear(rnn_hidden, rnn_hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(rnn_hidden, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.actor[-1].weight, gain=0.01)
        nn.init.xavier_uniform_(self.critic[-1].weight, gain=1.0)

    def init_hidden(self, batch_size: int, device: torch.device | str):
        return torch.zeros(self.rnn.num_layers, batch_size, self.rnn.hidden_size, device=device)

    def forward_step(self, features: torch.Tensor, hidden: torch.Tensor | None = None):
        """Single-step forward for online play."""
        x = F.relu(self.input_proj(features))
        x = x.unsqueeze(1)
        out, next_hidden = self.rnn(x, hidden)
        h = out[:, -1]
        logits = self.actor(h)
        value = self.critic(h).squeeze(-1)
        return logits, value, next_hidden

    def forward_sequence(self, features: torch.Tensor, hidden: torch.Tensor | None = None):
        """Sequence forward for offline training."""
        x = F.relu(self.input_proj(features))
        out, next_hidden = self.rnn(x, hidden)
        logits = self.actor(out)
        values = self.critic(out).squeeze(-1)
        return logits, values, next_hidden

    def get_action(self, features: torch.Tensor, action_mask: torch.Tensor,
                   hidden: torch.Tensor | None = None, deterministic: bool = False):
        """Sample action from policy with recurrent state."""
        logits, value, next_hidden = self.forward_step(features, hidden)
        masked_logits = logits.masked_fill(~action_mask, -1e9)
        masked_logits = torch.clamp(masked_logits, min=-1e8, max=1e8)
        probs = F.softmax(masked_logits, dim=-1)
        probs = torch.clamp(probs, min=1e-8)
        probs = probs / probs.sum(dim=-1, keepdim=True)
        dist = Categorical(probs)

        if deterministic:
            action = masked_logits.argmax(dim=-1)
        else:
            action = dist.sample()

        return action, dist.log_prob(action), value, dist.entropy(), next_hidden

    def evaluate(self, features: torch.Tensor, action_mask: torch.Tensor,
                 actions: torch.Tensor, hidden: torch.Tensor | None = None):
        """Evaluate actions for PPO-style losses (single-step or batched)."""
        if features.dim() == 3:
            logits, values, _ = self.forward_sequence(features, hidden)
        else:
            logits, values, _ = self.forward_step(features, hidden)
        masked_logits = logits.masked_fill(~action_mask, -1e9)
        masked_logits = torch.clamp(masked_logits, min=-1e8, max=1e8)
        probs = F.softmax(masked_logits, dim=-1)
        probs = torch.clamp(probs, min=1e-8)
        probs = probs / probs.sum(dim=-1, keepdim=True)
        dist = Categorical(probs)
        return dist.log_prob(actions), values, dist.entropy()
