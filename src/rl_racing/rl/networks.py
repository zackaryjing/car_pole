"""Neural network modules for RL algorithms."""

from __future__ import annotations

import torch
from torch import nn


class MLPQNetwork(nn.Module):
    def __init__(self, input_dim: int, action_count: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_count),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.net(observations)
