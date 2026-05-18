from __future__ import annotations

import torch
import torch.nn as nn


class HarmonicFusionHead(nn.Module):
    def __init__(self, in_dim: int = 72, hidden_dim: int = 48, n_classes: int = 2, dropout: float = 0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

