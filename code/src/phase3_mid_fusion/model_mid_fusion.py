from __future__ import annotations

import torch
import torch.nn as nn


class MidFusionHead(nn.Module):
    """Small classifier over concatenated frozen specialist latents."""

    def __init__(self, in_dim: int = 320, hidden_dim: int = 160, dropout: float = 0.20, n_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_classes),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class GuardNeckFusionHead(nn.Module):
    """Classifier over specialist latents plus guard evidence at the neck."""

    def __init__(self, in_dim: int = 324, hidden_dim: int = 192, dropout: float = 0.20, n_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_classes),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class FrozenSpecialistMidFusion(nn.Module):
    """Runs each fixed view through its matching frozen specialist encoder."""

    def __init__(self, specialists: list[nn.Module], head: MidFusionHead):
        super().__init__()
        self.specialists = nn.ModuleList(specialists)
        self.head = head
        for model in self.specialists:
            model.eval()
            for param in model.parameters():
                param.requires_grad_(False)

    @torch.no_grad()
    def encode_views(self, x_views: torch.Tensor) -> torch.Tensor:
        latents = []
        for view_idx, model in enumerate(self.specialists):
            latents.append(model.encode(x_views[:, view_idx]))
        return torch.cat(latents, dim=1)

    def forward(self, x_views: torch.Tensor) -> torch.Tensor:
        z = self.encode_views(x_views)
        return self.head(z)
