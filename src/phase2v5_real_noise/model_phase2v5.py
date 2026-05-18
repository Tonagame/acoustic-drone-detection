from __future__ import annotations

import torch
import torch.nn as nn


class DroneCNNV5(nn.Module):
    """Same CNN family as earlier phases, split into encoder and classifier."""

    def __init__(self, n_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, n_classes)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.gap(self.features(x)).view(x.size(0), -1)

    def classify_from_latent(self, z: torch.Tensor) -> torch.Tensor:
        return self.fc(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classify_from_latent(self.encode(x))


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = float(gamma)
        self.register_buffer("weight", weight if weight is not None else None)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = nn.functional.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        return (((1.0 - pt) ** self.gamma) * ce).mean()


def extract_model_state(checkpoint: dict) -> dict:
    if "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    return checkpoint

