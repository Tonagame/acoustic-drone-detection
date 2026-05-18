"""
Option 2 prediction: one generalist CNN evaluated across the five views.
"""

from dataclasses import dataclass

import numpy as np
import torch

from audio_views import audio_to_logmel, create_audio_views
from config_hybrid import NOISE_FLOOR, VIEW_WEIGHTS


@dataclass
class Option2Prediction:
    score: float
    per_view_probs: np.ndarray


@torch.no_grad()
def predict_option2(option2: dict, audio: np.ndarray, device=None) -> Option2Prediction:
    if device is None:
        device = next(option2["model"].parameters()).device
    if np.abs(audio).max() < NOISE_FLOOR:
        return Option2Prediction(0.0, np.zeros(5, dtype=np.float32))

    views = create_audio_views(audio)
    probs = np.zeros(5, dtype=np.float32)
    model = option2["model"]
    drone_idx = option2["drone_idx"]
    for vi, view in enumerate(views):
        lm = audio_to_logmel(view)
        x = lm.unsqueeze(0).unsqueeze(0).to(device)
        probs[vi] = torch.softmax(model(x), dim=1)[0, drone_idx].item()
    score = float(np.array(VIEW_WEIGHTS, dtype=np.float32) @ probs)
    return Option2Prediction(score, probs)
