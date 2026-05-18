"""
Option 3 prediction: five specialist CNNs, one per audio view.
"""

from dataclasses import dataclass

import numpy as np
import torch

from audio_views import audio_to_logmel, create_audio_views
from config_hybrid import (
    NOISE_FLOOR,
    OPTION3_ALONE_FMAX_THR,
    OPTION3_ALONE_SCORE_THR,
    OPTION3_ALONE_VOTE_THR,
    OPTION3_ALONE_VOTES_NEED,
    OPTION3_SCORE_METHOD,
    OPTION3_VOTE_THRESHOLD,
    VIEW_WEIGHTS,
)


@dataclass
class Option3Prediction:
    score: float
    filtered_max: float
    weighted_average: float
    vote_count: int
    per_view_probs: np.ndarray
    score_method: str
    detected_alone: bool


def option3_score_from_probs(
    probs: np.ndarray,
    method: str = OPTION3_SCORE_METHOD,
    vote_threshold: float = OPTION3_VOTE_THRESHOLD,
) -> tuple[float, float, float, int]:
    weights = np.array(VIEW_WEIGHTS, dtype=np.float32)
    weighted_average = float(weights @ probs)
    filtered_max = float(probs[1:].max())
    vote_count = int((probs > vote_threshold).sum())
    if method == "filtered_max":
        score = filtered_max
    elif method == "voting":
        score = vote_count / float(len(probs))
    elif method == "weighted_average":
        score = weighted_average
    else:
        raise ValueError(f"Unknown Option 3 score method: {method}")
    return float(score), filtered_max, weighted_average, vote_count


@torch.no_grad()
def predict_option3(option3: dict, audio: np.ndarray, device=None, method: str = OPTION3_SCORE_METHOD) -> Option3Prediction:
    if device is None:
        device = next(option3["models"][0].parameters()).device
    if np.abs(audio).max() < NOISE_FLOOR:
        probs = np.zeros(5, dtype=np.float32)
        return Option3Prediction(0.0, 0.0, 0.0, 0, probs, method, False)

    views = create_audio_views(audio)
    probs = np.zeros(5, dtype=np.float32)
    drone_idx = option3["drone_idx"]
    for vi, (model, view) in enumerate(zip(option3["models"], views)):
        lm = audio_to_logmel(view)
        x = lm.unsqueeze(0).unsqueeze(0).to(device)
        probs[vi] = torch.softmax(model(x), dim=1)[0, drone_idx].item()

    score, filtered_max, weighted_average, vote_count = option3_score_from_probs(probs, method)
    detected_alone = (
        filtered_max > OPTION3_ALONE_FMAX_THR
        or weighted_average > OPTION3_ALONE_SCORE_THR
        or vote_count >= OPTION3_ALONE_VOTES_NEED
    )
    return Option3Prediction(
        score=score,
        filtered_max=filtered_max,
        weighted_average=weighted_average,
        vote_count=vote_count,
        per_view_probs=probs,
        score_method=method,
        detected_alone=detected_alone,
    )
