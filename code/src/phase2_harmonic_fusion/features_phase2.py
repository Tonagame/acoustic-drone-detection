from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.harmonic_guard.harmonic_analyzer import analyze_harmonics
from src.phase2v5_real_noise.audio_phase2v5 import AudioPreprocessor
from src.phase2v5_real_noise.config_phase2v5 import VIEW_WEIGHTS
from src.phase2v5_real_noise.data_phase2v5 import build_pools
from src.phase2v5_real_noise.model_phase2v5 import DroneCNNV5, extract_model_state

from .config_phase2 import HARMONIC_FEATURE_NAMES, SAMPLE_RATE, SNR_LEVELS


class PoolArgs:
    def __init__(self, max_drone_files, max_nodrone_files, max_fsd_clips_per_label, seed):
        self.max_drone_files = max_drone_files
        self.max_nodrone_files = max_nodrone_files
        self.max_fsd_clips_per_label = max_fsd_clips_per_label
        self.seed = seed


def load_backbone(path: Path, device) -> tuple[DroneCNNV5, dict]:
    ckpt = torch.load(str(path), map_location=device, weights_only=False)
    model = DroneCNNV5(n_classes=2).to(device)
    model.load_state_dict(extract_model_state(ckpt))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, ckpt


def harmonic_vector(audio: np.ndarray) -> np.ndarray:
    h = analyze_harmonics(audio)
    return np.asarray(
        [
            np.clip(h.f0_hz / 150.0, 0.0, 1.5),
            h.hps_confidence,
            h.low_band_ratio,
            h.harmonicity_score,
            h.upper_harmonic_explained_ratio,
            h.impulse_score,
            h.vehicle_risk_score,
            np.clip(h.num_harmonics / 80.0, 0.0, 2.0),
        ],
        dtype=np.float32,
    )


@torch.no_grad()
def weighted_cnn_latent(model: DroneCNNV5, preproc: AudioPreprocessor, audio: np.ndarray, device) -> np.ndarray:
    latents = []
    for view in preproc.create_audio_views(audio):
        lm = preproc.audio_to_logmel(view).unsqueeze(0).unsqueeze(0).to(device)
        latents.append(model.encode(lm).squeeze(0).detach().cpu().numpy())
    stacked = np.stack(latents, axis=0).astype(np.float32)
    return (VIEW_WEIGHTS.reshape(-1, 1) * stacked).sum(axis=0).astype(np.float32)


def sample_audio(condition: str, drone_pool, fsd_pool, nodrone_pool, preproc: AudioPreprocessor, rng) -> np.ndarray:
    if condition == "drone_alone":
        return drone_pool.sample_window(rng)
    if condition == "drone_plus_fsd":
        drone = drone_pool.sample_window(rng)
        noise = fsd_pool.sample_window(rng)
        return preproc.mix_at_snr(drone, noise, rng.choice(SNR_LEVELS))
    if condition == "fsd_alone":
        return fsd_pool.sample_window(rng)
    if condition == "nodrone_alone":
        return nodrone_pool.sample_window(rng) if nodrone_pool is not None else fsd_pool.sample_window(rng)
    raise ValueError(f"Unknown condition: {condition}")


def build_feature_matrix(
    backbone_path: Path,
    out_path: Path,
    examples_per_class: int,
    max_drone_files: int,
    max_nodrone_files: int,
    max_fsd_clips_per_label: int,
    seed: int,
    device,
) -> Path:
    preproc = AudioPreprocessor(SAMPLE_RATE)
    pool_args = PoolArgs(max_drone_files, max_nodrone_files, max_fsd_clips_per_label, seed)
    drone_pool, fsd_pool, nodrone_pool = build_pools(pool_args, preproc)
    backbone, ckpt = load_backbone(backbone_path, device)
    rng = random.Random(seed)

    x_rows = []
    y_rows = []
    cond_rows = []
    score_rows = []
    total = examples_per_class * 2
    print(f"Building Phase 2 features -> {out_path}")
    print(f"Examples: {total} ({examples_per_class}/class), FSD clips={len(fsd_pool.records)}")

    for i in range(total):
        positive = i < examples_per_class
        if positive:
            condition = "drone_plus_fsd" if rng.random() < 0.85 else "drone_alone"
            label = 0
        else:
            condition = "fsd_alone" if rng.random() < 0.90 else "nodrone_alone"
            label = 1
        audio = sample_audio(condition, drone_pool, fsd_pool, nodrone_pool, preproc, rng)
        cnn = weighted_cnn_latent(backbone, preproc, audio, device)
        harm = harmonic_vector(audio)
        x_rows.append(np.concatenate([cnn, harm], axis=0))
        y_rows.append(label)
        cond_rows.append(condition)
        with torch.no_grad():
            logits = backbone.classify_from_latent(torch.from_numpy(cnn).float().unsqueeze(0).to(device))
            score_rows.append(float(torch.softmax(logits, dim=1)[0, 0].item()))
        if (i + 1) % 500 == 0:
            print(f"  features {i + 1}/{total}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        X=np.stack(x_rows).astype(np.float32),
        y=np.asarray(y_rows, dtype=np.int64),
        condition=np.asarray(cond_rows),
        backbone_score=np.asarray(score_rows, dtype=np.float32),
        harmonic_feature_names=np.asarray(HARMONIC_FEATURE_NAMES),
        backbone_path=str(backbone_path),
        backbone_phase=ckpt.get("phase", ""),
    )
    return out_path
