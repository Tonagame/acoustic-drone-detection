from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch

from src.phase2v5_real_noise.audio_phase2v5 import AudioPreprocessor
from src.phase2v5_real_noise.config_phase2v5 import DRONE_DIR, FSD50K_CANDIDATES_CSV, FSD_LABELS, NODRONE_DIR
from src.phase2v5_real_noise.data_phase2v5 import AudioFileWindowPool, FSD50KWindowPool
from src.phase3_real_noise_specialists.predict_phase3_hybrid import load_phase2_guard, load_specialist_bundle, predict_phase2_guard, predict_specialists

from .config_phase2b import FEATURE_NAMES, PHASE2_BACKBONE_PATH, PHASE2_GUARD_PATH, PHASE3_SPECIALIST_PATH, SAMPLE_RATE, SNR_LEVELS
from .pitch_features import crepe_pitch_features_batch


class PoolArgs:
    def __init__(self, max_drone_files, max_nodrone_files, max_fsd_clips_per_label, seed):
        self.max_drone_files = max_drone_files
        self.max_nodrone_files = max_nodrone_files
        self.max_fsd_clips_per_label = max_fsd_clips_per_label
        self.seed = seed


def build_pools(args, preproc):
    drone_pool = AudioFileWindowPool(DRONE_DIR, preproc, args.max_drone_files, args.seed)
    nodrone_pool = AudioFileWindowPool(NODRONE_DIR, preproc, args.max_nodrone_files, args.seed + 1) if NODRONE_DIR.exists() else None
    fsd_pool = FSD50KWindowPool(FSD50K_CANDIDATES_CSV, preproc, FSD_LABELS, args.max_fsd_clips_per_label, args.seed + 2)
    return drone_pool, fsd_pool, nodrone_pool


def sample_condition(condition: str, pools, preproc: AudioPreprocessor, rng):
    drone_pool, fsd_pool, nodrone_pool = pools
    if condition == "drone_alone":
        return drone_pool.sample_window(rng), 0
    if condition == "drone_plus_fsd":
        drone = drone_pool.sample_window(rng)
        noise = fsd_pool.sample_window(rng)
        return preproc.mix_at_snr(drone, noise, rng.choice(SNR_LEVELS)), 0
    if condition == "fsd_alone":
        return fsd_pool.sample_window(rng), 1
    if condition == "nodrone_alone":
        return (nodrone_pool.sample_window(rng) if nodrone_pool else fsd_pool.sample_window(rng)), 1
    raise ValueError(condition)


@torch.no_grad()
def base_feature_rows(audios: np.ndarray, specialists, guard, preproc: AudioPreprocessor):
    rows = []
    for audio in audios:
        sp = predict_specialists(specialists, preproc, audio)
        gd = predict_phase2_guard(guard, preproc, audio)
        rows.append([
            *sp.per_view_probs.tolist(),
            float(sp.score),
            float(sp.filtered_max),
            float(sp.vote_count) / 5.0,
            float(gd.score),
            float(gd.vehicle_risk_score),
            float(gd.f0_norm),
            float(gd.harmonicity_score),
        ])
    return np.asarray(rows, dtype=np.float32)


def build_feature_matrix(
    out_path: Path,
    examples_per_class: int,
    max_drone_files: int,
    max_nodrone_files: int,
    max_fsd_clips_per_label: int,
    seed: int,
    device,
    specialist_path: Path = PHASE3_SPECIALIST_PATH,
    phase2_path: Path = PHASE2_GUARD_PATH,
    backbone_path: Path = PHASE2_BACKBONE_PATH,
):
    preproc = AudioPreprocessor(SAMPLE_RATE)
    pool_args = PoolArgs(max_drone_files, max_nodrone_files, max_fsd_clips_per_label, seed)
    pools = build_pools(pool_args, preproc)
    specialists = load_specialist_bundle(specialist_path, device)
    guard = load_phase2_guard(phase2_path, backbone_path, device)
    rng = random.Random(seed)
    audios = []
    labels = []
    conditions = []
    total = examples_per_class * 2
    print(f"Building Phase 2b pitch-guard features -> {out_path}")
    print(f"Examples: {total} ({examples_per_class}/class)")
    for i in range(total):
        positive = i < examples_per_class
        if positive:
            condition = "drone_plus_fsd" if rng.random() < 0.85 else "drone_alone"
        else:
            condition = "fsd_alone" if rng.random() < 0.90 else "nodrone_alone"
        audio, label = sample_condition(condition, pools, preproc, rng)
        audios.append(audio.astype(np.float32))
        labels.append(label)
        conditions.append(condition)
    audios = np.stack(audios).astype(np.float32)
    base = base_feature_rows(audios, specialists, guard, preproc)
    pitch = crepe_pitch_features_batch(audios, device=device, batch_size=64)
    X = np.concatenate([base, pitch], axis=1).astype(np.float32)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        X=X,
        y=np.asarray(labels, dtype=np.int64),
        condition=np.asarray(conditions),
        feature_names=np.asarray(FEATURE_NAMES),
        specialist_path=str(specialist_path),
        phase2_path=str(phase2_path),
        backbone_path=str(backbone_path),
    )
    return out_path

