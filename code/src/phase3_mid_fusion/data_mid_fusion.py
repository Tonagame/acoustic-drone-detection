from __future__ import annotations

import random

import torch
from torch.utils.data import Dataset

from src.phase2v5_real_noise.audio_phase2v5 import AudioPreprocessor
from src.phase2v5_real_noise.data_phase2v5 import AudioFileWindowPool, FSD50KWindowPool


class RealNoiseMidFusionDataset(Dataset):
    """Returns all five log-mel views for one sampled 1-second audio window."""

    def __init__(
        self,
        preproc: AudioPreprocessor,
        drone_pool: AudioFileWindowPool,
        fsd_pool: FSD50KWindowPool,
        nodrone_pool: AudioFileWindowPool | None,
        examples_per_class: int,
        snr_levels: list[int],
        augment: bool = True,
        seed: int = 6101,
        positive_mix_prob: float = 0.95,
        negative_fsd_prob: float = 0.95,
    ):
        self.preproc = preproc
        self.drone_pool = drone_pool
        self.fsd_pool = fsd_pool
        self.nodrone_pool = nodrone_pool
        self.examples_per_class = int(examples_per_class)
        self.snr_levels = list(snr_levels)
        self.augment = bool(augment)
        self.seed = int(seed)
        self.positive_mix_prob = float(positive_mix_prob)
        self.negative_fsd_prob = float(negative_fsd_prob)

    def __len__(self) -> int:
        return self.examples_per_class * 2

    def _rng(self, idx: int):
        if self.augment:
            return random
        return random.Random(self.seed + idx * 104729)

    def __getitem__(self, idx: int):
        rng = self._rng(idx)
        is_drone = idx < self.examples_per_class
        if is_drone:
            drone = self.drone_pool.sample_window(rng)
            audio = drone
            if rng.random() < self.positive_mix_prob:
                noise = self.fsd_pool.sample_window(rng)
                audio = self.preproc.mix_at_snr(drone, noise, rng.choice(self.snr_levels))
            label = 0
        else:
            use_fsd = self.nodrone_pool is None or rng.random() < self.negative_fsd_prob
            audio = self.fsd_pool.sample_window(rng) if use_fsd else self.nodrone_pool.sample_window(rng)
            label = 1

        views = self.preproc.create_audio_views(audio)
        tensors = [self.preproc.audio_to_logmel(view).unsqueeze(0).float() for view in views]
        return torch.stack(tensors, dim=0), torch.tensor(label, dtype=torch.long)
