from __future__ import annotations

import csv
import random
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .audio_phase2v5 import AudioPreprocessor
from .config_phase2v5 import FSD_LABELS, VIEW_NAMES


def _list_wavs(folder: Path, max_files: int | None, seed: int) -> list[Path]:
    files = list(folder.glob("*.wav"))
    rng = random.Random(seed)
    rng.shuffle(files)
    return files[:max_files] if max_files else files


class SmallAudioCache:
    def __init__(self, preproc: AudioPreprocessor, max_items: int = 24):
        self.preproc = preproc
        self.max_items = max(1, int(max_items))
        self.cache: OrderedDict[str, np.ndarray] = OrderedDict()

    def get(self, path: Path) -> np.ndarray:
        key = str(path)
        if key in self.cache:
            audio = self.cache.pop(key)
            self.cache[key] = audio
            return audio
        audio = self.preproc.read_audio(path)
        self.cache[key] = audio
        while len(self.cache) > self.max_items:
            self.cache.popitem(last=False)
        return audio


class AudioFileWindowPool:
    def __init__(self, folder: Path, preproc: AudioPreprocessor, max_files: int | None, seed: int):
        self.paths = _list_wavs(folder, max_files, seed)
        self.preproc = preproc
        self.cache = SmallAudioCache(preproc)
        if not self.paths:
            raise FileNotFoundError(f"No WAV files found in {folder}")

    def sample_window(self, rng) -> np.ndarray:
        path = self.paths[rng.randrange(len(self.paths))]
        return self.preproc.random_window(self.cache.get(path), rng)


class FSD50KWindowPool:
    def __init__(
        self,
        csv_path: Path,
        preproc: AudioPreprocessor,
        labels: list[str] | None = None,
        max_clips_per_label: int | None = None,
        seed: int = 1337,
    ):
        self.preproc = preproc
        self.cache = SmallAudioCache(preproc)
        labels = labels or FSD_LABELS
        selected: list[dict] = []
        rng = random.Random(seed)
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for label in labels:
            matches = []
            for row in rows:
                matched = [x.strip() for x in str(row.get("matched_labels", "")).split(",") if x.strip()]
                if label in matched and Path(row["path"]).exists():
                    matches.append(row)
            rng.shuffle(matches)
            selected.extend(matches[:max_clips_per_label] if max_clips_per_label else matches)
        dedup = {}
        for row in selected:
            dedup[str(row["path"])] = row
        self.records = list(dedup.values())
        rng.shuffle(self.records)
        if not self.records:
            raise FileNotFoundError(f"No FSD50K candidate clips found from {csv_path}")

    def sample_window(self, rng) -> np.ndarray:
        row = self.records[rng.randrange(len(self.records))]
        audio = self.cache.get(Path(row["path"]))
        return self.preproc.random_window(audio, rng)


class RealNoiseGeneralistDataset(Dataset):
    """
    Generalist dataset for Phase 1.

    Label convention:
      0 = drone
      1 = no_drone
    """

    def __init__(
        self,
        preproc: AudioPreprocessor,
        drone_pool: AudioFileWindowPool,
        fsd_pool: FSD50KWindowPool,
        nodrone_pool: AudioFileWindowPool | None,
        examples_per_class: int,
        snr_levels: list[int],
        augment: bool = True,
        seed: int = 1337,
        positive_mix_prob: float = 0.85,
        negative_fsd_prob: float = 0.80,
        mix_positives: bool | None = None,
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
        self.mix_positives = self.augment if mix_positives is None else bool(mix_positives)

    def __len__(self) -> int:
        return self.examples_per_class * 2

    def _rng(self, idx: int):
        if self.augment:
            return random
        return random.Random(self.seed + idx * 7919)

    def __getitem__(self, idx: int):
        rng = self._rng(idx)
        is_drone = idx < self.examples_per_class
        if is_drone:
            drone = self.drone_pool.sample_window(rng)
            audio = drone
            if self.mix_positives and rng.random() < self.positive_mix_prob:
                noise = self.fsd_pool.sample_window(rng)
                audio = self.preproc.mix_at_snr(drone, noise, rng.choice(self.snr_levels))
            label = 0
        else:
            use_fsd = self.nodrone_pool is None or rng.random() < self.negative_fsd_prob
            audio = self.fsd_pool.sample_window(rng) if use_fsd else self.nodrone_pool.sample_window(rng)
            label = 1

        view_idx = rng.randrange(len(VIEW_NAMES)) if self.augment else (idx % len(VIEW_NAMES))
        audio_view = self.preproc.create_audio_views(audio)[view_idx]
        logmel = self.preproc.audio_to_logmel(audio_view)
        return logmel.unsqueeze(0).float(), torch.tensor(label, dtype=torch.long)


def build_pools(args, preproc: AudioPreprocessor):
    from .config_phase2v5 import DRONE_DIR, FSD50K_CANDIDATES_CSV, NODRONE_DIR

    drone_pool = AudioFileWindowPool(DRONE_DIR, preproc, args.max_drone_files, args.seed)
    nodrone_pool = None
    if NODRONE_DIR.exists():
        nodrone_pool = AudioFileWindowPool(NODRONE_DIR, preproc, args.max_nodrone_files, args.seed + 1)
    fsd_pool = FSD50KWindowPool(
        FSD50K_CANDIDATES_CSV,
        preproc,
        max_clips_per_label=args.max_fsd_clips_per_label,
        seed=args.seed + 2,
    )
    return drone_pool, fsd_pool, nodrone_pool
