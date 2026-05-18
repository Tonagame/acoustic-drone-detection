from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.signal as sig
import soundfile as sf
import torch
import torchaudio.functional as FA
import torchaudio.transforms as T

from .config_phase2v5 import FS, HOP_SAMPLES, NOISE_FLOOR, VIEW_NAMES, WIN_SAMPLES


class AudioPreprocessor:
    def __init__(self, sample_rate: int = FS):
        self.sample_rate = int(sample_rate)
        self.win_samples = int(self.sample_rate)
        self.hop_samples = int(0.5 * self.sample_rate)
        self.noise_floor = NOISE_FLOOR
        self.hp150 = sig.butter(4, 150, btype="high", fs=self.sample_rate, output="sos")
        self.hp250 = sig.butter(4, 250, btype="high", fs=self.sample_rate, output="sos")
        high_6k = min(6000, int(self.sample_rate * 0.45))
        self.bp200 = sig.butter(4, [200, high_6k], btype="band", fs=self.sample_rate, output="sos")
        self.bp500 = sig.butter(4, [500, high_6k], btype="band", fs=self.sample_rate, output="sos")
        self.mel = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=512,
            win_length=400,
            hop_length=160,
            n_mels=64,
            power=2.0,
        )

    @staticmethod
    def norm_view(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        p = float(np.max(np.abs(x))) if x.size else 0.0
        return (x / p).astype(np.float32) if p > 1e-6 else x.astype(np.float32)

    def create_audio_views(self, x: np.ndarray) -> list[np.ndarray]:
        x = self.norm_view(x)
        return [
            x,
            self.norm_view(sig.sosfiltfilt(self.hp150, x).astype(np.float32)),
            self.norm_view(sig.sosfiltfilt(self.hp250, x).astype(np.float32)),
            self.norm_view(sig.sosfiltfilt(self.bp200, x).astype(np.float32)),
            self.norm_view(sig.sosfiltfilt(self.bp500, x).astype(np.float32)),
        ]

    def audio_to_logmel(self, wav: np.ndarray) -> torch.Tensor:
        wav = np.asarray(wav, dtype=np.float32)
        if len(wav) < self.win_samples:
            padded = np.zeros(self.win_samples, dtype=np.float32)
            padded[: len(wav)] = wav
            wav = padded
        t = torch.from_numpy(wav).float()
        with torch.no_grad():
            mel = self.mel(t)
            mel = torch.log(mel + 1e-6)
            mel = (mel - mel.mean()) / (mel.std() + 1e-6)
        return mel

    def read_audio(self, path: Path) -> np.ndarray:
        audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != self.sample_rate and len(audio) > 0:
            t = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)
            audio = FA.resample(t, int(sr), self.sample_rate).squeeze(0).numpy()
        return self.norm_view(audio)

    def window_audio(self, audio: np.ndarray, max_windows: int | None = None) -> list[np.ndarray]:
        wins = []
        for start in range(0, len(audio) - self.win_samples + 1, self.hop_samples):
            wins.append(audio[start : start + self.win_samples].copy())
            if max_windows and len(wins) >= max_windows:
                break
        if not wins and len(audio) > 0:
            padded = np.zeros(self.win_samples, dtype=np.float32)
            n = min(len(audio), self.win_samples)
            padded[:n] = audio[:n]
            wins.append(padded)
        return wins

    def random_window(self, audio: np.ndarray, rng) -> np.ndarray:
        if len(audio) >= self.win_samples:
            start = rng.randrange(0, len(audio) - self.win_samples + 1)
            return audio[start : start + self.win_samples].copy()
        padded = np.zeros(self.win_samples, dtype=np.float32)
        n = min(len(audio), self.win_samples)
        padded[:n] = audio[:n]
        return padded

    def mix_at_snr(self, clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
        clean = self.random_or_pad(clean)
        noise = self.random_or_pad(noise)
        pc = float(np.mean(clean**2)) + 1e-9
        pn = float(np.mean(noise**2)) + 1e-9
        target_pn = pc / (10.0 ** (float(snr_db) / 10.0))
        mixed = clean + noise * np.sqrt(target_pn / pn)
        return self.norm_view(mixed)

    def random_or_pad(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if len(x) == self.win_samples:
            return x.copy()
        if len(x) > self.win_samples:
            return x[: self.win_samples].copy()
        padded = np.zeros(self.win_samples, dtype=np.float32)
        padded[: len(x)] = x
        return padded


def legacy_window_constants(sample_rate: int):
    if int(sample_rate) == FS:
        return WIN_SAMPLES, HOP_SAMPLES
    return int(sample_rate), int(0.5 * sample_rate)

