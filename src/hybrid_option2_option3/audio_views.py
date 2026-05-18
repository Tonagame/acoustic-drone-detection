"""
Audio preprocessing and scenario signal helpers for the hybrid detector.
"""

import random
from pathlib import Path

import numpy as np
import scipy.signal as sig
import soundfile as sf
import torch
import torchaudio.functional as FA
import torchaudio.transforms as T

from config_hybrid import FS, HOP_SAMPLES, NOISE_FLOOR, VIEW_NAMES, WIN_SAMPLES

_HP150 = sig.butter(4, 150, btype="high", fs=FS, output="sos")
_HP250 = sig.butter(4, 250, btype="high", fs=FS, output="sos")
_BP200 = sig.butter(4, [200, 6000], btype="band", fs=FS, output="sos")
_BP500 = sig.butter(4, [500, 6000], btype="band", fs=FS, output="sos")

_mel_cpu = T.MelSpectrogram(
    sample_rate=FS,
    n_fft=512,
    win_length=400,
    hop_length=160,
    n_mels=64,
    power=2.0,
)


def _norm_view(x: np.ndarray) -> np.ndarray:
    x = x - x.mean()
    pk = np.abs(x).max()
    return x / pk if pk > 1e-6 else x


def create_audio_views(x: np.ndarray) -> list[np.ndarray]:
    """Return raw, HPF-150, HPF-250, BPF-200-6k, BPF-500-6k views."""
    x = x.astype(np.float64)
    x = x - x.mean()
    pk = np.abs(x).max()
    if pk < 1e-6:
        return [np.zeros_like(x, dtype=np.float32)] * len(VIEW_NAMES)
    x = x / pk
    return [
        x.astype(np.float32),
        _norm_view(sig.sosfilt(_HP150, x)).astype(np.float32),
        _norm_view(sig.sosfilt(_HP250, x)).astype(np.float32),
        _norm_view(sig.sosfilt(_BP200, x)).astype(np.float32),
        _norm_view(sig.sosfilt(_BP500, x)).astype(np.float32),
    ]


def audio_to_logmel(wav: np.ndarray) -> torch.Tensor:
    pk = np.abs(wav).max()
    if pk < NOISE_FLOOR:
        return torch.zeros(64, 98)
    w = wav / pk
    t = torch.from_numpy(w.astype(np.float32)).unsqueeze(0)
    return torch.log10(_mel_cpu(t) + 1e-10).squeeze(0)


def load_wav(path: Path) -> np.ndarray:
    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != FS:
        t = torch.from_numpy(audio).unsqueeze(0)
        audio = FA.resample(t, sr, FS).squeeze(0).numpy()
    pk = np.abs(audio).max()
    return (audio / pk).astype(np.float32) if pk > 1e-4 else audio.astype(np.float32)


def window_audio(audio: np.ndarray, win=WIN_SAMPLES, hop=HOP_SAMPLES) -> list[np.ndarray]:
    return [audio[s:s + win].copy() for s in range(0, len(audio) - win + 1, hop)]


def _lp(s, taps):
    return np.convolve(s, np.ones(taps) / taps, mode="same")


def _norm(s, lv=0.85):
    p = np.abs(s).max()
    return (s / p * lv).astype(np.float32) if p > 1e-7 else s.astype(np.float32)


def synth_tank(n, t0=0.0):
    t = np.linspace(t0, t0 + n / FS, n, endpoint=False)
    rpm = 1.0 + 0.04 * np.sin(2 * np.pi * 0.3 * t)
    f0 = 45.0
    eng = (
        0.55 * np.sin(2 * np.pi * f0 * rpm * t)
        + 0.25 * np.sin(2 * np.pi * f0 * 2 * rpm * t)
        + 0.12 * np.sin(2 * np.pi * f0 * 3 * rpm * t)
        + 0.08 * np.sin(2 * np.pi * f0 * 4 * rpm * t)
    )
    clank = np.zeros(n)
    rng = np.random.default_rng(int(t0 * 100) % 9999)
    for pos in range(0, n, int(FS * 0.15)):
        b = min(int(FS * 0.01), n - pos)
        clank[pos:pos + b] = rng.standard_normal(b) * 0.4
    return _norm(eng + clank + _lp(np.random.randn(n), 64) * 0.3)


def synth_engine(n, t0=0.0):
    rng = np.random.default_rng(int(t0 * 1000 + 17) % 99991)
    f0 = rng.uniform(60.0, 120.0)
    t = np.linspace(t0, t0 + n / FS, n, endpoint=False)
    rpm = 1.0 + 0.05 * np.sin(2 * np.pi * 1.2 * t)
    ph = np.cumsum(rpm) * (f0 / FS) * 2 * np.pi
    harm = (
        0.55 * np.sin(ph)
        + 0.25 * np.sin(2 * ph)
        + 0.12 * np.sin(3 * ph)
        + 0.06 * np.sin(4 * ph)
        + 0.03 * np.sin(5 * ph)
    )
    exhaust = _lp(rng.standard_normal(n), max(1, int(FS / 2000))) * 0.7
    mech = np.zeros(n)
    pos = 0
    while pos < n:
        pos += int(rng.integers(max(1, int(FS * 0.03)), max(2, int(FS * 0.12))))
        if pos >= n:
            break
        b = min(int(rng.integers(1, 6)), n - pos)
        if b > 0:
            mech[pos:pos + b] = rng.standard_normal(b) * rng.uniform(0.05, 0.3)
    return _norm(harm + exhaust + mech)


def synth_crowd(n, t0=0.0):
    white = np.random.randn(n)
    bp = _lp(white - _lp(white, 80), 5)
    t = np.linspace(t0, t0 + n / FS, n, endpoint=False)
    am = 0.4 + 0.6 * np.abs(np.sin(2 * np.pi * 3.0 * t))
    return _norm(bp * am, 0.6)


def synth_pure_noise(n, t0=0.0):
    rng = np.random.default_rng(int(t0 * 1000 + 71) % 99991)
    return _norm(rng.standard_normal(n), 0.45)


def synth_wind(n, t0=0.0):
    rng = np.random.default_rng(int(t0 * 1000 + 31) % 99991)
    white = rng.standard_normal(n + 100)
    pink = np.cumsum(white)[100:]
    pink = np.diff(np.concatenate([[0], pink]))
    lp = sig.sosfilt(sig.butter(4, 800, btype="low", fs=FS, output="sos"), pink)
    t = np.linspace(t0, t0 + n / FS, n, endpoint=False)
    gust = 0.6 + 0.4 * np.abs(np.sin(2 * np.pi * rng.uniform(0.5, 2.0) * t))
    return _norm(lp * gust)


def mix_at_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    clean = clean.astype(np.float64)
    noise = noise.astype(np.float64)
    if len(noise) < len(clean):
        noise = np.tile(noise, int(np.ceil(len(clean) / len(noise))))
    if len(noise) > len(clean):
        start = random.randint(0, len(noise) - len(clean))
        noise = noise[start:start + len(clean)]
    pc = np.mean(clean ** 2) + 1e-12
    pn = np.mean(noise ** 2) + 1e-12
    mixed = clean + np.sqrt(pc / (pn * 10 ** (snr_db / 10.0))) * noise
    return _norm(mixed, 1.0)


def collect_wav_windows(paths: list[Path], max_windows: int) -> list[np.ndarray]:
    wins = []
    for folder in paths:
        if not folder.exists():
            continue
        for f in sorted(folder.rglob("*.wav")):
            if len(wins) >= max_windows:
                return wins[:max_windows]
            try:
                wins.extend(window_audio(load_wav(f)))
            except Exception:
                pass
    return wins[:max_windows]
