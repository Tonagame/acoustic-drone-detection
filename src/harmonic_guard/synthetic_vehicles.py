"""Synthetic generator/vehicle signals for guard-only smoke testing.

These are not replacements for real field data. They only help exercise the
harmonic analyzer when no real generator/vehicle recordings are available.
"""

import numpy as np

from .config_harmonic_guard import FS, WIN_SAMPLES


def _lp(s, taps):
    return np.convolve(s, np.ones(taps) / taps, mode="same")


def _norm(s, level=0.85):
    s = np.asarray(s, dtype=np.float32)
    peak = float(np.max(np.abs(s))) if s.size else 0.0
    return (s / peak * level).astype(np.float32) if peak > 1e-7 else s


def synth_generator(n=WIN_SAMPLES, t0=0.0):
    rng = np.random.default_rng(int(t0 * 1000 + 501) % 99991)
    f0 = rng.uniform(48.0, 62.0)
    t = np.linspace(t0, t0 + n / FS, n, endpoint=False)
    wobble = 1.0 + 0.015 * np.sin(2 * np.pi * 0.8 * t)
    ph = np.cumsum(wobble) * (f0 / FS) * 2 * np.pi
    hum = (
        0.65 * np.sin(ph)
        + 0.34 * np.sin(2 * ph)
        + 0.20 * np.sin(3 * ph)
        + 0.10 * np.sin(4 * ph)
        + 0.05 * np.sin(5 * ph)
    )
    buzz = _lp(rng.standard_normal(n), max(1, int(FS / 1200))) * 0.22
    return _norm(hum + buzz, 0.9)


def synth_vehicle(n=WIN_SAMPLES, t0=0.0):
    rng = np.random.default_rng(int(t0 * 1000 + 733) % 99991)
    f0 = rng.uniform(70.0, 115.0)
    t = np.linspace(t0, t0 + n / FS, n, endpoint=False)
    rpm = 1.0 + 0.07 * np.sin(2 * np.pi * 1.1 * t + 0.3)
    ph = np.cumsum(rpm) * (f0 / FS) * 2 * np.pi
    engine = (
        0.52 * np.sin(ph)
        + 0.27 * np.sin(2 * ph)
        + 0.16 * np.sin(3 * ph)
        + 0.08 * np.sin(4 * ph)
    )
    road = _lp(rng.standard_normal(n), max(1, int(FS / 700))) * 0.35
    ticks = np.zeros(n, dtype=np.float32)
    for pos in range(0, n, int(FS * 0.08)):
        b = min(int(rng.integers(2, 12)), n - pos)
        ticks[pos:pos + b] += rng.standard_normal(b).astype(np.float32) * rng.uniform(0.02, 0.12)
    return _norm(engine + road + ticks, 0.85)
