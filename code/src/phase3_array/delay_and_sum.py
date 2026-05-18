"""Delay-and-sum passive beamforming."""

import numpy as np

from .direction_grid import unit_vector_from_az_el
from .fractional_delay import apply_fractional_delay


def compute_delays(mic_positions: np.ndarray, unit_vector: np.ndarray, speed_of_sound: float):
    mic_positions = np.asarray(mic_positions, dtype=np.float64)
    unit_vector = np.asarray(unit_vector, dtype=np.float64)
    tau = -mic_positions @ unit_vector / float(speed_of_sound)
    tau = tau - tau.mean()
    return tau


def _safe_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = x - float(np.mean(x))
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak > 1e-6:
        x = x / peak
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def delay_and_sum_beamform(
    x_multi: np.ndarray,
    fs: int,
    mic_positions: np.ndarray,
    az_deg: float,
    el_deg: float,
    speed_of_sound: float = 343.0,
):
    x_multi = np.asarray(x_multi, dtype=np.float32)
    if x_multi.ndim != 2:
        raise ValueError(f"x_multi must be [samples, channels], got {x_multi.shape}")
    if x_multi.shape[1] != mic_positions.shape[0]:
        raise ValueError(
            f"Channel count {x_multi.shape[1]} does not match mic geometry {mic_positions.shape[0]}"
        )

    unit_vector = unit_vector_from_az_el(az_deg, el_deg)
    delays_sec = compute_delays(mic_positions, unit_vector, speed_of_sound)
    delays_samples = delays_sec * float(fs)

    aligned = []
    for ch in range(x_multi.shape[1]):
        aligned.append(apply_fractional_delay(x_multi[:, ch], -delays_samples[ch]))
    beam = np.mean(np.stack(aligned, axis=1), axis=1)
    beam = _safe_normalize(beam)

    debug = {
        "delays_sec": delays_sec,
        "delays_samples": delays_samples,
        "az_deg": float(az_deg),
        "el_deg": float(el_deg),
    }
    return beam, debug
