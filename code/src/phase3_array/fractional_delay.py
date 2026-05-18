"""Fractional-delay helpers."""

import numpy as np


def apply_fractional_delay(x: np.ndarray, delay_samples: float) -> np.ndarray:
    """
    Delay a 1D signal by a fractional number of samples.

    Positive delay_samples means the output occurs later in time.
    """
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if x.size == 0:
        return x.copy()
    if not np.isfinite(delay_samples):
        raise ValueError(f"delay_samples must be finite, got {delay_samples}")
    n = np.arange(x.size, dtype=np.float64)
    y = np.interp(n - float(delay_samples), n, x.astype(np.float64), left=0.0, right=0.0)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    return y.astype(np.float32)


def apply_fractional_delay_sinc_placeholder(x: np.ndarray, delay_samples: float) -> np.ndarray:
    """Future placeholder for FIR/sinc fractional delay."""
    return apply_fractional_delay(x, delay_samples)
