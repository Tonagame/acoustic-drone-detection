"""DSP harmonic analyzer for tank/engine/generator/vehicle-like audio.

The analyzer is intentionally non-destructive. It estimates low-frequency
fundamental structure and returns features that can be used by a guard layer.
It does not suppress harmonics or alter the waveform sent to the drone CNNs.
"""

from dataclasses import asdict, dataclass

import numpy as np
import scipy.signal as sig

from .config_harmonic_guard import (
    FS,
    HARMONIC_MAX_HZ,
    HARMONIC_TOLERANCE_HZ,
    HPS_HARMONICS,
    LOW_F0_MAX_HZ,
    LOW_F0_MIN_HZ,
)


@dataclass
class HarmonicFeatures:
    f0_hz: float
    hps_confidence: float
    low_band_ratio: float
    harmonicity_score: float
    upper_harmonic_explained_ratio: float
    impulse_score: float
    vehicle_risk_score: float
    num_harmonics: int
    harmonic_ladder_hz: list[float]

    def to_dict(self):
        d = asdict(self)
        d["harmonic_ladder_hz"] = ";".join(f"{x:.1f}" for x in self.harmonic_ladder_hz)
        return d


_BP_LOW = sig.butter(4, [LOW_F0_MIN_HZ, LOW_F0_MAX_HZ], btype="band", fs=FS, output="sos")


def _safe_norm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    x = x - float(np.mean(x)) if x.size else x
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    return (x / peak).astype(np.float32) if peak > 1e-7 else x.astype(np.float32)


def low_f0_bandpass(audio: np.ndarray) -> np.ndarray:
    """Return the 30-150 Hz bandpassed view used by the harmonic analyzer."""
    x = _safe_norm(audio)
    if x.size < 8:
        return np.zeros_like(x, dtype=np.float32)
    return sig.sosfilt(_BP_LOW, x).astype(np.float32)


def _spectrum(audio: np.ndarray):
    x = _safe_norm(audio)
    if x.size < 16:
        return np.array([0.0]), np.array([0.0])
    n_fft = max(4096, int(2 ** np.ceil(np.log2(len(x)))))
    win = np.hanning(len(x)).astype(np.float32)
    mag = np.abs(np.fft.rfft(x * win, n=n_fft)).astype(np.float64)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / FS)
    mag += 1e-12
    return freqs, mag


def _interp_mag(freqs: np.ndarray, mag: np.ndarray, hz: float) -> float:
    if hz <= freqs[0] or hz >= freqs[-1]:
        return 0.0
    return float(np.interp(hz, freqs, mag))


def estimate_f0_hps(audio: np.ndarray) -> tuple[float, float]:
    """Estimate low fundamental using a log Harmonic Product Spectrum."""
    freqs, mag = _spectrum(low_f0_bandpass(audio))
    candidates = np.arange(LOW_F0_MIN_HZ, LOW_F0_MAX_HZ + 0.25, 0.5)
    scores = np.zeros_like(candidates, dtype=np.float64)
    log_mag = np.log(mag)
    for i, f0 in enumerate(candidates):
        total = 0.0
        weight_sum = 0.0
        for h in range(1, HPS_HARMONICS + 1):
            hz = f0 * h
            if hz > LOW_F0_MAX_HZ:
                break
            total += _interp_mag(freqs, log_mag, hz) / h
            weight_sum += 1.0 / h
        scores[i] = total / max(weight_sum, 1e-12)

    best_i = int(np.argmax(scores))
    best = float(scores[best_i])
    median = float(np.median(scores))
    mad = float(np.median(np.abs(scores - median))) + 1e-9
    confidence = float(np.clip((best - median) / (8.0 * mad), 0.0, 1.0))
    return float(candidates[best_i]), confidence


def _band_energy(freqs: np.ndarray, mag: np.ndarray, fmin: float, fmax: float) -> float:
    mask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(mask):
        return 0.0
    power = mag[mask] ** 2
    return float(np.trapz(power, freqs[mask]))


def _harmonic_energy(freqs: np.ndarray, mag: np.ndarray, ladder: list[float], tolerance_hz: float) -> float:
    total = 0.0
    for hz in ladder:
        mask = (freqs >= hz - tolerance_hz) & (freqs <= hz + tolerance_hz)
        if np.any(mask):
            total += float(np.trapz(mag[mask] ** 2, freqs[mask]))
    return total


def _impulse_score(audio: np.ndarray) -> float:
    """Approximate clank/impulse content with frame RMS crest factor."""
    x = _safe_norm(audio)
    if len(x) < 256:
        return 0.0
    frame = 256
    hop = 128
    vals = []
    for start in range(0, len(x) - frame + 1, hop):
        seg = x[start:start + frame]
        vals.append(float(np.sqrt(np.mean(seg * seg) + 1e-12)))
    vals = np.asarray(vals, dtype=np.float32)
    crest = float(vals.max() / (vals.mean() + 1e-9))
    return float(np.clip((crest - 2.0) / 4.0, 0.0, 1.0))


def analyze_harmonics(audio: np.ndarray) -> HarmonicFeatures:
    f0, hps_conf = estimate_f0_hps(audio)
    freqs, mag = _spectrum(audio)

    low_energy = _band_energy(freqs, mag, LOW_F0_MIN_HZ, LOW_F0_MAX_HZ)
    broad_energy = _band_energy(freqs, mag, LOW_F0_MIN_HZ, HARMONIC_MAX_HZ)
    upper_energy = _band_energy(freqs, mag, LOW_F0_MAX_HZ, HARMONIC_MAX_HZ)
    low_band_ratio = float(np.clip(low_energy / (broad_energy + 1e-12), 0.0, 1.0))

    ladder = []
    h = 1
    while f0 * h <= HARMONIC_MAX_HZ:
        ladder.append(float(f0 * h))
        h += 1
    harmonic_energy = _harmonic_energy(freqs, mag, ladder, HARMONIC_TOLERANCE_HZ)
    upper_ladder = [hz for hz in ladder if hz > LOW_F0_MAX_HZ]
    upper_harm_energy = _harmonic_energy(freqs, mag, upper_ladder, HARMONIC_TOLERANCE_HZ)

    harmonicity = float(np.clip(harmonic_energy / (broad_energy + 1e-12), 0.0, 1.0))
    upper_explained = float(np.clip(upper_harm_energy / (upper_energy + 1e-12), 0.0, 1.0))
    impulse = _impulse_score(audio)

    risk = (
        0.25 * hps_conf
        + 0.22 * low_band_ratio
        + 0.25 * harmonicity
        + 0.23 * upper_explained
        + 0.05 * impulse
    )
    risk = float(np.clip(risk, 0.0, 1.0))

    return HarmonicFeatures(
        f0_hz=f0,
        hps_confidence=hps_conf,
        low_band_ratio=low_band_ratio,
        harmonicity_score=harmonicity,
        upper_harmonic_explained_ratio=upper_explained,
        impulse_score=impulse,
        vehicle_risk_score=risk,
        num_harmonics=len(ladder),
        harmonic_ladder_hz=ladder,
    )
