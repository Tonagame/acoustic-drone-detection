from __future__ import annotations

import numpy as np
import torch

from .config_phase2b import CREPE_FMAX, CREPE_FMIN, CREPE_MODEL, CREPE_STEP_MS, SAMPLE_RATE


def _fallback_pyin_batch(audios: np.ndarray) -> np.ndarray:
    import librosa

    rows = []
    for x in audios:
        f0, voiced, prob = librosa.pyin(
            x.astype(np.float32),
            fmin=CREPE_FMIN,
            fmax=CREPE_FMAX,
            sr=SAMPLE_RATE,
            frame_length=1024,
            hop_length=160,
        )
        valid = np.isfinite(f0)
        vals = f0[valid] if np.any(valid) else np.asarray([], dtype=np.float32)
        periodicity = prob[valid] if prob is not None and np.any(valid) else np.asarray([], dtype=np.float32)
        rows.append(_summarize_pitch(vals, periodicity))
    return np.stack(rows).astype(np.float32)


def _summarize_pitch(pitch_hz: np.ndarray, periodicity: np.ndarray) -> np.ndarray:
    pitch_hz = np.asarray(pitch_hz, dtype=np.float32)
    periodicity = np.asarray(periodicity, dtype=np.float32)
    valid = np.isfinite(pitch_hz) & (pitch_hz > 0)
    pitch_hz = pitch_hz[valid]
    periodicity = periodicity[valid] if periodicity.shape == valid.shape else periodicity
    periodicity = periodicity[np.isfinite(periodicity)] if periodicity.size else np.asarray([], dtype=np.float32)
    if pitch_hz.size == 0:
        return np.zeros(7, dtype=np.float32)
    p50 = float(np.median(pitch_hz))
    p25 = float(np.percentile(pitch_hz, 25))
    p75 = float(np.percentile(pitch_hz, 75))
    iqr = p75 - p25
    periodicity_mean = float(np.mean(periodicity)) if periodicity.size else 0.0
    periodicity_max = float(np.max(periodicity)) if periodicity.size else 0.0
    voiced_ratio = float(np.mean(valid))
    low_pitch_ratio = float(np.mean((pitch_hz >= 50.0) & (pitch_hz <= 150.0)))
    stability = float(1.0 / (1.0 + iqr / max(p50, 1.0)))
    return np.asarray([
        np.clip(p50 / CREPE_FMAX, 0.0, 1.5),
        np.clip(iqr / CREPE_FMAX, 0.0, 1.5),
        np.clip(periodicity_mean, 0.0, 1.0),
        np.clip(periodicity_max, 0.0, 1.0),
        np.clip(voiced_ratio, 0.0, 1.0),
        np.clip(low_pitch_ratio, 0.0, 1.0),
        np.clip(stability, 0.0, 1.0),
    ], dtype=np.float32)


@torch.no_grad()
def crepe_pitch_features_batch(audios: np.ndarray, device=None, batch_size: int = 64) -> np.ndarray:
    audios = np.asarray(audios, dtype=np.float32)
    if audios.ndim == 1:
        audios = audios[None, :]
    try:
        import torchcrepe
    except Exception:
        return _fallback_pyin_batch(audios)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = []
    for start in range(0, len(audios), batch_size):
        batch = torch.from_numpy(audios[start:start + batch_size]).float().to(device)
        pitch, periodicity = torchcrepe.predict(
            batch,
            SAMPLE_RATE,
            CREPE_STEP_MS,
            CREPE_FMIN,
            CREPE_FMAX,
            CREPE_MODEL,
            return_periodicity=True,
            device=device,
            batch_size=min(batch_size, len(batch)),
        )
        pitch_np = pitch.detach().cpu().numpy()
        periodicity_np = periodicity.detach().cpu().numpy()
        for p, c in zip(pitch_np, periodicity_np):
            voiced = c >= 0.35
            vals = p[voiced]
            conf = c[voiced]
            out.append(_summarize_pitch(vals, conf))
    return np.stack(out).astype(np.float32)

