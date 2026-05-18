"""Beam scanning over an azimuth/elevation grid."""

import numpy as np
import scipy.signal as sig

from .delay_and_sum import delay_and_sum_beamform
from .hybrid_detector_wrapper import predict_hybrid_on_mono_window


def drone_band_energy(x: np.ndarray, fs: int, fmin=200.0, fmax=6000.0) -> float:
    x = np.asarray(x, dtype=np.float32)
    if x.size < 8:
        return 0.0
    nperseg = min(512, x.size)
    freqs, psd = sig.welch(x, fs=fs, nperseg=nperseg)
    mask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(mask):
        return 0.0
    return float(np.trapz(psd[mask], freqs[mask]))


def beam_scan_window(x_window_multi, fs, mic_positions, directions, config, hybrid_detector=None):
    beams = []
    energies = []
    debug_by_dir = []

    for direction in directions:
        beam, dbg = delay_and_sum_beamform(
            x_window_multi,
            fs,
            mic_positions,
            direction["az_deg"],
            direction["el_deg"],
            speed_of_sound=getattr(config, "speed_of_sound", 343.0),
        )
        beams.append(beam)
        energies.append(drone_band_energy(beam, fs))
        debug_by_dir.append(dbg)

    energies = np.asarray(energies, dtype=np.float32)
    n_dirs = len(directions)
    hybrid_scores = np.full(n_dirs, np.nan, dtype=np.float32)
    detected_flags = np.zeros(n_dirs, dtype=bool)
    hybrid_debug = [None] * n_dirs

    if hybrid_detector is not None:
        use_prefilter = bool(getattr(config, "use_energy_prefilter", True))
        max_full = int(getattr(config, "max_directions_for_full_hybrid", n_dirs))
        if (not use_prefilter) or n_dirs <= max_full:
            selected = list(range(n_dirs))
        else:
            top_k = min(int(getattr(config, "top_k_beams_for_hybrid", 5)), n_dirs)
            selected = list(np.argsort(energies)[-top_k:][::-1])

        for idx in selected:
            score, detected, dbg = predict_hybrid_on_mono_window(beams[idx], fs, hybrid_detector)
            hybrid_scores[idx] = score
            detected_flags[idx] = detected
            hybrid_debug[idx] = dbg

    valid = np.isfinite(hybrid_scores)
    if valid.any():
        best_idx = int(np.nanargmax(hybrid_scores))
        best_score = float(hybrid_scores[best_idx])
    else:
        best_idx = int(np.argmax(energies)) if n_dirs else -1
        best_score = float(energies[best_idx]) if n_dirs else 0.0

    if best_idx >= 0:
        best_dir = directions[best_idx]
        best_az = float(best_dir["az_deg"])
        best_el = float(best_dir["el_deg"])
        best_energy = float(energies[best_idx])
    else:
        best_az = best_el = best_energy = 0.0

    return {
        "directions": directions,
        "beam_energy_scores": energies,
        "hybrid_scores": hybrid_scores,
        "detected_flags": detected_flags,
        "best_az": best_az,
        "best_el": best_el,
        "best_score": best_score,
        "best_direction_index": best_idx,
        "best_detected": bool(detected_flags[best_idx]) if best_idx >= 0 else False,
        "beam_energy_best": best_energy,
        "per_direction_debug": debug_by_dir,
        "hybrid_debug": hybrid_debug,
        "beamformed_best": beams[best_idx] if best_idx >= 0 else None,
    }
