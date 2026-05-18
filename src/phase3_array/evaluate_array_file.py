"""Evaluate one recorded multichannel array WAV file."""

import csv
import json
from collections import Counter, deque
from pathlib import Path

import numpy as np
import scipy.signal as sig
import soundfile as sf
import torch
import torchaudio.functional as FA

from .array_geometry import load_array_geometry
from .beam_scan import beam_scan_window
from .config_phase3 import beam_scan_results_dir, hop_samples, plots_dir, sample_rate_target, window_samples
from .direction_grid import make_direction_grid
from .hybrid_detector_wrapper import load_hybrid_detector
from .plot_phase3_results import plot_array_evaluation


def _read_wav(path: Path):
    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return audio.astype(np.float32), int(sr)


def _resample_multi(x: np.ndarray, src_fs: int, dst_fs: int) -> np.ndarray:
    if src_fs == dst_fs:
        return x.astype(np.float32)
    t = torch.from_numpy(x.T.astype(np.float32))
    y = FA.resample(t, src_fs, dst_fs).T.numpy()
    return y.astype(np.float32)


def _apply_common_highpass(x: np.ndarray, fs: int, highpass_hz):
    if highpass_hz is None:
        return x
    sos = sig.butter(4, float(highpass_hz), btype="high", fs=fs, output="sos")
    y = np.zeros_like(x, dtype=np.float32)
    for ch in range(x.shape[1]):
        y[:, ch] = sig.sosfilt(sos, x[:, ch]).astype(np.float32)
    return y


class Phase3Smoother:
    def __init__(self, mode: str, hop_sec: float):
        self.mode = mode
        self.hop_sec = hop_sec
        self.history = deque()

    def update(self, detected: bool) -> bool:
        if self.mode == "none":
            return bool(detected)
        if self.mode == "2_of_3":
            self.history.append(bool(detected))
            while len(self.history) > 3:
                self.history.popleft()
            return len(self.history) == 3 and sum(self.history) >= 2
        if self.mode == "3_of_5":
            self.history.append(bool(detected))
            while len(self.history) > 5:
                self.history.popleft()
            return len(self.history) == 5 and sum(self.history) >= 3
        if self.mode == "persist_1_5s":
            needed = max(1, round(1.5 / self.hop_sec))
            self.history.append(bool(detected))
            while len(self.history) > needed:
                self.history.popleft()
            return len(self.history) == needed and all(self.history)
        raise ValueError(f"Unknown smoothing mode: {self.mode}")


def _intervals(rows):
    intervals = []
    start = None
    last_end = None
    for row in rows:
        if row["smoothed_detected"] and start is None:
            start = row["time_start"]
        if not row["smoothed_detected"] and start is not None:
            intervals.append([start, last_end])
            start = None
        last_end = row["time_end"]
    if start is not None:
        intervals.append([start, last_end])
    return intervals


def _write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _jsonable_array(value):
    if value is None:
        return []
    if isinstance(value, str) and value == "":
        return []
    try:
        return np.asarray(value, dtype=float).reshape(-1).tolist()
    except Exception:
        return []


def evaluate_array_wav(wav_path, config, hybrid_detector=None):
    wav_path = Path(wav_path)
    config.ensure_output_dirs()
    if hybrid_detector is None:
        hybrid_detector = load_hybrid_detector(config)

    x, sr = _read_wav(wav_path)
    duration_sec = x.shape[0] / max(sr, 1)
    if duration_sec > getattr(config, "warn_if_file_longer_sec", 1800):
        print(f"[WARN] {wav_path.name} is {duration_sec/60:.1f} min; current implementation reads it fully.")
    x = _resample_multi(x, sr, getattr(config, "sample_rate_target", sample_rate_target))
    fs = getattr(config, "sample_rate_target", sample_rate_target)
    x = _apply_common_highpass(x, fs, getattr(config, "optional_highpass_hz", None))

    mic_positions = load_array_geometry(config, expected_channels=x.shape[1])
    directions = make_direction_grid(
        getattr(config, "azimuth_grid_deg"),
        getattr(config, "elevation_grid_deg"),
    )

    win = window_samples()
    hop = hop_samples()
    smoother = Phase3Smoother(getattr(config, "smoothing_mode", "2_of_3"), getattr(config, "hop_sec", 0.5))

    rows = []
    dir_rows = []
    for start in range(0, x.shape[0] - win + 1, hop):
        end = start + win
        t0 = start / fs
        t1 = end / fs
        result = beam_scan_window(
            x[start:end, :],
            fs,
            mic_positions,
            directions,
            config,
            hybrid_detector=hybrid_detector,
        )
        raw_det = bool(result["best_detected"])
        smooth_det = smoother.update(raw_det)
        idx = result["best_direction_index"]
        hdbg = result["hybrid_debug"][idx] if idx >= 0 and result["hybrid_debug"][idx] else {}

        rows.append({
            "time_start": t0,
            "time_end": t1,
            "best_az": result["best_az"],
            "best_el": result["best_el"],
            "best_score": result["best_score"],
            "raw_detected": int(raw_det),
            "smoothed_detected": int(smooth_det),
            "option2_score_best": hdbg.get("option2_score", ""),
            "option3_score_best": hdbg.get("option3_score", ""),
            "option3_filtered_max_best": hdbg.get("option3_filtered_max", ""),
            "option3_weighted_average_best": hdbg.get("option3_weighted_average", ""),
            "option3_vote_count_best": hdbg.get("option3_vote_count", ""),
            "option3_per_view_probs_best": json.dumps(_jsonable_array(hdbg.get("option3_per_view_probs", []))),
            "phase2_score_best": hdbg.get("phase2_score", ""),
            "phase2_threshold_best": hdbg.get("phase2_threshold", ""),
            "vehicle_risk_score_best": hdbg.get("vehicle_risk_score", ""),
            "f0_norm_best": hdbg.get("f0_norm", ""),
            "harmonicity_score_best": hdbg.get("harmonicity_score", ""),
            "fusion_reason_best": hdbg.get("fusion_reason", ""),
            "vetoed_best": int(bool(hdbg.get("vetoed", False))),
            "beam_energy_best": result["beam_energy_best"],
        })

        for di, direction in enumerate(directions):
            ddbg = result["hybrid_debug"][di] if result["hybrid_debug"][di] else {}
            dir_rows.append({
                "time_start": t0,
                "az": direction["az_deg"],
                "el": direction["el_deg"],
                "beam_energy": float(result["beam_energy_scores"][di]),
                "hybrid_score": "" if not np.isfinite(result["hybrid_scores"][di]) else float(result["hybrid_scores"][di]),
                "detected": int(result["detected_flags"][di]),
                "option2_score": ddbg.get("option2_score", ""),
                "option3_score": ddbg.get("option3_score", ""),
                "phase2_score": ddbg.get("phase2_score", ""),
                "vehicle_risk_score": ddbg.get("vehicle_risk_score", ""),
                "vetoed": int(bool(ddbg.get("vetoed", False))),
                "fusion_reason": ddbg.get("fusion_reason", ""),
            })

    base = wav_path.stem
    csv_path = beam_scan_results_dir / f"{base}_per_window.csv"
    dir_csv_path = beam_scan_results_dir / f"{base}_direction_scores.csv"
    summary_path = beam_scan_results_dir / f"{base}_summary.json"
    _write_csv(csv_path, rows)
    _write_csv(dir_csv_path, dir_rows)

    scores = [float(r["best_score"]) for r in rows]
    smoothed_count = sum(int(r["smoothed_detected"]) for r in rows)
    raw_count = sum(int(r["raw_detected"]) for r in rows)
    common_dir = Counter((r["best_az"], r["best_el"]) for r in rows).most_common(1)
    summary = {
        "file": str(wav_path),
        "sample_rate": fs,
        "channels": int(x.shape[1]),
        "duration_sec": duration_sec,
        "num_windows": len(rows),
        "max_score": max(scores) if scores else 0.0,
        "average_score": float(np.mean(scores)) if scores else 0.0,
        "detected_windows": raw_count,
        "smoothed_detected_windows": smoothed_count,
        "most_common_direction": {
            "az": common_dir[0][0][0],
            "el": common_dir[0][0][1],
            "count": common_dir[0][1],
        } if common_dir else None,
        "detection_intervals": _intervals(rows),
        "per_window_csv": str(csv_path),
        "direction_scores_csv": str(dir_csv_path),
        "detector_mode": getattr(config, "detector_mode", "hybrid_option2_option3"),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot_array_evaluation(csv_path, dir_csv_path, plots_dir)
    return summary
