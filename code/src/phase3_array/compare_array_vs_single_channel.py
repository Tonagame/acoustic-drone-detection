"""Compare beamformed array detection to a single-channel baseline."""

import csv
import json
from pathlib import Path

import numpy as np
import scipy.signal as sig
import soundfile as sf
import torch
import torchaudio.functional as FA

from .config_phase3 import comparisons_dir, hop_samples, sample_rate_target, window_samples
from .hybrid_detector_wrapper import load_hybrid_detector, predict_hybrid_on_mono_window
from .plot_phase3_results import plot_comparison


def _read_csv(path: Path):
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _read_wav(path: Path):
    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return audio.astype(np.float32), int(sr)


def _resample_multi(x: np.ndarray, src_fs: int, dst_fs: int):
    if src_fs == dst_fs:
        return x.astype(np.float32)
    t = torch.from_numpy(x.T.astype(np.float32))
    return FA.resample(t, src_fs, dst_fs).T.numpy().astype(np.float32)


def _apply_highpass(x: np.ndarray, fs: int, highpass_hz):
    if highpass_hz is None:
        return x
    sos = sig.butter(4, float(highpass_hz), btype="high", fs=fs, output="sos")
    y = np.zeros_like(x, dtype=np.float32)
    for ch in range(x.shape[1]):
        y[:, ch] = sig.sosfilt(sos, x[:, ch]).astype(np.float32)
    return y


def compare_array_vs_single_channel(wav_path, array_per_window_csv, config, hybrid_detector=None):
    wav_path = Path(wav_path)
    config.ensure_output_dirs()
    if hybrid_detector is None:
        hybrid_detector = load_hybrid_detector(config)

    x, sr = _read_wav(wav_path)
    fs = getattr(config, "sample_rate_target", sample_rate_target)
    x = _resample_multi(x, sr, fs)
    x = _apply_highpass(x, fs, getattr(config, "optional_highpass_hz", None))

    ch = int(getattr(config, "single_channel_index", 0))
    if ch < 0 or ch >= x.shape[1]:
        raise ValueError(f"single_channel_index={ch} outside channel range 0..{x.shape[1]-1}")

    win = window_samples()
    hop = hop_samples()
    rows = []
    for wi, start in enumerate(range(0, x.shape[0] - win + 1, hop)):
        end = start + win
        score, detected, debug = predict_hybrid_on_mono_window(x[start:end, ch], fs, hybrid_detector)
        rows.append({
            "time_start": start / fs,
            "time_end": end / fs,
            "single_channel_score": score,
            "single_channel_detected": int(detected),
            "single_option2_score": debug.get("option2_score", ""),
            "single_option3_score": debug.get("option3_score", ""),
            "single_phase2_score": debug.get("phase2_score", ""),
            "single_vehicle_risk_score": debug.get("vehicle_risk_score", ""),
            "single_fusion_reason": debug.get("fusion_reason", ""),
            "single_vetoed": int(bool(debug.get("vetoed", False))),
        })

    array_rows = _read_csv(Path(array_per_window_csv))
    merged = []
    for i, row in enumerate(rows):
        arr = array_rows[i] if i < len(array_rows) else {}
        merged.append({
            **row,
            "beamformed_score": arr.get("best_score", ""),
            "beamformed_raw_detected": arr.get("raw_detected", ""),
            "beamformed_smoothed_detected": arr.get("smoothed_detected", ""),
            "best_az": arr.get("best_az", ""),
            "best_el": arr.get("best_el", ""),
            "beam_option2_score": arr.get("option2_score_best", ""),
            "beam_option3_score": arr.get("option3_score_best", ""),
            "beam_phase2_score": arr.get("phase2_score_best", ""),
            "beam_vehicle_risk_score": arr.get("vehicle_risk_score_best", ""),
            "beam_fusion_reason": arr.get("fusion_reason_best", ""),
            "beam_vetoed": arr.get("vetoed_best", ""),
            "score_delta_beam_minus_single": (
                float(arr.get("best_score", 0.0)) - float(row["single_channel_score"])
                if arr.get("best_score", "") != "" else ""
            ),
        })

    base = wav_path.stem
    out_csv = comparisons_dir / f"{base}_array_vs_single_channel.csv"
    out_json = comparisons_dir / f"{base}_array_vs_single_channel_summary.json"
    _write_csv(out_csv, merged)

    single_rate = 100.0 * np.mean([r["single_channel_detected"] for r in rows]) if rows else 0.0
    beam_raw = [float(r["beamformed_raw_detected"]) for r in merged if r["beamformed_raw_detected"] != ""]
    beam_smooth = [float(r["beamformed_smoothed_detected"]) for r in merged if r["beamformed_smoothed_detected"] != ""]
    single_scores = [float(r["single_channel_score"]) for r in rows]
    beam_scores = [float(r["beamformed_score"]) for r in merged if r["beamformed_score"] != ""]
    n_pair = min(len(beam_scores), len(single_scores))
    mean_delta = (
        float(np.mean(np.asarray(beam_scores[:n_pair]) - np.asarray(single_scores[:n_pair])))
        if n_pair else 0.0
    )
    truth_path = wav_path.with_suffix(".json")
    is_positive = True
    if truth_path.exists():
        try:
            truth = json.loads(truth_path.read_text(encoding="utf-8"))
            source_kinds = [str(s.get("kind", "")).lower() for s in truth.get("sources", [])]
            is_positive = "drone" in source_kinds
        except Exception:
            is_positive = True

    summary = {
        "file": str(wav_path),
        "single_channel_index": ch,
        "single_channel_detection_rate_percent": single_rate,
        "beamformed_raw_detection_rate_percent": 100.0 * float(np.mean(beam_raw)) if beam_raw else 0.0,
        "beamformed_smoothed_detection_rate_percent": 100.0 * float(np.mean(beam_smooth)) if beam_smooth else 0.0,
        "mean_single_channel_score": float(np.mean(single_scores)) if single_scores else 0.0,
        "mean_beamformed_score": float(np.mean(beam_scores)) if beam_scores else 0.0,
        "mean_score_delta_beam_minus_single": mean_delta,
        "comparison_csv": str(out_csv),
    }
    if is_positive:
        helped = (
            summary["beamformed_smoothed_detection_rate_percent"] > summary["single_channel_detection_rate_percent"]
            or summary["mean_beamformed_score"] > summary["mean_single_channel_score"] + 0.02
        )
        maintained = (
            abs(summary["beamformed_smoothed_detection_rate_percent"] - summary["single_channel_detection_rate_percent"]) <= 1.0
            and abs(summary["mean_beamformed_score"] - summary["mean_single_channel_score"]) <= 0.02
        )
    else:
        helped = (
            summary["beamformed_smoothed_detection_rate_percent"] < summary["single_channel_detection_rate_percent"]
            or summary["mean_beamformed_score"] < summary["mean_single_channel_score"] - 0.02
        )
        maintained = (
            abs(summary["beamformed_smoothed_detection_rate_percent"] - summary["single_channel_detection_rate_percent"]) <= 1.0
            and abs(summary["mean_beamformed_score"] - summary["mean_single_channel_score"]) <= 0.02
        )
    if helped:
        summary["verdict"] = "beamforming helped"
    elif maintained:
        summary["verdict"] = "beamforming maintained performance"
    else:
        summary["verdict"] = "beamforming did not help"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot_comparison(out_csv, config.plots_dir)
    return summary
