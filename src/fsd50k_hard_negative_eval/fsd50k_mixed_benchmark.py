"""Mixed positive benchmark: real DADS drone + real FSD50K negative audio."""

import argparse
import csv
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio.functional as FA

from .config_fsd50k_eval import (
    DEFAULT_MAX_CLIPS_PER_LABEL,
    DEFAULT_MIXED_DRONE_WINDOWS_PER_LABEL,
    FS,
    FSD50K_CANDIDATES_CSV,
    HOP_SAMPLES,
    LABELS,
    MIXED_SNR_LEVELS,
    RANDOM_SEED,
    RESULTS_DIR,
    WIN_SAMPLES,
    ensure_dirs,
)

HYBRID_SRC_DIR = Path(__file__).resolve().parents[2] / "src" / "hybrid_option2_option3"
if str(HYBRID_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(HYBRID_SRC_DIR))

from config_hybrid import DRONE_DIR, VIEW_WEIGHTS  # noqa: E402
from fuse_option2_option3 import TemporalSmoother, fuse_predictions  # noqa: E402
from load_models import load_hybrid_models  # noqa: E402
from predict_option2 import predict_option2  # noqa: E402
from predict_option3 import predict_option3  # noqa: E402
from src.harmonic_guard.guard_fusion import apply_harmonic_guard  # noqa: E402
from src.harmonic_guard.harmonic_analyzer import analyze_harmonics  # noqa: E402


SYSTEMS = ["option2", "option3", "hybrid_smoothed", "harmonic_guard_smoothed"]


def _option2_detect_from_probs(probs: np.ndarray) -> bool:
    weights = np.asarray(VIEW_WEIGHTS, dtype=np.float32)
    ws = float(weights @ probs)
    fm = float(probs[1:].max())
    vc = int((probs > 0.60).sum())
    return (fm > 0.75) or (ws > 0.60) or (vc >= 2)


def _read_audio(path: Path) -> np.ndarray:
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != FS:
        t = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)
        audio = FA.resample(t, sr, FS).squeeze(0).numpy()
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    return (audio / peak).astype(np.float32) if peak > 1e-5 else audio.astype(np.float32)


def _window_audio(audio: np.ndarray):
    return [audio[s:s + WIN_SAMPLES].copy() for s in range(0, len(audio) - WIN_SAMPLES + 1, HOP_SAMPLES)]


def _fit_window(audio: np.ndarray, rng: random.Random) -> np.ndarray:
    if len(audio) >= WIN_SAMPLES:
        start = rng.randint(0, len(audio) - WIN_SAMPLES)
        return audio[start:start + WIN_SAMPLES].copy()
    out = np.zeros(WIN_SAMPLES, dtype=np.float32)
    if len(audio):
        reps = int(np.ceil(WIN_SAMPLES / len(audio)))
        tiled = np.tile(audio, reps)
        out[:] = tiled[:WIN_SAMPLES]
    return out


def _mix_at_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float):
    clean = clean.astype(np.float64)
    noise = noise.astype(np.float64)
    pc = np.mean(clean * clean) + 1e-12
    pn = np.mean(noise * noise) + 1e-12
    mixed = clean + np.sqrt(pc / (pn * 10 ** (snr_db / 10.0))) * noise
    peak = float(np.max(np.abs(mixed))) if mixed.size else 0.0
    return (mixed / peak).astype(np.float32) if peak > 1e-7 else mixed.astype(np.float32)


def _write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_fsd50k_candidates(max_clips_per_label: int, seed: int):
    df = pd.read_csv(FSD50K_CANDIDATES_CSV)
    rng = random.Random(seed)
    by_label = {}
    for label in LABELS:
        rows = df[df["matched_labels"].fillna("").str.split(",").apply(lambda xs: label in xs)].copy()
        records = rows.to_dict("records")
        rng.shuffle(records)
        selected = []
        for row in records:
            path = Path(row["path"])
            if path.exists():
                selected.append({
                    "benchmark_label": label,
                    "split": row["split"],
                    "fname": str(row["fname"]),
                    "path": str(path),
                    "matched_labels": row["matched_labels"],
                    "all_labels": row["all_labels"],
                })
            if len(selected) >= max_clips_per_label:
                break
        by_label[label] = selected
    return by_label


def load_drone_windows(max_files: int, seed: int):
    rng = random.Random(seed)
    files = [p for p in sorted(DRONE_DIR.glob("*.wav")) if p.is_file()]
    rng.shuffle(files)
    windows = []
    for path in files[:max_files]:
        try:
            windows.extend(_window_audio(_read_audio(path)))
        except Exception:
            pass
    rng.shuffle(windows)
    return windows


def _predict_all(models_bundle, audio, hybrid_smoother, guard_smoother, rule, enable_hybrid_veto):
    o2 = predict_option2(models_bundle["option2"], audio, models_bundle["device"])
    o3 = predict_option3(models_bundle["option3"], audio, models_bundle["device"], method="weighted_average")
    hybrid = fuse_predictions(o2.score, o3, rule, enable_hybrid_veto)
    feats = analyze_harmonics(audio)
    guard = apply_harmonic_guard(hybrid, o3, feats)
    hybrid_det = hybrid_smoother.update(hybrid.detected)
    guard_det = guard_smoother.update(guard.detected)
    base_score = min(1.0, 0.5 * float(o2.score) + 0.5 * float(o3.score))
    return {
        "dets": {
            "option2": _option2_detect_from_probs(o2.per_view_probs),
            "option3": bool(o3.detected_alone),
            "hybrid_smoothed": bool(hybrid_det),
            "harmonic_guard_smoothed": bool(guard_det),
        },
        "scores": {
            "option2": float(o2.score),
            "option3": float(o3.score),
            "hybrid_smoothed": base_score,
            "harmonic_guard_smoothed": base_score * (1.0 - 0.35 * float(guard.downgraded)),
        },
        "o2": o2,
        "o3": o3,
        "hybrid": hybrid,
        "guard": guard,
        "features": feats,
    }


def evaluate_mixed(models_bundle, drone_windows, candidates_by_label, windows_per_label, snr_levels, seed, rule, smoothing, enable_hybrid_veto):
    rng = random.Random(seed)
    totals = defaultdict(lambda: {s: {"windows": 0, "detections": 0, "scores": []} for s in SYSTEMS})
    debug_rows = []
    drone_pos = 0
    t0 = time.perf_counter()

    scenarios = [("drone alone", None, None)]
    for label in LABELS:
        for snr in snr_levels:
            scenarios.append((f"drone+{label}@{snr}dB", label, snr))

    for scenario_name, label, snr in scenarios:
        hybrid_smoother = TemporalSmoother(smoothing)
        guard_smoother = TemporalSmoother(smoothing)
        clips = candidates_by_label.get(label, []) if label else []
        clip_noises = {}

        for i in range(windows_per_label):
            if not drone_windows:
                raise RuntimeError("No drone windows available.")
            clean = drone_windows[drone_pos % len(drone_windows)].copy()
            drone_pos += 1
            clip = {}
            if label:
                clip = clips[i % len(clips)]
                path = clip["path"]
                if path not in clip_noises:
                    clip_noises[path] = _read_audio(Path(path))
                noise = _fit_window(clip_noises[path], rng)
                audio = _mix_at_snr(clean, noise, float(snr))
            else:
                audio = clean

            pred = _predict_all(models_bundle, audio, hybrid_smoother, guard_smoother, rule, enable_hybrid_veto)
            for system in SYSTEMS:
                totals[scenario_name][system]["windows"] += 1
                totals[scenario_name][system]["detections"] += int(pred["dets"][system])
                totals[scenario_name][system]["scores"].append(pred["scores"][system])
                totals["ALL"][system]["windows"] += 1
                totals["ALL"][system]["detections"] += int(pred["dets"][system])
                totals["ALL"][system]["scores"].append(pred["scores"][system])

            debug_rows.append({
                "scenario": scenario_name,
                "label": label or "",
                "snr_db": "" if snr is None else snr,
                "window_index": i,
                "fsd50k_path": clip.get("path", ""),
                "fsd50k_fname": clip.get("fname", ""),
                "fsd50k_all_labels": clip.get("all_labels", ""),
                "option2_detected": int(pred["dets"]["option2"]),
                "option3_detected": int(pred["dets"]["option3"]),
                "hybrid_smoothed_detected": int(pred["dets"]["hybrid_smoothed"]),
                "harmonic_guard_smoothed_detected": int(pred["dets"]["harmonic_guard_smoothed"]),
                "option2_score": pred["scores"]["option2"],
                "option3_score": pred["scores"]["option3"],
                "hybrid_score": pred["scores"]["hybrid_smoothed"],
                "guard_score": pred["scores"]["harmonic_guard_smoothed"],
                "guard_downgraded": int(pred["guard"].downgraded),
                "guard_reason": pred["guard"].reason,
                "vehicle_risk_score": float(pred["features"].vehicle_risk_score),
                "f0_hz": float(pred["features"].f0_hz),
                "harmonicity_score": float(pred["features"].harmonicity_score),
            })

    summary_rows = []
    for scenario_name in ["ALL"] + [s[0] for s in scenarios]:
        for system in SYSTEMS:
            t = totals[scenario_name][system]
            summary_rows.append({
                "scenario": scenario_name,
                "system": system,
                "windows": t["windows"],
                "detected_windows": t["detections"],
                "recall_percent": 100.0 * t["detections"] / max(t["windows"], 1),
                "mean_score": float(np.mean(t["scores"])) if t["scores"] else 0.0,
            })
    return {
        "summary_rows": summary_rows,
        "debug_rows": debug_rows,
        "elapsed_seconds": time.perf_counter() - t0,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark drone recall with real FSD50K interference.")
    parser.add_argument("--max-clips-per-label", type=int, default=DEFAULT_MAX_CLIPS_PER_LABEL)
    parser.add_argument("--windows-per-scenario", type=int, default=DEFAULT_MIXED_DRONE_WINDOWS_PER_LABEL)
    parser.add_argument("--max-drone-files", type=int, default=500)
    parser.add_argument("--snr", default=",".join(str(x) for x in MIXED_SNR_LEVELS))
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--rule", default="B", choices=["A", "B", "C", "D"])
    parser.add_argument("--smoothing", default="2of3", choices=["none", "2of3", "3of5", "persist_1_5s"])
    parser.add_argument("--no-hybrid-veto", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    ensure_dirs()
    device = torch.device("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    snr_levels = [float(x.strip()) for x in args.snr.split(",") if x.strip()]
    print("FSD50K mixed positive benchmark")
    print("Safety: no training, no model overwrite.")
    print(f"Device: {device}")
    print(f"Windows per scenario: {args.windows_per_scenario}")
    print(f"SNRs: {snr_levels}")

    candidates = load_fsd50k_candidates(args.max_clips_per_label, args.seed)
    drone_windows = load_drone_windows(args.max_drone_files, args.seed)
    print(f"Drone windows loaded: {len(drone_windows)}")
    models = load_hybrid_models(device=device)
    result = evaluate_mixed(
        models,
        drone_windows,
        candidates,
        windows_per_label=args.windows_per_scenario,
        snr_levels=snr_levels,
        seed=args.seed,
        rule=args.rule,
        smoothing=args.smoothing,
        enable_hybrid_veto=not args.no_hybrid_veto,
    )

    summary_path = RESULTS_DIR / "fsd50k_mixed_recall_summary.csv"
    debug_path = RESULTS_DIR / "fsd50k_mixed_recall_window_debug.csv"
    _write_csv(summary_path, result["summary_rows"])
    _write_csv(debug_path, result["debug_rows"])

    all_rows = [r for r in result["summary_rows"] if r["scenario"] == "ALL"]
    print()
    for row in all_rows:
        print(
            f"{row['system']:26s} "
            f"recall={row['recall_percent']:6.2f}% "
            f"detected={row['detected_windows']}/{row['windows']} "
            f"mean_score={row['mean_score']:.3f}"
        )
    print()
    print(f"Saved summary -> {summary_path}")
    print(f"Saved debug   -> {debug_path}")
    print(f"Elapsed       -> {result['elapsed_seconds']:.1f}s")


if __name__ == "__main__":
    main()
