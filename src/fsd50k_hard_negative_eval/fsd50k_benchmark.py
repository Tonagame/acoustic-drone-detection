"""Evaluate real FSD50K vehicle/engine clips as hard negatives."""

import argparse
import csv
import random
import shutil
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
    DEFAULT_MAX_WINDOWS_PER_CLIP,
    FS,
    FSD50K_CANDIDATES_CSV,
    HOP_SAMPLES,
    LABELS,
    RANDOM_SEED,
    RESULTS_DIR,
    WIN_SAMPLES,
    WORST_DIR,
    ensure_dirs,
)

HYBRID_SRC_DIR = Path(__file__).resolve().parents[2] / "src" / "hybrid_option2_option3"
if str(HYBRID_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(HYBRID_SRC_DIR))

from config_hybrid import VIEW_WEIGHTS  # noqa: E402
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


def _windows(audio: np.ndarray, max_windows: int):
    out = []
    for start in range(0, len(audio) - WIN_SAMPLES + 1, HOP_SAMPLES):
        out.append(audio[start:start + WIN_SAMPLES].copy())
        if len(out) >= max_windows:
            break
    if not out and len(audio) > 0:
        padded = np.zeros(WIN_SAMPLES, dtype=np.float32)
        n = min(len(audio), WIN_SAMPLES)
        padded[:n] = audio[:n]
        out.append(padded)
    return out


def _write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_candidates(max_clips_per_label: int, seed: int):
    df = pd.read_csv(FSD50K_CANDIDATES_CSV)
    rng = random.Random(seed)
    selected = []
    seen = set()
    for label in LABELS:
        rows = df[df["matched_labels"].fillna("").str.split(",").apply(lambda xs: label in xs)].copy()
        records = rows.to_dict("records")
        rng.shuffle(records)
        count = 0
        for row in records:
            key = (row["split"], str(row["fname"]), label)
            path = Path(row["path"])
            if not path.exists():
                continue
            selected.append({
                "benchmark_label": label,
                "split": row["split"],
                "fname": str(row["fname"]),
                "path": str(path),
                "matched_labels": row["matched_labels"],
                "all_labels": row["all_labels"],
            })
            seen.add(key)
            count += 1
            if count >= max_clips_per_label:
                break
    return selected


def evaluate_fsd50k(models_bundle, clips, max_windows_per_clip: int, rule: str, smoothing: str, enable_hybrid_veto: bool):
    totals = {s: {"windows": 0, "detections": 0, "scores": []} for s in SYSTEMS}
    label_totals = defaultdict(lambda: {s: {"windows": 0, "detections": 0, "scores": []} for s in SYSTEMS})
    debug_rows = []
    worst_rows = []
    t0 = time.perf_counter()

    for ci, clip in enumerate(clips, 1):
        try:
            audio = _read_audio(Path(clip["path"]))
        except Exception as exc:
            debug_rows.append({**clip, "window_index": "", "error": str(exc)})
            continue

        hybrid_smoother = TemporalSmoother(smoothing)
        guard_smoother = TemporalSmoother(smoothing)
        clip_windows = _windows(audio, max_windows_per_clip)
        clip_any = {s: False for s in SYSTEMS}
        clip_max_score = {s: 0.0 for s in SYSTEMS}
        clip_max_vehicle_risk = 0.0

        for wi, window in enumerate(clip_windows):
            o2 = predict_option2(models_bundle["option2"], window, models_bundle["device"])
            o3 = predict_option3(models_bundle["option3"], window, models_bundle["device"], method="weighted_average")
            hybrid = fuse_predictions(o2.score, o3, rule, enable_hybrid_veto)
            feats = analyze_harmonics(window)
            guard = apply_harmonic_guard(hybrid, o3, feats)

            hybrid_det = hybrid_smoother.update(hybrid.detected)
            guard_det = guard_smoother.update(guard.detected)
            base_score = min(1.0, 0.5 * float(o2.score) + 0.5 * float(o3.score))
            dets = {
                "option2": _option2_detect_from_probs(o2.per_view_probs),
                "option3": bool(o3.detected_alone),
                "hybrid_smoothed": bool(hybrid_det),
                "harmonic_guard_smoothed": bool(guard_det),
            }
            scores = {
                "option2": float(o2.score),
                "option3": float(o3.score),
                "hybrid_smoothed": base_score,
                "harmonic_guard_smoothed": base_score * (1.0 - 0.35 * float(guard.downgraded)),
            }

            for system in SYSTEMS:
                totals[system]["windows"] += 1
                totals[system]["detections"] += int(dets[system])
                totals[system]["scores"].append(scores[system])
                label_totals[clip["benchmark_label"]][system]["windows"] += 1
                label_totals[clip["benchmark_label"]][system]["detections"] += int(dets[system])
                label_totals[clip["benchmark_label"]][system]["scores"].append(scores[system])
                clip_any[system] = clip_any[system] or dets[system]
                clip_max_score[system] = max(clip_max_score[system], scores[system])

            clip_max_vehicle_risk = max(clip_max_vehicle_risk, float(feats.vehicle_risk_score))
            debug_rows.append({
                **clip,
                "clip_index": ci,
                "window_index": wi,
                "option2_detected": int(dets["option2"]),
                "option3_detected": int(dets["option3"]),
                "hybrid_smoothed_detected": int(dets["hybrid_smoothed"]),
                "harmonic_guard_smoothed_detected": int(dets["harmonic_guard_smoothed"]),
                "option2_score": float(o2.score),
                "option3_score": float(o3.score),
                "option3_filtered_max": float(o3.filtered_max),
                "option3_vote_count": int(o3.vote_count),
                "hybrid_score": base_score,
                "hybrid_reason": hybrid.reason,
                "hybrid_vetoed": int(hybrid.vetoed),
                "guard_downgraded": int(guard.downgraded),
                "guard_reason": guard.reason,
                "vehicle_risk_score": float(feats.vehicle_risk_score),
                "f0_hz": float(feats.f0_hz),
                "harmonicity_score": float(feats.harmonicity_score),
                "upper_harmonic_explained_ratio": float(feats.upper_harmonic_explained_ratio),
                "error": "",
            })

        for system in SYSTEMS:
            if clip_any[system]:
                worst_rows.append({
                    **clip,
                    "system": system,
                    "max_score": clip_max_score[system],
                    "max_vehicle_risk_score": clip_max_vehicle_risk,
                    "clip_windows": len(clip_windows),
                })

    summary_rows = []
    for system in SYSTEMS:
        t = totals[system]
        summary_rows.append({
            "system": system,
            "windows": t["windows"],
            "detected_windows": t["detections"],
            "false_alarm_rate_percent": 100.0 * t["detections"] / max(t["windows"], 1),
            "mean_score": float(np.mean(t["scores"])) if t["scores"] else 0.0,
        })

    label_rows = []
    for label in LABELS:
        for system in SYSTEMS:
            t = label_totals[label][system]
            label_rows.append({
                "label": label,
                "system": system,
                "windows": t["windows"],
                "detected_windows": t["detections"],
                "false_alarm_rate_percent": 100.0 * t["detections"] / max(t["windows"], 1),
                "mean_score": float(np.mean(t["scores"])) if t["scores"] else 0.0,
            })

    worst_rows.sort(key=lambda r: (r["system"], -float(r["max_score"])))
    return {
        "summary_rows": summary_rows,
        "label_rows": label_rows,
        "debug_rows": debug_rows,
        "worst_rows": worst_rows,
        "elapsed_seconds": time.perf_counter() - t0,
    }


def copy_worst_audio(worst_rows: list[dict], per_system: int):
    WORST_DIR.mkdir(parents=True, exist_ok=True)
    copied = []
    by_system = defaultdict(list)
    for row in worst_rows:
        by_system[row["system"]].append(row)
    for system, rows in by_system.items():
        dst_dir = WORST_DIR / system
        dst_dir.mkdir(parents=True, exist_ok=True)
        for row in rows[:per_system]:
            src = Path(row["path"])
            dst = dst_dir / f"{row['benchmark_label']}__{row['split']}__{row['fname']}.wav"
            if src.exists() and not dst.exists():
                shutil.copy2(src, dst)
            copied.append({**row, "copied_to": str(dst)})
    return copied


def main():
    parser = argparse.ArgumentParser(description="Benchmark detector false alarms on real FSD50K negatives.")
    parser.add_argument("--max-clips-per-label", type=int, default=DEFAULT_MAX_CLIPS_PER_LABEL)
    parser.add_argument("--max-windows-per-clip", type=int, default=DEFAULT_MAX_WINDOWS_PER_CLIP)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--rule", default="B", choices=["A", "B", "C", "D"])
    parser.add_argument("--smoothing", default="2of3", choices=["none", "2of3", "3of5", "persist_1_5s"])
    parser.add_argument("--no-hybrid-veto", action="store_true")
    parser.add_argument("--copy-worst", type=int, default=20)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    ensure_dirs()
    device = torch.device("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    print("FSD50K hard-negative benchmark")
    print("Safety: no training, no model overwrite, negative benchmark only.")
    print(f"Device: {device}")
    print(f"Max clips per label: {args.max_clips_per_label}")
    print(f"Max windows per clip: {args.max_windows_per_clip}")

    clips = load_candidates(args.max_clips_per_label, args.seed)
    print(f"Selected clips: {len(clips)}")
    models = load_hybrid_models(device=device)
    result = evaluate_fsd50k(
        models,
        clips,
        max_windows_per_clip=args.max_windows_per_clip,
        rule=args.rule,
        smoothing=args.smoothing,
        enable_hybrid_veto=not args.no_hybrid_veto,
    )
    copied_rows = copy_worst_audio(result["worst_rows"], args.copy_worst) if args.copy_worst > 0 else []

    summary_path = RESULTS_DIR / "fsd50k_benchmark_summary.csv"
    labels_path = RESULTS_DIR / "fsd50k_benchmark_by_label.csv"
    debug_path = RESULTS_DIR / "fsd50k_benchmark_window_debug.csv"
    worst_path = RESULTS_DIR / "fsd50k_worst_false_alarms.csv"
    copied_path = RESULTS_DIR / "fsd50k_copied_worst_false_alarms.csv"
    _write_csv(summary_path, result["summary_rows"])
    _write_csv(labels_path, result["label_rows"])
    _write_csv(debug_path, result["debug_rows"])
    _write_csv(worst_path, result["worst_rows"])
    _write_csv(copied_path, copied_rows)

    print()
    for row in result["summary_rows"]:
        print(
            f"{row['system']:26s} "
            f"false_alarm={row['false_alarm_rate_percent']:6.2f}% "
            f"detected={row['detected_windows']}/{row['windows']} "
            f"mean_score={row['mean_score']:.3f}"
        )
    print()
    print(f"Saved summary -> {summary_path}")
    print(f"Saved labels  -> {labels_path}")
    print(f"Saved debug   -> {debug_path}")
    print(f"Saved worst   -> {worst_path}")
    print(f"Elapsed       -> {result['elapsed_seconds']:.1f}s")


if __name__ == "__main__":
    main()
