from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.phase2v5_real_noise.audio_phase2v5 import AudioPreprocessor
from src.phase2v5_real_noise.config_phase2v5 import DRONE_DIR, FSD50K_CANDIDATES_CSV, FSD_LABELS, NODRONE_DIR
from src.phase2v5_real_noise.data_phase2v5 import AudioFileWindowPool, FSD50KWindowPool

from src.phase3_real_noise_specialists.config_phase3_specialists import (
    PHASE2_BACKBONE_PATH,
    PHASE2_GUARD_PATH,
    RESULTS_DIR,
    SNR_LEVELS,
    SPECIALIST_BUNDLE_PATH,
    TEMPORAL_SMOOTHING,
    ensure_dirs,
)
from src.phase3_real_noise_specialists.predict_phase3_hybrid import (
    TemporalSmoother,
    fuse_phase3,
    load_phase2_guard,
    load_specialist_bundle,
    predict_phase2_guard,
    predict_specialists,
)


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def build_pools(args, preproc):
    drone_pool = AudioFileWindowPool(DRONE_DIR, preproc, args.max_drone_files, args.seed)
    nodrone_pool = AudioFileWindowPool(NODRONE_DIR, preproc, args.max_nodrone_files, args.seed + 1) if NODRONE_DIR.exists() else None
    fsd_pool = FSD50KWindowPool(FSD50K_CANDIDATES_CSV, preproc, FSD_LABELS, args.max_fsd_clips_per_label, args.seed + 2)
    return drone_pool, fsd_pool, nodrone_pool


def sample_condition(condition, pools, preproc, rng):
    drone_pool, fsd_pool, nodrone_pool = pools
    if condition == "drone_alone":
        return drone_pool.sample_window(rng)
    if condition.startswith("drone_plus_fsd_snr_"):
        snr = int(condition.split("_snr_", 1)[1].replace("db", ""))
        drone = drone_pool.sample_window(rng)
        noise = fsd_pool.sample_window(rng)
        return preproc.mix_at_snr(drone, noise, snr)
    if condition == "fsd_real_noise_alone":
        return fsd_pool.sample_window(rng)
    if condition == "dads_no_drone_alone":
        return nodrone_pool.sample_window(rng) if nodrone_pool else fsd_pool.sample_window(rng)
    raise ValueError(condition)


def summarize(rows, system, condition, target_positive):
    sub = [r for r in rows if r["system"] == system and r["condition"] == condition]
    det = sum(int(r["detected"]) for r in sub)
    smooth = sum(int(r["smoothed_detected"]) for r in sub)
    return {
        "system": system,
        "condition": condition,
        "target": "positive" if target_positive else "negative",
        "windows": len(sub),
        "detected": det,
        "smoothed_detected": smooth,
        "detection_rate_percent": 100.0 * det / max(len(sub), 1),
        "smoothed_detection_rate_percent": 100.0 * smooth / max(len(sub), 1),
        "mean_score": float(np.mean([r["score"] for r in sub])) if sub else 0.0,
    }


def parse_args():
    ap = argparse.ArgumentParser(description="Benchmark Phase 3 real-noise specialist hybrid")
    ap.add_argument("--specialists", type=Path, default=SPECIALIST_BUNDLE_PATH)
    ap.add_argument("--phase2", type=Path, default=PHASE2_GUARD_PATH)
    ap.add_argument("--backbone", type=Path, default=PHASE2_BACKBONE_PATH)
    ap.add_argument("--windows-per-condition", type=int, default=250)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=4141)
    ap.add_argument("--no-gpu", action="store_true")
    ap.add_argument("--max-drone-files", type=int, default=12000)
    ap.add_argument("--max-nodrone-files", type=int, default=5000)
    ap.add_argument("--max-fsd-clips-per-label", type=int, default=500)
    return ap.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.windows_per_condition = min(args.windows_per_condition, 80)
        args.max_drone_files = min(args.max_drone_files, 1200)
        args.max_nodrone_files = min(args.max_nodrone_files, 800)
        args.max_fsd_clips_per_label = min(args.max_fsd_clips_per_label, 35)
    ensure_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_gpu else "cpu")
    preproc = AudioPreprocessor(16000)
    pools = build_pools(args, preproc)
    specialists = load_specialist_bundle(args.specialists, device)
    guard = load_phase2_guard(args.phase2, args.backbone, device)
    rng = random.Random(args.seed)

    conditions = [("drone_alone", True)]
    conditions.extend((f"drone_plus_fsd_snr_{snr:+d}db", True) for snr in SNR_LEVELS)
    conditions.extend([("fsd_real_noise_alone", False), ("dads_no_drone_alone", False)])
    systems = ["specialists_only", "phase2_guard", "phase3_hybrid"]
    rows = []
    t0 = time.perf_counter()
    print("Phase 3 real-noise specialist benchmark")
    print(f"Specialists: {args.specialists}")
    print(f"Guard      : {args.phase2}")
    print(f"Device     : {device}")

    for condition, target_positive in conditions:
        smoothers = {system: TemporalSmoother(TEMPORAL_SMOOTHING) for system in systems}
        for wi in range(args.windows_per_condition):
            audio = sample_condition(condition, pools, preproc, rng)
            sp = predict_specialists(specialists, preproc, audio)
            gd = predict_phase2_guard(guard, preproc, audio)
            hy = fuse_phase3(sp, gd)

            system_values = {
                "specialists_only": (sp.candidate, sp.score, "specialist_rule"),
                "phase2_guard": (gd.score >= 0.85, gd.score, "phase2_threshold_0.85"),
                "phase3_hybrid": (hy.detected, hy.score, hy.reason),
            }
            for system, (detected, score, reason) in system_values.items():
                smoothed = smoothers[system].update(bool(detected))
                rows.append({
                    "system": system,
                    "condition": condition,
                    "window_index": wi,
                    "target_positive": int(target_positive),
                    "detected": int(bool(detected)),
                    "smoothed_detected": int(bool(smoothed)),
                    "score": float(score),
                    "reason": reason,
                    "specialist_score": float(sp.score),
                    "specialist_filtered_max": float(sp.filtered_max),
                    "specialist_vote_count": int(sp.vote_count),
                    "p_raw": float(sp.per_view_probs[0]),
                    "p_hpf150": float(sp.per_view_probs[1]),
                    "p_hpf250": float(sp.per_view_probs[2]),
                    "p_bpf200": float(sp.per_view_probs[3]),
                    "p_bpf500": float(sp.per_view_probs[4]),
                    "phase2_score": float(gd.score),
                    "vehicle_risk_score": float(gd.vehicle_risk_score),
                    "f0_norm": float(gd.f0_norm),
                    "harmonicity_score": float(gd.harmonicity_score),
                })

    summary = []
    for condition, target_positive in conditions:
        for system in systems:
            summary.append(summarize(rows, system, condition, target_positive))

    for system in systems:
        pos = [r for r in rows if r["system"] == system and r["target_positive"] == 1]
        neg = [r for r in rows if r["system"] == system and r["target_positive"] == 0]
        summary.append({
            "system": system,
            "condition": "overall",
            "target": "mixed",
            "windows": len(pos) + len(neg),
            "detected": sum(int(r["detected"]) for r in pos + neg),
            "smoothed_detected": sum(int(r["smoothed_detected"]) for r in pos + neg),
            "detection_rate_percent": 100.0 * sum(int(r["detected"]) for r in pos + neg) / max(len(pos) + len(neg), 1),
            "smoothed_detection_rate_percent": 100.0 * sum(int(r["smoothed_detected"]) for r in pos + neg) / max(len(pos) + len(neg), 1),
            "positive_recall_percent": 100.0 * sum(int(r["smoothed_detected"]) for r in pos) / max(len(pos), 1),
            "negative_false_alarm_percent": 100.0 * sum(int(r["smoothed_detected"]) for r in neg) / max(len(neg), 1),
            "mean_score": float(np.mean([r["score"] for r in pos + neg])) if rows else 0.0,
        })

    tag = args.specialists.stem.replace("drone_cnn_", "")
    if args.quick:
        tag += "_quick"
    summary_path = RESULTS_DIR / f"benchmark_{tag}_summary.csv"
    debug_path = RESULTS_DIR / f"benchmark_{tag}_debug.csv"
    write_csv(summary_path, summary)
    write_csv(debug_path, rows)

    print("\nSummary")
    for row in summary:
        if row["condition"] == "overall":
            print(
                f"{row['system']:<18} recall={float(row['positive_recall_percent']):6.2f}% "
                f"FAR={float(row['negative_false_alarm_percent']):6.2f}%"
            )
    print(f"Saved summary -> {summary_path}")
    print(f"Saved debug -> {debug_path}")
    print(f"Elapsed: {time.perf_counter() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

