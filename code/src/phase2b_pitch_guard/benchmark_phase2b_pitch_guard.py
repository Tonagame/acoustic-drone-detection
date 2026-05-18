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

from src.phase2b_pitch_guard.config_phase2b import PHASE2_BACKBONE_PATH, PHASE2_GUARD_PATH, PHASE3_SPECIALIST_PATH, RESULTS_DIR, SAVE_PATH, SNR_LEVELS, ensure_dirs
from src.phase2b_pitch_guard.features_phase2b import PoolArgs, base_feature_rows, build_pools
from src.phase2b_pitch_guard.model_phase2b import PitchGuardMLP
from src.phase2b_pitch_guard.pitch_features import crepe_pitch_features_batch
from src.phase2v5_real_noise.audio_phase2v5 import AudioPreprocessor
from src.phase3_real_noise_specialists.predict_phase3_hybrid import TemporalSmoother, fuse_phase3, load_phase2_guard, load_specialist_bundle, predict_phase2_guard, predict_specialists


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def sample_condition(condition, pools, preproc, rng):
    drone_pool, fsd_pool, nodrone_pool = pools
    if condition == "drone_alone":
        return drone_pool.sample_window(rng)
    if condition.startswith("drone_plus_fsd_snr_"):
        snr = int(condition.split("_snr_", 1)[1].replace("db", ""))
        return preproc.mix_at_snr(drone_pool.sample_window(rng), fsd_pool.sample_window(rng), snr)
    if condition == "fsd_real_noise_alone":
        return fsd_pool.sample_window(rng)
    if condition == "dads_no_drone_alone":
        return nodrone_pool.sample_window(rng) if nodrone_pool else fsd_pool.sample_window(rng)
    raise ValueError(condition)


def summarize(rows, system, condition, target_positive):
    sub = [r for r in rows if r["system"] == system and r["condition"] == condition]
    det = sum(int(r["smoothed_detected"]) for r in sub)
    return {
        "system": system,
        "condition": condition,
        "target": "positive" if target_positive else "negative",
        "windows": len(sub),
        "smoothed_detected": det,
        "smoothed_detection_rate_percent": 100.0 * det / max(len(sub), 1),
        "mean_score": float(np.mean([r["score"] for r in sub])) if sub else 0.0,
    }


def parse_args():
    ap = argparse.ArgumentParser(description="Benchmark Phase 2b pitch guard against Phase 3")
    ap.add_argument("--model", type=Path, default=SAVE_PATH)
    ap.add_argument("--threshold", type=float, default=0.55)
    ap.add_argument("--phase3-threshold", type=float, default=0.55)
    ap.add_argument("--windows-per-condition", type=int, default=250)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=6262)
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
    pools = build_pools(PoolArgs(args.max_drone_files, args.max_nodrone_files, args.max_fsd_clips_per_label, args.seed), preproc)
    specialists = load_specialist_bundle(PHASE3_SPECIALIST_PATH, device)
    guard = load_phase2_guard(PHASE2_GUARD_PATH, PHASE2_BACKBONE_PATH, device)
    ckpt = torch.load(str(args.model), map_location=device, weights_only=False)
    model = PitchGuardMLP(in_dim=int(ckpt["metadata"]["input_dim"])).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    rng = random.Random(args.seed)
    conditions = [("drone_alone", True)]
    conditions.extend((f"drone_plus_fsd_snr_{snr:+d}db", True) for snr in SNR_LEVELS)
    conditions.extend([("fsd_real_noise_alone", False), ("dads_no_drone_alone", False)])
    rows = []
    t0 = time.perf_counter()
    print("Phase 2b pitch guard benchmark")
    print(f"Model: {args.model}")
    print(f"Device: {device}")
    for condition, target_positive in conditions:
        audios = np.stack([sample_condition(condition, pools, preproc, rng) for _ in range(args.windows_per_condition)]).astype(np.float32)
        base = base_feature_rows(audios, specialists, guard, preproc)
        pitch = crepe_pitch_features_batch(audios, device=device, batch_size=64)
        X = np.concatenate([base, pitch], axis=1).astype(np.float32)
        with torch.no_grad():
            pitch_scores = torch.softmax(model(torch.from_numpy(X).to(device)), dim=1)[:, 0].detach().cpu().numpy()
        smoothers = {"phase2b_pitch_guard": TemporalSmoother("2_of_3"), "phase3_hybrid_0.55": TemporalSmoother("2_of_3")}
        for i, audio in enumerate(audios):
            sp = predict_specialists(specialists, preproc, audio)
            gd = predict_phase2_guard(guard, preproc, audio)
            hy = fuse_phase3(sp, gd)
            phase3_score = float(hy.score)
            systems = {
                "phase2b_pitch_guard": (float(pitch_scores[i]), float(pitch_scores[i]) > args.threshold),
                "phase3_hybrid_0.55": (phase3_score, phase3_score > args.phase3_threshold),
            }
            for system, (score, detected) in systems.items():
                smoothed = smoothers[system].update(detected)
                rows.append({
                    "system": system,
                    "condition": condition,
                    "window_index": i,
                    "target_positive": int(target_positive),
                    "score": score,
                    "detected": int(detected),
                    "smoothed_detected": int(smoothed),
                    "phase3_score": phase3_score,
                    "phase2b_score": float(pitch_scores[i]),
                    "vehicle_risk_score": float(gd.vehicle_risk_score),
                    "specialist_score": float(sp.score),
                })
    summary = []
    for condition, target_positive in conditions:
        for system in ["phase2b_pitch_guard", "phase3_hybrid_0.55"]:
            summary.append(summarize(rows, system, condition, target_positive))
    for system in ["phase2b_pitch_guard", "phase3_hybrid_0.55"]:
        pos = [r for r in rows if r["system"] == system and r["target_positive"] == 1]
        neg = [r for r in rows if r["system"] == system and r["target_positive"] == 0]
        summary.append({
            "system": system,
            "condition": "overall",
            "target": "mixed",
            "windows": len(pos) + len(neg),
            "positive_recall_percent": 100.0 * sum(int(r["smoothed_detected"]) for r in pos) / max(len(pos), 1),
            "negative_false_alarm_percent": 100.0 * sum(int(r["smoothed_detected"]) for r in neg) / max(len(neg), 1),
            "mean_score": float(np.mean([r["score"] for r in pos + neg])) if rows else 0.0,
        })
    tag = args.model.stem.replace("drone_cnn_", "")
    if args.quick:
        tag += "_quick"
    summary_path = RESULTS_DIR / f"benchmark_{tag}_summary.csv"
    debug_path = RESULTS_DIR / f"benchmark_{tag}_debug.csv"
    write_csv(summary_path, summary)
    write_csv(debug_path, rows)
    print("\nSummary")
    for r in summary:
        if r["condition"] == "overall":
            print(f"{r['system']:<22} recall={r['positive_recall_percent']:6.2f}% FAR={r['negative_false_alarm_percent']:6.2f}%")
    print(f"Saved summary -> {summary_path}")
    print(f"Saved debug -> {debug_path}")
    print(f"Elapsed: {time.perf_counter() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

