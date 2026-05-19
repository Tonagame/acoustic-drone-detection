from __future__ import annotations

import argparse
import csv
import random
import time
from pathlib import Path

import numpy as np
import torch

from src.phase2v5_real_noise.audio_phase2v5 import AudioPreprocessor
from src.phase2v5_real_noise.config_phase2v5 import DRONE_DIR, FSD50K_CANDIDATES_CSV, FSD_LABELS, NODRONE_DIR
from src.phase2v5_real_noise.data_phase2v5 import AudioFileWindowPool, FSD50KWindowPool
from src.phase3_real_noise_specialists.config_phase3_specialists import (
    RESULTS_DIR,
    SNR_LEVELS,
    SPECIALIST_BUNDLE_PATH,
    TEMPORAL_SMOOTHING,
)
from src.phase3_real_noise_specialists.predict_phase3_hybrid import (
    TemporalSmoother,
    load_specialist_bundle,
    predict_specialists,
)


THRESHOLDS = [round(x, 2) for x in np.arange(0.05, 1.00, 0.05)]


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    keys: list[str] = []
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


def sample_condition(condition: str, pools, preproc: AudioPreprocessor, rng):
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


def condition_list():
    conditions = [("drone_alone", True)]
    conditions.extend((f"drone_plus_fsd_snr_{snr:+d}db", True) for snr in SNR_LEVELS)
    conditions.extend([("fsd_real_noise_alone", False), ("dads_no_drone_alone", False)])
    return conditions


def apply_smoothing(raw_detected: list[bool], mode: str = TEMPORAL_SMOOTHING) -> list[bool]:
    smoother = TemporalSmoother(mode)
    return [bool(smoother.update(x)) for x in raw_detected]


def summarize_system(rows: list[dict], system: str, threshold: float | str) -> dict:
    sub = [r for r in rows if r["system"] == system and r["threshold"] == threshold]
    pos = [r for r in sub if r["target_positive"] == 1]
    neg = [r for r in sub if r["target_positive"] == 0]
    recall = 100.0 * sum(int(r["smoothed_detected"]) for r in pos) / max(len(pos), 1)
    far = 100.0 * sum(int(r["smoothed_detected"]) for r in neg) / max(len(neg), 1)
    return {
        "system": system,
        "threshold": threshold,
        "positive_windows": len(pos),
        "negative_windows": len(neg),
        "positive_recall_percent": recall,
        "negative_false_alarm_percent": far,
        "score_index": recall - 2.0 * far,
        "mean_score_positive": float(np.mean([r["score"] for r in pos])) if pos else 0.0,
        "mean_score_negative": float(np.mean([r["score"] for r in neg])) if neg else 0.0,
    }


def parse_args():
    ap = argparse.ArgumentParser(description="Ablate raw-only specialist vs five-specialist ensemble")
    ap.add_argument("--specialists", type=Path, default=SPECIALIST_BUNDLE_PATH)
    ap.add_argument("--windows-per-condition", type=int, default=250)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=7373)
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

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_gpu else "cpu")
    preproc = AudioPreprocessor(16000)
    pools = build_pools(args, preproc)
    specialists = load_specialist_bundle(args.specialists, device)
    rng = random.Random(args.seed)

    sample_rows: list[dict] = []
    t0 = time.perf_counter()
    print("Raw-only vs five-specialist ablation")
    print(f"Specialists: {args.specialists}")
    print(f"Device     : {device}")
    print(f"Windows/condition: {args.windows_per_condition}")

    for condition, target_positive in condition_list():
        print(f"Sampling {condition}...")
        for wi in range(args.windows_per_condition):
            audio = sample_condition(condition, pools, preproc, rng)
            sp = predict_specialists(specialists, preproc, audio)
            sample_rows.append({
                "condition": condition,
                "window_index": wi,
                "target_positive": int(target_positive),
                "raw_score": float(sp.per_view_probs[0]),
                "five_weighted_score": float(sp.score),
                "five_rule_detected": int(sp.candidate),
                "five_filtered_max": float(sp.filtered_max),
                "five_vote_count": int(sp.vote_count),
                "p_raw": float(sp.per_view_probs[0]),
                "p_hpf150": float(sp.per_view_probs[1]),
                "p_hpf250": float(sp.per_view_probs[2]),
                "p_bpf200": float(sp.per_view_probs[3]),
                "p_bpf500": float(sp.per_view_probs[4]),
            })

    eval_rows: list[dict] = []
    for condition, target_positive in condition_list():
        cond_rows = [r for r in sample_rows if r["condition"] == condition]
        for threshold in THRESHOLDS:
            for system, score_key in [("raw_only", "raw_score"), ("five_weighted", "five_weighted_score")]:
                raw_det = [float(r[score_key]) >= threshold for r in cond_rows]
                smooth_det = apply_smoothing(raw_det)
                for r, det, sm in zip(cond_rows, raw_det, smooth_det):
                    eval_rows.append({
                        "system": system,
                        "threshold": threshold,
                        "condition": condition,
                        "window_index": r["window_index"],
                        "target_positive": int(target_positive),
                        "score": float(r[score_key]),
                        "detected": int(det),
                        "smoothed_detected": int(sm),
                    })

        rule_raw = [bool(r["five_rule_detected"]) for r in cond_rows]
        rule_smooth = apply_smoothing(rule_raw)
        for r, det, sm in zip(cond_rows, rule_raw, rule_smooth):
            eval_rows.append({
                "system": "five_rule",
                "threshold": "rule",
                "condition": condition,
                "window_index": r["window_index"],
                "target_positive": int(target_positive),
                "score": float(r["five_weighted_score"]),
                "detected": int(det),
                "smoothed_detected": int(sm),
            })

    summary_rows = [summarize_system(eval_rows, "raw_only", thr) for thr in THRESHOLDS]
    summary_rows += [summarize_system(eval_rows, "five_weighted", thr) for thr in THRESHOLDS]
    summary_rows += [summarize_system(eval_rows, "five_rule", "rule")]

    def best(system: str):
        options = [r for r in summary_rows if r["system"] == system]
        return max(options, key=lambda r: (r["score_index"], r["positive_recall_percent"], -r["negative_false_alarm_percent"]))

    best_raw = best("raw_only")
    best_five_weighted = best("five_weighted")
    five_rule = best("five_rule")

    condition_rows: list[dict] = []
    chosen = [
        ("raw_only_best", "raw_only", best_raw["threshold"]),
        ("five_weighted_best", "five_weighted", best_five_weighted["threshold"]),
        ("five_rule", "five_rule", "rule"),
    ]
    for label, system, threshold in chosen:
        for condition, target_positive in condition_list():
            sub = [r for r in eval_rows if r["system"] == system and r["threshold"] == threshold and r["condition"] == condition]
            condition_rows.append({
                "system": label,
                "condition": condition,
                "target": "positive" if target_positive else "negative",
                "threshold": threshold,
                "windows": len(sub),
                "smoothed_detection_rate_percent": 100.0 * sum(int(r["smoothed_detected"]) for r in sub) / max(len(sub), 1),
                "mean_score": float(np.mean([r["score"] for r in sub])) if sub else 0.0,
            })

    suffix = "quick" if args.quick else f"{args.windows_per_condition}w"
    sample_path = RESULTS_DIR / f"ablation_raw_vs_five_samples_{suffix}.csv"
    sweep_path = RESULTS_DIR / f"ablation_raw_vs_five_threshold_sweep_{suffix}.csv"
    condition_path = RESULTS_DIR / f"ablation_raw_vs_five_conditions_{suffix}.csv"
    write_csv(sample_path, sample_rows)
    write_csv(sweep_path, summary_rows)
    write_csv(condition_path, condition_rows)

    print("\nBest overall operating points")
    for row in [best_raw, best_five_weighted, five_rule]:
        print(
            f"{row['system']:<14} thr={str(row['threshold']):>4} "
            f"recall={row['positive_recall_percent']:6.2f}% "
            f"FAR={row['negative_false_alarm_percent']:6.2f}% "
            f"score_index={row['score_index']:6.2f}"
        )
    print(f"\nSaved samples    -> {sample_path}")
    print(f"Saved sweep      -> {sweep_path}")
    print(f"Saved conditions -> {condition_path}")
    print(f"Elapsed: {time.perf_counter() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
