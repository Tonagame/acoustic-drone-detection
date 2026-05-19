from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.phase2v5_real_noise.audio_phase2v5 import AudioPreprocessor
from src.phase2v5_real_noise.config_phase2v5 import DRONE_DIR, FSD50K_CANDIDATES_CSV, FSD_LABELS, NODRONE_DIR
from src.phase2v5_real_noise.data_phase2v5 import AudioFileWindowPool, FSD50KWindowPool
from src.phase3_real_noise_specialists.predict_phase3_hybrid import (
    TemporalSmoother,
    fuse_phase3,
    load_phase2_guard,
    load_specialist_bundle,
    predict_phase2_guard,
    predict_specialists,
)

from src.phase3_mid_fusion.config_mid_fusion import (
    MID_FUSION_INPUT_DIM,
    PHASE2_BACKBONE_PATH,
    PHASE2_GUARD_PATH,
    RESULTS_DIR,
    SAVE_PATH,
    SNR_LEVELS,
    SPECIALIST_BUNDLE_PATH,
    TEMPORAL_SMOOTHING,
    ensure_dirs,
)
from src.phase3_mid_fusion.model_mid_fusion import FrozenSpecialistMidFusion, MidFusionHead


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


def smooth_sequence(flags: list[int], mode: str = TEMPORAL_SMOOTHING) -> list[int]:
    if mode == "none":
        return [int(bool(x)) for x in flags]
    n, need = (3, 2) if mode == "2_of_3" else (5, 3)
    hist = deque()
    out = []
    for flag in flags:
        hist.append(bool(flag))
        while len(hist) > n:
            hist.popleft()
        out.append(int(len(hist) == n and sum(hist) >= need))
    return out


def summarize(rows, system, condition, target_positive, detected_key="smoothed_detected"):
    sub = [r for r in rows if r["system"] == system and r["condition"] == condition]
    det = sum(int(r[detected_key]) for r in sub)
    return {
        "system": system,
        "condition": condition,
        "target": "positive" if target_positive else "negative",
        "windows": len(sub),
        "smoothed_detection_rate_percent": 100.0 * det / max(len(sub), 1),
        "mean_score": float(np.mean([r["score"] for r in sub])) if sub else 0.0,
    }


@torch.no_grad()
def predict_mid(model, preproc: AudioPreprocessor, audio: np.ndarray, device):
    views = preproc.create_audio_views(audio)
    tensors = [preproc.audio_to_logmel(view).unsqueeze(0).float() for view in views]
    x = torch.stack(tensors, dim=0).unsqueeze(0).to(device)
    logits = model(x)
    return float(torch.softmax(logits, dim=1)[0, 0].item())


def load_mid_fusion(mid_path: Path, specialists_bundle, device):
    ckpt = torch.load(str(mid_path), map_location=device, weights_only=False)
    head = MidFusionHead(int(ckpt["metadata"].get("input_dim", MID_FUSION_INPUT_DIM))).to(device)
    head.load_state_dict(ckpt["head_state_dict"])
    head.eval()
    model = FrozenSpecialistMidFusion(specialists_bundle["models"], head).to(device)
    model.eval()
    return model, ckpt


def threshold_sweep(rows, conditions):
    sweep = []
    thresholds = [round(x, 2) for x in np.arange(0.05, 0.951, 0.05)]
    mid_rows = [r for r in rows if r["system"] == "mid_fusion"]
    for thr in thresholds:
        tmp = []
        for condition, target_positive in conditions:
            sub = [r for r in mid_rows if r["condition"] == condition]
            flags = [int(r["score"] >= thr) for r in sub]
            smooth = smooth_sequence(flags)
            rate = 100.0 * sum(smooth) / max(len(smooth), 1)
            tmp.append((condition, target_positive, rate))
            sweep.append({"threshold": thr, "condition": condition, "target": "positive" if target_positive else "negative", "smoothed_detection_rate_percent": rate})
        pos_rates = [rate for _, pos, rate in tmp if pos]
        neg_rates = [rate for _, pos, rate in tmp if not pos]
        sweep.append({
            "threshold": thr,
            "condition": "overall",
            "target": "mixed",
            "positive_recall_percent": float(np.mean(pos_rates)) if pos_rates else 0.0,
            "negative_false_alarm_percent": float(np.mean(neg_rates)) if neg_rates else 0.0,
            "score_index": (float(np.mean(pos_rates)) if pos_rates else 0.0) - 2.0 * (float(np.mean(neg_rates)) if neg_rates else 0.0),
        })
    return sweep


def plot_summary(summary_rows, sweep_rows, out_path: Path):
    systems = ["mid_fusion", "specialists_only", "phase3_hybrid"]
    overall = [r for r in summary_rows if r["condition"] == "overall"]
    recalls = [next((float(r["positive_recall_percent"]) for r in overall if r["system"] == s), 0.0) for s in systems]
    fars = [next((float(r["negative_false_alarm_percent"]) for r in overall if r["system"] == s), 0.0) for s in systems]
    sweep_overall = [r for r in sweep_rows if r["condition"] == "overall"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    x = np.arange(len(systems))
    axes[0].bar(x - 0.18, recalls, width=0.36, label="Recall")
    axes[0].bar(x + 0.18, fars, width=0.36, label="False alarm")
    axes[0].set_xticks(x, ["Mid fusion", "5-specialist rule", "Hybrid guard"], rotation=15, ha="right")
    axes[0].set_ylabel("Smoothed rate (%)")
    axes[0].set_title("Mid Fusion vs Existing")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].plot([float(r["threshold"]) for r in sweep_overall], [float(r["positive_recall_percent"]) for r in sweep_overall], marker="o", label="Recall")
    axes[1].plot([float(r["threshold"]) for r in sweep_overall], [float(r["negative_false_alarm_percent"]) for r in sweep_overall], marker="o", label="False alarm")
    axes[1].set_xlabel("Mid-fusion threshold")
    axes[1].set_ylabel("Rate (%)")
    axes[1].set_title("Mid-Fusion Threshold Sweep")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def parse_args():
    ap = argparse.ArgumentParser(description="Benchmark mid-fusion against existing Phase 3 systems")
    ap.add_argument("--mid-model", type=Path, default=SAVE_PATH)
    ap.add_argument("--specialists", type=Path, default=SPECIALIST_BUNDLE_PATH)
    ap.add_argument("--phase2", type=Path, default=PHASE2_GUARD_PATH)
    ap.add_argument("--backbone", type=Path, default=PHASE2_BACKBONE_PATH)
    ap.add_argument("--windows-per-condition", type=int, default=250)
    ap.add_argument("--threshold", type=float, default=0.50)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=6363)
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
        if args.mid_model == SAVE_PATH:
            quick_path = SAVE_PATH.with_name(SAVE_PATH.stem + "_quick" + SAVE_PATH.suffix)
            if quick_path.exists():
                args.mid_model = quick_path
    ensure_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_gpu else "cpu")
    preproc = AudioPreprocessor(16000)
    pools = build_pools(args, preproc)
    specialists = load_specialist_bundle(args.specialists, device)
    guard = load_phase2_guard(args.phase2, args.backbone, device)
    mid_model, _ = load_mid_fusion(args.mid_model, specialists, device)
    rng = random.Random(args.seed)

    conditions = [("drone_alone", True)]
    conditions.extend((f"drone_plus_fsd_snr_{snr:+d}db", True) for snr in SNR_LEVELS)
    conditions.extend([("fsd_real_noise_alone", False), ("dads_no_drone_alone", False)])
    systems = ["mid_fusion", "specialists_only", "phase3_hybrid"]
    smoothers = {system: TemporalSmoother(TEMPORAL_SMOOTHING) for system in systems}
    rows = []
    print("Phase 3 mid-fusion benchmark")
    print(f"Mid model  : {args.mid_model}")
    print(f"Device     : {device}")
    t0 = time.perf_counter()
    for condition, target_positive in conditions:
        for system in systems:
            smoothers[system] = TemporalSmoother(TEMPORAL_SMOOTHING)
        for wi in range(args.windows_per_condition):
            audio = sample_condition(condition, pools, preproc, rng)
            sp = predict_specialists(specialists, preproc, audio)
            gd = predict_phase2_guard(guard, preproc, audio)
            hy = fuse_phase3(sp, gd)
            mid_score = predict_mid(mid_model, preproc, audio, device)
            values = {
                "mid_fusion": (mid_score >= args.threshold, mid_score, f"mid_threshold_{args.threshold:.2f}"),
                "specialists_only": (sp.candidate, sp.score, "specialist_rule"),
                "phase3_hybrid": (hy.detected, hy.score, hy.reason),
            }
            for system, (detected, score, reason) in values.items():
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
                    "mid_score": float(mid_score),
                    "specialist_score": float(sp.score),
                    "specialist_filtered_max": float(sp.filtered_max),
                    "specialist_vote_count": int(sp.vote_count),
                    "phase2_score": float(gd.score),
                    "vehicle_risk_score": float(gd.vehicle_risk_score),
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
            "positive_recall_percent": 100.0 * sum(int(r["smoothed_detected"]) for r in pos) / max(len(pos), 1),
            "negative_false_alarm_percent": 100.0 * sum(int(r["smoothed_detected"]) for r in neg) / max(len(neg), 1),
            "mean_score": float(np.mean([r["score"] for r in pos + neg])) if pos or neg else 0.0,
        })
    sweep = threshold_sweep(rows, conditions)
    tag = args.mid_model.stem.replace("drone_cnn_", "")
    if args.quick and not tag.endswith("_quick"):
        tag += "_quick"
    summary_path = RESULTS_DIR / f"benchmark_{tag}_summary.csv"
    debug_path = RESULTS_DIR / f"benchmark_{tag}_debug.csv"
    sweep_path = RESULTS_DIR / f"benchmark_{tag}_threshold_sweep.csv"
    write_csv(summary_path, summary)
    write_csv(debug_path, rows)
    write_csv(sweep_path, sweep)
    graph_path = RESULTS_DIR / f"benchmark_{tag}.png"
    plot_summary(summary, sweep, graph_path)
    public_graph = Path(__file__).resolve().parents[3] / "results" / "graphs" / "mid_fusion_vs_existing.png"
    plot_summary(summary, sweep, public_graph)

    print("\nSummary")
    for row in summary:
        if row["condition"] == "overall":
            print(f"{row['system']:<18} recall={float(row['positive_recall_percent']):6.2f}% FAR={float(row['negative_false_alarm_percent']):6.2f}%")
    sweep_overall = [r for r in sweep if r["condition"] == "overall"]
    best = max(sweep_overall, key=lambda r: float(r["score_index"]))
    print(f"Best mid threshold by recall - 2*FAR: {float(best['threshold']):.2f} recall={float(best['positive_recall_percent']):.2f}% FAR={float(best['negative_false_alarm_percent']):.2f}%")
    print(f"Saved summary -> {summary_path}")
    print(f"Saved graph   -> {graph_path}")
    print(f"Elapsed: {time.perf_counter() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
