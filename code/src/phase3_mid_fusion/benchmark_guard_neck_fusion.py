from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.phase2v5_real_noise.audio_phase2v5 import AudioPreprocessor
from src.phase2v5_real_noise.config_phase2v5 import FSD_LABELS
from src.phase3_mid_fusion.benchmark_mid_fusion import (
    build_category_pools,
    build_pools,
    false_alarm_by_category,
    latency_summary,
    load_mid_fusion,
    sample_condition,
    smooth_sequence,
    summarize,
    timed_call,
)
from src.phase3_mid_fusion.config_mid_fusion import (
    GUARD_NECK_INPUT_DIM,
    GUARD_NECK_SAVE_PATH,
    RESULTS_DIR,
    SAVE_PATH,
    SNR_LEVELS,
    SPECIALIST_BUNDLE_PATH,
    TEMPORAL_SMOOTHING,
    PHASE2_GUARD_PATH,
    PHASE2_BACKBONE_PATH,
    ensure_dirs,
)
from src.phase3_mid_fusion.model_mid_fusion import GuardNeckFusionHead
from src.phase3_real_noise_specialists.predict_phase3_hybrid import (
    TemporalSmoother,
    fuse_phase3,
    fuse_phase3_soft_guard,
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


@torch.no_grad()
def specialist_latent(specialists_bundle, preproc: AudioPreprocessor, audio: np.ndarray):
    device = specialists_bundle["device"]
    views = preproc.create_audio_views(audio)
    latents = []
    for model, view in zip(specialists_bundle["models"], views):
        lm = preproc.audio_to_logmel(view).unsqueeze(0).unsqueeze(0).to(device)
        latents.append(model.encode(lm).squeeze(0).detach().cpu().numpy())
    return np.concatenate(latents).astype(np.float32)


@torch.no_grad()
def predict_guard_neck(model, specialists_bundle, preproc, audio, guard_pred, device):
    z = specialist_latent(specialists_bundle, preproc, audio)
    gf = np.asarray([guard_pred.score, guard_pred.vehicle_risk_score, guard_pred.f0_norm, guard_pred.harmonicity_score], dtype=np.float32)
    x = torch.from_numpy(np.concatenate([z, gf]).astype(np.float32)).unsqueeze(0).to(device)
    return float(torch.softmax(model(x), dim=1)[0, 0].item())


def load_guard_neck(path: Path, device):
    ckpt = torch.load(str(path), map_location=device, weights_only=False)
    model = GuardNeckFusionHead(int(ckpt["metadata"].get("input_dim", GUARD_NECK_INPUT_DIM))).to(device)
    model.load_state_dict(ckpt["head_state_dict"])
    model.eval()
    return model, ckpt


def threshold_sweep_v2(rows, conditions):
    sweep = []
    thresholds = [round(x, 2) for x in np.arange(0.05, 0.951, 0.05)]
    for system in ["mid_fusion_v1", "guard_neck_v2"]:
        sys_rows = [r for r in rows if r["system"] == system]
        for thr in thresholds:
            tmp = []
            for condition, target_positive in conditions:
                sub = [r for r in sys_rows if r["condition"] == condition]
                flags = [int(r["score"] >= thr) for r in sub]
                smooth = smooth_sequence(flags)
                rate = 100.0 * sum(smooth) / max(len(smooth), 1)
                tmp.append((condition, target_positive, rate))
                sweep.append({"system": system, "threshold": thr, "condition": condition, "target": "positive" if target_positive else "negative", "smoothed_detection_rate_percent": rate})
            pos_rates = [rate for _, pos, rate in tmp if pos]
            neg_rates = [rate for _, pos, rate in tmp if not pos]
            sweep.append({
                "system": system,
                "threshold": thr,
                "condition": "overall",
                "target": "mixed",
                "positive_recall_percent": float(np.mean(pos_rates)) if pos_rates else 0.0,
                "negative_false_alarm_percent": float(np.mean(neg_rates)) if neg_rates else 0.0,
                "score_index": (float(np.mean(pos_rates)) if pos_rates else 0.0) - 2.0 * (float(np.mean(neg_rates)) if neg_rates else 0.0),
            })
    return sweep


def plot_comparison(summary_rows, out_path: Path):
    systems = ["mid_fusion_v1", "guard_neck_v2", "specialists_only", "phase3_hybrid", "phase3_soft_guard"]
    labels = ["Mid v1", "Guard-neck v2", "5-specialist", "Hard guard", "Soft guard"]
    overall = [r for r in summary_rows if r["condition"] == "overall"]
    recalls = [next((float(r["positive_recall_percent"]) for r in overall if r["system"] == s), 0.0) for s in systems]
    fars = [next((float(r["negative_false_alarm_percent"]) for r in overall if r["system"] == s), 0.0) for s in systems]
    x = np.arange(len(systems))
    fig, ax = plt.subplots(figsize=(10, 4.4))
    ax.bar(x - 0.18, recalls, width=0.36, label="Recall")
    ax.bar(x + 0.18, fars, width=0.36, label="False alarm")
    ax.set_xticks(x, labels, rotation=18, ha="right")
    ax.set_ylabel("Smoothed rate (%)")
    ax.set_title("Guard-Neck Fusion vs Mid Fusion")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def parse_args():
    ap = argparse.ArgumentParser(description="Benchmark guard-neck fusion v2 against mid-fusion v1")
    ap.add_argument("--mid-v1", type=Path, default=SAVE_PATH)
    ap.add_argument("--guard-neck", type=Path, default=GUARD_NECK_SAVE_PATH)
    ap.add_argument("--mid-threshold", type=float, default=0.60)
    ap.add_argument("--guard-neck-threshold", type=float, default=0.50)
    ap.add_argument("--specialists", type=Path, default=SPECIALIST_BUNDLE_PATH)
    ap.add_argument("--phase2", type=Path, default=PHASE2_GUARD_PATH)
    ap.add_argument("--backbone", type=Path, default=PHASE2_BACKBONE_PATH)
    ap.add_argument("--windows-per-condition", type=int, default=250)
    ap.add_argument("--category-windows", type=int, default=120)
    ap.add_argument("--category-max-clips-per-label", type=int, default=160)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=7474)
    ap.add_argument("--no-gpu", action="store_true")
    ap.add_argument("--max-drone-files", type=int, default=12000)
    ap.add_argument("--max-nodrone-files", type=int, default=5000)
    ap.add_argument("--max-fsd-clips-per-label", type=int, default=500)
    return ap.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.windows_per_condition = min(args.windows_per_condition, 80)
        args.category_windows = min(args.category_windows, 35)
        args.max_drone_files = min(args.max_drone_files, 1200)
        args.max_nodrone_files = min(args.max_nodrone_files, 800)
        args.max_fsd_clips_per_label = min(args.max_fsd_clips_per_label, 35)
        q1 = SAVE_PATH.with_name(SAVE_PATH.stem + "_quick" + SAVE_PATH.suffix)
        q2 = GUARD_NECK_SAVE_PATH.with_name(GUARD_NECK_SAVE_PATH.stem + "_quick" + GUARD_NECK_SAVE_PATH.suffix)
        if args.mid_v1 == SAVE_PATH and q1.exists():
            args.mid_v1 = q1
        if args.guard_neck == GUARD_NECK_SAVE_PATH and q2.exists():
            args.guard_neck = q2
    ensure_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_gpu else "cpu")
    preproc = AudioPreprocessor(16000)
    pools = build_pools(args, preproc)
    category_pools = build_category_pools(args, preproc)
    specialists = load_specialist_bundle(args.specialists, device)
    guard = load_phase2_guard(args.phase2, args.backbone, device)
    mid_v1, _ = load_mid_fusion(args.mid_v1, specialists, device)
    guard_neck, _ = load_guard_neck(args.guard_neck, device)
    rng = random.Random(args.seed)
    conditions = [("drone_alone", True)]
    conditions.extend((f"drone_plus_fsd_snr_{snr:+d}db", True) for snr in SNR_LEVELS)
    conditions.extend([("fsd_real_noise_alone", False), ("dads_no_drone_alone", False)])
    category_conditions = [(f"fsd_label__{label}", False) for label in FSD_LABELS if label in category_pools]
    eval_conditions = conditions + category_conditions
    systems = ["mid_fusion_v1", "guard_neck_v2", "specialists_only", "phase3_hybrid", "phase3_soft_guard"]
    smoothers = {system: TemporalSmoother(TEMPORAL_SMOOTHING) for system in systems}
    rows = []
    latency_rows = []
    t0 = time.perf_counter()
    print("Guard-neck fusion v2 benchmark")
    print(f"Mid v1     : {args.mid_v1}")
    print(f"Guard-neck : {args.guard_neck}")
    print(f"Device     : {device}")
    for condition, target_positive in eval_conditions:
        for system in systems:
            smoothers[system] = TemporalSmoother(TEMPORAL_SMOOTHING)
        n_windows = args.category_windows if condition.startswith("fsd_label__") else args.windows_per_condition
        for wi in range(n_windows):
            audio = sample_condition(condition, pools, preproc, rng, category_pools)
            sp, sp_ms = timed_call(device, predict_specialists, specialists, preproc, audio)
            gd, gd_ms = timed_call(device, predict_phase2_guard, guard, preproc, audio)
            hy = fuse_phase3(sp, gd)
            soft = fuse_phase3_soft_guard(sp, gd)
            mid_score, mid_ms = timed_call(device, lambda: float(torch.softmax(mid_v1(torch.stack([preproc.audio_to_logmel(v).unsqueeze(0).float() for v in preproc.create_audio_views(audio)], dim=0).unsqueeze(0).to(device)), dim=1)[0, 0].item()))
            gn_score, gn_ms = timed_call(device, predict_guard_neck, guard_neck, specialists, preproc, audio, gd, device)
            values = {
                "mid_fusion_v1": (mid_score >= args.mid_threshold, mid_score, f"mid_v1_thr_{args.mid_threshold:.2f}", mid_ms),
                "guard_neck_v2": (gn_score >= args.guard_neck_threshold, gn_score, f"guard_neck_thr_{args.guard_neck_threshold:.2f}", gn_ms),
                "specialists_only": (sp.candidate, sp.score, "specialist_rule", sp_ms),
                "phase3_hybrid": (hy.detected, hy.score, hy.reason, sp_ms + gd_ms),
                "phase3_soft_guard": (soft.detected, soft.score, soft.reason, sp_ms + gd_ms),
            }
            for system, (detected, score, reason, latency_ms) in values.items():
                smoothed = smoothers[system].update(bool(detected))
                latency_rows.append({"system": system, "condition": condition, "window_index": wi, "latency_ms": float(latency_ms)})
                rows.append({
                    "system": system,
                    "condition": condition,
                    "window_index": wi,
                    "target_positive": int(target_positive),
                    "detected": int(bool(detected)),
                    "smoothed_detected": int(bool(smoothed)),
                    "score": float(score),
                    "reason": reason,
                    "latency_ms": float(latency_ms),
                    "mid_v1_score": float(mid_score),
                    "guard_neck_score": float(gn_score),
                    "specialist_score": float(sp.score),
                    "phase2_score": float(gd.score),
                    "vehicle_risk_score": float(gd.vehicle_risk_score),
                })
    summary = []
    for condition, target_positive in eval_conditions:
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
    sweep = threshold_sweep_v2(rows, conditions)
    latency = latency_summary(latency_rows)
    fa = false_alarm_by_category(rows)
    tag = args.guard_neck.stem.replace("drone_cnn_", "")
    if args.quick and not tag.endswith("_quick"):
        tag += "_quick"
    summary_path = RESULTS_DIR / f"benchmark_{tag}_summary.csv"
    debug_path = RESULTS_DIR / f"benchmark_{tag}_debug.csv"
    sweep_path = RESULTS_DIR / f"benchmark_{tag}_threshold_sweep.csv"
    latency_path = RESULTS_DIR / f"benchmark_{tag}_latency.csv"
    fa_path = RESULTS_DIR / f"benchmark_{tag}_false_alarms_by_category.csv"
    graph_path = RESULTS_DIR / f"benchmark_{tag}.png"
    public_graph = Path(__file__).resolve().parents[3] / "results" / "graphs" / "guard_neck_vs_mid_fusion.png"
    write_csv(summary_path, summary)
    write_csv(debug_path, rows)
    write_csv(sweep_path, sweep)
    write_csv(latency_path, latency)
    write_csv(fa_path, fa)
    plot_comparison(summary, graph_path)
    plot_comparison(summary, public_graph)
    print("\nSummary")
    for row in summary:
        if row["condition"] == "overall":
            print(f"{row['system']:<18} recall={float(row['positive_recall_percent']):6.2f}% FAR={float(row['negative_false_alarm_percent']):6.2f}%")
    best_v2 = max([r for r in sweep if r["system"] == "guard_neck_v2" and r["condition"] == "overall"], key=lambda r: float(r["score_index"]))
    print(f"Best guard-neck threshold by recall - 2*FAR: {float(best_v2['threshold']):.2f} recall={float(best_v2['positive_recall_percent']):.2f}% FAR={float(best_v2['negative_false_alarm_percent']):.2f}%")
    print(f"Saved summary -> {summary_path}")
    print(f"Saved graph   -> {graph_path}")
    print(f"Elapsed: {time.perf_counter() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
