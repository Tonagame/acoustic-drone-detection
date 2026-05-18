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
    from src.phase2_harmonic_fusion.config_phase2 import DEFAULT_SAVE_PATH, RESULTS_DIR, SNR_LEVELS, ensure_dirs
    from src.phase2_harmonic_fusion.features_phase2 import (
        PoolArgs,
        harmonic_vector,
        load_backbone,
        sample_audio,
        weighted_cnn_latent,
    )
    from src.phase2_harmonic_fusion.model_phase2 import HarmonicFusionHead
else:
    from .config_phase2 import DEFAULT_SAVE_PATH, RESULTS_DIR, SNR_LEVELS, ensure_dirs
    from .features_phase2 import PoolArgs, harmonic_vector, load_backbone, sample_audio, weighted_cnn_latent
    from .model_phase2 import HarmonicFusionHead

from src.phase2v5_real_noise.audio_phase2v5 import AudioPreprocessor
from src.phase2v5_real_noise.data_phase2v5 import build_pools


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


@torch.no_grad()
def predict_phase2(audio, backbone, head, preproc, device):
    cnn = weighted_cnn_latent(backbone, preproc, audio, device)
    harm = harmonic_vector(audio)
    x = torch.from_numpy(np.concatenate([cnn, harm]).astype(np.float32)).unsqueeze(0).to(device)
    score = float(torch.softmax(head(x), dim=1)[0, 0].item())
    return score, cnn, harm


@torch.no_grad()
def predict_backbone_score(audio, backbone, preproc, device):
    cnn = weighted_cnn_latent(backbone, preproc, audio, device)
    logits = backbone.classify_from_latent(torch.from_numpy(cnn).float().unsqueeze(0).to(device))
    return float(torch.softmax(logits, dim=1)[0, 0].item())


def summarize(rows, condition, target_positive, system, threshold):
    sub = [r for r in rows if r["condition"] == condition and r["system"] == system]
    det = sum(int(float(r["score"]) > threshold) for r in sub)
    return {
        "system": system,
        "condition": condition,
        "target": "positive" if target_positive else "negative",
        "threshold": threshold,
        "windows": len(sub),
        "detected": det,
        "detection_rate_percent": 100.0 * det / max(len(sub), 1),
        "mean_score": float(np.mean([r["score"] for r in sub])) if sub else 0.0,
    }


def parse_args():
    ap = argparse.ArgumentParser(description="Benchmark Phase 2 harmonic fusion")
    ap.add_argument("--phase2", type=Path, default=DEFAULT_SAVE_PATH)
    ap.add_argument("--backbone", type=Path, default=Path("models/phase2v5_real_noise/drone_cnn_phase2v5c_real_noise_balanced.pth"))
    ap.add_argument("--windows-per-condition", type=int, default=250)
    ap.add_argument("--threshold", type=float, default=0.60)
    ap.add_argument("--backbone-threshold", type=float, default=0.60)
    ap.add_argument("--seed", type=int, default=4040)
    ap.add_argument("--no-gpu", action="store_true")
    ap.add_argument("--max-drone-files", type=int, default=12000)
    ap.add_argument("--max-nodrone-files", type=int, default=5000)
    ap.add_argument("--max-fsd-clips-per-label", type=int, default=500)
    return ap.parse_args()


def main():
    args = parse_args()
    ensure_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_gpu else "cpu")
    preproc = AudioPreprocessor(16000)
    pool_args = PoolArgs(args.max_drone_files, args.max_nodrone_files, args.max_fsd_clips_per_label, args.seed)
    drone_pool, fsd_pool, nodrone_pool = build_pools(pool_args, preproc)
    backbone, _ = load_backbone(args.backbone, device)
    ckpt = torch.load(str(args.phase2), map_location=device, weights_only=False)
    input_dim = int(ckpt["metadata"]["input_dim"])
    head = HarmonicFusionHead(in_dim=input_dim).to(device)
    head.load_state_dict(ckpt["head_state_dict"])
    head.eval()
    rng = random.Random(args.seed)

    conditions = [("drone_alone", True)]
    conditions.extend((f"drone_plus_fsd_snr_{snr:+d}db", True) for snr in SNR_LEVELS)
    conditions.extend([("fsd_alone", False), ("nodrone_alone", False)])

    rows = []
    t0 = time.perf_counter()
    print("Phase 2 harmonic fusion benchmark")
    print(f"Phase2: {args.phase2}")
    print(f"Backbone: {args.backbone}")
    for condition, target_positive in conditions:
        for wi in range(args.windows_per_condition):
            if condition.startswith("drone_plus_fsd_snr_"):
                drone = drone_pool.sample_window(rng)
                noise = fsd_pool.sample_window(rng)
                snr = int(condition.split("_snr_", 1)[1].replace("db", ""))
                audio = preproc.mix_at_snr(drone, noise, snr)
            elif condition == "fsd_alone":
                audio = sample_audio("fsd_alone", drone_pool, fsd_pool, nodrone_pool, preproc, rng)
            elif condition == "nodrone_alone":
                audio = sample_audio("nodrone_alone", drone_pool, fsd_pool, nodrone_pool, preproc, rng)
            else:
                audio = sample_audio(condition, drone_pool, fsd_pool, nodrone_pool, preproc, rng)
            phase2_score, _cnn, harm = predict_phase2(audio, backbone, head, preproc, device)
            backbone_score = predict_backbone_score(audio, backbone, preproc, device)
            base = {
                "condition": condition,
                "window_index": wi,
                "target_positive": int(target_positive),
                "vehicle_risk_score": float(harm[6]),
            }
            rows.append({**base, "system": "phase2_harmonic", "score": phase2_score})
            rows.append({**base, "system": "v5c_backbone", "score": backbone_score})

    summary = []
    for condition, target_positive in conditions:
        summary.append(summarize(rows, condition, target_positive, "phase2_harmonic", args.threshold))
        summary.append(summarize(rows, condition, target_positive, "v5c_backbone", args.backbone_threshold))

    tag = args.phase2.stem.replace("drone_cnn_", "")
    tag = f"{tag}_thr_{str(args.threshold).replace('.', 'p')}_backbone_{str(args.backbone_threshold).replace('.', 'p')}"
    summary_path = RESULTS_DIR / f"benchmark_{tag}_summary.csv"
    debug_path = RESULTS_DIR / f"benchmark_{tag}_debug.csv"
    write_csv(summary_path, summary)
    write_csv(debug_path, rows)

    print("\nSummary")
    for row in summary:
        print(f"{row['system']:<16} {row['condition']:<26} det={row['detection_rate_percent']:6.2f}% mean={row['mean_score']:.3f}")
    print(f"Saved summary -> {summary_path}")
    print(f"Saved debug -> {debug_path}")
    print(f"Elapsed: {time.perf_counter() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
