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
    from src.phase2v5_real_noise.config_phase2v5 import QUICK_SAVE_PATH, RESULTS_DIR, SNR_LEVELS, VIEW_WEIGHTS, ensure_dirs
    from src.phase2v5_real_noise.data_phase2v5 import build_pools
    from src.phase2v5_real_noise.model_phase2v5 import DroneCNNV5, extract_model_state
else:
    from .audio_phase2v5 import AudioPreprocessor
    from .config_phase2v5 import QUICK_SAVE_PATH, RESULTS_DIR, SNR_LEVELS, VIEW_WEIGHTS, ensure_dirs
    from .data_phase2v5 import build_pools
    from .model_phase2v5 import DroneCNNV5, extract_model_state


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def predict_generalist(model, preproc, audio, device):
    if np.abs(audio).max() < preproc.noise_floor:
        return 0.0, np.zeros(5, dtype=np.float32), False
    probs = np.zeros(5, dtype=np.float32)
    for vi, view in enumerate(preproc.create_audio_views(audio)):
        lm = preproc.audio_to_logmel(view)
        x = lm.unsqueeze(0).unsqueeze(0).to(device)
        probs[vi] = torch.softmax(model(x), dim=1)[0, 0].item()
    ws = float(VIEW_WEIGHTS @ probs)
    fm = float(probs[1:].max())
    vc = int((probs > 0.60).sum())
    detected = (fm > 0.75) or (ws > 0.60) or (vc >= 2)
    return ws, probs, detected


def summarize(system: str, condition: str, target_positive: bool, rows: list[dict]) -> dict:
    n = len(rows)
    det = sum(int(r["detected"]) for r in rows)
    rate = 100.0 * det / max(n, 1)
    return {
        "system": system,
        "condition": condition,
        "target": "positive" if target_positive else "negative",
        "windows": n,
        "detected": det,
        "detection_rate_percent": rate,
        "mean_score": float(np.mean([r["score"] for r in rows])) if rows else 0.0,
    }


def parse_args():
    ap = argparse.ArgumentParser(description="Phase 1b benchmark for Phase 2v5 generalist")
    ap.add_argument("--checkpoint", type=Path, default=QUICK_SAVE_PATH)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--sample_rate", type=int, default=16000)
    ap.add_argument("--no-gpu", action="store_true")
    ap.add_argument("--seed", type=int, default=1441)
    ap.add_argument("--max-drone-files", type=int, default=1200)
    ap.add_argument("--max-nodrone-files", type=int, default=800)
    ap.add_argument("--max-fsd-clips-per-label", type=int, default=35)
    ap.add_argument("--windows-per-condition", type=int, default=250)
    ap.add_argument("--per-snr", action="store_true", help="Also report mixed drone+FSD recall at each SNR")
    ap.add_argument("--decision-mode", choices=["legacy", "score"], default="legacy")
    ap.add_argument("--score-thr", type=float, default=0.60)
    return ap.parse_args()


def result_tag(args) -> str:
    base = args.checkpoint.stem.replace("drone_cnn_", "")
    suffix = "_per_snr" if args.per_snr else ""
    quick = "_quick" if args.quick else ""
    mode = f"_{args.decision_mode}_thr_{str(args.score_thr).replace('.', 'p')}" if args.decision_mode != "legacy" else ""
    return f"{base}{suffix}{quick}{mode}"


def main():
    args = parse_args()
    if args.quick:
        args.windows_per_condition = min(args.windows_per_condition, 80)
        args.max_drone_files = min(args.max_drone_files, 1200)
        args.max_nodrone_files = min(args.max_nodrone_files, 800)
        args.max_fsd_clips_per_label = min(args.max_fsd_clips_per_label, 35)
    ensure_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_gpu else "cpu")
    ckpt = torch.load(str(args.checkpoint), map_location=device, weights_only=False)
    preproc = AudioPreprocessor(int(ckpt.get("sample_rate", args.sample_rate)))
    model = DroneCNNV5(n_classes=2).to(device)
    model.load_state_dict(extract_model_state(ckpt))
    model.eval()
    drone_pool, fsd_pool, nodrone_pool = build_pools(args, preproc)
    rng = random.Random(args.seed)

    print("Phase 1b benchmark: Phase 2v5 generalist")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {device}, windows_per_condition={args.windows_per_condition}")

    debug_rows = []
    summary_rows = []
    t0 = time.perf_counter()

    conditions = [
        ("drone_alone", True),
        ("drone_plus_fsd_real_noise", True),
        ("fsd_real_noise_alone", False),
        ("dads_no_drone_alone", False),
    ]
    if args.per_snr:
        conditions = [("drone_alone", True)]
        conditions.extend((f"drone_plus_fsd_snr_{snr:+d}db", True) for snr in SNR_LEVELS)
        conditions.extend([("fsd_real_noise_alone", False), ("dads_no_drone_alone", False)])
    for condition, target_positive in conditions:
        condition_rows = []
        for wi in range(args.windows_per_condition):
            if condition == "drone_alone":
                audio = drone_pool.sample_window(rng)
            elif condition == "drone_plus_fsd_real_noise":
                drone = drone_pool.sample_window(rng)
                noise = fsd_pool.sample_window(rng)
                audio = preproc.mix_at_snr(drone, noise, rng.choice(SNR_LEVELS))
            elif condition.startswith("drone_plus_fsd_snr_"):
                drone = drone_pool.sample_window(rng)
                noise = fsd_pool.sample_window(rng)
                snr = int(condition.split("_snr_", 1)[1].replace("db", ""))
                audio = preproc.mix_at_snr(drone, noise, snr)
            elif condition == "fsd_real_noise_alone":
                audio = fsd_pool.sample_window(rng)
            else:
                audio = nodrone_pool.sample_window(rng) if nodrone_pool else fsd_pool.sample_window(rng)
            score, probs, legacy_detected = predict_generalist(model, preproc, audio, device)
            detected = bool(score > args.score_thr) if args.decision_mode == "score" else bool(legacy_detected)
            row = {
                "condition": condition,
                "window_index": wi,
                "target_positive": int(target_positive),
                "detected": int(detected),
                "score": float(score),
                "p_raw": float(probs[0]),
                "p_hpf150": float(probs[1]),
                "p_hpf250": float(probs[2]),
                "p_bpf200": float(probs[3]),
                "p_bpf500": float(probs[4]),
            }
            condition_rows.append(row)
            debug_rows.append(row)
        summary_rows.append(summarize("phase2v5_generalist", condition, target_positive, condition_rows))

    pos = [r for r in debug_rows if r["target_positive"]]
    neg = [r for r in debug_rows if not r["target_positive"]]
    tp = sum(1 for r in pos if r["detected"])
    fn = len(pos) - tp
    fp = sum(1 for r in neg if r["detected"])
    tn = len(neg) - fp
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    summary_rows.append({
        "system": "phase2v5_generalist",
        "condition": "overall",
        "target": "mixed",
        "windows": len(debug_rows),
        "detected": tp + fp,
        "detection_rate_percent": 100.0 * (tp + fp) / max(len(debug_rows), 1),
        "mean_score": float(np.mean([r["score"] for r in debug_rows])) if debug_rows else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_alarm_rate": fp / max(len(neg), 1),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    })

    prefix = f"phase1b_{result_tag(args)}"
    summary_path = RESULTS_DIR / f"{prefix}_summary.csv"
    debug_path = RESULTS_DIR / f"{prefix}_window_debug.csv"
    write_csv(summary_path, summary_rows)
    write_csv(debug_path, debug_rows)

    print("\nSummary")
    for row in summary_rows:
        print(f"{row['condition']:<26} det={row['detection_rate_percent']:6.2f}% mean={row['mean_score']:.3f}")
    print(f"Saved summary -> {summary_path}")
    print(f"Saved debug -> {debug_path}")
    print(f"Elapsed: {time.perf_counter() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
