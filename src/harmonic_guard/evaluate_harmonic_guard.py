"""Evaluate the experimental harmonic guard against the existing hybrid."""

import argparse
import csv
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config_harmonic_guard import DEFAULT_N_WINDOWS, RANDOM_SEED, RESULTS_DIR, ensure_dirs
from .guard_fusion import apply_harmonic_guard
from .harmonic_analyzer import analyze_harmonics
from .hybrid_bridge import add_hybrid_path
from .synthetic_vehicles import synth_generator, synth_vehicle

add_hybrid_path()

from evaluate_hybrid import WindowPool, build_drone_windows  # noqa: E402
from audio_views import (  # noqa: E402
    WIN_SAMPLES,
    collect_wav_windows,
    mix_at_snr,
    synth_crowd,
    synth_engine,
    synth_pure_noise,
    synth_tank,
    synth_wind,
)
from config_hybrid import NODRONE_DIR, SPEECH_DIRS, WIND_DIRS  # noqa: E402
from fuse_option2_option3 import TemporalSmoother, fuse_predictions  # noqa: E402
from load_models import load_hybrid_models  # noqa: E402
from predict_option2 import predict_option2  # noqa: E402
from predict_option3 import predict_option3  # noqa: E402


@dataclass
class GuardEvalConfig:
    n_windows: int = DEFAULT_N_WINDOWS
    seed: int = RANDOM_SEED
    hybrid_rule: str = "B"
    option3_score_method: str = "weighted_average"
    smoothing_mode: str = "2of3"
    enable_hybrid_veto: bool = True


def _metrics(tp, fp, tn, fn):
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return precision, recall, f1


def _write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_guard_scenarios(n_windows: int):
    max_drone_files = max(80, min(300, n_windows * 8))
    drone = WindowPool(build_drone_windows(max_files=max_drone_files))
    speech = WindowPool(collect_wav_windows(SPEECH_DIRS, n_windows), synth_crowd)
    wind = WindowPool(collect_wav_windows(WIND_DIRS, n_windows), synth_wind)
    nodrone = WindowPool(collect_wav_windows([NODRONE_DIR], n_windows), synth_pure_noise)

    return [
        ("drone alone", True, lambda i: drone.next(i)),
        ("drone+tank 0dB", True, lambda i: mix_at_snr(drone.next(i), synth_tank(WIN_SAMPLES, i), 0)),
        ("drone+tank -5dB", True, lambda i: mix_at_snr(drone.next(i), synth_tank(WIN_SAMPLES, i), -5)),
        ("drone+tank -10dB", True, lambda i: mix_at_snr(drone.next(i), synth_tank(WIN_SAMPLES, i), -10)),
        ("drone+tank -20dB", True, lambda i: mix_at_snr(drone.next(i), synth_tank(WIN_SAMPLES, i), -20)),
        ("drone+engine", True, lambda i: mix_at_snr(drone.next(i), synth_engine(WIN_SAMPLES, i), 0)),
        ("drone+generator", True, lambda i: mix_at_snr(drone.next(i), synth_generator(WIN_SAMPLES, i), 0)),
        ("drone+vehicle", True, lambda i: mix_at_snr(drone.next(i), synth_vehicle(WIN_SAMPLES, i), 0)),
        ("drone+crowd", True, lambda i: mix_at_snr(drone.next(i), synth_crowd(WIN_SAMPLES, i), 0)),
        ("drone+speech", True, lambda i: mix_at_snr(drone.next(i), speech.next(i), 0)),
        ("tank alone", False, lambda i: synth_tank(WIN_SAMPLES, i)),
        ("engine alone", False, lambda i: synth_engine(WIN_SAMPLES, i)),
        ("generator alone", False, lambda i: synth_generator(WIN_SAMPLES, i)),
        ("vehicle alone", False, lambda i: synth_vehicle(WIN_SAMPLES, i)),
        ("crowd alone", False, lambda i: synth_crowd(WIN_SAMPLES, i)),
        ("speech alone", False, lambda i: speech.next(i)),
        ("pure noise", False, lambda i: synth_pure_noise(WIN_SAMPLES, i)),
        ("wind alone", False, lambda i: wind.next(i)),
        ("no_drone wav", False, lambda i: nodrone.next(i)),
    ]


def evaluate_guard(models_bundle, cfg: GuardEvalConfig):
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    scenarios = build_guard_scenarios(cfg.n_windows)

    systems = ["hybrid_smoothed", "guarded_hybrid_smoothed"]
    totals = {s: {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "scores": []} for s in systems}
    condition_rows = []
    raw_rows = []
    t0 = time.perf_counter()

    for cond_name, is_positive, gen_fn in scenarios:
        hybrid_smoother = TemporalSmoother(cfg.smoothing_mode)
        guard_smoother = TemporalSmoother(cfg.smoothing_mode)
        cond = {s: {"det": 0, "scores": [], "downgraded": 0} for s in systems}

        for i in range(cfg.n_windows):
            audio = gen_fn(i)
            o2 = predict_option2(models_bundle["option2"], audio, models_bundle["device"])
            o3 = predict_option3(
                models_bundle["option3"],
                audio,
                models_bundle["device"],
                method=cfg.option3_score_method,
            )
            hybrid = fuse_predictions(o2.score, o3, cfg.hybrid_rule, cfg.enable_hybrid_veto)
            feats = analyze_harmonics(audio)
            guard = apply_harmonic_guard(hybrid, o3, feats)

            hybrid_det = hybrid_smoother.update(hybrid.detected)
            guarded_det = guard_smoother.update(guard.detected)
            score = min(1.0, 0.5 * float(o2.score) + 0.5 * float(o3.score))
            guarded_score = score * (1.0 - 0.35 * float(guard.downgraded))

            dets = {
                "hybrid_smoothed": hybrid_det,
                "guarded_hybrid_smoothed": guarded_det,
            }
            scores = {
                "hybrid_smoothed": score,
                "guarded_hybrid_smoothed": guarded_score,
            }
            for sys_name in systems:
                det = bool(dets[sys_name])
                cond[sys_name]["det"] += int(det)
                cond[sys_name]["scores"].append(scores[sys_name])
                if sys_name == "guarded_hybrid_smoothed":
                    cond[sys_name]["downgraded"] += int(guard.downgraded)
                totals[sys_name]["scores"].append(scores[sys_name])
                if is_positive and det:
                    totals[sys_name]["tp"] += 1
                elif is_positive and not det:
                    totals[sys_name]["fn"] += 1
                elif not is_positive and det:
                    totals[sys_name]["fp"] += 1
                else:
                    totals[sys_name]["tn"] += 1

            raw_rows.append({
                "condition": cond_name,
                "window_index": i,
                "expected_positive": int(is_positive),
                "option2_score": float(o2.score),
                "option3_score": float(o3.score),
                "option3_filtered_max": float(o3.filtered_max),
                "option3_vote_count": int(o3.vote_count),
                "hybrid_raw_detected": int(hybrid.detected),
                "hybrid_smoothed_detected": int(hybrid_det),
                "hybrid_reason": hybrid.reason,
                "hybrid_vetoed": int(hybrid.vetoed),
                "guard_raw_detected": int(guard.detected),
                "guard_smoothed_detected": int(guarded_det),
                "guard_downgraded": int(guard.downgraded),
                "guard_reason": guard.reason,
                **feats.to_dict(),
            })

        for sys_name in systems:
            rate = 100.0 * cond[sys_name]["det"] / max(cfg.n_windows, 1)
            condition_rows.append({
                "system": sys_name,
                "condition": cond_name,
                "expected": "positive" if is_positive else "negative",
                "detection_rate_percent": rate,
                "false_alarm_rate_percent": "" if is_positive else rate,
                "recall_percent": rate if is_positive else "",
                "mean_score": float(np.mean(cond[sys_name]["scores"])),
                "downgraded_windows": cond[sys_name]["downgraded"],
            })

    summary_rows = []
    for sys_name in systems:
        t = totals[sys_name]
        precision, recall, f1 = _metrics(t["tp"], t["fp"], t["tn"], t["fn"])
        summary_rows.append({
            "system": sys_name,
            "tp": t["tp"],
            "fp": t["fp"],
            "tn": t["tn"],
            "fn": t["fn"],
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mean_score": float(np.mean(t["scores"])),
        })

    return {
        "summary_rows": summary_rows,
        "condition_rows": condition_rows,
        "raw_rows": raw_rows,
        "elapsed_seconds": time.perf_counter() - t0,
    }


def save_results(result):
    ensure_dirs()
    summary_path = RESULTS_DIR / "harmonic_guard_summary.csv"
    condition_path = RESULTS_DIR / "harmonic_guard_conditions.csv"
    raw_path = RESULTS_DIR / "harmonic_guard_window_debug.csv"
    _write_csv(summary_path, result["summary_rows"])
    _write_csv(condition_path, result["condition_rows"])
    _write_csv(raw_path, result["raw_rows"])
    return summary_path, condition_path, raw_path


def main():
    parser = argparse.ArgumentParser(description="Evaluate experimental harmonic guard.")
    parser.add_argument("--windows", type=int, default=DEFAULT_N_WINDOWS)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--rule", default="B", choices=["A", "B", "C", "D"])
    parser.add_argument("--smoothing", default="2of3", choices=["none", "2of3", "3of5", "persist_1_5s"])
    parser.add_argument("--no-hybrid-veto", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    import torch

    device = torch.device("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    cfg = GuardEvalConfig(
        n_windows=args.windows,
        seed=args.seed,
        hybrid_rule=args.rule,
        smoothing_mode=args.smoothing,
        enable_hybrid_veto=not args.no_hybrid_veto,
    )
    print("Experimental harmonic guard")
    print("Safety: no training, no audio suppression, no model overwrite.")
    print(f"Device: {device}")
    print(f"Windows per condition: {cfg.n_windows}")

    models = load_hybrid_models(device=device)
    result = evaluate_guard(models, cfg)
    summary_path, condition_path, raw_path = save_results(result)

    print()
    for row in result["summary_rows"]:
        print(
            f"{row['system']:24s} "
            f"precision={row['precision']*100:6.2f}% "
            f"recall={row['recall']*100:6.2f}% "
            f"f1={row['f1']*100:6.2f}% "
            f"fp={row['fp']} fn={row['fn']}"
        )
    print()
    print(f"Saved summary    -> {summary_path}")
    print(f"Saved conditions -> {condition_path}")
    print(f"Saved debug      -> {raw_path}")
    print(f"Elapsed seconds  -> {result['elapsed_seconds']:.1f}")


if __name__ == "__main__":
    main()
