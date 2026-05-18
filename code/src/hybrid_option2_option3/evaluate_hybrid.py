"""
Evaluation engine for the Option2 + Option3 hybrid.
"""

import csv
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from audio_views import (
    collect_wav_windows,
    load_wav,
    mix_at_snr,
    synth_crowd,
    synth_engine,
    synth_pure_noise,
    synth_tank,
    synth_wind,
    window_audio,
)
from config_hybrid import (
    CONFUSION_DIR,
    DEFAULT_N_WINDOWS,
    DRONE_DIR,
    MAX_DRONE_FILES,
    NODRONE_DIR,
    OPTION2_MODEL_PATH,
    OPTION2_THRESHOLD_SWEEP,
    OPTION3_MODEL_PATH,
    OPTION3_THRESHOLD_SWEEP,
    RANDOM_SEED,
    RESULTS_DIR,
    SPEECH_DIRS,
    TIMELINE_DIR,
    WIND_DIRS,
    WIN_SAMPLES,
    ensure_dirs,
)
from fuse_option2_option3 import TemporalSmoother, fuse_predictions
from load_models import load_hybrid_models
from predict_option2 import predict_option2
from predict_option3 import predict_option3


@dataclass
class EvalConfig:
    n_windows: int = DEFAULT_N_WINDOWS
    hybrid_rule: str = "B"
    option3_score_method: str = "weighted_average"
    smoothing_mode: str = "2of3"
    enable_veto: bool = True
    seed: int = RANDOM_SEED


def _option2_detect_from_probs(probs: np.ndarray) -> bool:
    ws = float(np.array([0.05, 0.20, 0.25, 0.35, 0.15], dtype=np.float32) @ probs)
    fm = float(probs[1:].max())
    vc = int((probs > 0.60).sum())
    return (fm > 0.75) or (ws > 0.60) or (vc >= 2)


def _metrics(tp, fp, tn, fn):
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return precision, recall, f1


class WindowPool:
    def __init__(self, windows: list[np.ndarray], fallback_fn=None):
        self.windows = windows
        self.fallback_fn = fallback_fn
        self.pos = 0

    def next(self, i: int):
        if self.windows:
            if self.pos >= len(self.windows):
                random.shuffle(self.windows)
                self.pos = 0
            w = self.windows[self.pos].copy()
            self.pos += 1
            return w
        if self.fallback_fn is not None:
            return self.fallback_fn(WIN_SAMPLES, i)
        return synth_pure_noise(WIN_SAMPLES, i)


def build_drone_windows(max_files=MAX_DRONE_FILES):
    files = [f for f in sorted(DRONE_DIR.glob("*.wav")) if sf.info(str(f)).frames >= WIN_SAMPLES]
    if max_files and len(files) > max_files:
        files = random.sample(files, max_files)
    wins = []
    for f in files:
        try:
            wins.extend(window_audio(load_wav(f)))
        except Exception:
            pass
    random.shuffle(wins)
    return wins


def build_scenarios(n_windows: int):
    drone = WindowPool(build_drone_windows())
    speech = WindowPool(collect_wav_windows(SPEECH_DIRS, n_windows), synth_crowd)
    wind = WindowPool(collect_wav_windows(WIND_DIRS, n_windows), synth_wind)
    nodrone = WindowPool(collect_wav_windows([NODRONE_DIR], n_windows), synth_pure_noise)

    scenarios = [
        ("drone alone", True, lambda i: drone.next(i)),
        ("drone+tank 0dB", True, lambda i: mix_at_snr(drone.next(i), synth_tank(WIN_SAMPLES, i), 0)),
        ("drone+tank -5dB", True, lambda i: mix_at_snr(drone.next(i), synth_tank(WIN_SAMPLES, i), -5)),
        ("drone+tank -10dB", True, lambda i: mix_at_snr(drone.next(i), synth_tank(WIN_SAMPLES, i), -10)),
        ("drone+tank -20dB", True, lambda i: mix_at_snr(drone.next(i), synth_tank(WIN_SAMPLES, i), -20)),
        ("drone+engine", True, lambda i: mix_at_snr(drone.next(i), synth_engine(WIN_SAMPLES, i), 0)),
        ("drone+crowd", True, lambda i: mix_at_snr(drone.next(i), synth_crowd(WIN_SAMPLES, i), 0)),
        ("drone+speech", True, lambda i: mix_at_snr(drone.next(i), speech.next(i), 0)),
        ("tank alone", False, lambda i: synth_tank(WIN_SAMPLES, i)),
        ("engine alone", False, lambda i: synth_engine(WIN_SAMPLES, i)),
        ("crowd alone", False, lambda i: synth_crowd(WIN_SAMPLES, i)),
        ("speech alone", False, lambda i: speech.next(i)),
        ("pure noise", False, lambda i: synth_pure_noise(WIN_SAMPLES, i)),
        ("wind alone", False, lambda i: wind.next(i)),
        ("no_drone wav", False, lambda i: nodrone.next(i)),
    ]
    return scenarios


def evaluate_systems(models_bundle, cfg: EvalConfig):
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    scenarios = build_scenarios(cfg.n_windows)
    smoother = TemporalSmoother(cfg.smoothing_mode)

    systems = ["option2", "option3", "hybrid_no_smoothing", "hybrid_smoothed"]
    totals = {s: {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "scores": []} for s in systems}
    rows = []
    raw_records = []

    t0 = time.perf_counter()
    for cond_name, is_positive, gen_fn in scenarios:
        smoother.reset()
        cond = {s: {"det": 0, "scores": []} for s in systems}

        for i in range(cfg.n_windows):
            audio = gen_fn(i)
            o2 = predict_option2(models_bundle["option2"], audio, models_bundle["device"])
            o3 = predict_option3(
                models_bundle["option3"],
                audio,
                models_bundle["device"],
                method=cfg.option3_score_method,
            )
            hybrid = fuse_predictions(o2.score, o3, cfg.hybrid_rule, cfg.enable_veto)
            hybrid_smooth_det = smoother.update(hybrid.detected)

            dets = {
                "option2": _option2_detect_from_probs(o2.per_view_probs),
                "option3": o3.detected_alone,
                "hybrid_no_smoothing": hybrid.detected,
                "hybrid_smoothed": hybrid_smooth_det,
            }
            scores = {
                "option2": o2.score,
                "option3": o3.score,
                "hybrid_no_smoothing": min(1.0, 0.5 * o2.score + 0.5 * o3.score),
                "hybrid_smoothed": min(1.0, 0.5 * o2.score + 0.5 * o3.score),
            }

            raw_records.append({
                "condition": cond_name,
                "window_index": i,
                "expected": int(is_positive),
                "option2_score": o2.score,
                "option3_score": o3.score,
                "option3_filtered_max": o3.filtered_max,
                "option3_weighted_average": o3.weighted_average,
                "option3_vote_count": o3.vote_count,
                "hybrid_detected": int(hybrid.detected),
                "hybrid_smoothed": int(hybrid_smooth_det),
                "hybrid_reason": hybrid.reason,
                "hybrid_vetoed": int(hybrid.vetoed),
            })

            for sys_name in systems:
                det = dets[sys_name]
                cond[sys_name]["det"] += int(det)
                cond[sys_name]["scores"].append(scores[sys_name])
                totals[sys_name]["scores"].append(scores[sys_name])
                if is_positive and det:
                    totals[sys_name]["tp"] += 1
                elif is_positive and not det:
                    totals[sys_name]["fn"] += 1
                elif not is_positive and det:
                    totals[sys_name]["fp"] += 1
                else:
                    totals[sys_name]["tn"] += 1

        for sys_name in systems:
            rate = 100.0 * cond[sys_name]["det"] / max(cfg.n_windows, 1)
            mean_score = float(np.mean(cond[sys_name]["scores"]))
            rows.append({
                "system": sys_name,
                "condition": cond_name,
                "expected": "positive" if is_positive else "negative",
                "detection_rate_percent": rate,
                "false_alarm_rate_percent": 0.0 if is_positive else rate,
                "recall_percent": rate if is_positive else "",
                "mean_drone_probability": mean_score,
            })

    summary = []
    for sys_name in systems:
        t = totals[sys_name]
        precision, recall, f1 = _metrics(t["tp"], t["fp"], t["tn"], t["fn"])
        summary.append({
            "system": sys_name,
            "tp": t["tp"],
            "fp": t["fp"],
            "tn": t["tn"],
            "fn": t["fn"],
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mean_drone_probability": float(np.mean(t["scores"])),
        })

    return {
        "condition_rows": rows,
        "summary_rows": summary,
        "raw_records": raw_records,
        "elapsed_seconds": time.perf_counter() - t0,
        "config": cfg,
    }


def run_threshold_sweep(models_bundle, cfg: EvalConfig):
    random.seed(cfg.seed + 1000)
    np.random.seed(cfg.seed + 1000)
    wanted = {
        "drone alone",
        "drone+tank 0dB",
        "drone+engine",
        "tank alone",
        "engine alone",
        "crowd alone",
    }
    scenarios = [s for s in build_scenarios(cfg.n_windows) if s[0] in wanted]
    cache = {}
    for cond_name, is_positive, gen_fn in scenarios:
        vals = []
        for i in range(cfg.n_windows):
            audio = gen_fn(i)
            o2 = predict_option2(models_bundle["option2"], audio, models_bundle["device"])
            o3 = predict_option3(
                models_bundle["option3"],
                audio,
                models_bundle["device"],
                method=cfg.option3_score_method,
            )
            vals.append((o2.score, o3.score))
        cache[cond_name] = np.array(vals, dtype=np.float32)

    rows = []
    for o3_thr in OPTION3_THRESHOLD_SWEEP:
        for o2_thr in OPTION2_THRESHOLD_SWEEP:
            row = {
                "option3_threshold": o3_thr,
                "option2_confirmation_threshold": o2_thr,
            }
            for cond_name, arr in cache.items():
                det = (arr[:, 1] > o3_thr) & (arr[:, 0] > o2_thr)
                row[cond_name] = float(det.mean() * 100.0)
            rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_confusion_matrices(summary_rows: list[dict]):
    for row in summary_rows:
        path = CONFUSION_DIR / f"{row['system']}_confusion_matrix.csv"
        write_csv(path, [
            {"actual": "positive", "predicted_positive": row["tp"], "predicted_negative": row["fn"]},
            {"actual": "negative", "predicted_positive": row["fp"], "predicted_negative": row["tn"]},
        ])


def write_timeline_plots(raw_records: list[dict]):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    by_condition = {}
    for row in raw_records:
        by_condition.setdefault(row["condition"], []).append(row)

    for condition, rows in by_condition.items():
        rows = sorted(rows, key=lambda r: int(r.get("window_index", 0)))
        x = [int(r.get("window_index", i)) for i, r in enumerate(rows)]
        option2 = [float(r["option2_score"]) for r in rows]
        option3 = [float(r["option3_score"]) for r in rows]
        hybrid = [int(r["hybrid_smoothed"]) for r in rows]

        fig, ax1 = plt.subplots(figsize=(10, 4))
        ax1.plot(x, option2, label="Option 2 score", linewidth=1.3)
        ax1.plot(x, option3, label="Option 3 score", linewidth=1.3)
        ax1.set_ylim(0, 1)
        ax1.set_xlabel("Window")
        ax1.set_ylabel("Score")
        ax1.grid(True, alpha=0.25)
        ax2 = ax1.twinx()
        ax2.step(x, hybrid, where="post", label="Hybrid smoothed detect", color="black", alpha=0.5)
        ax2.set_ylim(-0.05, 1.05)
        ax2.set_ylabel("Detection")
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc="upper right")
        fig.suptitle(condition)
        safe = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in condition)
        fig.tight_layout()
        fig.savefig(TIMELINE_DIR / f"{safe}.png", dpi=120)
        plt.close(fig)


def save_results(eval_result, sweep_rows):
    ensure_dirs()
    write_csv(RESULTS_DIR / "hybrid_metrics.csv", eval_result["condition_rows"])
    write_csv(RESULTS_DIR / "hybrid_comparison_table.csv", eval_result["summary_rows"])
    write_csv(RESULTS_DIR / "hybrid_threshold_sweep.csv", sweep_rows)
    write_csv(RESULTS_DIR / "logs" / "hybrid_window_log.csv", eval_result["raw_records"])
    write_confusion_matrices(eval_result["summary_rows"])
    write_timeline_plots(eval_result["raw_records"])


def recommend(eval_result):
    summaries = {r["system"]: r for r in eval_result["summary_rows"]}
    best = max(eval_result["summary_rows"], key=lambda r: r["f1"])
    if best["system"] == "hybrid_smoothed":
        verdict = "Hybrid is better"
    elif best["system"] == "hybrid_no_smoothing":
        verdict = "Hybrid is promising but needs tuning"
    elif best["system"] == "option3":
        verdict = "Option 3 is better"
    else:
        verdict = "Option 2 remains primary"

    option2 = summaries.get("option2", {})
    hybrid = summaries.get("hybrid_smoothed", {})
    if option2 and hybrid and hybrid.get("f1", 0) < option2.get("f1", 0):
        verdict = "Option 2 remains primary"
    return verdict, best


def evaluate_and_save(cfg: EvalConfig, option2_path=None, option3_path=None):
    ensure_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = load_hybrid_models(
        option2_path=option2_path if option2_path is not None else OPTION2_MODEL_PATH,
        option3_path=option3_path if option3_path is not None else OPTION3_MODEL_PATH,
        device=device,
    )
    eval_result = evaluate_systems(models, cfg)
    sweep_rows = run_threshold_sweep(models, cfg)
    save_results(eval_result, sweep_rows)
    verdict, best = recommend(eval_result)
    return eval_result, sweep_rows, verdict, best
