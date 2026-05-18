"""Compare Option2, Option3, Hybrid, and Hybrid+HarmonicGuard."""

import argparse
import csv
import random
import time
from pathlib import Path

import numpy as np

from .config_harmonic_guard import DEFAULT_N_WINDOWS, RANDOM_SEED, RESULTS_DIR, ensure_dirs
from .evaluate_harmonic_guard import build_guard_scenarios
from .guard_fusion import apply_harmonic_guard
from .harmonic_analyzer import analyze_harmonics
from .hybrid_bridge import add_hybrid_path

add_hybrid_path()

from config_hybrid import VIEW_WEIGHTS  # noqa: E402
from fuse_option2_option3 import TemporalSmoother, fuse_predictions  # noqa: E402
from load_models import load_hybrid_models  # noqa: E402
from predict_option2 import predict_option2  # noqa: E402
from predict_option3 import predict_option3  # noqa: E402


SYSTEMS = [
    "option2",
    "option3",
    "hybrid_smoothed",
    "harmonic_guard_smoothed",
]


def _option2_detect_from_probs(probs: np.ndarray) -> bool:
    weights = np.asarray(VIEW_WEIGHTS, dtype=np.float32)
    ws = float(weights @ probs)
    fm = float(probs[1:].max())
    vc = int((probs > 0.60).sum())
    return (fm > 0.75) or (ws > 0.60) or (vc >= 2)


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


def compare_iterations(models_bundle, n_windows: int, seed: int, rule: str, smoothing: str, enable_hybrid_veto: bool):
    random.seed(seed)
    np.random.seed(seed)
    scenarios = build_guard_scenarios(n_windows)

    totals = {s: {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "scores": []} for s in SYSTEMS}
    condition_rows = []
    debug_rows = []
    t0 = time.perf_counter()

    for cond_name, is_positive, gen_fn in scenarios:
        hybrid_smoother = TemporalSmoother(smoothing)
        guard_smoother = TemporalSmoother(smoothing)
        cond = {s: {"det": 0, "scores": [], "downgraded": 0} for s in SYSTEMS}

        for i in range(n_windows):
            audio = gen_fn(i)
            o2 = predict_option2(models_bundle["option2"], audio, models_bundle["device"])
            o3 = predict_option3(models_bundle["option3"], audio, models_bundle["device"], method="weighted_average")
            hybrid = fuse_predictions(o2.score, o3, rule, enable_hybrid_veto)
            feats = analyze_harmonics(audio)
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
                det = bool(dets[system])
                cond[system]["det"] += int(det)
                cond[system]["scores"].append(scores[system])
                totals[system]["scores"].append(scores[system])
                if system == "harmonic_guard_smoothed":
                    cond[system]["downgraded"] += int(guard.downgraded)
                if is_positive and det:
                    totals[system]["tp"] += 1
                elif is_positive and not det:
                    totals[system]["fn"] += 1
                elif not is_positive and det:
                    totals[system]["fp"] += 1
                else:
                    totals[system]["tn"] += 1

            debug_rows.append({
                "condition": cond_name,
                "window_index": i,
                "expected_positive": int(is_positive),
                "option2_detected": int(dets["option2"]),
                "option3_detected": int(dets["option3"]),
                "hybrid_smoothed_detected": int(dets["hybrid_smoothed"]),
                "harmonic_guard_smoothed_detected": int(dets["harmonic_guard_smoothed"]),
                "guard_downgraded": int(guard.downgraded),
                "guard_reason": guard.reason,
                "option2_score": float(o2.score),
                "option3_score": float(o3.score),
                "hybrid_score": base_score,
                "vehicle_risk_score": float(feats.vehicle_risk_score),
                "f0_hz": float(feats.f0_hz),
                "hps_confidence": float(feats.hps_confidence),
                "harmonicity_score": float(feats.harmonicity_score),
                "upper_harmonic_explained_ratio": float(feats.upper_harmonic_explained_ratio),
            })

        for system in SYSTEMS:
            rate = 100.0 * cond[system]["det"] / max(n_windows, 1)
            condition_rows.append({
                "system": system,
                "condition": cond_name,
                "expected": "positive" if is_positive else "negative",
                "detection_rate_percent": rate,
                "recall_percent": rate if is_positive else "",
                "false_alarm_rate_percent": "" if is_positive else rate,
                "mean_score": float(np.mean(cond[system]["scores"])),
                "downgraded_windows": cond[system]["downgraded"],
            })

    summary_rows = []
    for system in SYSTEMS:
        t = totals[system]
        precision, recall, f1 = _metrics(t["tp"], t["fp"], t["tn"], t["fn"])
        summary_rows.append({
            "system": system,
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
        "debug_rows": debug_rows,
        "elapsed_seconds": time.perf_counter() - t0,
    }


def main():
    parser = argparse.ArgumentParser(description="Compare detector iterations with harmonic guard.")
    parser.add_argument("--windows", type=int, default=DEFAULT_N_WINDOWS)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--rule", default="B", choices=["A", "B", "C", "D"])
    parser.add_argument("--smoothing", default="2of3", choices=["none", "2of3", "3of5", "persist_1_5s"])
    parser.add_argument("--no-hybrid-veto", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    import torch

    ensure_dirs()
    device = torch.device("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    print("Comparing detector iterations")
    print("Systems: Option2, Option3, Hybrid, Hybrid+HarmonicGuard")
    print("Safety: no training, no audio suppression, no model overwrite.")
    print(f"Device: {device}")
    print(f"Windows per condition: {args.windows}")

    models = load_hybrid_models(device=device)
    result = compare_iterations(
        models,
        n_windows=args.windows,
        seed=args.seed,
        rule=args.rule,
        smoothing=args.smoothing,
        enable_hybrid_veto=not args.no_hybrid_veto,
    )

    summary_path = RESULTS_DIR / "iteration_comparison_summary.csv"
    condition_path = RESULTS_DIR / "iteration_comparison_conditions.csv"
    debug_path = RESULTS_DIR / "iteration_comparison_window_debug.csv"
    _write_csv(summary_path, result["summary_rows"])
    _write_csv(condition_path, result["condition_rows"])
    _write_csv(debug_path, result["debug_rows"])

    print()
    for row in result["summary_rows"]:
        print(
            f"{row['system']:26s} "
            f"precision={row['precision']*100:6.2f}% "
            f"recall={row['recall']*100:6.2f}% "
            f"f1={row['f1']*100:6.2f}% "
            f"fp={row['fp']} fn={row['fn']}"
        )
    print()
    print(f"Saved summary    -> {summary_path}")
    print(f"Saved conditions -> {condition_path}")
    print(f"Saved debug      -> {debug_path}")
    print(f"Elapsed seconds  -> {result['elapsed_seconds']:.1f}")


if __name__ == "__main__":
    main()
