from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
PHASE3_RESULTS = ROOT / "results" / "phase3_real_noise_specialists"
PHASE2B_RESULTS = ROOT / "results" / "phase2b_pitch_guard"
OUT_DIR = PHASE3_RESULTS / "plots"


BASE_COMPARISON = PHASE3_RESULTS / "iteration_comparison_scores.csv"
PHASE2B_SWEEP = PHASE2B_RESULTS / "benchmark_phase2b_learned_pitch_guard_v1_medium_threshold_sweep.csv"


DISPLAY_NAMES = {
    "old_realFSD_option2": "Old generalist CNN\n(real FSD)",
    "old_realFSD_option3": "Old 5-specialist CNN\n(real FSD)",
    "old_realFSD_hybrid_smoothed": "Old hybrid rules\n(real FSD)",
    "v5b_recall_legacy": "Real-noise generalist\nsensitive",
    "v5c_score_thr_0.60": "Real-noise generalist\nbalanced",
    "phase2_harmonic_thr_0.85": "Harmonic DSP + ML\nfusion",
    "phase3_specialists_only_default": "5-specialist\nreal-noise ensemble",
    "phase3_phase3_hybrid_default": "5-specialist +\nharmonic guard",
    "phase3_hybrid_score_thr_0.55": "5-specialist +\nharmonic ML guard",
    "phase3_hybrid_score_thr_0.50": "5-specialist +\nharmonic ML guard\nsensitive",
    "phase2b_pitch_guard_thr_0.55": "5-specialist +\npitch-harmonic ML",
    "phase2b_pitch_guard_thr_0.45": "5-specialist +\npitch-harmonic ML\nsensitive",
    "old_hybrid_synthetic_hybrid_smoothed": "Old hybrid rules\nsynthetic test",
}


REAL_ORDER = [
    "old_realFSD_option2",
    "old_realFSD_option3",
    "old_realFSD_hybrid_smoothed",
    "v5b_recall_legacy",
    "v5c_score_thr_0.60",
    "phase2_harmonic_thr_0.85",
    "phase3_specialists_only_default",
    "phase3_phase3_hybrid_default",
    "phase3_hybrid_score_thr_0.55",
    "phase3_hybrid_score_thr_0.50",
    "phase2b_pitch_guard_thr_0.55",
    "phase2b_pitch_guard_thr_0.45",
]


SNR_ORDER = ["clean_drone", "mix_m20", "mix_m15", "mix_m10", "mix_m5", "mix_0", "mix_p5", "mix_p10"]
SNR_LABELS = ["clean", "-20", "-15", "-10", "-5", "0", "+5", "+10"]


def _style():
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 180,
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "legend.frameon": False,
        }
    )


def _load_comparison() -> pd.DataFrame:
    if not BASE_COMPARISON.exists():
        raise FileNotFoundError(f"Missing comparison table: {BASE_COMPARISON}")
    df = pd.read_csv(BASE_COMPARISON)

    if PHASE2B_SWEEP.exists():
        sweep = pd.read_csv(PHASE2B_SWEEP)
        extra = []
        for thr in (0.55, 0.45):
            row = sweep[np.isclose(sweep["thr"], thr)]
            if row.empty:
                continue
            r = row.iloc[0]
            fsd_far = float(r.get("fsd", np.nan))
            nodrone_far = float(r.get("nodrone", np.nan))
            avg_far = np.nanmean([fsd_far, nodrone_far])
            extra.append(
                {
                    "iteration": f"phase2b_pitch_guard_thr_{thr:.2f}",
                    "clean_drone": float(r["clean"]),
                    "mix_m20": float(r["m20"]),
                    "mix_m15": float(r["m15"]),
                    "mix_m10": float(r["m10"]),
                    "mix_m5": float(r["m5"]),
                    "mix_0": np.nan,
                    "mix_p5": np.nan,
                    "mix_p10": np.nan,
                    "mixed_recall": float(r["recall"]),
                    "fsd_far": fsd_far,
                    "nodrone_far": nodrone_far,
                    "avg_far": avg_far,
                    "score_index": float(r["recall"]) - 2.0 * avg_far,
                    "notes": "Learned pitch guard threshold sweep",
                }
            )
        if extra:
            df = pd.concat([df, pd.DataFrame(extra)], ignore_index=True)

    df["display"] = df["iteration"].map(DISPLAY_NAMES).fillna(df["iteration"])
    return df


def _real_subset(df: pd.DataFrame) -> pd.DataFrame:
    keep = [x for x in REAL_ORDER if x in set(df["iteration"])]
    sub = df[df["iteration"].isin(keep)].copy()
    sub["order"] = sub["iteration"].map({name: i for i, name in enumerate(keep)})
    return sub.sort_values("order")


def plot_recall_far(df: pd.DataFrame) -> Path:
    sub = _real_subset(df)
    _style()
    x = np.arange(len(sub))
    width = 0.38
    fig, ax = plt.subplots(figsize=(14, 5.8))
    ax.bar(x - width / 2, sub["mixed_recall"], width, label="mixed drone recall", color="#1f77b4")
    ax.bar(x + width / 2, sub["avg_far"], width, label="average false alarm", color="#d62728")
    ax.axhline(75, color="#2ca02c", linestyle="--", linewidth=1.2, label="75% recall gate")
    ax.axhline(5, color="#8c564b", linestyle=":", linewidth=1.2, label="5% FAR gate")
    ax.set_title("Real-Noise Approach Comparison: Recall vs False Alarm")
    ax.set_ylabel("Rate (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(sub["display"], rotation=35, ha="right")
    ax.set_ylim(0, 105)
    ax.legend(ncol=4, loc="upper left")
    fig.tight_layout()
    path = OUT_DIR / "all_iterations_recall_vs_far.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_score_index(df: pd.DataFrame) -> Path:
    sub = _real_subset(df)
    sub = sub.sort_values("score_index", ascending=True)
    colors = ["#2ca02c" if "phase2b" in it else "#1f77b4" for it in sub["iteration"]]
    _style()
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(sub["display"], sub["score_index"], color=colors)
    ax.set_title("Real-Noise Approach Score Index")
    ax.set_xlabel("mixed recall - 2 x average FAR")
    fig.tight_layout()
    path = OUT_DIR / "all_iterations_score_index.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_snr_curves(df: pd.DataFrame) -> Path:
    selected = [
        "v5b_recall_legacy",
        "v5c_score_thr_0.60",
        "phase2_harmonic_thr_0.85",
        "phase3_hybrid_score_thr_0.55",
        "phase2b_pitch_guard_thr_0.55",
        "phase2b_pitch_guard_thr_0.45",
    ]
    sub = df[df["iteration"].isin(selected)].copy()
    _style()
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    for _, row in sub.iterrows():
        vals = [row.get(col, np.nan) for col in SNR_ORDER]
        ax.plot(SNR_LABELS, vals, marker="o", linewidth=2, label=DISPLAY_NAMES.get(row["iteration"], row["iteration"]))
    ax.axhline(92, color="#2ca02c", linestyle="--", linewidth=1, label="clean target reference")
    ax.axhline(75, color="#8c564b", linestyle=":", linewidth=1, label="mixed recall gate")
    ax.set_title("Recall By SNR Across Main Approaches")
    ax.set_xlabel("Drone + FSD50K SNR (dB); clean = no added FSD noise")
    ax.set_ylabel("Recall (%)")
    ax.set_ylim(0, 105)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    path = OUT_DIR / "all_iterations_snr_recall_curves.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_synthetic_warning(df: pd.DataFrame) -> Path:
    names = [
        "old_hybrid_synthetic_hybrid_smoothed",
        "old_realFSD_hybrid_smoothed",
        "phase3_hybrid_score_thr_0.55",
        "phase2b_pitch_guard_thr_0.45",
    ]
    sub = df[df["iteration"].isin(names)].copy()
    sub["order"] = sub["iteration"].map({name: i for i, name in enumerate(names)})
    sub = sub.sort_values("order")
    _style()
    fig, ax = plt.subplots(figsize=(8.5, 5))
    colors = ["#7f7f7f", "#d62728", "#1f77b4", "#2ca02c"]
    ax.bar(sub["display"], sub["mixed_recall"], color=colors[: len(sub)])
    ax.set_title("Synthetic Success vs Real-Noise Progress")
    ax.set_ylabel("Mixed/benchmark recall (%)")
    ax.set_ylim(0, 105)
    ax.tick_params(axis="x", rotation=25)
    for i, (_, row) in enumerate(sub.iterrows()):
        ax.text(i, float(row["mixed_recall"]) + 2, f"{float(row['mixed_recall']):.1f}%", ha="center", fontsize=9)
    fig.tight_layout()
    path = OUT_DIR / "synthetic_vs_real_progress.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = _load_comparison()
    merged = OUT_DIR / "all_iterations_plot_source.csv"
    df.to_csv(merged, index=False)
    outputs = [
        plot_recall_far(df),
        plot_score_index(df),
        plot_snr_curves(df),
        plot_synthetic_warning(df),
    ]
    print("Saved graph source ->", merged)
    print("Saved plots:")
    for path in outputs:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
