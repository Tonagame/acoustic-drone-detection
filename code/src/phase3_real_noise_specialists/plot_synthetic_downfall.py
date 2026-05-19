from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "results" / "phase3_real_noise_specialists" / "plots"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    hybrid_path = ROOT / "results" / "hybrid_option2_option3" / "hybrid_comparison_table.csv"
    real_path = ROOT / "results" / "fsd50k_hard_negative_eval" / "fsd50k_mixed_recall_summary.csv"
    neg_path = ROOT / "results" / "fsd50k_hard_negative_eval" / "fsd50k_benchmark_summary.csv"

    hybrid = pd.read_csv(hybrid_path)
    real = pd.read_csv(real_path)
    neg = pd.read_csv(neg_path)

    old_hybrid = hybrid[hybrid["system"] == "hybrid_smoothed"].iloc[0]
    old_real = real[(real["system"] == "hybrid_smoothed") & (real["scenario"] == "ALL")].iloc[0]
    old_clean = real[(real["system"] == "hybrid_smoothed") & (real["scenario"] == "drone alone")].iloc[0]
    old_far = neg[neg["system"] == "hybrid_smoothed"].iloc[0]

    labels = [
        "Original hybrid\nsynthetic benchmark\nrecall",
        "Original hybrid\nclean drone on\nFSD benchmark",
        "Original hybrid\ndrone + real\nFSD noise",
        "Original hybrid\nreal FSD noise\nfalse alarm",
    ]
    values = [
        float(old_hybrid["recall"]) * 100.0,
        float(old_clean["recall_percent"]),
        float(old_real["recall_percent"]),
        float(old_far["false_alarm_rate_percent"]),
    ]
    colors = ["#2ca02c", "#1f77b4", "#d62728", "#ff7f0e"]

    plt.rcParams.update({
        "figure.dpi": 140,
        "savefig.dpi": 180,
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.grid": True,
        "grid.alpha": 0.25,
    })
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    bars = ax.bar(labels, values, color=colors)
    ax.set_title("The Synthetic-Noise Downfall")
    ax.set_ylabel("Rate (%)")
    ax.set_ylim(0, 105)
    ax.axhline(75, color="#444444", linestyle="--", linewidth=1, label="75% recall reference")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 2, f"{value:.1f}%", ha="center", fontsize=10)
    ax.legend(frameon=False)
    fig.tight_layout()
    out = OUT_DIR / "synthetic_downfall_old_hybrid.png"
    fig.savefig(out)
    plt.close(fig)

    pd.DataFrame({"metric": labels, "value_percent": values}).to_csv(
        OUT_DIR / "synthetic_downfall_old_hybrid_source.csv",
        index=False,
    )
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
