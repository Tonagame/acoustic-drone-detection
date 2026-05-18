from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = ROOT / "results" / "phase2_harmonic_fusion"
DEFAULT_SUMMARY = DEFAULT_RESULTS / "benchmark_phase2_harmonic_fusion_v1_thr_0p85_backbone_0p6_summary.csv"
DEFAULT_SWEEP = DEFAULT_RESULTS / "benchmark_phase2_harmonic_fusion_v1_threshold_sweep.csv"
DEFAULT_DEBUG = DEFAULT_RESULTS / "benchmark_phase2_harmonic_fusion_v1_thr_0p85_backbone_0p6_debug.csv"
DEFAULT_PLOTS = DEFAULT_RESULTS / "plots"


CONDITION_LABELS = {
    "drone_alone": "Drone\nalone",
    "drone_plus_fsd_snr_-20db": "-20 dB",
    "drone_plus_fsd_snr_-15db": "-15 dB",
    "drone_plus_fsd_snr_-10db": "-10 dB",
    "drone_plus_fsd_snr_-5db": "-5 dB",
    "drone_plus_fsd_snr_+0db": "0 dB",
    "drone_plus_fsd_snr_+5db": "+5 dB",
    "drone_plus_fsd_snr_+10db": "+10 dB",
    "fsd_alone": "FSD\nnoise",
    "nodrone_alone": "DADS\nno-drone",
}

POSITIVE_ORDER = [
    "drone_alone",
    "drone_plus_fsd_snr_-20db",
    "drone_plus_fsd_snr_-15db",
    "drone_plus_fsd_snr_-10db",
    "drone_plus_fsd_snr_-5db",
    "drone_plus_fsd_snr_+0db",
    "drone_plus_fsd_snr_+5db",
    "drone_plus_fsd_snr_+10db",
]

NEGATIVE_ORDER = ["fsd_alone", "nodrone_alone"]

COLORS = {
    "phase2_harmonic": "#1f77b4",
    "v5c_backbone": "#ff7f0e",
}


def _style():
    plt.rcParams.update({
        "figure.dpi": 140,
        "savefig.dpi": 180,
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.labelsize": 10,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "legend.frameon": False,
    })


def _bar_pair(ax, summary: pd.DataFrame, conditions: list[str], title: str, ylabel: str):
    x = np.arange(len(conditions))
    width = 0.38
    for offset, system in [(-width / 2, "phase2_harmonic"), (width / 2, "v5c_backbone")]:
        vals = []
        for cond in conditions:
            row = summary[(summary["system"] == system) & (summary["condition"] == cond)]
            vals.append(float(row["detection_rate_percent"].iloc[0]) if len(row) else 0.0)
        ax.bar(x + offset, vals, width, label=system.replace("_", " "), color=COLORS[system])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels([CONDITION_LABELS[c] for c in conditions])
    ax.set_ylim(0, 105)
    ax.legend()


def plot_condition_bars(summary_path: Path, out_dir: Path) -> list[Path]:
    summary = pd.read_csv(summary_path)
    _style()
    out = []

    fig, ax = plt.subplots(figsize=(11, 5))
    _bar_pair(ax, summary, POSITIVE_ORDER, "Drone Detection Recall By Condition", "Detection rate (%)")
    fig.tight_layout()
    path = out_dir / "phase2_vs_v5c_positive_recall.png"
    fig.savefig(path)
    plt.close(fig)
    out.append(path)

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    _bar_pair(ax, summary, NEGATIVE_ORDER, "False Alarm Rate On Negative Audio", "False alarm rate (%)")
    fig.tight_layout()
    path = out_dir / "phase2_vs_v5c_false_alarms.png"
    fig.savefig(path)
    plt.close(fig)
    out.append(path)
    return out


def plot_threshold_sweep(sweep_path: Path, out_dir: Path) -> list[Path]:
    sweep = pd.read_csv(sweep_path)
    _style()
    out = []

    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    for system in ["phase2_harmonic", "v5c_backbone"]:
        s = sweep[sweep["system"] == system].sort_values("thr")
        ax.plot(s["far"], s["pos_recall"], marker="o", label=system.replace("_", " "), color=COLORS[system])
        for _, row in s.iterrows():
            if row["thr"] in [0.60, 0.85]:
                ax.annotate(f"{row['thr']:.2f}", (row["far"], row["pos_recall"]), xytext=(5, 4), textcoords="offset points")
    ax.set_title("Recall vs False Alarm Tradeoff")
    ax.set_xlabel("Negative false alarm rate (%)")
    ax.set_ylabel("Positive recall (%)")
    ax.set_xlim(left=0)
    ax.set_ylim(0, 100)
    ax.legend()
    fig.tight_layout()
    path = out_dir / "phase2_threshold_tradeoff.png"
    fig.savefig(path)
    plt.close(fig)
    out.append(path)

    fig, ax = plt.subplots(figsize=(8, 5))
    s = sweep[sweep["system"] == "phase2_harmonic"].sort_values("thr")
    ax.plot(s["thr"], s["drone_plus_fsd_snr_-10db"], marker="o", label="-10 dB recall")
    ax.plot(s["thr"], s["drone_plus_fsd_snr_-5db"], marker="o", label="-5 dB recall")
    ax.plot(s["thr"], s["fsd_alone"], marker="o", label="FSD false alarm")
    ax.plot(s["thr"], s["nodrone_alone"], marker="o", label="DADS no-drone false alarm")
    ax.axvline(0.85, color="#444444", linestyle="--", linewidth=1, label="chosen 0.85")
    ax.set_title("Phase 2 Threshold Sweep")
    ax.set_xlabel("Phase 2 score threshold")
    ax.set_ylabel("Rate (%)")
    ax.set_ylim(0, 105)
    ax.legend()
    fig.tight_layout()
    path = out_dir / "phase2_threshold_sweep_detail.png"
    fig.savefig(path)
    plt.close(fig)
    out.append(path)
    return out


def plot_score_distributions(debug_path: Path, out_dir: Path) -> list[Path]:
    debug = pd.read_csv(debug_path)
    _style()
    out = []
    fig, ax = plt.subplots(figsize=(9, 5))
    phase2 = debug[debug["system"] == "phase2_harmonic"].copy()
    groups = [
        ("drone_alone", "drone alone"),
        ("drone_plus_fsd_snr_-10db", "-10 dB mix"),
        ("drone_plus_fsd_snr_-5db", "-5 dB mix"),
        ("fsd_alone", "FSD noise"),
        ("nodrone_alone", "DADS no-drone"),
    ]
    data = [phase2[phase2["condition"] == cond]["score"].values for cond, _ in groups]
    ax.boxplot(data, tick_labels=[label for _, label in groups], showfliers=False)
    ax.axhline(0.85, color="#444444", linestyle="--", linewidth=1, label="chosen threshold 0.85")
    ax.set_title("Phase 2 Score Distribution")
    ax.set_ylabel("Phase 2 drone score")
    ax.set_ylim(0, 1.02)
    ax.legend()
    fig.tight_layout()
    path = out_dir / "phase2_score_distribution.png"
    fig.savefig(path)
    plt.close(fig)
    out.append(path)
    return out


def parse_args():
    ap = argparse.ArgumentParser(description="Create Phase 2 benchmark graph PNGs")
    ap.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    ap.add_argument("--sweep", type=Path, default=DEFAULT_SWEEP)
    ap.add_argument("--debug", type=Path, default=DEFAULT_DEBUG)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_PLOTS)
    return ap.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    outputs.extend(plot_condition_bars(args.summary, args.out_dir))
    outputs.extend(plot_threshold_sweep(args.sweep, args.out_dir))
    outputs.extend(plot_score_distributions(args.debug, args.out_dir))
    print("Saved plots:")
    for path in outputs:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
