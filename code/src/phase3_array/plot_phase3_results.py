"""Plotting helpers for Phase 3 array evaluation."""

import csv
from pathlib import Path

import numpy as np


def _read_csv(path: Path):
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _safe_float(v, default=np.nan):
    try:
        if v == "":
            return default
        return float(v)
    except Exception:
        return default


def plot_array_evaluation(per_window_csv, direction_scores_csv, plots_dir):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    per_rows = _read_csv(Path(per_window_csv))
    dir_rows = _read_csv(Path(direction_scores_csv))
    if not per_rows:
        return
    base = Path(per_window_csv).stem.replace("_per_window", "")

    t = np.array([_safe_float(r["time_start"], 0.0) for r in per_rows])
    scores = np.array([_safe_float(r["best_score"], 0.0) for r in per_rows])
    raw = np.array([_safe_float(r["raw_detected"], 0.0) for r in per_rows])
    smooth = np.array([_safe_float(r["smoothed_detected"], 0.0) for r in per_rows])
    az = np.array([_safe_float(r["best_az"], 0.0) for r in per_rows])
    el = np.array([_safe_float(r["best_el"], 0.0) for r in per_rows])

    fig, ax1 = plt.subplots(figsize=(11, 4))
    ax1.plot(t, scores, label="Best hybrid score", linewidth=1.4)
    ax1.set_ylim(0, 1)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Score")
    ax1.grid(True, alpha=0.25)
    ax2 = ax1.twinx()
    ax2.step(t, raw, where="post", label="Raw detect", alpha=0.35)
    ax2.step(t, smooth, where="post", label="Smoothed detect", color="black", alpha=0.65)
    ax2.set_ylim(-0.05, 1.05)
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper right")
    fig.suptitle(f"{base}: detection timeline")
    fig.tight_layout()
    fig.savefig(plots_dir / f"{base}_timeline.png", dpi=120)
    plt.close(fig)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
    ax1.plot(t, az, linewidth=1.2)
    ax1.set_ylabel("Azimuth (deg)")
    ax1.grid(True, alpha=0.25)
    ax2.plot(t, el, linewidth=1.2, color="tab:orange")
    ax2.set_ylabel("Elevation (deg)")
    ax2.set_xlabel("Time (s)")
    ax2.grid(True, alpha=0.25)
    fig.suptitle(f"{base}: best direction timeline")
    fig.tight_layout()
    fig.savefig(plots_dir / f"{base}_direction_timeline.png", dpi=120)
    plt.close(fig)

    if dir_rows:
        vals = {}
        for row in dir_rows:
            key = (_safe_float(row["az"], 0.0), _safe_float(row["el"], 0.0))
            hs = _safe_float(row.get("hybrid_score", ""), np.nan)
            if np.isfinite(hs):
                vals.setdefault(key, []).append(hs)
            else:
                vals.setdefault(key, []).append(_safe_float(row["beam_energy"], 0.0))
        xs, ys, cs = [], [], []
        for (azv, elv), arr in vals.items():
            xs.append(azv)
            ys.append(elv)
            cs.append(float(np.mean(arr)))
        fig, ax = plt.subplots(figsize=(7, 5))
        sc = ax.scatter(xs, ys, c=cs, cmap="viridis", s=130, edgecolors="black")
        ax.set_xlabel("Azimuth (deg)")
        ax.set_ylabel("Elevation (deg)")
        ax.set_title(f"{base}: direction average score")
        fig.colorbar(sc, ax=ax, label="Average score")
        fig.tight_layout()
        fig.savefig(plots_dir / f"{base}_direction_scores.png", dpi=120)
        plt.close(fig)


def plot_comparison(comparison_csv, plots_dir):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    rows = _read_csv(Path(comparison_csv))
    if not rows:
        return
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    base = Path(comparison_csv).stem.replace("_array_vs_single_channel", "")

    t = np.array([_safe_float(r["time_start"], 0.0) for r in rows])
    single = np.array([_safe_float(r["single_channel_score"], 0.0) for r in rows])
    beam = np.array([_safe_float(r["beamformed_score"], 0.0) for r in rows])
    single_det = np.array([_safe_float(r["single_channel_detected"], 0.0) for r in rows])
    beam_det = np.array([_safe_float(r["beamformed_smoothed_detected"], 0.0) for r in rows])

    fig, ax1 = plt.subplots(figsize=(11, 4))
    ax1.plot(t, single, label="Single-channel score", linewidth=1.3)
    ax1.plot(t, beam, label="Beamformed score", linewidth=1.3)
    ax1.set_ylim(0, 1)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Score")
    ax1.grid(True, alpha=0.25)
    ax2 = ax1.twinx()
    ax2.step(t, single_det, where="post", label="Single detect", alpha=0.35)
    ax2.step(t, beam_det, where="post", label="Beam detect", color="black", alpha=0.65)
    ax2.set_ylim(-0.05, 1.05)
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper right")
    fig.suptitle(f"{base}: array vs single-channel")
    fig.tight_layout()
    fig.savefig(plots_dir / f"{base}_array_vs_single_channel.png", dpi=120)
    plt.close(fig)
