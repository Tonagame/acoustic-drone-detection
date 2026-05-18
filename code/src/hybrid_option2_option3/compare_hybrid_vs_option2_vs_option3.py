"""
Print a compact comparison from hybrid evaluation CSV outputs.

Run run_hybrid_test.py first, or use --run to evaluate before printing.
"""

import argparse
import csv

from config_hybrid import RESULTS_DIR
from evaluate_hybrid import EvalConfig, evaluate_and_save


def read_csv(path):
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def print_comparison():
    summary_path = RESULTS_DIR / "hybrid_comparison_table.csv"
    metrics_path = RESULTS_DIR / "hybrid_metrics.csv"
    if not summary_path.exists() or not metrics_path.exists():
        raise FileNotFoundError("Run run_hybrid_test.py first, or pass --run.")

    summary = read_csv(summary_path)
    metrics = read_csv(metrics_path)

    print("System comparison")
    print("-----------------")
    print(f"{'System':<22s} {'Precision':>10s} {'Recall':>10s} {'F1':>10s} {'FP':>7s} {'FN':>7s}")
    for row in summary:
        print(
            f"{row['system']:<22s} "
            f"{float(row['precision'])*100:9.2f}% "
            f"{float(row['recall'])*100:9.2f}% "
            f"{float(row['f1'])*100:9.2f}% "
            f"{row['fp']:>7s} {row['fn']:>7s}"
        )

    print("\nKey conditions")
    print("--------------")
    key = {
        "drone alone",
        "drone+tank 0dB",
        "drone+engine",
        "drone+crowd",
        "tank alone",
        "engine alone",
        "crowd alone",
        "pure noise",
    }
    systems = ["option2", "option3", "hybrid_no_smoothing", "hybrid_smoothed"]
    by_cond = {(r["condition"], r["system"]): r for r in metrics}
    print(f"{'Condition':<18s} " + " ".join(f"{s[:12]:>13s}" for s in systems))
    for cond in metrics:
        cname = cond["condition"]
        if cname not in key:
            continue
        vals = []
        for sys_name in systems:
            row = by_cond.get((cname, sys_name))
            vals.append(f"{float(row['detection_rate_percent']):12.2f}%" if row else "        n/a  ")
        print(f"{cname:<18s} " + " ".join(vals))
        key.remove(cname)
        if not key:
            break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", help="Run evaluation first")
    parser.add_argument("--windows", type=int, default=600)
    parser.add_argument("--rule", choices=["A", "B", "C", "D"], default="B")
    parser.add_argument("--smoothing", choices=["none", "2of3", "3of5", "persist_1_5s"], default="2of3")
    args = parser.parse_args()

    if args.run:
        cfg = EvalConfig(n_windows=args.windows, hybrid_rule=args.rule, smoothing_mode=args.smoothing)
        evaluate_and_save(cfg)
    print_comparison()


if __name__ == "__main__":
    main()
