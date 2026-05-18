"""
Run the experimental Option2 + Option3 hybrid evaluation.

Examples:
  python src/hybrid_option2_option3/run_hybrid_test.py
  python src/hybrid_option2_option3/run_hybrid_test.py --windows 120 --rule B --smoothing 2of3
"""

import argparse
from pathlib import Path

from evaluate_hybrid import EvalConfig, evaluate_and_save


def _print_summary(eval_result, verdict, best):
    print("\nPer-system summary")
    print("------------------")
    for row in eval_result["summary_rows"]:
        print(
            f"{row['system']:<20s} "
            f"precision={row['precision']*100:6.2f}% "
            f"recall={row['recall']*100:6.2f}% "
            f"F1={row['f1']*100:6.2f}% "
            f"TP={row['tp']} FP={row['fp']} TN={row['tn']} FN={row['fn']}"
        )

    print("\nCondition table")
    print("---------------")
    for row in eval_result["condition_rows"]:
        if row["system"] in ("option2", "option3", "hybrid_smoothed"):
            metric = "recall" if row["expected"] == "positive" else "FA"
            print(
                f"{row['system']:<20s} {row['condition']:<18s} "
                f"{metric}={row['detection_rate_percent']:6.2f}% "
                f"mean={row['mean_drone_probability']:.3f}"
            )

    print("\nVerdict")
    print("-------")
    print(verdict)
    print(
        f"Best by F1: {best['system']} "
        f"(precision={best['precision']*100:.2f}%, "
        f"recall={best['recall']*100:.2f}%, F1={best['f1']*100:.2f}%)"
    )
    print("Recommended starting point: Rule B, Option3 weighted_average, smoothing=2of3, veto=on")


def main():
    parser = argparse.ArgumentParser(description="Evaluate hybrid Option2+Option3 detector")
    parser.add_argument("--windows", type=int, default=600, help="Windows per condition")
    parser.add_argument("--rule", choices=["A", "B", "C", "D"], default="B")
    parser.add_argument(
        "--option3-score",
        choices=["weighted_average", "filtered_max", "voting"],
        default="weighted_average",
    )
    parser.add_argument(
        "--smoothing",
        choices=["none", "2of3", "3of5", "persist_1_5s"],
        default="2of3",
    )
    parser.add_argument("--no-veto", action="store_true")
    parser.add_argument("--option2-model", type=Path, default=None)
    parser.add_argument("--option3-model", type=Path, default=None)
    args = parser.parse_args()

    cfg = EvalConfig(
        n_windows=args.windows,
        hybrid_rule=args.rule,
        option3_score_method=args.option3_score,
        smoothing_mode=args.smoothing,
        enable_veto=not args.no_veto,
    )
    eval_result, sweep_rows, verdict, best = evaluate_and_save(
        cfg,
        option2_path=args.option2_model,
        option3_path=args.option3_model,
    )

    print(f"\nEvaluation finished in {eval_result['elapsed_seconds']:.1f}s")
    print("Wrote:")
    print("  results/hybrid_option2_option3/hybrid_metrics.csv")
    print("  results/hybrid_option2_option3/hybrid_threshold_sweep.csv")
    print("  results/hybrid_option2_option3/hybrid_comparison_table.csv")
    print("  results/hybrid_option2_option3/confusion_matrices/*.csv")
    print("  results/hybrid_option2_option3/logs/hybrid_window_log.csv")
    print(f"Threshold sweep rows: {len(sweep_rows)}")
    _print_summary(eval_result, verdict, best)


if __name__ == "__main__":
    main()
