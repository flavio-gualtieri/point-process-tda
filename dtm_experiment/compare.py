# dtm_experiment/compare.py

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "models"))

import evaluate_seeds as es

SEEDS = [
    56245, 189089, 2342344, 9278394, 91873097,
    908308920, 235498734453, 928374129038471,
    974924729845723, 9267492783429472,
]

FEATURES = ["betti_cnn_0", "betti_cnn_1", "pi_0", "pi_1", "new_feature"]

RESULTS_DIR = Path(__file__).resolve().parent / "results"
OUT_DIR = RESULTS_DIR / "summary"


def main() -> None:
    print(f"Loading results for {len(FEATURES)} feature(s) x {len(SEEDS)} seed(s) from {RESULTS_DIR} ...")
    results = es.collect_feature_results(RESULTS_DIR, FEATURES, SEEDS)

    features_present = [f for f in FEATURES if results.get(f)]
    if not features_present:
        raise SystemExit("No results.pt found for any feature/seed -- run train.py first.")

    es.print_feature_configs(results)
    es.print_loss_summary("Test loss across seeds (held-out split)", es.feature_loss_summary(results, "test_loss"))
    es.print_loss_summary("Adversarial loss across seeds", es.feature_loss_summary(results, "adversarial_loss"))
    es.print_paired_comparison(results, "test_loss")
    es.print_paired_comparison(results, "adversarial_loss")

    es.plot_training_curves(results, features_present, OUT_DIR / "training_curves.png")
    es.plot_loss_distribution(results, features_present, OUT_DIR / "loss_distribution.png")


if __name__ == "__main__":
    main()
