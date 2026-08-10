# dtm_experiment/compare.py
# sbatch dtm_experiment/run_compare.sh

from __future__ import annotations

import contextlib
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "models"))

import evaluate_seeds as es

# Must match dtm_experiment/train_k5.py::SEEDS -- collect_feature_results
# skips/warns per-seed if any are missing, so this is safe to keep even
# while a seed's array task (see run_train_gpu_k5.sh) is still running.
SEEDS = [
    9371,
"""     9372,
    9373,
    9374,
    9375,
    9376,
    9377,
    9378,
    9379,
    9380, """
]

# Full run-order sweep from train_k5.py::run_seed -- vihrs, the 3 branch-
# ablation arms (fusion_pi/fusion_betti/fusion_scalars, point 1 of the
# PH-value investigation), the original fusion/ph_combined/pi_01/
# betti_cnn_01 comparison. Trim this down (e.g. back to ["vihrs", "fusion"])
# for a smaller, focused report -- collect_feature_results skips/warns on
# any feature with no results.pt yet, so it's safe to run this against a
# partially-finished sweep.
FEATURES = [
#    "vihrs",
    "fusion_pi",
    "fusion_betti",
    "fusion_scalars",
    "fusion",
#    "ph_combined",
#    "pi_01",
#    "betti_cnn_01",
    ]

k = 5

RESULTS_DIR = Path(__file__).resolve().parent / f"results_k{k}"
OUT_DIR = RESULTS_DIR / "summary_fusion"


class _Tee(io.TextIOBase):
    """Duplicates writes to every stream given (e.g. real stdout + a report
    file), so the console output and the saved report are guaranteed identical."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, s: str) -> int:
        for stream in self.streams:
            stream.write(s)
        return len(s)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def _build_json_summary(
    results: dict, features_present: list[str],
    test_summary: dict, adv_summary: dict, train_summary: dict,
    test_per_target_summary: dict, adv_per_target_summary: dict,
) -> dict:
    return {
        "k": k,
        "results_dir": str(RESULTS_DIR),
        "seeds_requested": SEEDS,
        "features": {
            feature: {
                "n_seeds": len(results.get(feature, {})),
                "test_loss": test_summary.get(feature, {}),
                "adversarial_loss": adv_summary.get(feature, {}),
                "test_loss_per_target": test_per_target_summary.get(feature, {}),
                "adversarial_loss_per_target": adv_per_target_summary.get(feature, {}),
                "training": train_summary.get(feature, {}),
            }
            for feature in features_present
        },
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUT_DIR / "report.txt"
    json_path = OUT_DIR / "report.json"

    with open(report_path, "w") as report_file:
        with contextlib.redirect_stdout(_Tee(sys.stdout, report_file)):
            print(f"Loading results for {len(FEATURES)} feature(s) x {len(SEEDS)} seed(s) from {RESULTS_DIR} ...")
            results = es.collect_feature_results(RESULTS_DIR, FEATURES, SEEDS)

            features_present = [f for f in FEATURES if results.get(f)]
            if not features_present:
                raise SystemExit("No results.pt found for any feature/seed -- run train.py first.")

            es.print_feature_configs(results)

            train_summary = es.feature_training_summary(results)
            es.print_training_summary("Training dynamics across seeds (train/val loss, best epoch)", train_summary)
            es.print_overfit_flags(train_summary)

            test_summary = es.feature_loss_summary(results, "test_loss")
            adv_summary = es.feature_loss_summary(results, "adversarial_loss")
            es.print_loss_summary("Test loss across seeds (held-out split)", test_summary)
            es.print_loss_summary("Adversarial loss across seeds", adv_summary)
            es.print_paired_comparison(results, "test_loss")
            es.print_paired_comparison(results, "adversarial_loss")

            test_per_target_summary = es.feature_per_target_loss_summary(results, "test_loss_per_target")
            adv_per_target_summary = es.feature_per_target_loss_summary(results, "adversarial_loss_per_target")
            es.print_per_target_loss_summary(
                "Per-target test loss across seeds (which parameter drives each feature's loss)",
                test_per_target_summary,
            )
            es.print_per_target_loss_summary(
                "Per-target adversarial loss across seeds", adv_per_target_summary,
            )

            es.plot_training_curves(results, features_present, OUT_DIR / "training_curves.png")
            es.plot_loss_distribution(results, features_present, OUT_DIR / "loss_distribution.png")

    with open(json_path, "w") as f:
        json.dump(
            _build_json_summary(
                results, features_present, test_summary, adv_summary, train_summary,
                test_per_target_summary, adv_per_target_summary,
            ),
            f, indent=2,
        )

    print(f"\nSaved text report -> {report_path}")
    print(f"Saved JSON summary -> {json_path}")


if __name__ == "__main__":
    main()
