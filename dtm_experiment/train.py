# dtm_experiment/train.py

from __future__ import annotations

import sys
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "runners" / "params"))

from cloudforger.nn.experiments import build_experiment
import run_new_feature

SEEDS = [
    9371,
    9372,
    9373
]

DATA_DIR = ROOT / "data" / "params" / "2d" / "thomas"
RESULTS_DIR = Path(__file__).resolve().parent / "results_k10"

N_EPOCHS = 500

# Same betti_cnn/pi settings as the Rips-based fix, except n_epochs: the new
# clouds.pkl is ~7x larger than the dataset that 500-epoch/batch=32 choice
# was tuned against (34,000 vs 4,960 train_test clouds), so each epoch now
# does ~7x more gradient steps. Watch the train/val curves the way we did for
# the Rips-based betti_cnn overfitting fix, and adjust if it's still
# improving at the cutoff or overfitting well before it.
CFG_BASE = {
    "task": "params", "process": "thomas",
    "batch_size": 32, "n_epochs": N_EPOCHS, "lr": 0.001, "embedding_dim": 64,
}

# (method, dataset filename, adversarial filename)
BETTI_PI_METHODS = [
    ("betti_cnn_0", "betti_dtm_k10.pkl", "adversarial_betti_dtm_k10.pkl"),
    ("betti_cnn_1", "betti_dtm_k10.pkl", "adversarial_betti_dtm_k10.pkl"),
    ("pi_0", "images_dtm_k10.pkl", "adversarial_images_dtm_k10.pkl"),
    ("pi_1", "images_dtm_k10.pkl", "adversarial_images_dtm_k10.pkl"),
]

# new_feature's own clouds.pkl load + L(r)-r feature extraction is expensive
# and identical across seeds -- loaded lazily, once, and reused for every
# seed instead of redone per-seed now that seeds are the outer loop.
_NEW_FEATURE_DATA: dict | None = None


def _get_new_feature_data() -> dict | None:
    global _NEW_FEATURE_DATA
    if _NEW_FEATURE_DATA is None:
        clouds_path = DATA_DIR / "clouds.pkl"
        if not clouds_path.exists():
            return None
        _NEW_FEATURE_DATA = run_new_feature.prepare_data(
            clouds_path, DATA_DIR / "adversarial_clouds.pkl",
        )
    return _NEW_FEATURE_DATA


def train_betti_pi_for_seed(seed: int) -> None:
    for method, dataset_file, adv_file in BETTI_PI_METHODS:
        dataset_path = DATA_DIR / dataset_file
        adv_path = DATA_DIR / adv_file
        if not dataset_path.exists():
            print(f"! {dataset_path} missing -- run compute_features.py first. Skipping {method}.")
            continue

        output_dir = RESULTS_DIR / method / f"seed_{seed}"
        if (output_dir / "results.pt").exists():
            print(f"\n{method} | seed {seed}: already done, skipping.")
            continue

        print(f"\n{'=' * 80}\n{method} | seed {seed}\n{'=' * 80}")
        try:
            cfg = {**CFG_BASE, "method": method, "seed": seed}
            experiment = build_experiment(cfg)
            experiment.run(
                dataset_path, output_dir,
                adversarial_path=adv_path if adv_path.exists() else None,
            )
        except Exception:
            # Long unattended run -- one feature failing must not take down
            # the rest of the seed or the seeds after it.
            print(f"\n!! {method} | seed {seed} FAILED, continuing with next feature.\n{traceback.format_exc()}")


def train_new_feature_for_seed(seed: int) -> None:
    output_root = RESULTS_DIR / "new_feature"
    output_dir = output_root / f"seed_{seed}"
    if (output_dir / "results.pt").exists():
        print(f"\nnew_feature | seed {seed}: already done, skipping.")
        return

    print(f"\n{'#' * 90}\n### new_feature | seed {seed}\n{'#' * 90}")
    try:
        data = _get_new_feature_data()
        if data is None:
            print(f"! {DATA_DIR / 'clouds.pkl'} missing. Skipping new_feature for seed {seed}.")
            return
        run_new_feature.run_one_seed(
            seed,
            train_records=data["train_records"],
            train_features=data["train_features"],
            adversarial_features=data["adversarial_features"],
            adversarial_path=data["adversarial_path"],
            r_grid=data["r_grid"],
            output_root=output_root,
            n_epochs=N_EPOCHS,
            batch_size=100,  # paper default, matches run_new_feature.py's own CLI default
            lr=0.001,
            device_pref=None,
            skip_mincontrast=True,  # bonus analysis only; drop this to include it
            mc_cache={},
            mc_rng=np.random.default_rng(run_new_feature.MC_SEED),
        )
    except Exception:
        print(f"\n!! new_feature | seed {seed} FAILED, continuing with next seed.\n{traceback.format_exc()}")


def main() -> None:
    # Seed-outer, feature-inner: every feature finishes (and saves its own
    # results.pt/results.json under its seed_<seed>/ dir) for the current
    # seed before moving to the next seed, so results are comparable across
    # all features as soon as the first seed completes, while later seeds
    # are still training.
    for seed in SEEDS:
        print(f"\n{'*' * 90}\n*** seed {seed}\n{'*' * 90}")
        #train_betti_pi_for_seed(seed)
        train_new_feature_for_seed(seed)
        print(f"\n--- seed {seed} done. Compare progress so far with: python dtm_experiment/compare.py ---")

    print("\nAll training done. Run dtm_experiment/compare.py next.")


if __name__ == "__main__":
    main()
