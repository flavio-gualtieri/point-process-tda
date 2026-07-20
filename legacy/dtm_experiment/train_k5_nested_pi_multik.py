# dtm_experiment/train_k5_nested_pi_multik.py
"""
Runs vihrs_checkpointed and pi_multik_fusion on nested_thomas, both for 500
epochs. This file's scope has shifted from its previous version (vihrs +
pi_multik, both plain/un-fused) now that those two are done and analyzed:

  - Plain vihrs (no checkpoint selection, by design -- see run_vihrs.py) hit
    its best val loss (0.4699) at epoch 67 of the 500-epoch run, then
    overfit for the rest of training with no way to recover that checkpoint
    -- its reported test_loss (1.075) reflects the epoch-500 weights, not
    the epoch-67 ones. vihrs_checkpointed.py isolates that one variable
    (add best-val-checkpoint selection, change nothing else) to get an
    honest test_loss for the epoch-67-quality model instead.
  - pi_multik (k=5/10/15 stacked images) is already trained and analyzed
    (results_k5_nested_pi_multik/pi_multik/) -- NOT rerun here. It's reused
    as one branch of pi_multik_fusion below via
    pi_multik_fusion_model.load_fusion_split, which re-reads the same
    images_dtm_k<k>.pkl files pi_multik_model.load_multik_split always has,
    just also joined against vihrs's L(r)-r population this time.
  - pi_multik_fusion is new: vihrs's L(r)-r+n(x) branch fused with
    pi_multik's stacked-image branch, motivated directly by
    vihrs_checkpointed's result -- if L(r)-r carries information the images
    don't, fusing should do at least as well as the better of the two alone.

Requires the same images_dtm_k5/10/15.pkl (+ adversarial_) as
train_k5_nested_pi_multik.py's original version -- see
pi_multik_model.py's module docstring.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "runners" / "params"))

import run_vihrs
import vihrs_checkpointed
import pi_multik_model
import pi_multik_fusion_model

# Same 5 nested_thomas targets as train_k5_nested.py -- see that file's
# NESTED_THOMAS_LABEL_NAMES comment for why "parent_intensity" names the
# coarse/root layer specifically. Every module below reads run_vihrs.
# LABEL_NAMES at call time (verified by reading each), so retargeting it
# here retargets target selection/model output width/saved label_names
# everywhere downstream -- same monkey-patching convention
# train_k5_nested.py documents for fusion_model.py.
NESTED_THOMAS_LABEL_NAMES = (
    "parent_intensity", "meta_offspring", "meta_cluster_scale", "mean_offspring", "cluster_scale",
)
run_vihrs.LABEL_NAMES = NESTED_THOMAS_LABEL_NAMES

# The k sweep pi_multik_fusion's image branch uses -- must match whatever
# pi_multik was actually trained with, since its results aren't being
# recomputed here.
pi_multik_model.K_VALUES = [5, 10, 15]

# Same 7 seeds as train_k5_nested.py, for direct comparability.
SEEDS = [
    9374,
    9375,
    9376,
    9377,
    9378,
    9379,
    9380,
]

DATA_DIR = ROOT / "data" / "params" / "2d" / "nested_thomas"
RESULTS_DIR = Path(__file__).resolve().parent / "results_k5_nested_pi_multik"
pi_multik_fusion_model.RESULTS_DIR = RESULTS_DIR / "pi_multik_fusion"

N_EPOCHS = 500

_VIHRS_DATA: dict | None = None


def _get_vihrs_data() -> dict | None:
    global _VIHRS_DATA
    if _VIHRS_DATA is None:
        clouds_path = DATA_DIR / "clouds.pkl"
        if not clouds_path.exists():
            return None
        _VIHRS_DATA = run_vihrs.prepare_data(
            clouds_path, DATA_DIR / "adversarial_clouds.pkl",
        )
    return _VIHRS_DATA


def train_vihrs_checkpointed_for_seed(seed: int) -> None:
    output_root = RESULTS_DIR / "vihrs_checkpointed"
    output_dir = output_root / f"seed_{seed}"
    if (output_dir / "results.pt").exists():
        print(f"\nvihrs_checkpointed | seed {seed}: already done, skipping.")
        return

    print(f"\n{'#' * 90}\n### vihrs_checkpointed | seed {seed}\n{'#' * 90}")
    try:
        data = _get_vihrs_data()
        if data is None:
            print(f"! {DATA_DIR / 'clouds.pkl'} missing. Skipping vihrs_checkpointed for seed {seed}.")
            return
        vihrs_checkpointed.run_one_seed(
            seed,
            train_features=data["train_features"],
            adversarial_features=data["adversarial_features"],
            adversarial_path=data["adversarial_path"],
            r_grid=data["r_grid"],
            output_root=output_root,
            n_epochs=N_EPOCHS,
            batch_size=100,  # paper default, matches run_vihrs.py's own CLI default -- unchanged, see that file
            lr=0.001,
            device_pref=None,
        )
    except Exception:
        print(f"\n!! vihrs_checkpointed | seed {seed} FAILED, continuing with next seed.\n{traceback.format_exc()}")


_PI_MULTIK_FUSION_DATA: dict | None = None


def _get_pi_multik_fusion_data() -> dict | None:
    global _PI_MULTIK_FUSION_DATA
    if _PI_MULTIK_FUSION_DATA is None:
        lr_data = _get_vihrs_data()
        if lr_data is None:
            return None
        train_split = pi_multik_fusion_model.load_fusion_split(
            pi_multik_model.K_VALUES, DATA_DIR, lr_data["train_features"], tag="train_test",
        )
        if train_split is None:
            return None
        adv_split = None
        if lr_data["adversarial_features"] is not None:
            adv_split = pi_multik_fusion_model.load_fusion_split(
                pi_multik_model.K_VALUES, DATA_DIR, lr_data["adversarial_features"], tag="adversarial",
            )
        _PI_MULTIK_FUSION_DATA = {"train_split": train_split, "adv_split": adv_split}
    return _PI_MULTIK_FUSION_DATA


def train_pi_multik_fusion_for_seed(seed: int) -> None:
    output_dir = pi_multik_fusion_model.RESULTS_DIR / f"seed_{seed}"
    if (output_dir / "results.pt").exists():
        print(f"\npi_multik_fusion | seed {seed}: already done, skipping.")
        return

    print(f"\n{'#' * 90}\n### pi_multik_fusion | seed {seed}\n{'#' * 90}")
    try:
        data = _get_pi_multik_fusion_data()
        if data is None or data["train_split"] is None:
            print(
                f"! pi_multik_fusion data missing for seed {seed} -- run "
                f"`compute_features.py --process nested_thomas --k <k>` for k in "
                f"{pi_multik_model.K_VALUES} first. Skipping."
            )
            return
        pi_multik_fusion_model.run_one_seed(seed, data["train_split"], data["adv_split"])
    except Exception:
        print(f"\n!! pi_multik_fusion | seed {seed} FAILED, continuing with next seed.\n{traceback.format_exc()}")


def run_seed(seed: int) -> None:
    print(f"\n{'*' * 90}\n*** seed {seed}\n{'*' * 90}")
    train_vihrs_checkpointed_for_seed(seed)
    train_pi_multik_fusion_for_seed(seed)
    print(f"\n--- seed {seed} done. ---")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train vihrs_checkpointed and pi_multik_fusion on the nested_thomas dataset, "
        "both for 500 epochs, for one seed (--seed, for one-GPU-job-per-seed parallelism) or all "
        "SEEDS in sequence (no --seed). Does NOT retrain plain vihrs or pi_multik -- see this "
        "file's module docstring."
    )
    p.add_argument("--seed", type=int, default=None, choices=SEEDS)
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    seeds = [args.seed] if args.seed is not None else SEEDS

    for seed in seeds:
        run_seed(seed)

    print(f"\nAll requested training done. Results under {RESULTS_DIR}")


if __name__ == "__main__":
    main()
