# dtm_experiment/train_k5.py

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "runners" / "params"))

from cloudforger.nn.experiments import build_experiment
import run_vihrs
import fusion_model

# 10 seeds, trained in parallel (one GPU job per seed via --seed / the SLURM
# array in run_train_gpu_k5.sh), each job iterating through every feature
# below for its own seed.
SEEDS = [
    9371,
    9372,
    9373,
    9374,
    9375,
    9376,
    9377,
    9378,
    9379,
    9380,
]

DATA_DIR = ROOT / "data" / "params" / "2d" / "thomas"
RESULTS_DIR = Path(__file__).resolve().parent / "results_k5"

N_EPOCHS = 500

# Vihrs (2022)'s own paper-faithful epoch count (Appendix A.2: chosen once
# via a held-out-curve check, no early stopping) -- deliberately NOT the same
# budget as the TDA methods below, since run_vihrs.py is a standalone
# replication of the paper's own recipe, not tuned for parity with them.
VIHRS_N_EPOCHS = 20

# Same betti_cnn/pi settings as the Rips-based fix, except n_epochs: the new
# clouds.pkl is ~7x larger than the dataset that 500-epoch/batch=32 choice
# was tuned against (34,000 vs 4,960 train_test clouds), so each epoch now
# does ~7x more gradient steps. Watch the train/val curves the way we did for
# the Rips-based betti_cnn overfitting fix, and adjust if it's still
# improving at the cutoff or overfitting well before it.
CFG_BASE = {
    "task": "params", "process": "thomas",
    "batch_size": 32, "n_epochs": N_EPOCHS, "lr": 0.001, "embedding_dim": 64,
    # The new clouds.pkl's "params" dict carries "edge_buffer" alongside the
    # 3 real unknown parameters (NeymanScottProcess.params exposes it for
    # simulation bookkeeping -- it's a deterministic edge_buffer=4*cluster_scale,
    # not a free parameter). Without this, _select_labels in base.py passes
    # all 4 columns through untouched, so every Experiment-framework method
    # here silently predicts/gets scored on 4 targets instead of 3, diluting
    # test_loss with an artificially-easy 4th column. vihrs never had
    # this problem -- it hardcodes exactly LABEL_NAMES regardless of what
    # else is in the params dict.
    "target_label_names": list(run_vihrs.LABEL_NAMES),
}

# (method, dataset filename, adversarial filename) -- trained in this exact
# order for every seed, right after vihrs and fusion (see run_seed below).
BETTI_PI_METHODS = [
    ("ph_combined", "images_dtm_k5.pkl", "adversarial_images_dtm_k5.pkl"),
    ("pi_01", "images_dtm_k5.pkl", "adversarial_images_dtm_k5.pkl"),
    ("betti_cnn_01", "betti_dtm_k5.pkl", "adversarial_betti_dtm_k5.pkl"),
]

# vihrs's own clouds.pkl load + L(r)-r feature extraction is expensive
# and identical across seeds -- loaded lazily, once, and reused for every
# seed instead of redone per-seed now that seeds are the outer loop.
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


def train_vihrs_for_seed(seed: int) -> None:
    output_root = RESULTS_DIR / "vihrs"
    output_dir = output_root / f"seed_{seed}"
    if (output_dir / "results.pt").exists():
        print(f"\nvihrs | seed {seed}: already done, skipping.")
        return

    print(f"\n{'#' * 90}\n### vihrs | seed {seed}\n{'#' * 90}")
    try:
        data = _get_vihrs_data()
        if data is None:
            print(f"! {DATA_DIR / 'clouds.pkl'} missing. Skipping vihrs for seed {seed}.")
            return
        run_vihrs.run_one_seed(
            seed,
            train_records=data["train_records"],
            train_features=data["train_features"],
            adversarial_features=data["adversarial_features"],
            adversarial_path=data["adversarial_path"],
            r_grid=data["r_grid"],
            output_root=output_root,
            n_epochs=VIHRS_N_EPOCHS,
            batch_size=100,  # paper default, matches run_vihrs.py's own CLI default
            lr=0.001,
            device_pref=None,
            skip_mincontrast=True,  # bonus analysis only; drop this to include it
            mc_cache={},
            mc_rng=np.random.default_rng(run_vihrs.MC_SEED),
        )
    except Exception:
        print(f"\n!! vihrs | seed {seed} FAILED, continuing with next seed.\n{traceback.format_exc()}")


# fusion's L(r)-r side reuses _get_vihrs_data()'s already-cached clouds/
# features (same expensive load, same object) rather than reloading; only the
# TDA-side alignment (join by seed against the DTM-filtered images/betti
# files) is fusion-specific and gets cached separately here.
_FUSION_DATA: dict | None = None


def _get_fusion_data() -> dict | None:
    global _FUSION_DATA
    if _FUSION_DATA is None:
        images_path = DATA_DIR / "images_dtm_k5.pkl"
        betti_path = DATA_DIR / "betti_dtm_k5.pkl"
        adv_images_path = DATA_DIR / "adversarial_images_dtm_k5.pkl"
        adv_betti_path = DATA_DIR / "adversarial_betti_dtm_k5.pkl"
        if not images_path.exists() or not betti_path.exists():
            return None

        lr_data = _get_vihrs_data()
        if lr_data is None:
            return None

        train_split = fusion_model.load_fusion_split(
            images_path, betti_path, lr_data["train_features"], tag="train_test",
        )
        adv_split = None
        if lr_data["adversarial_features"] is not None:
            adv_split = fusion_model.load_fusion_split(
                adv_images_path, adv_betti_path, lr_data["adversarial_features"], tag="adversarial",
            )
        _FUSION_DATA = {"train_split": train_split, "adv_split": adv_split}
    return _FUSION_DATA


# The 3 new branch-ablation arms (point 1 of the PH-value investigation):
# vihrs's L(r)-r + n(x) is always on, each arm adds exactly one PH piece --
# the PI image branch, the Betti-curve branch, or just the two cheap
# persistence-entropy scalars with no CNN branch at all. Trained BEFORE the
# original fusion/ph_combined/pi_01/betti_cnn_01 sweep below (per seed), so
# the ablation results are available as soon as possible rather than only
# after the full original sweep finishes.
FUSION_ABLATION_VARIANTS = ["fusion_pi", "fusion_betti", "fusion_scalars"]


def train_fusion_variant_for_seed(seed: int, variant: str) -> None:
    output_dir = fusion_model.output_dir_for_variant(variant, seed)
    if (output_dir / "results.pt").exists():
        print(f"\n{variant} | seed {seed}: already done, skipping.")
        return

    print(f"\n{'#' * 90}\n### {variant} | seed {seed}\n{'#' * 90}")
    try:
        data = _get_fusion_data()
        if data is None or data["train_split"] is None:
            print(f"! fusion data missing for seed {seed} -- run compute_features.py first. Skipping.")
            return
        fusion_model.run_one_seed(seed, data["train_split"], data["adv_split"], variant=variant)
    except Exception:
        print(f"\n!! {variant} | seed {seed} FAILED, continuing with next seed.\n{traceback.format_exc()}")


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


def run_seed(seed: int) -> None:
    # vihrs -> fusion_pi -> fusion_betti -> fusion_scalars -> fusion ->
    # ph_combined -> pi_01 -> betti_cnn_01, in that order; each feature's
    # train_*/experiment.run() call saves its own results.pt the moment it
    # finishes, so progress is visible/comparable as soon as any single
    # feature completes rather than only at the end of the whole seed.
    # vihrs runs first because every fusion variant reuses its L(r)-r data
    # (_get_vihrs_data(), cached); the 3 branch-ablation arms run next so
    # their results land before the rest of the (longer) original sweep.
    print(f"\n{'*' * 90}\n*** seed {seed}\n{'*' * 90}")
    train_vihrs_for_seed(seed)
    for variant in FUSION_ABLATION_VARIANTS:
        train_fusion_variant_for_seed(seed, variant)
    train_fusion_variant_for_seed(seed, "fusion")
    train_betti_pi_for_seed(seed)
    print(f"\n--- seed {seed} done. Compare progress so far with: python dtm_experiment/compare.py ---")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train vihrs/fusion_pi/fusion_betti/fusion_scalars/fusion/ph_combined/pi_01/"
        "betti_cnn_01 for one seed (--seed, for one-GPU-job-per-seed parallelism) or all SEEDS "
        "in sequence (no --seed)."
    )
    p.add_argument("--seed", type=int, default=None, choices=SEEDS)
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    seeds = [args.seed] if args.seed is not None else SEEDS

    for seed in seeds:
        run_seed(seed)

    print("\nAll requested training done. Run dtm_experiment/compare.py next.")


if __name__ == "__main__":
    main()
