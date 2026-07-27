#!/usr/bin/env python3
# scripts/featurize_sigma_sweep.py
"""Persistence-image calibration sweep: for a grid of (sigma_pixels,
pd_calibration_coverage) combinations, build calibrated persistence images
for the DTM k channels in the given config's `filtration:` list (k=5,10,15
for nested_thomas_pi_multik_k5k10k15.yaml), one isolated directory per combo
-- so the sweep never touches (or overwrites) the regular
data/<process>/dtm_k<k>/persistence_image.pkl the rest of the pipeline uses,
and combos are safe to run in parallel (e.g. one per SLURM array task).

Diagrams do NOT depend on sigma_pixels/coverage (those are imaging-only
params) -- this script never recomputes them. It loads each k's diagrams
ONCE from the regular cache scripts/featurize.py already built
(data/<process>/dtm_k<k>/diagrams.pkl), and reuses them across every combo
in the sweep; only the imaging step (build_calibrated_imager + .transform)
is redone per combo. Run scripts/featurize.py for this config first if
those diagrams.pkl files don't exist yet -- this script will not build them.

Output layout (mirrors the regular per-k persistence_image.pkl schema
exactly, so scripts/train.py only needs its dataset-path lookup pointed
here temporarily, not its loading logic):
    data/<process>/sigma_sweep/sigma<S>_cov<C>/dtm_k<k>/persistence_image.pkl
    data/<process>/sigma_sweep/sigma<S>_cov<C>/dtm_k<k>/adversarial_persistence_image.pkl

Usage:
    # full grid, run locally
    python scripts/featurize_sigma_sweep.py configs/runs/nested_thomas/nested_thomas_pi_multik_k5k10k15.yaml

    # one combo per invocation, e.g. from a SLURM array task
    python scripts/featurize_sigma_sweep.py configs/runs/nested_thomas/nested_thomas_pi_multik_k5k10k15.yaml \\
        --sigma-pixels 1.5 --coverage 0.95
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cloudforger.config import load_config
from cloudforger.core.io import dump_pickle
from cloudforger.core.records import load_diagrams
from cloudforger.features import REGISTRY as FEATURE_REGISTRY
from cloudforger.filtration import REGISTRY as FILTRATION_REGISTRY
from cloudforger.paths import DEFAULT_DATA_ROOT, DataPaths
from cloudforger.vectorizers.calibrated import build_calibrated_imager

SWEEP_DIR_NAME = "sigma_sweep"

# Live area of work -- override with --sigma-pixels/--coverage rather than
# editing these; they're just sane defaults for an unparameterized run.
# Extended below 1.5 after the first sweep (1.5/2.0/3.0) found monotonically
# decreasing loss as sigma shrank, with no sign of turning over yet.
DEFAULT_SIGMA_PIXELS = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]
DEFAULT_COVERAGE = [0.95, 0.99, 0.999]


def combo_tag(sigma_pixels: float, coverage: float) -> str:
    return f"sigma{sigma_pixels:g}_cov{coverage:g}"


def _entropy_by_dim(diagrams: list, homology_dims: tuple[int, ...]) -> dict[int, np.ndarray]:
    entropy_feature = FEATURE_REGISTRY.build("persistence_entropy", homology_dims=homology_dims)
    per_diagram = [entropy_feature.compute(d) for d in diagrams]
    return {dim: np.array([pd[dim] for pd in per_diagram]) for dim in homology_dims}


def build_combo_images(
    train_diagrams: list,
    train_bundle: dict,
    adv_diagrams: list | None,
    adv_bundle: dict | None,
    homology_dims: tuple[int, ...],
    resolution: int,
    sigma_pixels: float,
    coverage: float,
    out_path: Path,
    adv_out_path: Path,
) -> None:
    imager = build_calibrated_imager(
        train_diagrams, homology_dims=homology_dims, resolution=resolution,
        sigma_pixels=sigma_pixels, coverage=coverage,
    )

    def _payload(diagrams: list, bundle: dict) -> dict:
        images = [imager.transform(d) for d in diagrams]
        image_tensors = {dim: np.stack([im[dim] for im in images]) for dim in homology_dims}
        entropies = _entropy_by_dim(diagrams, homology_dims)
        return {
            **bundle, "image_tensors": image_tensors, "homology_dims": list(homology_dims),
            "imager_params": imager.params, "persistence_entropy": entropies,
        }

    dump_pickle(out_path, _payload(train_diagrams, train_bundle))
    print(f"    saved -> {out_path}")
    if adv_diagrams is not None:
        dump_pickle(adv_out_path, _payload(adv_diagrams, adv_bundle))
        print(f"    saved -> {adv_out_path}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path, help="RunConfig YAML (process/filtration/data_root + base resolution+homology_dims)")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="path.to.field=value")
    parser.add_argument(
        "--sigma-pixels", type=float, nargs="+", default=DEFAULT_SIGMA_PIXELS,
        help=f"sigma_pixels values to sweep, cartesian product with --coverage (default: {DEFAULT_SIGMA_PIXELS})",
    )
    parser.add_argument(
        "--coverage", type=float, nargs="+", default=DEFAULT_COVERAGE,
        help=f"pd_calibration_coverage values to sweep (default: {DEFAULT_COVERAGE})",
    )
    parser.add_argument("--force", action="store_true", help="recompute even if a combo's images already exist")
    args = parser.parse_args(argv)

    cfg = load_config(args.config, overrides=args.overrides)
    if not cfg.filtration:
        raise ValueError(f"{args.config} has no `filtration:` entries -- nothing to sweep k channels from.")
    data_paths = DataPaths(cfg.process.name, root=cfg.data_root or DEFAULT_DATA_ROOT)
    sweep_root = data_paths.process_dir / SWEEP_DIR_NAME

    pi_cfg = next((f for f in cfg.features if f.name == "persistence_image"), None)
    if pi_cfg is None:
        raise ValueError(f"{args.config} has no features: persistence_image entry -- nothing to size the imager from.")
    homology_dims = tuple(pi_cfg.params.get("homology_dims", (0, 1)))
    resolution = int(pi_cfg.params.get("resolution", 64))

    # Load each k's diagrams once, up front -- reused unchanged across every
    # combo below (see module docstring: diagrams are combo-independent).
    per_k_diagrams: dict[int, tuple] = {}
    for f_cfg in cfg.filtration:
        k = f_cfg.params.get("k")
        filtration = FILTRATION_REGISTRY.build(f_cfg.name, **f_cfg.params)
        diagrams_path = data_paths.diagrams([filtration])
        adv_diagrams_path = data_paths.diagrams([filtration], adversarial=True)
        if not diagrams_path.exists():
            raise FileNotFoundError(
                f"{diagrams_path} not found -- run scripts/featurize.py {args.config} first "
                f"(diagrams only; that run's persistence_image output isn't needed here)."
            )
        train_diagrams, train_bundle = load_diagrams(diagrams_path)
        adv_diagrams, adv_bundle = (None, None)
        if adv_diagrams_path.exists():
            adv_diagrams, adv_bundle = load_diagrams(adv_diagrams_path)
        per_k_diagrams[k] = (train_diagrams, train_bundle, adv_diagrams, adv_bundle)
        print(
            f"[dtm_k{k}] loaded {len(train_diagrams)} train diagrams"
            + (f", {len(adv_diagrams)} adversarial" if adv_diagrams is not None else ", no adversarial split")
        )

    combos = list(itertools.product(args.sigma_pixels, args.coverage))
    print(f"\n{len(combos)} combo(s): {combos}")

    for sigma_pixels, coverage in combos:
        tag = combo_tag(sigma_pixels, coverage)
        print(f"\n=== {tag} ===")
        for k, (train_diagrams, train_bundle, adv_diagrams, adv_bundle) in per_k_diagrams.items():
            out_dir = sweep_root / tag / f"dtm_k{k}"
            out_path = out_dir / "persistence_image.pkl"
            adv_out_path = out_dir / "adversarial_persistence_image.pkl"
            if out_path.exists() and not args.force:
                print(f"  [dtm_k{k}] {out_path} exists; skipping (pass --force to recompute).")
                continue
            print(f"  [dtm_k{k}] fitting imager (sigma_pixels={sigma_pixels}, coverage={coverage}) ...")
            try:
                build_combo_images(
                    train_diagrams, train_bundle, adv_diagrams, adv_bundle,
                    homology_dims, resolution, sigma_pixels, coverage, out_path, adv_out_path,
                )
            except ValueError as e:
                # axis_bounds can raise on a degenerate birth axis at some
                # (coverage, homology_dim) combos -- record the failure
                # rather than aborting the rest of the grid.
                print(f"  [dtm_k{k}] SKIPPED ({tag}): {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
