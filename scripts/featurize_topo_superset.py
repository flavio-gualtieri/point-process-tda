#!/usr/bin/env python3
# scripts/featurize_topo_superset.py
"""Build persistence images for a chosen subset of the topo_superset
mass-fraction DTM diagrams (scripts/precompute_topo_superset.py) -- the
"channel selection" step that script's docstring deliberately deferred to
a later step.

Reuses the exact transform scripts/featurize.py's persistence_image
feature handler uses (build_calibrated_imager, fit per m on TRAIN diagrams
only, then .transform per diagram), with the same
resolution/sigma_pixels/homology_dims as the config's
features.persistence_image block by default, so these channels are
directly comparable to the existing dtm_k5/dtm_k10/dtm_k15 ones.

topo_superset diagrams are stored flat
(topo_superset/dtm_m<value>_diagrams.pkl), not under the per-k DataPaths
convention scripts/featurize.py expects -- mass-fraction DTM needs a
per-cloud k = round(m*N), which the fixed-k Filtration/DataPaths path has
no hook for (see precompute_topo_superset.py's docstring). So this writes
flat sibling topo_superset/dtm_m<value>_persistence_image.pkl files (same
schema as dtm_k5/persistence_image.pkl: image_tensors, persistence_entropy,
imager_params) instead of going through scripts/featurize.py.

The m-channel subset is a live area of work -- override with --m-values
rather than editing DEFAULT_M_VALUES.

Usage:
    python scripts/featurize_topo_superset.py configs/runs/nested_thomas_pi_multik.yaml
    python scripts/featurize_topo_superset.py configs/runs/foo.yaml \\
        --m-values 0.01 0.02 0.04 0.07 0.10 0.15 0.20 --force
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cloudforger.config import load_config
from cloudforger.core.io import dump_pickle
from cloudforger.core.records import load_diagrams
from cloudforger.features import REGISTRY as FEATURE_REGISTRY
from cloudforger.paths import DEFAULT_DATA_ROOT, DataPaths
from cloudforger.vectorizers.calibrated import build_calibrated_imager

# Even log-spaced coverage of precompute_topo_superset.py's full 11-point m
# grid (0.01 ... 0.90): fine end matches the existing k=5,10,15 regime,
# coarse end reaches into the previously-unsampled parent-scale territory.
# This is a live area of work -- override with --m-values, don't edit here.
DEFAULT_M_VALUES = [0.01, 0.02, 0.04, 0.10, 0.20, 0.45, 0.90]

HOMOLOGY_DIMS = (0, 1)


def _entropy_by_dim(diagrams: list, homology_dims: tuple[int, ...]) -> dict[int, np.ndarray]:
    entropy_feature = FEATURE_REGISTRY.build("persistence_entropy", homology_dims=homology_dims)
    per_diagram = [entropy_feature.compute(d) for d in diagrams]
    return {dim: np.array([pd[dim] for pd in per_diagram]) for dim in homology_dims}


def _build_images(
    train_diagrams: list, train_bundle: dict, adv_diagrams: list | None, adv_bundle: dict | None,
    homology_dims: tuple[int, ...], resolution: int, sigma_pixels: float,
    out_path: Path, adv_out_path: Path,
) -> None:
    imager = build_calibrated_imager(
        train_diagrams, homology_dims=homology_dims, resolution=resolution,
        sigma_pixels=sigma_pixels,
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
    print(f"  saved -> {out_path}")
    if adv_diagrams is not None:
        dump_pickle(adv_out_path, _payload(adv_diagrams, adv_bundle))
        print(f"  saved -> {adv_out_path}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path, help="RunConfig YAML path (process name / data_root / feature params)")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="path.to.field=value")
    parser.add_argument(
        "--m-values", type=float, nargs="+", default=DEFAULT_M_VALUES,
        help=f"mass-fraction channels to featurize (default: {DEFAULT_M_VALUES})",
    )
    parser.add_argument("--force", action="store_true", help="recompute even if outputs already exist")
    args = parser.parse_args(argv)

    cfg = load_config(args.config, overrides=args.overrides)
    data_paths = DataPaths(cfg.process.name, root=cfg.data_root or DEFAULT_DATA_ROOT)
    superset_dir = data_paths.process_dir / "topo_superset"

    pi_cfg = next((f for f in cfg.features if f.name == "persistence_image"), None)
    if pi_cfg is None:
        raise ValueError(f"{args.config} has no features: persistence_image entry -- nothing to size the imager from.")
    homology_dims = tuple(pi_cfg.params.get("homology_dims", HOMOLOGY_DIMS))
    resolution = int(pi_cfg.params.get("resolution", 64))
    sigma_pixels = float(pi_cfg.params.get("sigma_pixels", 2.0))

    print(f"m channels: {args.m_values}")
    print(f"imager params: resolution={resolution} sigma_pixels={sigma_pixels} homology_dims={homology_dims}")

    for m in args.m_values:
        tag = f"m={m:.2f}"
        diagrams_path = superset_dir / f"dtm_m{m:.2f}_diagrams.pkl"
        adv_diagrams_path = superset_dir / f"adversarial_dtm_m{m:.2f}_diagrams.pkl"
        out_path = superset_dir / f"dtm_m{m:.2f}_persistence_image.pkl"
        adv_out_path = superset_dir / f"adversarial_dtm_m{m:.2f}_persistence_image.pkl"

        if not diagrams_path.exists():
            raise FileNotFoundError(f"{diagrams_path} not found -- run scripts/precompute_topo_superset.py first.")
        if out_path.exists() and not args.force:
            print(f"[{tag}] {out_path} exists; skipping (pass --force to recompute).")
            continue

        train_diagrams, train_bundle = load_diagrams(diagrams_path)
        adv_diagrams, adv_bundle = (None, None)
        if adv_diagrams_path.exists():
            adv_diagrams, adv_bundle = load_diagrams(adv_diagrams_path)

        print(f"[{tag}] fitting imager on {len(train_diagrams)} train diagrams ...")
        _build_images(
            train_diagrams, train_bundle, adv_diagrams, adv_bundle,
            homology_dims, resolution, sigma_pixels, out_path, adv_out_path,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
