#!/usr/bin/env python3
# scripts/featurize_bifiltration.py
"""Compute signed measures (and their convolution images) from already-generated
clouds, for each configured bifiltration.

Two independently cached stages:

    Stage A  clouds -> signed measures   (~1 s/cloud; the expensive one)
    Stage B  signed measures -> images   (~0.02 s/cloud; the one you'll sweep)

They are cached separately on purpose. Every hyperparameter worth sweeping
(sigma_pixels, image resolution) lives in Stage B, so fusing the stages would
make each sweep re-pay Stage A's ~2 core-hours per 8k clouds.

The ordering difference from featurize.py that matters: there, diagrams are a
fixed fact about a cloud and calibration only decides how to draw them, so it
happens afterwards. Here the grid determines *which module gets computed* -- a
grid coarser than the smallest cluster scale rounds fine structure away with no
error -- so calibration runs first, on TRAIN clouds only, and is then frozen and
stored in the bundle.

Usage:
    python scripts/featurize_bifiltration.py configs/runs/nested_thomas/nested_thomas_mph.yaml
    python scripts/featurize_bifiltration.py <cfg> --set bifiltration.0.params.dtm_mass=0.15 --force
    python scripts/featurize_bifiltration.py <cfg> --stage b --force   # re-image only
    python scripts/featurize_bifiltration.py <cfg> --limit 500         # smoke test
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cloudforger.config import BifiltrationConfig, RunConfig, load_config
from cloudforger.calibration import bifiltration_grid, check_grid_resolves
from cloudforger.core.io import dump_pickle, load_pickle
from cloudforger.core.records import (
    load_signed_measures,
    signed_measure_to_record,
    to_pointcloud,
)
from cloudforger.data_generation.filtration import BIFILTRATION_REGISTRY
from cloudforger.data_generation.filtration.bifiltration import Bifiltration
from cloudforger.paths import DEFAULT_DATA_ROOT, DataPaths
from cloudforger.vectorization.persistence_images.signed_measure_image import build_signed_measure_imagers

# Calibration defaults. Coverage is 0.95 rather than 0.99 because the repo's own
# sigma/coverage sweep (results/nested_thomas/sigma_sweep) found tighter bounds
# beat looser ones on every fine-scale target; the same reasoning applies to the
# codensity axis here.
GRID_RESOLUTION_MIN = 50
GRID_RESOLUTION_MAX = 600
GRID_COVERAGE = 0.95
GRID_SAMPLE = 500


# ---------------------------------------------------------------------------
# Stage A: clouds -> signed measures
# ---------------------------------------------------------------------------


def _build_bundle(measures: list, records: list[dict], grid, bifiltration: Bifiltration) -> dict:
    label_names = list(records[0]["params"].keys())
    labels = np.array([[r["params"][k] for k in label_names] for r in records], dtype=float)
    return {
        "signed_measures": [signed_measure_to_record(m) for m in measures],
        # Grid stored at bundle level as well as per record: Stage B needs it
        # before touching any individual measure, and one authoritative copy
        # means a mismatched record cannot pass unnoticed.
        "grid": (np.asarray(grid[0]), np.asarray(grid[1])),
        "labels": labels,
        "label_names": label_names,
        "seeds": [r["seed"] for r in records],
        "params": [r["params"] for r in records],
        "process": records[0].get("process", ""),
        "bifiltration": bifiltration.name,
        "bifiltration_params": dict(bifiltration.params),
    }


def _report(measures: list, timings: np.ndarray, grid, tag: str) -> None:
    """The three things worth eyeballing before committing to a full run."""
    print(f"  [{tag}] timing: median {np.median(timings):.2f}s, "
          f"p99 {np.percentile(timings, 99):.2f}s, max {np.max(timings):.2f}s "
          f"(total {timings.sum() / 60:.1f} min)")
    for dim in measures[0].dimensions():
        n_atoms = np.array([len(m.atoms(dim)[0]) for m in measures])
        masses = np.array([m.atoms(dim)[1].sum() for m in measures])
        reach = np.array([m.atoms(dim)[0][:, 0].max() if len(m.atoms(dim)[0]) else 0.0
                          for m in measures])
        print(f"  [{tag}] H{dim}: atoms median {int(np.median(n_atoms))} "
              f"(min {n_atoms.min()}, max {n_atoms.max()}), "
              f"empty {int((n_atoms == 0).sum())}")
        print(f"  [{tag}] H{dim}: total mass {masses.min()}..{masses.max()} "
              f"(nonzero is expected; the paper's null-mass convention is not enforced)")
        # Far below the grid top => threshold_radius is wastefully large.
        # Equal to it => atoms are being clipped and the cutoff is too small.
        print(f"  [{tag}] H{dim}: max radius reached {reach.max():.4f} "
              f"vs grid top {grid[0][-1]:.4f}")


def _compute_measures(records: list[dict], bifiltration: Bifiltration, grid, tag: str) -> list:
    measures, timings = [], []
    for i, rec in enumerate(records):
        t0 = time.perf_counter()
        measures.append(bifiltration.compute(to_pointcloud(rec), grid))
        timings.append(time.perf_counter() - t0)
        if (i + 1) % 250 == 0 or i + 1 == len(records):
            done = np.asarray(timings)
            eta = done.mean() * (len(records) - i - 1)
            print(f"  [{tag}] {i + 1}/{len(records)} measures "
                  f"(median {np.median(done):.2f}s, max {done.max():.2f}s, "
                  f"ETA {eta / 60:.1f} min)", flush=True)
    _report(measures, np.asarray(timings), grid, tag)
    return measures


def stage_a(cfg: RunConfig, bifiltration: Bifiltration, data_paths: DataPaths,
            force: bool, limit: int | None) -> None:
    tag = bifiltration.path_tag()
    out = data_paths.signed_measures([bifiltration])

    if out.exists() and not force:
        print(f"  [{tag}] {out} exists; skipping stage A (pass --force to recompute).")
        return

    clouds_path = data_paths.clouds()
    if not clouds_path.exists():
        raise FileNotFoundError(f"No clouds at {clouds_path}; run scripts/generate.py first.")
    clouds = load_pickle(clouds_path)
    if limit:
        clouds = clouds[:limit]
    n_pts = [len(np.asarray(r["points"])) for r in clouds]
    print(f"  [{tag}] {len(clouds)} train clouds "
          f"(n_points median {int(np.median(n_pts))}, max {max(n_pts)})")

    # --- calibration: TRAIN clouds only, before any persistence computation ---
    #
    # Resolution is derived from the data, not fixed. The binding constraint is
    # DATASET-wide, not per-cloud: one grid is shared by every cloud, so it must
    # resolve the smallest cluster_scale anywhere in the set. The sweep's
    # sigma1/sigma2 <= 12 constraint bounds the ratio WITHIN a cloud and does
    # not bound this -- cluster_scale still varies ~12x across clouds on its
    # own, so the requirement is threshold_radius / min(cluster_scale), which
    # came out at 167 on real data where a per-cloud reading suggested 50.
    # Cost of complying is negligible: measured 0.40 -> 0.46 s/cloud going from
    # resolution 50 to 200 (atom count roughly triples, which is the real cost,
    # and it is still only ~14 MB for 8k clouds).
    resolution = GRID_RESOLUTION_MIN
    smallest = None
    if "cluster_scale" in clouds[0]["params"]:
        smallest = min(float(r["params"]["cluster_scale"]) for r in clouds)
        needed = int(np.ceil(float(bifiltration.params["threshold_radius"]) / smallest)) + 1  # linspace(0,t,n) has n-1 intervals
        resolution = int(np.clip(needed, GRID_RESOLUTION_MIN, GRID_RESOLUTION_MAX))
        print(f"  [{tag}] smallest cluster_scale {smallest:.5f} -> "
              f"needs resolution {needed}, using {resolution}")

    sample = [to_pointcloud(r) for r in clouds[:GRID_SAMPLE]]
    grid = bifiltration_grid(sample, bifiltration, resolution=resolution,
                             coverage=GRID_COVERAGE, n_sample=GRID_SAMPLE)
    print(f"  [{tag}] grid: axis0 {grid[0][0]:.4f}..{grid[0][-1]:.4f} "
          f"(step {grid[0][1] - grid[0][0]:.5f}), "
          f"axis1 {grid[1][0]:.4f}..{grid[1][-1]:.4f}")

    # Hard stop rather than a warning: too coarse a grid silently rounds the
    # fine layer away with no error anywhere downstream.
    if smallest is not None:
        check_grid_resolves(grid, smallest)
        print(f"  [{tag}] grid resolves smallest cluster_scale: OK")

    measures = _compute_measures(clouds, bifiltration, grid, tag)
    dump_pickle(out, _build_bundle(measures, clouds, grid, bifiltration))
    print(f"  [{tag}] saved -> {out}")

    # Adversarial set reuses the SAME frozen grid -- same discipline as every
    # other normalization in this repo.
    adv_path = data_paths.clouds(adversarial=True)
    if cfg.use_adversarial and adv_path.exists():
        adv = load_pickle(adv_path)
        if limit:
            adv = adv[:limit]
        print(f"  [{tag}] {len(adv)} adversarial clouds (frozen grid)")
        adv_measures = _compute_measures(adv, bifiltration, grid, f"{tag}/adv")
        adv_out = data_paths.signed_measures([bifiltration], adversarial=True)
        dump_pickle(adv_out, _build_bundle(adv_measures, adv, grid, bifiltration))
        print(f"  [{tag}] saved -> {adv_out}")


# ---------------------------------------------------------------------------
# Stage B: signed measures -> images
# ---------------------------------------------------------------------------


def _image_params(cfg: RunConfig) -> dict:
    for f in cfg.features:
        if f.name in ("mph_image", "signed_measure_image"):
            return {"resolution": int(f.params.get("resolution", 64)),
                    "sigma_pixels": float(f.params.get("sigma_pixels", 0.75)),
                    "clip_mass": bool(f.params.get("clip_mass", True))}
    return {"resolution": 64, "sigma_pixels": 0.75, "clip_mass": True}


def stage_b(cfg: RunConfig, bifiltration: Bifiltration, data_paths: DataPaths,
            force: bool) -> None:
    tag = bifiltration.path_tag()
    feat_params = _image_params(cfg)
    homology_dims = tuple(bifiltration.params["homology_dims"])

    jobs = [(data_paths.signed_measures([bifiltration]),
             data_paths.feature([bifiltration], "mph_image"), tag, False),
            (data_paths.signed_measures([bifiltration], adversarial=True),
             data_paths.feature([bifiltration], "mph_image", adversarial=True),
             f"{tag}/adv", True)]

    for src, dst, label, is_adv in jobs:
        if not src.exists():
            if is_adv:
                continue
            raise FileNotFoundError(f"No signed measures at {src}; run stage A first.")
        if dst.exists() and not force:
            print(f"  [{label}] {dst} exists; skipping (pass --force).")
            continue

        measures, bundle = load_signed_measures(src)
        grid = bundle["grid"]
        imagers = build_signed_measure_imagers(grid, homology_dims=homology_dims, **feat_params)

        # (n_clouds, n_dims, res, res) -- the layout build_pi_tensor expects, so
        # pi_multik consumes this unchanged.
        images = np.stack([np.stack([imagers.transform(m)[d] for d in homology_dims])
                           for m in measures])

        # Mass conservation is the invariant that caught the boundary-truncation
        # bug; assert it here so a future change to _box_mass or the grid can't
        # regress silently.
        true_mass = np.array([[m.atoms(d)[1].sum() for d in homology_dims] for m in measures])
        mass_err = np.abs(images.sum(axis=(2, 3)) - true_mass).max()
        if feat_params["clip_mass"] and mass_err > 1e-6:
            raise AssertionError(f"mass not conserved: max error {mass_err:.2e}")

        print(f"  [{label}] images {images.shape}, "
              f"range {images.min():.3f}..{images.max():.3f}, "
              f"nonzero {100 * (np.abs(images) > 1e-9).mean():.0f}%, "
              f"mass err {mass_err:.1e}")
        dump_pickle(dst, {
            "image_tensors": images,
            "grid": grid,
            "labels": bundle["labels"],
            "label_names": bundle["label_names"],
            "seeds": bundle["seeds"],
            "params": bundle["params"],
            "process": bundle["process"],
            "imager_params": imagers.params,
        })
        print(f"  [{label}] saved -> {dst}")


# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path, help="RunConfig YAML path")
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        metavar="path.to.field=value")
    parser.add_argument("--force", action="store_true", help="recompute even if outputs exist")
    parser.add_argument("--stage", choices=["a", "b", "both"], default="both",
                        help="a = measures only, b = images only (needs a done)")
    parser.add_argument("--limit", type=int, default=None,
                        help="only process the first N clouds (smoke test)")
    args = parser.parse_args(argv)

    cfg = load_config(args.config, overrides=args.overrides)
    if not cfg.bifiltration:
        raise ValueError(f"{args.config} has no 'bifiltration:' block.")
    data_paths = DataPaths(cfg.process.name, root=cfg.data_root or DEFAULT_DATA_ROOT)

    for bcfg in cfg.bifiltration:
        if not isinstance(bcfg, BifiltrationConfig):
            raise TypeError(
                "cfg.bifiltration holds raw dicts, not BifiltrationConfig -- "
                "RunConfig.from_dict is not parsing the 'bifiltration' block."
            )
        bifiltration = BIFILTRATION_REGISTRY.build(bcfg.name, **bcfg.params)
        print(f"[{bifiltration.path_tag()}] {bifiltration.name} params={bifiltration.params}")
        if args.stage in ("a", "both"):
            stage_a(cfg, bifiltration, data_paths, args.force, args.limit)
        if args.stage in ("b", "both"):
            stage_b(cfg, bifiltration, data_paths, args.force)


if __name__ == "__main__":
    main()