#!/usr/bin/env python3
# scripts/featurize.py
"""Compute persistence diagrams (for each configured filtration -- a list
for a multi-k sweep) and features/vectorizers from already-generated
clouds. Persistence-image output also carries a bundled
"persistence_entropy" column (cheap to compute in the same diagram pass,
and every multi-source method in this repo expects it alongside whichever
structural feature it's paired with) -- matching
dtm_experiment/compute_features.py's original convention.

Betti curves are deliberately NOT a `features:` entry computed here: their
calibration is a population-level fit (build_calibrated_betti), so
precomputing them here -- before scripts/train.py's train/val/test split
exists -- would bake in the same leakage bug pi_multik.py's module
docstring documents and fixes for persistence images. The betti_multik
method (experiments/pi_multik/betti_multik.py) computes them on the fly
instead, per seed, from the diagrams.pkl this script always writes,
calibrated on that seed's train rows only.

Usage:
    python scripts/featurize.py configs/runs/thomas/thomas_pi_multik_k5k10k15.yaml
    python scripts/featurize.py configs/runs/foo.yaml --set filtration.0.params.k=10 --force
"""

from __future__ import annotations

import os

# Single-threaded BLAS/OpenMP per process, set before numpy is imported
# anywhere in the chain below (including in ProcessPoolExecutor workers,
# which re-import this module under macOS's 'spawn' start method) --
# _compute_diagrams's per-cloud DTM/Rips computation turned out to already
# be internally multi-threaded, so N worker PROCESSES x M internal threads
# each was oversubscribing the machine's cores far worse than N alone would
# (observed: a single diagram computation took 58s under 2-process
# contention vs ~2-4s typical) -- one thread per worker process instead.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cloudforger.config import FeatureConfig, RunConfig, load_config
from cloudforger.core.io import dump_pickle, load_pickle
from cloudforger.core.records import diagram_to_record, load_diagrams, to_pointcloud
from cloudforger.vectorization.scalar_features import REGISTRY as FEATURE_REGISTRY
from cloudforger.data_generation.filtration import REGISTRY as FILTRATION_REGISTRY
from cloudforger.data_generation.filtration.base import Filtration
from cloudforger.paths import DEFAULT_DATA_ROOT, DataPaths
from cloudforger.vectorization.persistence_images.calibrated import build_calibrated_imager


def _compute_one_diagram(rec: dict, filtration: Filtration):
    """Module-level (picklable) target for ProcessPoolExecutor workers --
    a bound/nested function can't cross the process boundary."""
    return filtration.compute(to_pointcloud(rec))


def _compute_diagrams(clouds_records: list[dict], filtration: Filtration, tag: str) -> list:
    n = len(clouds_records)
    # Capped well below os.cpu_count(): each worker is a fresh spawned
    # process re-importing numpy/scipy/gudhi/multipers's full C++ backends,
    # and running all 10 cores' worth concurrently pushed this machine's
    # memory hard enough to crash the VS Code host process outright (not
    # just slow things down) -- a few workers is a much safer default.
    # This machine has only 16GB total RAM. Even bounded (max_tasks_per_child
    # below), each worker's per-cloud DTM peak still reaches ~1-2GB before a
    # recycle, and 4 concurrent workers left the system with under 150MB free
    # -- too close to swapping/OOM given everything else running (editor,
    # browser, ...). 2 workers leaves real headroom.
    # Machine is at 16GB RAM and baseline load (editor, browser, ...) alone
    # was already leaving well under 1GB free before this job adds anything --
    # after 3 crashes, correctness/stability beats speed here: 1 worker only.
    workers = max(1, min(1, os.cpu_count() or 1))
    # Small workloads (e.g. a --limit smoke run) aren't worth process-pool
    # spawn overhead -- plain sequential loop, same as before.
    # NOTE: workers==1 still goes through the pool below (not this branch) so
    # it gets max_tasks_per_child recycling too -- the per-cloud DTM leak
    # doesn't care whether it's leaking in a worker or the main process, and a
    # 7000-call unbounded leak in a single process is just as dangerous, only
    # slower to manifest. Only truly tiny workloads skip the pool entirely.
    if n < 200:
        diagrams = []
        for i, rec in enumerate(clouds_records):
            diagrams.append(_compute_one_diagram(rec, filtration))
            if (i + 1) % 500 == 0 or i + 1 == n:
                print(f"  [{tag}] {i + 1}/{n} diagrams computed", flush=True)
        return diagrams

    diagrams: list = [None] * n
    done = 0
    # max_tasks_per_child: the underlying DTM/persistent-homology C++
    # backend leaks memory per call (observed: a single worker reached
    # 8GB RSS after ~20-30 calls, unbounded growth, not a fixed import
    # cost) -- respawning each worker after a handful of tasks resets
    # that growth instead of letting it run away and exhaust memory.
    with ProcessPoolExecutor(max_workers=workers, max_tasks_per_child=10) as pool:
        futures = {pool.submit(_compute_one_diagram, rec, filtration): i for i, rec in enumerate(clouds_records)}
        for future in as_completed(futures):
            i = futures[future]
            diagrams[i] = future.result()
            done += 1
            if done % 500 == 0 or done == n:
                print(f"  [{tag}] {done}/{n} diagrams computed ({workers} workers)", flush=True)
    return diagrams


def _build_bundle(diagrams: list, records: list[dict]) -> dict:
    label_names = list(records[0]["params"].keys())
    labels = np.array([[r["params"][k] for k in label_names] for r in records], dtype=float)
    return {
        "diagrams": [diagram_to_record(d) for d in diagrams],
        "labels": labels,
        "label_names": label_names,
        "seeds": [r["seed"] for r in records],
        "params": [r["params"] for r in records],
        "process": records[0].get("process", ""),
    }


def _get_or_compute_diagrams(
    clouds_path: Path, out_path: Path, filtration: Filtration, force: bool, tag: str
) -> tuple[list, dict] | None:
    if not clouds_path.exists():
        return None
    if out_path.exists() and not force:
        print(f"  [{tag}] {out_path} exists; loading cached diagrams (pass --force to recompute).")
        return load_diagrams(out_path)

    clouds = load_pickle(clouds_path)
    print(f"  [{tag}] computing {filtration.name} (params={filtration.params}) diagrams for {len(clouds)} clouds ...")
    diagrams = _compute_diagrams(clouds, filtration, tag)
    bundle = _build_bundle(diagrams, clouds)
    dump_pickle(out_path, bundle)
    print(f"  [{tag}] saved diagrams -> {out_path}")
    return diagrams, bundle


def _entropy_by_dim(diagrams: list, homology_dims: tuple[int, ...]) -> dict[int, np.ndarray]:
    entropy_feature = FEATURE_REGISTRY.build("persistence_entropy", homology_dims=homology_dims)
    per_diagram = [entropy_feature.compute(d) for d in diagrams]  # each: {dim: float}
    return {dim: np.array([pd[dim] for pd in per_diagram]) for dim in homology_dims}


def _compute_persistence_image(
    train_diagrams: list, train_bundle: dict, adv_diagrams: list | None, adv_bundle: dict | None,
    feat_cfg: FeatureConfig, out_path: Path, adv_out_path: Path,
) -> None:
    homology_dims = tuple(feat_cfg.params.get("homology_dims", (0, 1)))
    imager = build_calibrated_imager(
        train_diagrams,
        homology_dims=homology_dims,
        resolution=int(feat_cfg.params.get("resolution", 64)),
        sigma_pixels=float(feat_cfg.params.get("sigma_pixels", 2.0)),
        coverage=float(feat_cfg.params.get("pd_calibration_coverage", 0.99))
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
    print(f"  saved persistence_image -> {out_path}")
    if adv_diagrams is not None:
        dump_pickle(adv_out_path, _payload(adv_diagrams, adv_bundle))
        print(f"  saved adversarial persistence_image -> {adv_out_path}")


def _compute_persistence_entropy_standalone(
    train_diagrams: list, train_bundle: dict, adv_diagrams: list | None, adv_bundle: dict | None,
    feat_cfg: FeatureConfig, out_path: Path, adv_out_path: Path,
) -> None:
    homology_dims = tuple(feat_cfg.params.get("homology_dims", (0, 1)))

    def _payload(diagrams: list, bundle: dict) -> dict:
        return {**bundle, "persistence_entropy": _entropy_by_dim(diagrams, homology_dims)}

    dump_pickle(out_path, _payload(train_diagrams, train_bundle))
    print(f"  saved persistence_entropy -> {out_path}")
    if adv_diagrams is not None:
        dump_pickle(adv_out_path, _payload(adv_diagrams, adv_bundle))
        print(f"  saved adversarial persistence_entropy -> {adv_out_path}")


_FEATURE_HANDLERS = {
    "persistence_image": _compute_persistence_image,
    "persistence_entropy": _compute_persistence_entropy_standalone,
}


def run_one_filtration(cfg: RunConfig, filt_cfg, data_paths: DataPaths, force: bool) -> None:
    filtration = FILTRATION_REGISTRY.build(filt_cfg.name, **filt_cfg.params)
    tag = filtration.path_tag()

    diagrams_path = data_paths.diagrams([filtration])
    adv_diagrams_path = data_paths.diagrams([filtration], adversarial=True)

    train = _get_or_compute_diagrams(data_paths.clouds(), diagrams_path, filtration, force, tag)
    if train is None:
        raise FileNotFoundError(f"{data_paths.clouds()} not found -- run scripts/generate.py first.")
    train_diagrams, train_bundle = train

    adv = _get_or_compute_diagrams(data_paths.clouds(adversarial=True), adv_diagrams_path, filtration, force, f"{tag} adversarial")
    adv_diagrams, adv_bundle = adv if adv is not None else (None, None)

    if not cfg.features:
        print(f"  [{tag}] no features configured; diagrams only.")
        return

    for feat_cfg in cfg.features:
        handler = _FEATURE_HANDLERS.get(feat_cfg.name)
        if handler is None:
            raise ValueError(f"Unknown feature {feat_cfg.name!r}; expected one of {sorted(_FEATURE_HANDLERS)}")
        out_path = data_paths.feature([filtration], feat_cfg.name)
        adv_out_path = data_paths.feature([filtration], feat_cfg.name, adversarial=True)
        if out_path.exists() and not force:
            print(f"  [{tag}] {out_path} exists; skipping {feat_cfg.name} (pass --force to recompute).")
            continue
        handler(train_diagrams, train_bundle, adv_diagrams, adv_bundle, feat_cfg, out_path, adv_out_path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path, help="RunConfig YAML path")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="path.to.field=value")
    parser.add_argument("--force", action="store_true", help="recompute even if outputs already exist")
    args = parser.parse_args(argv)

    cfg = load_config(args.config, overrides=args.overrides)
    data_paths = DataPaths(cfg.process.name, root=cfg.data_root or DEFAULT_DATA_ROOT)

    if not cfg.filtration:
        print("No filtration configured; nothing to do.")
        return

    for filt_cfg in cfg.filtration:
        run_one_filtration(cfg, filt_cfg, data_paths, args.force)

    print("\nDone.")


if __name__ == "__main__":
    main()
