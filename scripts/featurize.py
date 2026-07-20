#!/usr/bin/env python3
# scripts/featurize.py
"""Compute persistence diagrams (for each configured filtration -- a list
for a multi-k sweep) and features/vectorizers from already-generated
clouds. Betti-curve and persistence-image outputs also carry a bundled
"persistence_entropy" column (cheap to compute in the same diagram pass,
and every multi-source method in this repo expects it alongside whichever
structural feature it's paired with) -- matching
dtm_experiment/compute_features.py's original convention.

Usage:
    python scripts/featurize.py configs/runs/thomas_dtm_k5_betti_cnn.yaml
    python scripts/featurize.py configs/runs/foo.yaml --set filtration.0.params.k=10 --force
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cloudforger.config import FeatureConfig, RunConfig, load_config
from cloudforger.core.io import dump_pickle, load_pickle
from cloudforger.core.records import diagram_to_record, load_diagrams, to_pointcloud
from cloudforger.features import REGISTRY as FEATURE_REGISTRY
from cloudforger.filtration import REGISTRY as FILTRATION_REGISTRY
from cloudforger.filtration.base import Filtration
from cloudforger.paths import DEFAULT_DATA_ROOT, DataPaths
from cloudforger.vectorizers.calibrated import build_calibrated_imager


def _compute_diagrams(clouds_records: list[dict], filtration: Filtration, tag: str) -> list:
    diagrams = []
    for i, rec in enumerate(clouds_records):
        cloud = to_pointcloud(rec)
        diagrams.append(filtration.compute(cloud))
        if (i + 1) % 500 == 0 or i + 1 == len(clouds_records):
            print(f"  [{tag}] {i + 1}/{len(clouds_records)} diagrams computed", flush=True)
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


def _compute_betti_curve(
    train_diagrams: list, train_bundle: dict, adv_diagrams: list | None, adv_bundle: dict | None,
    feat_cfg: FeatureConfig, out_path: Path, adv_out_path: Path,
) -> None:
    from cloudforger.features.calibrated import build_calibrated_betti_curves

    homology_dims = tuple(feat_cfg.params.get("homology_dims", (0, 1)))
    grid_size = int(feat_cfg.params.get("grid_size", 512))
    weighted = bool(feat_cfg.params.get("weight_by_persistence", False))

    betti_by_dim = build_calibrated_betti_curves(train_diagrams, homology_dims, grid_size, weighted)

    def _payload(diagrams: list, bundle: dict) -> dict:
        matrices = {dim: np.stack([betti_by_dim[dim].compute(d).curves[dim] for d in diagrams]) for dim in homology_dims}
        entropies = _entropy_by_dim(diagrams, homology_dims)
        payload = {**bundle, "betti_params": {dim: b.params for dim, b in betti_by_dim.items()}, "persistence_entropy": entropies}
        for dim in homology_dims:
            payload[f"betti{dim}_matrix"] = matrices[dim]
        return payload

    dump_pickle(out_path, _payload(train_diagrams, train_bundle))
    print(f"  saved betti_curve -> {out_path}")
    if adv_diagrams is not None:
        dump_pickle(adv_out_path, _payload(adv_diagrams, adv_bundle))
        print(f"  saved adversarial betti_curve -> {adv_out_path}")


def _compute_persistence_image(
    train_diagrams: list, train_bundle: dict, adv_diagrams: list | None, adv_bundle: dict | None,
    feat_cfg: FeatureConfig, out_path: Path, adv_out_path: Path,
) -> None:
    homology_dims = tuple(feat_cfg.params.get("homology_dims", (0, 1)))
    imager = build_calibrated_imager(
        train_diagrams, homology_dims=homology_dims,
        resolution=int(feat_cfg.params.get("resolution", 128)), sigma=float(feat_cfg.params.get("sigma", 0.05)),
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
    "betti_curve": _compute_betti_curve,
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
