# scripts/processing/params/delete.py

from __future__ import annotations

import argparse
import pickle
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np


DEFAULT_DATA_ROOT = Path(
    "/Users/qp252676/Desktop/point-process-tda/data/params/2d/inhom_thomas"
)

SPLITS: dict[str, dict[str, str]] = {
    "train_test": {
        "clouds": "clouds.pkl",
        "diagrams": "diagrams.pkl",
        "images": "images.pkl",
        "features": "features.pkl",
    },
    "adversarial": {
        "clouds": "adversarial_clouds.pkl",
        "diagrams": "adversarial_diagrams.pkl",
        "images": "adversarial_images.pkl",
        "features": "adversarial_features.pkl",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Vectorize diagrams and compute pair-distance features, with covariates "
            "propagated into the saved pickle bundles."
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Directory containing clouds.pkl, diagrams.pkl, etc.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help=(
            "Project root. If omitted, inferred from --data-root when possible. "
            "Use this if imports fail."
        ),
    )
    parser.add_argument("--dimension", type=int, default=2)
    parser.add_argument("--process", default="inhom_thomas")
    parser.add_argument("--homology-dims", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--image-resolution", type=int, default=64)
    parser.add_argument("--sigma", type=float, default=0.05)
    parser.add_argument("--pair-n-samples", type=int, default=5_000)
    parser.add_argument("--pair-grid-size", type=int, default=64)
    parser.add_argument("--pair-seed", type=int, default=20240601)
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=["vectorize", "pairwise"],
        default=["vectorize", "pairwise"],
        help="Which stages to run.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=sorted(SPLITS),
        default=["train_test", "adversarial"],
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not create .bak backups of existing output files before overwriting.",
    )
    return parser.parse_args()


def infer_project_root(data_root: Path) -> Path:
    """Infer the repository root from .../point-process-tda/data/params/..."""
    data_root = data_root.resolve()
    for parent in [data_root, *data_root.parents]:
        if (parent / "scripts" / "processing" / "params" / "pipeline_lib").exists():
            return parent
    # With the provided default, this is usually data_root.parents[3].
    if len(data_root.parents) >= 4:
        return data_root.parents[3]
    return Path.cwd()


def configure_imports(project_root: Path) -> None:
    candidates = [
        project_root,
        project_root / "src",
        project_root / "scripts" / "processing" / "params",
    ]
    for path in candidates:
        s = str(path)
        if s not in sys.path:
            sys.path.insert(0, s)


def load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def dump_pickle(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def backup_once(path: Path, enabled: bool = True) -> None:
    if not enabled or not path.exists():
        return
    backup = path.with_suffix(path.suffix + ".bak")
    if not backup.exists():
        shutil.copy2(path, backup)
        print(f"  Backed up {path.name} -> {backup.name}")


def get_items(bundle: Any, key: str) -> list[Any]:
    if isinstance(bundle, dict) and key in bundle:
        return list(bundle[key])
    return list(bundle)


def get_seed(record: Any) -> Any:
    if isinstance(record, dict):
        return record.get("seed")
    return getattr(record, "seed", None)


def get_params(record: Any) -> dict[str, Any]:
    if isinstance(record, dict):
        return dict(record.get("params", {}))
    return dict(getattr(record, "generator_params", {}))


def get_points(record: Any) -> np.ndarray | None:
    if isinstance(record, dict):
        pts = record.get("points")
    else:
        pts = getattr(record, "points", None)
    return None if pts is None else np.asarray(pts)


def get_covariates(record: Any) -> Any:
    if isinstance(record, dict):
        return record.get("covariates")
    return getattr(record, "covariates", None)


def scalarize_params(params: dict[str, Any]) -> dict[str, float]:
    """Flatten vector-valued parameters such as beta into scalar label columns."""
    out: dict[str, float] = {}
    for key, value in params.items():
        arr = np.asarray(value)
        if arr.ndim == 0:
            out[key] = float(arr)
        else:
            for j, item in enumerate(arr.ravel()):
                out[f"{key}_{j}"] = float(item)
    return out


def build_numeric_labels(records: Iterable[Any]) -> tuple[np.ndarray, list[str], list[dict[str, float]]]:
    params = [scalarize_params(get_params(r)) for r in records]
    if not params:
        raise ValueError("Cannot build labels from an empty record list.")

    label_names = list(params[0].keys())
    for i, p in enumerate(params):
        if list(p.keys()) != label_names:
            raise ValueError(
                f"Parameter keys are inconsistent at index {i}: "
                f"{list(p.keys())} != {label_names}"
            )

    labels = np.asarray([[p[name] for name in label_names] for p in params], dtype=float)
    return labels, label_names, params


def cloud_with_scalar_params(cloud: Any) -> Any:
    if not isinstance(cloud, dict):
        return cloud
    record = dict(cloud)
    record["params"] = scalarize_params(dict(cloud.get("params", {})))
    return record


def aligned_covariates_from_clouds(
    data_root: Path,
    split_name: str,
    target_records: list[Any],
) -> tuple[list[Any], np.ndarray]:
    clouds_path = data_root / SPLITS[split_name]["clouds"]
    clouds = get_items(load_pickle(clouds_path), "clouds")

    if len(clouds) != len(target_records):
        raise ValueError(
            f"{split_name}: length mismatch while aligning covariates: "
            f"{len(clouds)=}, {len(target_records)=}"
        )

    covariates: list[Any] = []
    n_points: list[int] = []

    for i, (cloud, target) in enumerate(zip(clouds, target_records)):
        cloud_seed = get_seed(cloud)
        target_seed = get_seed(target)
        if cloud_seed != target_seed:
            raise ValueError(
                f"{split_name}: seed mismatch at index {i}: "
                f"cloud seed {cloud_seed!r}, target seed {target_seed!r}"
            )

        cov = get_covariates(cloud)
        if cov is None:
            raise ValueError(f"{split_name}: cloud index {i} has no covariates")

        pts = get_points(cloud)
        covariates.append(cov)
        n_points.append(0 if pts is None else int(pts.shape[0]))

    return covariates, np.asarray(n_points, dtype=int)


def vectorize_split_with_covariates(
    *,
    data_root: Path,
    split_name: str,
    imager: Any,
    homology_dims: list[int],
    resolution: int,
    dimension: int,
    backup: bool,
) -> None:
    from pipeline_lib.records import load_diagrams
    from pipeline_lib.images import stack_images_by_dim

    diagrams_path = data_root / SPLITS[split_name]["diagrams"]
    out_path = data_root / SPLITS[split_name]["images"]

    diagrams, bundle = load_diagrams(diagrams_path)
    raw_diagrams = get_items(bundle, "diagrams")
    covariates, n_points = aligned_covariates_from_clouds(data_root, split_name, raw_diagrams)

    print(f"\nVectorizing {split_name}: {len(diagrams)} diagrams from {diagrams_path}")
    images = [imager.transform(d) for d in diagrams]
    image_tensors = stack_images_by_dim(images, homology_dims, resolution)

    labels, label_names, params = build_numeric_labels(raw_diagrams)
    seeds = [get_seed(d) for d in raw_diagrams]

    payload: dict[str, Any] = {
        "images": images,
        "image_tensors": image_tensors,
        "homology_dims": list(imager.dimensions),
        "imager_params": imager.params,
        "dimension": dimension,
        "split": split_name,
        "source_diagrams": str(diagrams_path),
        "labels": labels,
        "label_names": label_names,
        "params": params,
        "seeds": seeds,
        "covariates": covariates,
        "n_points": n_points,
    }

    for key in ("process", "filtration_params"):
        if isinstance(bundle, dict) and key in bundle:
            payload[key] = bundle[key]

    backup_once(out_path, enabled=backup)
    dump_pickle(out_path, payload)
    shapes = {int(k): v.shape for k, v in image_tensors.items()}
    print(f"  Saved {len(images)} persistence images -> {out_path}")
    print(f"  Image tensors by dimension: {shapes}")


def make_pair_distance_cdf(n_samples: int, grid_size: int, seed: int) -> Any:
    from cloudforger.stats.pair_dist import PairDistanceCDF

    attempts = [
        lambda: PairDistanceCDF(n_samples=n_samples, grid_size=grid_size, seed=seed),
        lambda: PairDistanceCDF(n_samples=n_samples, grid_size=grid_size),
        lambda: PairDistanceCDF(n_samples=n_samples, resolution=grid_size, seed=seed),
        lambda: PairDistanceCDF(n_samples=n_samples, resolution=grid_size),
        lambda: PairDistanceCDF(n_samples, grid_size, seed),
        lambda: PairDistanceCDF(n_samples, grid_size),
    ]

    errors: list[str] = []
    for attempt in attempts:
        try:
            return attempt()
        except TypeError as exc:
            errors.append(str(exc))

    raise TypeError(
        "Could not construct PairDistanceCDF. Constructor attempts failed with:\n"
        + "\n".join(f"  - {e}" for e in errors)
    )


def statistic_length(statistic: Any, default_grid_size: int) -> int:
    params = getattr(statistic, "params", {}) or {}
    for key in ("grid_size", "resolution", "n_grid"):
        if key in params:
            return int(params[key])
    for attr in ("grid_size", "_grid_size", "resolution", "_resolution"):
        if hasattr(statistic, attr):
            value = getattr(statistic, attr)
            if isinstance(value, (int, np.integer)):
                return int(value)
    return int(default_grid_size)


def compute_statistic_safely(statistic: Any, pointcloud: Any, default_grid_size: int) -> np.ndarray:
    pts = np.asarray(pointcloud.points)

    # PairDistanceCDF samples two distinct points. For clouds with 0 or 1 point,
    # no pairwise distances exist, so use an all-zero CDF vector and continue.
    if getattr(statistic, "name", "") == "pair_distance_cdf" and pts.shape[0] < 2:
        return np.zeros(statistic_length(statistic, default_grid_size), dtype=float)

    return np.asarray(statistic.compute(pointcloud), dtype=float).ravel()


def compute_pairwise_split_with_covariates(
    *,
    data_root: Path,
    split_name: str,
    statistic: Any,
    process: str,
    grid_size: int,
    backup: bool,
) -> None:
    from cloudforger.core.features import CorrelationFeatures
    from pipeline_lib.records import features_to_record, to_pointcloud

    clouds_path = data_root / SPLITS[split_name]["clouds"]
    out_path = data_root / SPLITS[split_name]["features"]

    raw_clouds = get_items(load_pickle(clouds_path), "clouds")
    clouds = [cloud_with_scalar_params(c) for c in raw_clouds]

    labels, label_names, params = build_numeric_labels(clouds)
    seeds = [get_seed(c) for c in clouds]
    covariates = [get_covariates(c) for c in raw_clouds]
    n_points = np.asarray(
        [0 if get_points(c) is None else int(get_points(c).shape[0]) for c in raw_clouds],
        dtype=int,
    )

    if any(cov is None for cov in covariates):
        first = next(i for i, cov in enumerate(covariates) if cov is None)
        raise ValueError(f"{split_name}: cloud index {first} has no covariates")

    statistics = [statistic]
    stat_params = {s.name: s.params for s in statistics}
    order = sorted(s.name for s in statistics)

    print(
        f"\nComputing pairwise features for {split_name}: "
        f"{len(clouds)} clouds {[s.name for s in statistics]} from {clouds_path}"
    )

    feature_records: list[dict[str, Any]] = []
    rows: list[np.ndarray] = []

    for i, (cloud, cov, n) in enumerate(zip(clouds, covariates, n_points)):
        pc = to_pointcloud(cloud)
        vectors = {
            s.name: compute_statistic_safely(s, pc, default_grid_size=grid_size)
            for s in statistics
        }
        row = np.concatenate([vectors[name] for name in order])
        rows.append(row)

        cf = CorrelationFeatures(
            features=vectors,
            generator_name=pc.generator_name,
            generator_params=pc.generator_params,
            seed=pc.seed,
            statistic_params=stat_params,
        )
        record = features_to_record(cf)
        record["covariates"] = cov
        record["n_points"] = int(n)
        feature_records.append(record)

        print(f"\r  {i + 1}/{len(clouds)}", end="", flush=True)
    print()

    feature_matrix = np.stack(rows) if rows else np.empty((0, 0), dtype=float)

    payload: dict[str, Any] = {
        "features": feature_records,
        "feature_matrix": feature_matrix,
        "feature_order": order,
        "labels": labels,
        "label_names": label_names,
        "params": params,
        "seeds": seeds,
        "process": process,
        "statistic_params": stat_params,
        "covariates": covariates,
        "n_points": n_points,
    }

    backup_once(out_path, enabled=backup)
    dump_pickle(out_path, payload)
    print(f"  Saved pairwise features -> {out_path}")
    print(f"  feature_matrix shape: {feature_matrix.shape}")


def main() -> None:
    args = parse_args()
    data_root = args.data_root.expanduser().resolve()
    project_root = (args.project_root.expanduser().resolve() if args.project_root else infer_project_root(data_root))
    configure_imports(project_root)

    from pipeline_lib.records import load_diagrams
    from pipeline_lib.images import build_imagers

    print(f"Data root: {data_root}")
    print(f"Project root: {project_root}")

    backup = not args.no_backup

    if "vectorize" in args.stages:
        calibration_path = data_root / SPLITS["train_test"]["diagrams"]
        calibration_diagrams, _ = load_diagrams(calibration_path)
        print(f"\nCalibrating persistence imager on {len(calibration_diagrams)} train/test diagrams")
        imager = build_imagers(
            calibration_diagrams,
            homology_dims=list(args.homology_dims),
            resolution=int(args.image_resolution),
            sigma=float(args.sigma),
        )

        for split_name in args.splits:
            vectorize_split_with_covariates(
                data_root=data_root,
                split_name=split_name,
                imager=imager,
                homology_dims=list(args.homology_dims),
                resolution=int(args.image_resolution),
                dimension=int(args.dimension),
                backup=backup,
            )

    if "pairwise" in args.stages:
        statistic = make_pair_distance_cdf(
            n_samples=int(args.pair_n_samples),
            grid_size=int(args.pair_grid_size),
            seed=int(args.pair_seed),
        )
        for split_name in args.splits:
            compute_pairwise_split_with_covariates(
                data_root=data_root,
                split_name=split_name,
                statistic=statistic,
                process=args.process,
                grid_size=int(args.pair_grid_size),
                backup=backup,
            )


if __name__ == "__main__":
    main()
