#!/usr/bin/env python3
"""Stage 2 (statistics branch): clouds -> correlation features.

Reads the clouds produced by ``generate_clouds_params.py``
(default ``data/params/<process>/clouds.pkl``) and writes correlation features
(default ``data/params/<process>/features.pkl``), carrying the parameter labels
through so every feature vector stays paired with its generating parameters.

Rerun if a statistic's config (n_samples, grid_size) changes.

Each cloud record is reconstructed into a ``PointCloud`` (with its ``Box``
region) before the statistics run, because the statistics read ``cloud.points``,
``cloud.seed`` and -- for PairDistanceCDF -- require ``cloud.region``.

Output pickle (dict):
    {
        "features":         list, one entry per cloud (aligned with labels);
                            a record {"features": {stat: vector}, "params": {...},
                            "seed", "process", "statistic_params"} under
                            --format dict, or a CorrelationFeatures under
                            --format object,
        "feature_matrix":   (N, D) float array, statistic vectors concatenated
                            in sorted statistic-name order,
        "feature_order":    list[str], the statistic names in that order,
        "labels":           (N, P) float array of generating parameters,
        "label_names":      list[str] naming the P label columns,
        "params":           list[dict], the raw per-cloud parameters,
        "seeds":            list[int],
        "process":          str,
        "statistic_params": dict,
    }
With --format dict (default) the file reloads with numpy alone.

Examples
--------
    python scripts/processing/params/compute_pairwise_params.py
    python scripts/processing/params/compute_pairwise_params.py --n-samples 10000 --grid-size 128
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np

# --- make the cloudforger package importable when run as a plain script ------
# scripts/processing/params/<this file>  ->  parents[3] == project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]
_SRC = PROJECT_ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from cloudforger.core.cloud import PointCloud          # noqa: E402
from cloudforger.core.features import CorrelationFeatures  # noqa: E402
from cloudforger.core.region import Box                # noqa: E402
from cloudforger.stats.pair_dist import PairDistanceCDF  # noqa: E402


def build_statistics(n_samples: int, grid_size: int) -> list:
    """The statistics to compute. Add TripleAngleCDF (etc.) here later -- the
    rest of the script picks it up with no other changes."""
    return [
        PairDistanceCDF(n_samples=n_samples, grid_size=grid_size),
    ]


# --- record accessors --------------------------------------------------------
# Work whether a cloud was saved as a dict (default) or a PointCloud instance.

def to_pointcloud(cloud: Any) -> PointCloud:
    """Rebuild a PointCloud (with its Box region) from a dict record.

    The statistics need a PointCloud; PairDistanceCDF in particular requires the
    region, so it is reconstructed here rather than dropped.
    """
    if not isinstance(cloud, dict):
        return cloud
    region = None
    reg = cloud.get("region")
    if reg is not None:
        try:
            region = Box(low=np.asarray(reg["low"], dtype=float),
                         high=np.asarray(reg["high"], dtype=float))
        except Exception:
            region = None
    return PointCloud(
        points=np.asarray(cloud["points"]),
        generator_name=cloud.get("process", ""),
        generator_params=dict(cloud.get("params", {})),
        seed=cloud.get("seed"),
        region=region,
    )


def cloud_params(cloud: Any) -> dict[str, Any]:
    if isinstance(cloud, dict):
        return dict(cloud["params"])
    return dict(cloud.generator_params)


def cloud_seed(cloud: Any) -> Any:
    return cloud["seed"] if isinstance(cloud, dict) else cloud.seed


def load_clouds(path: Path) -> list:
    with open(path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict) and "clouds" in data:
        return data["clouds"]
    return data


def build_labels(clouds: list) -> tuple[np.ndarray, list[str], list[dict]]:
    params = [cloud_params(c) for c in clouds]
    label_names = list(params[0].keys())
    for p in params:
        if list(p.keys()) != label_names:
            raise ValueError("Clouds have inconsistent parameter keys; cannot stack labels.")
    labels = np.array([[p[k] for k in label_names] for p in params], dtype=float)
    return labels, label_names, params


def features_to_record(cf: CorrelationFeatures) -> dict[str, Any]:
    """Flatten CorrelationFeatures into a plain, picklable dict labelled by params."""
    return {
        "features": {k: np.asarray(v) for k, v in cf.features.items()},
        "params": dict(cf.generator_params),   # <-- label
        "seed": cf.seed,
        "process": cf.generator_name,
        "statistic_params": dict(cf.statistic_params),
    }


def compute_features(
    clouds_path: Path,
    out_path: Path,
    statistics: list,
    process: str,
    feature_format: str = "dict",
) -> None:
    clouds = load_clouds(clouds_path)
    n = len(clouds)
    labels, label_names, params = build_labels(clouds)
    seeds = [cloud_seed(c) for c in clouds]
    stat_params = {s.name: s.params for s in statistics}
    order = sorted(s.name for s in statistics)  # matches CorrelationFeatures.names()

    computed: list[CorrelationFeatures] = []
    rows: list[np.ndarray] = []
    print(f"  Computing features for {n} clouds {[s.name for s in statistics]} "
          f"from {clouds_path} ...")
    for i, cloud in enumerate(clouds):
        pc = to_pointcloud(cloud)
        vectors = {s.name: np.asarray(s.compute(pc)) for s in statistics}
        rows.append(np.concatenate([vectors[name] for name in order]))
        computed.append(CorrelationFeatures(
            features=vectors,
            generator_name=pc.generator_name,
            generator_params=pc.generator_params,
            seed=pc.seed,
            statistic_params=stat_params,
        ))
        print(f"\r    {i + 1}/{n}", end="", flush=True)
    print()

    feature_matrix = np.stack(rows)
    features = (
        [features_to_record(cf) for cf in computed]
        if feature_format == "dict"
        else computed
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(
            {
                "features": features,
                "feature_matrix": feature_matrix,
                "feature_order": order,
                "labels": labels,
                "label_names": label_names,
                "params": params,
                "seeds": seeds,
                "process": process,
                "statistic_params": stat_params,
            },
            f,
        )
    print(f"  Saved features for {len(computed)} clouds, "
          f"feature_matrix {feature_matrix.shape} \u2192 {out_path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--process", default="thomas",
        help="Process name; sets default in/out paths (default: thomas).",
    )
    parser.add_argument(
        "--clouds", type=Path, default=None,
        help="Input clouds .pkl (default: data/params/<process>/clouds.pkl).",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output features .pkl (default: data/params/<process>/features.pkl).",
    )
    parser.add_argument(
        "--format", default="dict", choices=("dict", "object"),
        help="dict: records labelled by 'params' (numpy-only to reload). "
             "object: CorrelationFeatures instances (needs cloudforger). "
             "Default: dict.",
    )
    parser.add_argument(
        "--n-samples", type=int, default=5000,
        help="Pairs sampled per cloud for the CDF statistic (default: 5000).",
    )
    parser.add_argument(
        "--grid-size", type=int, default=64,
        help="CDF grid resolution (default: 64).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    clouds_path = args.clouds or (
        PROJECT_ROOT / "data" / "params" / args.process / "clouds.pkl"
    )
    out_path = args.output or (
        PROJECT_ROOT / "data" / "params" / args.process / "features.pkl"
    )
    statistics = build_statistics(args.n_samples, args.grid_size)
    compute_features(clouds_path, out_path, statistics, args.process, args.format)
    print("\nDone.")


if __name__ == "__main__":
    main()