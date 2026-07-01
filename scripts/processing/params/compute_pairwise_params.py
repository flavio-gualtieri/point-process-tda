# scripts/processing/params/compute_pairwise_params.py

from __future__ import annotations

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

from cloudforger.core.cloud import PointCloud
from cloudforger.core.features import CorrelationFeatures
from cloudforger.core.region import Box
from cloudforger.stats.pair_dist import PairDistanceCDF


CONFIG = {
    "process": "thomas",
    "dimensions": [2],

    "splits": {
        "train_test": {
            "input_name": "clouds.pkl",
            "output_name": "features.pkl",
        },
        "adversarial": {
            "input_name": "adversarial_clouds.pkl",
            "output_name": "adversarial_features.pkl",
        },
    },

    "format": "dict",
    "pair_distance": {
        "n_samples": 5_000,
        "grid_size": 64,
    },
}


def normalize_dimensions(value: int | list[int]) -> list[int]:
    if isinstance(value, int):
        return [value]
    if isinstance(value, list) and all(isinstance(v, int) for v in value):
        return value
    raise TypeError("CONFIG['dimensions'] must be an int or a list of ints.")


def to_pointcloud(cloud: Any) -> PointCloud:
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
    labels = np.array([[p[k] for k in label_names] for p in params], dtype=float) # study this

    return labels, label_names, params


def features_to_record(cf: CorrelationFeatures) -> dict[str, Any]:
    return {
        "features": {k: np.asarray(v) for k, v in cf.features.items()},
        "params": dict(cf.generator_params),   # <-- label
        "seed": cf.seed,
        "process": cf.generator_name,
        "statistic_params": dict(cf.statistic_params),
    }


def compute_features(clouds_path: Path, out_path: Path,
                     statistics: list, process: str,
                     feature_format: str = "dict") -> None:
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


def main() -> None:
    process = CONFIG["process"]
    dimensions = normalize_dimensions(CONFIG["dimensions"])
    feature_format = CONFIG.get("format", "dict")

    if feature_format not in {"dict", "object"}:
        raise ValueError("CONFIG['format'] must be 'dict' or 'object'.")

    pair_cfg = CONFIG["pair_distance"]
    statistics = [PairDistanceCDF(n_samples=int(pair_cfg["n_samples"]), grid_size=int(pair_cfg["grid_size"]))]

    for dim in dimensions:
        base_dir = PROJECT_ROOT / "data" / "params" / f"{dim}d" / process

        for split_name, split_cfg in CONFIG["splits"].items():
            clouds_path = base_dir / split_cfg["input_name"]
            out_path = base_dir / split_cfg["output_name"]

            if not clouds_path.exists():
                print(f"[dim={dim}] Skipping {split_name}: missing {clouds_path}")
                continue

            print(f"[dim={dim}] Computing {split_name} features")
            compute_features(
                clouds_path=clouds_path,
                out_path=out_path,
                statistics=statistics,
                process=process,
                feature_format=feature_format,
            )

    print("\nDone.")


if __name__ == "__main__":
    main()