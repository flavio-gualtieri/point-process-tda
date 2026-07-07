# scripts/processing/params/pipeline_lib/clouds.py
"""Stage: generate_clouds — cloud-design expansion and point-cloud sampling."""

from __future__ import annotations

import inspect
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
import yaml

from cloudforger.core.base import PointProcess
from cloudforger.core.cloud import PointCloud
from cloudforger.core.region import Box, Region
from cloudforger.processes.matern import MaternHardCoreProcess
from cloudforger.processes.poisson import PoissonProcess
from cloudforger.processes.thomas import ThomasProcess
from cloudforger.processes.inhom_thomas import InhomThomas

from pipeline_lib.io import dump_pickle
from pipeline_lib.records import cloud_to_record

PROCESS_REGISTRY: dict[str, type[PointProcess]] = {
    "thomas": ThomasProcess,
    "inhom_thomas": InhomThomas,
    "matern": MaternHardCoreProcess,
    "poisson": PoissonProcess,
}


@dataclass(frozen=True)
class CloudDesignBundle:
    config: dict[str, Any]
    train_test_vectors: list[dict[str, float]]
    adversarial_vectors: list[dict[str, float]]
    adversarial_indices: list[int]
    train_test_design: list[dict[str, float]]
    adversarial_design: list[dict[str, float]]
    base_seed: int
    output_format: str
    n_hint: int
    adversarial_seed_offset: int
    dimension_seed_stride: int


def default_region(dimension: int) -> Region:
    return Box(low=np.zeros(dimension), high=np.ones(dimension))


def build_region(config: dict[str, Any], dimension: int) -> Region:
    region_cfg = config.get("region")
    if region_cfg is None:
        return Box(low=np.zeros(dimension), high=np.ones(dimension))

    low = np.asarray(region_cfg["low"], dtype=float)
    high = np.asarray(region_cfg["high"], dtype=float)

    if low.shape != (dimension,) or high.shape != (dimension,):
        raise ValueError("region.low and region.high must match dimension.")

    return Box(low=low, high=high)


def expand_grid_axis(spec: Any) -> list[float]:
    if isinstance(spec, list):
        return [float(v) for v in spec]

    if not isinstance(spec, dict):
        raise ValueError(f"Grid axis must be a list or mapping, got {spec!r}.")

    if "values" in spec:
        return [float(v) for v in spec["values"]]

    kind = spec.get("kind")
    low = float(spec["low"])
    high = float(spec["high"])
    num = int(spec["num"])
    if num < 1:
        raise ValueError(f"Grid axis num must be >= 1, got {num}.")

    if kind == "linspace":
        return [float(v) for v in np.linspace(low, high, num)]
    if kind == "logspace":
        if low <= 0 or high <= 0:
            raise ValueError("logspace grid bounds must be positive.")
        return [float(v) for v in np.exp(np.linspace(np.log(low), np.log(high), num))]

    raise ValueError(f"Unknown grid kind {kind!r}; use values, linspace, or logspace.")


def iter_param_grid(grid: dict[str, Any]) -> Iterator[dict[str, float]]:
    keys = list(grid)
    axes = [expand_grid_axis(grid[k]) for k in keys]
    for combo in itertools.product(*axes):
        yield dict(zip(keys, combo))


def sample_param_vector(
    ranges: dict[str, dict[str, Any]],
    rng: np.random.Generator,
) -> dict[str, float]:
    params: dict[str, float] = {}

    for key, spec in ranges.items():
        low = float(spec["low"])
        high = float(spec["high"])
        scale = spec.get("scale", "linear")

        if scale == "log":
            if low <= 0 or high <= 0:
                raise ValueError(f"Log-scaled parameter {key!r} must have positive bounds.")
            value = np.exp(rng.uniform(np.log(low), np.log(high)))
        elif scale == "linear":
            value = rng.uniform(low, high)
        else:
            raise ValueError(f"Unknown scale {scale!r} for parameter {key!r}.")

        params[key] = float(value)

    return params


def build_param_vectors(
    config: dict[str, Any],
    design_rng: np.random.Generator,
) -> list[dict[str, float]]:
    design = config["design"]
    mode = design.get("mode", "grid")

    if mode == "grid":
        return list(iter_param_grid(design["grid"]))

    if mode == "random":
        random_cfg = design["random"]
        n_param_vectors = int(random_cfg["n_param_vectors"])
        ranges = random_cfg["ranges"]
        return [sample_param_vector(ranges, design_rng) for _ in range(n_param_vectors)]

    raise ValueError(f"Unknown design.mode {mode!r}; use 'grid' or 'random'.")


def split_adversarial_vectors(
    param_vectors: list[dict[str, float]],
    config: dict[str, Any],
    base_seed: int,
) -> tuple[list[dict[str, float]], list[dict[str, float]], list[int]]:
    adv_cfg = config.get("adversarial", {})
    if not adv_cfg.get("enabled", False):
        return param_vectors, [], []

    n_total = len(param_vectors)
    exact_n = adv_cfg.get("n_param_vectors")
    if exact_n is None:
        fraction = float(adv_cfg.get("fraction", 0.0))
        n_adv = int(round(fraction * n_total))
    else:
        n_adv = int(exact_n)

    if n_adv < 0 or n_adv >= n_total:
        raise ValueError(
            f"adversarial holdout size must be in [0, {n_total - 1}], got {n_adv}."
        )

    seed_offset = int(adv_cfg.get("seed_offset", 100_000))
    holdout_rng = np.random.default_rng(base_seed + seed_offset)
    adversarial_idx = set(int(i) for i in holdout_rng.choice(n_total, size=n_adv, replace=False))

    train_test_vectors = [p for i, p in enumerate(param_vectors) if i not in adversarial_idx]
    adversarial_vectors = [p for i, p in enumerate(param_vectors) if i in adversarial_idx]
    return train_test_vectors, adversarial_vectors, sorted(adversarial_idx)


def repeat_vectors(param_vectors: Iterable[dict[str, float]], reps: int) -> list[dict[str, float]]:
    return [dict(params) for params in param_vectors for _ in range(reps)]


def save_clouds(path: Path, clouds: list[PointCloud], output_format: str) -> None:
    payload = [cloud_to_record(c) for c in clouds] if output_format == "dict" else clouds
    dump_pickle(path, payload)


def cloud_stats(clouds: list[PointCloud]) -> dict[str, int]:
    if not clouds:
        return {"n_clouds": 0, "total_points": 0, "min_points": 0, "max_points": 0}
    return {
        "n_clouds": len(clouds),
        "total_points": int(sum(c.n_points for c in clouds)),
        "min_points": int(min(c.n_points for c in clouds)),
        "max_points": int(max(c.n_points for c in clouds)),
    }


def generate_clouds_for_design(
    process_name: str,
    region: Region,
    design: list[dict[str, float]],
    base_seed: int,
    n_hint: int = 0,
) -> list[PointCloud]:
    if process_name not in PROCESS_REGISTRY:
        raise ValueError(f"Unknown process {process_name!r}. Available: {sorted(PROCESS_REGISTRY)}")

    cls = PROCESS_REGISTRY[process_name]
    valid_params = set(inspect.signature(cls.__init__).parameters) - {"self"}
    clouds: list[PointCloud] = []

    for offset, params in enumerate(design):
        filtered = {k: v for k, v in params.items() if k in valid_params}
        clouds.append(cls(**filtered).sample(n=n_hint, region=region, seed=base_seed + offset))

    return clouds


def write_cloud_manifest(
    path: Path,
    config: dict[str, Any],
    train_test_vectors: list[dict[str, float]],
    adversarial_vectors: list[dict[str, float]],
    adversarial_indices: list[int],
    train_test_clouds: list[PointCloud],
    adversarial_clouds: list[PointCloud],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "config": config,
        "n_train_test_param_vectors": len(train_test_vectors),
        "n_adversarial_param_vectors": len(adversarial_vectors),
        "adversarial_param_indices": adversarial_indices,
        "adversarial_params": adversarial_vectors,
        "train_test_stats": cloud_stats(train_test_clouds),
        "adversarial_stats": cloud_stats(adversarial_clouds),
    }
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(manifest, f, sort_keys=False)
