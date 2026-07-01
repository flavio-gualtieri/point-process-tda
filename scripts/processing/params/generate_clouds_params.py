# scripts/processing/params/generate_clouds_params.py

from __future__ import annotations

import itertools
import pickle
import sys
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

import numpy as np
import yaml

# scripts/processing/params/<this file> -> parents[3] == project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]
_SRC = PROJECT_ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from cloudforger.core.base import PointProcess
from cloudforger.core.cloud import PointCloud
from cloudforger.core.region import Box, Region
from cloudforger.processes.thomas import ThomasProcess

CONFIG_PATH = PROJECT_ROOT / "configs" / "params" / "cloud_generation.yaml"

ProcessBuilder = Callable[[dict[str, Any]], PointProcess]

PROCESS_REGISTRY: dict[str, dict[str, Any]] = {
    "thomas": {
        "build": lambda p: ThomasProcess(**p),
    },
}


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Config {path} must contain a YAML mapping.")
    return config


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def default_region(dimension: int) -> Region:
    """Unit box [0, 1]^d."""
    return Box(low=np.zeros(dimension), high=np.ones(dimension))


def normalize_dimensions(value: int | list[int]) -> list[int]:
    if isinstance(value, int):
        return [value]

    if isinstance(value, list) and all(isinstance(v, int) for v in value):
        return value

    raise TypeError(
        "`dimension` must be either an int, e.g. 2, "
        "or a list of ints, e.g. [2, 3, 5]."
        )

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


def sample_param_vector(ranges: dict[str, dict[str, Any]], rng: np.random.Generator) -> dict[str, float]:
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


def build_param_vectors(config: dict[str, Any], design_rng: np.random.Generator) -> list[dict[str, float]]:
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


def split_adversarial_vectors(param_vectors: list[dict[str, float]], config: dict[str, Any],
                              base_seed: int) -> tuple[list[dict[str, float]], list[dict[str, float]], list[int]]:
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

    holdout_rng = np.random.default_rng(base_seed + int(adv_cfg.get("seed_offset", 100000)))
    adversarial_idx = set(int(i) for i in holdout_rng.choice(n_total, size=n_adv, replace=False))

    train_test_vectors = [p for i, p in enumerate(param_vectors) if i not in adversarial_idx]
    adversarial_vectors = [p for i, p in enumerate(param_vectors) if i in adversarial_idx]
    return train_test_vectors, adversarial_vectors, sorted(adversarial_idx)


def repeat_vectors(param_vectors: Iterable[dict[str, float]], reps: int) -> list[dict[str, float]]:
    return [dict(params) for params in param_vectors for _ in range(reps)]


def cloud_to_record(cloud: PointCloud) -> dict[str, Any]:
    record: dict[str, Any] = {
        "points": cloud.points,
        "params": dict(cloud.generator_params),
        "process": cloud.generator_name,
        "seed": cloud.seed,
        "n_points": cloud.n_points,
        "dimension": cloud.dimension,
    }
    region = cloud.region
    if isinstance(region, Box):
        record["region"] = {"low": region.low, "high": region.high}
    return record


def generate_clouds_for_design(process_name: str, region: Region, design: list[dict[str, float]],
                               base_seed: int, n_hint: int = 0) -> list[PointCloud]:
    if process_name not in PROCESS_REGISTRY:
        raise ValueError(
            f"Unknown process {process_name!r}. Available: {sorted(PROCESS_REGISTRY)}"
        )

    build: ProcessBuilder = PROCESS_REGISTRY[process_name]["build"]
    clouds: list[PointCloud] = []
    for offset, params in enumerate(design):
        process = build(params)
        clouds.append(process.sample(n=n_hint, region=region, seed=base_seed + offset))
    return clouds


def save_payload(path: Path, clouds: list[PointCloud], output_format: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [cloud_to_record(c) for c in clouds] if output_format == "dict" else clouds
    with open(path, "wb") as f:
        pickle.dump(payload, f)


def cloud_stats(clouds: list[PointCloud]) -> dict[str, int | float]:
    if not clouds:
        return {"n_clouds": 0, "total_points": 0, "min_points": 0, "max_points": 0}
    return {
        "n_clouds": len(clouds),
        "total_points": int(sum(c.n_points for c in clouds)),
        "min_points": int(min(c.n_points for c in clouds)),
        "max_points": int(max(c.n_points for c in clouds)),
    }


def write_manifest(path: Path, config: dict[str, Any], train_test_vectors: list[dict[str, float]],
                   adversarial_vectors: list[dict[str, float]], adversarial_indices: list[int],
                   train_test_clouds: list[PointCloud], adversarial_clouds: list[PointCloud]) -> None:
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


def main() -> None:
    config = load_config()

    process_name = config.get("process", "thomas")
    dimensions = normalize_dimensions(config.get("dimension", 2))
    base_seed = int(config.get("seed", 0))
    output_format = config.get("format", "dict")
    n_hint = int(config.get("n_hint", 0))
    reps = int(config["design"].get("reps", 1))

    if output_format not in {"dict", "object"}:
        raise ValueError("format must be 'dict' or 'object'.")
    if reps < 1:
        raise ValueError("design.reps must be >= 1.")

    # Build and split the parameter vectors once, so the same held-out
    # parameter combinations are used for every dimension.
    design_rng = np.random.default_rng(base_seed)
    param_vectors = build_param_vectors(config, design_rng)
    train_test_vectors, adversarial_vectors, adversarial_indices = split_adversarial_vectors(
        param_vectors, config, base_seed
    )

    train_test_design = repeat_vectors(train_test_vectors, reps)
    adversarial_design = repeat_vectors(adversarial_vectors, reps)

    adversarial_seed_offset = int(
        config.get("adversarial", {}).get("seed_offset", 100_000)
    )

    dimension_seed_stride = int(config.get("dimension_seed_stride", 1_000_937_000))

    for dim_index, dim in enumerate(dimensions):
        dim_seed = base_seed + dim_index * dimension_seed_stride

        region = default_region(dim)

        train_test_clouds = generate_clouds_for_design(
            process_name=process_name,
            region=region,
            design=train_test_design,
            base_seed=dim_seed,
            n_hint=n_hint,
        )

        adversarial_clouds = generate_clouds_for_design(
            process_name=process_name,
            region=region,
            design=adversarial_design,
            base_seed=dim_seed + adversarial_seed_offset,
            n_hint=n_hint,
        )

        train_test_output = resolve_path(
            f"data/params/{dim}d/{process_name}/clouds.pkl"
        )
        adversarial_output = resolve_path(
            f"data/params/{dim}d/{process_name}/adversarial_clouds.pkl"
        )
        manifest_output = resolve_path(
            f"data/params/{dim}d/{process_name}/cloud_generation_manifest.yaml"
        )

        save_payload(train_test_output, train_test_clouds, output_format)
        save_payload(adversarial_output, adversarial_clouds, output_format)

        manifest_config = dict(config)
        manifest_config["dimension"] = dim
        manifest_config["dimension_seed"] = dim_seed

        write_manifest(
            manifest_output,
            manifest_config,
            train_test_vectors,
            adversarial_vectors,
            adversarial_indices,
            train_test_clouds,
            adversarial_clouds,
        )

        train_stats = cloud_stats(train_test_clouds)
        adv_stats = cloud_stats(adversarial_clouds)

        print(
            f"[dim={dim}] Generated {train_stats['n_clouds']} train/test "
            f"'{process_name}' clouds from {len(train_test_vectors)} distinct "
            f"parameter vectors x {reps} rep(s)."
        )
        print(
            f"[dim={dim}] Generated {adv_stats['n_clouds']} adversarial "
            f"'{process_name}' clouds from {len(adversarial_vectors)} held-out "
            f"parameter vectors x {reps} rep(s)."
        )
        print(
            f"[dim={dim}] Train/test points: total {train_stats['total_points']}, "
            f"min {train_stats['min_points']}, max {train_stats['max_points']}."
        )
        print(
            f"[dim={dim}] Adversarial points: total {adv_stats['total_points']}, "
            f"min {adv_stats['min_points']}, max {adv_stats['max_points']}."
        )
        print(f"[dim={dim}] Saved train/test to {train_test_output}")
        print(f"[dim={dim}] Saved adversarial to {adversarial_output}")
        print(f"[dim={dim}] Saved manifest to {manifest_output}")


if __name__ == "__main__":
    main()
