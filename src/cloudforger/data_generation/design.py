# src/cloudforger/data_generation/design.py
"""Point-process design-space sampling: turning a design spec (grid or
random parameter-vector sampling, with an optional adversarial holdout)
into concrete PointCloud realizations. Promoted from
scripts/processing/params/pipeline_lib/clouds.py, generalized off its
per-dimension looping (this repo is 2D-only for now -- see RunConfig) and
using cloudforger.data_generation.point_processes.REGISTRY instead of a
duplicate local mapping."""

from __future__ import annotations

import math
import inspect
import itertools

import numpy as np

from dataclasses import dataclass
from typing import Any, Iterable, Iterator

from .point_processes import REGISTRY as PROCESS_REGISTRY
from ..core.cloud import PointCloud
from ..core.region import Box, Region

DEFAULT_ADVERSARIAL_SEED_OFFSET = 100_000

_SAFE = {"sqrt": math.sqrt, "log": math.log, "exp": math.exp, "pi": math.pi}


def apply_derived(params: dict, derived: dict | None) -> dict:
    out = dict(params)
    for name, expr in (derived or {}).items():
        out[name] = float(eval(expr, {"__builtins__": {}}, {**_SAFE, **out}))
    return out


def passes(params: dict, constraints: list[str] | None) -> bool:
    return all(bool(eval(c, {"__builtins__": {}}, {**_SAFE, **params}))
               for c in (constraints or []))


def default_region(dimension: int = 2) -> Region:
    return Box(low=np.zeros(dimension), high=np.ones(dimension))


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


def build_param_vectors(design: dict[str, Any], design_rng: np.random.Generator) -> list[dict[str, float]]:
    mode = design.get("mode", "grid")

    if mode == "grid":
        return list(iter_param_grid(design["grid"]))

    if mode == "random":
        random_cfg = design["random"]
        n = int(random_cfg["n_param_vectors"])
        ranges     = random_cfg["ranges"]
        derived    = random_cfg.get("derived")
        constraints = random_cfg.get("constraints")
        out, tries = [], 0
        while len(out) < n:
            tries += 1
            if tries > 200 * n:
                raise RuntimeError("constraints reject nearly everything; loosen them")
            p = apply_derived(sample_param_vector(ranges, design_rng), derived)
            if passes(p, constraints):
                out.append(p)
        return out

    raise ValueError(f"Unknown design.mode {mode!r}; use 'grid' or 'random'.")


def split_adversarial_vectors(
    param_vectors: list[dict[str, float]],
    adversarial: dict[str, Any],
    base_seed: int,
) -> tuple[list[dict[str, float]], list[dict[str, float]], list[int]]:
    if not adversarial.get("enabled", False):
        return param_vectors, [], []

    n_total = len(param_vectors)
    exact_n = adversarial.get("n_param_vectors")
    if exact_n is None:
        fraction = float(adversarial.get("fraction", 0.0))
        n_adv = int(round(fraction * n_total))
    else:
        n_adv = int(exact_n)

    if n_adv < 0 or n_adv >= n_total:
        raise ValueError(f"adversarial holdout size must be in [0, {n_total - 1}], got {n_adv}.")

    seed_offset = int(adversarial.get("seed_offset", DEFAULT_ADVERSARIAL_SEED_OFFSET))
    holdout_rng = np.random.default_rng(base_seed + seed_offset)
    adversarial_idx = set(int(i) for i in holdout_rng.choice(n_total, size=n_adv, replace=False))

    train_test_vectors = [p for i, p in enumerate(param_vectors) if i not in adversarial_idx]
    adversarial_vectors = [p for i, p in enumerate(param_vectors) if i in adversarial_idx]
    return train_test_vectors, adversarial_vectors, sorted(adversarial_idx)


def repeat_vectors(param_vectors: Iterable[dict[str, float]], reps: int) -> list[dict[str, float]]:
    return [dict(params) for params in param_vectors for _ in range(reps)]


def generate_clouds_for_design(
    process_name: str,
    region: Region,
    design: list[dict[str, float]],
    base_seed: int,
    n_hint: int = 0,
) -> list[PointCloud]:
    cls = PROCESS_REGISTRY.get(process_name)
    valid_params = set(inspect.signature(cls.__init__).parameters) - {"self"}
    clouds: list[PointCloud] = []

    for offset, params in enumerate(design):
        filtered = {k: v for k, v in params.items() if k in valid_params}
        clouds.append(cls(**filtered).sample(n=n_hint, region=region, seed=base_seed + offset))

    return clouds


def cloud_stats(clouds: list[PointCloud]) -> dict[str, int]:
    if not clouds:
        return {"n_clouds": 0, "total_points": 0, "min_points": 0, "max_points": 0}
    return {
        "n_clouds": len(clouds),
        "total_points": int(sum(c.n_points for c in clouds)),
        "min_points": int(min(c.n_points for c in clouds)),
        "max_points": int(max(c.n_points for c in clouds)),
    }


@dataclass
class CloudDesign:
    """Result of expanding a process config's design spec into concrete
    per-cloud parameter vectors, split into train/test vs. held-out
    adversarial, and repeated `reps` times each."""

    process_name: str
    region: Region
    train_test_vectors: list[dict[str, float]]
    adversarial_vectors: list[dict[str, float]]
    adversarial_indices: list[int]
    train_test_design: list[dict[str, float]]
    adversarial_design: list[dict[str, float]]
    base_seed: int
    adversarial_seed_offset: int
    n_hint: int = 0

    @classmethod
    def build(
        cls,
        process_name: str,
        seed: int,
        design: dict[str, Any],
        adversarial: dict[str, Any] | None = None,
        region: Region | None = None,
        n_hint: int = 0,
    ) -> "CloudDesign":
        adversarial = adversarial or {}
        region = region or default_region(2)
        reps = int(design.get("reps", 1))
        if reps < 1:
            raise ValueError("design.reps must be >= 1.")

        design_rng = np.random.default_rng(seed)
        param_vectors = build_param_vectors(design, design_rng)
        train_test_vectors, adversarial_vectors, adversarial_indices = split_adversarial_vectors(
            param_vectors, adversarial, seed
        )
        seed_offset = int(adversarial.get("seed_offset", DEFAULT_ADVERSARIAL_SEED_OFFSET))

        return cls(
            process_name=process_name,
            region=region,
            train_test_vectors=train_test_vectors,
            adversarial_vectors=adversarial_vectors,
            adversarial_indices=adversarial_indices,
            train_test_design=repeat_vectors(train_test_vectors, reps),
            adversarial_design=repeat_vectors(adversarial_vectors, reps),
            base_seed=seed,
            adversarial_seed_offset=seed_offset,
            n_hint=n_hint,
        )

    def generate(self) -> tuple[list[PointCloud], list[PointCloud]]:
        train_test = generate_clouds_for_design(
            self.process_name, self.region, self.train_test_design, self.base_seed, self.n_hint,
        )
        adversarial = generate_clouds_for_design(
            self.process_name, self.region, self.adversarial_design,
            self.base_seed + self.adversarial_seed_offset, self.n_hint,
        )
        return train_test, adversarial

    def manifest(self, train_test_clouds: list[PointCloud], adversarial_clouds: list[PointCloud]) -> dict[str, Any]:
        return {
            "process": self.process_name,
            "seed": self.base_seed,
            "n_train_test_param_vectors": len(self.train_test_vectors),
            "n_adversarial_param_vectors": len(self.adversarial_vectors),
            "adversarial_param_indices": self.adversarial_indices,
            "adversarial_params": self.adversarial_vectors,
            "train_test_stats": cloud_stats(train_test_clouds),
            "adversarial_stats": cloud_stats(adversarial_clouds),
        }
