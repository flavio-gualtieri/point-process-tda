# src/cloudforger/generation/samplers.py
"""Model parameters + a case's PATTERN stream -> points in W = [0,1]^2.

Thin adapters over the existing samplers in data_generation/point_processes,
which are reused unchanged. This module is the only place that knows the
mapping from DV3 manifest names (kappa, sigma, ...) to constructor arguments.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from ..core.region import Box
from ..data_generation.point_processes import PoissonProcess, ThomasProcess
from .prior import PriorDraw

WINDOW = Box(low=np.zeros(2), high=np.ones(2))


@dataclass(frozen=True)
class Simulated:
    points: np.ndarray           # (n, 2) float64
    diagnostics: dict[str, Any]  # manifest "sampler" group


def _poisson(draw: PriorDraw, rng: np.random.Generator) -> tuple[np.ndarray, dict[str, Any]]:
    # N ~ Poisson(nbar |W|), then N iid uniform points in W. Exact.
    cloud = PoissonProcess(intensity=draw.nbar).sample(WINDOW, rng=rng)
    return cloud.points, {"sampler": "exact"}


def _thomas(draw: PriorDraw, rng: np.random.Generator) -> tuple[np.ndarray, dict[str, Any]]:
    # Parents ~ Poisson(kappa) on W expanded by b = 3.719 sigma (the kernel's
    # 1e-4 tail quantile), Poisson(mu) children each with N(0, sigma^2 I)
    # offsets, keep children in W. Exact up to a < 1e-5 truncation.
    proc = ThomasProcess(
        parent_intensity=draw.model["kappa"],
        mean_offspring=draw.design["mu"],
        cluster_scale=draw.model["sigma"],
    )
    cloud = proc.sample(WINDOW, rng=rng)
    return cloud.points, {"sampler": "exact", "buffer": proc.edge_buffer, "n_parents": proc.last_n_parents}


SAMPLERS: dict[str, Callable[[PriorDraw, np.random.Generator], tuple[np.ndarray, dict[str, Any]]]] = {
    "poisson": _poisson,
    "thomas": _thomas,
}


def points_sha1(points: np.ndarray) -> str:
    return hashlib.sha1(np.ascontiguousarray(points, dtype="<f8").tobytes()).hexdigest()


def simulate(family: str, draw: PriorDraw, rng: np.random.Generator) -> Simulated:
    if family not in SAMPLERS:
        raise NotImplementedError(f"no sampler implemented for {family!r}")
    t0 = time.perf_counter()
    points, diag = SAMPLERS[family](draw, rng)
    wall_s = time.perf_counter() - t0
    points = np.ascontiguousarray(points, dtype=np.float64).reshape(-1, 2)
    return Simulated(points, {**diag, "wall_s": wall_s, "sha1": points_sha1(points)})
