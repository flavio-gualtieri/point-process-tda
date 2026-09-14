# src/cloudforger/generation/samplers.py
"""Model parameters + a case's PATTERN stream -> points in W = [0,1]^2.

Thin adapters over the samplers in data_generation/point_processes. This
module is the only place that knows the mapping from DV3 manifest names
(kappa, sigma, ...) to constructor arguments.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from ..core.region import Box
from ..data_generation.point_processes import (
    CirculantLGCPProcess, MaternHardCoreProcess, NestedThomasProcess, PoissonProcess, ThomasProcess,
)
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


def _nested(draw: PriorDraw, rng: np.random.Generator) -> tuple[np.ndarray, dict[str, Any]]:
    # Meta-parents ~ Poisson(kappa) on W (+) (b1 + b2); Poisson(mu1) parents at
    # N(0, sigma1^2), kept on W (+) b2; Poisson(mu2) children at N(0, sigma2^2),
    # kept on W. b_i = 3.719 sigma_i. Exact up to two < 1e-5 truncations.
    proc = NestedThomasProcess(
        meta_parent_intensity=draw.model["kappa"],
        meta_offspring=draw.design["mu1"],
        meta_cluster_scale=draw.model["sigma1"],
        mean_offspring=draw.design["mu2"],
        cluster_scale=draw.model["sigma"],
    )
    cloud = proc.sample(WINDOW, rng=rng)
    return cloud.points, {"sampler": "exact", "buffer": proc.edge_buffer + proc.meta_process.edge_buffer,
                          "n_parents": proc.last_n_parents}


def _matern2(draw: PriorDraw, rng: np.random.Generator) -> tuple[np.ndarray, dict[str, Any]]:
    # Primaries ~ Poisson(lam_p) on W (+) R with U(0,1) marks, type II thinning, crop to W. Exact.
    # n_parents records the number of primary (candidate) points.
    proc = MaternHardCoreProcess(parent_intensity=draw.model["lam_p"], hardcore_radius=draw.model["R"])
    cloud = proc.sample(WINDOW, rng=rng)
    return cloud.points, {"sampler": "exact", "buffer": draw.model["R"], "n_parents": proc.last_n_primary}


def _lgcp(draw: PriorDraw, rng: np.random.Generator) -> tuple[np.ndarray, dict[str, Any]]:
    # Field by circulant embedding on the plan's M x M grid, Poisson cell counts,
    # uniform locations within cells. Exact for the grid LGCP.
    proc = CirculantLGCPProcess(
        mu=draw.model["mu_log"], sigma2=draw.design["sigma2"], s=draw.model["s_abs"],
        grid_M=draw.numerics["grid_M"],
    )
    cloud = proc.sample(WINDOW, rng=rng)
    return cloud.points, {"sampler": "exact", "pad_P": proc.last_pad_P, "min_eig": proc.last_min_eig}


SAMPLERS: dict[str, Callable[[PriorDraw, np.random.Generator], tuple[np.ndarray, dict[str, Any]]]] = {
    "poisson": _poisson,
    "thomas": _thomas,
    "nested": _nested,
    "matern2": _matern2,
    "lgcp": _lgcp,
}


def points_sha1(points: np.ndarray) -> str:
    return hashlib.sha1(np.ascontiguousarray(points, dtype="<f8").tobytes()).hexdigest()


def simulate(family: str, draw: PriorDraw, rng: np.random.Generator, n_min: int = 0,
             max_tries: int = 10_000) -> Simulated:
    """One pattern from the model conditioned on n >= n_min: redraw from the
    same stream until it holds (an exact sampler of the conditional law)."""
    if family not in SAMPLERS:
        raise NotImplementedError(f"no sampler implemented for {family!r}")
    t0 = time.perf_counter()
    for tries in range(1, max_tries + 1):
        points, diag = SAMPLERS[family](draw, rng)
        if len(points) >= n_min:
            break
    else:
        raise RuntimeError(f"{family}: no pattern with n >= {n_min} in {max_tries} draws")
    wall_s = time.perf_counter() - t0
    points = np.ascontiguousarray(points, dtype=np.float64).reshape(-1, 2)
    return Simulated(points, {**diag, "pattern_tries": tries, "wall_s": wall_s, "sha1": points_sha1(points)})
