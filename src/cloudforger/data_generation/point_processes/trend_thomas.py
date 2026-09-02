# src/cloudforger/data_generation/point_processes/trend_thomas.py
"""Thomas process with a deterministic log-linear *parent-intensity trend*
and an optional anisotropic offspring kernel.

The parent (cluster-centre) intensity is inhomogeneous:

    rho_parent(u) = parent_intensity * exp( beta . (u - c) )

with c the window centre, so `parent_intensity` is the parent rate *at the
centre* and `beta` (one coefficient per axis) is the log-intensity gradient
-- the standard spatstat trend parametrisation, kppm(X ~ x + y, "Thomas").
Cluster centres concentrate where beta . u is large; the within-cluster law
(mean_offspring, offspring kernel) is unchanged and stays stationary.

This is the *parent* route to inhomogeneity. The offspring-retention route
-- thin offspring by a covariate field -- is the separate `inhom_thomas`
(InhomThomas). Both collapse to ThomasProcess when the trend is switched
off (beta = 0, cluster_aspect = 1).

Offspring kernel:
  cluster_aspect == 1  -> isotropic GaussianKernel(cluster_scale), identical
                          to ThomasProcess.
  cluster_aspect  > 1  -> AnisotropicGaussianKernel with
                            sigma_1 = cluster_scale * sqrt(cluster_aspect)
                            sigma_2 = cluster_scale / sqrt(cluster_aspect)
                          rotated by cluster_theta. The geometric-mean scale
                          sqrt(sigma_1 sigma_2) == cluster_scale is held
                          fixed, so a design reuses the isotropic
                          cluster_scale range unchanged and varies
                          cluster_aspect (>= 1) as a separate axis.

Identifiability note: parent_intensity (kappa_0) trades off against the
total count exactly as in the homogeneous case. Holding E[N] fixed across a
beta sweep -- so the count cannot leak the trend -- is a config-layer
choice (a `derived` parent_intensity in the design spec), deliberately not
baked in here.

beta is accepted either as a per-axis sequence (`beta=[b0, b1]`, the
canonical form) or, as a stopgap until the design spec samples vectors
(ROADMAP_TO_PUBLICATION.md sec 8.2), as scalars `beta_0`, `beta_1` that a
`random` design's per-key ranges can produce directly.
"""

from __future__ import annotations

import math

import numpy as np

from ...core.region import Box, Region
from .neyman_scott import NeymanScottProcess, poisson_counts
from .kernels import AnisotropicGaussianKernel, GaussianKernel


def _thin_loglinear_poisson(
    region: Box,
    rng: np.random.Generator,
    kappa0: float,
    beta: np.ndarray,
) -> np.ndarray:
    """Realisation of an inhomogeneous Poisson process with intensity
    kappa0 * exp(beta . (u - c)) over `region` (c = region centre), by
    Lewis-Shedler thinning of a homogeneous Poisson at the exact
    region-corner maximum rate.

    `region` is the edge-expanded box NeymanScottProcess samples parents in;
    Box.expanded pads symmetrically, so its centre is still the original
    window centre and `beta` keeps its meaning."""
    if not isinstance(region, Box):
        raise TypeError("trend_thomas requires a Box region")

    d = region.dimension
    if beta.shape != (d,):
        raise ValueError(
            f"beta must have length {d} (region dimension), got shape {beta.shape}"
        )

    if not np.any(beta):
        n = int(rng.poisson(kappa0 * region.volume))
        return region.sample_uniform(n, rng) if n else np.empty((0, d))

    centre = 0.5 * (region.low + region.high)
    half_extent = 0.5 * (region.high - region.low)
    # the linear map beta . (u - c) attains its max over the box at a corner
    eta_max = float(np.sum(np.abs(beta) * half_extent))
    lam_max = kappa0 * math.exp(eta_max)

    n = int(rng.poisson(lam_max * region.volume))
    if n == 0:
        return np.empty((0, d))
    cand = region.sample_uniform(n, rng)
    eta = (cand - centre) @ beta
    keep = rng.random(n) < np.exp(eta - eta_max)  # eta <= eta_max exactly, no clip
    return cand[keep]


class TrendThomasProcess(NeymanScottProcess):

    def __init__(
        self,
        parent_intensity: float,
        mean_offspring: float,
        cluster_scale: float,
        beta=None,
        beta_0: float | None = None,
        beta_1: float | None = None,
        cluster_aspect: float = 1.0,
        cluster_theta: float = 0.0,
        edge_buffer: float | None = None,
    ):
        if beta is None:
            if beta_0 is None and beta_1 is None:
                raise ValueError(
                    "provide either `beta` (a per-axis sequence) or scalar `beta_0`/`beta_1`"
                )
            beta = [0.0 if beta_0 is None else beta_0, 0.0 if beta_1 is None else beta_1]
        beta = np.asarray(beta, dtype=float).ravel()

        if parent_intensity <= 0:
            raise ValueError("parent_intensity must be positive")
        if mean_offspring <= 0:
            raise ValueError("mean_offspring must be positive")
        if cluster_scale <= 0:
            raise ValueError("cluster_scale must be positive")
        if cluster_aspect < 1.0:
            raise ValueError("cluster_aspect must be >= 1 (sigma_1 is the long axis)")
        if not np.all(np.isfinite(beta)):
            raise ValueError("beta must be finite")

        self._kappa0 = float(parent_intensity)
        self._beta = beta

        root = math.sqrt(cluster_aspect)
        if cluster_aspect == 1.0:
            kernel = GaussianKernel(cluster_scale)
        else:
            kernel = AnisotropicGaussianKernel(
                sigma_1=cluster_scale * root,
                sigma_2=cluster_scale / root,
                theta=cluster_theta,
            )

        def _parent_sampler(expanded: Region, rng: np.random.Generator) -> np.ndarray:
            return _thin_loglinear_poisson(expanded, rng, self._kappa0, self._beta)

        super().__init__(
            parent_intensity=parent_intensity,
            kernel=kernel,
            offspring_count_sampler=poisson_counts(mean_offspring),
            edge_buffer=edge_buffer,
            process_name="trend_thomas",
            param_dict={
                "mean_offspring": float(mean_offspring),
                "cluster_scale": float(cluster_scale),
                "cluster_aspect": float(cluster_aspect),
                "cluster_theta": float(cluster_theta),
                "cluster_sigma_1": float(cluster_scale * root),
                "cluster_sigma_2": float(cluster_scale / root),
                **{f"beta_{i}": float(b) for i, b in enumerate(beta)},
                "beta_norm": float(np.linalg.norm(beta)),
            },
            parent_sampler=_parent_sampler,
        )
