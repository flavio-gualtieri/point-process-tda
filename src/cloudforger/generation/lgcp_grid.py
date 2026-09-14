# src/cloudforger/generation/lgcp_grid.py
"""LGCP grid resolution (generation.tex, LGCP "Resolution criterion").

The grid LGCP (intensity constant on each of M x M cells, D = 1/M) has pair
correlation, averaged over the grid's position,

    gbar_D(h) = sum_{k1,k2} p(k1 | h1) p(k2 | h2) exp(C(D |k|)),

where k_i in {floor(h_i/D), floor(h_i/D) + 1} with probabilities 1 - frac and
frac. That is the bilinear interpolation of exp(C) sampled on the lattice D Z^2.
M is accepted if the discretisation bias of K is invisible to the statistics:

    sup_{r in R} |K_D(r) - K(r)| <= tol * s0_K(r; nbar),    s0_K = 2 pi r s0,

with s0 the null s.d. of L-hat - r at nbar (nulls.py) and tol = 0.1. The
chosen M is the smallest power of two >= M_min that passes. It is a
deterministic function of (sigma2, s, nbar), computed per case by the plan.
The bias integral uses a midpoint rule on a D/4 grid over one quadrant; K and
K_D share the grid, so its pixelation of the disc cancels in the difference.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np

from .nulls import L_RMIN, R_GRID, s0_at

_SUB = 4  # quadrature points per cell side


@lru_cache(maxsize=8)
def _quadrature(M: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-axis midpoints h, interpolation matrix A (h -> lattice), and the
    R_GRID bin of every |h| (len(R_GRID) = outside the largest radius)."""
    D = 1.0 / M
    dh = D / _SUB
    h = (np.arange(int(np.ceil(R_GRID[-1] / dh))) + 0.5) * dh
    f = np.floor(h / D).astype(int)
    t = h / D - f
    kmax = int(f.max()) + 2
    A = np.zeros((len(h), kmax))
    A[np.arange(len(h)), f] = 1.0 - t
    A[np.arange(len(h)), f + 1] = t
    radius = np.hypot(h[:, None], h[None, :])
    bins = np.searchsorted(R_GRID, radius, side="left")
    return h, A, radius, bins


def grid_bias(sigma2: float, s: float, M: int) -> np.ndarray:
    """K_D(r) - K(r) on R_GRID for the exponential covariance sigma2 exp(-r/s)."""
    D = 1.0 / M
    h, A, radius, bins = _quadrature(M)
    k = np.arange(A.shape[1])
    lattice = np.exp(sigma2 * np.exp(-D * np.hypot(k[:, None], k[None, :]) / s))
    gbar = A @ lattice @ A.T
    diff = (gbar - np.exp(sigma2 * np.exp(-radius / s))) * (4.0 * (D / _SUB) ** 2)
    per_bin = np.bincount(bins.ravel(), weights=diff.ravel(), minlength=len(R_GRID) + 1)
    return np.cumsum(per_bin[:len(R_GRID)])


def choose_grid_M(sigma2: float, s: float, nbar: float, tabs: dict[str, Any],
                  tol: float = 0.1, M_min: int = 256, M_max: int = 4096) -> tuple[int, float]:
    """(M, worst |bias| / (tol s0_K)) for the smallest passing power of two."""
    sel = R_GRID >= L_RMIN
    s0K = 2.0 * np.pi * R_GRID[sel] * s0_at(nbar, "L", tabs)[sel]
    M = M_min
    while M <= M_max:
        ratio = float(np.max(np.abs(grid_bias(sigma2, s, M)[sel]) / (tol * s0K)))
        if ratio <= 1.0:
            return M, ratio
        M *= 2
    raise RuntimeError(f"no grid up to M={M_max} passes for sigma2={sigma2}, s={s}, nbar={nbar}")
