"""The F, G and J summary functions on the unit square W = [0,1]^2, reduced-sample correction.

F  empty-space (spherical contact) function: the distance from an arbitrary fixed location to the
   nearest point. Sees VOID structure, which K largely does not.
G  nearest-neighbour function: the distance from a typical point to its nearest other point. Sees
   the short-range clumping or inhibition that K averages over.
J  (1 - G) / (1 - F): 1 under CSR, < 1 for clustering, > 1 for inhibition, and -- the reason it is
   here -- NOT a function of the second-order structure alone, so it discriminates where K cannot.

Both F and G use the reduced-sample (border) correction, spatstat's `correction="rs"`: restrict to
the points / test locations further than r from the boundary, so nothing can be censored by the
window.

    G(r) = #{i : d_i <= r and b_i > r} / #{i : b_i > r}
    F(r) = #{u : e_u <= r and c_u > r} / #{u : c_u > r}

with d_i point i's nearest-neighbour distance, b_i its distance to the window edge, and (e_u, c_u)
the same two quantities for a test location u. Border correction over Kaplan-Meier because it is
exactly unbiased, has no tuning, and vectorizes -- the point here is a strong STANDARD baseline,
not a novel estimator.

The counts come from sorting rather than from an (N, m) mask, using

    #{d <= r AND b > r} = #{d <= r} - #{max(d, b) <= r}

so each function costs O(N log N + m log N) instead of O(N m).
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.spatial import cKDTree

# Test locations for F. 64x64 = 4096 interior locations is ~15x oversampled against this sweep's
# median n of 281 (range 44-961), and the sort-based counting makes the grid size nearly free.
F_GRID_SIZE = 64

# J is only defined while F(r) < 1. Past the largest radius where the denominator is numerically
# safe the estimate is held flat at its last valid value (spatstat truncates the plot there); the
# ratio is also clipped, because an unbounded spike entering a z-scored channel would swamp it.
J_MIN_DENOM = 0.05
J_CLIP = 5.0


def border(locations: np.ndarray) -> np.ndarray:
    """Distance from each row to the nearest face of the unit square."""
    return np.minimum(locations, 1.0 - locations).min(axis=1)


@lru_cache(maxsize=4)
def test_grid(grid_size: int = F_GRID_SIZE) -> tuple[np.ndarray, np.ndarray]:
    """F's test locations and their border distances -- identical for every pattern, so cached.

    An interior grid: `linspace` endpoints would sit exactly ON the boundary, where the border
    distance is 0 and the location is eligible at no radius at all.
    """
    axis = np.linspace(0.0, 1.0, grid_size + 2)[1:-1]
    mesh = np.stack(np.meshgrid(axis, axis, indexing="ij"), axis=-1).reshape(-1, 2)
    return mesh, border(mesh)


def _reduced_sample_cdf(dist: np.ndarray, edge: np.ndarray, radii: np.ndarray) -> np.ndarray:
    """Border-corrected CDF #{d <= r, b > r} / #{b > r} on `radii`.

    Once r exceeds every border distance the denominator is 0 and the estimator is undefined; the
    last well-defined value is held flat from there, keeping the channel monotone and finite.
    """
    out = np.zeros(len(radii))
    if len(dist) == 0:
        return out

    n_d = np.searchsorted(np.sort(dist), radii, side="right")                  # #{d <= r}
    n_b = np.searchsorted(np.sort(edge), radii, side="right")                  # #{b <= r}
    n_m = np.searchsorted(np.sort(np.maximum(dist, edge)), radii, side="right")  # #{max(d,b) <= r}

    denom, numer = len(dist) - n_b, n_d - n_m
    valid = denom > 0
    out[valid] = numer[valid] / denom[valid]
    if not valid.all():
        defined = np.flatnonzero(valid)
        out[~valid] = out[defined[-1]] if len(defined) else 0.0
    # Border correction is exactly unbiased but not monotone in finite samples (the eligible set
    # shrinks with r); enforce the CDF property so the channel is a genuine distribution function.
    return np.maximum.accumulate(out)


def g_function(points: np.ndarray, radii: np.ndarray) -> np.ndarray:
    """Border-corrected nearest-neighbour distance distribution G(r)."""
    if len(points) < 2:
        return np.zeros(len(radii))
    # k=2 because the first neighbour returned is the query point itself.
    nn = cKDTree(points).query(points, k=2)[0][:, 1]
    return _reduced_sample_cdf(nn, border(points), radii)


def f_function(points: np.ndarray, radii: np.ndarray, grid_size: int = F_GRID_SIZE) -> np.ndarray:
    """Border-corrected empty-space function F(r) on a regular test grid."""
    if len(points) == 0:
        return np.zeros(len(radii))
    mesh, edge = test_grid(grid_size)
    return _reduced_sample_cdf(cKDTree(points).query(mesh, k=1)[0], edge, radii)


def j_function(f: np.ndarray, g: np.ndarray) -> np.ndarray:
    """J(r) = (1 - G(r)) / (1 - F(r)), held flat where F is too close to 1.

    The r = 0 value and the post-truncation fill are both 1 -- "no evidence of departure from
    Poisson" -- which is the right neutral value for a z-scored channel.
    """
    denom = 1.0 - f
    valid = denom > J_MIN_DENOM
    out = np.ones(len(f))
    out[valid] = (1.0 - g[valid]) / denom[valid]
    if valid.any() and not valid.all():
        # F is non-decreasing, so the invalid region is a suffix; hold the last trustworthy value
        # rather than letting the ratio blow up as the denominator goes to 0.
        last = np.flatnonzero(valid)[-1]
        out[last + 1:] = out[last]
    return np.clip(out, 0.0, J_CLIP)
