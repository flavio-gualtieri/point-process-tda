# src/cloudforger/baselines/summstats.py
"""The F, G and J summary functions -- the model-discrimination companions to
Ripley's K/L that `vihrs.py` already implements.

WHY THIS EXISTS. The vihrs baseline consumes exactly one summary function,
the isotropic-corrected `L(r) - r`, plus `n(x)`. Reporting a topological
classifier's margin against that alone invites the reading "L(r) is an
impoverished feature set" rather than "topology carries extra information".
The standard spatial-statistics answer to *model discrimination* (as opposed
to parameter estimation) is not K alone but K together with:

  * F -- the empty-space (spherical contact) function: distribution of the
    distance from an arbitrary fixed location to the nearest point. Sees
    VOID structure, which K largely does not.
  * G -- the nearest-neighbour distance function: distribution of the
    distance from a typical point to its nearest other point. Sees the
    short-range clumping/inhibition K averages over.
  * J = (1 - G) / (1 - F) -- Van Lieshout & Baddeley's classic discrimination
    statistic. J == 1 for Poisson, J < 1 for clustering, J > 1 for inhibition,
    and crucially it is NOT a function of the second-order structure alone.

Feeding {L, F, G, J} to the *same* 1-D CNN vihrs already uses turns that
baseline into "the union of the standard summary functions", which is the
comparison a spatial statistician would actually make.

ESTIMATORS. Both F and G use the reduced-sample (border) correction, which
is spatstat's `correction="rs"`: restrict to the sample points / test
locations whose distance to the window boundary exceeds r, so no observation
can be censored by the window.

    G(r) = #{i : d_i <= r and b_i > r} / #{i : b_i > r}
    F(r) = #{u : e_u <= r and c_u > r} / #{u : c_u > r}

with d_i the nearest-neighbour distance of point i, b_i its distance to the
window edge, and (e_u, c_u) the same two quantities for a test location u on
a regular grid. Border correction is chosen over Kaplan-Meier because it is
exactly unbiased, has no tuning, and is trivially vectorized -- and because
the point here is a strong *standard* baseline, not a novel estimator.

The counts are evaluated by sorting rather than by broadcasting an (N, m)
mask, using

    #{d <= r AND b > r} = #{d <= r} - #{max(d, b) <= r}

so each function costs O(N log N + m log N) instead of O(N * m). That keeps
the whole four-channel featurization comparable in cost to the existing
O(n^2) L(r) computation it sits next to, rather than dominating it.

J is only defined while F(r) < 1. Beyond the largest radius where the
denominator is numerically safe the estimate is held flat at its last valid
value (spatstat truncates the plot there instead); the ratio is also clipped,
because an unbounded spike entering a z-scored CNN channel would swamp it.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial import cKDTree

# Test-location grid for F. 64x64 = 4096 interior locations on the unit
# square is ~1 test point per 1.5e-4 of area, i.e. dense relative to the
# 150-800 points these designs generate, and the sort-based counting below
# makes the grid size nearly free in the radius loop.
F_GRID_SIZE = 64

# J is held flat once (1 - F) drops below this; see the module docstring.
J_MIN_DENOM = 0.05
J_CLIP = 5.0


def _border_distance(locations: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    """Distance from each row of `locations` to the nearest window face."""
    return np.minimum(locations - low, high - locations).min(axis=1)


def _reduced_sample_cdf(
    dist: np.ndarray, border: np.ndarray, r_grid: np.ndarray
) -> np.ndarray:
    """Border-corrected CDF #{d_i <= r, b_i > r} / #{b_i > r} on `r_grid`.

    Uses  #{d <= r AND b > r} = #{d <= r} - #{max(d, b) <= r}  so all three
    counts come from `searchsorted` on sorted copies, making this O(N log N)
    rather than O(N * len(r_grid)).

    Once r exceeds every border distance the denominator is 0 and the
    estimator is undefined; the last well-defined value is held flat from
    there on, which keeps the channel monotone and finite for the CNN.
    """
    n = len(dist)
    out = np.zeros(len(r_grid), dtype=np.float64)
    if n == 0:
        return out

    d_sorted = np.sort(dist)
    b_sorted = np.sort(border)
    m_sorted = np.sort(np.maximum(dist, border))

    n_d_le = np.searchsorted(d_sorted, r_grid, side="right")      # #{d <= r}
    n_b_le = np.searchsorted(b_sorted, r_grid, side="right")      # #{b <= r}
    n_m_le = np.searchsorted(m_sorted, r_grid, side="right")      # #{max(d,b) <= r}

    denom = n - n_b_le                                            # #{b > r}
    numer = n_d_le - n_m_le                                       # #{d <= r, b > r}

    valid = denom > 0
    out[valid] = numer[valid] / denom[valid]
    if not valid.all():
        # r beyond every border distance: hold the last defined value (0 if
        # the window is degenerate and nothing was ever eligible).
        last = np.flatnonzero(valid)
        out[~valid] = out[last[-1]] if len(last) else 0.0
    # Border correction is exactly unbiased but not monotone in finite
    # samples (the eligible set shrinks with r); enforce the CDF property so
    # the channel is a genuine distribution function.
    return np.maximum.accumulate(out)


def g_function(
    points: np.ndarray, region_low: Any, region_high: Any, r_grid: np.ndarray
) -> np.ndarray:
    """Border-corrected nearest-neighbour distance distribution G(r)."""
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 2:
        return np.zeros(len(r_grid), dtype=np.float64)
    low = np.asarray(region_low, dtype=np.float64)
    high = np.asarray(region_high, dtype=np.float64)

    # k=2 because the first neighbour returned is the query point itself.
    nn = cKDTree(points).query(points, k=2)[0][:, 1]
    return _reduced_sample_cdf(nn, _border_distance(points, low, high), r_grid)


def f_function(
    points: np.ndarray,
    region_low: Any,
    region_high: Any,
    r_grid: np.ndarray,
    grid_size: int = F_GRID_SIZE,
) -> np.ndarray:
    """Border-corrected empty-space function F(r) on a regular test grid."""
    points = np.asarray(points, dtype=np.float64)
    low = np.asarray(region_low, dtype=np.float64)
    high = np.asarray(region_high, dtype=np.float64)
    if len(points) == 0:
        return np.zeros(len(r_grid), dtype=np.float64)

    # Interior grid: linspace endpoints sit exactly ON the boundary, where the
    # border distance is 0 and the location is eligible at no radius at all.
    axes = [np.linspace(lo, hi, grid_size + 2)[1:-1] for lo, hi in zip(low, high)]
    mesh = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, len(low))

    empty = cKDTree(points).query(mesh, k=1)[0]
    return _reduced_sample_cdf(empty, _border_distance(mesh, low, high), r_grid)


def j_function(
    f: np.ndarray,
    g: np.ndarray,
    min_denom: float = J_MIN_DENOM,
    clip: float = J_CLIP,
) -> np.ndarray:
    """J(r) = (1 - G(r)) / (1 - F(r)), held flat where F is too close to 1.

    J == 1 under CSR, < 1 for clustering, > 1 for inhibition -- so the
    r = 0 value and the post-truncation fill are both 1, i.e. "no evidence
    of departure from Poisson", which is the right neutral value for a
    z-scored CNN channel.
    """
    f = np.asarray(f, dtype=np.float64)
    g = np.asarray(g, dtype=np.float64)
    denom = 1.0 - f
    valid = denom > min_denom

    out = np.ones(len(f), dtype=np.float64)
    out[valid] = (1.0 - g[valid]) / denom[valid]
    if valid.any() and not valid.all():
        # F is non-decreasing, so the invalid region is a suffix; hold the
        # last trustworthy value across it rather than letting the ratio blow
        # up as the denominator goes to 0.
        last = np.flatnonzero(valid)[-1]
        out[last + 1:] = out[last]
    return np.clip(out, 0.0, clip)


# Channel registry. "L" is produced by vihrs._isotropic_l_minus_r and is
# handled there; the three below are computed here. Keys are the names a
# config's `summary_channels` list uses, values the key each lands under in
# the cached feature dict.
CHANNEL_KEYS: dict[str, str] = {
    "L": "l_minus_r",
    "F": "f_func",
    "G": "g_func",
    "J": "j_func",
}
ALL_CHANNELS: tuple[str, ...] = ("L", "F", "G", "J")


def normalize_channels(channels: Any) -> tuple[str, ...]:
    """Validate/canonicalize a `summary_channels` config value."""
    if channels is None:
        return ("L",)
    if isinstance(channels, str):
        channels = [c for c in channels.replace(",", " ").split() if c]
    out = tuple(str(c).strip().upper() for c in channels)
    if not out:
        raise ValueError("summary_channels is empty -- give at least one of L, F, G, J.")
    bad = [c for c in out if c not in CHANNEL_KEYS]
    if bad:
        raise ValueError(f"unknown summary channel(s) {bad}; valid: {list(CHANNEL_KEYS)}")
    if len(set(out)) != len(out):
        raise ValueError(f"duplicate summary channels in {out}")
    return out


def compute_fgj(
    points: np.ndarray,
    region_low: Any,
    region_high: Any,
    fg_grid: np.ndarray,
    grid_size: int = F_GRID_SIZE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(F, G, J) on `fg_grid` for one cloud -- the one call the feature
    extractor in vihrs.py makes."""
    f = f_function(points, region_low, region_high, fg_grid, grid_size=grid_size)
    g = g_function(points, region_low, region_high, fg_grid)
    return f, g, j_function(f, g)


def stack_channels(
    features: dict[str, np.ndarray], channels: tuple[str, ...]
) -> np.ndarray:
    """(N, C, m) float32 stack of the requested channels, in the given order.

    A single channel still returns (N, 1, m); VihrsCNN accepts either that or
    the legacy (N, m), so the one-channel path stays numerically identical to
    the pre-existing L-only baseline.
    """
    missing = [c for c in channels if CHANNEL_KEYS[c] not in features]
    if missing:
        raise KeyError(
            f"summary channel(s) {missing} requested but not in the cached features "
            f"(have: {sorted(features)}). Rebuild the cache with recompute_features=true."
        )
    return np.stack([features[CHANNEL_KEYS[c]] for c in channels], axis=1).astype(np.float32)
