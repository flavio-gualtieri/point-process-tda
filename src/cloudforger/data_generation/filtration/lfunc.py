# src/cloudforger/data_generation/filtration/lfunc.py
"""L-reparameterized filtrations: (b, d) -> (L(b), L(d)).

Takes any base filtration and re-indexes its persistence diagram by the
cloud's OWN empirical Ripley L function instead of by raw distance.

Why this is a filtration and not a post-hoc edit of a diagram: the empirical
K used here (baselines.vihrs._isotropic_l_minus_r) is a cumulative sum of
positive isotropic-correction weights over pairs with d <= r, so K is
monotone non-decreasing and L = sqrt(K/pi) is too. Applying a monotone
increasing phi to a filtration's values yields exactly the persistence
diagram of the reparameterized filtration G_s = F_{phi^-1(s)}:
Dgm(G) = phi(Dgm(F)) pointwise. So this changes the clock, not the geometry
-- points stay above the diagonal by construction.

Interpretation of the new axis: L(r) = r exactly under CSR, so L-units are
"equivalent-Poisson radius". A feature dying at L-value 0.08 survived until
the cloud had accumulated as many neighbours as a Poisson process would by
radius 0.08.

What it costs: the transform moves the second-order trend OUT of the diagram
and into L itself. Used alone the representation is strictly weaker; used
alongside the L curve (pi_multik's `include_lfunc` extra-channel) the pair is
a factorization -- L carries the trend, the transformed diagram carries the
topology with that trend divided out -- which is the point. Never evaluate
this filtration without also running the arm that feeds L back in.

Two caveats baked into the implementation:
  * L is only estimated on [0, r_max] (spatstat's side/4 default = 0.25 on
    the unit square; beyond that the isotropic edge correction degrades).
    ~1% of DTM_k5 diagram values exceed it. Values above r_max are extended
    with slope 1 -- principled, since beyond the correlation range the
    process decorrelates, K(r) -> pi r^2 and dL/dr -> 1.
  * Non-finite entries (the essential H0 bar, death = inf) pass through
    untouched.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ...core.cloud import PointCloud
from .base import Filtration
from .dtm import DTMFiltration
from .rips import RipsFiltration

# spatstat Kest/Lest geometric defaults for the unit square; kept in sync with
# baselines.vihrs.R_MAX / N_R so the L used here is the SAME L the vihrs
# baseline consumes (that identity is the whole point of the comparison).
R_MAX = 0.25
N_R = 513


class LFuncFiltration(Filtration):
    """Wraps a base Filtration, reparameterizing its diagram by empirical L."""

    def __init__(self, base: Filtration, r_max: float = R_MAX, n_r: int = N_R):
        self._base = base
        self._r_max = float(r_max)
        self._n_r = int(n_r)
        self._r_grid = np.linspace(0.0, self._r_max, self._n_r)

    @property
    def name(self) -> str:
        return f"l_{self._base.name}"

    @property
    def params(self) -> dict[str, Any]:
        return {**self._base.params, "r_max": self._r_max, "n_r": self._n_r}

    def path_tag(self) -> str:
        return f"l_{self._base.path_tag()}"

    # -- the reparameterization ------------------------------------------

    def cloud_l(self, cloud: PointCloud) -> np.ndarray:
        """Monotone empirical L on self._r_grid for this cloud."""
        # Imported lazily: baselines.vihrs pulls in torch and the experiment
        # machinery, which has no business loading at filtration-import time.
        # Sharing the function (rather than reimplementing) guarantees this is
        # bit-for-bit the L the vihrs baseline uses.
        from ...baselines.vihrs import _isotropic_l_minus_r

        region = cloud.region
        low = getattr(region, "low", None)
        high = getattr(region, "high", None)
        if low is None or high is None:
            low, high = [0.0, 0.0], [1.0, 1.0]
        l_minus_r = _isotropic_l_minus_r(np.asarray(cloud.points, dtype=np.float64), low, high, self._r_grid)
        return monotone_l(np.asarray(l_minus_r, dtype=np.float64) + self._r_grid)

    def apply_to_pairs(self, pairs: np.ndarray, l_curve: np.ndarray) -> np.ndarray:
        return apply_l_transform(pairs, l_curve, self._r_grid)

    def _compute_diagrams(self, cloud: PointCloud) -> dict[int, np.ndarray]:
        l_curve = self.cloud_l(cloud)
        return {
            dim: self.apply_to_pairs(pairs, l_curve)
            for dim, pairs in self._base._compute_diagrams(cloud).items()
        }


# ---------------------------------------------------------------------------
# Free functions -- also used by scripts/processing/transform_diagrams_lfunc.py,
# which transforms ALREADY-COMPUTED diagrams (no persistence recompute) using
# the cached L curves. Keeping the maths in one place keeps that fast path and
# the from-scratch path identical.
# ---------------------------------------------------------------------------


def monotone_l(l_curve: np.ndarray) -> np.ndarray:
    """Enforce monotonicity. K is a cumulative sum of positive weights so L is
    monotone by construction (verified: 0 violations over 200 sampled Thomas
    clouds); this only guards float noise. NOTE the cached array is L - r,
    which is NOT monotone -- add r back BEFORE calling this."""
    return np.maximum.accumulate(np.asarray(l_curve, dtype=np.float64))


def apply_l_transform(pairs: np.ndarray, l_curve: np.ndarray, r_grid: np.ndarray) -> np.ndarray:
    """(b, d) -> (L(b), L(d)) for an (m, 2) diagram array.

    Finite values are linearly interpolated on r_grid; values above
    r_grid[-1] get the slope-1 tail L(r_max) + (r - r_max); non-finite
    entries (the essential H0 bar) pass through."""
    out = np.asarray(pairs, dtype=np.float64).copy()
    if out.size == 0:
        return out.reshape(-1, 2)
    r_max = float(r_grid[-1])
    finite = np.isfinite(out)
    v = out[finite]
    out[finite] = np.where(
        v > r_max,
        float(l_curve[-1]) + (v - r_max),
        np.interp(v, r_grid, l_curve),
    )
    return out


# ---------------------------------------------------------------------------
# Registry-facing presets. Config selects these by name, e.g.
#   filtration: [{name: l_dtm, params: {k: 5, q: 2.0, maxdim: 1}}]
# -> path tag l_dtm_k5 -> data/<process>/l_dtm_k5/diagrams.pkl
# ---------------------------------------------------------------------------


class LRipsFiltration(LFuncFiltration):
    """L-reparameterized Rips -- the strict version: L is defined on
    inter-point distances and Rips filtration values ARE inter-point
    distances, so this is a true reparameterization. Caveat: Rips H0 births
    are all 0 and L(0) = 0, so the H0 birth axis stays degenerate and
    build_calibrated_imager will still reject it -- use homology_dims: [1],
    or the vec_multik landscape path, for this one."""

    def __init__(self, maxdim: int = 1, thresh: float | None = None, r_max: float = R_MAX, n_r: int = N_R):
        super().__init__(RipsFiltration(maxdim=maxdim, thresh=thresh), r_max=r_max, n_r=n_r)


class LDTMFiltration(LFuncFiltration):
    """L-reparameterized DTM. DTM values are k-NN *average* distances rather
    than inter-point distances, so composing with L is a type-compatible
    monotone normalization rather than a strict reparameterization -- weaker
    justification than LRips, but DTM has non-degenerate H0 births, so this
    is the arm that drops straight into the existing pi_multik H0+H1 image
    pipeline and compares directly against every number in the tables."""

    def __init__(
        self,
        maxdim: int = 1,
        k: int = 5,
        q: float = 2.0,
        thresh: float | None = None,
        r_max: float = R_MAX,
        n_r: int = N_R,
    ):
        super().__init__(DTMFiltration(maxdim=maxdim, k=k, q=q, thresh=thresh), r_max=r_max, n_r=n_r)
