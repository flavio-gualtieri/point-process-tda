# src/cloudforger/vectorization/scalar_features/calibrated.py
"""Build a BettiCurve feature with a grid_range auto-calibrated from a
diagram sample, instead of hand-picked bounds.

build_calibrated_betti fits ONE shared grid_range across every requested
homology dimension (pooling finite death times from all of them together),
not a separate range per dimension. That's deliberate, not the obvious
choice: a separate per-dim range is what persistence images do (H0 and H1
are different pixel grids, see persistence_images/calibrated.py's
MultiChannelImager) and is what this module's own build_calibrated_betti
used to do too, before the Euler-characteristic channel existed. But
chi(t) = sum_d (-1)^d * beta_d(t) (euler_characteristic_curve below) is only
meaningful pointwise, which requires beta_0(t), beta_1(t), ... to already be
evaluated at the identical t -- i.e. one shared grid, not one per dimension.
BettiCurve already supports this: a single instance's self._grid is one
array shared across every dim in self._homology_dims (see betti_curve.py),
so build_calibrated_betti only has to pick that one grid_range."""

from __future__ import annotations

import numpy as np

from ...core.diagram import PersistenceDiagram
from .betti_curve import BettiCurve


def _shared_grid_hi(
    diagrams: list[PersistenceDiagram], homology_dims: tuple[int, ...], coverage: float, pad: float
) -> float:
    """Upper bound for a Betti-curve bundle's single shared filtration-value
    grid axis: the coverage-percentile death time, pooled over every finite
    (birth, death) pair across diagrams AND across every dim in
    homology_dims (pooling across dims, not just across diagrams, is the one
    difference from betti_curve.py's original _betti_grid_hi -- see module
    docstring for why a shared axis is required here). Same "no birth axis"
    reasoning as the function it replaces: a Betti curve has no birth axis
    of its own (BettiCurve only takes one grid_range -- see
    betti_curve.py's _curve_from_pairs: alive = births <= grid < deaths), so
    a degenerate/constant birth (e.g. plain Rips H0) is fine here."""
    per_diagram_deaths = [d.finite_pairs(dim)[:, 1] for d in diagrams for dim in homology_dims]
    nonempty = [d for d in per_diagram_deaths if len(d)]
    if not nonempty:
        return 1.0
    deaths = np.concatenate(nonempty)
    return float(pad * np.percentile(deaths, 100.0 * coverage))


def build_calibrated_betti(
    diagrams: list[PersistenceDiagram],
    homology_dims: tuple[int, ...] = (0, 1),
    grid_size: int = 128,
    coverage: float = 0.99,
    range_pad: float = 1.1,
    weight_by_persistence: bool = False,
) -> BettiCurve:
    """One BettiCurve covering every dim in homology_dims, all on the same
    grid_size-sample grid (see module docstring) -- .compute(diagram).curves
    is then a {dim: (grid_size,)} dict ready for euler_characteristic_curve
    below, or for stacking into a multi-channel [beta_d0, beta_d1, ...,
    euler] tensor (see experiments/pi_multik/betti_multik.py's
    build_betti_tensor)."""
    grid_hi = _shared_grid_hi(diagrams, homology_dims, coverage, range_pad)
    return BettiCurve(
        homology_dims=homology_dims,
        grid_size=grid_size,
        grid_range=(0.0, grid_hi),
        drop_infinite=True,
        normalize=False,
        weight_by_persistence=weight_by_persistence,
    )


def euler_characteristic_curve(curves: dict[int, np.ndarray]) -> np.ndarray:
    """chi(t) = sum_d (-1)^d * beta_d(t), pointwise. Requires every beta_d
    array in `curves` to already share the same t-grid (see
    build_calibrated_betti) -- shape-agnostic otherwise, so this works
    equally on a single diagram's {dim: (grid_size,)} curves (as returned by
    BettiCurve.compute(d).curves) or a batched {dim: (N, grid_size)}."""
    dims = sorted(curves)
    if not dims:
        raise ValueError("euler_characteristic_curve: `curves` is empty.")
    out = np.zeros_like(curves[dims[0]], dtype=np.float64)
    for d in dims:
        out = out + ((-1.0) ** d) * curves[d]
    return out
