# src/cloudforger/features/calibrated.py
"""Build BettiCurve features with ranges auto-calibrated from a diagram
sample, instead of hand-picked bounds. Generalizes
dtm_experiment/compute_features.py's build_calibrated_betti."""

from __future__ import annotations

import numpy as np

from ..core.diagram import PersistenceDiagram
from .betti_curve import BettiCurve


def _betti_grid_hi(diagrams: list[PersistenceDiagram], homology_dim: int, coverage: float, pad: float) -> float:
    """Upper bound for a Betti curve's single filtration-value grid axis:
    the coverage-percentile death time, pooled over every finite (birth,
    death) pair across diagrams. Deliberately NOT core.calibration.axis_bounds
    -- that function calibrates a 2D (birth, persistence) IMAGE and rejects a
    degenerate birth axis (raises for e.g. plain Rips H0, where every birth
    is 0). A Betti curve has no birth axis of its own (BettiCurve only takes
    one grid_range, see features/betti_curve.py's _curve_from_pairs: alive =
    births <= grid < deaths), so a degenerate/constant birth is completely
    fine here -- it just means every H0 feature comes alive at the same
    point, not that the curve is unbuildable."""
    deaths = [d.finite_pairs(homology_dim)[:, 1] for d in diagrams]
    deaths = np.concatenate([d for d in deaths if len(d)])
    if len(deaths) == 0:
        return 1.0
    return float(pad * np.percentile(deaths, 100.0 * coverage))


def build_calibrated_betti_curves(
    diagrams: list[PersistenceDiagram],
    homology_dims: tuple[int, ...] = (0, 1),
    grid_size: int = 512,
    weight_by_persistence: bool = False,
    coverage: float = 0.99,
    range_pad: float = 1.1,
) -> dict[int, BettiCurve]:
    betti_by_dim: dict[int, BettiCurve] = {}
    for dim in homology_dims:
        grid_hi = _betti_grid_hi(diagrams, dim, coverage, range_pad)
        grid_range = (0.0, grid_hi)
        betti_by_dim[dim] = BettiCurve(
            homology_dims=(dim,),
            grid_size=grid_size,
            grid_range=grid_range,
            drop_infinite=True,
            normalize=False,
            weight_by_persistence=weight_by_persistence,
        )
    return betti_by_dim
