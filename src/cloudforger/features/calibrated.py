# src/cloudforger/features/calibrated.py
"""Build BettiCurve features with ranges auto-calibrated from a diagram
sample, instead of hand-picked bounds. Generalizes
dtm_experiment/compute_features.py's build_calibrated_betti."""

from __future__ import annotations

from ..core.calibration import axis_bounds, calibrate
from ..core.diagram import PersistenceDiagram
from .betti_curve import BettiCurve


def build_calibrated_betti_curves(
    diagrams: list[PersistenceDiagram],
    homology_dims: tuple[int, ...] = (0, 1),
    grid_size: int = 512,
    weight_by_persistence: bool = False,
    range_pad: float = 1.1,
) -> dict[int, BettiCurve]:
    stats = calibrate(diagrams=diagrams, homology_dims=homology_dims)
    betti_by_dim: dict[int, BettiCurve] = {}
    for dim in homology_dims:
        birth_hi, pers_hi = axis_bounds(stats, dim)
        grid_range = (0.0, (birth_hi + pers_hi) * range_pad)
        betti_by_dim[dim] = BettiCurve(
            homology_dims=(dim,),
            grid_size=grid_size,
            grid_range=grid_range,
            drop_infinite=True,
            normalize=False,
            weight_by_persistence=weight_by_persistence,
        )
    return betti_by_dim
