# src/cloudforger/calibration/__init__.py
"""Persistence-image (and general vectorizer) calibration: turning a sample
of diagrams into axis bounds / resolution / sigma_pixels choices. See
diagram_calibration.py for the implementation; vectorization/persistence_images/
calibrated.py and scripts/featurize_sigma_sweep.py are the main callers."""

from .diagram_calibration import (
    axis_bounds,
    bifiltration_grid,
    calibrate,
    calibrate_report,
    check_grid_resolves,
    diagram_stats,
)

__all__ = [
    "axis_bounds",
    "bifiltration_grid",
    "calibrate",
    "calibrate_report",
    "check_grid_resolves",
    "diagram_stats",
]
