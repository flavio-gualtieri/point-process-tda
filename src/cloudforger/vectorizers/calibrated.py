# src/cloudforger/vectorizers/calibrated.py
"""Build a MultiChannelImager with ranges auto-calibrated from a diagram
sample, instead of hand-picked bounds. Reconciles the two independent
implementations that existed before this refactor
(dtm_experiment/compute_features.py::build_calibrated_imager and
scripts/processing/params/pipeline_lib/images.py::build_imagers) into one."""

from __future__ import annotations

from ..core.calibration import axis_bounds, calibrate, calibrate_report
from ..core.diagram import PersistenceDiagram
from .multi_channel import MultiChannelImager
from .persistence_image import PersistenceImager, DEFAULT_SIGMA_PIXELS


def build_calibrated_imager(
    diagrams: list[PersistenceDiagram],
    homology_dims: tuple[int, ...] = (0, 1),
    resolution: int = 128,
    sigma_pixels: float = DEFAULT_SIGMA_PIXELS,
    verbose: bool = True,
) -> MultiChannelImager:
    if verbose:
        print(calibrate_report(diagrams))
    stats = calibrate(diagrams)

    imagers: dict[int, PersistenceImager] = {}
    for dim in homology_dims:
        birth_hi, pers_hi = axis_bounds(stats, dim)
        imagers[dim] = PersistenceImager(
            birth_range=(0.0, birth_hi),
            pers_range=(0.0, pers_hi),
            resolution=resolution,
            sigma_pixels=sigma_pixels,
        )
        if verbose:
            print(
                f"  [pi] dim={dim}: birth_range=(0.0, {birth_hi:.4f}) "
                f"pers_range=(0.0, {pers_hi:.4f}) sigma_pixels={sigma_pixels:g}"
            )

    return MultiChannelImager(imagers)
