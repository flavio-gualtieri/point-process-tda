# src/cloudforger/vectorization/persistence_images/calibrated.py
"""Build a MultiChannelImager with ranges auto-calibrated from a diagram
sample, instead of hand-picked bounds. Reconciles the two independent
implementations that existed before this refactor
(dtm_experiment/compute_features.py::build_calibrated_imager and
scripts/processing/params/pipeline_lib/images.py::build_imagers) into one."""

from __future__ import annotations

from ...calibration import axis_bounds, calibrate, calibrate_report, diagram_stats
from ...core.diagram import PersistenceDiagram
from .multi_channel import MultiChannelImager
from .persistence_image import PersistenceImager


def build_calibrated_imager(
    diagrams: list[PersistenceDiagram],
    homology_dims: tuple[int, ...] = (0, 1),
    resolution: int = 128,
    sigma_pixels: float = 2.0,
    coverage: float = 0.99,
    verbose: bool = True,
) -> MultiChannelImager:
    if verbose:
        print(calibrate_report(diagrams))

    imagers: dict[int, PersistenceImager] = {}
    for dim in homology_dims:
        birth_range, pers_range = axis_bounds(
            diagrams=diagrams,
            homology_dim=dim,
            coverage=coverage
            )
        imagers[dim] = PersistenceImager(
            birth_range=birth_range,
            pers_range=pers_range,
            resolution=resolution,
            sigma_pixels=sigma_pixels,
        )
        if verbose:
            print(
                f"  [pi] dim={dim}: birth_range=({birth_range[0]:.4f}, {birth_range[1]:.4f}) "
                f"pers_range=(0.0, {pers_range[1]:.4f}) sigma_pixels={sigma_pixels:g}"
            )

    return MultiChannelImager(imagers)
