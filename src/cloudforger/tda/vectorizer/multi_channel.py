# src/cloudforger/tda/vectorizer/multi_channel.py
from typing import Any

import numpy as np

from ...core.diagram import PersistenceDiagram
from .persistence_image import PersistenceImager


class MultiChannelImager:
    """Applies one PersistenceImager per homology dimension.

    Each dimension has its own imager with independently calibrated
    bounds, so H0 and H1 are vectorized on their own coordinate systems.
    Output is keyed by dimension, for routing to per-dimension encoders.
    """

    def __init__(self, imagers: dict[int, PersistenceImager]):
        # e.g. {0: imager_h0, 1: imager_h1}
        self._imagers = imagers

    def transform(self, diagram: PersistenceDiagram) -> dict[int, np.ndarray]:
        """Return {dim: (resolution, resolution) image} for each
        configured dimension."""
        return {
            dim: imager.transform(diagram, dim)
            for dim, imager in self._imagers.items()
        }

    @property
    def dimensions(self) -> list[int]:
        return sorted(self._imagers.keys())

    @property
    def params(self) -> dict[str, Any]:
        return {dim: im.params for dim, im in self._imagers.items()}