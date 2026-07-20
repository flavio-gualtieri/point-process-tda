# src/cloudforger/vectorizers/persistence_image.py

from __future__ import annotations

from typing import Any, Callable

from ..core.diagram import PersistenceDiagram

import numpy as np


def linear_weight(persistence: np.ndarray) -> np.ndarray:
    """Weight each point by its persistence. Zero weight on the diagonal,
    rising linearly. The standard, mild choice."""
    return persistence


class PersistenceImager:
    """Vectorizes one homology dimension of a PersistenceDiagram into a
    fixed-size raster, in birth-persistence coordinates.

    Bounds are fixed at construction and never inferred from data, so a
    pixel means the same thing across every image in the dataset.
    """

    def __init__(
        self,
        birth_range: tuple[float, float],
        pers_range: tuple[float, float],
        resolution: int = 64,
        sigma_frac: float = 0.1,
        sigma_stat: str = "median",
        min_sigma_pixels: float = 1.0,
        weight: bool = False,
        weight_fn: Callable[[np.ndarray], np.ndarray] = linear_weight,
    ):
        self._birth_range = birth_range
        self._pers_range = pers_range
        self._resolution = resolution
        self._sigma_frac = sigma_frac
        self._sigma_stat = sigma_stat
        self._min_sigma_pixels = min_sigma_pixels
        self._weight = weight
        self._weight_fn = weight_fn

        # Pixel-center coordinates along each axis, computed once.
        self._birth_axis = np.linspace(*birth_range, resolution)
        self._pers_axis = np.linspace(*pers_range, resolution)

    def transform(self, diagram: PersistenceDiagram, dim: int) -> np.ndarray:
        """Return a (resolution, resolution) image for homology `dim`.
        Axis 0 is persistence (descending), axis 1 is birth."""
        pairs = diagram.finite_pairs(dim)
        image = np.zeros((self._resolution, self._resolution))

        if len(pairs) == 0:
            return image  # empty diagram -> all-zero image

        births = pairs[:, 0]
        persistences = pairs[:, 1] - pairs[:, 0]  # birth-death -> birth-pers
        if self._sigma_stat == "median":
            spread = np.median(persistences) 
        elif self._sigma_stat == "iqr":
            q1, q3 = np.percentile(persistences, [25, 75])
            spread = q3 - q1
        else:
            raise ValueError(
                f"Unsupported sigma statistic: {self._sigma_stat!r}. "
                "Expected 'median' or 'iqr'."
            )
        if not np.isfinite(spread) or spread <= 0:
            spread = np.finfo(float).eps
    
        pixel = (self._pers_range[1] - self._pers_range[0]) / self._resolution
        sigma_floor = self._min_sigma_pixels * pixel
        sigma_diagram = max(self._sigma_frac * spread, sigma_floor)

        weights = self._weight_fn(persistences) if self._weight else np.ones_like(persistences)

        # Gaussian per point, summed onto the grid. Centers outside the
        # grid still deposit their in-range tail; nothing clipped/dropped.
        bb, pp = np.meshgrid(self._birth_axis, self._pers_axis)
        for b, p, w in zip(births, persistences, weights):
            kernel = np.exp(
                -((bb - b) ** 2 + (pp - p) ** 2) / (2 * sigma_diagram ** 2)
            )
            image += w * kernel

        # Flip so persistence increases upward when viewed as an image.
        return np.flipud(image)

    @property
    def params(self) -> dict[str, Any]:
        return {
            "birth_range": self._birth_range,
            "pers_range": self._pers_range,
            "resolution": self._resolution,
            "weight_fn": self._weight_fn.__name__ if self._weight else None,
            "sigma_frac": self._sigma_frac,
            "sigma_stat": self._sigma_stat,
            "min_sigma_pixels": self._min_sigma_pixels,
        }