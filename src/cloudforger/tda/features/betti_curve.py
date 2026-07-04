# src/cloudforger/tda/features/betti_curve.py

from __future__ import annotations

from typing import Any

from .base import DiagramFeature
from ...core.betti import BettiCurveFeature
from ...core.diagram import PersistenceDiagram

import numpy as np


class BettiCurve(DiagramFeature):

    def __init__(
            self,
            homology_dims: tuple[int] | int = (0, 1),
            grid_size: int = 128,
            grid_range: tuple[float, float] = (0.0, 1.0),
            drop_infinite: bool = True,
            normalize: bool = False,
    ):
        if grid_size < 2:
            raise ValueError("grid_size must be >= 2.")

        if grid_range[0] >= grid_range[1]:
            raise ValueError("grid_range must satisfy low < high.")

        self._homology_dims = tuple(homology_dims) if isinstance(homology_dims, tuple) else [homology_dims]
        self._grid_size = int(grid_size)
        self._grid_range = tuple(float(x) for x in grid_range)
        self._grid = np.linspace(*self._grid_range, self._grid_size)
        self._drop_infinite = bool(drop_infinite)
        self._normalize = bool(normalize)

    @property
    def name(self) -> str:
        return "bett_curve"

    @property
    def params(self) -> dict[str, Any]:
       return {
            "name": self.name,
            "homology_dims": self._homology_dims,
            "grid_size": self._grid_size,
            "grid_range": self._grid_range,
            "drop_infinite": self._drop_infinite,
            "normalize": self._normalize,
        }

    @property
    def grid(self) -> np.ndarray:
        return self._grid

    def _curve_from_pairs(self, pairs: np.ndarray) -> np.ndarray:
        if pairs.size == 0:
            return np.zeros(self._grid_size, dtype=float)

        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ValueError(
                f"Expected persistence pairs with shape (M, 2), got {pairs.shape}."
            )

        births = pairs[:, 0]
        deaths = pairs[:, 1]

        finite_births = np.isfinite(births)

        if self._drop_infinite:
            valid = finite_births & np.isfinite(deaths)
            births = births[valid]
            deaths = deaths[valid]
        else:
            valid = finite_births
            births = births[valid]
            deaths = deaths[valid]
            deaths = np.where(np.isinf(deaths), self._grid_range[1], deaths)

        if len(births) == 0:
            return np.zeros(self._grid_size, dtype=float)

        alive = (births[:, None] <= self._grid[None, :]) & (
            self._grid[None, :] < deaths[:, None]
        )
        curve = alive.sum(axis=0).astype(float)

        if self._normalize and len(births) > 0:
            curve = curve / float(len(births))

        return curve
  
    def compute(self, diagram: PersistenceDiagram) -> BettiCurveFeature:
        curves: dict[int, np.ndarray] = {}

        for dim in self._homology_dims:
            pairs = diagram.diagrams.get(dim)
            if pairs is None:
                pairs = np.empty((0, 2), dtype=float)

            curves[dim] = self._curve_from_pairs(np.asarray(pairs, dtype=float))

        return BettiCurveFeature(
            curves=curves,
            generator_name=diagram.generator_name,
            generator_params=diagram.generator_params,
            seed=diagram.seed,
            filtration_name=diagram.filtration_name,
            filtration_params=diagram.filtration_params,
            feature_name=self.name,
            feature_params=self.params,
        )