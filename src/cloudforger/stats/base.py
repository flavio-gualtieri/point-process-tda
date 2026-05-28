# src/cloudforger/stats/base.py
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from ..core.cloud import PointCloud


class CloudStatistic(ABC):

    def __init__(self, n_samples: int, grid_size: int):
        self._n_samples = n_samples
        self._grid_size = grid_size
        self._grid = np.linspace(*self.grid_range, grid_size)

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def grid_range(self) -> tuple[float, float]:
        ...

    @abstractmethod
    def _sample_values(self, cloud: PointCloud, rng: np.random.Generator) -> np.ndarray:
        ...

    def compute(self, cloud: PointCloud) -> np.ndarray:
        rng = np.random.default_rng(
            None if cloud.seed is None else cloud.seed + 1_000_003
        )
        values = self._sample_values(cloud, rng)
        values = np.sort(values)
      
        counts = np.searchsorted(values, self._grid, side="right")
        return counts / len(values)

    @property
    def params(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "n_samples": self._n_samples,
            "grid_size": self._grid_size,
            "grid_range": self.grid_range,
        }