# src/pointforge/core/base.py
from abc import ABC, abstractmethod
from typing import Any
import numpy as np
from .cloud import PointCloud
from .region import Region

class PointProcess(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def params(self) -> dict[str, Any]:
        ...

    @abstractmethod
    def _sample_points(
        self, n: int, region: Region, rng: np.random.Generator
    ) -> np.ndarray:
        ...

    def sample(
        self, n: int, region: Region, seed: int | None = None
    ) -> PointCloud:
        """Public API. Wraps _sample_points with seeding and metadata."""
        rng = np.random.default_rng(seed)
        points = self._sample_points(n, region, rng)
        return PointCloud(
            points=points,
            generator_name=self.name,
            generator_params=self.params,
            seed=seed,
            region=region
        )