# src/cloudforger/core/base.py

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
            self,
            n: int | None,
            region: Region,
            rng: np.random.Generator
    ) -> np.ndarray:
        ...

    def sample(
            self,
            n: int | Region | None = None,
            region: Region | None = None,
            seed: int | None = None
    ) -> PointCloud:
        if region is None and isinstance(n, Region):
            region = n
            n = None
        if region is None:
            raise TypeError("Must specify a region to sample from.")
        
        rng = np.random.default_rng(seed)
        points = self._sample_points(n, region, rng)
        
        return PointCloud(
            points=points,
            generator_name=self.name,
            generator_params=self.params,
            seed=seed,
            region=region
        )