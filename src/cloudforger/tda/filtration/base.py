# cloudforger/tda/filtration/base.py

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ...core.cloud import PointCloud
from ...core.diagram import PersistenceDiagram

import numpy as np


class Filtration(ABC):

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def params(self) -> dict[str, Any]:
        ...

    @abstractmethod
    def _compute_diagrams(self, cloud: PointCloud) -> dict[int, np.ndarray]:
        ...

    def compute(self, cloud: PointCloud) -> PersistenceDiagram:
        diagrams = self._compute_diagrams(cloud)
        return PersistenceDiagram(
            diagrams=diagrams,
            generator_name=cloud.generator_name,
            generator_params=cloud.generator_params,
            seed=cloud.seed,
            filtration_name=self.name,
            filtration_params=self.params,
        )