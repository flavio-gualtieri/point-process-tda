# src/cloudforger/filtration/base.py

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..core.cloud import PointCloud
from ..core.diagram import PersistenceDiagram

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

    def path_tag(self) -> str:
        """Directory-name-safe tag identifying this filtration + its
        path-significant params (e.g. DTM's k). Default: just the name;
        override when a param should be visible in data/results paths."""
        return self.name

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
