# src/cloudforger/filtration/rips.py

from __future__ import annotations

from typing import Any
from ripser import ripser

from ..core.cloud import PointCloud
from .base import Filtration

import numpy as np


class RipsFiltration(Filtration):

    def __init__(self, maxdim: int = 1, thresh: float | None = None):
        self._maxdim = maxdim
        self._thresh = thresh

    @property
    def name(self) -> str:
        return "rips"

    @property
    def params(self) -> dict[str, Any]:
        return {
            "maxdim": self._maxdim,
            "thresh": self._thresh
        }

    def _compute_diagrams(self, cloud: PointCloud) -> dict[int, np.ndarray]:
        kwargs: dict[str, Any] = {"maxdim": self._maxdim}
        if self._thresh is not None:
            kwargs["thresh"] = self._thresh
        dgms = ripser(cloud.points, **kwargs)["dgms"]
        return {dim: dgms[dim] for dim in range(len(dgms))}
