# src/cloudforger/filtration/bifiltration.py

from __future__ import annotations
from typing import Any
from abc import ABC, abstractmethod

import numpy as np
import multipers as mp
import multipers.filtrations as F

from multipers.filtrations.density import DTM

from ...core.signed_measure import SignedMeasure
from ...core.cloud import PointCloud

class Bifiltration(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def params(self) -> dict[str, Any]: 
        ...

    @abstractmethod
    def vertex_function(self, cloud: PointCloud) -> np.ndarray:
        """The second filtration parameter, one value per point.
        Swapping this is how you get the other candidate designs."""
        ...

    @abstractmethod
    def _compute_measures(self, cloud, grid) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        ...

    def path_tag(self) -> str:
        return self.name

    def compute(self, cloud: PointCloud, grid) -> SignedMeasure:
        return SignedMeasure(
            measures=self._compute_measures(cloud, grid),
            grid=grid, axis_names=self.axis_names,
            generator_name=cloud.generator_name,
            generator_params=cloud.generator_params,
            seed=cloud.seed,
            filtration_name=self.name, filtration_params=self.params,
        )


class DtmRipsBifiltration(Bifiltration):
    axis_names = ("rips_radius", "dtm_codensity")

    def __init__(self, dtm_mass=0.05, homology_dims=(0,), threshold_radius=0.25):
        self._m, self._dims, self._thresh = dtm_mass, tuple(homology_dims), threshold_radius

    @property
    def name(self): return "mph_dtm"

    @property
    def params(self):
        return {"dtm_mass": self._m, "homology_dims": list(self._dims),
                "threshold_radius": self._thresh}

    def path_tag(self):
        return f"mph_dtm{self._m}"          # groups as mph_dtm0.05+0.15 via _TAG_FAMILY_RE

    def vertex_function(self, cloud):
        pts = np.asarray(cloud.points)
        return np.asarray(DTM(masses=[self._m]).fit(pts).score_samples(pts)).ravel()

    def _compute_measures(self, cloud, grid):
        pts = np.asarray(cloud.points)
        st = F.RipsLowerstar(points=pts, function=self.vertex_function(cloud),
                             threshold_radius=self._thresh)
        st.collapse_edges(-1)
        st.expansion(max(self._dims) + 1)
        sms = mp.signed_measure(st, degrees=list(self._dims),
                                grid_strategy="precomputed", grid=list(grid))
        return {d: (np.asarray(a, float), np.asarray(w, int))
                for d, (a, w) in zip(self._dims, sms)}