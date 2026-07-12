# cloudforger/tda/filtration/dtm.py

from __future__ import annotations

from typing import Any
from gudhi.dtm_rips_complex import DTMRipsComplex

from ...core.cloud import PointCloud
from .base import Filtration

import numpy as np


class DTMFiltration(Filtration):
    """Density-sensitive filtration based on the Distance-to-Measure (DTM).

    Instead of building the Rips complex on raw pairwise distances, each point
    is assigned a weight equal to its DTM value -- roughly the (q-averaged)
    distance to its ``k`` nearest neighbours. Points in dense regions get small
    weights and enter the filtration early; points in sparse regions enter late.
    The result is a weighted-Rips filtration (Anai et al. 2020) whose birth/death
    structure reflects *local density variation* rather than pure geometry.

    ``k`` is the key knob: it sets the scale over which density is estimated.
    Small ``k`` is sensitive to fine-grained clustering but noisier; large ``k``
    smooths over local fluctuations. Treat it the way you'd treat the radius ``r``
    in an L-function -- you may want several values.

    Note: DTM filtration values live on a different scale than raw Rips distances,
    so ``birth_range`` / ``pers_range`` for the persistence imager must be
    re-calibrated (run ``calibrate`` on DTM diagrams, don't reuse Rips ranges).
    """

    def __init__(
        self,
        maxdim: int = 1,
        k: int = 5,
        q: float = 2.0,
        thresh: float | None = None,
    ):
        self._maxdim = maxdim
        self._k = k
        self._q = q
        self._thresh = thresh

    @property
    def name(self) -> str:
        return "dtm"

    @property
    def params(self) -> dict[str, Any]:
        return {
            "maxdim": self._maxdim,
            "k": self._k,
            "q": self._q,
            "thresh": self._thresh,
        }

    def _compute_diagrams(self, cloud: PointCloud) -> dict[int, np.ndarray]:
        max_filtration = np.inf if self._thresh is None else self._thresh
        dtm_rips = DTMRipsComplex(
            points=np.asarray(cloud.points),
            k=self._k,
            q=self._q,
            max_filtration=max_filtration,
        )
        # A (maxdim)-dimensional feature needs (maxdim + 1)-simplices to die.
        st = dtm_rips.create_simplex_tree(max_dimension=self._maxdim + 1)
        st.compute_persistence()

        diagrams: dict[int, np.ndarray] = {}
        for dim in range(self._maxdim + 1):
            pairs = st.persistence_intervals_in_dimension(dim)
            # Normalise empty results to a well-shaped (0, 2) array so downstream
            # code (finite_pairs / calibrate) can index columns unconditionally.
            if pairs.size == 0:
                pairs = np.empty((0, 2), dtype=float)
            diagrams[dim] = np.asarray(pairs, dtype=float)
        return diagrams