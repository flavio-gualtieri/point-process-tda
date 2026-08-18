# src/cloudforger/vectorization/scalar_features/persistence_statistics.py
"""The floor-control vectorization: a handful of scalar summaries of one
diagram's persistence-length distribution, no grid/calibration at all (a
pure per-diagram function, same "calibration-free" status as
persistence_entropy.py -- see that module's docstring). Trained MLP-only
(see experiments/pi_multik/vectorized_multik.py): there's no raster/curve
shape here for a CNN to exploit, so pairing this with the native CoordConv/
Conv1d encoders would be meaningless."""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import DiagramFeature
from ...core.diagram import PersistenceDiagram

_QUANTILES = (10.0, 25.0, 50.0, 75.0, 90.0)

STAT_NAMES: tuple[str, ...] = (
    "count", "total_persistence", "persistent_entropy",
    *(f"birth_q{int(q)}" for q in _QUANTILES),
    *(f"death_q{int(q)}" for q in _QUANTILES),
    *(f"lifetime_q{int(q)}" for q in _QUANTILES),
)  # 3 + 5*3 = 18 statistics


def _entropy(lifetimes: np.ndarray) -> float:
    total = lifetimes.sum()
    if total <= 0:
        return 0.0
    probs = lifetimes / total
    return float(-(probs * np.log(probs)).sum())


def _stats_from_pairs(pairs: np.ndarray) -> np.ndarray:
    """(18,) statistics vector for one dimension's finite pairs -- an empty
    diagram (no points survived to this dim) returns an all-zero row, same
    "missing features -> zero rows" convention as landscape/silhouette
    (tent.py's tent() on an empty pair array)."""
    if len(pairs) == 0:
        return np.zeros(len(STAT_NAMES), dtype=float)

    births = pairs[:, 0]
    deaths = pairs[:, 1]
    lifetimes = deaths - births

    count = float(len(pairs))
    total_persistence = float(lifetimes.sum())
    entropy = _entropy(lifetimes)
    birth_q = np.percentile(births, _QUANTILES)
    death_q = np.percentile(deaths, _QUANTILES)
    lifetime_q = np.percentile(lifetimes, _QUANTILES)

    return np.concatenate([[count, total_persistence, entropy], birth_q, death_q, lifetime_q])


class PersistenceStatistics(DiagramFeature):
    """count, total persistence, persistent entropy, and 10/25/50/75/90
    quantiles of birth/death/lifetime, per homology dimension -- 18 scalars
    per dim, computed directly from finite (birth, death) pairs."""

    def __init__(self, homology_dims: tuple[int, ...] = (0, 1)):
        self._homology_dims = tuple(homology_dims)

    @property
    def name(self) -> str:
        return "persistence_statistics"

    @property
    def params(self) -> dict[str, Any]:
        return {"homology_dims": self._homology_dims, "stat_names": STAT_NAMES}

    def compute(self, diagram: PersistenceDiagram) -> dict[int, np.ndarray]:
        return {dim: _stats_from_pairs(diagram.finite_pairs(dim)) for dim in self._homology_dims}
