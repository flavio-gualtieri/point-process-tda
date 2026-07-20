# src/cloudforger/features/persistence_entropy.py

from __future__ import annotations

from typing import Any

from .base import DiagramFeature
from ..core.diagram import PersistenceDiagram

import numpy as np


def _entropy_from_pairs(pairs: np.ndarray) -> float:
    finite = pairs[np.isfinite(pairs).all(axis=1)]
    if len(finite) == 0:
        return 0.0
    p = finite[:, 1] - finite[:, 0]
    total = p.sum()
    if total <= 0:
        return 0.0
    probs = p / total
    return float(-(probs * np.log(probs)).sum())


class PersistenceEntropy(DiagramFeature):
    """Shannon entropy of the normalized persistence-length distribution,
    per homology dimension. Promoted from dtm_experiment/compute_features.py
    (previously a local free function, not a registered feature)."""

    def __init__(self, homology_dims: tuple[int, ...] = (0, 1)):
        self._homology_dims = tuple(homology_dims)

    @property
    def name(self) -> str:
        return "persistence_entropy"

    @property
    def params(self) -> dict[str, Any]:
        return {"homology_dims": self._homology_dims}

    def compute(self, diagram: PersistenceDiagram) -> dict[int, float]:
        return {
            dim: _entropy_from_pairs(np.asarray(diagram.diagrams.get(dim, np.empty((0, 2))), dtype=float))
            for dim in self._homology_dims
        }
