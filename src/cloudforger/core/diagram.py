from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class PersistenceDiagram:
    diagrams: dict[int, np.ndarray]
    generator_name: str
    generator_params: dict[str, Any]
    seed: int | None = None
    
    filtration_name: str = ""
    filtration_params: dict[str, Any] = field(default_factory=dict)

    def dimensions(self) -> list[int]:
        return sorted(self.diagrams.keys())

    def finite_pairs(self, dim: int) -> np.ndarray:
        pairs = self.diagrams.get(dim)
        if pairs is None or len(pairs) == 0:
            return np.empty((0, 2), dtype=float)
        finite_mask = np.isfinite(pairs).all(axis=1)
        return pairs[finite_mask]