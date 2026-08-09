# src/cloudforger/core/signed_measure.py

from __future__ import annotations
from typing import Any

import numpy as np

from dataclasses import dataclass, field

@dataclass
class SignedMeasure:
    
    measures: dict[int, tuple[np.ndarray, np.ndarray]]
    grid: tuple[np.ndarray, np.ndarray]
    axis_names: tuple[str, str]
    generator_name: str
    generator_params: dict[str, Any]
    seed: int | None = None
    filtration_name: str = ""
    filtration_params: dict[str, Any] = field(default_factory=dict)

    def dimensions(self) -> list[int]:
        return sorted(self.measures.keys())

    def atoms(self, dim: int) -> tuple[np.ndarray, np.ndarray]:
        a, w = self.measures.get(dim, (None, None))
        if a is None or len(a) == 0:
            return np.empty((0, 2)), np.empty((0,), dtype=int)
        return a, w