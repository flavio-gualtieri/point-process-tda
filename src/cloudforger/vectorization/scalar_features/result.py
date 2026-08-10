# src/cloudforger/features/result.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class BettiCurveFeature:

    curves: dict[int, np.ndarray]
    generator_name: str
    generator_params: dict[str, Any]
    seed: int | None = None
    filtration_name: str = ""
    filtration_params: dict[str, Any] = field(default_factory=dict)
    feature_name: str = "betti_curve"
    feature_params: dict[str, Any] = field(default_factory=dict)

    def dimensions(self) -> list[int]:
        return sorted(self.curves.keys())

    def vector(self, dimensions: list[int] | None = None) -> np.ndarray:
        dims = self.dimensions() if dimensions is None else dimensions
        return np.concatenate([self.curves[d] for d in dims])
