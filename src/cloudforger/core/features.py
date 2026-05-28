from dataclasses import dataclass, field
from typing import Any

import numpy as np

@dataclass
class CorrelationFeatures:
    features: dict[str, np.ndarray]

    generator_name: str
    generator_params: dict[str, Any]
    seed: int | None = None

    statistic_params: dict[str, Any] = field(default_factory=dict)

    def names(self) -> list[str]:
        return sorted(self.features.keys())
    
    def vector(self, names: list[str] | None = None) -> np.ndarray:
        keys = self.names() if names is None else names
        return np.concatenate([self.features[k] for k in keys])