from dataclasses import dataclass, field
from typing import Any, Optional
import numpy as np
from .region import Region

@dataclass
class PointCloud:
    points: np.ndarray #shape (N, D)
    generator_name: str
    generator_params: dict[str, Any] = field(default_factory=dict)
    seed: int | None = None
    region: Optional[Region] = None

    @property
    def n_points(self) -> int:
        return self.points.shape[0]
    
    @property
    def dimension(self) -> int:
        return self.points.shape[1]
    
    def __repr__(self) -> str:
        return (
            f"PointCloud(n={self.n_points}, d={self.dimension}, "
            f"generator={self.generator_name!r}, seed={self.seed})"
        )