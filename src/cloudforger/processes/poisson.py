import numpy as np
from ..core.base import PointProcess
from ..core.region import Region

class PoissonProcess(PointProcess):
    def __init__(self, intensity: float | None = None):
        if intensity is not None and intensity <= 0:
            raise ValueError("Intensity must be positive.")
        self.intensity = intensity

    @property
    def name(self) -> str:
        return "poisson"
    
    @property
    def params(self) -> dict[str, float | None]:
        return {"intensity": self.intensity}
    
    def _sample_points(
        self, n: int, region: Region, rng: np.random.Generator
    ) -> np.ndarray:
        if self.intensity is None:
            count = n
        else:
            expected = self.intensity * region.volume
            count = int(rng.poisson(expected))

        return region.sample_uniform(count, rng)