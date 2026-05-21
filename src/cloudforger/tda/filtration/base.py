from abc import ABC, abstractmethod
from cloudforger.core.cloud import PointCloud
from cloudforger.core.diagram import PersistenceDiagram

class Filtration(ABC):
    def compute(self, cloud: PointCloud) -> PersistenceDiagram:
        diagrams = self._compute_diagrams(cloud)
        return PersistenceDiagram(
            diagrams=diagrams,
            generator_name=cloud.generator_name,
            generator_params=cloud.generator_params,
            seed=cloud.seed,
            filtration_name=self.name,
            filtration_params=self.params,
        )
    
    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def params(self) -> dict[str, any]:
        ...

    @abstractmethod
    def _compute_diagrams(self, cloud: PointCloud) -> dict[int, np.ndarray]:
        ...