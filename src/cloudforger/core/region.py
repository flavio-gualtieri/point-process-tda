from abc import ABC, abstractmethod
import numpy as np

class Region(ABC):
    @property
    @abstractmethod
    def dimension(self) -> int:
        ...

    @property
    @abstractmethod
    def volume(self) -> float:
        ...

    @abstractmethod
    def contains(self, point: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def sample_uniform(self, n: int, rng: np.random.Generator) -> np.ndarray:
        ...


class Box(Region):
    def __init__(self, low: np.ndarray, high: np.ndarray):
        self.low = np.asarray(low, dtype=float)
        self.high = np.asarray(high, dtype=float)
        if self.low.shape != self.high.shape:
            raise ValueError("Low and high must have the same shape.")
        if np.any(self.low >= self.high):
            raise ValueError("Low must be less than high in all dimensions.")
        
    @property
    def dimension(self) -> int:
        return self.low.size
    
    @property
    def volume(self) -> float:
        return float(np.prod(self.high - self.low))
    
    def contains(self, points: np.ndarray) -> np.ndarray:
        return np.all((points >= self.low) & (points <= self.high), axis=1)

    def sample_uniform(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(self.low, self.high, size=(n, self.dimension))