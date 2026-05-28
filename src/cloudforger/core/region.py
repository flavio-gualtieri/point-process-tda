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

    @abstractmethod
    def expanded(self, pad: float) -> "Region":
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
    
    def expanded(self, pad: float) -> "Box":
        return Box(low=self.low - pad, high=self.high + pad)
    

class Torus(Region):
    def __init__(self, R: float, r: float):
        self.center = np.asarray(np.zeros(3), dtype=float)
        if R <= 0 or r <= 0:
            raise ValueError("Radii must be positive.")
        if r >= R:
            raise ValueError("Minor radius must be smaller than major radius.")
        self.R = float(R)
        self.r = float(r)

    @property
    def dimension(self) -> int:
        return 3

    @property
    def volume(self) -> float:
        # Pappus's theorem: V = 2 * pi^2 * R * r^2
        return float(2 * np.pi**2 * self.R * self.r**2)

    def contains(self, points: np.ndarray) -> np.ndarray:
        p = points - self.center
        # Distance from the z-axis (the torus's axis of symmetry)
        rho = np.sqrt(p[:, 0]**2 + p[:, 1]**2)
        # Distance from the central circle of the tube
        d_sq = (rho - self.R)**2 + p[:, 2]**2
        return d_sq <= self.r**2
    
    def expanded(self, pad: float) -> "Torus":
        return Torus(R=self.R + pad, r=self.r + pad)

    def sample_uniform(self, n: int, rng: np.random.Generator) -> np.ndarray:
        # Rejection sampling: the Jacobian for (theta, phi, s) -> (x,y,z)
        # is s*(R + s*cos(phi)), so uniform sampling in (theta, phi, s) is biased.
        # We accept with probability (R + s*cos(phi)) / (R + r).
        samples = []
        remaining = n
        while remaining > 0:
            # Oversample to reduce loop iterations; acceptance rate is R/(R+r) on average
            batch = int(remaining * (self.R + self.r) / self.R * 1.1) + 1
            theta = rng.uniform(0, 2*np.pi, batch)        # around main axis
            phi   = rng.uniform(0, 2*np.pi, batch)        # around tube
            s     = np.sqrt(rng.uniform(0, 1, batch)) * self.r  # radial in tube
            
            accept = rng.uniform(0, 1, batch) < (self.R + s*np.cos(phi)) / (self.R + self.r)
            theta, phi, s = theta[accept], phi[accept], s[accept]
            
            x = (self.R + s*np.cos(phi)) * np.cos(theta)
            y = (self.R + s*np.cos(phi)) * np.sin(theta)
            z = s * np.sin(phi)
            samples.append(np.stack([x, y, z], axis=1))
            remaining -= len(theta)
        
        pts = np.concatenate(samples, axis=0)[:n]
        return pts + self.center