"""Persistence pairs -> persistence image (Adams et al. 2017), 2-D or 1-D.

Pairs are the (m, 2) finite (birth, death) arrays of data/featurization/<family>/<tag>/diagrams.npz.
Each point becomes a Gaussian in (birth, persistence), weighted by its persistence; a pixel is the
exact mass of that surface over its box, not a point sample. Bounds are fitted once on training
diagrams, so a pixel means the same thing everywhere.

Mass outside the box is dropped rather than clamped, so a point sitting on the persistence = 0 edge
keeps half its Gaussian. The linear weight makes such points nearly weightless, so the loss is small.

1-D (birth_range=None): the persistence axis alone, for filtrations whose H0 births are all 0 (rips,
alpha). DTM H0 births are 2 f(x), a density, so DTM H0 is 2-D like H1.

Scaling makes patterns of different n comparable, and is the one knob to turn off to get raw
diagrams back: coordinates in units of mean point spacing (pairs * sqrt(n), the unit window's
1/sqrt(n) spacing), and image mass per point (/ n, since H0 has n - 1 pairs). Without it a pattern
at nbar=800 fills a third of the axis of one at nbar=100 and carries ~7x the mass, so the effective
resolution and dynamic range would both track the nbar regime axis.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.special import erf


@dataclass(frozen=True)
class Scaling:
    coords: str = "sqrt_n"   # "sqrt_n" | "none"
    density: bool = True     # divide the image by n

    def pairs(self, pairs: np.ndarray, n: int) -> np.ndarray:
        if self.coords == "none":
            return pairs
        if self.coords != "sqrt_n":
            raise ValueError(f"Scaling.coords must be 'sqrt_n' or 'none', got {self.coords!r}")
        return pairs * math.sqrt(n)

    def mass(self, image: np.ndarray, n: int) -> np.ndarray:
        return image / n if self.density else image


def _box_mass(edges: np.ndarray, centers: np.ndarray, sigma: float) -> np.ndarray:
    """Mass of N(centers, sigma^2) in each [edges[i], edges[i+1]): (len(edges) - 1, len(centers))."""
    z = (edges[:, None] - centers[None, :]) / (sigma * math.sqrt(2.0))
    return np.diff(0.5 * (1.0 + erf(z)), axis=0)


class PersistenceImager:
    """One (birth, persistence) box -> images of a fixed shape.

    sigma is in pixels, shared by both axes: the birth and persistence spans differ, so one absolute
    sigma would stretch the Gaussian's footprint by whatever their ratio happens to be, confounding
    how much smoothing with how distorted it is.
    """

    def __init__(
        self,
        pers_range: tuple[float, float],
        birth_range: tuple[float, float] | None = None,
        resolution: int = 64,
        sigma_pixels: float = 1.0,
        scaling: Scaling = Scaling(),
    ):
        if sigma_pixels <= 0:
            raise ValueError(f"sigma_pixels must be positive, got {sigma_pixels!r}")
        self.pers_range, self.birth_range = pers_range, birth_range
        self.resolution, self.sigma_pixels, self.scaling = resolution, sigma_pixels, scaling
        self._pers_edges = np.linspace(*pers_range, resolution + 1)
        self._sigma_pers = sigma_pixels * (pers_range[1] - pers_range[0]) / resolution
        if birth_range is not None:
            self._birth_edges = np.linspace(*birth_range, resolution + 1)
            self._sigma_birth = sigma_pixels * (birth_range[1] - birth_range[0]) / resolution

    @property
    def shape(self) -> tuple[int, ...]:
        return (self.resolution,) if self.birth_range is None else (self.resolution, self.resolution)

    def transform(self, pairs: np.ndarray, n: int) -> np.ndarray:
        """Image of one diagram; `n` is its pattern's point count (see Scaling).

        2-D: axis 0 is persistence (increasing upward, i.e. flipped for viewing), axis 1 is birth.
        """
        pairs = np.asarray(pairs, dtype=float).reshape(-1, 2)
        if len(pairs) == 0:
            return np.zeros(self.shape)
        pairs = self.scaling.pairs(pairs, n)
        persistence = pairs[:, 1] - pairs[:, 0]
        weights = persistence                      # linear weight (Adams et al.)
        pers_mass = _box_mass(self._pers_edges, persistence, self._sigma_pers)
        if self.birth_range is None:
            image = pers_mass @ weights
        else:
            birth_mass = _box_mass(self._birth_edges, pairs[:, 0], self._sigma_birth)
            image = np.flipud((weights[None, :] * pers_mass) @ birth_mass.T)
        return self.scaling.mass(image, n)

    @property
    def params(self) -> dict:
        return {"pers_range": self.pers_range, "birth_range": self.birth_range,
                "resolution": self.resolution, "sigma_pixels": self.sigma_pixels,
                "coords": self.scaling.coords, "density": self.scaling.density}


def fit_imager(
    pairs: list[np.ndarray],
    n_points: np.ndarray,
    birth_axis: bool,
    resolution: int = 64,
    sigma_pixels: float = 1.0,
    coverage: float = 0.99,
    pad: float = 1.05,
    scaling: Scaling = Scaling(),
) -> PersistenceImager:
    """Imager whose box holds the FULL support of `coverage` of the given (training) diagrams.

    Per-diagram extrema, not pooled pairs: the box must contain whole diagrams, so that clipping is
    a property of a pattern rather than of its busiest corner.
    """
    lo, hi, pers_hi = [], [], []
    for p, n in zip(pairs, n_points):
        p = np.asarray(p, dtype=float).reshape(-1, 2)
        if len(p) == 0:
            continue
        p = scaling.pairs(p, int(n))
        lo.append(p[:, 0].min())
        hi.append(p[:, 0].max())
        pers_hi.append((p[:, 1] - p[:, 0]).max())
    if not pers_hi:
        raise ValueError("fit_imager: every diagram is empty")
    q = 100.0 * coverage
    pers_range = (0.0, float(pad * np.percentile(pers_hi, q)))
    birth_range = None
    if birth_axis:
        birth_range = (float(np.percentile(lo, 100.0 - q)), float(pad * np.percentile(hi, q)))
        if birth_range[1] <= birth_range[0]:
            raise ValueError("birth axis is degenerate; fit with birth_axis=False (rips/alpha H0)")
    return PersistenceImager(pers_range, birth_range, resolution, sigma_pixels, scaling)
