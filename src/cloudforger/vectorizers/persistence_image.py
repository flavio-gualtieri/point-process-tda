# src/cloudforger/vectorizers/persistence_image.py

from __future__ import annotations

from typing import Any, Callable

from ..core.diagram import PersistenceDiagram

import numpy as np
from scipy.special import erf

# Chosen from a speckle/component-count diagnostic sweep (0.25-6px) over the
# real nested_thomas dtm_k5/10/15 diagrams: H1 stays heavily speckled below
# this value, H0 starts collapsing to a single blob above it. Not yet
# cross-checked against training loss (scripts/sweep_pi_sigma.py); revisit if
# that sweep, or a similar check on a different dataset, says otherwise.


def linear_weight(persistence: np.ndarray) -> np.ndarray:
    """Weight each point by its persistence. Zero weight on the diagonal,
    rising linearly. The standard choice from Adams et al. (2017)."""
    return persistence


def _box_mass(edges: np.ndarray, centers: np.ndarray, sigma: float) -> np.ndarray:
    """Exact probability mass of a 1-D N(centers, sigma^2) distribution
    inside each [edges[i], edges[i+1]) bin, via the Gaussian CDF. Returns
    shape (len(edges)-1, len(centers))."""
    z = (edges[:, None] - centers[None, :]) / (sigma * np.sqrt(2.0))
    cdf = 0.5 * (1.0 + erf(z))
    return np.diff(cdf, axis=0)


class PersistenceImager:
    """Vectorizes one homology dimension of a PersistenceDiagram into a
    fixed-size raster, following Adams et al. (2017), "Persistence Images:
    A Stable Vector Representation of Persistent Homology".

    Each diagram point becomes a normalized 2-D Gaussian in birth-persistence
    coordinates (the "persistence surface"), weighted by `weight_fn` of its
    persistence value. Each pixel is the exact integral of that surface over
    the pixel's box -- computed in closed form from the Gaussian CDF via
    `_box_mass`, since the 2-D Gaussian is separable into independent birth
    and persistence factors -- not a point sample of the surface at the
    pixel center.

    Bounds are fixed at construction and never inferred from a single
    diagram, so a pixel means the same thing across every image in the
    dataset, and two images built from this imager remain comparable under
    the paper's stability results (which assume a shared bandwidth).

    Bandwidth is specified as `sigma_pixels`, a single number of pixels
    shared by both axes, rather than one absolute value in birth/persistence
    units. birth_range and pers_range are calibrated independently and
    typically have very different spans, so a pixel is a different physical
    size on each axis; an absolute sigma shared between them would make the
    Gaussian isotropic in data space but stretched or squashed in pixel
    space by whatever that span ratio happens to be -- confounding "how much
    smoothing" with "how distorted the footprint is" the moment the two
    ranges differ. Deriving each axis's absolute sigma from one pixel count
    keeps the footprint the same shape in pixel space regardless of the
    calibrated ranges, so sweeping `sigma_pixels` varies only smoothing.
    """

    def __init__(
        self,
        birth_range: tuple[float, float],
        pers_range: tuple[float, float],
        resolution: int = 64,
        sigma_pixels: float = 2.0,
        weight_fn: Callable[[np.ndarray], np.ndarray] = linear_weight,
    ):
        if sigma_pixels <= 0:
            raise ValueError(f"sigma_pixels must be positive, got {sigma_pixels!r}")

        self._birth_range = birth_range
        self._pers_range = pers_range
        self._resolution = resolution
        self._sigma_pixels = sigma_pixels
        self._weight_fn = weight_fn

        birth_pixel = (birth_range[1] - birth_range[0]) / resolution
        pers_pixel = (pers_range[1] - pers_range[0]) / resolution
        self._sigma_birth = sigma_pixels * birth_pixel
        self._sigma_pers = sigma_pixels * pers_pixel

        # Pixel edges along each axis (resolution+1 boundaries), computed once.
        self._birth_edges = np.linspace(*birth_range, resolution + 1)
        self._pers_edges = np.linspace(*pers_range, resolution + 1)

    def transform(self, diagram: PersistenceDiagram, dim: int) -> np.ndarray:
        """Return a (resolution, resolution) image for homology `dim`.
        Axis 0 is persistence (descending), axis 1 is birth."""
        pairs = diagram.finite_pairs(dim)
        if len(pairs) == 0:
            return np.zeros((self._resolution, self._resolution))

        births = pairs[:, 0]
        persistences = pairs[:, 1] - pairs[:, 0]  # birth-death -> birth-pers
        weights = self._weight_fn(persistences)

        birth_mass = _box_mass(self._birth_edges, births, self._sigma_birth)
        pers_mass = _box_mass(self._pers_edges, persistences, self._sigma_pers)

        # image[j, k] = sum_i weight_i * pers_mass[j, i] * birth_mass[k, i]
        image = (weights[None, :] * pers_mass) @ birth_mass.T

        # Flip so persistence increases upward when viewed as an image.
        return np.flipud(image)

    @property
    def params(self) -> dict[str, Any]:
        return {
            "birth_range": self._birth_range,
            "pers_range": self._pers_range,
            "resolution": self._resolution,
            "sigma_pixels": self._sigma_pixels,
            "sigma_birth": self._sigma_birth,
            "sigma_pers": self._sigma_pers,
            "weight_fn": self._weight_fn.__name__,
        }
