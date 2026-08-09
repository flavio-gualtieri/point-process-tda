# src/cloudforger/vectorizers/signed_measure_image.py

from __future__ import annotations

from typing import Any

from ..core.signed_measure import SignedMeasure
from .persistence_image import _box_mass

import numpy as np


class SignedMeasureImager:
    """Vectorizes one homology dimension of a SignedMeasure into a
    fixed-size raster, following Loiseaux et al. (2023), "Stable
    Vectorization of Multiparameter Persistent Homology using Signed
    Barcodes as Measures" (Definition 6, the convolution vectorization).

    This is the bifiltration counterpart of PersistenceImager, and reuses its
    `_box_mass`: each pixel is the exact integral of the smoothed measure over
    the pixel's box, computed in closed form from the Gaussian CDF, not a point
    sample at the pixel centre. That is strictly better than the point-sampled
    convolution the paper describes, and costs the same.

    Three deliberate differences from PersistenceImager:

    1.  No weight function. A persistence image needs `linear_weight` to
        suppress near-diagonal points, which are the noisy ones. A signed
        measure has no diagonal: its integer weights are the invariant itself
        (they come from the Mobius inversion of the Hilbert function), and
        rescaling them would destroy the reconstruction property that makes
        the measure a faithful encoding of the module. Weights are used as-is,
        signs included.

    2.  Ranges are the bifiltration's own computation grid, not a separately
        calibrated pair of ranges. The grid already defines what the two axes
        mean -- it is the lattice the atoms were snapped onto -- so calibrating
        the image independently could only introduce disagreement, placing
        atoms in pixels that do not correspond to the cells they came from.
        Passing the grid through is what keeps the two stages consistent.

    3.  No vertical flip. PersistenceImager flips so that persistence increases
        upward, matching the usual way persistence diagrams are drawn. Both
        axes here are filtration coordinates with no such convention, so the
        raw orientation is kept: axis 0 (rows) is the second filtration
        parameter, axis 1 (columns) is the first. CoordConvPIEncoder appends
        symmetric linspace coordinate channels, so it is insensitive to the
        choice -- what matters is only that it is the same for every image.

    Bandwidth is `sigma_pixels`, one pixel count shared by both axes, for
    exactly the reason given in PersistenceImager's docstring: the two axes
    have unrelated spans (a radius axis of ~0.25 against a codensity axis of
    ~0.13 here), so an absolute sigma shared between them would confound
    "how much smoothing" with "how distorted the footprint is".
    """

    def __init__(
        self,
        grid: tuple[np.ndarray, np.ndarray],
        resolution: int = 64,
        sigma_pixels: float = 0.75,
        clip_mass: bool = True,
    ):
        if sigma_pixels <= 0:
            raise ValueError(f"sigma_pixels must be positive, got {sigma_pixels!r}")

        axis0, axis1 = np.asarray(grid[0], float), np.asarray(grid[1], float)
        if axis0.ndim != 1 or axis1.ndim != 1 or len(axis0) < 2 or len(axis1) < 2:
            raise ValueError("grid must be two 1-D arrays of length >= 2.")

        self._axis0_range = (float(axis0[0]), float(axis0[-1]))
        self._axis1_range = (float(axis1[0]), float(axis1[-1]))
        self._resolution = resolution
        self._sigma_pixels = sigma_pixels
        self._clip_mass = clip_mass

        pixel0 = (self._axis0_range[1] - self._axis0_range[0]) / resolution
        pixel1 = (self._axis1_range[1] - self._axis1_range[0]) / resolution
        self._sigma0 = sigma_pixels * pixel0
        self._sigma1 = sigma_pixels * pixel1

        self._edges0 = np.linspace(*self._axis0_range, resolution + 1)
        self._edges1 = np.linspace(*self._axis1_range, resolution + 1)

    def transform(self, measure: SignedMeasure, dim: int) -> np.ndarray:
        """Return a (resolution, resolution) image for homology `dim`.
        Rows are the second filtration parameter, columns the first.
        Values are signed."""
        atoms, weights = measure.atoms(dim)
        if len(atoms) == 0:
            return np.zeros((self._resolution, self._resolution))

        weights = np.asarray(weights, dtype=float)
        mass0 = _box_mass(self._edges0, atoms[:, 0], self._sigma0)
        mass1 = _box_mass(self._edges1, atoms[:, 1], self._sigma1)

        if self._clip_mass:
            # Renormalize each atom's smoothing kernel to unit mass inside the
            # grid, i.e. fold the off-grid tail back onto the boundary pixels.
            #
            # Without this the imager is NOT mass-preserving, and the loss is
            # asymmetric in a way that tracks the labels: H0 atoms with weight
            # +1 pile up at axis0 == 0 (every vertex is born at radius zero),
            # sitting exactly on the lower edge, so half of each one's Gaussian
            # falls off the grid -- while the negative atoms, which record
            # merges at positive radius, sit in the interior and keep ~all of
            # theirs. Measured on real nested_thomas H0: positives retained
            # 0.755 of their mass, negatives 0.954, turning a true total mass
            # of +1 into an image summing to -195. Since how many atoms land on
            # the boundary depends on the point pattern, that is a
            # parameter-dependent distortion, not a constant offset a later
            # z-score could absorb.
            #
            # Deliberately done here rather than in _box_mass, which is shared
            # with PersistenceImager: changing it there would silently alter
            # every persistence image already computed.
            total0, total1 = mass0.sum(axis=0), mass1.sum(axis=0)
            alive = (total0 > 1e-6) & (total1 > 1e-6)
            if not alive.all():
                # Atoms essentially outside the grid: clipping them would dump
                # their full weight onto an edge pixel, which is worse than
                # dropping them. Expected to be rare -- bifiltration_grid uses
                # percentile coverage, so a few percent of the second axis is
                # outside by construction.
                mass0, mass1, weights = mass0[:, alive], mass1[:, alive], weights[alive]
                total0, total1 = total0[alive], total1[alive]
                if weights.size == 0:
                    return np.zeros((self._resolution, self._resolution))
            mass0 = mass0 / total0[None, :]
            mass1 = mass1 / total1[None, :]

        # image[j, k] = sum_i weight_i * mass1[j, i] * mass0[k, i]
        return (weights[None, :] * mass1) @ mass0.T

    @property
    def params(self) -> dict[str, Any]:
        return {
            "axis0_range": self._axis0_range,
            "axis1_range": self._axis1_range,
            "resolution": self._resolution,
            "sigma_pixels": self._sigma_pixels,
            "clip_mass": self._clip_mass,
            "sigma0": self._sigma0,
            "sigma1": self._sigma1,
        }


class MultiDegreeSignedMeasureImager:
    """Applies one SignedMeasureImager per homology degree, sharing the grid.

    Counterpart of MultiChannelImager. Unlike that class, every degree shares
    one grid rather than getting independently calibrated bounds: the grid is a
    property of the bifiltration, not of a degree, and H0 and H1 of the same
    module live on the same two axes by construction.
    """

    def __init__(self, imagers: dict[int, SignedMeasureImager]):
        self._imagers = imagers

    def transform(self, measure: SignedMeasure) -> dict[int, np.ndarray]:
        return {dim: im.transform(measure, dim) for dim, im in self._imagers.items()}

    @property
    def dimensions(self) -> list[int]:
        return sorted(self._imagers)

    @property
    def params(self) -> dict[str, Any]:
        return {dim: im.params for dim, im in self._imagers.items()}


def build_signed_measure_imagers(
    grid: tuple[np.ndarray, np.ndarray],
    homology_dims: tuple[int, ...] = (0,),
    resolution: int = 64,
    sigma_pixels: float = 0.75,
    clip_mass: bool = True,
) -> MultiDegreeSignedMeasureImager:
    """Counterpart of build_calibrated_imager. Takes no diagrams, because the
    calibration already happened upstream in calibration.bifiltration_grid():
    the grid was needed before any measure could be computed."""
    return MultiDegreeSignedMeasureImager(
        {
            dim: SignedMeasureImager(grid, resolution=resolution,
                                sigma_pixels=sigma_pixels, clip_mass=clip_mass)
            for dim in homology_dims
        }
    )