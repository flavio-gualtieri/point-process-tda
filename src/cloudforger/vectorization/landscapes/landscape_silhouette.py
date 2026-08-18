# src/cloudforger/vectorization/landscapes/landscape_silhouette.py
"""Per-dimension transformers over a fixed, calibrated 1-D grid, built on
tent.py's primitives -- the landscape/silhouette analogue of
persistence_images/persistence_image.py's PersistenceImager, and
multi_channel.py's MultiChannelImager for the "one transformer per homology
dim" wrapper. See calibrated.py for how grid/K/p get fit from a diagram
sample instead of hand-picked."""

from __future__ import annotations

from typing import Any

import numpy as np

from ...core.diagram import PersistenceDiagram
from .tent import landscape_from_tents, silhouette_from_tents, tent


class LandscapeTransformer:
    """Vectorizes one homology dimension into a (K, G) persistence-
    landscape raster: grid/K are fixed at construction (never inferred from
    a single diagram, same reasoning as PersistenceImager's fixed
    birth_range/pers_range -- a grid point means the same filtration value
    across every diagram this transformer is applied to)."""

    def __init__(self, t_min: float, T: float, G: int, K: int):
        if not (T > t_min):
            raise ValueError(f"LandscapeTransformer: T={T} must exceed t_min={t_min}.")
        if K < 1:
            raise ValueError(f"LandscapeTransformer: K must be >= 1, got {K}.")
        self._t_min = float(t_min)
        self._T = float(T)
        self._G = int(G)
        self._K = int(K)
        self._grid = np.linspace(self._t_min, self._T, self._G)

    @property
    def grid(self) -> np.ndarray:
        return self._grid

    def transform(self, diagram: PersistenceDiagram, dim: int) -> np.ndarray:
        """Return the (K, G) landscape for homology `dim`."""
        pairs = diagram.finite_pairs(dim)
        tents = tent(pairs, self._grid)
        return landscape_from_tents(tents, self._K)

    @property
    def params(self) -> dict[str, Any]:
        return {"t_min": self._t_min, "T": self._T, "G": self._G, "K": self._K}


class SilhouetteTransformer:
    """Vectorizes one homology dimension into a (G,) persistence-silhouette
    curve (plus its unnormalized numerator, see tent.silhouette_from_tents)
    on a fixed, calibrated grid. p is an experimental axis (see
    calibrated.py's module docstring) fixed per transformer instance, not
    swept inside one run."""

    def __init__(self, t_min: float, T: float, G: int, p: float):
        if not (T > t_min):
            raise ValueError(f"SilhouetteTransformer: T={T} must exceed t_min={t_min}.")
        if p < 0:
            raise ValueError(f"SilhouetteTransformer: p must be >= 0, got {p}.")
        self._t_min = float(t_min)
        self._T = float(T)
        self._G = int(G)
        self._p = float(p)
        self._grid = np.linspace(self._t_min, self._T, self._G)

    @property
    def grid(self) -> np.ndarray:
        return self._grid

    def transform(self, diagram: PersistenceDiagram, dim: int) -> dict[str, np.ndarray]:
        """Return {"silhouette": (G,), "silhouette_unnormalized": (G,)} for
        homology `dim` -- both emitted (see tent.silhouette_from_tents'
        docstring for why the unnormalized numerator is kept, not just the
        normalized curve: the denominator discards feature count, which
        likely matters for parameter recovery)."""
        pairs = diagram.finite_pairs(dim)
        tents = tent(pairs, self._grid)
        normalized, unnormalized = silhouette_from_tents(tents, pairs, self._p)
        return {"silhouette": normalized, "silhouette_unnormalized": unnormalized}

    @property
    def params(self) -> dict[str, Any]:
        return {"t_min": self._t_min, "T": self._T, "G": self._G, "p": self._p}


class MultiChannelLandscape:
    """Applies one LandscapeTransformer per homology dimension -- the
    landscape analogue of persistence_images/multi_channel.py's
    MultiChannelImager."""

    def __init__(self, transformers: dict[int, LandscapeTransformer]):
        self._transformers = transformers

    def transform(self, diagram: PersistenceDiagram) -> dict[int, np.ndarray]:
        return {dim: t.transform(diagram, dim) for dim, t in self._transformers.items()}

    @property
    def dimensions(self) -> list[int]:
        return sorted(self._transformers.keys())

    @property
    def params(self) -> dict[str, Any]:
        return {dim: t.params for dim, t in self._transformers.items()}


class MultiChannelSilhouette:
    """Applies one SilhouetteTransformer per homology dimension -- the
    silhouette analogue of MultiChannelImager."""

    def __init__(self, transformers: dict[int, SilhouetteTransformer]):
        self._transformers = transformers

    def transform(self, diagram: PersistenceDiagram) -> dict[int, dict[str, np.ndarray]]:
        return {dim: t.transform(diagram, dim) for dim, t in self._transformers.items()}

    @property
    def dimensions(self) -> list[int]:
        return sorted(self._transformers.keys())

    @property
    def params(self) -> dict[str, Any]:
        return {dim: t.params for dim, t in self._transformers.items()}
