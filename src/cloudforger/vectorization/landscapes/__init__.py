# src/cloudforger/vectorization/landscapes/__init__.py
"""Persistence landscapes + silhouettes: diagram -> fixed-size (K, G) or
(G,) array, built from the shared tent primitive (tent.py) on a calibrated
1-D grid (calibrated.py) -- the sibling of vectorization/persistence_images/
for 1-D-grid vectorizers, registry-free like that package (see
persistence_images/__init__.py's own docstring for why)."""

from .tent import landscape_from_tents, silhouette_from_tents, tent
from .landscape_silhouette import (
    LandscapeTransformer,
    MultiChannelLandscape,
    MultiChannelSilhouette,
    SilhouetteTransformer,
)
from .calibrated import build_calibrated_landscape, build_calibrated_silhouette, choose_K

__all__ = [
    "tent",
    "landscape_from_tents",
    "silhouette_from_tents",
    "LandscapeTransformer",
    "SilhouetteTransformer",
    "MultiChannelLandscape",
    "MultiChannelSilhouette",
    "build_calibrated_landscape",
    "build_calibrated_silhouette",
    "choose_K",
]
