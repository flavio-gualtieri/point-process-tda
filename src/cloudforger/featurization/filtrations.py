"""Point pattern -> finite persistence pairs {dim: (m, 2) array of (birth, death)}, dims 0..maxdim.

rips   Vietoris-Rips on Euclidean distance.
dtm    DTM-Rips (Anai et al. 2020) in GUDHI's DTMRipsComplex convention: f(x) = (mean of d^q over the k nearest
       neighbours of x, x itself included)^(1/q); vertex x enters at 2 f(x), edge xy at
       max(d(x, y) + f(x) + f(y), 2 f(x), 2 f(y)). A flag filtration, so ripser on that matrix gives GUDHI's
       diagrams exactly, in O(n^2) memory instead of GUDHI's O(n^3).
alpha  Alpha complex. GUDHI's values are squared radii r^2; `scale` maps them to r^2, r, or 2r (the Rips axis).

Essential classes (death = inf) are dropped.
"""

from __future__ import annotations

import numpy as np
from gudhi import AlphaComplex
from ripser import ripser
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

ALPHA_SCALES = {"squared_radius": lambda a: a, "radius": np.sqrt, "diameter": lambda a: 2 * np.sqrt(a)}


def _finite(pairs) -> np.ndarray:
    pairs = np.asarray(pairs, dtype=float).reshape(-1, 2)
    return pairs[np.isfinite(pairs[:, 1])]


def rips(points: np.ndarray, maxdim: int = 1) -> dict[int, np.ndarray]:
    return {d: _finite(p) for d, p in enumerate(ripser(points, maxdim=maxdim)["dgms"])}


def dtm(points: np.ndarray, k: int, q: float = 2.0, maxdim: int = 1) -> dict[int, np.ndarray]:
    knn, _ = cKDTree(points).query(points, k=np.arange(1, k + 1))
    f = np.mean(knn**q, axis=1) ** (1 / q)
    m = np.maximum(cdist(points, points) + f[:, None] + f[None, :], 2 * np.maximum(f[:, None], f[None, :]))
    np.fill_diagonal(m, 2 * f)
    return {d: _finite(p) for d, p in enumerate(ripser(m, maxdim=maxdim, distance_matrix=True)["dgms"])}


def alpha(points: np.ndarray, scale: str = "diameter", maxdim: int = 1) -> dict[int, np.ndarray]:
    st = AlphaComplex(points=points).create_simplex_tree()
    st.compute_persistence()
    return {d: ALPHA_SCALES[scale](_finite(st.persistence_intervals_in_dimension(d))) for d in range(maxdim + 1)}


FILTRATIONS = {"rips": rips, "dtm": dtm, "alpha": alpha}


def tag(spec: dict) -> str:
    """Directory name for one config entry: rips, dtm_k5, alpha_diameter."""
    name = spec["name"]
    if name == "dtm":
        return f"dtm_k{spec['k']}"
    if name == "alpha":
        return f"alpha_{spec.get('scale', 'diameter')}"
    return name
