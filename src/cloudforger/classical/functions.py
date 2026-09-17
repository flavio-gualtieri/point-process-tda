"""Point pattern -> summary-function curves on a radius grid, and the two grid conventions.

The registry mirrors featurization.filtrations: a name maps to one curve of fixed length, and
`curves` computes all four in one pass because J is a function of F and G.

    L   Ripley's L(r) - r, isotropic edge correction (lfunction.py)
    F   empty-space function, G  nearest-neighbour function, J  (1 - G) / (1 - F)  (fgj.py)

Each curve is its own channel with its own encoder, so the four need not share a grid, a length or
an r_max -- they only ever meet as concatenated embeddings. Two conventions are available:

    fixed   r = RADII, spatstat's Kest axis (window side / 4, 512 intervals). What the VIHRS paper
            uses, and the only grid on which L is directly comparable to it.
    sqrtn   r = u / sqrt(n) for u on (0, u_max]. F and G are distance CDFs, so they saturate around
            the mean nearest-neighbour distance ~ 1/(2 sqrt(n)): on the fixed axis they sit at a
            constant 1.0 over 37-85% of the grid, and the saturation point moves 4x across this
            sweep's n range (44-961), so one encoder sees the same rise at wildly different
            positions. In u units it is near n-invariant (measured u99: median 1.2, q99 1.74,
            max 2.04). This is the same rescaling the PH arm applies to diagram coordinates
            (vectorization.persistence_images.Scaling, coords="sqrt_n").

Which convention wins is an empirical question, so it is a featurization-time tag rather than a
default: unlike a diagram, a gridded curve cannot be re-gridded losslessly afterwards (F and G are
step functions with up to n jumps, and for n ~ 900 a fixed-axis curve puts almost all of them
inside its first ~130 samples).
"""

from __future__ import annotations

import numpy as np

from . import fgj
from .lfunction import RADII, l_minus_r

NAMES = ("L", "F", "G", "J")
GRID_SIZE = len(RADII)      # every curve has this length, whatever the convention


def radii(spec: dict, n: int) -> np.ndarray:
    """The r values one pattern's curves are evaluated on."""
    if spec["name"] == "fixed":
        return RADII
    u = spec["u_max"] * np.arange(1, GRID_SIZE + 1) / GRID_SIZE
    return u / np.sqrt(n)


def axis(spec: dict) -> np.ndarray:
    """The grid's own axis, shared by every pattern: r under `fixed`, u = r sqrt(n) under `sqrtn`.
    Stored alongside the curves so a file is self-describing for plots and diagnostics."""
    if spec["name"] == "fixed":
        return RADII
    return spec["u_max"] * np.arange(1, GRID_SIZE + 1) / GRID_SIZE


def curves(points: np.ndarray, spec: dict, f_grid_size: int = fgj.F_GRID_SIZE) -> dict[str, np.ndarray]:
    """All four curves for one pattern. F and G are computed once and J derived from them."""
    r = radii(spec, len(points))
    f = fgj.f_function(points, r, f_grid_size)
    g = fgj.g_function(points, r)
    return {"L": l_minus_r(points, r), "F": f, "G": g, "J": fgj.j_function(f, g)}


def tag(spec: dict) -> str:
    """Directory name for one config entry: fixed, sqrtn_u2."""
    return "fixed" if spec["name"] == "fixed" else f"sqrtn_u{spec['u_max']:g}"
