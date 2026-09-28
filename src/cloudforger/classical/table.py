"""The classical feature table: a short vector of summary-function samples per bank pattern, the input
of the table learners' `classical` source.

Input   <data>/bank/<family>/{points.npz, manifest.csv}
Output  <data>/tables/classical/<family>.npz, rows in manifest order
            case_id   (P,)
            X         (P, 111)   float32
            names     (111,)     column names, e.g. L_fixed@0.0625, G_sqrtn@1.0, log_n

Two grids, because the two departures from CSR live at different scales:
  L on the fixed axis (r up to 1/4 of the window)  -- large-scale clustering (LGCP, nested parents)
  L, F, G, J on the sqrt(n) axis (u = r sqrt(n) <= 2) -- small-scale repulsion and tight clusters
Each is subsampled at SAMPLES log-spaced grid positions, plus log n.
"""

from __future__ import annotations

import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd

from ..paths import BANK, TABLES
from . import fgj
from .functions import GRID_SIZE, axis, radii
from .lfunction import l_minus_r

FIXED, SQRTN = {"name": "fixed"}, {"name": "sqrtn", "u_max": 2.0}
SAMPLES = 24
IDX = np.unique(np.geomspace(1, GRID_SIZE, SAMPLES).round().astype(int) - 1)


def names() -> list[str]:
    fx, sq = axis(FIXED)[IDX], axis(SQRTN)[IDX]
    cols = [f"L_fixed@{r:.4g}" for r in fx]
    cols += [f"{f}_sqrtn@{u:.3g}" for f in ("L", "F", "G", "J") for u in sq]
    return cols + ["log_n"]


def featurize(points: np.ndarray) -> np.ndarray:
    n = len(points)
    r_sq = radii(SQRTN, n)
    f = fgj.f_function(points, r_sq)
    g = fgj.g_function(points, r_sq)
    parts = [l_minus_r(points, radii(FIXED, n)) * np.sqrt(n),   # sqrt(n): less n-dependent
             l_minus_r(points, r_sq) * np.sqrt(n),
             f, g, fgj.j_function(f, g)]
    return np.concatenate([p[IDX] for p in parts] + [[np.log(n)]])


_POINTS = _OFFSETS = None


def _rows(bounds: tuple[int, int]) -> np.ndarray:
    lo, hi = bounds
    return np.stack([featurize(_POINTS[_OFFSETS[i]:_OFFSETS[i + 1]]) for i in range(lo, hi)])


def run(family: str, workers: int) -> Path:
    """Every pattern of one family. Skips the work if the output already exists."""
    global _POINTS, _OFFSETS
    path = TABLES / "classical" / f"{family}.npz"
    if path.exists():
        return path
    manifest = pd.read_csv(BANK / family / "manifest.csv")
    z = np.load(BANK / family / "points.npz")
    _POINTS, _OFFSETS = z["points"], z["offsets"]
    chunks = [(lo, min(lo + 500, len(manifest))) for lo in range(0, len(manifest), 500)]
    with mp.get_context("fork").Pool(workers) as pool:
        X = np.concatenate(pool.map(_rows, chunks)).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez(tmp, case_id=manifest.case_id.to_numpy(str), X=X, names=np.array(names()))
    tmp.replace(path)
    return path
