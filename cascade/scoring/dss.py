"""Dawid-Sebastiani score (Gaussian synthetic log-likelihood) on scale-free geometric statistics.

    S(P, x) = log det Sigma_P + (T(x) - mu_P)' Sigma_P^-1 (T(x) - mu_P)       lower is better

mu_P, Sigma_P are the mean and Ledoit-Wolf-shrunk covariance of T over M patterns simulated from P.
Proper (not strictly: two models agreeing on the first two moments of T tie). Invariant to affine
maps of T, so the statistics need no standardization. With the same M and d for every model on a
cloud, the finite-M bias of log det is common to all of them and cancels in a comparison.

Statistics, 14 in six groups (GROUPS), all on coordinates x sqrt(n) so the mean spacing is ~1:
  voronoi    log CV, skewness, q10, q90 of Voronoi cell areas / mean area
  neighbours log variance of the number of Delaunay neighbours
  delaunay   q10, q50 of the smallest angle of each Delaunay triangle
  mst        q50, q90 of minimum-spanning-tree edges that are not nearest-neighbour edges
  euler      Euler characteristic per point of the union of discs, r = 0.25, 0.5, 0.75 spacings
  h1         log total and log max H1 persistence (alpha filtration), per point
CIRCULARITY: mst is the alpha H0 diagram (its deaths ARE the MST edges), euler is beta0 - beta1 of
the alpha diagrams, h1 is the H1 diagram. They are fair only when no pipeline stage uses PH
features; NO_PH is the subset for when one does.
BOUNDARY: nothing is excluded. Simulations share the unit window, so edge effects hit x and every
simulation alike; cells are clipped to the window exactly (the reflection trick).
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial import Delaunay, Voronoi, cKDTree
from scipy.stats import skew
from sklearn.covariance import LedoitWolf

GROUPS = {"voronoi": 4, "neighbours": 1, "delaunay": 2, "mst": 2, "euler": 3, "h1": 2}
NO_PH = ("voronoi", "neighbours", "delaunay")
EULER_RADII = (0.25, 0.5, 0.75)


def columns(groups) -> np.ndarray:
    """Column indices of T for a subset of GROUPS."""
    start, out = 0, []
    for g, width in GROUPS.items():
        if g in groups:
            out += range(start, start + width)
        start += width
    return np.array(out)


def _voronoi_areas(X: np.ndarray, L: float) -> np.ndarray:
    """Cell areas clipped to [0, L]^2: reflecting the points across the four sides makes the
    original points' cells exactly their clipped cells."""
    refl = [X, X * [-1, 1], X * [1, -1], [2 * L, 0] + X * [-1, 1], [0, 2 * L] + X * [1, -1]]
    vor = Voronoi(np.concatenate(refl))
    areas = np.empty(len(X))
    for i in range(len(X)):
        v = vor.vertices[vor.regions[vor.point_region[i]]]
        v = v[np.argsort(np.arctan2(*(v - v.mean(0)).T[::-1]))]
        areas[i] = 0.5 * abs(np.dot(v[:, 0], np.roll(v[:, 1], 1)) - np.dot(v[:, 1], np.roll(v[:, 0], 1)))
    return areas


def _alpha(X: np.ndarray):
    import gudhi
    st = gudhi.AlphaComplex(points=X).create_simplex_tree()   # filtration value = squared radius
    st.compute_persistence()
    return st.persistence_intervals_in_dimension(0), st.persistence_intervals_in_dimension(1)


def statistics(points: np.ndarray) -> np.ndarray:
    n = len(points)
    L = np.sqrt(n)
    X = points * L

    a = _voronoi_areas(X, L)
    a = a / a.mean()
    t = [np.log(a.std() / a.mean()), skew(a), *np.quantile(a, [0.1, 0.9])]

    tri = Delaunay(X)
    s = tri.simplices
    edges = np.unique(np.sort(np.concatenate([s[:, [0, 1]], s[:, [1, 2]], s[:, [0, 2]]])), axis=0)
    degree = np.bincount(edges.ravel(), minlength=n)
    t.append(np.log(degree.var() + 1e-9))

    p = X[s]
    sides = [np.linalg.norm(p[:, (k + 1) % 3] - p[:, (k + 2) % 3], axis=1) for k in range(3)]
    angles = []
    for k in range(3):                               # law of cosines, angle opposite side k
        a_, b_, c_ = sides[k], sides[(k + 1) % 3], sides[(k + 2) % 3]
        angles.append(np.arccos(np.clip((b_**2 + c_**2 - a_**2) / (2 * b_ * c_ + 1e-12), -1, 1)))
    t += list(np.quantile(np.min(angles, axis=0), [0.1, 0.5]))

    length = np.linalg.norm(X[edges[:, 0]] - X[edges[:, 1]], axis=1)
    mst = minimum_spanning_tree(coo_matrix((length, (edges[:, 0], edges[:, 1])), shape=(n, n))).tocoo()
    nn = cKDTree(X).query(X, k=2)[1][:, 1]
    is_nn = (nn[mst.row] == mst.col) | (nn[mst.col] == mst.row)
    rest = mst.data[~is_nn]
    t += list(np.quantile(rest, [0.5, 0.9])) if len(rest) else [np.nan, np.nan]

    h0, h1 = _alpha(X)
    for r in EULER_RADII:
        b0 = np.sum((h0[:, 0] <= r**2) & (h0[:, 1] > r**2))
        b1 = np.sum((h1[:, 0] <= r**2) & (h1[:, 1] > r**2)) if len(h1) else 0
        t.append((b0 - b1) / n)
    pers = np.sqrt(h1[:, 1]) - np.sqrt(h1[:, 0]) if len(h1) else np.zeros(1)
    t += [np.log(pers.sum() / n + 1e-6), np.log(pers.max() + 1e-6)]
    return np.array(t, dtype=float)


def score(t_x: np.ndarray, t_sims: np.ndarray) -> float:
    """DSS of the observed statistics against the simulated ones (M x d)."""
    fill = np.nanmedian(t_sims, axis=0)
    t_sims = np.where(np.isnan(t_sims), fill, t_sims)
    t_x = np.where(np.isnan(t_x), fill, t_x)
    lw = LedoitWolf().fit(t_sims)
    cov = lw.covariance_ + 1e-9 * np.eye(len(t_x))
    diff = t_x - lw.location_
    sign, logdet = np.linalg.slogdet(cov)
    return float(logdet + diff @ np.linalg.solve(cov, diff))
