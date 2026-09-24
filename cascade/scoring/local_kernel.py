"""Kernel score on local configurations: the energy score, made local and Hilbertian.

    S(P, x) = 1/2 E k(Phi, Phi')  -  mean_i E k(phi_i, Phi)                 lower is better

phi_i are x's local configurations, Phi, Phi' configurations of patterns simulated from P, drawn
from DIFFERENT simulations (configurations of one pattern overlap and are dependent, and the
expectation assumes independent draws). With k characteristic this is strictly proper for the law
of radius-R configurations: E_Q S = 1/2 ||mu_P - mu_Q||^2 + const in the kernel's feature space.

A configuration: the points within R of a centre, translated so the centre is the origin, divided
by R. Coordinates are x sqrt(n) first, so R is in mean spacings. Two kinds of centre:
  point  the pattern's own points (the centre itself excluded): sees hard cores and cluster bursts
  grid   a regular grid of spacing R: sees the voids between clusters
Each (R, kind) is a component with its own kernel; the total is their sum (a sum of proper
scores is proper).

EMBEDDING (the one liberty taken with the definition): mu_phi = sum_y kappa_h(. - y) is evaluated
on a G x G grid over [-1, 1]^2 rather than in closed form, so ||mu_phi - mu_psi|| is a Euclidean
distance between vectors. Costs O(configurations x points x G^2) whatever the cluster sizes (the
closed form is O(points^2) per PAIR, and a tight cluster puts hundreds of points in one
configuration), and the kernel stays Hilbertian, so propriety is untouched. h >= grid spacing keeps
the rasterization faithful.

KERNEL WIDTH tau is fixed per cloud and component BEFORE any model is scored: tau0 = the median
distance between x's own configurations, times each multiplier in TAU_MULTS. The same tau for
every model on a cloud is what makes their scores comparable. `lin` is the tau -> infinity limit,
up to scale  E d^2(phi, Phi) - 1/2 E d^2(Phi, Phi'): it only sees the MEAN embedding, which for
point centres is a smoothed pair correlation (circular with K/g). The gap between finite tau and
`lin` is the higher-order structure the score adds.

BOUNDARY: with the window side sqrt(n) spacings, centres are kept at least R from the edge when that
leaves >= MIN_INTERIOR of the window, and configurations are otherwise cut off by the window.
The mode is chosen from x's n and applied to every simulation of the cloud.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

TAU_MULTS = (0.5, 1.0, 2.0, 4.0)
MIN_INTERIOR = 0.25


class Component:
    """One (R, kind) component, with its kernel width fixed from the observed pattern."""

    def __init__(self, x: np.ndarray, R: float, kind: str, h: float, grid: int, centres: int, rng):
        self.R, self.kind, self.h, self.max_centres = R, kind, h, centres
        L = np.sqrt(len(x))
        self.erode = (L - 2 * R) ** 2 >= MIN_INTERIOR * L**2
        g = np.linspace(-1, 1, grid)
        self.nodes = np.stack(np.meshgrid(g, g), -1).reshape(-1, 2)
        self.ex = self.embed(x, rng)
        d = cdist(self.ex, self.ex)
        pos = d[np.triu_indices(len(d), 1)]
        pos = pos[pos > 0]
        self.tau0 = float(np.median(pos)) if len(pos) else 1.0

    def embed(self, points: np.ndarray, rng, max_centres: int | None = None) -> np.ndarray:
        """(configurations, G^2) embeddings of one pattern's configurations."""
        max_centres = max_centres or self.max_centres
        L = np.sqrt(len(points))
        X = points * L
        lo, hi = (self.R, L - self.R) if self.erode else (0.0, L)
        if hi <= lo:
            return np.empty((0, len(self.nodes)))
        if self.kind == "point":
            idx = np.flatnonzero(np.all((X >= lo) & (X <= hi), axis=1))
            idx = rng.choice(idx, min(max_centres, len(idx)), replace=False) if len(idx) else idx
            centres = X[idx]
        else:
            axis = np.arange(lo + self.R / 2, hi, self.R)
            centres = np.stack(np.meshgrid(axis, axis), -1).reshape(-1, 2)
            if len(centres) > max_centres:
                centres = centres[rng.choice(len(centres), max_centres, replace=False)]
            idx = None
        tree = cKDTree(X)
        out = np.zeros((len(centres), len(self.nodes)))
        for c, nbrs in enumerate(tree.query_ball_point(centres, self.R)):
            if idx is not None:
                nbrs = [j for j in nbrs if j != idx[c]]
            if nbrs:
                off = (X[nbrs] - centres[c]) / self.R
                out[c] = np.exp(-cdist(self.nodes, off, "sqeuclidean") / (2 * self.h**2)).sum(1)
        return out

    def scores(self, sims: list[np.ndarray], rng, pool: int) -> dict[str, float]:
        """{tau label: score} for one model, from its simulated patterns."""
        per = max(1, pool // len(sims))
        embs = [self.embed(s, rng, per) for s in sims]
        which = np.concatenate([np.full(len(e), i) for i, e in enumerate(embs)])
        E = np.concatenate(embs)
        if len(E) == 0 or len(self.ex) == 0:
            return {k: np.nan for k in [*map(str, TAU_MULTS), "lin"]}
        d_cross = cdist(self.ex, E, "sqeuclidean")
        d_self = cdist(E, E, "sqeuclidean")[which[:, None] != which[None, :]]
        out = {}
        for m in TAU_MULTS:
            t2 = 2 * (m * self.tau0) ** 2
            out[str(m)] = float(0.5 * np.exp(-d_self / t2).mean() - np.exp(-d_cross / t2).mean())
        out["lin"] = float(d_cross.mean() - 0.5 * d_self.mean())
        return out
