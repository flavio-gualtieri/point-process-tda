"""The score that failed, kept as the baseline on the same clouds: an energy score with exact W_1
between whole patterns, both thinned to a common size <= max_points and matched by assignment.
    ES(P, x) = E W(S, x) - 1/2 E W(S, S')                                   lower is better
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


def _w1(a, b, rng, max_points):
    k = min(len(a), len(b), max_points)
    a = a[rng.choice(len(a), k, replace=False)]
    b = b[rng.choice(len(b), k, replace=False)]
    cost = cdist(a, b)
    i, j = linear_sum_assignment(cost)
    return cost[i, j].mean()


def score(x: np.ndarray, sims: list[np.ndarray], rng, max_points: int = 400) -> float:
    cross = np.mean([_w1(x, s, rng, max_points) for s in sims])
    within = np.mean([_w1(sims[i], sims[j], rng, max_points)
                      for i in range(len(sims)) for j in range(i + 1, len(sims))])
    return float(cross - within / 2)
