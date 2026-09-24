"""Wasserstein-based score of a fitted model against one observed pattern.

The question the evaluation asks: does the pipeline's fitted model (family_hat, theta_hat)
generate patterns like the observed one? It does not ask whether family_hat is the true label,
which is the point: a near-CSR Thomas cloud fitted as poisson is fine if poisson patterns look like it.

CHOICE 1 -- the score is an ENERGY SCORE with Wasserstein as the distance between patterns

    ES(model, x) = E W(S, x) - 1/2 E W(S, S')          S, S' independent patterns from the model

  Why not the plain W(x, S) between the observed pattern and one simulated from the fit? Because two
  independent patterns of the SAME stationary process are nowhere near each other point-by-point
  (clusters land in different places), so W(x, S) is large even for the true model, and how large
  depends on the family, the parameters and n. A single W(x, S) cannot tell a good fit from a bad one.
  The energy score subtracts the model's own pattern-to-pattern spread: under the true model, x is
  just another draw S'' and the two terms balance. Lower is better, and it never touches a summary
  function (L, F, G, J, delta-tilde), which is what keeps the evaluation off the features the
  classifiers use.
  Caveat: the energy score is guaranteed PROPER (the true model minimises it in expectation) when
  the distance is of negative type. Exact W_1 between planar measures is not guaranteed to be;
  sliced W_1 is (an average of 1-D W_1's, each an L1 distance between CDFs). So with `exact` the
  empirical check is the oracle itself: mean regret should come out >= 0. If it does not, switch
  to a sliced ground (CHOICE 2).

  Because ES still carries a cloud-dependent offset, it is only compared ACROSS MODELS ON THE SAME
  CLOUD. evaluate.py scores three models per cloud and reports differences:
      regret = ES(fit) - ES(oracle)      oracle = the true family at the true theta (simulation only)
      gain   = ES(csr) - ES(oracle)      csr    = poisson at nbar = n, the no-structure baseline
      skill  = 1 - sum(regret) / sum(gain)  over a set of clouds: 1 = as good as the truth, 0 = no
                                            better than CSR. Only defined for cells where gain > 0.
  Change: `energy_score` is the only place the combination is formed.

CHOICE 2 -- W between patterns as normalized empirical measures, on equal-size random subsamples

  Patterns have different n. The exact W between the normalized measures (mass 1/n per point) is
  an LP, and POT (the usual solver) is not installed. Instead both patterns are uniformly thinned
  to the same k = min(n_x, n_s, max_points) points and matched exactly by linear assignment
  (scipy.optimize.linear_sum_assignment): W_p = (mean matched cost)^(1/p).
  Why thinning is acceptable: independent random thinning leaves a stationary process's pair
  correlation function unchanged, so the SHAPE of the clustering or repulsion survives; only the
  intensity drops. Consequences to know:
    - Intensity is not scored at all (normalized measures). An estimate with the right structure
      and the wrong nbar is not penalized here -- it is visible in stage 3's nbar error instead.
      Change: add a count term to `distance`, e.g. + lam * |n_x - n_s| / sqrt(n_x).
    - max_points trades speed for small-scale resolution: thinning to k points makes structure at
      scales below the new mean spacing ~ 1/sqrt(k) harder to see (the Matern II hard core is the
      first thing to go). config evaluation.max_points; None disables the cap.
  Change: `ground: exact` is the only ground implemented. A `sliced` ground (1-D Wasserstein on
  random projections, O(n log n), no thinning needed) or POT's exact unequal-mass solver would
  each be a new branch in `distance`.

CHOICE 3 -- p = 1

  W_1 is a metric, which the energy score needs (W_2^2 is not), and it is less dominated by a few
  far-transported points than W_2 -- in a unit window those are the edge points, whose placement
  says more about the boundary than about the process. config evaluation.p.

CHOICE 4 -- estimator of the two expectations

  With M simulated patterns: E W(S, x) is the mean over the M, E W(S, S') the mean over all
  M(M-1)/2 pairs (unbiased, and it reuses every simulation). config evaluation.sims. Cost per
  model is M + M(M-1)/2 assignments; M = 8 gives 36.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


def thin(points: np.ndarray, k: int, rng) -> np.ndarray:
    return points if len(points) == k else points[rng.choice(len(points), k, replace=False)]


def w_exact(a: np.ndarray, b: np.ndarray, p: float) -> float:
    """W_p between two equal-size point sets with uniform weights."""
    cost = cdist(a, b) ** p
    i, j = linear_sum_assignment(cost)
    return float(cost[i, j].mean() ** (1 / p))


def distance(x: np.ndarray, y: np.ndarray, rng, ground: str = "exact", p: float = 1,
             max_points: int | None = 400) -> float:
    if ground != "exact":
        raise ValueError(f"unknown ground `{ground}` (only `exact` is implemented)")
    k = min(len(x), len(y), max_points or np.inf)
    return w_exact(thin(x, k, rng), thin(y, k, rng), p)


def energy_score(x: np.ndarray, sims: list[np.ndarray], rng, **kw) -> dict:
    """ES = mean_i W(S_i, x) - 1/2 mean_{i<j} W(S_i, S_j); lower is better."""
    cross = np.mean([distance(x, s, rng, **kw) for s in sims])
    within = np.mean([distance(sims[i], sims[j], rng, **kw)
                      for i in range(len(sims)) for j in range(i + 1, len(sims))])
    return {"es": float(cross - within / 2), "cross": float(cross), "within": float(within)}
