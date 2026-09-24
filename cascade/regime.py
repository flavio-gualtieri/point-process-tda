"""Regime coordinates, and the frozen cutoffs that decide which bank clouds each component trains on.

A cutoff is fitted ONCE, by cascade/regime_analysis.py, from stage 1's out-of-fold routing on the
train split, and written to cutoffs.json. From then on it is a fixed rule on manifest columns:
training a component never looks at any other component's predictions.

The rule, per family, with coordinate x (config regime.coordinates) and expected count nbar:

    u = log x + a * log nbar              a fitted by logistic regression of
                                          1{stage 1 routes to own group} on (log x, log nbar)
    P(own | u)                            isotonic in u (direction fitted)
    in regime at tau  <=>  P(own | u) >= tau, i.e. u past the boundary u*(tau)

A logistic model's 50% contour is a straight line in (log x, log nbar), so this is the 2-D
(x, nbar) cutoff expressed as ONE combined coordinate x * nbar^a -- e.g. for Thomas "cluster
overlap omega, corrected for how many points there are to see it with". When nbar does not matter
the fit gives a ~ 0 and the rule is the 1-D cutoff on x. tau = 0 keeps everything.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# Physically meaningful coordinates, per family, from the manifest's parameter columns. The `zeta`s
# are back-of-envelope pair-count signal-to-noise ratios (excess pairs at the process's own scale
# over the square root of CSR's pairs there); cascade/regime_analysis.py tests them against the rest.
DERIVED = {
    "thomas": {
        "omega": lambda r: r.sigma * np.sqrt(r.kappa),                  # cluster overlap (smearing)
        "zeta": lambda r: np.sqrt(r.mu) / (r.sigma * np.sqrt(r.kappa)),  # sqrt(richness) / overlap
    },
    "nested": {
        "omega_outer": lambda r: np.hypot(r.sigma1, r.sigma2) * np.sqrt(r.kappa),
        "omega_inner": lambda r: r.sigma2 * np.sqrt(r.kappa * r.mu1),
        "meta_size": lambda r: r.mu1 * r.mu2,                           # points per meta-cluster
        "zeta_outer": lambda r: np.sqrt(r.mu1 * r.mu2) / (np.hypot(r.sigma1, r.sigma2) * np.sqrt(r.kappa)),
        "zeta_inner": lambda r: np.sqrt(r.mu2) / (r.sigma2 * np.sqrt(r.kappa * r.mu1)),
        "zeta": lambda r: np.hypot(np.sqrt(r.mu1 * r.mu2) / (np.hypot(r.sigma1, r.sigma2) * np.sqrt(r.kappa)),
                                   np.sqrt(r.mu2) / (r.sigma2 * np.sqrt(r.kappa * r.mu1))),
    },
    "matern2": {
        "core": lambda r: r.R * np.sqrt(r.nbar),                        # hard core, in mean spacings
        "zeta": lambda r: r.R * r.nbar,                                  # ~ sqrt(n * packing fraction)
    },
    "lgcp": {
        "s_rel": lambda r: r.s * np.sqrt(r.nbar),                       # correlation range, mean spacings
        "zeta": lambda r: np.expm1(r.sigma2) * r.s * r.nbar,
    },
}


def values(rows: pd.DataFrame, family: str, name: str) -> np.ndarray:
    """One coordinate for rows of one family: a DERIVED name or a manifest column."""
    fn = DERIVED.get(family, {}).get(name)
    return np.asarray(fn(rows) if fn else rows[name], float)


# ------------------------------------------------------------------------------ fitting (analysis)

def fit_cutoff(x: np.ndarray, nbar: np.ndarray, y: np.ndarray, taus) -> dict:
    """The combined coordinate u = log x + a log nbar and its boundary per tau. See module doc."""
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    Z = np.column_stack([np.log(x), np.log(nbar)])
    b = LogisticRegression(C=1e6, max_iter=1000).fit(Z, y).coef_[0]
    a = float(b[1] / b[0])
    u = Z[:, 0] + a * Z[:, 1]
    iso = IsotonicRegression(increasing="auto", out_of_bounds="clip").fit(u, y)
    grid = np.sort(np.unique(u))
    p = iso.predict(grid)
    out = {"nbar_exponent": a, "direction": "increasing" if iso.increasing_ else "decreasing",
           "u_boundary": {}, "pool_fraction": {}}
    for tau in taus:
        hit = np.flatnonzero(p >= tau)
        ub = None if not len(hit) else float(grid[hit[0]] if iso.increasing_ else grid[hit[-1]])
        out["u_boundary"][str(tau)] = ub
        out["pool_fraction"][str(tau)] = float((iso.predict(u) >= tau).mean())
    return out


# ------------------------------------------------------------------------------ applying (training)

class Cutoffs:
    """cutoffs.json -> which rows of a family are in regime at a given tau."""

    def __init__(self, path: Path):
        self.path = Path(path)
        if not self.path.exists():
            raise SystemExit(f"{self.path} missing -- run cascade/regime_analysis.py first")
        self.table = json.loads(self.path.read_text())

    def u(self, rows: pd.DataFrame, family: str) -> np.ndarray:
        c = self.table[family]
        return np.log(values(rows, family, c["coordinate"])) + c["nbar_exponent"] * np.log(rows.nbar.to_numpy(float))

    def mask(self, rows: pd.DataFrame, family: str, tau: float) -> np.ndarray:
        """Boolean over rows (all of `family`): in regime at tau. tau = 0 keeps everything."""
        if tau == 0 or family == "poisson":
            return np.ones(len(rows), bool)
        c = self.table[family]
        ub = c["u_boundary"][str(tau)]
        if ub is None:
            return np.zeros(len(rows), bool)
        u = self.u(rows, family)
        return u >= ub if c["direction"] == "increasing" else u <= ub

    def describe(self, family: str, tau: float) -> str:
        c = self.table[family]
        ub = c["u_boundary"].get(str(tau))
        if tau == 0 or ub is None:
            return "all" if tau == 0 else "none"
        op = ">=" if c["direction"] == "increasing" else "<="
        return f"{c['coordinate']} * nbar^{c['nbar_exponent']:.2f} {op} {np.exp(ub):.4g}"
