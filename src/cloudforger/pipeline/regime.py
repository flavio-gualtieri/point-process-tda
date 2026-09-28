"""The regime: where a family is distinguishable from CSR, as a frozen rule on theta.

Per family, with a physical coordinate x (config regime.coordinates) and expected count nbar:

    u = log x + a * log nbar              a fitted by logistic regression of the event on (log x, log nbar)
    P(event | u)                          isotonic in u (direction fitted)
    in regime at tau  <=>  P(event | u) >= tau, i.e. u past the boundary u*(tau)

The event is the reference classifier's call (config regime.classifier) on its out-of-fold train
predictions, else on val: `detected` = not called poisson, `identified` = called its own family. A
logistic model's 50% contour is a straight line in (log x, log nbar), so this is the 2-D (x, nbar)
cutoff expressed as ONE combined coordinate x * nbar^a -- e.g. for Thomas "cluster overlap omega,
corrected for how many points there are to see it with". The rule is on theta alone, so every
model is scored on the same in-regime clouds. A family with no coordinate is in regime everywhere.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .core import load_classifier, log

# Physical coordinates from the manifest's parameter columns. omega = cluster (or ring) overlap,
# zeta = a back-of-envelope pair-count signal-to-noise ratio.
COORDINATES = {
    "thomas": {"omega": lambda r: r.sigma * np.sqrt(r.kappa)},
    "nested": {"omega_inner": lambda r: r.sigma2 * np.sqrt(r.kappa * r.mu1)},
    "lgcp": {"zeta": lambda r: np.expm1(r.sigma2) * r.s * r.nbar},
    "matern2": {"zeta": lambda r: r.R * r.nbar},
    "ring": {"omega": lambda r: r.rho * np.sqrt(r.kappa)},
    "matern1": {"zeta": lambda r: r.R * r.nbar},
}


def coordinate(r: pd.DataFrame, family: str, name: str) -> np.ndarray:
    """A named coordinate of one family's rows: COORDINATES, or any manifest column."""
    fn = COORDINATES.get(family, {}).get(name)
    return np.asarray(fn(r) if fn else r[name], float)


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


def fit_cutoffs(cfg: dict, r: pd.DataFrame) -> dict:
    """family -> fitted cutoff, from the reference classifier's calls (see module doc)."""
    reg = cfg["regime"]
    pred = load_classifier(cfg, reg["classifier"])
    if pred is None:
        log(f"regime: reference classifier {reg['classifier']} not trained -- every cloud in regime")
        return {}
    fit_split = "train" if (pred.split == "train").any() else "val"
    pred = pred[pred.split == fit_split]
    classes = [c for c in pred.columns if c != "split"]
    call = np.array(classes)[pred[classes].to_numpy().argmax(1)]
    rows = r.loc[pred.index]
    out = {}
    for f, coord in reg["coordinates"].items():
        if coord is None or f not in cfg["families"]:
            continue
        m = (rows.family == f).to_numpy()
        y = call[m] != "poisson" if reg["event"] == "detected" else call[m] == f
        if y.all() or not y.any():                               # one outcome: no boundary to fit
            c = {"constant": bool(y.all()), "nbar_exponent": 0.0, "direction": None, "u_boundary": {}}
        else:
            c = fit_cutoff(coordinate(rows[m], f, coord), rows.nbar.to_numpy(float)[m], y, reg["taus"])
        out[f] = {**c, "coordinate": coord, "event_rate": float(y.mean()), "fitted_on": fit_split,
                  "n": int(m.sum())}
    return out


def in_regime(cuts: dict, rows: pd.DataFrame, tau: float) -> np.ndarray:
    """Boolean over `rows`: in regime at tau under the fitted cutoffs."""
    keep = np.ones(len(rows), bool)
    for f, c in cuts.items():
        m = (rows.family == f).to_numpy()
        if "constant" in c:                                      # always (or never) detected
            keep[m] = c["constant"]
            continue
        ub = c["u_boundary"][str(tau)]
        if ub is None:
            keep[m] = False
            continue
        u = np.log(coordinate(rows[m], f, c["coordinate"])) + c["nbar_exponent"] * np.log(rows.nbar.to_numpy(float)[m])
        keep[m] = u >= ub if c["direction"] == "increasing" else u <= ub
    return keep
