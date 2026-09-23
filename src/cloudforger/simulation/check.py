"""Samplers against closed forms: mean of the unbiased K-hat vs K on R(nbar).

max_abs_z is only meaningful where K-hat is not badly skewed. For an LGCP the intensity is
log-normal, so moments of K-hat scale like exp(k sigma^2): by sigma^2 = 4 a single realisation
reaches 87x the mean and the s.d. estimated from two halves of one sample differs 3-fold, so the
studentised z is unusable. That is a limit of this diagnostic, not of the sampler -- the grid
process's K matches the closed form to ~1e-5, which lgcp_grid.grid_bias computes exactly.
"""

from __future__ import annotations

import numpy as np

from ..classical.lfunction import RADII, k_function
from ..departure.tables import Tables
from .families import FAMILIES, Rules
from .processes import SAMPLERS, lgcp_eigenvalues
from .seeding import PATTERN, case_rng
from .bank import Config, draw_theta

OFFSET = 10**9   # pilot indices above the null-curve range


def check_theta(fam_name: str, index: int, patterns: int, cfg: Config, tables: Tables, radii: int = 16) -> dict:
    fam = FAMILIES[fam_name](Rules.load())
    theta = draw_theta(fam, tables, cfg, index)
    model = dict(theta["model"])
    if fam_name == "lgcp":
        model["root_lam"] = lgcp_eigenvalues(model["sigma2"], model["s"], model["M"])
    nbar = theta["nbar"]
    k = np.empty((patterns, len(RADII)))
    for p in range(patterns):
        pts = SAMPLERS[fam_name](case_rng(cfg.root, "pilot", fam_name, OFFSET + index * 10_000 + p, PATTERN), **model)
        n = len(pts)
        k[p] = k_function(pts) * n * (n - 1) / nbar**2 if n >= 2 else 0.0
    K = np.pi * RADII**2 + fam.excess(RADII, theta["model"])
    ok = tables.window(nbar)[0] & (K > 0)
    sel = np.flatnonzero(ok)[:: max(1, ok.sum() // radii)]
    z = (k[:, sel].mean(0) - K[sel]) / (k[:, sel].std(0, ddof=1) / np.sqrt(patterns))
    return {"family": fam_name, "theta": index, "nbar": round(nbar, 1), "cv": round(theta["cv"], 3),
            "max_abs_z": float(np.abs(z).max()), "max_rel_err": float(np.abs(k[:, sel].mean(0) / K[sel] - 1).max())}
