"""Minimum contrast on Ripley's K, per family: the classical estimator the pipeline is compared with.

For family f with free parameters x (log scale, intensity fixed at n on the unit window):

    x_hat = argmin_x  mean_{r <= r_max} ( K_hat(r)^c - K_f(r; x)^c )^2

K_hat is classical.lfunction.k_function (isotropic edge correction, the estimator the classical
features use); K_f = pi r^2 + Family.excess, the closed forms the bank's generator is built on, so
the baseline fits the exact model that simulated the data. The search box is the family's
train-prior range on the FREE parameters, starts are the best `starts` of `init` train-prior draws,
then bounded Nelder-Mead. The dependent parameters (mu = n / kappa, ...) are NOT clipped: clipping
them independently breaks the fit's intensity = n and leaves a theta the sampler cannot realise.

Selection (the classical classifier): every cloud is fitted under every family's model; the call is
    argmin_f  log D_f + lambda * p_f          (poisson: D of pi r^2, p = 0)
over the run's families that have a K to fit (`models`). Cell has K = pi r^2 exactly and Strauss has
no closed form, so neither is ever called; when one is the true family (oracle routing) its estimate
is FALLBACK: intensity n and the train-prior median of the rest.
"""

from __future__ import annotations

import itertools
import math
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import minimize
from scipy.special import lambertw

from ..classical.lfunction import RADII
from ..pipeline.core import TARGETS
from ..simulation.families import FAMILIES, Rules, ring_step

FITTABLE = ("thomas", "nested", "lgcp", "matern2", "ring", "matern1")    # families with a K to fit
N_FREE = {"thomas": 2, "nested": 4, "lgcp": 2, "matern2": 1, "ring": 3, "matern1": 1}


def models(families) -> list[str]:
    """The families the baseline fits: the run's families with a K to fit, in FITTABLE order."""
    return [f for f in FITTABLE if f in families]


# Families K cannot fit, as (family's train rows, n per cloud) -> theta in TARGETS order. A length is
# taken at its median in mean-spacing units (R sqrt(nbar)), so it scales with the cloud's n.
_median = lambda x: float(np.exp(np.median(np.log(x))))
FALLBACK = {
    "cell": lambda train, n: np.column_stack([n, np.full(len(n), _median(train.k))]),
    "strauss": lambda train, n: np.column_stack([n, np.full(len(n), _median(train.q)),
                                                 _median(train.R * np.sqrt(train.nbar)) / np.sqrt(n)]),
}


# ---------------------------------------------------------------------------------------- models

# Free parameters on the log scale, and how they (with n) become the family's parameters. Intensity
# is n on the unit window, the classical estimate; everything K cannot see is fixed by it.
def _matern2(core, n):
    R, x = core / math.sqrt(n), math.pi * core**2
    return {"R": R, "lam_p": n * -math.log1p(-x) / x}


def _matern1(core, n):
    R = core / math.sqrt(n)
    a = math.pi * R**2
    return {"R": R, "lam_p": float(-lambertw(-a * n, 0).real) / a}


FREE = {
    "thomas": (lambda r: [np.log(r.kappa), np.log(r.sigma)],
               lambda e, n: {"kappa": e[0], "sigma": e[1], "mu": n / e[0]}),
    "nested": (lambda r: [np.log(r.kappa), np.log(r.mu1), np.log(r.sigma1), np.log(r.sigma1 / r.sigma2)],
               lambda e, n: {"kappa": e[0], "mu1": e[1], "sigma1": e[2], "sigma2": e[2] / e[3],
                             "mu2": n / (e[0] * e[1])}),
    "lgcp": (lambda r: [np.log(r.sigma2), np.log(r.s)],
             lambda e, n: {"sigma2": e[0], "s": e[1], "nbar": n}),
    "matern2": (lambda r: [np.log(r.R * np.sqrt(r.nbar))], lambda e, n: _matern2(e[0], n)),
    "ring": (lambda r: [np.log(r.kappa), np.log(r.rho), np.log(r.sigma / r.rho)],
             lambda e, n: {"kappa": e[0], "rho": e[1], "sigma": e[2] * e[1], "mu": n / e[0]}),
    "matern1": (lambda r: [np.log(r.R * np.sqrt(r.nbar))], lambda e, n: _matern1(e[0], n)),
}


def ring_table(path: Path) -> None:
    """ring_step is scale-free, G(r / rho, sigma / rho); tabulate it once (it costs ~50 ms a call)."""
    t = np.exp(np.linspace(np.log(0.03), np.log(1.5), 120))
    u = np.linspace(0, 2 + 8 * math.sqrt(2) * t[-1], 2400)
    G = np.stack([ring_step(u, 1.0, ti) for ti in t])
    np.savez(path, log_t=np.log(t), u=u, G=G)


class Excess:
    """K(r) - pi r^2 of each fittable family; ring by interpolation in its table (ring_table)."""

    def __init__(self, ring_table_path: Path):
        rules = Rules.load()
        self.fam = {f: FAMILIES[f](rules) for f in FITTABLE}
        z = np.load(ring_table_path)
        self.ring = RegularGridInterpolator((z["log_t"], z["u"]), z["G"], bounds_error=False, fill_value=None)
        self.u_max = z["u"][-1]

    def __call__(self, f: str, r: np.ndarray, p: dict) -> np.ndarray:
        if f != "ring":
            return self.fam[f].excess(r, p)
        u = r / p["rho"]
        g = self.ring(np.column_stack([np.full(len(r), math.log(p["sigma"] / p["rho"])), np.minimum(u, self.u_max)]))
        return np.where(u >= self.u_max, 1.0, np.clip(g, 0.0, 1.0)) / p["kappa"]


# ------------------------------------------------------------------------------------------ fit

def settings(mc: dict) -> list[tuple[float, float]]:
    """Every (contrast exponent c, r_max) of the config's grid."""
    return list(itertools.product(mc["c"], mc["r_max"]))


def radii_mask(mc: dict, r_max: float) -> np.ndarray:
    m = np.zeros(len(RADII), bool)
    m[mc["radii_step"] - 1::mc["radii_step"]] = True
    return m & (RADII <= r_max + 1e-12)


_W = {}


def init_worker(model, mc, box, prior_x, ring_table_path):
    _W.update(model=model, mc=mc, box=box, prior=prior_x, excess=Excess(ring_table_path), setts=settings(mc),
              masks=[radii_mask(mc, rm) for _, rm in settings(mc)])


def fit_one(job):
    """All settings for one cloud under the worker's model: (theta_hat per setting, D, at_bound)."""
    khat, n = job
    f, mc, (lo, hi), X0, ex = _W["model"], _W["mc"], _W["box"], _W["prior"], _W["excess"]
    to_params = FREE[f][1]
    T = TARGETS[f]
    theta, D, bound = np.full((len(_W["setts"]), len(T)), np.nan), np.full(len(_W["setts"]), np.nan), np.zeros(len(_W["setts"]), bool)
    for s, ((c, _), m) in enumerate(zip(_W["setts"], _W["masks"])):
        r, target = RADII[m], np.maximum(khat[m], 0.0) ** c

        def contrast(x):
            try:
                k = np.pi * r**2 + ex(f, r, to_params(np.exp(x), n))
            except (ValueError, ZeroDivisionError, OverflowError):
                return 1e6
            v = float(np.mean((target - np.maximum(k, 0.0) ** c) ** 2))
            return v if np.isfinite(v) else 1e6

        order = np.argsort([contrast(x0) for x0 in X0])[:mc["starts"]]
        best = None
        for x0 in X0[order]:
            res = minimize(contrast, x0, method="Nelder-Mead", bounds=list(zip(lo, hi)),
                           options={"xatol": 1e-4, "fatol": 1e-14, "maxiter": 300 * len(lo)})
            if best is None or res.fun < best.fun:
                best = res
        p = to_params(np.exp(best.x), n)
        theta[s] = [p[t] for t in T]
        D[s] = best.fun
        bound[s] = bool(np.any(np.minimum(best.x - lo, hi - best.x) < 1e-3))
    return theta, D, bound
