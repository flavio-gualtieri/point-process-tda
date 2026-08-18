# src/cloudforger/baselines/mincontrast_g.py
"""Minimum-contrast estimator for the stationary Thomas process, fit
against the empirical pair correlation function g(r) instead of
mincontrast.py's Ripley K-function -- the "minimum contrast on G" baseline
(writeup's ssec:exp-performance \\paragraph{Comparison against classical
estimators}: "minimum contrast on K and on G"). New module: unlike
palm.py (which reuses mincontrast.py's K_thomas/crop_and_rescale but adds
its own Palm-LIKELIHOOD objective), this file adds the missing
CONTRAST-based fit against g -- there was no empirical pair-correlation
estimator anywhere in this codebase before this file.

NOTE, same as palm.py's own docstring: validate this estimator on simulated
Thomas clouds with known parameters before trusting it on data -- the
derivation below is internally consistent (see empirical_g's docstring),
but has not been checked against ground truth the way mincontrast.py's
K-based fit implicitly has (via the simulator validation in the writeup's
SS5.4, which checks empirical K against K_thomas, not g against g_thomas
directly through THIS estimator).

------------------------------------------------------------------------------
Derivation (so a reviewer can check it):

mincontrast.py's empirical_K estimates
    K_hat(t) = (1 / lambda_hat^2) * sum_{i != j, d_ij <= t} 1 / w_ij
where w_ij = wx_ij * wy_ij is the translation edge-correction weight (area
of the unit square overlapped with itself shifted by x_i - x_j) and
lambda_hat = n (points per unit area on [0,1]^2). The pair correlation
function is the density version of K: K(t) = 2*pi * integral_0^t r g(r) dr,
so g(t) = (1 / (2 pi t)) * dK/dt. Replacing the indicator 1[d_ij <= t] in
K_hat's sum with a smoothing kernel kernel_h(t - d_ij) (a kernel bump
integrates to 1 over t, same role the indicator's jump plays for K) gives
the edge-corrected kernel pair-correlation estimator implemented below:
    g_hat(t) = (1 / (2 pi t lambda_hat^2)) * sum_{i != j} kernel_h(t - d_ij) / w_ij
using the SAME lambda_hat and w_ij as empirical_K, so the two estimators
are consistent with each other by construction, not independently derived.
Kernel: Gaussian, kernel_h(x) = exp(-(x/h)^2 / 2) / (h * sqrt(2 pi)).
Bandwidth default h = 0.15 / sqrt(lambda_hat), a standard density-scaled
rule of thumb (Ripley's-rule-of-thumb style, e.g. spatstat's bw.pcf
defaults similarly to the mean nearest-neighbour spacing).

Contrast objective: sum_t (g_hat(t)^c - g_theo(t)^c)^2 dt, the same
power-transformed-contrast SHAPE mincontrast.py uses for K (Waagpetersen
2007), for structural/API consistency -- but default c=1.0 (no power
transform) rather than K's c=0.25: K(t) grows like pi*t^2 and needs a
variance-stabilizing transform, g(t) is already O(1)-scaled (g -> 1 as
r -> infinity for a stationary process) so the same justification for
c < 1 doesn't obviously apply. c is still exposed for anyone who wants to
sweep it.
------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize

from .mincontrast import crop_and_rescale, load_cloud, TRAIN_TARGET_N
from .palm import g_thomas

__all__ = [
    "crop_and_rescale",
    "empirical_g",
    "fit",
    "fit_multistart",
    "estimate",
]


def empirical_g(points: np.ndarray, t_values: np.ndarray, bandwidth: float | None = None) -> tuple[np.ndarray, float]:
    """Edge-corrected Gaussian-kernel pair-correlation estimate -- see
    module docstring for the derivation (consistent with mincontrast.py's
    empirical_K by construction: same lambda_hat, same translation
    edge-correction weight w_ij)."""
    n = len(points)
    if n < 2:
        raise ValueError("Need at least 2 points to estimate g.")

    lam_hat = float(n)  # points per unit area on [0,1]^2
    bandwidth = bandwidth if bandwidth is not None else 0.15 / np.sqrt(lam_hat)

    dx = points[:, 0:1] - points[:, 0]  # (n, n)
    dy = points[:, 1:2] - points[:, 1]
    dist = np.sqrt(dx**2 + dy**2)

    wx = np.clip(1.0 - np.abs(dx), 0.0, None)
    wy = np.clip(1.0 - np.abs(dy), 0.0, None)
    weight = wx * wy
    np.fill_diagonal(weight, 1.0)

    iu = np.triu_indices(n, k=1)  # each unordered pair once; kernel is symmetric in +-(t - d)
    d_pairs = dist[iu]
    w_pairs = weight[iu]
    inv_w = 1.0 / w_pairs

    g_vals = np.empty(len(t_values))
    for k, t in enumerate(t_values):
        u = (t - d_pairs) / bandwidth
        kernel = np.exp(-0.5 * u**2) / (bandwidth * np.sqrt(2.0 * np.pi))
        # x2: triu_indices dropped the (j, i) mirror of each (i, j) pair,
        # but empirical_K's sum runs over ordered pairs (both (i,j) and
        # (j,i), each contributing 1/w_ij) -- same convention here.
        g_vals[k] = 2.0 * np.sum(kernel * inv_w) / (2.0 * np.pi * t * lam_hat**2)

    return g_vals, lam_hat


def _contrast_objective(
        log_params: np.ndarray,
        t_values: np.ndarray,
        g_emp: np.ndarray,
        c: float,
        dt: float,
) -> float:
    kappa = np.exp(log_params[0])
    sigma = np.exp(log_params[1])
    g_theo = g_thomas(t_values, kappa, sigma)
    return float(np.sum((g_emp**c - g_theo**c) ** 2) * dt)


def fit(
        points: np.ndarray,
        r_low: float = 1e-3,
        r_high: float = 0.25,
        n_t: int = 200,
        c: float = 1.0,
        bandwidth: float | None = None,
        kappa0: float | None = None,
        sigma0: float | None = None,
) -> dict[str, Any]:
    points = np.asarray(points, dtype=float)
    t_values = np.linspace(r_low, r_high, n_t)
    dt = t_values[1] - t_values[0]

    g_emp, lam_hat = empirical_g(points, t_values, bandwidth=bandwidth)

    kappa0 = kappa0 if kappa0 is not None else lam_hat / 5.0
    sigma0 = sigma0 if sigma0 is not None else r_high / 5.0
    x0 = [np.log(kappa0), np.log(sigma0)]

    res = minimize(
        _contrast_objective,
        x0,
        args=(t_values, g_emp, c, dt),
        method="Nelder-Mead",
        options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 5000},
    )

    kappa_hat = float(np.exp(res.x[0]))
    sigma_hat = float(np.exp(res.x[1]))
    mu_hat = lam_hat / kappa_hat  # mean_offspring = lambda / kappa

    return {
        # keys match the training params schema -- same convention as mincontrast.fit
        "parent_intensity": kappa_hat,
        "cluster_scale": sigma_hat,
        "mean_offspring": mu_hat,
        # diagnostics
        "lambda_hat": lam_hat,
        "converged": bool(res.success),
        "contrast_value": float(res.fun),
        "t_values": t_values,
        "g_emp": g_emp,
    }


def fit_multistart(points, n_starts: int = 10, rng=None, **kwargs):
    rng = rng or np.random.default_rng(420937)
    best_result, best_contrast = None, np.inf
    for _ in range(n_starts):
        kappa0 = np.exp(rng.uniform(-2, 6))
        sigma0 = np.exp(rng.uniform(-4, 0))
        result = fit(points, kappa0=kappa0, sigma0=sigma0, **kwargs)
        if result["contrast_value"] < best_contrast:
            best_result, best_contrast = result, result["contrast_value"]

    return best_result


def estimate(
        cloud_path: Path | str,
        seed: int = 0,
        target_n: int = TRAIN_TARGET_N,
        r_low: float = 1e-3,
        r_high: float = 0.25,
        n_t: int = 200,
        c: float = 1.0,
        n_starts: int = 10,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    cloud = load_cloud(Path(cloud_path))

    points = np.asarray(cloud["points"], dtype=float)
    reg = cloud.get("region", {})
    low = np.asarray(reg.get("low", [0.0, 0.0]), dtype=float)
    high = np.asarray(reg.get("high", [1.0, 1.0]), dtype=float)

    points_unit = crop_and_rescale(points, low, high, target_n=target_n, rng=rng)

    return fit_multistart(points_unit, n_starts=n_starts, rng=rng, r_low=r_low, r_high=r_high, n_t=n_t, c=c)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cloud", type=Path, required=True, help="path to a *_cloud.pkl file")
    p.add_argument("--seed", type=int, default=0, help="RNG seed for the sub-window crop (default: 0)")
    p.add_argument("--target-n", type=int, default=TRAIN_TARGET_N, help="target point count after crop (default: %(default)s)")
    p.add_argument("--r-high", type=float, default=0.25, help="upper distance limit in [0,1]^2 units (default: 0.25)")
    p.add_argument("--n-t", type=int, default=200, help="quadrature points for the Riemann sum (default: 200)")
    p.add_argument("--c", type=float, default=1.0, help="contrast exponent (default: 1.0, no power transform)")
    p.add_argument("--n-starts", type=int, default=10, help="multistart restart count (default: 10)")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    result = estimate(
        cloud_path=args.cloud, seed=args.seed, target_n=args.target_n,
        r_high=args.r_high, n_t=args.n_t, c=args.c, n_starts=args.n_starts,
    )
    print(f"\nMinimum-contrast-on-G estimates for {args.cloud.name}")
    print(f"  parent_intensity : {result['parent_intensity']:.6g}")
    print(f"  cluster_scale    : {result['cluster_scale']:.6g}")
    print(f"  mean_offspring   : {result['mean_offspring']:.6g}")
    print(f"  lambda_hat       : {result['lambda_hat']:.4g}  (points on unit square)")
    print(f"  converged        : {result['converged']}")
    print(f"  contrast value   : {result['contrast_value']:.6g}")


if __name__ == "__main__":
    main()
