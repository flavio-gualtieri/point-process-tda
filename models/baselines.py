# models/baselines.py
"""Classical baseline estimators for the homogeneous Thomas process.

This module adds a *second* classical estimator family alongside the
K-function minimum-contrast estimator in ``mincontrast.py``: a Palm
(composite) likelihood estimator in the Tanaka-Ogata-Stoyan (2008) sense,
specialised to the stationary Thomas process.

The public interface deliberately mirrors ``mincontrast.py`` so the estimator
is a *drop-in* for the same evaluation harness:

    crop_and_rescale(...)      # re-exported from mincontrast (identical preprocessing)
    fit_palm(points, ...) -> dict
    fit_multistart(points, n_starts=10, rng=None, **kwargs) -> dict
    estimate(cloud_path, ...) -> dict

Every fit returns the training-params schema
    parent_intensity, cluster_scale, mean_offspring
plus diagnostics
    lambda_hat, converged, contrast_value
where ``contrast_value`` holds the minimised *negative* log-likelihood, so the
harness's "lower is better" multistart bookkeeping still applies unchanged.

------------------------------------------------------------------------------
Derivation of the objective (so a reviewer can check it):

For a stationary Thomas process on the unit square the pair correlation is
    g(r) = 1 + exp(-r^2 / (4 sigma^2)) / (4 pi kappa sigma^2).
Following Tanaka et al. (2008), treat the ordered inter-point distances
{ r_ij = ||x_i - x_j|| : i != j, r_ij <= R } as an inhomogeneous Poisson
process on (0, R] with intensity
    nu(r) = n * lambda * 2 pi r * g(r).
The (theta-dependent part of the) log-likelihood is then
    l(kappa, sigma) = sum_{i!=j, r_ij<=R} log g(r_ij)  -  n * lambda * K(R),
because  integral_0^R nu(r) dr = n * lambda * K(R)  for the Thomas K-function
    K(R) = pi R^2 + (1/kappa) (1 - exp(-R^2 / (4 sigma^2))).
We estimate lambda = n on the unit square and maximise l over (kappa, sigma).
mean_offspring is recovered as mu = lambda / kappa, exactly as in mincontrast.

NOTE: validate this estimator on simulated Thomas clouds with known parameters
before trusting it on data (see estimate() / __main__); the constants above are
theta-independent and dropped, but you should confirm calibration empirically.
------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

# Re-use the *identical* preprocessing / K function from the Waagpetersen module
# so the two classical estimators see exactly the same points.
from mincontrast import crop_and_rescale, K_thomas, load_cloud, TRAIN_TARGET_N

__all__ = [
    "crop_and_rescale",
    "fit_palm",
    "fit_multistart",
    "estimate",
]


def g_thomas(r: np.ndarray, kappa: float, sigma: float) -> np.ndarray:
    """Pair correlation function of the stationary Thomas process."""
    return 1.0 + np.exp(-r**2 / (4.0 * sigma**2)) / (4.0 * np.pi * kappa * sigma**2)


def _palm_negloglik(
    log_params: np.ndarray,
    r: np.ndarray,
    n: int,
    lam: float,
    t_max: float,
) -> float:
    kappa = np.exp(log_params[0])
    sigma = np.exp(log_params[1])
    g = np.clip(g_thomas(r, kappa, sigma), 1e-12, None)
    K_R = float(K_thomas(np.array([t_max]), kappa, sigma)[0])
    loglik = float(np.sum(np.log(g)) - n * lam * K_R)
    return -loglik


def fit_palm(
    points: np.ndarray,
    t_max: float = 0.25,
    kappa0: float | None = None,
    sigma0: float | None = None,
) -> dict[str, Any]:
    points = np.asarray(points, dtype=float)
    n = len(points)
    if n < 2:
        raise ValueError("Need at least 2 points for the Palm likelihood.")

    lam = float(n)  # intensity on the unit square

    d = pdist(points)            # unordered pairwise distances, length n(n-1)/2
    d = d[d <= t_max]
    if d.size == 0:
        raise ValueError("No inter-point distances within t_max.")
    r = np.repeat(d, 2)          # ordered pairs: each unordered distance twice

    kappa0 = kappa0 if kappa0 is not None else lam / 5.0
    sigma0 = sigma0 if sigma0 is not None else t_max / 5.0
    x0 = [np.log(kappa0), np.log(sigma0)]

    # Bound the search in log-space. sigma cannot plausibly exceed the
    # observation radius t_max, and kappa cannot be astronomically larger than
    # the point count; without these bounds Nelder-Mead occasionally escapes to
    # sigma -> large (where kappa becomes unidentifiable) and kappa blows up.
    bounds = [
        (np.log(max(lam, 1.0) * 1e-4), np.log(max(lam, 1.0) * 1e2)),  # kappa
        (np.log(1e-3), np.log(max(t_max, 2e-3))),                     # sigma <= t_max
    ]
    x0 = [float(np.clip(x0[0], *bounds[0])), float(np.clip(x0[1], *bounds[1]))]

    res = minimize(
        _palm_negloglik,
        x0,
        args=(r, n, lam, t_max),
        method="Nelder-Mead",
        bounds=bounds,
        options={"xatol": 1e-8, "fatol": 1e-8, "maxiter": 5000},
    )

    kappa_hat = float(np.exp(res.x[0]))
    sigma_hat = float(np.exp(res.x[1]))
    mu_hat = lam / kappa_hat

    return {
        "parent_intensity": kappa_hat,
        "cluster_scale": sigma_hat,
        "mean_offspring": mu_hat,
        "lambda_hat": lam,
        "converged": bool(res.success),
        "contrast_value": float(res.fun),  # minimised negative log-likelihood
        "t_max": t_max,
    }


def fit_multistart(
    points: np.ndarray,
    n_starts: int = 10,
    rng: np.random.Generator | None = None,
    # signature-compatible with mincontrast.fit_multistart; r_high acts as t_max.
    r_low: float = 1e-3,
    r_high: float = 0.25,
    n_t: int = 200,
    c: float = 0.25,
    **_ignored: Any,
) -> dict[str, Any]:
    """Multistart Palm-likelihood fit. Same call signature as
    ``mincontrast.fit_multistart`` so the evaluation harness can pass the same
    ``mc_kwargs`` dict to either estimator. ``r_high`` is reused as ``t_max``.
    """
    rng = rng or np.random.default_rng(420937)
    best, best_val = None, np.inf
    for _ in range(n_starts):
        kappa0 = np.exp(rng.uniform(-2, 6))
        sigma0 = np.exp(rng.uniform(-4, 0))
        try:
            res = fit_palm(points, t_max=r_high, kappa0=kappa0, sigma0=sigma0)
        except Exception:
            continue
        if res["contrast_value"] < best_val:
            best, best_val = res, res["contrast_value"]
    if best is None:
        raise RuntimeError("All Palm-likelihood starts failed.")
    return best


def estimate(
    cloud_path: Path | str,
    seed: int = 0,
    target_n: int = TRAIN_TARGET_N,
    r_high: float = 0.25,
    n_starts: int = 10,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    cloud = load_cloud(Path(cloud_path))
    points = np.asarray(cloud["points"], dtype=float)
    reg = cloud.get("region", {})
    low = np.asarray(reg.get("low", [0.0, 0.0]), dtype=float)
    high = np.asarray(reg.get("high", [1.0, 1.0]), dtype=float)
    points_unit = crop_and_rescale(points, low, high, target_n=target_n, rng=rng)
    return fit_multistart(points_unit, n_starts=n_starts, rng=rng, r_high=r_high)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cloud", type=Path, required=True, help="path to a *_cloud.pkl file")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--r-high", type=float, default=0.25, help="Palm t_max on [0,1]^2")
    p.add_argument("--n-starts", type=int, default=10)
    return p


def main() -> None:
    args = _build_parser().parse_args()
    res = estimate(args.cloud, seed=args.seed, r_high=args.r_high, n_starts=args.n_starts)
    print(f"\nPalm-likelihood estimates for {args.cloud.name}")
    print(f"  parent_intensity : {res['parent_intensity']:.6g}")
    print(f"  cluster_scale    : {res['cluster_scale']:.6g}")
    print(f"  mean_offspring   : {res['mean_offspring']:.6g}")
    print(f"  lambda_hat       : {res['lambda_hat']:.4g}")
    print(f"  converged        : {res['converged']}")
    print(f"  neg-loglik       : {res['contrast_value']:.6g}")


if __name__ == "__main__":
    main()