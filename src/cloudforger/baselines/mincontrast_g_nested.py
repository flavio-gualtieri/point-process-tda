# src/cloudforger/baselines/mincontrast_g_nested.py
"""Minimum-contrast estimator for the two-level Nested Thomas process, fit
against the empirical pair correlation function g(r) instead of
mincontrast_nested.py's Ripley K-function -- the nested-process counterpart
of mincontrast_g.py's g_thomas fit, and the g-based half of the same gap
mincontrast_nested.py fills (Table~\\ref{tab:classical-comparison}'s
footnote $^d$; the "Outstanding experiments" bullet "Run minimum contrast
on nested Thomas against \\eqref{eq:g-nested}, and re-run it on g with the
exponent c swept" -- this module covers the first half; c is exposed as a
parameter/CLI flag below for the second half).

Closed form (writeup ssec:validation, \\eqref{eq:g-nested}):
    g(h) = 1
         + exp(-h^2 / (4*sigma2^2))          / (4*pi*sigma2^2           * kappa*mu1)
         + exp(-h^2 / (4*(sigma1^2+sigma2^2))) / (4*pi*(sigma1^2+sigma2^2) * kappa)
same parameter names as mincontrast_nested.py's K_nested_thomas: kappa =
parent_intensity, mu1 = meta_offspring, sigma1 = meta_cluster_scale,
sigma2 = cluster_scale. As with K, mu2 (mean_offspring) does not appear in
g -- it is recovered after the fit from rho = kappa*mu1*mu2 (rho estimated
as lambda_hat), exactly as mincontrast_nested.fit does. See that module's
docstring for the rest of the K-vs-g parameter discussion; it applies here
unchanged.

Uses empirical_g from mincontrast_g.py unmodified (the edge-corrected
Gaussian-kernel pair-correlation estimator makes no assumption about which
process generated the points).

Registered under method name "mincontrast_g_nested" -- see
mincontrast_nested.py's docstring for the same dispatch-by-exact-name
reasoning (baselines/__init__.py, scripts/train.py's
CLASSICAL_BASELINE_NAMES, scripts/evaluate.py's RAW_TAG_METHODS).

Default c=1.0, same as mincontrast_g.py and the same reasoning (g(r) is
already O(1)-scaled, so K's variance-stabilizing c=0.25 doesn't obviously
carry over) -- and the same open question: the writeup's discussion of
mincontrast_g.py's Thomas results found every seed worse than chance under
c=1.0 and flagged c as the likely cause (untuned, unlike K's validated
c=1/4). Sweep --c here before trusting this baseline's numbers any more
than that one's.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize

from .mincontrast import crop_and_rescale, load_cloud, TRAIN_TARGET_N
from .mincontrast_g import empirical_g

__all__ = [
    "crop_and_rescale",
    "g_nested_thomas",
    "fit",
    "fit_multistart",
    "estimate",
]

# Same box constraints, same reasoning, as mincontrast_nested.py's
# _LOG_BOUNDS -- see that module's comment. Kept as a separate copy (not a
# shared import) so each module's bounds can be retuned independently if the
# K- and g-contrast surfaces turn out to need different margins.
LOG_KAPPA_BOUNDS = (-2.0, 8.0)     # kappa  in [0.135, 2981]
LOG_MU1_BOUNDS = (-2.0, 4.0)       # mu1    in [0.135, 54.6]
LOG_SIGMA1_BOUNDS = (-8.0, -0.5)   # sigma1 in [3.4e-4, 0.607]
LOG_SIGMA2_BOUNDS = (-9.0, -1.0)   # sigma2 in [1.2e-4, 0.368]
_LOG_BOUNDS = [LOG_KAPPA_BOUNDS, LOG_MU1_BOUNDS, LOG_SIGMA1_BOUNDS, LOG_SIGMA2_BOUNDS]


def g_nested_thomas(r: np.ndarray, kappa: float, mu1: float, sigma1: float, sigma2: float) -> np.ndarray:
    """Pair correlation function of the two-level Nested Thomas process (eq:g-nested)."""
    return (
        1.0
        + np.exp(-r**2 / (4.0 * sigma2**2)) / (4.0 * np.pi * sigma2**2 * kappa * mu1)
        + np.exp(-r**2 / (4.0 * (sigma1**2 + sigma2**2))) / (4.0 * np.pi * (sigma1**2 + sigma2**2) * kappa)
    )


def _contrast_objective(
        log_params: np.ndarray,
        t_values: np.ndarray,
        g_emp: np.ndarray,
        c: float,
        dt: float,
) -> float:
    kappa = np.exp(log_params[0])
    mu1 = np.exp(log_params[1])
    sigma1 = np.exp(log_params[2])
    sigma2 = np.exp(log_params[3])
    g_theo = g_nested_thomas(t_values, kappa, mu1, sigma1, sigma2)
    return float(np.sum((g_emp**c - g_theo**c) ** 2) * dt)


def fit(
        points: np.ndarray,
        r_low: float = 1e-3,
        r_high: float = 0.25,
        n_t: int = 200,
        c: float = 1.0,
        bandwidth: float | None = None,
        kappa0: float | None = None,
        mu1_0: float | None = None,
        sigma1_0: float | None = None,
        sigma2_0: float | None = None,
) -> dict[str, Any]:
    points = np.asarray(points, dtype=float)
    t_values = np.linspace(r_low, r_high, n_t)
    dt = t_values[1] - t_values[0]

    g_emp, lam_hat = empirical_g(points, t_values, bandwidth=bandwidth)

    kappa0 = kappa0 if kappa0 is not None else lam_hat / 15.0
    mu1_0 = mu1_0 if mu1_0 is not None else 3.0
    sigma1_0 = sigma1_0 if sigma1_0 is not None else r_high / 10.0
    sigma2_0 = sigma2_0 if sigma2_0 is not None else r_high / 25.0
    x0 = [np.log(kappa0), np.log(mu1_0), np.log(sigma1_0), np.log(sigma2_0)]

    res = minimize(
        _contrast_objective,
        x0,
        args=(t_values, g_emp, c, dt),
        method="Nelder-Mead",
        bounds=_LOG_BOUNDS,
        # same 4D-vs-2D reasoning as mincontrast_nested.fit's raised caps
        options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 20000, "maxfev": 20000},
    )

    kappa_hat = float(np.exp(res.x[0]))
    mu1_hat = float(np.exp(res.x[1]))
    sigma1_hat = float(np.exp(res.x[2]))
    sigma2_hat = float(np.exp(res.x[3]))
    mu2_hat = lam_hat / (kappa_hat * mu1_hat)   # mean_offspring = lambda / (kappa*mu1)

    return {
        # keys match the nested_thomas training params schema -- same convention as mincontrast_nested.fit
        "parent_intensity": kappa_hat,
        "meta_offspring": mu1_hat,
        "meta_cluster_scale": sigma1_hat,
        "mean_offspring": mu2_hat,
        "cluster_scale": sigma2_hat,
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
        # Sampled from the exact same box the optimizer is bounded to (see
        # _LOG_BOUNDS above), so every start begins inside the feasible region.
        kappa0 = np.exp(rng.uniform(*LOG_KAPPA_BOUNDS))
        mu1_0 = np.exp(rng.uniform(*LOG_MU1_BOUNDS))
        sigma1_0 = np.exp(rng.uniform(*LOG_SIGMA1_BOUNDS))
        sigma2_0 = np.exp(rng.uniform(*LOG_SIGMA2_BOUNDS))
        result = fit(points, kappa0=kappa0, mu1_0=mu1_0, sigma1_0=sigma1_0, sigma2_0=sigma2_0, **kwargs)
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
    print(f"\nMinimum-contrast-on-G (nested Thomas) estimates for {args.cloud.name}")
    print(f"  parent_intensity   : {result['parent_intensity']:.6g}")
    print(f"  meta_offspring     : {result['meta_offspring']:.6g}")
    print(f"  meta_cluster_scale : {result['meta_cluster_scale']:.6g}")
    print(f"  mean_offspring     : {result['mean_offspring']:.6g}")
    print(f"  cluster_scale      : {result['cluster_scale']:.6g}")
    print(f"  lambda_hat         : {result['lambda_hat']:.4g}  (points on unit square)")
    print(f"  converged          : {result['converged']}")
    print(f"  contrast value     : {result['contrast_value']:.6g}")


if __name__ == "__main__":
    main()
