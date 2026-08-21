# src/cloudforger/baselines/mincontrast_nested.py
"""Minimum-contrast estimator for the two-level Nested Thomas process, fit
against Ripley's K-function -- the nested-process counterpart of
mincontrast.py's K_thomas fit. Fills the gap flagged by
Table~\\ref{tab:classical-comparison}'s footnote $^d$ in the writeup
("Not run: the closed form of \\eqref{eq:g-nested} makes this comparison
available in principle") and the matching "Outstanding experiments" bullet
("Run minimum contrast on nested Thomas against \\eqref{eq:g-nested}").

Closed form (writeup ssec:validation, \\eqref{eq:g-nested}):
    K(t) = pi*t^2
         + (1 - exp(-t^2 / (4*sigma2^2))) / (kappa*mu1)
         + (1 - exp(-t^2 / (4*(sigma1^2+sigma2^2)))) / kappa
where kappa=parent_intensity (the root Poisson layer), mu1=meta_offspring
(mean number of level-1 [Thomas] cluster centres per root parent),
sigma1=meta_cluster_scale, sigma2=cluster_scale -- the same parameter names
NestedThomasProcess.params exposes (see data_generation/point_processes/
nested_thomas.py).

Note mean_offspring (mu2, the fine-level offspring count per level-1
cluster) does NOT appear in K -- same fact as mincontrast.py's single-level
K_thomas not depending on mu: K is a purely second-order/spatial-arrangement
statistic, so it fixes kappa, mu1, sigma1, sigma2 (four free parameters,
fit below) but not mu2. mu2 is instead recovered from the intensity
identity rho = kappa*mu1*mu2 (rho estimated as lambda_hat, the point count
on the unit square), exactly as mincontrast.py recovers
mean_offspring = lambda_hat / kappa for the single-level process.

Registered under method name "mincontrast_nested", not "mincontrast" --
scripts/train.py's run_classical_baseline dispatches by exact method name
via getattr(baselines, method_name), and mincontrast.py's K_thomas has no
branch for a different closed form. See baselines/__init__.py and
scripts/train.py's CLASSICAL_BASELINE_NAMES for the registration, and
scripts/evaluate.py's RAW_TAG_METHODS for why results are found there.

NOT yet validated end-to-end against ground truth the way mincontrast.py's
K_thomas fit is (writeup SS5.4, thomas-only): \\eqref{eq:g-nested}'s own
derivation note says "the derivation needs independent checking before
either claim is relied on" -- spot-check the first array task's recovered
parameters against known simulator inputs before trusting the numbers.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize

from .mincontrast import crop_and_rescale, empirical_K, load_cloud, TRAIN_TARGET_N

__all__ = [
    "crop_and_rescale",
    "K_nested_thomas",
    "fit",
    "fit_multistart",
    "estimate",
]

# Box constraints on the four log-parameters, in both fit()'s minimize() call
# and fit_multistart()'s init sampling (so every start begins inside the box
# the optimizer is confined to). Added after the first real run: unbounded
# Nelder-Mead on this 4-parameter objective regularly walked kappa/mu1/sigma1
# to the point of `exp` overflow (RuntimeWarning: overflow encountered in
# exp/divide) on clouds where the K-contrast surface is flat in the
# near-Poisson direction (kappa, mu1 -> infinity together leaves both
# clustering terms -> 0, same "near-Poisson flatness" the writeup already
# reports for the single-level K fit, just an extra degenerate direction
# here) -- test losses of 65-395 (vs. the single-level K fit's already-bad
# 0.855 +- 0.682) traced back to this, not to a merely-hard estimation
# problem. Bounds are wide margins around
# configs/runs/nested_thomas/nested_thomas_mph.yaml's design grid (kappa in
# [15,120], mu1 in [1.5,7], meta_cluster_scale in ~[0.005,0.12],
# cluster_scale in ~[0.0005,0.04]), chosen so exp() at either edge is nowhere
# near float64 overflow (exp(8) ~ 2981, exp(-9) ~ 1.2e-4).
LOG_KAPPA_BOUNDS = (-2.0, 8.0)     # kappa  in [0.135, 2981]
LOG_MU1_BOUNDS = (-2.0, 4.0)       # mu1    in [0.135, 54.6]
LOG_SIGMA1_BOUNDS = (-8.0, -0.5)   # sigma1 in [3.4e-4, 0.607]
LOG_SIGMA2_BOUNDS = (-9.0, -1.0)   # sigma2 in [1.2e-4, 0.368]
_LOG_BOUNDS = [LOG_KAPPA_BOUNDS, LOG_MU1_BOUNDS, LOG_SIGMA1_BOUNDS, LOG_SIGMA2_BOUNDS]


def K_nested_thomas(t: np.ndarray, kappa: float, mu1: float, sigma1: float, sigma2: float) -> np.ndarray:
    """Ripley's K for the two-level Nested Thomas process (eq:g-nested)."""
    return (
        np.pi * t**2
        + (1.0 - np.exp(-t**2 / (4.0 * sigma2**2))) / (kappa * mu1)
        + (1.0 - np.exp(-t**2 / (4.0 * (sigma1**2 + sigma2**2)))) / kappa
    )


def _contrast_objective(
        log_params: np.ndarray,
        t_values: np.ndarray,
        K_emp: np.ndarray,
        c: float,
        dt: float,
) -> float:
    kappa = np.exp(log_params[0])
    mu1 = np.exp(log_params[1])
    sigma1 = np.exp(log_params[2])
    sigma2 = np.exp(log_params[3])
    K_theo = K_nested_thomas(t_values, kappa, mu1, sigma1, sigma2)
    return float(np.sum((K_emp**c - K_theo**c) ** 2) * dt)


def fit(
        points: np.ndarray,
        r_low: float = 1e-3,
        r_high: float = 0.25,
        n_t: int = 200,
        c: float = 0.25,
        kappa0: float | None = None,
        mu1_0: float | None = None,
        sigma1_0: float | None = None,
        sigma2_0: float | None = None,
) -> dict[str, Any]:
    points = np.asarray(points, dtype=float)
    t_values = np.linspace(r_low, r_high, n_t)
    dt = t_values[1] - t_values[0]

    K_emp, lam_hat = empirical_K(points, t_values)

    kappa0 = kappa0 if kappa0 is not None else lam_hat / 15.0
    mu1_0 = mu1_0 if mu1_0 is not None else 3.0
    sigma1_0 = sigma1_0 if sigma1_0 is not None else r_high / 10.0
    sigma2_0 = sigma2_0 if sigma2_0 is not None else r_high / 25.0
    x0 = [np.log(kappa0), np.log(mu1_0), np.log(sigma1_0), np.log(sigma2_0)]

    res = minimize(
        _contrast_objective,
        x0,
        args=(t_values, K_emp, c, dt),
        method="Nelder-Mead",
        bounds=_LOG_BOUNDS,
        # 4 free params vs mincontrast.py's 2 -- Nelder-Mead's simplex needs
        # more iterations/evals to converge in the higher dimension, so both
        # caps are raised well above scipy's dimension-scaled default
        # (~200*ndim fevals) rather than reusing mincontrast.py's 5000.
        options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 20000, "maxfev": 20000},
    )

    kappa_hat = float(np.exp(res.x[0]))
    mu1_hat = float(np.exp(res.x[1]))
    sigma1_hat = float(np.exp(res.x[2]))
    sigma2_hat = float(np.exp(res.x[3]))
    mu2_hat = lam_hat / (kappa_hat * mu1_hat)   # mean_offspring = lambda / (kappa*mu1)

    return {
        # ── keys that match the nested_thomas training params schema ────────
        "parent_intensity": kappa_hat,
        "meta_offspring": mu1_hat,
        "meta_cluster_scale": sigma1_hat,
        "mean_offspring": mu2_hat,
        "cluster_scale": sigma2_hat,
        # ── diagnostics ─────────────────────────────────────────────────────
        "lambda_hat": lam_hat,
        "converged": bool(res.success),
        "contrast_value": float(res.fun),
        "t_values": t_values,
        "K_emp": K_emp,
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
        c: float = 0.25,
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
    p.add_argument("--r-high", type=float, default=0.25, help="upper K integration limit in [0,1]^2 units (default: 0.25)")
    p.add_argument("--n-t", type=int, default=200, help="quadrature points for the Riemann sum (default: 200)")
    p.add_argument("--c", type=float, default=0.25, help="contrast exponent (default: 0.25, matching mincontrast.py's K fit)")
    p.add_argument("--n-starts", type=int, default=10, help="multistart restart count (default: 10)")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    result = estimate(
        cloud_path=args.cloud, seed=args.seed, target_n=args.target_n,
        r_high=args.r_high, n_t=args.n_t, c=args.c, n_starts=args.n_starts,
    )
    print(f"\nMinimum-contrast (K, nested Thomas) estimates for {args.cloud.name}")
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
