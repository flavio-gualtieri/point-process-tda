"""
mincontrast.py
==============

Minimum-contrast estimation of Thomas process parameters from a point-cloud
pickle file in the project's standard format (the same format produced by
process_barro.py and consumed by processing.py / model.py).

Theory
------
The Thomas process has pair correlation function

    g(t) = 1 + exp(-t^2 / (4 sigma^2)) / (kappa * (4 pi sigma^2))

and K-function

    K(t) = pi t^2 + (1 - exp(-t^2 / (4 sigma^2))) / kappa

where kappa = parent_intensity, sigma = cluster_scale.
The overall intensity is lambda = kappa * mu (mu = mean_offspring), so mu is
recovered as lambda_hat / kappa_hat after fitting (kappa, sigma) from K alone.

Minimum contrast (Diggle et al. 1985; Waagepetersen & Guan 2009 eq. 2):

    argmin_{kappa, sigma}  integral_{r_low}^{r_high}
        [ K_hat(t)^c - K(t; kappa, sigma)^c ]^2 dt

with contrast exponent c = 0.25 (standard choice that stabilises variance).

Coordinate space
----------------
processing.py crops a square sub-window of the raw cloud and rescales it to
[0,1]^2 before computing persistence features. To be comparable, mincontrast
must operate in that same unit-square space. The helper `crop_and_rescale`
replicates exactly what sample_unit_box_window in processing.py does, so the
two pipelines see identical point configurations.

Usage
-----
    python mincontrast.py                            # default barro census 1
    python mincontrast.py --cloud data/barro/bci.tree1_cloud.pkl
    python mincontrast.py --cloud data/barro/bci.tree1_cloud.pkl --seed 42
    python mincontrast.py --cloud data/barro/bci.tree1_cloud.pkl --r-high 0.3 --n-t 300
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize

# ── shared constant with processing.py ────────────────────────────────────────
TRAIN_TARGET_N = 900   # target point count after sub-window crop


# ══════════════════════════════════════════════════════════════════════════════
# I/O  –  reading the project's cloud pickle format
# ══════════════════════════════════════════════════════════════════════════════

def load_cloud(path: Path) -> dict[str, Any]:
    """Load a cloud pickle (1-element list or bare dict) → dict."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, list):
        data = data[0]
    return data


# ══════════════════════════════════════════════════════════════════════════════
# Pre-processing  –  mirrors processing.py / sample_unit_box_window exactly
# ══════════════════════════════════════════════════════════════════════════════

def crop_and_rescale(
        points: np.ndarray,
        region_low: np.ndarray,
        region_high: np.ndarray,
        target_n: int = TRAIN_TARGET_N,
        rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Crop a square sub-window sized for ~target_n points, rescale to [0,1]^2.

    Replicates sample_unit_box_window from processing.py so that mincontrast
    sees the same point configuration as the TDA / neural-network pipeline.
    All subsequent calculations (K-function, parameter estimates) are therefore
    in the same unit-square coordinate system as the training data.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    extent = region_high - region_low
    n_points = len(points)
    volume = float(np.prod(extent))

    density = n_points / volume
    window = float(np.sqrt(target_n / density))
    window = min(window, float(extent.min()))

    slack = extent - window
    corner = region_low + rng.uniform(0.0, 1.0, size=extent.shape) * slack

    mask = np.all(
        (points >= corner) & (points <= corner + window), axis=1
    )
    rescaled = (points[mask] - corner) / window   # now in [0,1]^2
    return rescaled


# ══════════════════════════════════════════════════════════════════════════════
# Empirical K-function  –  translation edge correction (Ohser 1983)
# ══════════════════════════════════════════════════════════════════════════════

def empirical_K(
        points: np.ndarray,
        t_values: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Translation-edge-corrected K-function on the unit square [0,1]^2.

    The edge-correction weight for a pair separated by vector h = (hx, hy) is
    |W ∩ W_h| = (1 - |hx|)(1 - |hy|), the area of the intersection of the
    unit square with its translate by h.

    Returns
    -------
    K_vals   : array, shape (len(t_values),)
    lam_hat  : float, estimated intensity (= n on the unit square)
    """
    n = len(points)
    if n < 2:
        raise ValueError("Need at least 2 points to estimate K.")

    lam_hat = float(n)   # points per unit area on [0,1]^2

    dx = points[:, 0:1] - points[:, 0]   # (n, n)
    dy = points[:, 1:2] - points[:, 1]

    dist = np.sqrt(dx**2 + dy**2)

    # Edge-correction weights: area of W ∩ W_h for unit square
    wx = np.clip(1.0 - np.abs(dx), 0.0, None)
    wy = np.clip(1.0 - np.abs(dy), 0.0, None)
    weight = wx * wy

    np.fill_diagonal(dist, np.inf)
    np.fill_diagonal(weight, 1.0)   # never accessed (dist=inf on diagonal)

    K_vals = np.empty(len(t_values))
    for k, t in enumerate(t_values):
        mask = dist <= t
        K_vals[k] = np.sum(1.0 / weight[mask]) / lam_hat**2

    return K_vals, lam_hat


# ══════════════════════════════════════════════════════════════════════════════
# Theoretical K-function for the Thomas process
# ══════════════════════════════════════════════════════════════════════════════

def K_thomas(t: np.ndarray, kappa: float, sigma: float) -> np.ndarray:
    """Theoretical K-function: K(t) = pi t^2 + (1 - exp(-t^2/(4sigma^2)))/kappa."""
    return np.pi * t**2 + (1.0 - np.exp(-t**2 / (4.0 * sigma**2))) / kappa


# ══════════════════════════════════════════════════════════════════════════════
# Minimum contrast objective and optimisation
# ══════════════════════════════════════════════════════════════════════════════

def _contrast_objective(
        log_params: np.ndarray,
        t_values: np.ndarray,
        K_emp: np.ndarray,
        c: float,
        dt: float,
) -> float:
    kappa = np.exp(log_params[0])
    sigma = np.exp(log_params[1])
    K_theo = K_thomas(t_values, kappa, sigma)
    return float(np.sum((K_emp**c - K_theo**c) ** 2) * dt)


def fit(
        points: np.ndarray,
        r_low: float = 1e-3,
        r_high: float = 0.25,
        n_t: int = 200,
        c: float = 0.25,
        kappa0: float | None = None,
        sigma0: float | None = None,
        rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """Fit Thomas process parameters by minimum contrast on K in [0,1]^2.

    Parameters
    ----------
    points  : (N, 2) array of points already in [0,1]^2 (post crop_and_rescale)
    r_low   : lower integration limit (unit-square units; default 1e-3)
    r_high  : upper integration limit (unit-square units; default 0.25,
              i.e. a quarter of the window side — standard rule of thumb)
    n_t     : number of quadrature points for the Riemann sum
    c       : contrast exponent (0.25 is the standard choice)
    kappa0  : initial guess for parent_intensity (default: lam_hat / 5)
    sigma0  : initial guess for cluster_scale   (default: r_high / 5)

    Returns
    -------
    dict with keys matching the training params schema:
        parent_intensity, cluster_scale, mean_offspring
    plus diagnostics: lambda_hat, converged, contrast_value, t_values, K_emp
    """
    points = np.asarray(points, dtype=float)
    t_values = np.linspace(r_low, r_high, n_t)
    dt = t_values[1] - t_values[0]

    K_emp, lam_hat = empirical_K(points, t_values)

    kappa0 = kappa0 if kappa0 is not None else lam_hat / 5.0
    sigma0 = sigma0 if sigma0 is not None else r_high / 5.0
    x0 = [np.log(kappa0), np.log(sigma0)]

    res = minimize(
        _contrast_objective,
        x0,
        args=(t_values, K_emp, c, dt),
        method="Nelder-Mead",
        options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 5000},
    )

    kappa_hat = float(np.exp(res.x[0]))
    sigma_hat = float(np.exp(res.x[1]))
    mu_hat = lam_hat / kappa_hat   # mean_offspring = lambda / kappa

    return {
        # ── keys that match the training params schema ──────────────────────
        "parent_intensity": kappa_hat,
        "cluster_scale":    sigma_hat,
        "mean_offspring":   mu_hat,
        # ── diagnostics ─────────────────────────────────────────────────────
        "lambda_hat":       lam_hat,
        "converged":        bool(res.success),
        "contrast_value":   float(res.fun),
        "t_values":         t_values,
        "K_emp":            K_emp,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Public entry point: estimate() — takes a cloud path, returns params dict
# ══════════════════════════════════════════════════════════════════════════════

def estimate(
        cloud_path: Path | str,
        seed: int = 0,
        target_n: int = TRAIN_TARGET_N,
        r_low: float = 1e-3,
        r_high: float = 0.25,
        n_t: int = 200,
        c: float = 0.25,
) -> dict[str, Any]:
    """Full pipeline: load cloud → crop/rescale → fit → return params.

    This is the function called by compare.py (and any future harness).
    The crop uses the same RNG seed as processing.py (default 0) so that
    mincontrast and the TDA model see the same sub-window of the plot.

    Parameters
    ----------
    cloud_path : path to a *_cloud.pkl file (project format)
    seed       : RNG seed for the random sub-window crop

    Returns
    -------
    dict with keys: parent_intensity, cluster_scale, mean_offspring
    (plus diagnostics — compare.py picks only the three parameter keys)
    """
    rng = np.random.default_rng(seed)
    cloud = load_cloud(Path(cloud_path))

    points = np.asarray(cloud["points"], dtype=float)
    reg = cloud.get("region", {})
    low  = np.asarray(reg.get("low",  [0.0, 0.0]), dtype=float)
    high = np.asarray(reg.get("high", [1.0, 1.0]), dtype=float)

    points_unit = crop_and_rescale(points, low, high, target_n=target_n, rng=rng)
    return fit(points_unit, r_low=r_low, r_high=r_high, n_t=n_t, c=c)


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cloud", type=Path,
                   default=Path("data/barro/bci.tree1_cloud.pkl"),
                   help="path to a *_cloud.pkl file")
    p.add_argument("--seed", type=int, default=0,
                   help="RNG seed for the sub-window crop (default: 0)")
    p.add_argument("--target-n", type=int, default=TRAIN_TARGET_N,
                   help="target point count after crop (default: %(default)s)")
    p.add_argument("--r-high", type=float, default=0.25,
                   help="upper K integration limit in [0,1]^2 units (default: 0.25)")
    p.add_argument("--n-t", type=int, default=200,
                   help="quadrature points for the Riemann sum (default: 200)")
    p.add_argument("--c", type=float, default=0.25,
                   help="contrast exponent (default: 0.25)")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    result = estimate(
        cloud_path=args.cloud,
        seed=args.seed,
        target_n=args.target_n,
        r_high=args.r_high,
        n_t=args.n_t,
        c=args.c,
    )
    print(f"\nMinimum-contrast estimates for {args.cloud.name}")
    print(f"  parent_intensity : {result['parent_intensity']:.6g}")
    print(f"  cluster_scale    : {result['cluster_scale']:.6g}")
    print(f"  mean_offspring   : {result['mean_offspring']:.6g}")
    print(f"  lambda_hat       : {result['lambda_hat']:.4g}  (points on unit square)")
    print(f"  converged        : {result['converged']}")
    print(f"  contrast value   : {result['contrast_value']:.6g}")


if __name__ == "__main__":
    main()