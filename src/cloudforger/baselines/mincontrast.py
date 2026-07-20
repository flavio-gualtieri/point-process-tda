# src/cloudforger/baselines/mincontrast.py
"""Minimum-contrast estimator for the stationary Thomas process (Waagpetersen-
style power-transformed K-function contrast). Promoted from
models/mincontrast.py, with n_starts finally threaded through as an explicit,
exposed parameter -- previously estimate() silently used fit_multistart's
hardcoded default with no way to override it from the CLI or evaluation
harness."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize

TRAIN_TARGET_N = 900   # target point count after sub-window crop


def load_cloud(path: Path) -> dict[str, Any]:
    with open(path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, list):
        data = data[0]
    return data


def crop_and_rescale(
        points: np.ndarray,
        region_low: np.ndarray,
        region_high: np.ndarray,
        target_n: int = TRAIN_TARGET_N,
        rng: np.random.Generator | None = None,
) -> np.ndarray:
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


def empirical_K(
        points: np.ndarray,
        t_values: np.ndarray,
) -> tuple[np.ndarray, float]:
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


def K_thomas(t: np.ndarray, kappa: float, sigma: float) -> np.ndarray:
    return np.pi * t**2 + (1.0 - np.exp(-t**2 / (4.0 * sigma**2))) / kappa


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
) -> dict[str, Any]:
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
        "cluster_scale": sigma_hat,
        "mean_offspring": mu_hat,
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
        c: float = 0.25,
        n_starts: int = 10,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    cloud = load_cloud(Path(cloud_path))

    points = np.asarray(cloud["points"], dtype=float)
    reg = cloud.get("region", {})
    low  = np.asarray(reg.get("low",  [0.0, 0.0]), dtype=float)
    high = np.asarray(reg.get("high", [1.0, 1.0]), dtype=float)

    points_unit = crop_and_rescale(points, low, high, target_n=target_n, rng=rng)

    return fit_multistart(points_unit, n_starts=n_starts, rng=rng, r_low=r_low, r_high=r_high, n_t=n_t, c=c)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cloud", type=Path, required=True, help="path to a *_cloud.pkl file")
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
    p.add_argument("--n-starts", type=int, default=10,
                   help="multistart restart count (default: 10)")

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
        n_starts=args.n_starts,
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
