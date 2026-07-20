# src/cloudforger/core/metrics.py
"""Generic, estimator-agnostic metric formulas shared by every method
(neural or classical) in a results comparison. Promoted from
models/evaluate.py -- kept deliberately small and pure-numpy so any new
estimator (baseline or nn method) can depend on it without pulling in that
file's classical-estimator-specific comparison harness (see
cloudforger.evaluate / Stage 5 for that)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def finite_array(values: list[float] | np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=float)


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def json_default(obj: Any) -> Any:
    """JSON serializer for numpy scalars/arrays and pathlib paths."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def compute_marginal_metrics(truth: np.ndarray, estimates: np.ndarray) -> dict[str, float]:
    """Compute one-estimator metrics after dropping its own non-finite rows."""
    truth = finite_array(truth)
    estimates = finite_array(estimates)
    mask = np.isfinite(truth) & np.isfinite(estimates)

    if mask.sum() < 2:
        return {
            "n": int(mask.sum()),
            "bias": float("nan"),
            "mae": float("nan"),
            "rmse": float("nan"),
            "rel_bias": float("nan"),
            "rel_rmse": float("nan"),
            "median_ratio": float("nan"),
            "log_bias": float("nan"),
            "log_mae": float("nan"),
            "log_rmse": float("nan"),
            "median_abs_log_error": float("nan"),
        }

    t, e = truth[mask], estimates[mask]
    errors = e - t
    denom = np.clip(t, 1e-12, None)
    rel_errors = errors / denom
    ratios = e / denom

    positive = (t > 0) & (e > 0)
    if positive.sum() >= 2:
        log_errors = np.log(e[positive]) - np.log(t[positive])
        log_bias = float(np.mean(log_errors))
        log_mae = float(np.mean(np.abs(log_errors)))
        log_rmse = float(np.sqrt(np.mean(log_errors**2)))
        med_abs_log = float(np.median(np.abs(log_errors)))
    else:
        log_bias = log_mae = log_rmse = med_abs_log = float("nan")

    return {
        "n": int(mask.sum()),
        "bias": float(np.mean(errors)),
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "rel_bias": float(np.mean(rel_errors)),
        "rel_rmse": float(np.sqrt(np.mean(rel_errors**2))),
        "median_ratio": float(np.median(ratios)),
        "log_bias": log_bias,
        "log_mae": log_mae,
        "log_rmse": log_rmse,
        "median_abs_log_error": med_abs_log,
    }


def bootstrap_mean_ci(
    values: np.ndarray,
    *,
    rng: np.random.Generator,
    n_boot: int = 1000,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of paired differences."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    if n < 3 or n_boot <= 0:
        return float("nan"), float("nan")

    boot_means = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        sample = rng.choice(values, size=n, replace=True)
        boot_means[b] = np.mean(sample)

    lo, hi = np.percentile(boot_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def compute_paired_metrics(
    truth: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    *,
    first_name: str,
    second_name: str,
    rng: np.random.Generator,
    n_boot: int = 1000,
) -> dict[str, float | str]:
    """Compare two estimators on the same finite, positive clouds.

    Delta quantities are ``second - first``. Negative delta_log_mae therefore
    means the second estimator has lower absolute log error.
    """
    truth = finite_array(truth)
    first = finite_array(first)
    second = finite_array(second)

    mask = (
        np.isfinite(truth)
        & np.isfinite(first)
        & np.isfinite(second)
        & (truth > 0)
        & (first > 0)
        & (second > 0)
    )

    if mask.sum() < 2:
        return {
            "first": first_name,
            "second": second_name,
            "n_common": int(mask.sum()),
            "first_log_mae": float("nan"),
            "second_log_mae": float("nan"),
            "delta_log_mae": float("nan"),
            "delta_log_mae_ci_low": float("nan"),
            "delta_log_mae_ci_high": float("nan"),
            "median_delta_abs_log_error": float("nan"),
            "second_win_rate": float("nan"),
            "first_win_rate": float("nan"),
            "tie_rate": float("nan"),
            "first_log_rmse": float("nan"),
            "second_log_rmse": float("nan"),
            "delta_log_rmse": float("nan"),
        }

    t, a, b = truth[mask], first[mask], second[mask]
    abs_log_a = np.abs(np.log(a) - np.log(t))
    abs_log_b = np.abs(np.log(b) - np.log(t))
    delta_abs_log = abs_log_b - abs_log_a

    ci_low, ci_high = bootstrap_mean_ci(delta_abs_log, rng=rng, n_boot=n_boot)

    tol = 1e-12
    second_wins = abs_log_b < abs_log_a - tol
    first_wins = abs_log_a < abs_log_b - tol
    ties = ~(second_wins | first_wins)

    log_err_a = np.log(a) - np.log(t)
    log_err_b = np.log(b) - np.log(t)

    return {
        "first": first_name,
        "second": second_name,
        "n_common": int(mask.sum()),
        "first_log_mae": float(np.mean(abs_log_a)),
        "second_log_mae": float(np.mean(abs_log_b)),
        "delta_log_mae": float(np.mean(delta_abs_log)),
        "delta_log_mae_ci_low": ci_low,
        "delta_log_mae_ci_high": ci_high,
        "median_delta_abs_log_error": float(np.median(delta_abs_log)),
        "second_win_rate": float(np.mean(second_wins)),
        "first_win_rate": float(np.mean(first_wins)),
        "tie_rate": float(np.mean(ties)),
        "first_log_rmse": float(np.sqrt(np.mean(log_err_a**2))),
        "second_log_rmse": float(np.sqrt(np.mean(log_err_b**2))),
        "delta_log_rmse": float(np.sqrt(np.mean(log_err_b**2)) - np.sqrt(np.mean(log_err_a**2))),
    }
