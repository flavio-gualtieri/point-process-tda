# models/evaluate.py

from __future__ import annotations

import argparse
import itertools
import json
import pickle
import time
import traceback
import warnings
from pathlib import Path
from typing import Any, Callable

import numpy as np

# ── estimator imports ─────────────────────────────────────────────────────────
try:
    import mincontrast as mc

    _MC = True
except ImportError:
    mc = None  # type: ignore[assignment]
    _MC = False
    warnings.warn("mincontrast.py not found — 'mincontrast' estimator unavailable.")

try:
    import phnn

    _PHNN = True
except ImportError:
    phnn = None  # type: ignore[assignment]
    _PHNN = False
    warnings.warn("phnn.py not importable — 'neural' estimator unavailable.")

try:
    import baselines

    _PALM = True
except ImportError:
    baselines = None  # type: ignore[assignment]
    _PALM = False
    warnings.warn("baselines.py not importable — 'palm' estimator unavailable.")

try:
    import pandas as pd

    _PANDAS = True
except ImportError:
    pd = None  # type: ignore[assignment]
    _PANDAS = False


# ══════════════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════════════

PARAMS: tuple[str, ...] = ("parent_intensity", "cluster_scale", "mean_offspring")

DEFAULT_CLOUDS = Path("data/params/2d/thomas/adversarial_clouds.pkl")

REGIME_CRITERIA: dict[str, dict[str, str]] = {
    "strong_tight": {"parent_intensity": "HIGH", "mean_offspring": "HIGH", "cluster_scale": "LOW"},
    "strong_diffuse": {"parent_intensity": "HIGH", "mean_offspring": "HIGH", "cluster_scale": "HIGH"},
    "weak": {"parent_intensity": "LOW", "mean_offspring": "LOW", "cluster_scale": "MID"},
}

ESTIMATOR_PREFIXES: dict[str, str] = {
    "mincontrast": "mc",
    "neural": "nn",
    "palm": "palm",
}

# Estimators that operate on the raw point cloud and expose the mincontrast
# interface (crop_and_rescale + fit_multistart). Neural is handled separately.
def _points_modules() -> dict[str, Any]:
    return {
        "mincontrast": mc if _MC else None,
        "palm": baselines if _PALM else None,
    }

# When two estimators are paired, the one with the HIGHER priority is placed
# "second", so delta_log_mae = second - first and a negative delta favours the
# more advanced method (matching the original neural-vs-mincontrast convention).
PAIR_PRIORITY: dict[str, int] = {"mincontrast": 0, "palm": 1, "neural": 2}


# ══════════════════════════════════════════════════════════════════════════════
# Small utilities
# ══════════════════════════════════════════════════════════════════════════════

def now() -> float:
    return time.perf_counter()


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


# ══════════════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════════════

def load_clouds(path: Path) -> list[dict[str, Any]]:
    """Load a clouds pickle as a list of dicts with at least points/params/seed."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    if not isinstance(data, list):
        data = [data]
    return data


def is_unit_square(cloud: dict[str, Any]) -> bool:
    """Check whether the cloud region is [0,1]^2, or absent and assumed [0,1]^2."""
    reg = cloud.get("region", {})
    low = np.asarray(reg.get("low", [0.0, 0.0]), dtype=float)
    high = np.asarray(reg.get("high", [1.0, 1.0]), dtype=float)
    return np.allclose(low, 0.0) and np.allclose(high, 1.0)


def infer_betti_path(clouds_path: Path) -> Path:
    """Infer the sibling betti.pkl for a clouds.pkl path."""
    name = clouds_path.name
    if "clouds" not in name:
        raise ValueError(f"Cannot infer betti path from {clouds_path} because filename has no 'clouds'.")
    return clouds_path.with_name(name.replace("clouds", "betti", 1))


def load_betti_curves(path: Path, homology_dim: int = 0) -> tuple[dict[Any, np.ndarray], str]:
    """Load betti curves and return ({seed: curve}, process_name)."""
    with open(path, "rb") as f:
        payload = pickle.load(f)

    seeds = payload["seeds"]
    matrix_key = f"betti{homology_dim}_matrix"
    if matrix_key not in payload:
        raise KeyError(
            f"{path} has no {matrix_key!r}; recompute Betti curves with homology_dim={homology_dim}."
        )

    matrix = payload[matrix_key]
    if len(seeds) != len(matrix):
        raise ValueError(f"{path}: {len(seeds)} seeds but {len(matrix)} curves — corrupt betti.pkl.")

    by_seed: dict[Any, np.ndarray] = {}
    for seed, curve in zip(seeds, matrix):
        if seed in by_seed:
            raise ValueError(f"{path}: duplicate seed {seed!r}; cannot uniquely join to clouds.")
        by_seed[seed] = np.asarray(curve)

    return by_seed, payload.get("process", "")


def attach_betti_curves(clouds: list[dict[str, Any]], betti_path: Path, homology_dim: int = 0) -> None:
    """Join betti curves to clouds in-place by seed.

    The process stored in clouds and betti.pkl is also checked, because seeds are
    commonly reused across train/test/adversarial splits and across processes.
    """
    by_seed, betti_process = load_betti_curves(betti_path, homology_dim=homology_dim)

    cloud_processes = {c.get("process") for c in clouds if c.get("process")}
    if betti_process and cloud_processes and cloud_processes != {betti_process}:
        raise ValueError(
            f"Process mismatch: clouds are {sorted(cloud_processes)} but {betti_path} "
            f"was computed for process={betti_process!r}. Refusing to join by seed."
        )

    seeds_seen: set[Any] = set()
    n_matched = 0
    curve_key = f"betti{homology_dim}_curve"

    for cloud in clouds:
        seed = cloud.get("seed")
        if seed is None:
            raise ValueError("Cloud record is missing 'seed' — cannot join Betti curves.")
        if seed in seeds_seen:
            raise ValueError(f"Duplicate seed {seed!r} among clouds — cannot uniquely join Betti curves.")
        seeds_seen.add(seed)

        curve = by_seed.get(seed)
        if curve is not None:
            cloud[curve_key] = curve
            # Backwards-compatible key expected by the original PH-NN runner.
            if homology_dim == 0:
                cloud["betti0_curve"] = curve
            n_matched += 1

    if n_matched < len(clouds):
        warnings.warn(
            f"Only {n_matched}/{len(clouds)} clouds matched a Betti curve by seed "
            f"from {betti_path}. Unmatched clouds will fail the neural estimator."
        )


# ══════════════════════════════════════════════════════════════════════════════
# Regime assignment
# ══════════════════════════════════════════════════════════════════════════════

def assign_terciles(values: np.ndarray) -> np.ndarray:
    """Assign LOW/MID/HIGH tercile labels to positive values on a log scale."""
    values = np.asarray(values, dtype=float)
    t1, t2 = np.percentile(values, [33.3, 66.7])
    return np.where(values <= t1, "LOW", np.where(values <= t2, "MID", "HIGH"))


def assign_regimes(clouds: list[dict[str, Any]], ridge_log_width: float | None = None) -> np.ndarray:
    """Return a regime label for each cloud.

    Named regimes are based on terciles of log-parameters. Optionally, clouds
    can also be labelled as ``ridge`` when parent_intensity * mean_offspring is
    close to the sample median on a log scale and the two factors trade off.
    This optional rule is disabled by default because ridge definitions are
    experiment-specific.
    """
    params_arr = {p: np.array([c["params"][p] for c in clouds], dtype=float) for p in PARAMS}
    terciles = {p: assign_terciles(np.log(params_arr[p])) for p in PARAMS}

    regimes = np.full(len(clouds), "other", dtype=object)

    for regime_name, criteria in REGIME_CRITERIA.items():
        mask = np.ones(len(clouds), dtype=bool)
        for param, level in criteria.items():
            mask &= terciles[param] == level
        regimes[mask] = regime_name

    if ridge_log_width is not None and ridge_log_width > 0:
        log_product = np.log(params_arr["parent_intensity"] * params_arr["mean_offspring"])
        near_constant_intensity = np.abs(log_product - np.median(log_product)) <= ridge_log_width
        inverse_tradeoff = (
            ((terciles["parent_intensity"] == "HIGH") & (terciles["mean_offspring"] == "LOW"))
            | ((terciles["parent_intensity"] == "LOW") & (terciles["mean_offspring"] == "HIGH"))
        )
        regimes[near_constant_intensity & inverse_tradeoff] = "ridge"

    return regimes


# ══════════════════════════════════════════════════════════════════════════════
# Estimator runners
# ══════════════════════════════════════════════════════════════════════════════

def blank_estimate() -> dict[str, float]:
    return {p: float("nan") for p in PARAMS}


def run_points_estimator(
    cloud: dict[str, Any],
    module: Any,
    *,
    rng: np.random.Generator | None = None,
    include_tracebacks: bool = False,
    **kwargs: Any,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Run any point-cloud estimator exposing the mincontrast interface
    (``crop_and_rescale`` + ``fit_multistart``) and return (estimates, diagnostics).

    Used for both 'mincontrast' and 'palm'."""
    diagnostics: dict[str, Any] = {
        "ok": False,
        "time_sec": float("nan"),
        "error": "",
        "traceback": "",
        "converged": None,
        "contrast_value": float("nan"),
        "lambda_hat": float("nan"),
    }
    t0 = now()

    try:
        if module is None:
            raise RuntimeError("estimator module is unavailable")

        points = np.asarray(cloud["points"], dtype=float)
        if len(points) < 5:
            raise ValueError(f"Need at least 5 points; got {len(points)}")

        if not is_unit_square(cloud):
            reg = cloud.get("region", {})
            low = np.asarray(reg.get("low", [0.0, 0.0]), dtype=float)
            high = np.asarray(reg.get("high", [1.0, 1.0]), dtype=float)
            points = module.crop_and_rescale(points, low, high)
            if len(points) < 5:
                raise ValueError(f"Crop/rescale left fewer than 5 points; got {len(points)}")

        result = module.fit_multistart(points, rng=rng, **kwargs)
        if result is None:
            raise RuntimeError("fit_multistart returned None")

        estimates = {p: safe_float(result.get(p, float("nan"))) for p in PARAMS}
        diagnostics.update(
            ok=all(np.isfinite(estimates[p]) for p in PARAMS),
            converged=bool(result.get("converged", False)),
            contrast_value=safe_float(result.get("contrast_value", float("nan"))),
            lambda_hat=safe_float(result.get("lambda_hat", float("nan"))),
        )
        return estimates, diagnostics

    except Exception as exc:  # noqa: BLE001 — failure is recorded per cloud.
        diagnostics["error"] = repr(exc)
        if include_tracebacks:
            diagnostics["traceback"] = traceback.format_exc()
        return blank_estimate(), diagnostics

    finally:
        diagnostics["time_sec"] = now() - t0


def run_mincontrast(
    cloud: dict[str, Any],
    *,
    rng: np.random.Generator | None = None,
    include_tracebacks: bool = False,
    **kwargs: Any,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Backwards-compatible wrapper: minimum contrast via run_points_estimator."""
    return run_points_estimator(
        cloud, mc if _MC else None, rng=rng, include_tracebacks=include_tracebacks, **kwargs
    )


def run_palm(
    cloud: dict[str, Any],
    *,
    rng: np.random.Generator | None = None,
    include_tracebacks: bool = False,
    **kwargs: Any,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Convenience wrapper: Palm likelihood via run_points_estimator."""
    return run_points_estimator(
        cloud, baselines if _PALM else None, rng=rng, include_tracebacks=include_tracebacks, **kwargs
    )


def build_neural_model(model_path: Path | None = None, results_path: Path | None = None) -> Any:
    """Construct the PH-NN model, optionally overriding saved paths."""
    if not _PHNN or phnn is None:
        raise RuntimeError("phnn module is unavailable")

    kwargs: dict[str, str] = {}
    if model_path is not None:
        kwargs["model_path"] = str(model_path)
    if results_path is not None:
        kwargs["results_path"] = str(results_path)
    return phnn.Model(**kwargs)


def run_neural(
    cloud: dict[str, Any],
    *,
    model: Any,
    homology_dim: int = 0,
    include_tracebacks: bool = False,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Run PH-NN and return (estimates, diagnostics)."""
    diagnostics: dict[str, Any] = {
        "ok": False,
        "time_sec": float("nan"),
        "error": "",
        "traceback": "",
    }
    t0 = now()

    try:
        if model is None:
            raise RuntimeError("neural model was not initialized")

        curve_key = f"betti{homology_dim}_curve"
        curve = cloud.get(curve_key)
        if curve is None and homology_dim == 0:
            curve = cloud.get("betti0_curve")
        if curve is None:
            raise ValueError(f"cloud has no {curve_key!r}; attach Betti curves before evaluation")

        import torch

        t = torch.as_tensor(np.asarray(curve, dtype=np.float32))
        result = model.forward(betti_0=t)

        if isinstance(result, dict):
            estimates = {p: safe_float(result.get(p, float("nan"))) for p in PARAMS}
        else:
            arr = np.asarray(result.detach().cpu(), dtype=float)
            if arr.ndim == 2:
                arr = arr[0]
            estimates = {p: safe_float(arr[i]) for i, p in enumerate(PARAMS)}

        missing = [p for p in PARAMS if not np.isfinite(estimates[p])]
        if missing:
            raise ValueError(f"PH-NN returned missing/non-finite estimates for {missing}")

        diagnostics["ok"] = True
        return estimates, diagnostics

    except Exception as exc:  # noqa: BLE001 — failure is recorded per cloud.
        diagnostics["error"] = repr(exc)
        if include_tracebacks:
            diagnostics["traceback"] = traceback.format_exc()
        return blank_estimate(), diagnostics

    finally:
        diagnostics["time_sec"] = now() - t0


# ══════════════════════════════════════════════════════════════════════════════
# Metrics
# ══════════════════════════════════════════════════════════════════════════════

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


def collect_param_arrays(
    raw_rows: list[dict[str, Any]],
    mask: np.ndarray,
    param: str,
    prefix: str | None = None,
) -> np.ndarray:
    if prefix is None:
        key = f"true_{param}"
    else:
        key = f"{prefix}_{param}"
    return np.array([row[key] for row, keep in zip(raw_rows, mask) if keep], dtype=float)


def ordered_pair(a: str, b: str) -> tuple[str, str]:
    """Return (first, second) with the higher-priority estimator second."""
    return tuple(sorted((a, b), key=lambda e: PAIR_PRIORITY.get(e, 99)))  # type: ignore[return-value]


# ══════════════════════════════════════════════════════════════════════════════
# Main evaluation loop
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(
    clouds: list[dict[str, Any]],
    estimator_names: list[str],
    *,
    mc_kwargs: dict[str, Any] | None = None,
    mc_seed: int = 420937,
    neural_model: Any | None = None,
    homology_dim: int = 0,
    ridge_log_width: float | None = None,
    bootstrap: int = 1000,
    bootstrap_seed: int = 20240707,
    include_tracebacks: bool = False,
) -> dict[str, Any]:
    """Run estimators on all clouds and return raw rows plus aggregate metrics."""
    mc_kwargs = mc_kwargs or {}
    requested = list(dict.fromkeys(estimator_names))
    regimes = assign_regimes(clouds, ridge_log_width=ridge_log_width)

    availability = {"mincontrast": _MC, "neural": _PHNN, "palm": _PALM}
    active_estimators: list[str] = []
    for est in requested:
        if est in availability and availability[est]:
            active_estimators.append(est)
        else:
            warnings.warn(f"Skipping unavailable or unknown estimator: {est!r}")

    if "neural" in active_estimators and neural_model is None:
        neural_model = build_neural_model()

    points_modules = _points_modules()
    points_estimators = [e for e in active_estimators if e in points_modules]
    use_neural = "neural" in active_estimators

    # Independent RNG stream per point-cloud estimator so their crops/starts do
    # not share state.
    rng_by_est = {
        name: np.random.default_rng(mc_seed + i) for i, name in enumerate(points_estimators)
    }
    boot_rng = np.random.default_rng(bootstrap_seed)

    raw_rows: list[dict[str, Any]] = []
    n_clouds = len(clouds)

    for i, cloud in enumerate(clouds):
        if i == 0 or (i + 1) % 100 == 0 or i + 1 == n_clouds:
            print(f"  cloud {i + 1}/{n_clouds} ...", flush=True)

        row: dict[str, Any] = {
            "index": i,
            "seed": cloud.get("seed"),
            "process": cloud.get("process", ""),
            "regime": regimes[i],
            "n_points": len(cloud["points"]),
        }
        for p in PARAMS:
            row[f"true_{p}"] = safe_float(cloud["params"].get(p, float("nan")))

        for name in points_estimators:
            prefix = ESTIMATOR_PREFIXES[name]
            estimates, diag = run_points_estimator(
                cloud,
                points_modules[name],
                rng=rng_by_est[name],
                include_tracebacks=include_tracebacks,
                **mc_kwargs,
            )
            for p in PARAMS:
                row[f"{prefix}_{p}"] = estimates[p]
            for k, v in diag.items():
                row[f"{prefix}_{k}"] = v

        if use_neural:
            prefix = ESTIMATOR_PREFIXES["neural"]
            estimates, diag = run_neural(
                cloud,
                model=neural_model,
                homology_dim=homology_dim,
                include_tracebacks=include_tracebacks,
            )
            for p in PARAMS:
                row[f"{prefix}_{p}"] = estimates[p]
            for k, v in diag.items():
                row[f"{prefix}_{k}"] = v

        raw_rows.append(row)

    metrics = compute_all_metrics(
        raw_rows,
        regimes=regimes,
        active_estimators=active_estimators,
        bootstrap=bootstrap,
        rng=boot_rng,
    )

    summary = {
        "n_clouds": n_clouds,
        "active_estimators": active_estimators,
        "params": list(PARAMS),
        "regime_counts": {str(r): int(np.sum(regimes == r)) for r in sorted(set(regimes))},
    }

    return {"raw": raw_rows, "metrics": metrics, "regimes": regimes, "summary": summary}


def compute_all_metrics(
    raw_rows: list[dict[str, Any]],
    *,
    regimes: np.ndarray,
    active_estimators: list[str],
    bootstrap: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Build marginal and (all-pairwise) paired metric dictionaries."""
    regime_labels = sorted(set(regimes)) + ["ALL"]

    metrics: dict[str, Any] = {
        "marginal": {},
        "paired": {},
        "failures": {},
        "runtime": {},
    }

    for regime in regime_labels:
        if regime == "ALL":
            mask = np.ones(len(raw_rows), dtype=bool)
        else:
            mask = np.array([row["regime"] == regime for row in raw_rows], dtype=bool)

        if not mask.any():
            continue

        metrics["marginal"][regime] = {}
        metrics["failures"][regime] = {}
        metrics["runtime"][regime] = {}

        for est_name in active_estimators:
            prefix = ESTIMATOR_PREFIXES[est_name]
            metrics["marginal"][regime][est_name] = {}

            ok_values = np.array([bool(row.get(f"{prefix}_ok", False)) for row, keep in zip(raw_rows, mask) if keep])
            metrics["failures"][regime][est_name] = {
                "n_total": int(mask.sum()),
                "n_ok": int(ok_values.sum()),
                "n_failed": int(len(ok_values) - ok_values.sum()),
                "failure_rate": float(1.0 - ok_values.mean()) if len(ok_values) else float("nan"),
            }

            times = np.array([row.get(f"{prefix}_time_sec", float("nan")) for row, keep in zip(raw_rows, mask) if keep], dtype=float)
            finite_times = times[np.isfinite(times)]
            metrics["runtime"][regime][est_name] = {
                "n_timed": int(len(finite_times)),
                "mean_time_sec": float(np.mean(finite_times)) if len(finite_times) else float("nan"),
                "median_time_sec": float(np.median(finite_times)) if len(finite_times) else float("nan"),
                "total_time_sec": float(np.sum(finite_times)) if len(finite_times) else float("nan"),
            }

            for p in PARAMS:
                truth = collect_param_arrays(raw_rows, mask, p)
                estimates = collect_param_arrays(raw_rows, mask, p, prefix)
                metrics["marginal"][regime][est_name][p] = compute_marginal_metrics(truth, estimates)

        # All pairwise comparisons among the active estimators.
        if len(active_estimators) >= 2:
            metrics["paired"].setdefault(regime, {})
            for a, b in itertools.combinations(active_estimators, 2):
                first, second = ordered_pair(a, b)
                pair_key = f"{first}_vs_{second}"
                metrics["paired"][regime][pair_key] = {}
                for p in PARAMS:
                    truth = collect_param_arrays(raw_rows, mask, p)
                    first_arr = collect_param_arrays(raw_rows, mask, p, ESTIMATOR_PREFIXES[first])
                    second_arr = collect_param_arrays(raw_rows, mask, p, ESTIMATOR_PREFIXES[second])
                    metrics["paired"][regime][pair_key][p] = compute_paired_metrics(
                        truth,
                        first_arr,
                        second_arr,
                        first_name=first,
                        second_name=second,
                        rng=rng,
                        n_boot=bootstrap,
                    )

    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# Output
# ══════════════════════════════════════════════════════════════════════════════

def print_marginal_metrics(metrics: dict[str, Any]) -> None:
    """Print one-estimator metrics."""
    for regime, by_est in sorted(metrics["marginal"].items()):
        print(f"\n{'═' * 96}")
        print(f"  Marginal metrics — Regime: {regime}")
        print(f"{'═' * 96}")
        header = (
            f"  {'estimator':<14} {'param':<18} {'n':>5} "
            f"{'bias':>11} {'rmse':>11} {'rel_rmse':>11} {'log_mae':>11} {'med_ratio':>11}"
        )
        print(header)
        print(f"  {'-' * (len(header) - 2)}")

        for est_name, by_param in sorted(by_est.items()):
            for p, m in by_param.items():
                print(
                    f"  {est_name:<14} {p:<18} {m['n']:>5d} "
                    f"{m['bias']:>11.4g} {m['rmse']:>11.4g} {m['rel_rmse']:>11.4g} "
                    f"{m['log_mae']:>11.4g} {m['median_ratio']:>11.4g}"
                )


def print_paired_metrics(metrics: dict[str, Any]) -> None:
    """Print same-cloud paired comparison metrics for every estimator pair."""
    if not metrics.get("paired"):
        return

    for regime, by_pair in sorted(metrics["paired"].items()):
        for pair_name, by_param in sorted(by_pair.items()):
            # Names are stored on each metric dict; read them from any param.
            sample = next(iter(by_param.values()))
            first = str(sample.get("first", "first"))
            second = str(sample.get("second", "second"))

            print(f"\n{'═' * 112}")
            print(f"  Paired metrics — {pair_name} — Regime: {regime}")
            print(f"  Delta is {second} - {first}; negative delta_log_mae favors {second}.")
            print(f"{'═' * 112}")
            f_lbl = f"{first[:10]}_lmae"
            s_lbl = f"{second[:10]}_lmae"
            w_lbl = f"{second[:6]}_win%"
            header = (
                f"  {'param':<18} {'n_common':>8} {f_lbl:>13} {s_lbl:>13} "
                f"{'delta':>12} {'CI_low':>12} {'CI_high':>12} {w_lbl:>10}"
            )
            print(header)
            print(f"  {'-' * (len(header) - 2)}")

            for p, m in by_param.items():
                print(
                    f"  {p:<18} {m['n_common']:>8d} "
                    f"{m['first_log_mae']:>13.4g} {m['second_log_mae']:>13.4g} "
                    f"{m['delta_log_mae']:>12.4g} {m['delta_log_mae_ci_low']:>12.4g} "
                    f"{m['delta_log_mae_ci_high']:>12.4g} {100 * m['second_win_rate']:>9.1f}%"
                )


def print_failure_and_runtime(metrics: dict[str, Any]) -> None:
    """Print failure rates and runtime summary."""
    all_failures = metrics.get("failures", {}).get("ALL", {})
    all_runtime = metrics.get("runtime", {}).get("ALL", {})
    if not all_failures and not all_runtime:
        return

    print(f"\n{'═' * 82}")
    print("  Overall failure and runtime summary")
    print(f"{'═' * 82}")
    header = f"  {'estimator':<14} {'n_total':>8} {'n_failed':>9} {'fail%':>8} {'mean_sec':>11} {'total_sec':>11}"
    print(header)
    print(f"  {'-' * (len(header) - 2)}")

    for est_name in sorted(set(all_failures) | set(all_runtime)):
        f = all_failures.get(est_name, {})
        r = all_runtime.get(est_name, {})
        print(
            f"  {est_name:<14} {f.get('n_total', 0):>8d} {f.get('n_failed', 0):>9d} "
            f"{100 * f.get('failure_rate', float('nan')):>7.1f}% "
            f"{r.get('mean_time_sec', float('nan')):>11.4g} {r.get('total_time_sec', float('nan')):>11.4g}"
        )


def print_metrics(metrics: dict[str, Any]) -> None:
    print_marginal_metrics(metrics)
    print_paired_metrics(metrics)
    print_failure_and_runtime(metrics)


def save_raw_results(raw_rows: list[dict[str, Any]], out_path: Path) -> None:
    """Save per-cloud results to CSV."""
    if not _PANDAS or pd is None:
        print("pandas not available — skipping CSV export.")
        return
    df = pd.DataFrame(raw_rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved per-cloud results → {out_path}")


def save_metrics(metrics: dict[str, Any], out_path: Path) -> None:
    """Save aggregate metrics to JSON."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=json_default)
    print(f"Saved aggregate metrics → {out_path}")


def load_raw_results(path: Path) -> list[dict[str, Any]]:
    """Load a previously saved per-cloud CSV (see save_raw_results) back into rows."""
    if not _PANDAS or pd is None:
        raise RuntimeError("pandas is required to load cached raw results from CSV")
    df = pd.read_csv(path)
    return df.to_dict(orient="records")


def infer_active_estimators(raw_rows: list[dict[str, Any]], requested: list[str]) -> list[str]:
    """Infer which requested estimators actually have columns in cached raw rows."""
    columns = set(raw_rows[0].keys()) if raw_rows else set()
    return [
        est
        for est in dict.fromkeys(requested)
        if est in ESTIMATOR_PREFIXES and f"{ESTIMATOR_PREFIXES[est]}_ok" in columns
    ]


def metrics_from_cached_raw(
    raw_rows: list[dict[str, Any]],
    estimator_names: list[str],
    *,
    bootstrap: int = 1000,
    bootstrap_seed: int = 20240707,
) -> dict[str, Any]:
    """Recompute the full metrics dict from previously saved per-cloud raw rows,
    without re-running any estimator."""
    regimes = np.array([str(row.get("regime", "other")) for row in raw_rows], dtype=object)
    active_estimators = infer_active_estimators(raw_rows, estimator_names)
    boot_rng = np.random.default_rng(bootstrap_seed)

    metrics = compute_all_metrics(
        raw_rows,
        regimes=regimes,
        active_estimators=active_estimators,
        bootstrap=bootstrap,
        rng=boot_rng,
    )

    summary = {
        "n_clouds": len(raw_rows),
        "active_estimators": active_estimators,
        "params": list(PARAMS),
        "regime_counts": {str(r): int(np.sum(regimes == r)) for r in sorted(set(regimes))},
    }

    return {"raw": raw_rows, "metrics": metrics, "regimes": regimes, "summary": summary}


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    p.add_argument(
        "--clouds",
        type=Path,
        default=DEFAULT_CLOUDS,
        help=(
            "path to clouds.pkl, containing point clouds with points/params/seed. "
            f"Default: {DEFAULT_CLOUDS}"
        ),
    )
    p.add_argument(
        "--betti",
        type=Path,
        default=None,
        help="path to betti.pkl for neural estimator; default is inferred from --clouds",
    )
    p.add_argument(
        "--homology-dim",
        type=int,
        default=0,
        help="homology dimension to load from betti.pkl; default: 0",
    )
    p.add_argument(
        "--estimators",
        nargs="+",
        default=["mincontrast", "neural", "palm"],
        choices=["mincontrast", "neural", "palm"],
        help="estimators to run; default: mincontrast neural palm",
    )
    p.add_argument("--max-clouds", type=int, default=None, help="limit number of clouds for quick testing")

    # Point-cloud estimator controls (shared by mincontrast and palm).
    # For palm, --r-high is reused as the Palm t_max and --n-t / --mc-c are ignored.
    p.add_argument("--r-low", type=float, default=1e-3, help="minimum K integration radius; default: 1e-3")
    p.add_argument("--r-high", type=float, default=0.25, help="maximum K radius / Palm t_max; default: 0.25")
    p.add_argument("--n-t", type=int, default=200, help="number of K integration grid points; default: 200")
    p.add_argument("--mc-c", type=float, default=0.25, help="minimum-contrast exponent; default: 0.25")
    p.add_argument("--mc-starts", type=int, default=10, help="multistart count (mincontrast & palm); default: 10")
    p.add_argument("--mc-seed", type=int, default=420937, help="base RNG seed for point-cloud estimators")

    # PH-NN controls.
    p.add_argument("--nn-model", type=Path, default=None, help="path to PH-NN model.pt; default from phnn.Model")
    p.add_argument("--nn-results", type=Path, default=None, help="path to PH-NN results.pt; default from phnn.Model")

    # Evaluation controls.
    p.add_argument(
        "--ridge-log-width",
        type=float,
        default=None,
        help=(
            "optional ridge regime width on log(parent_intensity * mean_offspring); "
            "disabled by default"
        ),
    )
    p.add_argument(
        "--bootstrap",
        type=int,
        default=1000,
        help="paired bootstrap resamples for delta_log_mae CIs; use 0 to disable",
    )
    p.add_argument("--bootstrap-seed", type=int, default=20240707, help="RNG seed for paired bootstrap")
    p.add_argument(
        "--tracebacks",
        action="store_true",
        help="store full exception tracebacks in raw CSV; useful for debugging failed fits",
    )

    # Output.
    p.add_argument("--out", type=Path, default=None, help="path for per-cloud CSV output")
    p.add_argument("--metrics-out", type=Path, default=None, help="path for aggregate metrics JSON output")
    p.add_argument(
        "--force",
        action="store_true",
        help="ignore any cached per-cloud CSV at --out and re-run all estimators",
    )

    return p


def main() -> None:
    args = _build_parser().parse_args()

    use_cache = not args.force and args.out is not None and args.out.exists()

    if use_cache:
        print(f"Found existing per-cloud results at {args.out} — reusing them instead of re-running estimators.")
        print("  (pass --force to ignore this cache and re-run everything)")
        t0 = time.time()
        raw_rows = load_raw_results(args.out)
        if args.max_clouds is not None:
            raw_rows = raw_rows[: args.max_clouds]
        print(f"  {len(raw_rows)} cached rows loaded.")

        results = metrics_from_cached_raw(
            raw_rows,
            args.estimators,
            bootstrap=args.bootstrap,
            bootstrap_seed=args.bootstrap_seed,
        )
        elapsed = time.time() - t0
        print(f"\nStats recomputed from cached results in {elapsed:.1f}s")

    else:
        if args.clouds == DEFAULT_CLOUDS:
            print(f"No --clouds given; defaulting to adversarial held-out split: {DEFAULT_CLOUDS}")

        print(f"Loading clouds from {args.clouds} ...")
        clouds = load_clouds(args.clouds)
        if args.max_clouds is not None:
            clouds = clouds[: args.max_clouds]
        print(f"  {len(clouds)} clouds loaded.")

        neural_model = None
        if "neural" in args.estimators:
            betti_path = args.betti or infer_betti_path(args.clouds)
            print(f"Loading Betti curves from {betti_path} and joining to clouds by seed ...")
            attach_betti_curves(clouds, betti_path, homology_dim=args.homology_dim)
            curve_key = f"betti{args.homology_dim}_curve"
            n_with_curve = sum(curve_key in c or (args.homology_dim == 0 and "betti0_curve" in c) for c in clouds)
            print(f"  {n_with_curve}/{len(clouds)} clouds matched a Betti curve.")

            print("Loading PH-NN model ...")
            neural_model = build_neural_model(args.nn_model, args.nn_results)

        mc_kwargs = {
            "r_low": args.r_low,
            "r_high": args.r_high,
            "n_t": args.n_t,
            "c": args.mc_c,
            "n_starts": args.mc_starts,
        }

        t0 = time.time()
        results = evaluate(
            clouds,
            args.estimators,
            mc_kwargs=mc_kwargs,
            mc_seed=args.mc_seed,
            neural_model=neural_model,
            homology_dim=args.homology_dim,
            ridge_log_width=args.ridge_log_width,
            bootstrap=args.bootstrap,
            bootstrap_seed=args.bootstrap_seed,
            include_tracebacks=args.tracebacks,
        )
        elapsed = time.time() - t0

        print(f"\nEvaluation completed in {elapsed:.1f}s")

    print_metrics(results["metrics"])

    if args.out and not use_cache:
        save_raw_results(results["raw"], args.out)
    if args.metrics_out:
        save_metrics(results["metrics"], args.metrics_out)


if __name__ == "__main__":
    main()