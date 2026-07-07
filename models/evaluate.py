# models/evaluate.py

from __future__ import annotations

import argparse
import pickle
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np

# ── estimator imports ─────────────────────────────────────────────────────────
try:
    import mincontrast as mc
    _MC = True
except ImportError:
    _MC = False
    warnings.warn("mincontrast.py not found — 'mincontrast' estimator unavailable.")

try:
    import phnn
    _PHNN = True
except ImportError:
    _PHNN = False
    warnings.warn("phnn.py not importable — 'neural' estimator unavailable.")

try:
    import pandas as pd
    _PANDAS = True
except ImportError:
    _PANDAS = False


# ══════════════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════════════

PARAMS = ["parent_intensity", "cluster_scale", "mean_offspring"]

# The adversarial split is the only dataset guaranteed unseen by a trained
# neural estimator — see the module docstring's Design notes.
DEFAULT_CLOUDS = Path("data/params/2d/thomas/adversarial_clouds.pkl")

REGIMES = {
    "strong_tight": {"parent_intensity": "HIGH", "mean_offspring": "HIGH", "cluster_scale": "LOW"},
    "strong_diffuse": {"parent_intensity": "HIGH", "mean_offspring": "HIGH", "cluster_scale": "HIGH"},
    "weak": {"parent_intensity": "LOW",  "mean_offspring": "LOW",  "cluster_scale": "MID"},
    "ridge": None,  # special handling: fixed kappa*omega product
}


# ══════════════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════════════

def load_clouds(path: Path) -> list[dict[str, Any]]:
    """Load the clouds pickle — list of dicts with 'points' and 'params'."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    if not isinstance(data, list):
        data = [data]
    return data


def is_unit_square(cloud: dict) -> bool:
    """Check if the cloud region is [0,1]^2 (or absent, implying unit square)."""
    reg = cloud.get("region", {})
    low = np.asarray(reg.get("low", [0.0, 0.0]))
    high = np.asarray(reg.get("high", [1.0, 1.0]))
    return np.allclose(low, 0.0) and np.allclose(high, 1.0)


def infer_betti_path(clouds_path: Path) -> Path:
    """Infer the sibling betti.pkl for a clouds.pkl, e.g.

    data/params/2d/thomas/clouds.pkl -> data/params/2d/thomas/betti.pkl
    data/params/2d/thomas/adversarial_clouds.pkl -> .../adversarial_betti.pkl
    """
    name = clouds_path.name
    if "clouds" not in name:
        raise ValueError(f"Cannot infer betti path from {clouds_path} (no 'clouds' in filename).")
    return clouds_path.with_name(name.replace("clouds", "betti", 1))


def load_betti_curves(path: Path, homology_dim: int = 0) -> tuple[dict[Any, np.ndarray], str]:
    """Load betti.pkl and return ({seed: betti_k_curve}, process) for the given homology dim.

    betti.pkl is produced by pipeline_lib/betti.py as a dict with parallel
    'seeds' and 'betti{k}_matrix' arrays — betti_matrix[i] is the curve for
    seeds[i], in the same order the clouds were originally generated in.
    """
    with open(path, "rb") as f:
        payload = pickle.load(f)
    seeds = payload["seeds"]
    matrix_key = f"betti{homology_dim}_matrix"
    if matrix_key not in payload:
        raise KeyError(f"{path} has no {matrix_key!r}; available homology dims were not computed with dim={homology_dim}.")
    matrix = payload[matrix_key]
    if len(seeds) != len(matrix):
        raise ValueError(f"{path}: {len(seeds)} seeds but {len(matrix)} curves — corrupt betti.pkl.")

    by_seed: dict[Any, np.ndarray] = {}
    for seed, curve in zip(seeds, matrix):
        if seed in by_seed:
            raise ValueError(f"{path}: duplicate seed {seed!r} in betti.pkl — cannot uniquely join to clouds.")
        by_seed[seed] = np.asarray(curve)
    return by_seed, payload.get("process", "")


def attach_betti_curves(clouds: list[dict], betti_path: Path, homology_dim: int = 0) -> None:
    """Join betti curves onto clouds in-place by seed, setting 'betti0_curve'.

    Every cloud is expected to carry a unique 'seed' (set by cloud_to_record
    in the generation pipeline). Clouds whose seed has no match in betti.pkl
    are left without a 'betti0_curve' and will be skipped by run_neural.

    'seed' is only unique *within* one pipeline run (every process/split
    restarts numbering from the same base_seed), so it cannot by itself catch
    e.g. a thomas clouds.pkl accidentally paired with a matern betti.pkl —
    both use seeds 0..N-1. The 'process' field recorded in both clouds.pkl
    and betti.pkl is checked as well to guard against that case.
    """
    by_seed, betti_process = load_betti_curves(betti_path, homology_dim=homology_dim)

    cloud_processes = {c.get("process") for c in clouds if c.get("process")}
    if betti_process and cloud_processes and cloud_processes != {betti_process}:
        raise ValueError(
            f"Process mismatch: clouds are {sorted(cloud_processes)} but {betti_path} "
            f"was computed for process={betti_process!r}. Refusing to join — 'seed' "
            "alone is not unique across processes, so this would silently pair clouds "
            "with the wrong curves. Pass the correct --betti path."
        )

    seeds_seen: set[Any] = set()
    n_matched = 0
    for cloud in clouds:
        seed = cloud.get("seed")
        if seed is None:
            raise ValueError("Cloud record is missing 'seed' — cannot join betti curves to clouds.")
        if seed in seeds_seen:
            raise ValueError(f"Duplicate seed {seed!r} among clouds — cannot uniquely join betti curves.")
        seeds_seen.add(seed)

        curve = by_seed.get(seed)
        if curve is not None:
            cloud["betti0_curve"] = curve
            n_matched += 1

    if n_matched < len(clouds):
        warnings.warn(
            f"Only {n_matched}/{len(clouds)} clouds matched a betti curve by seed "
            f"(betti.pkl: {betti_path}). Unmatched clouds will be skipped by the "
            "neural estimator — check that --clouds and --betti come from the same "
            "pipeline run/split."
        )


# ══════════════════════════════════════════════════════════════════════════════
# Regime assignment
# ══════════════════════════════════════════════════════════════════════════════

def assign_terciles(values: np.ndarray) -> np.ndarray:
    """Assign LOW / MID / HIGH tercile labels to an array of values."""
    t1, t2 = np.percentile(values, [33.3, 66.7])
    labels = np.where(values <= t1, "LOW",
             np.where(values <= t2, "MID", "HIGH"))
    return labels


def assign_regimes(clouds: list[dict]) -> np.ndarray:
    """Return a string array of regime labels, one per cloud."""
    params_arr = {p: np.array([c["params"][p] for c in clouds]) for p in PARAMS}
    terciles = {p: assign_terciles(np.log(params_arr[p])) for p in PARAMS}

    regimes = np.full(len(clouds), "other", dtype=object)
    for regime_name, criteria in REGIMES.items():
        if criteria is None:
            continue
        mask = np.ones(len(clouds), dtype=bool)
        for param, level in criteria.items():
            mask &= (terciles[param] == level)
        regimes[mask] = regime_name

    return regimes


# ══════════════════════════════════════════════════════════════════════════════
# Estimator runners
# ══════════════════════════════════════════════════════════════════════════════

def run_mincontrast(cloud: dict, **kwargs) -> dict[str, float] | None:
    """Run mincontrast.fit() on a cloud's points.  Returns None on failure."""
    points = np.asarray(cloud["points"], dtype=float)
    if len(points) < 5:
        return None
    if not is_unit_square(cloud):
        reg = cloud.get("region", {})
        low = np.asarray(reg.get("low", [0.0, 0.0]))
        high = np.asarray(reg.get("high", [1.0, 1.0]))
        points = mc.crop_and_rescale(points, low, high)
    try:
        result = mc.fit_multistart(points, **kwargs)
        return {p: result[p] for p in PARAMS}
    except Exception:
        return None


def run_neural(cloud: dict, model=None) -> dict[str, float] | None:
    """Run PH-NN on a cloud's pre-computed betti0_curve.  Returns None if missing."""
    curve = cloud.get("betti0_curve")
    if curve is None:
        return None
    try:
        import torch
        t = torch.as_tensor(np.asarray(curve, dtype=np.float32))
        result = model.forward(betti_0=t)
        if not isinstance(result, dict):
            result = {p: float(result[0, i]) for i, p in enumerate(PARAMS)}
        return {p: result[p] for p in PARAMS if p in result}
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# Metrics
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(
    truth: np.ndarray,
    estimates: np.ndarray,
) -> dict[str, float]:
    """Compute bias, RMSE, and relative RMSE for one parameter.

    Both arrays are 1D, same length, containing the true and estimated values
    for one parameter across multiple clouds.  NaN entries are dropped.
    """
    mask = np.isfinite(truth) & np.isfinite(estimates)
    if mask.sum() < 2:
        return {"n": 0, "bias": float("nan"), "rmse": float("nan"),
                "rel_rmse": float("nan"), "median_ratio": float("nan")}

    t, e = truth[mask], estimates[mask]
    errors = e - t
    ratios = e / np.clip(t, 1e-12, None)

    return {
        "n":            int(mask.sum()),
        "bias":         float(np.mean(errors)),
        "rmse":         float(np.sqrt(np.mean(errors ** 2))),
        "rel_rmse":     float(np.sqrt(np.mean((errors / np.clip(t, 1e-12, None)) ** 2))),
        "median_ratio": float(np.median(ratios)),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Main evaluation loop
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(
    clouds: list[dict],
    estimator_names: list[str],
    mc_kwargs: dict | None = None,
) -> dict[str, Any]:
    """Run estimators on all clouds, return structured results.

    Returns
    -------
    {
        "raw": list of per-cloud dicts  (truth + estimates),
        "metrics": {regime: {estimator: {param: {bias, rmse, ...}}}},
        "regimes": np.ndarray of regime labels,
    }
    """
    mc_kwargs = mc_kwargs or {}
    regimes = assign_regimes(clouds)

    # Pre-load neural model once if needed
    neural_model = None
    if "neural" in estimator_names and _PHNN:
        neural_model = phnn.Model()

    # ── per-cloud estimation ──────────────────────────────────────────────────
    raw_rows: list[dict] = []
    n = len(clouds)

    for i, cloud in enumerate(clouds):
        if (i + 1) % 100 == 0 or i == 0:
            print(f"  cloud {i+1}/{n} ...", flush=True)

        row: dict[str, Any] = {
            "index": i,
            "regime": regimes[i],
            "n_points": len(cloud["points"]),
        }
        for p in PARAMS:
            row[f"true_{p}"] = cloud["params"][p]

        if "mincontrast" in estimator_names and _MC:
            est = run_mincontrast(cloud, **mc_kwargs)
            for p in PARAMS:
                row[f"mc_{p}"] = est[p] if est else float("nan")

        if "neural" in estimator_names and _PHNN:
            est = run_neural(cloud, model=neural_model)
            for p in PARAMS:
                row[f"nn_{p}"] = est[p] if est else float("nan")

        raw_rows.append(row)

    # ── aggregate metrics ─────────────────────────────────────────────────────
    regime_labels = sorted(set(regimes)) + ["ALL"]
    est_prefixes = []
    if "mincontrast" in estimator_names and _MC:
        est_prefixes.append(("mincontrast", "mc"))
    if "neural" in estimator_names and _PHNN:
        est_prefixes.append(("neural", "nn"))

    metrics: dict[str, dict[str, dict[str, dict[str, float]]]] = {}

    for regime in regime_labels:
        if regime == "ALL":
            mask = np.ones(len(raw_rows), dtype=bool)
        else:
            mask = np.array([r["regime"] == regime for r in raw_rows])

        if mask.sum() == 0:
            continue

        metrics[regime] = {}
        for est_name, prefix in est_prefixes:
            metrics[regime][est_name] = {}
            for p in PARAMS:
                truth = np.array([raw_rows[j][f"true_{p}"] for j in range(len(raw_rows)) if mask[j]])
                ests  = np.array([raw_rows[j][f"{prefix}_{p}"] for j in range(len(raw_rows)) if mask[j]])
                metrics[regime][est_name][p] = compute_metrics(truth, ests)

    return {"raw": raw_rows, "metrics": metrics, "regimes": regimes}


# ══════════════════════════════════════════════════════════════════════════════
# Output
# ══════════════════════════════════════════════════════════════════════════════

def print_metrics(metrics: dict) -> None:
    """Print a formatted metrics table to stdout."""
    for regime, by_est in sorted(metrics.items()):
        print(f"\n{'═' * 70}")
        print(f"  Regime: {regime}")
        print(f"{'═' * 70}")
        header = f"  {'estimator':<14}  {'param':<20}  {'n':>5}  {'bias':>10}  {'rmse':>10}  {'rel_rmse':>10}  {'med_ratio':>10}"
        print(header)
        print(f"  {'-' * (len(header) - 2)}")

        for est_name, by_param in sorted(by_est.items()):
            for p, m in by_param.items():
                print(f"  {est_name:<14}  {p:<20}  {m['n']:>5d}  {m['bias']:>10.4f}  "
                      f"{m['rmse']:>10.4f}  {m['rel_rmse']:>10.4f}  {m['median_ratio']:>10.4f}")


def save_results(raw_rows: list[dict], out_path: Path) -> None:
    """Save per-cloud results to CSV (requires pandas)."""
    if not _PANDAS:
        print("pandas not available — skipping CSV export.")
        return
    df = pd.DataFrame(raw_rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved per-cloud results → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--clouds", type=Path, default=DEFAULT_CLOUDS,
                   help="path to clouds.pkl (list of cloud dicts with 'points', 'params', 'seed'). "
                        f"Default: {DEFAULT_CLOUDS} (the adversarial/held-out split — see module "
                        "docstring for why train_test/clouds.pkl is not the default).")
    p.add_argument("--betti", type=Path, default=None,
                   help="path to betti.pkl for the 'neural' estimator (default: inferred "
                        "sibling of --clouds, e.g. clouds.pkl -> betti.pkl). Curves are "
                        "joined to clouds by 'seed'.")
    p.add_argument("--estimators", nargs="+", default=["mincontrast", "neural"],
                   help="estimators to run (default: mincontrast neural)")
    p.add_argument("--max-clouds", type=int, default=None,
                   help="limit number of clouds (for quick testing)")
    p.add_argument("--out", type=Path, default=None,
                   help="path for per-cloud CSV output")
    p.add_argument("--r-high", type=float, default=0.25,
                   help="mincontrast upper K integration limit (default: 0.25)")
    p.add_argument("--mc-c", type=float, default=0.25,
                   help="mincontrast contrast exponent (default: 0.25)")
    return p


def main() -> None:
    args = _build_parser().parse_args()

    if args.clouds == DEFAULT_CLOUDS:
        print(f"No --clouds given; defaulting to the adversarial (held-out) split: {DEFAULT_CLOUDS}")
    print(f"Loading clouds from {args.clouds} ...")
    clouds = load_clouds(args.clouds)
    if args.max_clouds:
        clouds = clouds[:args.max_clouds]
    print(f"  {len(clouds)} clouds loaded.")

    if "neural" in args.estimators:
        betti_path = args.betti or infer_betti_path(args.clouds)
        print(f"Loading betti curves from {betti_path} and joining to clouds by seed ...")
        attach_betti_curves(clouds, betti_path)
        n_with_curve = sum("betti0_curve" in c for c in clouds)
        print(f"  {n_with_curve}/{len(clouds)} clouds matched a betti curve.")

    mc_kwargs = {"r_high": args.r_high, "c": args.mc_c}

    t0 = time.time()
    results = evaluate(clouds, args.estimators, mc_kwargs=mc_kwargs)
    elapsed = time.time() - t0
    print(f"\nEvaluation completed in {elapsed:.1f}s")

    print_metrics(results["metrics"])

    if args.out:
        save_results(results["raw"], args.out)


if __name__ == "__main__":
    main()