#!/usr/bin/env python3
# scripts/eval_loglog_stats_baseline.py
"""Track A baseline: fit an actual regressor -- not just eyeball marginal
Pearson r -- from the nine log(log(death/birth)) summary statistics per
(homology dim, k) explored in
notebooks/loglog_death_birth_vs_params_nested_thomas.ipynb, and report
held-out accuracy on the SAME population/split/label-normalization
pi_multik uses, so the numbers are directly comparable to
results/<process>/dtm_k.../pi_multik/seed_<seed>/results.json's
test_loss_per_target.

Why this exists: the notebook's correlation heatmap is 9 stats x 5 params
x 3 k's x 2 dims = 270 marginal Pearson r's -- useful for spotting which
stats/params are worth pursuing, but (a) marginal correlation isn't
predictive accuracy (it ignores interactions between the 54 stats, and
looking at 270 numbers for the few big ones is a multiple-comparisons trap),
and (b) it has no train/test split, so "correlation" here includes rows a
real estimator would never get to see labels for. This script closes both
gaps: a proper multivariate fit, evaluated only on held-out rows.

Data plumbing deliberately reuses the exact functions pi_multik itself
uses, rather than re-deriving them, so the comparison is apples-to-apples:
  - cloudforger.experiments.pi_multik.pi_multik.load_multik_split for the
    seed-aligned-across-k population (identical to what pi_multik trains
    on -- some clouds get dropped per k by the empty-diagram filter, and
    the intersection is what both this baseline and pi_multik actually see).
  - cloudforger.core.splits.train_val_test_indices(n, seed) for the split
    -- same seed value as a results/.../seed_<seed>/ directory reproduces
    that run's exact train/val/test partition.
  - cloudforger.baselines.vihrs.{fit_log_zscore,apply_log_zscore} for the
    label normalization -- fit on TRAIN rows only, matching
    experiments/common.py's fit_label_norm with log_label_names set to
    every param (see configs/runs/nested_thomas/pi_multik.yaml). Models
    are fit and scored in this same standardized log space, so
    test_z_mse below sits on the same scale as pi_multik's
    test_loss_per_target -- no unit conversion needed to compare them.

Two model families, both plain numpy/scipy (this repo has no scikit-learn
dependency today -- swap in sklearn.linear_model.RidgeCV /
sklearn.neighbors.KNeighborsRegressor / HistGradientBoostingRegressor here
if that changes; the closed-form/brute-force versions below are exact
equivalents for what this script needs, just without the extra dependency):
  - ridge: closed-form L2-regularized linear regression on standardized,
    median-imputed features (NaN where a diagram had too few finite pairs
    in that dim -- see loglog_stats_for_diagram). Alpha chosen per-target
    per-seed by held-out val z-MSE.
  - knn: inverse-distance-weighted k-nearest-neighbors in the same
    standardized feature space -- a cheap nonlinear ceiling, catching any
    signal a linear model misses. k chosen the same way as ridge's alpha.

Usage:
    python scripts/eval_loglog_stats_baseline.py
    python scripts/eval_loglog_stats_baseline.py --process nested_thomas --k-values 5 10 15
    python scripts/eval_loglog_stats_baseline.py --out notebooks/out/loglog_stats_baseline.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import skew, kurtosis

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cloudforger.baselines.vihrs import apply_log_zscore, fit_log_zscore, invert_log_zscore
from cloudforger.core.splits import train_val_test_indices
from cloudforger.experiments.pi_multik.pi_multik import load_multik_split

DEFAULT_PARAMS = ["parent_intensity", "meta_offspring", "meta_cluster_scale", "mean_offspring", "cluster_scale"]
DEFAULT_K_VALUES = [5, 10, 15]
DEFAULT_SEEDS = [9371, 9372, 9373, 9374, 9375, 9376, 9377, 9378, 9379, 9380]

RIDGE_ALPHAS = np.logspace(-2, 3, 12)
KNN_KS = [5, 10, 20, 40, 80]

# Same nine statistics, same transform, as the notebook -- kept in sync
# deliberately rather than imported, since the notebook computes them over
# a 600-diagram sample per k (for fast plotting) while this script needs
# them over the FULL population (for a real train/val/test split).
STATS = {
    "mean": np.mean,
    "median": np.median,
    "min": np.min,
    "max": np.max,
    "std": np.std,
    "variance": np.var,
    "iqr": lambda a: np.percentile(a, 75) - np.percentile(a, 25),
    "skew": skew,
    "kurtosis": kurtosis,
}


# ---------------------------------------------------------------------------
# Features: log(log(death/birth)) summary stats, per diagram per dim per k
# ---------------------------------------------------------------------------


def loglog_stats_for_diagram(diagram, dim: int) -> dict[str, float]:
    """NaN (not 0) when a diagram has no finite pairs in this dim -- an
    absent homology class is missing information, not a zero value. The
    preprocessor below imputes with the train-set median; the alternative
    of silently zero-filling would tell the model "this diagram's H1 stats
    are identically the dataset's most extreme value," which is wrong."""
    pairs = diagram.finite_pairs(dim)
    if len(pairs) == 0:
        return {name: np.nan for name in STATS}
    ratio = pairs[:, 1] / pairs[:, 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        loglog = np.log(np.log(ratio))
    loglog = loglog[np.isfinite(loglog)]
    if len(loglog) == 0:
        return {name: np.nan for name in STATS}
    out = {}
    with np.errstate(invalid="ignore"):  # skew/kurtosis of a 1-pair diagram is 0/0 -> nan by design
        for name, fn in STATS.items():
            out[name] = float(fn(loglog))
    return out


def build_feature_matrix(
    diagrams_per_k: dict[int, list], k_values: list[int], homology_dims: tuple[int, ...]
) -> tuple[np.ndarray, list[str]]:
    n = len(diagrams_per_k[k_values[0]])
    columns: list[str] = []
    data: dict[str, np.ndarray] = {}
    for k in k_values:
        diagrams = diagrams_per_k[k]
        assert len(diagrams) == n, f"k={k} has {len(diagrams)} diagrams, expected {n} -- alignment bug upstream."
        for dim in homology_dims:
            per_diagram = [loglog_stats_for_diagram(d, dim) for d in diagrams]
            for stat_name in STATS:
                col = f"{stat_name}_loglog_dim{dim}_k{k}"
                columns.append(col)
                data[col] = np.array([row[stat_name] for row in per_diagram], dtype=np.float64)
    X = np.column_stack([data[c] for c in columns])
    return X, columns


# ---------------------------------------------------------------------------
# Preprocessing: train-fit median-impute + standardize, applied frozen
# ---------------------------------------------------------------------------


def fit_preprocessor(X_train: np.ndarray) -> dict[str, np.ndarray]:
    median = np.nanmedian(X_train, axis=0)
    imputed = np.where(np.isnan(X_train), median, X_train)
    mean = imputed.mean(axis=0)
    std = imputed.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return {"median": median, "mean": mean, "std": std}


def apply_preprocessor(X: np.ndarray, prep: dict[str, np.ndarray]) -> np.ndarray:
    imputed = np.where(np.isnan(X), prep["median"], X)
    return (imputed - prep["mean"]) / prep["std"]


# ---------------------------------------------------------------------------
# Models: closed-form ridge, brute-force weighted KNN (numpy-only, no
# scikit-learn dependency -- see module docstring)
# ---------------------------------------------------------------------------


def _ridge_fit(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    # X, y are already train-standardized/centered -- no explicit intercept term.
    n_features = X.shape[1]
    return np.linalg.solve(X.T @ X + alpha * np.eye(n_features), X.T @ y)


def fit_predict_ridge(X_train, y_train, X_val, y_val, X_test):
    best_val_mse, best_alpha, best_w = None, None, None
    for alpha in RIDGE_ALPHAS:
        w = _ridge_fit(X_train, y_train, alpha)
        val_mse = float(np.mean((X_val @ w - y_val) ** 2))
        if best_val_mse is None or val_mse < best_val_mse:
            best_val_mse, best_alpha, best_w = val_mse, alpha, w
    return X_val @ best_w, X_test @ best_w, {"alpha": float(best_alpha)}


def _pairwise_sqdist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return np.clip(
        (A**2).sum(axis=1, keepdims=True) - 2 * A @ B.T + (B**2).sum(axis=1)[None, :], 0, None
    )


def _knn_predict_from_dist(d2: np.ndarray, y_train: np.ndarray, k: int) -> np.ndarray:
    k = min(k, d2.shape[1])
    nn_idx = np.argpartition(d2, kth=k - 1, axis=1)[:, :k]
    preds = np.empty(d2.shape[0], dtype=np.float64)
    for i in range(d2.shape[0]):
        idx = nn_idx[i]
        dist = np.sqrt(d2[i, idx])
        weights = 1.0 / (dist + 1e-6)
        preds[i] = np.average(y_train[idx], weights=weights)
    return preds


def fit_predict_knn(X_train, y_train, X_val, y_val, X_test):
    d2_val = _pairwise_sqdist(X_val, X_train)
    best_val_mse, best_k = None, None
    for k in KNN_KS:
        val_mse = float(np.mean((_knn_predict_from_dist(d2_val, y_train, k) - y_val) ** 2))
        if best_val_mse is None or val_mse < best_val_mse:
            best_val_mse, best_k = val_mse, k
    d2_test = _pairwise_sqdist(X_test, X_train)
    pred_val = _knn_predict_from_dist(d2_val, y_train, best_k)
    pred_test = _knn_predict_from_dist(d2_test, y_train, best_k)
    return pred_val, pred_test, {"k": int(best_k)}


MODEL_FNS = {"ridge": fit_predict_ridge, "knn": fit_predict_knn}


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    if ss_tot < 1e-12:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


# ---------------------------------------------------------------------------
# Per-seed fit/eval, matching pi_multik's split + label normalization
# ---------------------------------------------------------------------------


def fit_and_eval_one_seed(X: np.ndarray, y_raw: np.ndarray, param_names: list[str], seed: int) -> dict:
    n = len(y_raw)
    train_idx, val_idx, test_idx = train_val_test_indices(n, seed)  # same default fractions pi_multik uses

    prep = fit_preprocessor(X[train_idx])
    Xp = apply_preprocessor(X, prep)

    label_norm = fit_log_zscore(y_raw[train_idx])  # fit on TRAIN targets only, mirrors experiments/common.py
    z = apply_log_zscore(y_raw, label_norm)

    out = {m: {"val_z_mse": {}, "test_z_mse": {}, "test_r2": {}, "test_mae": {}, "chosen": {}} for m in MODEL_FNS}

    for j, name in enumerate(param_names):
        for model_name, fit_predict in MODEL_FNS.items():
            pred_val, pred_test, chosen = fit_predict(
                Xp[train_idx], z[train_idx, j], Xp[val_idx], z[val_idx, j], Xp[test_idx]
            )
            out[model_name]["val_z_mse"][name] = float(np.mean((pred_val - z[val_idx, j]) ** 2))
            out[model_name]["test_z_mse"][name] = float(np.mean((pred_test - z[test_idx, j]) ** 2))
            out[model_name]["chosen"][name] = chosen

            norm_j = {"mean": label_norm["mean"][j], "std": label_norm["std"][j]}
            pred_test_raw = invert_log_zscore(pred_test, norm_j)
            out[model_name]["test_r2"][name] = _r2(y_raw[test_idx, j], pred_test_raw)
            out[model_name]["test_mae"][name] = float(np.mean(np.abs(y_raw[test_idx, j] - pred_test_raw)))

    return out


# ---------------------------------------------------------------------------
# Comparison against pi_multik's own results.json (best-effort)
# ---------------------------------------------------------------------------


def load_pi_multik_test_loss(results_dir: Path, seeds: list[int]) -> dict[int, dict[str, float]]:
    out = {}
    for seed in seeds:
        path = results_dir / f"seed_{seed}" / "results.json"
        if not path.exists():
            continue
        with open(path) as f:
            payload = json.load(f)
        per_target = payload.get("test_loss_per_target")
        if per_target:
            out[seed] = per_target
    return out


def _agg(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float("nan"), float("nan")
    return float(arr.mean()), float(arr.std())


def print_report(per_seed: dict[int, dict], cnn_losses: dict[int, dict], param_names: list[str]) -> None:
    seeds = sorted(per_seed)
    model_names = list(MODEL_FNS)

    def _table(title: str, extract, model_names_: list[str], show_cnn: bool) -> None:
        print(f"\n{title}")
        print("-" * len(title))
        header = f"{'param':>20}" + "".join(f"{m:>16}" for m in model_names_)
        if show_cnn:
            header += f"{'pi_multik CNN':>16}"
        print(header)
        for name in param_names:
            row = f"{name:>20}"
            for m in model_names_:
                mean, std = _agg([per_seed[s][m][extract][name] for s in seeds])
                row += f"{mean:>10.3f}±{std:<5.3f}"
            if show_cnn:
                cnn_vals = [cnn_losses[s][name] for s in seeds if s in cnn_losses and name in cnn_losses[s]]
                if cnn_vals:
                    mean, std = _agg(cnn_vals)
                    row += f"{mean:>10.3f}±{std:<5.3f}"
                else:
                    row += f"{'n/a':>16}"
            print(row)

    print("\n" + "=" * 100)
    print(f"Aggregated over {len(seeds)} seeds: {seeds}")
    print("=" * 100)
    _table(
        "Test z-log MSE (standardized log-param space -- same scale as pi_multik's test_loss_per_target)",
        "test_z_mse", model_names, show_cnn=True,
    )
    _table("Test R2 (raw parameter units, inverted from z-log space)", "test_r2", model_names, show_cnn=False)
    _table("Test MAE (raw parameter units)", "test_mae", model_names, show_cnn=False)


# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--process", default="nested_thomas")
    parser.add_argument("--k-values", type=int, nargs="+", default=DEFAULT_K_VALUES)
    parser.add_argument("--params", nargs="+", default=DEFAULT_PARAMS)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--homology-dims", type=int, nargs="+", default=[0, 1])
    parser.add_argument(
        "--pi-multik-results", type=Path, default=None,
        help="results/<process>/dtm_k.../pi_multik dir to compare against "
             "(default: derived from --process/--k-values).",
    )
    parser.add_argument("--out", type=Path, default=None, help="where to write the JSON report")
    args = parser.parse_args(argv)

    data_dir = ROOT / "data" / args.process
    diagram_paths = [data_dir / f"dtm_k{k}" / "diagrams.pkl" for k in args.k_values]
    clouds_path = data_dir / "clouds.pkl"

    print(f"[load] aligning diagrams across k={args.k_values} for process={args.process!r} ...")
    split = load_multik_split(
        args.k_values, diagram_paths, clouds_path, tuple(args.params),
        tag=args.process, homology_dims=tuple(args.homology_dims),
    )
    if split is None:
        raise FileNotFoundError(f"one or more of {diagram_paths} is missing -- run scripts/featurize.py first.")

    n = len(split["targets"])
    print(f"[load] {n} clouds common to all k -- same population pi_multik trains/evaluates on.")

    print("[features] computing log(log(death/birth)) summary stats ...")
    t0 = time.time()
    X, columns = build_feature_matrix(split["diagrams_per_k"], args.k_values, tuple(args.homology_dims))
    n_nan_cols = int(np.isnan(X).any(axis=0).sum())
    print(
        f"[features] {X.shape[1]} columns built in {time.time() - t0:.1f}s "
        f"({n_nan_cols} contain at least one NaN row -- median-imputed before fitting)."
    )

    y_raw = split["targets"]

    if args.pi_multik_results is None:
        k_tag = "+".join(str(k) for k in sorted(args.k_values))
        args.pi_multik_results = ROOT / "results" / args.process / f"dtm_k{k_tag}" / "pi_multik"
    cnn_losses = load_pi_multik_test_loss(args.pi_multik_results, args.seeds)
    if cnn_losses:
        print(f"[compare] found pi_multik results.json for {len(cnn_losses)}/{len(args.seeds)} seeds "
              f"under {args.pi_multik_results}")
    else:
        print(f"[compare] no pi_multik results.json found under {args.pi_multik_results} -- baseline-only report.")

    per_seed = {}
    for seed in args.seeds:
        t0 = time.time()
        per_seed[seed] = fit_and_eval_one_seed(X, y_raw, args.params, seed)
        print(f"[fit] seed={seed} done in {time.time() - t0:.1f}s")

    print_report(per_seed, cnn_losses, args.params)

    if args.out is None:
        args.out = ROOT / "notebooks" / "out" / f"loglog_stats_baseline_{args.process}.json"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(
            {
                "process": args.process,
                "k_values": args.k_values,
                "homology_dims": args.homology_dims,
                "n_common_clouds": n,
                "feature_columns": columns,
                "per_seed": per_seed,
                "pi_multik_test_loss_per_target": cnn_losses,
            },
            f, indent=2,
        )
    print(f"\n[done] report -> {args.out}")


if __name__ == "__main__":
    main()
