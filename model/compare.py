"""
compare.py
==========

Compare parameter estimates for the BCI Thomas process across multiple
estimators, one census at a time, and produce a summary table + plots.

Architecture
------------
Each estimator is a callable registered in ESTIMATORS:

    ESTIMATORS: dict[str, Callable[[Path], dict[str, float]]]

The callable receives a *_cloud.pkl path (or a *_betti0.pkl path, when
`input_key="betti0"` is set) and returns a dict whose keys are a subset of:

    parent_intensity, cluster_scale, mean_offspring

Two estimators ship by default:

  "mincontrast"  –  classical minimum-contrast on the K-function
                    (mincontrast.py).  Input: *_cloud.pkl.
  "neural"       –  TDA + neural network  (model.py / processing.py).
                    Input: *_betti0.pkl (must be pre-computed by processing.py).

Adding a new estimator
----------------------
    from compare import ESTIMATORS

    def my_estimator(cloud_path: Path) -> dict[str, float]:
        ...
        return {"parent_intensity": ..., "cluster_scale": ..., "mean_offspring": ...}

    ESTIMATORS["my_name"] = my_estimator

Or pass --estimators to select a subset from the CLI.

Usage
-----
    # run all estimators on all 8 BCI censuses
    python compare.py

    # only mincontrast, save table and plots to results/
    python compare.py --estimators mincontrast --out-dir results

    # custom data directory
    python compare.py --data-dir /path/to/barro
"""

from __future__ import annotations

import argparse
import pickle
import sys
import warnings
from pathlib import Path
from typing import Any, Callable

import numpy as np

# ── optional imports (graceful degradation) ───────────────────────────────────
try:
    import pandas as pd
    _PANDAS = True
except ImportError:
    _PANDAS = False

try:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    _MPL = True
except ImportError:
    _MPL = False

# ══════════════════════════════════════════════════════════════════════════════
# Estimator registry
# ══════════════════════════════════════════════════════════════════════════════

PARAMS = ["parent_intensity", "cluster_scale", "mean_offspring"]

# Each value: (callable, input_suffix)
#   callable    : Path -> dict[str, float]
#   input_suffix: which file to pass ('_cloud.pkl' or '_betti0.pkl')
_REGISTRY: dict[str, tuple[Callable[[Path], dict[str, float]], str]] = {}


def register(name: str, input_suffix: str = "_cloud.pkl"):
    """Decorator to register an estimator function."""
    def decorator(fn: Callable[[Path], dict[str, float]]):
        _REGISTRY[name] = (fn, input_suffix)
        return fn
    return decorator


# ── mincontrast ───────────────────────────────────────────────────────────────
try:
    import mincontrast as _mc

    @register("mincontrast", input_suffix="_cloud.pkl")
    def _run_mincontrast(cloud_path: Path) -> dict[str, float]:
        result = _mc.estimate(cloud_path)
        return {p: result[p] for p in PARAMS}

except ImportError:
    warnings.warn("mincontrast.py not found; 'mincontrast' estimator unavailable.")


# ── neural (TDA + model.py) ───────────────────────────────────────────────────
try:
    import model as _model_mod

    _neural_model: _model_mod.Model | None = None

    def _get_neural_model() -> _model_mod.Model:
        global _neural_model
        if _neural_model is None:
            _neural_model = _model_mod.Model()
        return _neural_model

    @register("neural", input_suffix="_betti0.pkl")
    def _run_neural(betti0_path: Path) -> dict[str, float]:
        betti_0 = _model_mod.load_betti_0(str(betti0_path))
        result = _get_neural_model().forward(betti_0=betti_0)
        if not isinstance(result, dict):
            # batch return — take first row
            result = {p: float(result[0, i]) for i, p in enumerate(PARAMS)}
        return {p: result[p] for p in PARAMS if p in result}

except ImportError:
    warnings.warn("model.py not importable; 'neural' estimator unavailable.")


# Public handle so callers can add their own without touching _REGISTRY directly
ESTIMATORS = _REGISTRY


# ══════════════════════════════════════════════════════════════════════════════
# Core comparison logic
# ══════════════════════════════════════════════════════════════════════════════

def run_all(
        data_dir: Path,
        estimator_names: list[str],
        n_censuses: int = 8,
) -> dict[str, dict[int, dict[str, float]]]:
    """
    Returns
    -------
    results[estimator_name][census_index] = {param: value, ...}
    """
    results: dict[str, dict[int, dict[str, float]]] = {
        name: {} for name in estimator_names
    }

    for name in estimator_names:
        fn, suffix = ESTIMATORS[name]
        print(f"\n── {name} ──")
        for census in range(1, n_censuses + 1):
            path = data_dir / f"bci.tree{census}{suffix}"
            if not path.exists():
                print(f"  census {census}: {path.name} not found — skipping")
                continue
            try:
                est = fn(path)
                results[name][census] = est
                parts = "  ".join(f"{p}={est[p]:.4g}" for p in PARAMS if p in est)
                print(f"  census {census}: {parts}")
            except Exception as exc:
                print(f"  census {census}: ERROR — {exc}")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# Output helpers
# ══════════════════════════════════════════════════════════════════════════════

def _census_years() -> dict[int, str]:
    return {
        1: "1981–83", 2: "1985", 3: "1990–92", 4: "1995–96",
        5: "2000–01", 6: "2005–06", 7: "2010–11", 8: "2013–16",
    }


def print_table(results: dict[str, dict[int, dict[str, float]]]) -> None:
    years = _census_years()
    estimator_names = list(results.keys())

    # header
    col_w = 14
    header_parts = [f"{'census':<10}", f"{'years':<10}"]
    for name in estimator_names:
        for p in PARAMS:
            label = f"{name[:6]}/{p[:8]}"
            header_parts.append(f"{label:>{col_w}}")
    print("\n" + "  ".join(header_parts))
    print("-" * (10 + 10 + (col_w + 2) * len(estimator_names) * len(PARAMS)))

    all_censuses = sorted({c for est in results.values() for c in est})
    for census in all_censuses:
        row = [f"{census:<10}", f"{years.get(census,'?'):<10}"]
        for name in estimator_names:
            for p in PARAMS:
                val = results[name].get(census, {}).get(p)
                row.append(f"{val:>{col_w}.4g}" if val is not None else f"{'N/A':>{col_w}}")
        print("  ".join(row))


def save_csv(results: dict[str, dict[int, dict[str, float]]], out_path: Path) -> None:
    if not _PANDAS:
        print("pandas not available — skipping CSV export")
        return
    years = _census_years()
    rows = []
    all_censuses = sorted({c for est in results.values() for c in est})
    for census in all_censuses:
        row: dict[str, Any] = {"census": census, "years": years.get(census, "?")}
        for name, est_results in results.items():
            for p in PARAMS:
                val = est_results.get(census, {}).get(p)
                row[f"{name}_{p}"] = val
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"Saved CSV → {out_path}")


def save_plots(results: dict[str, dict[int, dict[str, float]]], out_dir: Path) -> None:
    if not _MPL:
        print("matplotlib not available — skipping plots")
        return

    years = _census_years()
    estimator_names = list(results.keys())
    all_censuses = sorted({c for est in results.values() for c in est})
    x = np.array(all_censuses)
    x_labels = [years.get(c, str(c)) for c in all_censuses]

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_map = {name: colors[i % len(colors)] for i, name in enumerate(estimator_names)}

    param_labels = {
        "parent_intensity": r"$\hat{\kappa}$ (parent intensity)",
        "cluster_scale":    r"$\hat{\sigma}$ (cluster scale)",
        "mean_offspring":   r"$\hat{\mu}$ (mean offspring)",
    }

    fig, axes = plt.subplots(len(PARAMS), 1, figsize=(9, 3.5 * len(PARAMS)), sharex=True)
    if len(PARAMS) == 1:
        axes = [axes]

    for ax, param in zip(axes, PARAMS):
        for name in estimator_names:
            vals = [results[name].get(c, {}).get(param) for c in all_censuses]
            y = np.array([v if v is not None else np.nan for v in vals])
            ax.plot(x, y, marker="o", label=name, color=color_map[name], linewidth=1.8)

        ax.set_ylabel(param_labels.get(param, param), fontsize=11)
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.yaxis.get_major_formatter().set_scientific(False)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(x_labels, rotation=30, ha="right", fontsize=9)
    axes[-1].set_xlabel("Census", fontsize=11)

    fig.suptitle("Thomas process parameter estimates — BCI 50-ha plot", fontsize=13)
    fig.tight_layout()

    out_path = out_dir / "compare_estimates.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot → {out_path}")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--data-dir", type=Path, default=Path("data/barro"),
                   help="directory containing the *_cloud.pkl and *_betti0.pkl files")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="directory for CSV and PNG outputs (default: --data-dir)")
    p.add_argument("--estimators", nargs="+", default=None,
                   metavar="NAME",
                   help="estimators to run (default: all registered); "
                        f"available: {list(_REGISTRY)}")
    p.add_argument("--censuses", type=int, default=8,
                   help="number of censuses to process (default: 8)")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    out_dir = args.out_dir or args.data_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    available = list(_REGISTRY)
    if not available:
        sys.exit("No estimators registered. Check that mincontrast.py / model.py are importable.")

    selected = args.estimators or available
    unknown = [n for n in selected if n not in _REGISTRY]
    if unknown:
        sys.exit(f"Unknown estimator(s): {unknown}. Available: {available}")

    results = run_all(args.data_dir, selected, n_censuses=args.censuses)
    print_table(results)
    save_csv(results, out_dir / "compare_estimates.csv")
    save_plots(results, out_dir)


if __name__ == "__main__":
    main()