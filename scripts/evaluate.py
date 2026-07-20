#!/usr/bin/env python3
# scripts/evaluate.py
"""Aggregate saved results.pt across seeds for one or more methods (any mix
of nn.experiments methods, vihrs, or classical baselines -- they all wrote
the identical schema, see cloudforger.nn.experiments.common.save_results)
and compare them: summary statistics, a paired Wilcoxon test against the
best method, training-dynamics/overfitting diagnostics, and comparison
plots. Generalizes models/evaluate_seeds.py (previously hardcoded to
thomas's 3 params under a single models-root) to any process/label set.

Usage:
    python scripts/evaluate.py configs/runs/thomas_dtm_k5_betti_cnn.yaml \\
        --methods betti_cnn_01 vihrs mincontrast pi_multik_fusion
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cloudforger.config import RunConfig, load_config
from cloudforger.filtration import REGISTRY as FILTRATION_REGISTRY
from cloudforger.paths import DEFAULT_RESULTS_ROOT, ResultsPaths

# Methods whose results always live under the "raw" filtration tag,
# regardless of what's configured under `filtration:` -- mirrors
# scripts/train.py's FILTRATION_INDEPENDENT_FILE_KEYS + baseline dispatch.
RAW_TAG_METHODS = {"raw_pc", "pairwise", "vihrs", "vihrs_checkpointed", "mincontrast", "palm"}

PALETTE = [
    "#2a78d6", "#1baf7a", "#eda100", "#008300",
    "#4a3aa7", "#e34948", "#e87ba4", "#eb6834",
]
GRID_COLOR = "#e1e0d9"
SURFACE_COLOR = "#fcfcfb"


# ══════════════════════════════════════════════════════════════════════════════
# Loading
# ══════════════════════════════════════════════════════════════════════════════

def result_path(results_paths: ResultsPaths, filtrations: list, method: str, seed: int) -> Path:
    tag_filtrations = [] if method in RAW_TAG_METHODS else filtrations
    return results_paths.seed_dir(tag_filtrations, method, seed) / "results.pt"


def load_seed_result(results_paths: ResultsPaths, filtrations: list, method: str, seed: int) -> dict[str, Any] | None:
    path = result_path(results_paths, filtrations, method, seed)
    if not path.exists():
        return None
    return torch.load(path, map_location="cpu", weights_only=False)


def collect_method_results(
    results_paths: ResultsPaths, filtrations: list, methods: list[str], seeds: list[int]
) -> dict[str, dict[int, dict[str, Any]]]:
    results: dict[str, dict[int, dict[str, Any]]] = {}
    for method in methods:
        by_seed: dict[int, dict[str, Any]] = {}
        for seed in seeds:
            result = load_seed_result(results_paths, filtrations, method, seed)
            if result is None:
                print(f"  ! missing results.pt for method={method!r} seed={seed}; skipping")
                continue
            by_seed[seed] = result
        results[method] = by_seed
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Statistics
# ══════════════════════════════════════════════════════════════════════════════

def _metric_values(by_seed: dict[int, dict[str, Any]], metric: str) -> dict[int, float]:
    out = {}
    for seed, result in by_seed.items():
        value = result.get(metric)
        if value is not None and np.isfinite(value):
            out[seed] = float(value)
    return out


def _summary_stats(values: np.ndarray) -> dict[str, float]:
    n = len(values)
    if n == 0:
        return {k: float("nan") for k in
                ("n", "mean", "std", "median", "min", "max", "ci95_low", "ci95_high")} | {"n": 0}
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if n > 1 else 0.0
    half_width = float(stats.t.ppf(0.975, n - 1) * std / np.sqrt(n)) if n > 1 else float("nan")
    return {
        "n": n, "mean": mean, "std": std, "median": float(np.median(values)),
        "min": float(np.min(values)), "max": float(np.max(values)),
        "ci95_low": mean - half_width, "ci95_high": mean + half_width,
    }


def method_loss_summary(results: dict[str, dict[int, dict[str, Any]]], metric: str) -> dict[str, dict[str, float]]:
    return {
        method: _summary_stats(np.array(list(_metric_values(by_seed, metric).values())))
        for method, by_seed in results.items()
    }


def print_loss_summary(title: str, summary: dict[str, dict[str, float]]) -> None:
    print(f"\n{'=' * 100}\n  {title}\n{'=' * 100}")
    print(f"  {'method':<20}{'n':>4}{'mean':>11}{'std':>11}{'median':>11}{'min':>11}{'max':>11}{'95% CI':>22}")
    print(f"  {'-' * 100}")
    for method in sorted(summary, key=lambda f: summary[f]["mean"]):
        s = summary[method]
        if s["n"] == 0:
            print(f"  {method:<20}{0:>4d}  (no results)")
            continue
        ci = f"[{s['ci95_low']:.4f}, {s['ci95_high']:.4f}]"
        print(f"  {method:<20}{s['n']:>4d}{s['mean']:>11.4f}{s['std']:>11.4f}{s['median']:>11.4f}{s['min']:>11.4f}{s['max']:>11.4f}{ci:>22}")


def _per_target_metric_values(by_seed: dict[int, dict[str, Any]], metric: str) -> dict[str, dict[int, float]]:
    out: dict[str, dict[int, float]] = {}
    for seed, result in by_seed.items():
        per_target = result.get(metric)
        if not per_target:
            continue
        for name, value in per_target.items():
            if value is not None and np.isfinite(value):
                out.setdefault(name, {})[seed] = float(value)
    return out


def method_per_target_loss_summary(
    results: dict[str, dict[int, dict[str, Any]]], metric: str
) -> dict[str, dict[str, dict[str, float]]]:
    summary: dict[str, dict[str, dict[str, float]]] = {}
    for method, by_seed in results.items():
        per_target = _per_target_metric_values(by_seed, metric)
        summary[method] = {name: _summary_stats(np.array(list(values.values()))) for name, values in per_target.items()}
    return summary


def print_per_target_loss_summary(title: str, summary: dict[str, dict[str, dict[str, float]]]) -> None:
    print(f"\n{'=' * 100}\n  {title}\n{'=' * 100}")
    label_names: list[str] = []
    for per_label in summary.values():
        for name in per_label:
            if name not in label_names:
                label_names.append(name)
    if not label_names:
        print("  (no per-target breakdown available for these results)")
        return
    for name in label_names:
        print(f"\n  -- {name} --")
        print(f"  {'method':<20}{'n':>4}{'mean':>11}{'std':>11}{'median':>11}{'min':>11}{'max':>11}")
        print(f"  {'-' * 78}")
        rows = [(m, per_label[name]) for m, per_label in summary.items() if name in per_label and per_label[name]["n"] > 0]
        for method, s in sorted(rows, key=lambda ms: ms[1]["mean"]):
            print(f"  {method:<20}{s['n']:>4d}{s['mean']:>11.4f}{s['std']:>11.4f}{s['median']:>11.4f}{s['min']:>11.4f}{s['max']:>11.4f}")


def _training_dynamics(result: dict[str, Any]) -> dict[str, float] | None:
    """Per-seed train/val trajectory summary. Absent for methods with no
    trained model (mincontrast/palm) -- returns None, filtered out below."""
    history = result.get("history")
    if not history or "train_loss" not in history or "val_loss" not in history:
        return None
    train, val = np.asarray(history["train_loss"]), np.asarray(history["val_loss"])
    if len(train) == 0 or len(val) == 0:
        return None
    val_min, val_last = float(np.min(val)), float(val[-1])
    train_min, train_last = float(np.min(train)), float(train[-1])
    return {
        "n_epochs": len(val), "best_epoch": int(np.argmin(val)) + 1,
        "train_min": train_min, "train_last": train_last,
        "val_min": val_min, "val_last": val_last,
        "overfit_gap": val_last - val_min,
        "overfit_ratio": val_last / val_min if val_min else float("nan"),
    }


def method_training_summary(results: dict[str, dict[int, dict[str, Any]]]) -> dict[str, dict[str, dict[str, float]]]:
    summary: dict[str, dict[str, dict[str, float]]] = {}
    for method, by_seed in results.items():
        per_seed = [d for r in by_seed.values() if (d := _training_dynamics(r)) is not None]
        if not per_seed:
            summary[method] = {}
            continue
        metrics = per_seed[0].keys()
        summary[method] = {m: _summary_stats(np.array([d[m] for d in per_seed])) for m in metrics}
    return summary


def print_training_summary(title: str, summary: dict[str, dict[str, dict[str, float]]]) -> None:
    print(f"\n{'=' * 100}\n  {title}\n{'=' * 100}")
    print(f"  {'method':<20}{'n':>4}{'train_min':>12}{'val_min':>12}{'val_last':>12}{'best_ep':>10}{'overfit_gap':>13}{'overfit_x':>11}")
    print(f"  {'-' * 94}")
    for method in sorted(summary, key=lambda f: summary[f].get("val_min", {}).get("mean", float("inf"))):
        s = summary[method]
        if not s or s["val_min"]["n"] == 0:
            print(f"  {method:<20}{0:>4d}  (no history)")
            continue
        best_ep = f"{s['best_epoch']['mean']:.0f}/{s['n_epochs']['mean']:.0f}"
        print(
            f"  {method:<20}{s['val_min']['n']:>4d}{s['train_min']['mean']:>12.4f}{s['val_min']['mean']:>12.4f}"
            f"{s['val_last']['mean']:>12.4f}{best_ep:>10}{s['overfit_gap']['mean']:>+13.4f}{s['overfit_ratio']['mean']:>11.2f}"
        )


def print_overfit_flags(summary: dict[str, dict[str, dict[str, float]]], ratio_threshold: float = 1.1) -> None:
    flagged = {m: s for m, s in summary.items() if s and s["val_min"]["n"] > 0 and s["overfit_ratio"]["mean"] > ratio_threshold}
    if not flagged:
        return
    print(f"\n{'-' * 88}\n  Overfitting check (val_last / val_min > {ratio_threshold:g})\n{'-' * 88}")
    for method, s in sorted(flagged.items(), key=lambda kv: -kv[1]["overfit_ratio"]["mean"]):
        pct = (s["overfit_ratio"]["mean"] - 1.0) * 100
        print(f"  ! {method}: val loss rose {pct:.0f}% after its best epoch ({s['best_epoch']['mean']:.0f}/{s['n_epochs']['mean']:.0f}).")


def print_paired_comparison(results: dict[str, dict[int, dict[str, Any]]], metric: str) -> dict[str, Any]:
    """Wilcoxon signed-rank test of every method against the method with the
    lowest mean, matched on common seeds (delta = method - best)."""
    per_method = {m: _metric_values(by_seed, metric) for m, by_seed in results.items()}
    means = {m: np.mean(list(v.values())) for m, v in per_method.items() if v}
    comparison: dict[str, Any] = {"metric": metric, "best": None, "vs_best": {}}
    if len(means) < 2:
        return comparison
    best = min(means, key=means.get)
    comparison["best"] = best

    print(f"\n{'=' * 88}\n  Paired comparison vs best method ({best}) on {metric}\n{'=' * 88}")
    print(f"  {'method':<20}{'n_seeds':>9}{'mean_delta':>14}{'wilcoxon_p':>14}")
    print(f"  {'-' * 57}")
    best_values = per_method[best]
    for method in sorted(per_method):
        if method == best:
            continue
        common = sorted(set(per_method[method]) & set(best_values))
        if len(common) < 2:
            continue
        delta = np.array([per_method[method][s] - best_values[s] for s in common])
        try:
            _, p_value = stats.wilcoxon(delta)
        except ValueError:
            p_value = float("nan")
        comparison["vs_best"][method] = {"n_seeds": len(common), "mean_delta": float(np.mean(delta)), "wilcoxon_p": float(p_value)}
        print(f"  {method:<20}{len(common):>9d}{np.mean(delta):>14.4f}{p_value:>14.4f}")
    return comparison


# ══════════════════════════════════════════════════════════════════════════════
# Plotting
# ══════════════════════════════════════════════════════════════════════════════

def _style_axis(ax) -> None:
    ax.set_facecolor(SURFACE_COLOR)
    ax.grid(True, which="both", color=GRID_COLOR, linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def plot_training_curves(results: dict[str, dict[int, dict[str, Any]]], methods: list[str], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True)
    for ax, split in zip(axes, ("train_loss", "val_loss")):
        for i, method in enumerate(methods):
            curves = [np.asarray(r["history"][split]) for r in results.get(method, {}).values() if r.get("history")]
            if not curves:
                continue
            min_len = min(len(c) for c in curves)
            stacked = np.stack([c[:min_len] for c in curves])
            mean, std = stacked.mean(axis=0), stacked.std(axis=0)
            epochs = np.arange(1, min_len + 1)
            color = PALETTE[i % len(PALETTE)]
            ax.plot(epochs, mean, color=color, linewidth=2, label=method)
            ax.fill_between(epochs, mean - std, mean + std, color=color, alpha=0.15, linewidth=0)
        ax.set_yscale("log")
        ax.set_xlabel("epoch")
        ax.set_title(split.replace("_", " "))
        _style_axis(ax)
    axes[0].set_ylabel("loss (log scale)")
    axes[-1].legend(loc="upper right", frameon=False)
    fig.suptitle("Training loss by method (mean ± std across seeds)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved training-curve comparison -> {out_path}")


def plot_loss_distribution(results: dict[str, dict[int, dict[str, Any]]], methods: list[str], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, metric in zip(axes, ("test_loss", "adversarial_loss")):
        data, labels, colors = [], [], []
        for i, method in enumerate(methods):
            values = list(_metric_values(results.get(method, {}), metric).values())
            if not values:
                continue
            data.append(values)
            labels.append(method)
            colors.append(PALETTE[i % len(PALETTE)])
        if not data:
            continue
        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.55, medianprops={"color": "#0b0b0b"})
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.35)
            patch.set_edgecolor(color)
        ax.set_title(metric.replace("_", " "))
        ax.tick_params(axis="x", rotation=30)
        _style_axis(ax)
    axes[0].set_ylabel("loss")
    fig.suptitle("Final loss spread by method across seeds")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved loss-distribution comparison -> {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path, help="RunConfig YAML path (process/filtration/seeds source)")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="path.to.field=value")
    parser.add_argument("--methods", nargs="+", required=True, help="method subdirectories to compare")
    parser.add_argument("--name", default="compare", help="name for this comparison's output subdirectory")
    args = parser.parse_args(argv)

    cfg = load_config(args.config, overrides=args.overrides)
    filtrations = [FILTRATION_REGISTRY.build(f.name, **f.params) for f in cfg.filtration]
    results_paths = ResultsPaths(cfg.process.name, root=cfg.results_root or DEFAULT_RESULTS_ROOT)
    out_dir = results_paths.compare_dir(filtrations, args.name)

    print(f"Loading results for {len(args.methods)} method(s) x {len(cfg.seeds)} seed(s) for process={cfg.process.name!r} ...")
    results = collect_method_results(results_paths, filtrations, args.methods, cfg.seeds)

    methods_present = [m for m in args.methods if results.get(m)]
    if not methods_present:
        raise SystemExit("No results.pt found for any method/seed -- check the config's process/filtration/seeds and --methods.")

    test_summary = method_loss_summary(results, "test_loss")
    adv_summary = method_loss_summary(results, "adversarial_loss")
    print_loss_summary("Test loss across seeds (held-out split)", test_summary)
    print_loss_summary("Adversarial loss across seeds", adv_summary)
    print_per_target_loss_summary("Per-target test loss across seeds", method_per_target_loss_summary(results, "test_loss_per_target"))

    training_summary = method_training_summary(results)
    print_training_summary("Training dynamics (trained methods only)", training_summary)
    print_overfit_flags(training_summary)

    test_comparison = print_paired_comparison(results, "test_loss")
    adv_comparison = print_paired_comparison(results, "adversarial_loss")

    plot_training_curves(results, methods_present, out_dir / "training_curves.png")
    plot_loss_distribution(results, methods_present, out_dir / "loss_distribution.png")

    summary_json = {
        "process": cfg.process.name,
        "methods": methods_present,
        "seeds": cfg.seeds,
        "test_loss": test_summary,
        "adversarial_loss": adv_summary,
        "test_loss_paired": test_comparison,
        "adversarial_loss_paired": adv_comparison,
    }
    import json

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary_json, f, indent=2, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o))
    print(f"\nSaved comparison summary -> {out_dir / 'summary.json'}")

    return summary_json


if __name__ == "__main__":
    main()
