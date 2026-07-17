# models/evaluate_seeds.py

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats

DEFAULT_SEEDS = [
    56245,
    189089,
    2342344,
    9278394,
    91873097,
    908308920,
    235498734453,
    928374129038471,
    974924729845723,
    9267492783429472,
]

DEFAULT_FEATURES = ["betti_0", "betti_1", "pairwise", "pi_0", "pi_1", "raw_pc", "vihrs", "betti_cnn_0"]

# Fixed categorical order (never cycled/reassigned) so a feature keeps its color
# across both figures.
PALETTE = [
    "#2a78d6",  # blue
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
    "#e87ba4",  # magenta
    "#eb6834",  # orange
]
GRID_COLOR = "#e1e0d9"
SURFACE_COLOR = "#fcfcfb"


# ══════════════════════════════════════════════════════════════════════════════
# Loading
# ══════════════════════════════════════════════════════════════════════════════

def result_path(models_root: Path, feature: str, seed: int) -> Path:
    return models_root / feature / f"seed_{seed}" / "results.pt"


def load_seed_result(models_root: Path, feature: str, seed: int) -> dict[str, Any] | None:
    path = result_path(models_root, feature, seed)
    if not path.exists():
        return None
    return torch.load(path, map_location="cpu", weights_only=False)


def collect_feature_results(
    models_root: Path, features: list[str], seeds: list[int]
) -> dict[str, dict[int, dict[str, Any]]]:
    results: dict[str, dict[int, dict[str, Any]]] = {}
    for feature in features:
        by_seed: dict[int, dict[str, Any]] = {}
        for seed in seeds:
            result = load_seed_result(models_root, feature, seed)
            if result is None:
                print(f"  ! missing results.pt for feature={feature!r} seed={seed}; skipping")
                continue
            by_seed[seed] = result
        results[feature] = by_seed
    return results


def print_feature_configs(results: dict[str, dict[int, dict[str, Any]]]) -> None:
    print(f"\n{'=' * 88}\n  Feature configs (from first available seed)\n{'=' * 88}")
    print(f"  {'feature':<14}{'n_seeds':>9}{'n_epochs':>10}{'lr':>10}{'embed_dim':>11}")
    print(f"  {'-' * 54}")
    for feature, by_seed in results.items():
        if not by_seed:
            print(f"  {feature:<14}{0:>9d}{'--':>10}{'--':>10}{'--':>11}")
            continue
        cfg = next(iter(by_seed.values()))["config"]
        print(
            f"  {feature:<14}{len(by_seed):>9d}{cfg['n_epochs']:>10d}"
            f"{cfg['lr']:>10.4g}{cfg['embedding_dim']:>11d}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# Statistics
# ══════════════════════════════════════════════════════════════════════════════

def _metric_values(by_seed: dict[int, dict[str, Any]], metric: str) -> dict[int, float]:
    """Seed -> metric value, dropping seeds where the metric is missing/None."""
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
        "n": n,
        "mean": mean,
        "std": std,
        "median": float(np.median(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
    }


def feature_loss_summary(
    results: dict[str, dict[int, dict[str, Any]]], metric: str
) -> dict[str, dict[str, float]]:
    return {
        feature: _summary_stats(np.array(list(_metric_values(by_seed, metric).values())))
        for feature, by_seed in results.items()
    }


def print_loss_summary(title: str, summary: dict[str, dict[str, float]]) -> None:
    print(f"\n{'=' * 100}\n  {title}\n{'=' * 100}")
    print(
        f"  {'feature':<14}{'n':>4}{'mean':>11}{'std':>11}{'median':>11}"
        f"{'min':>11}{'max':>11}{'95% CI':>22}"
    )
    print(f"  {'-' * 94}")
    for feature in sorted(summary, key=lambda f: summary[f]["mean"]):
        s = summary[feature]
        if s["n"] == 0:
            print(f"  {feature:<14}{0:>4d}  (no results)")
            continue
        ci = f"[{s['ci95_low']:.4f}, {s['ci95_high']:.4f}]"
        print(
            f"  {feature:<14}{s['n']:>4d}{s['mean']:>11.4f}{s['std']:>11.4f}"
            f"{s['median']:>11.4f}{s['min']:>11.4f}{s['max']:>11.4f}{ci:>22}"
        )


def _per_target_metric_values(
    by_seed: dict[int, dict[str, Any]], metric: str
) -> dict[str, dict[int, float]]:
    """label_name -> {seed -> value}, for a *_per_target metric (a dict
    keyed by label name saved per seed, e.g. result['test_loss_per_target']).
    Seeds/targets missing or non-finite are dropped, same convention as
    _metric_values. Silently yields {} for results that predate this metric
    (e.g. old results.pt without a per-target breakdown) rather than erroring,
    so this can be called on a mix of old and new runs."""
    out: dict[str, dict[int, float]] = {}
    for seed, result in by_seed.items():
        per_target = result.get(metric)
        if not per_target:
            continue
        for name, value in per_target.items():
            if value is not None and np.isfinite(value):
                out.setdefault(name, {})[seed] = float(value)
    return out


def feature_per_target_loss_summary(
    results: dict[str, dict[int, dict[str, Any]]], metric: str
) -> dict[str, dict[str, dict[str, float]]]:
    """feature -> label_name -> across-seed summary stats, for a
    *_per_target metric (e.g. 'test_loss_per_target')."""
    summary: dict[str, dict[str, dict[str, float]]] = {}
    for feature, by_seed in results.items():
        per_target = _per_target_metric_values(by_seed, metric)
        summary[feature] = {
            name: _summary_stats(np.array(list(values.values())))
            for name, values in per_target.items()
        }
    return summary


def print_per_target_loss_summary(title: str, summary: dict[str, dict[str, dict[str, float]]]) -> None:
    """One table per target label, each row a feature, sorted by mean loss
    (lowest first) -- mirrors print_loss_summary's layout/columns, just
    split out per target instead of aggregated across all of them."""
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
        print(
            f"  {'feature':<16}{'n':>4}{'mean':>11}{'std':>11}{'median':>11}"
            f"{'min':>11}{'max':>11}"
        )
        print(f"  {'-' * 74}")
        rows = [(f, per_label[name]) for f, per_label in summary.items() if name in per_label and per_label[name]["n"] > 0]
        for feature, s in sorted(rows, key=lambda fs: fs[1]["mean"]):
            print(
                f"  {feature:<16}{s['n']:>4d}{s['mean']:>11.4f}{s['std']:>11.4f}"
                f"{s['median']:>11.4f}{s['min']:>11.4f}{s['max']:>11.4f}"
            )


def _training_dynamics(result: dict[str, Any]) -> dict[str, float] | None:
    """Per-seed train/val trajectory summary: best (min) and final (last)
    loss for each split, the epoch the best val checkpoint came from, and how
    much val degraded between that checkpoint and the final epoch. A large
    positive overfit_gap/overfit_ratio is the standard overfitting signature:
    train loss still falling while val loss has already turned back up."""
    history = result.get("history")
    if not history or "train_loss" not in history or "val_loss" not in history:
        return None
    train, val = np.asarray(history["train_loss"]), np.asarray(history["val_loss"])
    if len(train) == 0 or len(val) == 0:
        return None
    val_min, val_last = float(np.min(val)), float(val[-1])
    train_min, train_last = float(np.min(train)), float(train[-1])
    return {
        "n_epochs": len(val),
        "best_epoch": int(np.argmin(val)) + 1,
        "train_min": train_min,
        "train_last": train_last,
        "val_min": val_min,
        "val_last": val_last,
        "overfit_gap": val_last - val_min,
        "overfit_ratio": val_last / val_min if val_min else float("nan"),
    }


def feature_training_summary(
    results: dict[str, dict[int, dict[str, Any]]]
) -> dict[str, dict[str, dict[str, float]]]:
    """feature -> {metric -> across-seed summary stats} for every metric
    _training_dynamics reports (train_min, val_last, overfit_gap, ...)."""
    summary: dict[str, dict[str, dict[str, float]]] = {}
    for feature, by_seed in results.items():
        per_seed = [d for r in by_seed.values() if (d := _training_dynamics(r)) is not None]
        if not per_seed:
            summary[feature] = {}
            continue
        metrics = per_seed[0].keys()
        summary[feature] = {m: _summary_stats(np.array([d[m] for d in per_seed])) for m in metrics}
    return summary


def print_training_summary(title: str, summary: dict[str, dict[str, dict[str, float]]]) -> None:
    print(f"\n{'=' * 100}\n  {title}\n{'=' * 100}")
    print(
        f"  {'feature':<14}{'n':>4}{'train_min':>12}{'val_min':>12}{'val_last':>12}"
        f"{'best_ep':>10}{'overfit_gap':>13}{'overfit_x':>11}"
    )
    print(f"  {'-' * 94}")
    for feature in sorted(summary, key=lambda f: summary[f].get("val_min", {}).get("mean", float("inf"))):
        s = summary[feature]
        if not s or s["val_min"]["n"] == 0:
            print(f"  {feature:<14}{0:>4d}  (no history)")
            continue
        best_ep = f"{s['best_epoch']['mean']:.0f}/{s['n_epochs']['mean']:.0f}"
        print(
            f"  {feature:<14}{s['val_min']['n']:>4d}"
            f"{s['train_min']['mean']:>12.4f}"
            f"{s['val_min']['mean']:>12.4f}"
            f"{s['val_last']['mean']:>12.4f}"
            f"{best_ep:>10}"
            f"{s['overfit_gap']['mean']:>+13.4f}"
            f"{s['overfit_ratio']['mean']:>11.2f}"
        )
    print(
        "  (best_ep = mean epoch of the best val checkpoint, out of n_epochs trained;"
        " overfit_gap = val_last - val_min; overfit_x = val_last / val_min)"
    )


def print_overfit_flags(summary: dict[str, dict[str, dict[str, float]]], ratio_threshold: float = 1.1) -> None:
    """Call out any feature whose val loss climbed back up by more than
    `ratio_threshold` between its best checkpoint and the final epoch --
    i.e. it kept training well past the point of diminishing/negative
    returns on held-out data."""
    flagged = {
        f: s for f, s in summary.items()
        if s and s["val_min"]["n"] > 0 and s["overfit_ratio"]["mean"] > ratio_threshold
    }
    if not flagged:
        return
    print(f"\n{'-' * 88}\n  Overfitting check (val_last / val_min > {ratio_threshold:g})\n{'-' * 88}")
    for feature, s in sorted(flagged.items(), key=lambda kv: -kv[1]["overfit_ratio"]["mean"]):
        pct = (s["overfit_ratio"]["mean"] - 1.0) * 100
        print(
            f"  ! {feature}: val loss rose {pct:.0f}% after its best epoch "
            f"({s['best_epoch']['mean']:.0f}/{s['n_epochs']['mean']:.0f}) while training continued to epoch "
            f"{s['n_epochs']['mean']:.0f} -- train loss kept falling (train_min={s['train_min']['mean']:.4f}) "
            f"while val degraded (val_min={s['val_min']['mean']:.4f} -> val_last={s['val_last']['mean']:.4f})."
        )


def print_paired_comparison(results: dict[str, dict[int, dict[str, Any]]], metric: str) -> None:
    """Wilcoxon signed-rank test of every feature against the feature with the
    lowest mean, matched on common seeds (delta = feature - best)."""
    per_feature = {f: _metric_values(by_seed, metric) for f, by_seed in results.items()}
    means = {f: np.mean(list(v.values())) for f, v in per_feature.items() if v}
    if len(means) < 2:
        return
    best = min(means, key=means.get)

    print(f"\n{'=' * 88}\n  Paired comparison vs best feature ({best}) on {metric}\n{'=' * 88}")
    print(f"  {'feature':<14}{'n_seeds':>9}{'mean_delta':>14}{'wilcoxon_p':>14}")
    print(f"  {'-' * 51}")
    best_values = per_feature[best]
    for feature in sorted(per_feature):
        if feature == best:
            continue
        common = sorted(set(per_feature[feature]) & set(best_values))
        if len(common) < 2:
            continue
        delta = np.array([per_feature[feature][s] - best_values[s] for s in common])
        try:
            _, p_value = stats.wilcoxon(delta)
        except ValueError:
            p_value = float("nan")
        print(f"  {feature:<14}{len(common):>9d}{np.mean(delta):>14.4f}{p_value:>14.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# Plotting
# ══════════════════════════════════════════════════════════════════════════════

def _style_axis(ax) -> None:
    ax.set_facecolor(SURFACE_COLOR)
    ax.grid(True, which="both", color=GRID_COLOR, linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def plot_training_curves(
    results: dict[str, dict[int, dict[str, Any]]], features: list[str], out_path: Path
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True)
    for ax, split in zip(axes, ("train_loss", "val_loss")):
        for i, feature in enumerate(features):
            curves = [
                np.asarray(r["history"][split])
                for r in results.get(feature, {}).values()
                if "history" in r
            ]
            if not curves:
                continue
            min_len = min(len(c) for c in curves)
            stacked = np.stack([c[:min_len] for c in curves])
            mean, std = stacked.mean(axis=0), stacked.std(axis=0)
            epochs = np.arange(1, min_len + 1)
            color = PALETTE[i % len(PALETTE)]
            ax.plot(epochs, mean, color=color, linewidth=2, label=feature)
            ax.fill_between(epochs, mean - std, mean + std, color=color, alpha=0.15, linewidth=0)
        ax.set_yscale("log")
        ax.set_xlabel("epoch")
        ax.set_title(split.replace("_", " "))
        _style_axis(ax)
    axes[0].set_ylabel("loss (log scale)")
    axes[-1].legend(loc="upper right", frameon=False)
    fig.suptitle("Training loss by feature (mean ± std across seeds)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved training-curve comparison -> {out_path}")


def plot_loss_distribution(
    results: dict[str, dict[int, dict[str, Any]]], features: list[str], out_path: Path
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, metric in zip(axes, ("test_loss", "adversarial_loss")):
        data, labels, colors = [], [], []
        for i, feature in enumerate(features):
            values = list(_metric_values(results.get(feature, {}), metric).values())
            if not values:
                continue
            data.append(values)
            labels.append(feature)
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
    fig.suptitle("Final loss spread by feature across seeds")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved loss-distribution comparison -> {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--models-root", type=Path, required=True,
        help="e.g. results/params/2d/thomas — parent directory holding one subdirectory per feature",
    )
    ap.add_argument("--features", nargs="+", default=DEFAULT_FEATURES,
                     help="feature/method subdirectories to compare")
    ap.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    ap.add_argument("--out-dir", type=Path, default=None,
                     help="directory for output plots; default: <models-root>/summary")
    args = ap.parse_args()

    out_dir = args.out_dir or (args.models_root / "summary")

    print(f"Loading results for {len(args.features)} feature(s) x {len(args.seeds)} seed(s) "
          f"from {args.models_root} ...")
    results = collect_feature_results(args.models_root, args.features, args.seeds)

    features_present = [f for f in args.features if results.get(f)]
    if not features_present:
        raise SystemExit("No results.pt found for any feature/seed — check --models-root/--features/--seeds.")

    print_feature_configs(results)
    print_loss_summary("Test loss across seeds (held-out split)", feature_loss_summary(results, "test_loss"))
    print_loss_summary("Adversarial loss across seeds", feature_loss_summary(results, "adversarial_loss"))
    print_paired_comparison(results, "test_loss")
    print_paired_comparison(results, "adversarial_loss")

    plot_training_curves(results, features_present, out_dir / "training_curves.png")
    plot_loss_distribution(results, features_present, out_dir / "loss_distribution.png")


if __name__ == "__main__":
    main()
