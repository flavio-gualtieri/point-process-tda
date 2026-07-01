# scripts/analysis/analysis.py

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIG: dict[str, Any] = {
    "process": "thomas",
    "dimensions": [2],
    "methods": ["raw_pc", "pi_0", "pi_1", "pairwise", "betti_0", "betti_1"],
    "results_root": PROJECT_ROOT / "results" / "params",
    "analysis_root": PROJECT_ROOT / "results" / "analysis" / "params",
    "results_filename": "results.pt",
    "json_filename": "results.json",
    "top_k_epochs_for_late_val": 5,
}


@dataclass
class ExperimentResult:
    process: str
    dimension: int
    method: str
    path: Path
    seed: int | None
    n_epochs: int
    test_loss: float
    best_val_loss: float
    best_epoch: int
    final_train_loss: float
    final_val_loss: float
    min_train_loss: float
    train_val_gap_at_best: float
    final_train_val_gap: float
    overfit_ratio: float
    convergence_epoch_105: int | None
    val_auc: float
    train_auc: float
    late_val_mean: float
    late_val_std: float
    label_names: list[str]
    history: dict[str, list[float]]
    config: dict[str, Any]


def normalize_dimensions(value: int | list[int]) -> list[int]:
    if isinstance(value, int):
        return [value]
    if isinstance(value, list) and all(isinstance(v, int) for v in value):
        return value
    raise TypeError("CONFIG['dimensions'] must be an int or a list of ints.")


def dim_dir_name(dim: int) -> str:
    return f"{dim}d"


def result_path(process: str, dim: int, method: str) -> Path:
    return (
        Path(CONFIG["results_root"])
        / dim_dir_name(dim)
        / process
        / method
        / CONFIG["results_filename"]
    )


def json_path(process: str, dim: int, method: str) -> Path:
    return (
        Path(CONFIG["results_root"])
        / dim_dir_name(dim)
        / process
        / method
        / CONFIG["json_filename"]
    )


def analysis_dir(process: str, dim: int) -> Path:
    return Path(CONFIG["analysis_root"]) / dim_dir_name(dim) / process


def safe_torch_load(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return json.load(f)


def auc(values: list[float]) -> float:
    if len(values) == 0:
        return float("nan")
    if len(values) == 1:
        return float(values[0])
    return float(np.trapz(np.asarray(values, dtype=float)) / (len(values) - 1))


def first_epoch_within_threshold(values: list[float], target: float, multiplier: float) -> int | None:
    threshold = multiplier * target
    for i, value in enumerate(values, start=1):
        if value <= threshold:
            return i
    return None


def load_experiment(process: str, dim: int, method: str) -> ExperimentResult | None:
    pt_path = result_path(process, dim, method)
    meta_path = json_path(process, dim, method)

    if not pt_path.exists():
        print(f"Missing: {pt_path}")
        return None

    payload = safe_torch_load(pt_path)
    json_meta = read_json_if_exists(meta_path)

    history = payload["history"]
    train_loss = [float(x) for x in history["train_loss"]]
    val_loss = [float(x) for x in history["val_loss"]]

    if len(train_loss) != len(val_loss):
        raise ValueError(f"{pt_path}: train_loss and val_loss lengths differ.")

    if len(train_loss) == 0:
        raise ValueError(f"{pt_path}: empty history.")

    best_idx = int(np.argmin(val_loss))
    best_epoch = best_idx + 1
    best_val_loss = float(val_loss[best_idx])

    train_at_best = float(train_loss[best_idx])
    final_train = float(train_loss[-1])
    final_val = float(val_loss[-1])
    min_train = float(np.min(train_loss))
    test_loss = float(payload.get("test_loss", json_meta.get("test_loss", float("nan"))))

    train_val_gap_at_best = best_val_loss - train_at_best
    final_train_val_gap = final_val - final_train
    overfit_ratio = final_val / final_train if final_train > 0 else float("inf")
    convergence_epoch_105 = first_epoch_within_threshold(val_loss, best_val_loss, 1.05)

    k = int(CONFIG["top_k_epochs_for_late_val"])
    late_vals = np.asarray(val_loss[-k:], dtype=float)

    cfg = payload.get("config", {})
    seed = cfg.get("seed", json_meta.get("seed"))

    return ExperimentResult(
        process=process,
        dimension=dim,
        method=method,
        path=pt_path,
        seed=seed,
        n_epochs=len(train_loss),
        test_loss=test_loss,
        best_val_loss=best_val_loss,
        best_epoch=best_epoch,
        final_train_loss=final_train,
        final_val_loss=final_val,
        min_train_loss=min_train,
        train_val_gap_at_best=float(train_val_gap_at_best),
        final_train_val_gap=float(final_train_val_gap),
        overfit_ratio=float(overfit_ratio),
        convergence_epoch_105=convergence_epoch_105,
        val_auc=auc(val_loss),
        train_auc=auc(train_loss),
        late_val_mean=float(late_vals.mean()),
        late_val_std=float(late_vals.std()),
        label_names=list(payload.get("label_names", [])),
        history={"train_loss": train_loss, "val_loss": val_loss},
        config=cfg,
    )


def collect_results(process: str, dim: int, methods: list[str]) -> list[ExperimentResult]:
    results: list[ExperimentResult] = []

    for method in methods:
        result = load_experiment(process, dim, method)
        if result is not None:
            results.append(result)

    if not results:
        raise FileNotFoundError(
            f"No experiment results found for process={process}, dim={dim}."
        )

    return results


def results_to_frame(results: list[ExperimentResult]) -> pd.DataFrame:
    rows = []

    for r in results:
        rows.append(
            {
                "process": r.process,
                "dimension": r.dimension,
                "method": r.method,
                "seed": r.seed,
                "n_epochs": r.n_epochs,
                "test_loss": r.test_loss,
                "best_val_loss": r.best_val_loss,
                "best_epoch": r.best_epoch,
                "final_train_loss": r.final_train_loss,
                "final_val_loss": r.final_val_loss,
                "min_train_loss": r.min_train_loss,
                "train_val_gap_at_best": r.train_val_gap_at_best,
                "final_train_val_gap": r.final_train_val_gap,
                "overfit_ratio": r.overfit_ratio,
                "convergence_epoch_105": r.convergence_epoch_105,
                "train_auc": r.train_auc,
                "val_auc": r.val_auc,
                "late_val_mean": r.late_val_mean,
                "late_val_std": r.late_val_std,
                "path": str(r.path),
            }
        )

    df = pd.DataFrame(rows)
    df = df.sort_values("test_loss", ascending=True).reset_index(drop=True)
    df["test_rank"] = np.arange(1, len(df) + 1)

    best_test = float(df["test_loss"].min())
    df["relative_test_loss"] = df["test_loss"] / best_test
    df["percent_worse_than_best"] = 100.0 * (df["test_loss"] - best_test) / best_test

    return df


def plot_overlay_curves(results: list[ExperimentResult], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))

    for r in results:
        epochs = np.arange(1, r.n_epochs + 1)
        ax.plot(epochs, r.history["train_loss"], linestyle="--", alpha=0.65, label=f"{r.method} train")
        ax.plot(epochs, r.history["val_loss"], linestyle="-", alpha=0.95, label=f"{r.method} val")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training and validation loss by feature")
    ax.legend(fontsize=8, ncols=2)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "overlay_train_val_curves.png", dpi=200)
    plt.close(fig)


def plot_validation_only(results: list[ExperimentResult], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))

    for r in results:
        epochs = np.arange(1, r.n_epochs + 1)
        ax.plot(epochs, r.history["val_loss"], label=r.method)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation loss")
    ax.set_title("Validation loss by feature")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "validation_curves.png", dpi=200)
    plt.close(fig)


def plot_test_loss_ranking(df: pd.DataFrame, out_dir: Path) -> None:
    ordered = df.sort_values("test_loss", ascending=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(ordered["method"], ordered["test_loss"])
    ax.set_xlabel("Method")
    ax.set_ylabel("Test loss")
    ax.set_title("Test loss ranking")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "test_loss_ranking.png", dpi=200)
    plt.close(fig)


def plot_best_val_vs_test(df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))

    ax.scatter(df["best_val_loss"], df["test_loss"])

    for _, row in df.iterrows():
        ax.annotate(
            row["method"],
            (row["best_val_loss"], row["test_loss"]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
        )

    low = min(df["best_val_loss"].min(), df["test_loss"].min())
    high = max(df["best_val_loss"].max(), df["test_loss"].max())
    ax.plot([low, high], [low, high], linestyle="--", linewidth=1)

    ax.set_xlabel("Best validation loss")
    ax.set_ylabel("Test loss")
    ax.set_title("Validation-test agreement")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "best_val_vs_test.png", dpi=200)
    plt.close(fig)


def plot_overfitting_gap(df: pd.DataFrame, out_dir: Path) -> None:
    ordered = df.sort_values("final_train_val_gap", ascending=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(ordered["method"], ordered["final_train_val_gap"])
    ax.set_xlabel("Method")
    ax.set_ylabel("Final validation loss - final train loss")
    ax.set_title("Final generalization gap")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "generalization_gap.png", dpi=200)
    plt.close(fig)


def plot_convergence(df: pd.DataFrame, out_dir: Path) -> None:
    view = df.copy()
    max_epoch = int(view["n_epochs"].max())
    view["convergence_epoch_105_plot"] = view["convergence_epoch_105"].fillna(max_epoch + 1)
    ordered = view.sort_values("convergence_epoch_105_plot", ascending=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(ordered["method"], ordered["convergence_epoch_105_plot"])
    ax.set_xlabel("Method")
    ax.set_ylabel("First epoch within 5% of best validation loss")
    ax.set_title("Convergence speed")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "convergence_speed.png", dpi=200)
    plt.close(fig)


def plot_metric_heatmap(df: pd.DataFrame, out_dir: Path) -> None:
    metrics = [
        "test_loss",
        "best_val_loss",
        "final_train_val_gap",
        "overfit_ratio",
        "val_auc",
        "late_val_std",
    ]

    matrix = df.set_index("method")[metrics].copy()

    normalized = matrix.copy()
    for col in normalized.columns:
        values = normalized[col].astype(float)
        lo = values.min()
        hi = values.max()
        if math.isclose(lo, hi):
            normalized[col] = 0.0
        else:
            normalized[col] = (values - lo) / (hi - lo)

    fig, ax = plt.subplots(figsize=(10, max(4, 0.5 * len(normalized))))
    im = ax.imshow(normalized.values, aspect="auto")

    ax.set_xticks(np.arange(len(metrics)))
    ax.set_yticks(np.arange(len(normalized.index)))
    ax.set_xticklabels(metrics, rotation=35, ha="right")
    ax.set_yticklabels(normalized.index)
    ax.set_title("Normalized diagnostic metrics; lower is better")

    for i in range(normalized.shape[0]):
        for j in range(normalized.shape[1]):
            raw = matrix.iloc[i, j]
            ax.text(j, i, f"{raw:.3g}", ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_dir / "diagnostic_heatmap.png", dpi=200)
    plt.close(fig)


def write_markdown_report(df: pd.DataFrame, out_dir: Path, process: str, dim: int) -> None:
    best = df.iloc[0]
    worst = df.sort_values("test_loss", ascending=False).iloc[0]
    fastest = df.sort_values("convergence_epoch_105", ascending=True, na_position="last").iloc[0]
    smallest_gap = df.sort_values("final_train_val_gap", ascending=True).iloc[0]
    most_stable = df.sort_values("late_val_std", ascending=True).iloc[0]

    lines = []
    lines.append(f"# Parameter-estimation feature comparison: {process}, {dim}d")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Best test loss: `{best['method']}` with `{best['test_loss']:.6g}`.")
    lines.append(f"- Worst test loss: `{worst['method']}` with `{worst['test_loss']:.6g}`.")
    lines.append(f"- Fastest convergence: `{fastest['method']}` at epoch `{fastest['convergence_epoch_105']}`.")
    lines.append(f"- Smallest final train/validation gap: `{smallest_gap['method']}` with gap `{smallest_gap['final_train_val_gap']:.6g}`.")
    lines.append(f"- Most stable late validation curve: `{most_stable['method']}` with late-val std `{most_stable['late_val_std']:.6g}`.")
    lines.append("")
    lines.append("## Ranking by test loss")
    lines.append("")
    lines.append(df[
        [
            "test_rank",
            "method",
            "test_loss",
            "best_val_loss",
            "best_epoch",
            "final_train_val_gap",
            "convergence_epoch_105",
            "percent_worse_than_best",
        ]
    ].to_markdown(index=False))
    lines.append("")
    lines.append("## How to read the diagnostics")
    lines.append("")
    lines.append("- `test_loss` is the main held-out score from the runner's normal train/val/test split.")
    lines.append("- `best_val_loss` checks whether the model selected by validation also generalizes well.")
    lines.append("- `final_train_val_gap` is a simple overfitting diagnostic.")
    lines.append("- `convergence_epoch_105` is the first epoch whose validation loss is within 5% of the best validation loss.")
    lines.append("- `val_auc` summarizes the whole validation curve; lower means better average validation performance during training.")
    lines.append("- `late_val_std` measures how noisy or unstable validation loss was near the end of training.")
    lines.append("")
    lines.append("## Generated figures")
    lines.append("")
    lines.append("- `overlay_train_val_curves.png`")
    lines.append("- `validation_curves.png`")
    lines.append("- `test_loss_ranking.png`")
    lines.append("- `best_val_vs_test.png`")
    lines.append("- `generalization_gap.png`")
    lines.append("- `convergence_speed.png`")
    lines.append("- `diagnostic_heatmap.png`")
    lines.append("")

    with open(out_dir / "analysis_report.md", "w") as f:
        f.write("\n".join(lines))


def analyze_dimension(process: str, dim: int, methods: list[str]) -> None:
    out_dir = analysis_dir(process, dim)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = collect_results(process, dim, methods)
    df = results_to_frame(results)

    df.to_csv(out_dir / "summary.csv", index=False)
    df.to_json(out_dir / "summary.json", orient="records", indent=2)

    plot_overlay_curves(results, out_dir)
    plot_validation_only(results, out_dir)
    plot_test_loss_ranking(df, out_dir)
    plot_best_val_vs_test(df, out_dir)
    plot_overfitting_gap(df, out_dir)
    plot_convergence(df, out_dir)
    plot_metric_heatmap(df, out_dir)
    write_markdown_report(df, out_dir, process, dim)

    print(f"\nAnalysis complete for {process}, {dim}d")
    print(f"Saved outputs to: {out_dir}")
    print("")
    print(df[
        [
            "test_rank",
            "method",
            "test_loss",
            "best_val_loss",
            "best_epoch",
            "final_train_val_gap",
            "convergence_epoch_105",
            "percent_worse_than_best",
        ]
    ].to_string(index=False))


def main() -> None:
    process = CONFIG["process"]
    dimensions = normalize_dimensions(CONFIG["dimensions"])
    methods = list(CONFIG["methods"])

    for dim in dimensions:
        analyze_dimension(process, dim, methods)


if __name__ == "__main__":
    main()