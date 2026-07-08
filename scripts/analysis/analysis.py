# scripts/analysis/analysis.py

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIG: dict[str, Any] = {
    "process": "inhom_thomas",
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
    adversarial_loss: float | None
    adversarial_path: str | None
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


def figs_dir(process: str, dim: int) -> Path:
    return analysis_dir(process, dim) / "figs"


def summaries_dir(process: str, dim: int) -> Path:
    return analysis_dir(process, dim) / "summaries"


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


def optional_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().item()
    try:
        out = float(value)
    except TypeError:
        return None
    if np.isnan(out):
        return None
    return out


def auc(values: list[float]) -> float:
    if len(values) == 0:
        return float("nan")
    if len(values) == 1:
        return float(values[0])

    arr = np.asarray(values, dtype=float)

    if hasattr(np, "trapezoid"):
        area = np.trapezoid(arr)
    else:
        area = np.trapz(arr)

    return float(area / (len(arr) - 1))


def first_epoch_within_threshold(
    values: list[float],
    target: float,
    multiplier: float,
) -> int | None:
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

    adversarial_loss = optional_float(
        payload.get("adversarial_loss", json_meta.get("adversarial_loss"))
    )
    adversarial_path = payload.get(
        "adversarial_path",
        json_meta.get("adversarial_path"),
    )

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
        adversarial_loss=adversarial_loss,
        adversarial_path=str(adversarial_path) if adversarial_path is not None else None,
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
        if r.adversarial_loss is None:
            adversarial_minus_test = np.nan
            adversarial_ratio_to_test = np.nan
        else:
            adversarial_minus_test = r.adversarial_loss - r.test_loss
            adversarial_ratio_to_test = (
                r.adversarial_loss / r.test_loss
                if r.test_loss > 0
                else np.nan
            )

        rows.append(
            {
                "process": r.process,
                "dimension": r.dimension,
                "method": r.method,
                "seed": r.seed,
                "n_epochs": r.n_epochs,
                "test_loss": r.test_loss,
                "adversarial_loss": np.nan if r.adversarial_loss is None else r.adversarial_loss,
                "adversarial_minus_test": adversarial_minus_test,
                "adversarial_ratio_to_test": adversarial_ratio_to_test,
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
                "adversarial_path": r.adversarial_path,
            }
        )

    df = pd.DataFrame(rows)

    df = df.sort_values("test_loss", ascending=True).reset_index(drop=True)
    df["test_rank"] = np.arange(1, len(df) + 1)

    best_test = float(df["test_loss"].min())
    df["relative_test_loss"] = df["test_loss"] / best_test
    df["percent_worse_than_best"] = 100.0 * (df["test_loss"] - best_test) / best_test

    if df["adversarial_loss"].notna().any():
        finite_adv = df["adversarial_loss"].dropna()
        best_adv = float(finite_adv.min())
        df["adversarial_rank"] = df["adversarial_loss"].rank(
            method="min",
            ascending=True,
            na_option="bottom",
        ).astype("Int64")
        df["relative_adversarial_loss"] = df["adversarial_loss"] / best_adv
        df["percent_worse_than_best_adversarial"] = (
            100.0 * (df["adversarial_loss"] - best_adv) / best_adv
        )
    else:
        df["adversarial_rank"] = pd.Series([pd.NA] * len(df), dtype="Int64")
        df["relative_adversarial_loss"] = np.nan
        df["percent_worse_than_best_adversarial"] = np.nan

    return df


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    columns = list(df.columns)

    formatted_rows = []
    for _, row in df.iterrows():
        formatted_rows.append(
            [
                "" if pd.isna(row[col]) else str(row[col])
                for col in columns
            ]
        )

    widths = [
        max(
            len(str(col)),
            *(len(row[i]) for row in formatted_rows),
        )
        for i, col in enumerate(columns)
    ]

    header = "| " + " | ".join(
        str(col).ljust(widths[i])
        for i, col in enumerate(columns)
    ) + " |"

    separator = "| " + " | ".join(
        "-" * widths[i]
        for i in range(len(columns))
    ) + " |"

    body = [
        "| " + " | ".join(
            row[i].ljust(widths[i])
            for i in range(len(columns))
        ) + " |"
        for row in formatted_rows
    ]

    return "\n".join([header, separator, *body])


def plot_overlay_curves(results: list[ExperimentResult], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))

    for r in results:
        epochs = np.arange(1, r.n_epochs + 1)
        ax.plot(
            epochs,
            r.history["train_loss"],
            linestyle="--",
            alpha=0.65,
            label=f"{r.method} train",
        )
        ax.plot(
            epochs,
            r.history["val_loss"],
            linestyle="-",
            alpha=0.95,
            label=f"{r.method} val",
        )

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
    ax.set_title("Normal test loss ranking")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "test_loss_ranking.png", dpi=200)
    plt.close(fig)


def plot_adversarial_loss_ranking(df: pd.DataFrame, out_dir: Path) -> None:
    available = df.dropna(subset=["adversarial_loss"])
    if available.empty:
        return

    ordered = available.sort_values("adversarial_loss", ascending=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(ordered["method"], ordered["adversarial_loss"])
    ax.set_xlabel("Method")
    ax.set_ylabel("Adversarial loss")
    ax.set_title("Adversarial test loss ranking")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "adversarial_loss_ranking.png", dpi=200)
    plt.close(fig)


def plot_test_vs_adversarial_bars(df: pd.DataFrame, out_dir: Path) -> None:
    available = df.dropna(subset=["adversarial_loss"])
    if available.empty:
        return

    ordered = available.sort_values("test_loss", ascending=True)
    x = np.arange(len(ordered))
    width = 0.38

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, ordered["test_loss"], width, label="normal test")
    ax.bar(x + width / 2, ordered["adversarial_loss"], width, label="adversarial")
    ax.set_xticks(x)
    ax.set_xticklabels(ordered["method"], rotation=35, ha="right")
    ax.set_xlabel("Method")
    ax.set_ylabel("Loss")
    ax.set_title("Normal test vs adversarial test loss")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "test_vs_adversarial_bars.png", dpi=200)
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
    ax.set_ylabel("Normal test loss")
    ax.set_title("Validation-test agreement")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "best_val_vs_test.png", dpi=200)
    plt.close(fig)


def plot_test_vs_adversarial_scatter(df: pd.DataFrame, out_dir: Path) -> None:
    available = df.dropna(subset=["adversarial_loss"])
    if available.empty:
        return

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.scatter(available["test_loss"], available["adversarial_loss"])

    for _, row in available.iterrows():
        ax.annotate(
            row["method"],
            (row["test_loss"], row["adversarial_loss"]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
        )

    low = min(available["test_loss"].min(), available["adversarial_loss"].min())
    high = max(available["test_loss"].max(), available["adversarial_loss"].max())
    ax.plot([low, high], [low, high], linestyle="--", linewidth=1)

    ax.set_xlabel("Normal test loss")
    ax.set_ylabel("Adversarial test loss")
    ax.set_title("Normal vs adversarial generalization")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "test_vs_adversarial_scatter.png", dpi=200)
    plt.close(fig)


def plot_overfitting_gap(df: pd.DataFrame, out_dir: Path) -> None:
    ordered = df.sort_values("final_train_val_gap", ascending=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(ordered["method"], ordered["final_train_val_gap"])
    ax.set_xlabel("Method")
    ax.set_ylabel("Final validation loss - final train loss")
    ax.set_title("Final train/validation gap")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "train_val_gap.png", dpi=200)
    plt.close(fig)


def plot_adversarial_gap(df: pd.DataFrame, out_dir: Path) -> None:
    available = df.dropna(subset=["adversarial_minus_test"])
    if available.empty:
        return

    ordered = available.sort_values("adversarial_minus_test", ascending=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(ordered["method"], ordered["adversarial_minus_test"])
    ax.set_xlabel("Method")
    ax.set_ylabel("Adversarial loss - normal test loss")
    ax.set_title("Adversarial generalization gap")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "adversarial_gap.png", dpi=200)
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


def write_markdown_report(
    df: pd.DataFrame,
    out_dir: Path,
    process: str,
    dim: int,
) -> None:
    best = df.iloc[0]
    worst = df.sort_values("test_loss", ascending=False).iloc[0]
    fastest = df.sort_values(
        "convergence_epoch_105",
        ascending=True,
        na_position="last",
    ).iloc[0]
    smallest_gap = df.sort_values("final_train_val_gap", ascending=True).iloc[0]
    most_stable = df.sort_values("late_val_std", ascending=True).iloc[0]

    has_adversarial = df["adversarial_loss"].notna().any()

    lines = []
    lines.append(f"# Parameter-estimation feature comparison: {process}, {dim}d")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Best normal test loss: `{best['method']}` with `{best['test_loss']:.6g}`.")
    lines.append(f"- Worst normal test loss: `{worst['method']}` with `{worst['test_loss']:.6g}`.")
    lines.append(f"- Fastest convergence: `{fastest['method']}` at epoch `{fastest['convergence_epoch_105']}`.")
    lines.append(f"- Smallest final train/validation gap: `{smallest_gap['method']}` with gap `{smallest_gap['final_train_val_gap']:.6g}`.")
    lines.append(f"- Most stable late validation curve: `{most_stable['method']}` with late-val std `{most_stable['late_val_std']:.6g}`.")

    if has_adversarial:
        adv_df = df.dropna(subset=["adversarial_loss"])
        best_adv = adv_df.sort_values("adversarial_loss", ascending=True).iloc[0]
        worst_adv = adv_df.sort_values("adversarial_loss", ascending=False).iloc[0]
        smallest_adv_gap = adv_df.sort_values("adversarial_minus_test", ascending=True).iloc[0]
        largest_adv_gap = adv_df.sort_values("adversarial_minus_test", ascending=False).iloc[0]

        lines.append(f"- Best adversarial loss: `{best_adv['method']}` with `{best_adv['adversarial_loss']:.6g}`.")
        lines.append(f"- Worst adversarial loss: `{worst_adv['method']}` with `{worst_adv['adversarial_loss']:.6g}`.")
        lines.append(f"- Smallest adversarial gap: `{smallest_adv_gap['method']}` with adversarial-minus-test `{smallest_adv_gap['adversarial_minus_test']:.6g}`.")
        lines.append(f"- Largest adversarial gap: `{largest_adv_gap['method']}` with adversarial-minus-test `{largest_adv_gap['adversarial_minus_test']:.6g}`.")
    else:
        lines.append("- No adversarial losses were found in the loaded result files.")

    lines.append("")
    lines.append("## Ranking by normal test loss")
    lines.append("")

    normal_cols = [
        "test_rank",
        "method",
        "test_loss",
        "best_val_loss",
        "best_epoch",
        "final_train_val_gap",
        "convergence_epoch_105",
        "percent_worse_than_best",
    ]
    lines.append(dataframe_to_markdown(df[normal_cols]))

    if has_adversarial:
        lines.append("")
        lines.append("## Ranking by adversarial loss")
        lines.append("")

        adv_cols = [
            "adversarial_rank",
            "method",
            "adversarial_loss",
            "test_loss",
            "adversarial_minus_test",
            "adversarial_ratio_to_test",
            "percent_worse_than_best_adversarial",
        ]

        adv_table = df.dropna(subset=["adversarial_loss"]).sort_values(
            "adversarial_loss",
            ascending=True,
        )
        lines.append(dataframe_to_markdown(adv_table[adv_cols]))

    lines.append("")
    lines.append("## How to read the diagnostics")
    lines.append("")
    lines.append("- `test_loss` is the held-out score from the normal train/val/test split.")
    lines.append("- `adversarial_loss` is the loss on parameter combinations intentionally excluded from the normal dataset.")
    lines.append("- `adversarial_minus_test` measures how much performance degrades on unseen parameter combinations.")
    lines.append("- `best_val_loss` checks whether the model selected by validation also generalizes well.")
    lines.append("- `final_train_val_gap` is a simple train/validation overfitting diagnostic.")
    lines.append("- `convergence_epoch_105` is the first epoch whose validation loss is within 5% of the best validation loss.")
    lines.append("- `val_auc` summarizes the whole validation curve; lower means better average validation performance during training.")
    lines.append("- `late_val_std` measures how noisy or unstable validation loss was near the end of training.")
    lines.append("")
    lines.append("## Generated figures")
    lines.append("")
    lines.append("- `figs/overlay_train_val_curves.png`")
    lines.append("- `figs/validation_curves.png`")
    lines.append("- `figs/test_loss_ranking.png`")
    lines.append("- `figs/best_val_vs_test.png`")
    lines.append("- `figs/train_val_gap.png`")
    lines.append("- `figs/convergence_speed.png`")

    if has_adversarial:
        lines.append("- `figs/adversarial_loss_ranking.png`")
        lines.append("- `figs/test_vs_adversarial_bars.png`")
        lines.append("- `figs/test_vs_adversarial_scatter.png`")
        lines.append("- `figs/adversarial_gap.png`")

    lines.append("")

    with open(out_dir / "analysis_report.md", "w") as f:
        f.write("\n".join(lines))


def analyze_dimension(process: str, dim: int, methods: list[str]) -> None:
    fig_out = figs_dir(process, dim)
    summary_out = summaries_dir(process, dim)

    fig_out.mkdir(parents=True, exist_ok=True)
    summary_out.mkdir(parents=True, exist_ok=True)

    results = collect_results(process, dim, methods)
    df = results_to_frame(results)

    df.to_csv(summary_out / "summary.csv", index=False)
    df.to_json(summary_out / "summary.json", orient="records", indent=2)

    plot_overlay_curves(results, fig_out)
    plot_validation_only(results, fig_out)
    plot_test_loss_ranking(df, fig_out)
    plot_best_val_vs_test(df, fig_out)
    plot_overfitting_gap(df, fig_out)
    plot_convergence(df, fig_out)

    if df["adversarial_loss"].notna().any():
        plot_adversarial_loss_ranking(df, fig_out)
        plot_test_vs_adversarial_bars(df, fig_out)
        plot_test_vs_adversarial_scatter(df, fig_out)
        plot_adversarial_gap(df, fig_out)

    write_markdown_report(df, summary_out, process, dim)

    print(f"\nAnalysis complete for {process}, {dim}d")
    print(f"Saved figures to:   {fig_out}")
    print(f"Saved summaries to: {summary_out}")
    print("")

    display_cols = [
        "test_rank",
        "method",
        "test_loss",
        "adversarial_rank",
        "adversarial_loss",
        "adversarial_minus_test",
        "best_val_loss",
        "best_epoch",
        "final_train_val_gap",
        "convergence_epoch_105",
        "percent_worse_than_best",
    ]

    print(df[display_cols].to_string(index=False))


def main() -> None:
    process = CONFIG["process"]
    dimensions = normalize_dimensions(CONFIG["dimensions"])
    methods = list(CONFIG["methods"])

    for dim in dimensions:
        analyze_dimension(process, dim, methods)


if __name__ == "__main__":
    main()