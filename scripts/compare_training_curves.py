#!/usr/bin/env python3
# scripts/compare_training_curves.py
"""Quick, ad-hoc overlay of training curves (train/val loss, mean over seeds)
for an arbitrary set of methods within a process. For fast one-off checks
only -- output always lands under results/<process>/_compare/temp/ and gets
overwritten by later runs with the same method set (see scripts/evaluate.py
for the archived, RunConfig-driven comparison with full metrics).

Each method is auto-discovered by scanning results/<process>/*/<method>/ for
filtration-tag subdirectories. If a method name exists under more than one
filtration tag, disambiguate with "<filtration_tag>:<method>", e.g.
"dtm_k5+10+15:pi_multik_scaleconv".

Usage:
    python scripts/compare_training_curves.py --process nested_thomas \
        --methods pi_multik pi_multik_scaleconv vihrs_500 --seeds 9371 9372
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cloudforger.paths import COMPARE_DIR_NAME, DEFAULT_RESULTS_ROOT

# Mirrors scripts/evaluate.py's palette -- no shared plotting module in this
# repo, each plotting script keeps its own copy.
PALETTE = [
    "#2a78d6", "#1baf7a", "#eda100", "#008300",
    "#4a3aa7", "#e34948", "#e87ba4", "#eb6834",
]
GRID_COLOR = "#e1e0d9"
SURFACE_COLOR = "#fcfcfb"


def _style_axis(ax) -> None:
    ax.set_facecolor(SURFACE_COLOR)
    ax.grid(True, which="both", color=GRID_COLOR, linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def resolve_method_dir(process_dir: Path, spec: str) -> tuple[str, Path]:
    """spec is either "method" (auto-discovered across filtration tags) or
    an explicit "tag:method". Returns (label, method_dir)."""
    if ":" in spec:
        tag, method = spec.split(":", 1)
        method_dir = process_dir / tag / method
        if not method_dir.is_dir():
            raise SystemExit(f"No results at {method_dir}")
        return spec, method_dir

    matches = sorted(
        m for m in process_dir.glob(f"*/{spec}") if m.is_dir() and m.parent.name != COMPARE_DIR_NAME
    )
    if not matches:
        raise SystemExit(f"No method dir named {spec!r} found under {process_dir}")
    if len(matches) > 1:
        options = ", ".join(f"{m.parent.name}:{spec}" for m in matches)
        raise SystemExit(f"{spec!r} is ambiguous across filtration tags -- disambiguate with one of: {options}")
    return spec, matches[0]


def load_seed_results(method_dir: Path, seeds: list[int]) -> dict[int, dict[str, Any]]:
    by_seed = {}
    for seed in seeds:
        path = method_dir / f"seed_{seed}" / "results.pt"
        if not path.exists():
            print(f"  ! missing {path}; skipping")
            continue
        by_seed[seed] = torch.load(path, map_location="cpu", weights_only=False)
    return by_seed


def plot_curves(curves_by_label: dict[str, dict[int, dict[str, Any]]], out_path: Path, title: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True)
    for ax, split in zip(axes, ("train_loss", "val_loss")):
        for i, (label, by_seed) in enumerate(curves_by_label.items()):
            curves = [
                np.asarray(r["history"][split])
                for r in by_seed.values()
                if r.get("history") and split in r["history"]
            ]
            if not curves:
                continue
            min_len = min(len(c) for c in curves)
            stacked = np.stack([c[:min_len] for c in curves])
            mean = stacked.mean(axis=0)
            epochs = np.arange(1, min_len + 1)
            color = PALETTE[i % len(PALETTE)]
            ax.plot(epochs, mean, color=color, linewidth=2, label=label)
            if len(curves) > 1:
                std = stacked.std(axis=0)
                ax.fill_between(epochs, mean - std, mean + std, color=color, alpha=0.15, linewidth=0)
        ax.set_yscale("log")
        ax.set_xlabel("epoch")
        ax.set_title(split.replace("_", " "))
        _style_axis(ax)
    axes[0].set_ylabel("loss (log scale)")
    axes[-1].legend(loc="upper right", frameon=False, fontsize=9)
    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved -> {out_path}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--process", required=True)
    parser.add_argument("--methods", nargs="+", required=True, help='method name, or "tag:method" if ambiguous')
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    args = parser.parse_args(argv)

    process_dir = Path(DEFAULT_RESULTS_ROOT) / args.process

    curves_by_label: dict[str, dict[int, dict[str, Any]]] = {}
    for spec in args.methods:
        label, method_dir = resolve_method_dir(process_dir, spec)
        by_seed = load_seed_results(method_dir, args.seeds)
        if not by_seed:
            print(f"  ! no results found for {label!r}; excluding from plot")
            continue
        curves_by_label[label] = by_seed

    if not curves_by_label:
        raise SystemExit("No results found for any requested method/seed combination.")

    print(f"\n{'method':<45}{'seeds found':>12}{'test_loss':>12}{'n_epochs':>10}")
    for label, by_seed in curves_by_label.items():
        test_losses = [r["test_loss"] for r in by_seed.values() if r.get("test_loss") is not None]
        test_str = f"{np.mean(test_losses):.4f}" if test_losses else "n/a"
        n_ep = min((len(r["history"]["train_loss"]) for r in by_seed.values() if r.get("history")), default=0)
        print(f"{label:<45}{len(by_seed):>12d}{test_str:>12}{n_ep:>10d}")

    seeds_str = ",".join(str(s) for s in args.seeds)
    subdir = "_vs_".join(label.replace(":", "-") for label in curves_by_label)
    out_dir = process_dir / COMPARE_DIR_NAME / "temp" / subdir
    title = f"Training curves, {args.process} (seeds {seeds_str})"
    plot_curves(curves_by_label, out_dir / "training_curves.png", title)


if __name__ == "__main__":
    main()
