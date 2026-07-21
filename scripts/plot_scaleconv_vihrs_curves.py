#!/usr/bin/env python3
# scripts/plot_scaleconv_vihrs_curves.py
"""One-off training-curve comparison: the two pi_multik_scaleconv variants
(k=5,10,15 baseline scales vs the 7-channel topo_superset mass-fraction
scales) against the flat-concat pi_multik baseline and the vihrs_500
neural baseline, for a single seed.

Per-seed curves, not mean+-std across seeds (only one seed exists for the
scaleconv variants so far) -- mirrors scripts/evaluate.py's
plot_training_curves styling, but spans three different results/ filtration
tags in one plot (pi_multik*/dtm_k5+10+15,
pi_multik_scaleconv/dtm_m0.01+...+0.90, vihrs_500/raw), which
evaluate.py's single-tag --methods flag can't do in one invocation.

Usage:
    python scripts/plot_scaleconv_vihrs_curves.py --seed 9371
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cloudforger.paths import DEFAULT_RESULTS_ROOT, ExplicitTag, ResultsPaths

# Mirrors scripts/evaluate.py's palette -- no shared plotting module in this
# repo, each plotting script keeps its own copy.
PALETTE = ["#2a78d6", "#1baf7a", "#eda100", "#e34948", "#4a3aa7", "#e87ba4"]
GRID_COLOR = "#e1e0d9"
SURFACE_COLOR = "#fcfcfb"

M_VALUES = [0.01, 0.02, 0.04, 0.10, 0.20, 0.45, 0.90]


def _style_axis(ax) -> None:
    ax.set_facecolor(SURFACE_COLOR)
    ax.grid(True, which="both", color=GRID_COLOR, linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=9371)
    parser.add_argument("--process", default="nested_thomas")
    args = parser.parse_args(argv)

    results_paths = ResultsPaths(args.process, root=DEFAULT_RESULTS_ROOT)
    k_tag = [ExplicitTag("dtm_k5"), ExplicitTag("dtm_k10"), ExplicitTag("dtm_k15")]
    m_tag = [ExplicitTag(f"dtm_m{m:.2f}") for m in M_VALUES]

    runs = {
        "pi_multik (flat concat, k5/10/15)": results_paths.seed_dir(k_tag, "pi_multik", args.seed),
        "pi_multik_scaleconv (k5/10/15)": results_paths.seed_dir(k_tag, "pi_multik_scaleconv", args.seed),
        "pi_multik_scaleconv (7ch topo_superset)": results_paths.seed_dir(m_tag, "pi_multik_scaleconv", args.seed),
        "vihrs_500": results_paths.seed_dir([], "vihrs_500", args.seed),
    }

    results: dict[str, dict] = {}
    for name, seed_dir in runs.items():
        path = seed_dir / "results.pt"
        if not path.exists():
            print(f"  ! missing {path}; skipping {name!r}")
            continue
        results[name] = torch.load(path, map_location="cpu", weights_only=False)

    if not results:
        raise SystemExit(f"No results.pt found for seed {args.seed} under any of the expected paths.")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True)
    for ax, split in zip(axes, ("train_loss", "val_loss")):
        for i, (name, r) in enumerate(results.items()):
            history = r.get("history")
            if not history or split not in history:
                continue
            curve = np.asarray(history[split])
            epochs = np.arange(1, len(curve) + 1)
            ax.plot(epochs, curve, color=PALETTE[i % len(PALETTE)], linewidth=2, label=name)
        ax.set_yscale("log")
        ax.set_xlabel("epoch")
        ax.set_title(split.replace("_", " "))
        _style_axis(ax)
    axes[0].set_ylabel("loss (log scale)")
    axes[-1].legend(loc="upper right", frameon=False, fontsize=9)
    fig.suptitle(f"Training curves, seed {args.seed}: scale-aware pi_multik vs vihrs")
    fig.tight_layout()

    out_dir = results_paths.root / args.process / "_compare" / f"scaleconv_vihrs_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "training_curves.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved -> {out_path}")

    print(f"\n{'method':<42}{'test_loss':>12}{'adv_loss':>12}{'n_epochs':>10}")
    for name, r in results.items():
        n_ep = len(r["history"]["train_loss"]) if r.get("history") else 0
        adv = r.get("adversarial_loss")
        adv_str = f"{adv:.4f}" if adv is not None else "n/a"
        print(f"{name:<42}{r['test_loss']:>12.4f}{adv_str:>12}{n_ep:>10d}")


if __name__ == "__main__":
    main()
