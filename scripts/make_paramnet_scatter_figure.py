#!/usr/bin/env python3
# scripts/make_paramnet_scatter_figure.py
"""Build the predicted-vs-true scatter figure requested by the \\figtodo in
docs/final_report.tex Section~\\ref{ssec:results} (Estimation performance and
baseline comparison): one panel per Thomas parameter, ParamNet (the default
DTM$_5$ / persistence-image / shared-encoder model of
configs/runs/thomas/pi_multik_k5.yaml -- results/thomas/dtm_k5/pi_multik/)
on the shared Thomas test split, identity line drawn, both axes in
log-normalized units (the label_norm space: log-transform then z-score
against the train split's mean/std, i.e. exactly the space test_loss is
computed in).

There is no cached prediction array in results/thomas/dtm_k5/pi_multik --
only the trained weights (model.pt / results.pt["model_state"]) and the
scalar losses (results.json). This script reloads one seed's checkpoint,
rebuilds the identical (deterministic, seed-keyed) data pipeline
PIMultiKExperiment.run() used to produce it -- same train/val/test split,
same train-only-fit label_norm, same train-only-calibrated persistence
imagers (see pi_multik.py's build_pi_tensor) -- and runs a forward pass over
the test split only. No retraining.

Intuition-building rather than publication-rigorous, matching the figtodo:
one trained seed (9371, the first of the ten reported in
Table~\\ref{tab:perparam}), no error bars or per-point uncertainty.

Usage:
    python scripts/make_paramnet_scatter_figure.py

Output:
    figs/paramnet_scatter.png
    figs/paramnet_scatter.pdf
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cloudforger import baselines
from cloudforger.core.splits import train_val_test_indices
from cloudforger.experiments.pi_multik.pi_multik import build_extra, build_pi_tensor, load_multik_split
from cloudforger.experiments.common import prepare_device
from cloudforger.paths import DataPaths, ExplicitTag

SEED = 9371
RESULT_DIR = ROOT / "results" / "thomas" / "dtm_k5" / "pi_multik" / f"seed_{SEED}"
OUT_PNG = ROOT / "figs" / "paramnet_scatter.png"
OUT_PDF = ROOT / "figs" / "paramnet_scatter.pdf"

PALETTE = ["#2a78d6", "#1baf7a", "#eda100"]
GRID_COLOR = "#e1e0d9"
SURFACE_COLOR = "#fcfcfb"

PRETTY = {
    "parent_intensity": r"$\kappa$ (parent intensity)",
    "mean_offspring": r"$\mu$ (mean offspring)",
    "cluster_scale": r"$\sigma$ (cluster scale)",
}


def _style_axis(ax) -> None:
    ax.set_facecolor(SURFACE_COLOR)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#333333")
    ax.grid(True, color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax.tick_params(colors="#333333")


def main() -> None:
    payload = torch.load(RESULT_DIR / "results.pt", map_location="cpu", weights_only=False)
    cfg = dict(payload["config"])
    label_names = list(payload["label_names"])
    k_values = list(cfg["k_values"])
    homology_dims = tuple(cfg.get("homology_dims", (0, 1)))
    resolution = int(cfg.get("resolution", 64))
    sigma_pixels = float(cfg.get("sigma_pixels", 0.5))
    coverage = float(cfg.get("pd_calibration_coverage", 0.95))
    include_entropy = bool(cfg.get("include_entropy", False))
    include_log_n = bool(cfg.get("include_log_n", True))
    device = prepare_device(SEED)

    data_paths = DataPaths("thomas")
    filtrations = [ExplicitTag(f"dtm_m{k}.00") for k in k_values] if False else None
    # pi_multik reads per-k diagrams.pkl directly, under filtration tag "dtm_k{k}".
    from cloudforger.data_generation.filtration import REGISTRY as FILTRATION_REGISTRY

    filtrations = [FILTRATION_REGISTRY.build("dtm", k=k, q=2.0, maxdim=1) for k in k_values]
    image_paths = [data_paths.diagrams([f]) for f in filtrations]
    clouds_path = data_paths.clouds()

    split = load_multik_split(
        k_values, image_paths, clouds_path, tuple(label_names), tag="reload", homology_dims=homology_dims,
    )
    n = len(split["targets"])
    train_idx, val_idx, test_idx = train_val_test_indices(n, SEED)

    label_norm = baselines.vihrs.fit_log_zscore(split["targets"][train_idx])
    targets_std = baselines.vihrs.apply_log_zscore(split["targets"], label_norm).astype(np.float32)
    pi_img, imagers = build_pi_tensor(
        split, k_values, homology_dims=homology_dims, resolution=resolution,
        sigma_pixels=sigma_pixels, coverage=coverage, train_idx=train_idx,
    )
    extra, n_norm, entropy_norms = build_extra(
        split, train_idx, include_entropy=include_entropy, include_log_n=include_log_n,
    )

    # Sanity check: the label_norm/label_names this reload just fit should
    # match what the saved checkpoint reports (same seed -> same train_idx
    # -> same fit).
    saved_norm = payload["label_norm"]
    assert np.allclose(saved_norm["mean"], label_norm["mean"], atol=1e-6), "label_norm mean mismatch on reload"
    assert np.allclose(saved_norm["std"], label_norm["std"], atol=1e-6), "label_norm std mismatch on reload"

    from cloudforger.experiments.pi_multik.pi_multik import PIMultiKExperiment, _resolve_fusion_kwargs

    exp = PIMultiKExperiment(cfg)
    model = exp._build_model(
        in_channels=len(homology_dims),
        embedding_dim=cfg["embedding_dim"],
        n_k=len(k_values),
        n_extra=extra.shape[1],
        n_targets=len(label_names),
        conv_channels=tuple(cfg.get("conv_channels", (32, 64, 128))),
        dropout=cfg.get("dropout", 0.2),
        head_hidden_dims=tuple(cfg.get("head_hidden_dims", (64, 32))),
        head_dropout=cfg.get("head_dropout", 0.1),
        pool_type=str(cfg.get("pool_type", "max")),
        use_coords=bool(cfg.get("coordconv", True)),
        **_resolve_fusion_kwargs(cfg),
    ).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()

    x_img = torch.from_numpy(pi_img[test_idx]).to(device)
    x_extra = torch.from_numpy(extra[test_idx]).to(device)
    y_true = targets_std[test_idx]
    with torch.no_grad():
        y_pred = model(x_img, x_extra).cpu().numpy()

    test_loss = float(np.mean((y_pred - y_true) ** 2))
    print(f"Recomputed test MSE (all targets): {test_loss:.4f} "
          f"(saved: {payload['test_loss']:.4f})")

    fig, axes = plt.subplots(1, len(label_names), figsize=(4.4 * len(label_names), 4.2))
    if len(label_names) == 1:
        axes = [axes]

    for ax, color, name in zip(axes, PALETTE, label_names):
        j = label_names.index(name)
        t, p = y_true[:, j], y_pred[:, j]
        lo, hi = min(t.min(), p.min()), max(t.max(), p.max())
        pad = 0.05 * (hi - lo)
        lo, hi = lo - pad, hi + pad
        ax.plot([lo, hi], [lo, hi], color="#999999", linewidth=1.2, linestyle="--", zorder=1)
        ax.scatter(t, p, s=14, color=color, alpha=0.55, edgecolors="none", zorder=2)
        mse_j = float(np.mean((p - t) ** 2))
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("true (log-normalized)")
        ax.set_ylabel("predicted (log-normalized)")
        ax.set_title(f"{PRETTY.get(name, name)}\n" r"$\mathcal{L}=$" f"{mse_j:.3f}")
        _style_axis(ax)

    fig.suptitle(
        f"ParamNet (DTM$_5$, persistence image), Thomas test split, seed {SEED}",
        y=1.03,
    )
    fig.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    print(f"Wrote {OUT_PNG}")
    print(f"Wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
