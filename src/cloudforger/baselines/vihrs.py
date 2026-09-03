# src/cloudforger/baselines/vihrs.py

"""
Self-contained comparison baseline replicating:

    Vihrs, N. (2022). "Estimating parameters of the Thomas/Neyman-Scott and
    related cluster point process models via a neural network trained on the
    centred Ripley L-function L(r) - r plus the point count n(x)."

Purpose
-------
This repo's own pipeline trains networks on TDA-derived features (persistence
images, Betti curves, ...). This module trains an INDEPENDENT network using
the classical-statistics feature set from the paper above, on the same
clouds/seeds/split convention as the rest of the pipeline, and writes results
in the same results.pt / results.json / model.pt schema so it drops straight
into a comparison alongside every other method (see cloudforger's train/
evaluate CLI scripts).

This module does NOT import or modify anything under cloudforger.experiments/
encoders/models/training -- it is a fully independent implementation of the
paper's own recipe. It DOES reuse
two small, framework-agnostic pieces of this package, on purpose, for a fair
and exact comparison: cloudforger.core.metrics' marginal-metric formulas (so
"same metrics" means literally the same code, not a re-derivation that could
drift), and cloudforger.baselines.mincontrast's minimum-contrast estimator
(the paper's own evaluation protocol calls for a comparison against minimum
contrast, and this repo already has that estimator).

Deliberate implementation choices (the paper is silent, or the environment
differs, on each of these)
----------------------------------------------------------------------------
* Keras -> PyTorch. No TensorFlow/Keras is installed anywhere in this repo's
  environment (and the rest of the codebase is 100% PyTorch); the CNN below
  is a layer-for-layer PyTorch re-implementation (identical filter counts,
  kernel sizes, pooling, dense sizes, loss, optimizer, batch size, epochs)
  rather than adding a second deep-learning framework as a dependency.
* Ripley's isotropic edge correction has no existing implementation anywhere
  in this repo (cloudforger.baselines.mincontrast only has a translation/
  minus-sampling correction) and neither spatstat/rpy2 nor astropy/pointpats
  is installed, so it is implemented directly below (`_isotropic_l_minus_r`)
  from the standard closed-form rectangle formula: for a point at distances
  (b_left, b_right, b_bottom, b_top) to the 4 sides of the window and a pair
  distance r, each side i contributes an excluded arc of half-angle
  alpha_i = arccos(clip(b_i / r, 0, 1)); adjacent sides' excluded arcs
  overlap near a rectangle corner by max(0, alpha_i + alpha_j - pi/2) once
  the circle reaches past that corner. This was independently re-derived and
  cross-checked against the "point at rectangle corner => quarter of the
  circle is inside" limiting case. K_hat uses spatstat's own normalisation,
  Area / (n*(n-1)), confirmed against spatstat's Kest documentation, not
  Area / n^2 (the two differ appreciably for this dataset's smallest clouds,
  which have as few as 4-5 points).
* r-grid: spatstat's Kest/Lest default is rmax = min(side/4, sqrt(1000 / (pi
  * lambda))), evaluated with 513 equally-spaced points. The second
  (density-dependent) term would give every cloud a DIFFERENT r-grid, which
  can't be batched into one fixed-size CNN input, so we fix
  rmax = side/4 = 0.25 for every cloud on the unit square (the geometric
  term, which also dominates for all but the very densest clouds in this
  dataset).
* L(r)-r standardization: Section 2.2 step 2(d) standardizes the L(r)-r
  curve itself with a SINGLE global scalar mean/std pooled across every
  (cloud, r) entry in the training data — "the mean and standard deviation
  were calculated both over all n_train simulations and over all m values
  for r meaning that all values ... were scaled by the same amount" — not a
  per-r-position z-score, which would distort the curve's shape. Implemented
  below as fit_zscore_global/apply_zscore_global, fit once on the current
  seed's train split and frozen for val/test/adversarial.
* Target and n(x) standardization: the paper's step 2(d) is explicit here
  too — plain (non-log) z-score, fit on the training split only and frozen
  for test/adversarial (item 3b). We deliberately deviate and log-transform
  first, matching cloudforger.experiments.base._normalize_labels_by_name:
  this repo's process designs sample parameters on wide, log-uniform ranges
  (e.g. parent_intensity 10-200, cluster_scale 0.005-0.1), where plain
  z-score would leave MSE dominated by the top of the range. Same reasoning
  for n(x): counts span a similarly wide, right-skewed range.
* Paper defaults used as-is, including the training procedure: batch_size=100,
  epochs=20 (Appendix A.2), Adam with lr=0.001 and no weight decay, no
  dropout (Figure 1's architecture has neither), and by default NO
  checkpoint selection -- the paper trains a fixed number of epochs and
  evaluates the final network. Pass checkpoint_best=True to instead select
  the best-val-loss checkpoint before the final test/adversarial evaluation,
  matching the convention every other method in this repo uses (see
  cloudforger.experiments.base._train_and_eval) -- useful when you want
  vihrs's test loss to get the same fair treatment every other method here
  gets, rather than the paper's literal no-checkpointing recipe.

Feature extraction (the expensive, O(points^2) part) is cached to a sibling
.npz file next to each source pickle so repeated seeds/re-runs don't
recompute it.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from scipy.spatial.distance import cdist
from torch.utils.data import DataLoader, TensorDataset

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..core import metrics as classical_evaluate  # marginal/paired metric formulas, shared with every method
from ..core.io import intersect_seeds
from ..core.records import load_diagram_bundle
from ..core.splits import train_val_test_indices
from ..provenance import append_ledger_entry, provenance_stamp
from . import mincontrast as mc

PROJECT_ROOT = Path(__file__).resolve().parents[3]

# ══════════════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════════════

# Must match cloudforger.evaluate's DEFAULT_SEEDS (duplicated here, not
# imported, to keep this module independent of the rest of the pipeline).
SEEDS: list[int] = [
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

DEFAULT_LABEL_NAMES: tuple[str, ...] = ("parent_intensity", "mean_offspring", "cluster_scale")

DEFAULT_CLOUDS = PROJECT_ROOT / "data" / "thomas" / "clouds.pkl"
DEFAULT_ADVERSARIAL = PROJECT_ROOT / "data" / "thomas" / "adversarial_clouds.pkl"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results" / "thomas" / "raw" / "vihrs"

R_MAX = 0.25  # spatstat Kest/Lest geometric default (window side / 4) for the unit square
N_R = 513  # spatstat Kest/Lest default resolution (512 intervals + 1 points)

MC_SEED = 420937  # for reproducibility of the bonus mincontrast comparison


# ══════════════════════════════════════════════════════════════════════════════
# Feature extraction: Ripley's isotropic-corrected L(r) - r, plus n(x)
# ══════════════════════════════════════════════════════════════════════════════

def default_r_grid(r_max: float = R_MAX, n_r: int = N_R) -> np.ndarray:
    return np.linspace(0.0, r_max, n_r)


def _isotropic_l_minus_r(
    points: np.ndarray,
    region_low: Any,
    region_high: Any,
    r_grid: np.ndarray,
) -> np.ndarray:
    """Ripley's isotropic-edge-corrected L(r) - r on a fixed r_grid.

        K_hat(r) = Area / (n*(n-1)) * sum_{i != j, d_ij <= r} w(i, d_ij)
        L_hat(r) = sqrt(K_hat(r) / pi)                      (2D: omega_d = pi)

    w(i, d) is point i's isotropic correction weight at radius d: 2*pi
    divided by the fraction of the circle of radius d centred at point i that
    lies within the (rectangular) window, from i's distances to the 4 sides.
    See the module docstring for the derivation.
    """
    n = len(points)
    if n < 2:
        return -r_grid.copy()

    low = np.asarray(region_low, dtype=np.float64)
    high = np.asarray(region_high, dtype=np.float64)
    area = float(np.prod(high - low))

    b_left = (points[:, 0] - low[0]).astype(np.float32)
    b_right = (high[0] - points[:, 0]).astype(np.float32)
    b_bottom = (points[:, 1] - low[1]).astype(np.float32)
    b_top = (high[1] - points[:, 1]).astype(np.float32)

    d = cdist(points, points).astype(np.float32)
    np.fill_diagonal(d, np.inf)

    r_max = float(r_grid[-1])
    mask = d <= r_max
    if not mask.any():
        return -r_grid.copy()

    rows, cols = np.nonzero(mask)
    d_pairs = d[rows, cols]

    half_pi = np.float32(np.pi / 2)
    two_pi = np.float32(2 * np.pi)

    def _alpha(boundary: np.ndarray) -> np.ndarray:
        return np.arccos(np.clip(boundary[rows] / d_pairs, 0.0, 1.0))

    a_l, a_r, a_b, a_t = _alpha(b_left), _alpha(b_right), _alpha(b_bottom), _alpha(b_top)

    gamma_rt = np.clip(a_r + a_t - half_pi, 0.0, None)
    gamma_tl = np.clip(a_t + a_l - half_pi, 0.0, None)
    gamma_lb = np.clip(a_l + a_b - half_pi, 0.0, None)
    gamma_br = np.clip(a_b + a_r - half_pi, 0.0, None)

    theta_excluded = 2.0 * (a_l + a_r + a_b + a_t) - (gamma_rt + gamma_tl + gamma_lb + gamma_br)
    weight = two_pi / np.clip(two_pi - theta_excluded, 1e-6, None)

    order = np.argsort(d_pairs)
    d_sorted = d_pairs[order]
    cumw = np.concatenate([[0.0], np.cumsum(weight[order], dtype=np.float64)])

    idx = np.searchsorted(d_sorted, r_grid, side="right")
    k_hat = (area / (n * (n - 1))) * cumw[idx]
    l_hat = np.sqrt(np.clip(k_hat, 0.0, None) / np.pi)
    return l_hat - r_grid


def load_cloud_records(path: Path) -> list[dict[str, Any]]:
    with open(path, "rb") as f:
        data = pickle.load(f)
    if not isinstance(data, list):
        data = [data]
    return data


def _load_pickle(path: Path) -> Any:
    """Plain unpickle -- unlike load_cloud_records, does NOT wrap a non-list
    payload into a one-element list (the classification clouds.pkl is a
    {"clouds": [...], "labels": [...]} dict, consumed as-is)."""
    with open(path, "rb") as f:
        return pickle.load(f)


def extract_features(
    records: list[dict[str, Any]],
    r_grid: np.ndarray,
    *,
    label_names: tuple[str, ...] = DEFAULT_LABEL_NAMES,
    tag: str = "",
) -> dict[str, np.ndarray]:
    n = len(records)
    m = len(r_grid)
    l_minus_r = np.empty((n, m), dtype=np.float32)
    n_points = np.empty(n, dtype=np.float64)
    targets = np.empty((n, len(label_names)), dtype=np.float64)
    cloud_seeds = np.empty(n, dtype=np.int64)

    t0 = time.perf_counter()
    for i, rec in enumerate(records):
        points = np.asarray(rec["points"], dtype=np.float64)
        region = rec.get("region", {})
        low = region.get("low", [0.0, 0.0])
        high = region.get("high", [1.0, 1.0])

        l_minus_r[i] = _isotropic_l_minus_r(points, low, high, r_grid)
        n_points[i] = len(points)
        targets[i] = [rec["params"][name] for name in label_names]
        cloud_seeds[i] = rec.get("seed", i)

        if (i + 1) % 250 == 0 or i + 1 == n:
            elapsed = time.perf_counter() - t0
            print(f"  [{tag}] featurized {i + 1}/{n} clouds ({elapsed:.1f}s elapsed)", flush=True)

    return {
        "l_minus_r": l_minus_r,
        "n_points": n_points,
        "targets": targets,
        "cloud_seeds": cloud_seeds,
    }


def get_features(
    records: list[dict[str, Any]],
    r_grid: np.ndarray,
    *,
    label_names: tuple[str, ...],
    cache_path: Path | None,
    force: bool,
    tag: str,
) -> dict[str, np.ndarray]:
    if cache_path is not None and cache_path.exists() and not force:
        cached = np.load(cache_path)
        if cached["r_grid"].shape == r_grid.shape and np.allclose(cached["r_grid"], r_grid):
            print(f"  [{tag}] loaded cached features <- {cache_path}")
            return {k: cached[k] for k in ("l_minus_r", "n_points", "targets", "cloud_seeds")}
        print(f"  [{tag}] cache at {cache_path} used a different r_grid; recomputing.")

    features = extract_features(records, r_grid, label_names=label_names, tag=tag)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, r_grid=r_grid, **features)
        print(f"  [{tag}] cached features -> {cache_path}")

    return features


# ══════════════════════════════════════════════════════════════════════════════
# Label / input normalization (mirrors cloudforger.experiments.base)
# ══════════════════════════════════════════════════════════════════════════════

def fit_log_zscore(values: np.ndarray) -> dict[str, np.ndarray]:
    """Mirrors _normalize_labels_by_name in cloudforger.experiments.base:
    log-transform then z-score, fit on the given (training) values."""
    if np.any(values <= 0):
        raise ValueError("Cannot log-transform non-positive values.")
    log_values = np.log(values)
    mean = log_values.mean(axis=0)
    std = log_values.std(axis=0)
    std = np.where(std == 0, 1.0, std)
    return {"mean": mean, "std": std}


def apply_log_zscore(values: np.ndarray, norm: dict[str, np.ndarray]) -> np.ndarray:
    return (np.log(values) - norm["mean"]) / norm["std"]


def invert_log_zscore(standardized: np.ndarray, norm: dict[str, np.ndarray]) -> np.ndarray:
    return np.exp(standardized * norm["std"] + norm["mean"])


def fit_zscore_global(values: np.ndarray) -> dict[str, float]:
    """Single scalar mean/std pooled across every entry of `values` (every
    cloud AND every r together), matching Vihrs (2022) Section 2.2 step
    2(d): "the mean and standard deviation were calculated both over all
    n_train simulations and over all m values for r meaning that all values
    ... were scaled by the same amount." Deliberately NOT a per-r-position
    z-score, which would rescale each point of the curve differently and
    distort its shape."""
    mean = float(values.mean())
    std = float(values.std())
    return {"mean": mean, "std": std if std else 1.0}


def apply_zscore_global(values: np.ndarray, norm: dict[str, float]) -> np.ndarray:
    return (values - norm["mean"]) / norm["std"]


# ══════════════════════════════════════════════════════════════════════════════
# Network (PyTorch translation of the paper's Keras architecture)
# ══════════════════════════════════════════════════════════════════════════════

class VihrsCNN(nn.Module):
    """Two-branch network: Conv1D(64,7)-Pool(5)-Conv1D(64,7)-Pool(5)-
    Conv1D(64,7)-Flatten on the L(r)-r sequence, concatenated with the
    (standardized) scalar n(x), then Dense(64)-Dense(32)-Dense(k, linear)."""

    def __init__(self, seq_len: int, n_targets: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7), nn.ReLU(), nn.MaxPool1d(5),
            nn.Conv1d(64, 64, kernel_size=7), nn.ReLU(), nn.MaxPool1d(5),
            nn.Conv1d(64, 64, kernel_size=7), nn.ReLU(),
        )
        with torch.no_grad():
            flat_dim = self.conv(torch.zeros(1, 1, seq_len)).flatten(1).shape[1]
        self.merge_dim = flat_dim + 1
        # No dropout: Figure 1's architecture has none, and the paper's own
        # training procedure (fixed 20 epochs, no early stopping) doesn't
        # rely on it for regularization either.
        self.head = nn.Sequential(
            nn.Linear(self.merge_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, n_targets),
        )

    def forward(self, sequence: torch.Tensor, n_scalar: torch.Tensor) -> torch.Tensor:
        x = self.conv(sequence.unsqueeze(1)).flatten(start_dim=1)
        merged = torch.cat([x, n_scalar.unsqueeze(1)], dim=1)
        return self.head(merged)


# ══════════════════════════════════════════════════════════════════════════════
# Training / evaluation loops
# ══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, optimizer, loss_fn, device) -> float:
    model.train()
    total, count = 0.0, 0
    for seq, n_scalar, y in loader:
        seq, n_scalar, y = seq.to(device), n_scalar.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(seq, n_scalar)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(y)
        count += len(y)
    return total / count


@torch.no_grad()
def evaluate_loss(model, loader, loss_fn, device) -> float:
    model.eval()
    total, count = 0.0, 0
    for seq, n_scalar, y in loader:
        seq, n_scalar, y = seq.to(device), n_scalar.to(device), y.to(device)
        pred = model(seq, n_scalar)
        loss = loss_fn(pred, y)
        total += loss.item() * len(y)
        count += len(y)
    return total / count


@torch.no_grad()
def evaluate_per_target_loss(model, loader, device) -> np.ndarray:
    """Per-target (standardized-space) MSE -- same quantity evaluate_loss
    averages down to one scalar, kept per-column so it's directly comparable
    to cloudforger.training.train.evaluate_per_target's output for the TDA methods
    (same standardized log-zscore space, same label_names order)."""
    model.eval()
    total_sq_err, count = None, 0
    for seq, n_scalar, y in loader:
        seq, n_scalar, y = seq.to(device), n_scalar.to(device), y.to(device)
        pred = model(seq, n_scalar)
        sq_err = (pred - y).pow(2).sum(dim=0)
        total_sq_err = sq_err if total_sq_err is None else total_sq_err + sq_err
        count += len(y)
    return (total_sq_err / count).cpu().numpy()


@torch.no_grad()
def predict_all(model, loader, device) -> np.ndarray:
    model.eval()
    preds = []
    for seq, n_scalar, _y in loader:
        seq, n_scalar = seq.to(device), n_scalar.to(device)
        preds.append(model(seq, n_scalar).cpu().numpy())
    return np.concatenate(preds, axis=0)


# ══════════════════════════════════════════════════════════════════════════════
# Paper evaluation protocol: scatter / error-by-true-value / error-vs-n plots
# ══════════════════════════════════════════════════════════════════════════════

def _make_paper_plots(
    plot_dir: Path,
    true_raw: np.ndarray,
    pred_raw: np.ndarray,
    n_points: np.ndarray,
    *,
    label_names: tuple[str, ...],
) -> None:
    plot_dir.mkdir(parents=True, exist_ok=True)

    for j, name in enumerate(label_names):
        t, p = true_raw[:, j], pred_raw[:, j]
        error = p - t

        fig, ax = plt.subplots(figsize=(5, 5))
        lo, hi = min(t.min(), p.min()), max(t.max(), p.max())
        ax.plot([lo, hi], [lo, hi], color="#999999", linewidth=1, linestyle="--")
        ax.scatter(t, p, s=10, alpha=0.5, color="#2a78d6")
        ax.set_xlabel(f"true {name}")
        ax.set_ylabel(f"estimated {name}")
        ax.set_title(f"{name}: estimate vs true")
        fig.tight_layout()
        fig.savefig(plot_dir / f"scatter_{name}.png", dpi=150)
        plt.close(fig)

        # Bin by the actual simulated grid values (params are drawn from a
        # small discrete logspace grid), not arbitrary quantile cuts.
        unique_vals = np.unique(t)
        boxes = [error[t == v] for v in unique_vals]
        labels = [f"{v:.3g}" for v in unique_vals]

        fig, ax = plt.subplots(figsize=(max(6, 0.8 * len(unique_vals)), 5))
        ax.axhline(0, color="#999999", linewidth=1)
        ax.boxplot(boxes, tick_labels=labels)
        ax.set_xlabel(f"true {name}")
        ax.set_ylabel("error (estimate - true)")
        ax.set_title(f"{name}: error by true value")
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        fig.savefig(plot_dir / f"error_boxplot_{name}.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.axhline(0, color="#999999", linewidth=1)
        ax.scatter(n_points, error, s=10, alpha=0.5, color="#eda100")
        ax.set_xscale("log")
        ax.set_xlabel("number of points n(x)")
        ax.set_ylabel("error (estimate - true)")
        ax.set_title(f"{name}: error vs n(x)")
        fig.tight_layout()
        fig.savefig(plot_dir / f"error_vs_npoints_{name}.png", dpi=150)
        plt.close(fig)


def _mincontrast_comparison(
    plot_dir: Path,
    test_records: list[dict[str, Any]],
    test_true_raw: np.ndarray,
    nn_pred_raw: np.ndarray,
    *,
    label_names: tuple[str, ...],
    mc_cache: dict[Any, Any],
    mc_rng: np.random.Generator,
) -> dict[str, Any] | None:
    """Bonus comparison against minimum contrast, per the paper's own
    evaluation protocol (section 4). Best-effort: any failure here must not
    break the main NN training/saving, which has already completed."""
    mc_pred = np.full_like(test_true_raw, np.nan)
    n_fit = 0
    for i, rec in enumerate(test_records):
        cloud_seed = rec.get("seed", i)
        if cloud_seed not in mc_cache:
            points = np.asarray(rec["points"], dtype=float)
            if len(points) < 5:
                mc_cache[cloud_seed] = None
            else:
                try:
                    mc_cache[cloud_seed] = mc.fit_multistart(points, rng=mc_rng)
                except Exception as exc:  # noqa: BLE001 — best-effort bonus analysis
                    warnings.warn(f"mincontrast fit failed for cloud seed={cloud_seed}: {exc!r}")
                    mc_cache[cloud_seed] = None
            n_fit += 1
        result = mc_cache[cloud_seed]
        if result is not None:
            for j, name in enumerate(label_names):
                mc_pred[i, j] = result.get(name, np.nan)

    print(f"  [mincontrast] {n_fit} new fits this seed ({len(mc_cache)} unique clouds fit so far)")

    comparison = {}
    for j, name in enumerate(label_names):
        comparison[name] = classical_evaluate.compute_paired_metrics(
            test_true_raw[:, j],
            mc_pred[:, j],
            nn_pred_raw[:, j],
            first_name="mincontrast",
            second_name="vihrs",
            rng=np.random.default_rng(20240707),
        )

    plot_dir.mkdir(parents=True, exist_ok=True)
    for j, name in enumerate(label_names):
        t = test_true_raw[:, j]
        finite_mc = np.isfinite(mc_pred[:, j])
        fig, ax = plt.subplots(figsize=(5, 5))
        all_vals = np.concatenate([t, mc_pred[finite_mc, j], nn_pred_raw[:, j]])
        lo, hi = float(np.min(all_vals)), float(np.max(all_vals))
        ax.plot([lo, hi], [lo, hi], color="#999999", linewidth=1, linestyle="--")
        ax.scatter(t[finite_mc], mc_pred[finite_mc, j], s=10, alpha=0.4, color="#e34948", label="mincontrast")
        ax.scatter(t, nn_pred_raw[:, j], s=10, alpha=0.4, color="#2a78d6", label="vihrs (NN)")
        ax.set_xlabel(f"true {name}")
        ax.set_ylabel(f"estimated {name}")
        ax.set_title(f"{name}: NN vs minimum contrast")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(plot_dir / f"mincontrast_comparison_{name}.png", dpi=150)
        plt.close(fig)

    return comparison


# ══════════════════════════════════════════════════════════════════════════════
# Per-seed run
# ══════════════════════════════════════════════════════════════════════════════

def run_one_seed(
    seed: int,
    *,
    train_records: list[dict[str, Any]],
    train_features: dict[str, np.ndarray],
    adversarial_features: dict[str, np.ndarray] | None,
    adversarial_path: Path | None,
    r_grid: np.ndarray,
    output_root: Path,
    n_epochs: int,
    batch_size: int,
    lr: float,
    label_names: tuple[str, ...] = DEFAULT_LABEL_NAMES,
    checkpoint_best: bool = False,
    early_stopping_patience: int | None = None,
    device_pref: str | None = None,
    skip_mincontrast: bool = False,
    mc_cache: dict[Any, Any] | None = None,
    mc_rng: np.random.Generator | None = None,
    results_root: Path | None = None,
    run_tag: str | None = None,
) -> dict[str, Any]:
    output_dir = output_root / f"seed_{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    mc_cache = {} if mc_cache is None else mc_cache
    mc_rng = mc_rng or np.random.default_rng(MC_SEED)

    torch.manual_seed(seed)
    if device_pref:
        device = device_pref
    elif torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    n = len(train_features["targets"])
    train_idx, val_idx, test_idx = train_val_test_indices(n, seed)

    # Fit every normalization statistic on the train split ONLY, then apply
    # the frozen stats to val/test/adversarial below.
    label_norm = fit_log_zscore(train_features["targets"][train_idx])
    n_norm = fit_log_zscore(train_features["n_points"][train_idx])
    lr_norm = fit_zscore_global(train_features["l_minus_r"][train_idx])

    targets_std = apply_log_zscore(train_features["targets"], label_norm).astype(np.float32)
    n_std = apply_log_zscore(train_features["n_points"], n_norm).astype(np.float32)
    seq = apply_zscore_global(train_features["l_minus_r"], lr_norm).astype(np.float32)

    def _loader(idx: np.ndarray, shuffle: bool) -> DataLoader:
        ds = TensorDataset(
            torch.from_numpy(seq[idx]),
            torch.from_numpy(n_std[idx]),
            torch.from_numpy(targets_std[idx]),
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    train_loader = _loader(train_idx, True)
    val_loader = _loader(val_idx, False)
    test_loader = _loader(test_idx, False)

    model = VihrsCNN(seq_len=len(r_grid), n_targets=len(label_names)).to(device)
    # No weight_decay: Adam is used with its plain defaults per the paper.
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # By default (checkpoint_best=False) val_loss is tracked purely as a
    # diagnostic curve, matching the paper's fixed-epoch-count, no-early-
    # stopping recipe -- see the module docstring. With checkpoint_best=True,
    # the best-val-loss state is reloaded before the final evaluation,
    # matching every other method in this repo's convention instead.
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
    best_val_loss, best_state, best_epoch = float("inf"), None, 0
    epochs_no_improve = 0

    for epoch in range(1, n_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss = evaluate_loss(model, val_loader, loss_fn, device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        if checkpoint_best and val_loss < best_val_loss:
            best_val_loss, best_epoch = val_loss, epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        elif checkpoint_best:
            epochs_no_improve += 1
        print(f"[vihrs seed={seed}] epoch {epoch:3d} | train {train_loss:.4f} | val {val_loss:.4f}")
        if checkpoint_best and early_stopping_patience is not None and epochs_no_improve >= early_stopping_patience:
            print(f"[vihrs seed={seed}] early stopping at epoch {epoch} (no val improvement for {early_stopping_patience} epochs)")
            break

    if checkpoint_best:
        model.load_state_dict(best_state)
        print(f"[vihrs seed={seed}] best checkpoint: epoch {best_epoch}/{n_epochs}, val_loss {best_val_loss:.4f}")
        final_state = best_state
    else:
        final_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    test_loss = evaluate_loss(model, test_loader, loss_fn, device)
    test_loss_per_target = dict(zip(label_names, evaluate_per_target_loss(model, test_loader, device).tolist()))
    print(f"\n[vihrs seed={seed}] test loss {test_loss:.4f}")

    adversarial_loss = None
    adversarial_loss_per_target = None
    if adversarial_features is not None:
        adv_targets_std = apply_log_zscore(adversarial_features["targets"], label_norm).astype(np.float32)
        adv_n_std = apply_log_zscore(adversarial_features["n_points"], n_norm).astype(np.float32)
        adv_seq = apply_zscore_global(adversarial_features["l_minus_r"], lr_norm).astype(np.float32)
        adv_ds = TensorDataset(torch.from_numpy(adv_seq), torch.from_numpy(adv_n_std), torch.from_numpy(adv_targets_std))
        adv_loader = DataLoader(adv_ds, batch_size=batch_size, shuffle=False)
        adversarial_loss = evaluate_loss(model, adv_loader, loss_fn, device)
        adversarial_loss_per_target = dict(
            zip(label_names, evaluate_per_target_loss(model, adv_loader, device).tolist())
        )
        print(f"[vihrs seed={seed}] adversarial loss {adversarial_loss:.4f}")

    # ---- save, matching this repo's shared results.pt / results.json / model.pt schema ----
    cfg = {
        "task": "params",
        "method": "vihrs",
        "paper": "Vihrs (2022)",
        "batch_size": batch_size,
        "n_epochs": n_epochs,
        "lr": lr,
        "embedding_dim": model.merge_dim,
        "seed": seed,
        "checkpoint_best": checkpoint_best,
        "best_epoch": best_epoch if checkpoint_best else n_epochs,
        "r_max": float(r_grid[-1]),
        "n_r": len(r_grid),
        "output_dir": str(output_dir),
        "results_root": str(results_root) if results_root else None,
        "run_tag": run_tag,
    }

    # Provenance: same stamp shape cloudforger.experiments.common.save_results
    # uses, so results.json is self-sufficient (commit/dirty/when/run_tag)
    # without loading the torch file, and this run gets a row in the shared
    # results/experiments.jsonl ledger regardless of which code path wrote it.
    stamp = provenance_stamp(run_tag=run_tag)

    torch.save(
        {
            "model_state": final_state,
            "history": history,
            "config": cfg,
            "test_loss": test_loss,
            "test_loss_per_target": test_loss_per_target,
            "label_names": list(label_names),
            "label_norm": label_norm,
            "label_log_mean": label_norm["mean"],
            "label_log_std": label_norm["std"],
            "label_transforms": ["log"] * len(label_names),
            "n_points_norm": n_norm,
            "lr_norm": lr_norm,
            "adversarial_loss": adversarial_loss,
            "adversarial_loss_per_target": adversarial_loss_per_target,
            "adversarial_path": str(adversarial_path) if adversarial_path else None,
            **stamp,
        },
        output_dir / "results.pt",
    )
    torch.save(model.cpu(), output_dir / "model.pt")
    model.to(device)

    json_payload: dict[str, Any] = {
        "task": "params",
        "method": "vihrs",
        "paper": "Vihrs (2022)",
        "test_loss": test_loss,
        "test_loss_per_target": test_loss_per_target,
        "seed": seed,
        **stamp,
        "config": cfg,
    }
    if checkpoint_best:
        json_payload["best_epoch"] = best_epoch
        json_payload["best_val_loss"] = best_val_loss
    if adversarial_loss is not None:
        json_payload["adversarial_loss"] = adversarial_loss
        json_payload["adversarial_loss_per_target"] = adversarial_loss_per_target
        json_payload["adversarial_path"] = str(adversarial_path)
    with open(output_dir / "results.json", "w") as f:
        json.dump(json_payload, f, indent=2, default=str)

    append_ledger_entry(
        output_dir=output_dir,
        results_root=results_root,
        method="vihrs",
        seed=seed,
        test_loss=test_loss,
        adversarial_loss=adversarial_loss,
        stamp=stamp,
    )

    # ---- same marginal metrics as cloudforger.core.metrics, on de-standardized test predictions ----
    test_pred_std = predict_all(model, test_loader, device)
    test_pred_raw = invert_log_zscore(test_pred_std, label_norm)
    test_true_raw = train_features["targets"][test_idx]
    test_n_points = train_features["n_points"][test_idx]

    marginal_metrics = {
        name: classical_evaluate.compute_marginal_metrics(test_true_raw[:, j], test_pred_raw[:, j])
        for j, name in enumerate(label_names)
    }

    result: dict[str, Any] = {
        "seed": seed,
        "test_loss": test_loss,
        "test_loss_per_target": test_loss_per_target,
        "adversarial_loss": adversarial_loss,
        "adversarial_loss_per_target": adversarial_loss_per_target,
        "marginal_metrics": marginal_metrics,
    }

    if not skip_mincontrast:
        try:
            test_records = [train_records[i] for i in test_idx]
            mc_comparison = _mincontrast_comparison(
                output_dir / "plots",
                test_records,
                test_true_raw,
                test_pred_raw,
                label_names=label_names,
                mc_cache=mc_cache,
                mc_rng=mc_rng,
            )
            if mc_comparison is not None:
                marginal_metrics = {**marginal_metrics, "_vs_mincontrast": mc_comparison}
                result["mincontrast_comparison"] = mc_comparison
        except Exception as exc:  # noqa: BLE001 — bonus analysis must not break the core run
            warnings.warn(f"[vihrs seed={seed}] mincontrast comparison failed: {exc!r}")

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(marginal_metrics, f, indent=2, default=classical_evaluate.json_default)

    _make_paper_plots(
        output_dir / "plots", test_true_raw, test_pred_raw, test_n_points, label_names=label_names,
    )

    return result


# ══════════════════════════════════════════════════════════════════════════════
# Data preparation (called once, reused across every seed)
# ══════════════════════════════════════════════════════════════════════════════

def prepare_data(
    clouds_path: Path,
    adversarial_path: Path | None = None,
    *,
    label_names: tuple[str, ...] | None = DEFAULT_LABEL_NAMES,
    r_max: float = R_MAX,
    n_r: int = N_R,
    cache_dir: Path | None = None,
    force: bool = False,
    max_clouds: int | None = None,
) -> dict[str, Any]:
    """Load clouds_path (+ adversarial_path, if given) and extract/cache the
    L(r)-r + n(x) feature set that run_one_seed needs.

    label_names=None adapts to whatever labels this process's clouds
    actually carry (every key in the first record's `params`, in that
    record's own order) instead of assuming a fixed set -- lets callers
    driven by RunConfig.target_label_names (None when the YAML doesn't pin
    an explicit target list) pass that straight through. The resolved tuple
    is echoed back in the return dict so callers know what was used."""
    r_grid = default_r_grid(r_max, n_r)
    cache_dir = cache_dir or clouds_path.parent

    print(f"Loading clouds from {clouds_path} ...")
    train_records = load_cloud_records(clouds_path)
    if max_clouds is not None:
        train_records = train_records[:max_clouds]
    print(f"  {len(train_records)} clouds loaded.")

    if label_names is None:
        label_names = tuple(train_records[0]["params"].keys())
        print(f"  no label_names given -- adapting to this process's own params: {label_names}")

    train_cache = cache_dir / (clouds_path.stem + ".lfunc_cache.npz")
    train_features = get_features(
        train_records, r_grid, label_names=label_names, cache_path=train_cache, force=force, tag="train_test"
    )

    adversarial_features: dict[str, np.ndarray] | None = None
    if adversarial_path is not None and adversarial_path.exists():
        print(f"Loading adversarial clouds from {adversarial_path} ...")
        adv_records = load_cloud_records(adversarial_path)
        if max_clouds is not None:
            adv_records = adv_records[:max_clouds]
        print(f"  {len(adv_records)} adversarial clouds loaded.")
        adv_cache = cache_dir / (adversarial_path.stem + ".lfunc_cache.npz")
        adversarial_features = get_features(
            adv_records, r_grid, label_names=label_names, cache_path=adv_cache, force=force, tag="adversarial"
        )
    else:
        print(f"No adversarial clouds found at {adversarial_path}; skipping adversarial evaluation.")

    return {
        "train_records": train_records,
        "train_features": train_features,
        "adversarial_features": adversarial_features,
        "adversarial_path": adversarial_path if adversarial_features is not None else None,
        "r_grid": r_grid,
        "label_names": label_names,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Classification variant (task="classify")
# ══════════════════════════════════════════════════════════════════════════════
#
# The paper is a *parameter-estimation* method; this section reuses its exact
# feature set (Ripley's isotropic-corrected L(r)-r + n(x)) and network
# (VihrsCNN, unchanged -- its final Linear just emits `n_classes` logits
# instead of `n_targets` regression outputs) to instead classify the
# point-process family, so it drops in as a comparison arm alongside the
# pi_multik classification run (experiments/pi_multik/pi_multik.py, task=
# "classify"). Differences from the regression path above:
#
#   * loss: CrossEntropyLoss; metric: accuracy + per-class recall (mirrors
#     _per_class_accuracy in pi_multik.py). No log-zscore on the target --
#     class indices pass through verbatim.
#   * split parity: the classification data ships as a
#     {"clouds": [PointCloud], "labels": int[N], ...} bundle (NOT the
#     list-of-records shape prepare_data() consumes), and pi_multik's
#     train/val/test split is train_val_test_indices(n, seed) over the seeds
#     COMMON to all per-k diagram bundles, in k_values[0]'s order. To train
#     and test on precisely the same clouds/split, this reproduces that
#     intersection from the diagram bundles' `seeds` arrays (the diagrams
#     themselves are never touched) and aligns clouds.pkl to it by seed.
#   * always checkpoints + early-stops on val loss (like every trained
#     method in this repo, and like pi_multik's classify path) -- the
#     paper's fixed-epoch, no-checkpoint recipe is regression-only.
#   * results.json carries the same classification keys pi_multik writes:
#     class_names / test_accuracy / test_accuracy_per_class /
#     adversarial_accuracy / adversarial_accuracy_per_class.


CLASSIFY_LFUNC_CACHE = "clouds.lfunc_classify_cache.npz"
CLASSIFY_ADV_LFUNC_CACHE = "adversarial_clouds.lfunc_classify_cache.npz"


def _load_classification_alignment(
    diagram_paths: list[Path], k_values: list[int], tag: str
) -> dict[str, Any]:
    """Reproduce load_multik_split's seed alignment for task="classify"
    without building persistence images: load each k's diagram bundle,
    N-way-intersect their `seeds` arrays (kept in k_values[0]'s order, same
    as intersect_seeds does), and carry the bundle's 1-D integer class
    labels through verbatim. Returns the common seed vector, the aligned
    class-index targets, the class-name list, and the seed_offset used to
    decode (class_index, original_seed) pairs back to clouds.pkl."""
    bundles = []
    for k, path in zip(k_values, diagram_paths):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"[{tag}] classification vihrs needs the per-k diagram bundles only to reproduce "
                f"pi_multik's train/val/test split; missing k={k} bundle at {path}."
            )
        bundles.append(load_diagram_bundle(path))

    seed_arrays = [np.asarray(b["seeds"]) for b in bundles]
    idx_per_k, common_seeds = intersect_seeds(seed_arrays)
    per_k_totals = ", ".join(f"{k}: {len(s)}" for k, s in zip(k_values, seed_arrays))
    print(f"  [{tag}] {len(common_seeds)} clouds common to all k in {k_values} (per-k totals: {{{per_k_totals}}}).")

    class_names = list(bundles[0]["label_names"])
    targets = np.asarray(bundles[0]["labels"]).reshape(-1)[idx_per_k[0]].astype(np.int64)
    for k, b, idx in zip(k_values[1:], bundles[1:], idx_per_k[1:]):
        targets_k = np.asarray(b["labels"]).reshape(-1)[idx].astype(np.int64)
        if not np.array_equal(targets, targets_k):
            raise AssertionError(
                f"[{tag}] class-label mismatch between k={k_values[0]} and k={k} after seed alignment -- alignment bug."
            )
    seed_offset = int(bundles[0].get("config", {}).get("seed_offset", 0))
    return {
        "common_seeds": common_seeds,
        "targets": targets,
        "class_names": class_names,
        "seed_offset": seed_offset,
    }


def extract_cloud_features(clouds: list[Any], r_grid: np.ndarray, *, tag: str = "") -> dict[str, np.ndarray]:
    """L(r)-r + n(x) for a list of PointCloud objects (or dicts), in the
    given order. No targets -- the class labels come from the diagram
    bundles (see _load_classification_alignment), joined by seed afterwards."""
    n = len(clouds)
    m = len(r_grid)
    l_minus_r = np.empty((n, m), dtype=np.float32)
    n_points = np.empty(n, dtype=np.float64)
    cloud_seeds = np.empty(n, dtype=np.int64)

    t0 = time.perf_counter()
    for i, c in enumerate(clouds):
        is_dict = isinstance(c, dict)
        points = np.asarray(c["points"] if is_dict else c.points, dtype=np.float64)
        region = c.get("region") if is_dict else getattr(c, "region", None)
        if region is None:
            low, high = [0.0, 0.0], [1.0, 1.0]
        elif isinstance(region, dict):
            low, high = region.get("low", [0.0, 0.0]), region.get("high", [1.0, 1.0])
        else:  # cloudforger.core.region.Box
            low, high = region.low, region.high

        l_minus_r[i] = _isotropic_l_minus_r(points, low, high, r_grid)
        n_points[i] = len(points)
        seed = c.get("seed", i) if is_dict else getattr(c, "seed", i)
        cloud_seeds[i] = i if seed is None else int(seed)

        if (i + 1) % 5000 == 0 or i + 1 == n:
            print(f"  [{tag}] featurized {i + 1}/{n} clouds ({time.perf_counter() - t0:.1f}s elapsed)", flush=True)

    return {"l_minus_r": l_minus_r, "n_points": n_points, "cloud_seeds": cloud_seeds}


def get_cloud_features(
    clouds: list[Any],
    r_grid: np.ndarray,
    *,
    cache_path: Path | None,
    force: bool,
    tag: str,
) -> dict[str, np.ndarray]:
    """extract_cloud_features + a sibling .npz cache. Cache content is
    split-independent (whole file, file order), so every seed reuses it.
    The write is atomic (tmp file + os.replace) so concurrent SLURM-array
    tasks that each recompute it can't tear each other's file."""
    if cache_path is not None and cache_path.exists() and not force:
        try:
            cached = np.load(cache_path)
            if cached["r_grid"].shape == r_grid.shape and np.allclose(cached["r_grid"], r_grid) \
                    and len(cached["cloud_seeds"]) == len(clouds):
                print(f"  [{tag}] loaded cached L(r)-r features <- {cache_path}")
                return {k: cached[k] for k in ("l_minus_r", "n_points", "cloud_seeds")}
            print(f"  [{tag}] cache at {cache_path} is stale (r_grid or cloud count differs); recomputing.")
        except (OSError, KeyError, ValueError, EOFError) as exc:
            print(f"  [{tag}] cache at {cache_path} unreadable ({exc!r}); recomputing.")

    features = extract_cloud_features(clouds, r_grid, tag=tag)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        # tmp MUST end in .npz: np.savez_compressed appends ".npz" to any path
        # that lacks it, so a bare ".tmp.<pid>" name lands at ".tmp.<pid>.npz"
        # and the os.replace below then fails on a path that was never written.
        tmp = cache_path.with_name(f"{cache_path.name}.tmp.{os.getpid()}.npz")
        np.savez_compressed(tmp, r_grid=r_grid, **features)
        os.replace(tmp, cache_path)
        print(f"  [{tag}] cached L(r)-r features -> {cache_path}")

    return features


def _classification_features_for_alignment(
    clouds_path: Path,
    align: dict[str, Any],
    r_grid: np.ndarray,
    cache_dir: Path,
    force: bool,
    *,
    tag: str,
    cache_name: str,
) -> dict[str, np.ndarray]:
    """L(r)-r + n(x) + class targets for exactly the clouds in
    align["common_seeds"], in that order. Mirrors _classification_n_points
    in pi_multik.py: prefer a direct {global seed -> cloud} join, fall back
    to a (class_index, original_seed) decode when clouds.pkl's own seeds
    aren't globally unique."""
    payload = _load_pickle(clouds_path)
    clouds = payload["clouds"] if isinstance(payload, dict) else payload
    cloud_labels = (
        np.asarray(payload["labels"]).reshape(-1)
        if isinstance(payload, dict) and "labels" in payload else None
    )

    feats = get_cloud_features(
        clouds, r_grid, cache_path=Path(cache_dir) / cache_name, force=force, tag=tag,
    )
    cloud_seeds = feats["cloud_seeds"]
    common_seeds = np.asarray(align["common_seeds"], dtype=np.int64)
    seed_offset = int(align["seed_offset"])

    if len(set(int(s) for s in cloud_seeds)) == len(cloud_seeds):
        row_by_seed = {int(s): i for i, s in enumerate(cloud_seeds)}
        try:
            rows = np.array([row_by_seed[int(s)] for s in common_seeds], dtype=np.int64)
        except KeyError as exc:
            raise KeyError(f"[{tag}] diagram seed {exc} is not present in {clouds_path}.") from exc
    else:
        if cloud_labels is None or not seed_offset:
            raise KeyError(
                f"[{tag}] {clouds_path} has non-unique cloud seeds and no `labels`/seed_offset to "
                "disambiguate them against the globally-unique diagram seeds."
            )
        row_by_cls_orig = {
            (int(cloud_labels[i]), int(cloud_seeds[i])): i for i in range(len(cloud_seeds))
        }
        rows = np.array(
            [row_by_cls_orig[(int(s) // seed_offset, int(s) % seed_offset)] for s in common_seeds],
            dtype=np.int64,
        )

    return {
        "l_minus_r": feats["l_minus_r"][rows],
        "n_points": feats["n_points"][rows],
        "targets": np.asarray(align["targets"], dtype=np.int64),
        "seeds": common_seeds,
    }


def prepare_data_classify(
    clouds_path: Path,
    adversarial_clouds_path: Path | None,
    *,
    diagram_paths: list[Path],
    adversarial_diagram_paths: list[Path] | None,
    k_values: list[int],
    label_names: tuple[str, ...] | None = None,
    r_max: float = R_MAX,
    n_r: int = N_R,
    cache_dir: Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Classification counterpart of prepare_data: build the L(r)-r + n(x)
    feature set for the clouds/split pi_multik's classify run uses, plus the
    matching adversarial set. `diagram_paths` are per-k diagram bundles,
    read only for their `seeds`/`labels` (to reproduce the exact split)."""
    r_grid = default_r_grid(r_max, n_r)
    cache_dir = Path(cache_dir) if cache_dir is not None else Path(clouds_path).parent

    align = _load_classification_alignment(diagram_paths, k_values, tag="train_test")
    class_names = align["class_names"]
    if label_names is not None and list(label_names) != class_names:
        raise ValueError(
            f"target_label_names {list(label_names)} must equal the diagram bundle's class list "
            f"{class_names} exactly (position i == class label i)."
        )

    print(f"Loading classification clouds from {clouds_path} ...")
    train_features = _classification_features_for_alignment(
        clouds_path, align, r_grid, cache_dir, force, tag="train_test", cache_name=CLASSIFY_LFUNC_CACHE,
    )

    adversarial_features = None
    have_adv = (
        adversarial_clouds_path is not None
        and Path(adversarial_clouds_path).exists()
        and adversarial_diagram_paths is not None
        and all(Path(p).exists() for p in adversarial_diagram_paths)
    )
    if have_adv:
        adv_align = _load_classification_alignment(adversarial_diagram_paths, k_values, tag="adversarial")
        if adv_align["class_names"] != class_names:
            raise ValueError("adversarial diagram bundle's class list differs from the training bundle's.")
        print(f"Loading adversarial classification clouds from {adversarial_clouds_path} ...")
        adversarial_features = _classification_features_for_alignment(
            adversarial_clouds_path, adv_align, r_grid, cache_dir, force,
            tag="adversarial", cache_name=CLASSIFY_ADV_LFUNC_CACHE,
        )
    else:
        print("No adversarial clouds / diagram bundles found; skipping adversarial evaluation.")

    return {
        "train_features": train_features,
        "adversarial_features": adversarial_features,
        "adversarial_path": Path(adversarial_clouds_path) if adversarial_features is not None else None,
        "class_names": class_names,
        "r_grid": r_grid,
    }


def _ce_train_one_epoch(model, loader, optimizer, device) -> tuple[float, float]:
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    total_loss, correct, count = 0.0, 0, 0
    for seq, n_scalar, y in loader:
        seq, n_scalar, y = seq.to(device), n_scalar.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(seq, n_scalar)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
        correct += int((logits.argmax(dim=-1) == y).sum())
        count += len(y)
    return total_loss / count, correct / count


@torch.no_grad()
def _ce_evaluate(model, loader, device) -> tuple[float, float]:
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss, correct, count = 0.0, 0, 0
    for seq, n_scalar, y in loader:
        seq, n_scalar, y = seq.to(device), n_scalar.to(device), y.to(device)
        logits = model(seq, n_scalar)
        total_loss += loss_fn(logits, y).item() * len(y)
        correct += int((logits.argmax(dim=-1) == y).sum())
        count += len(y)
    return total_loss / count, correct / count


@torch.no_grad()
def _classify_report(
    model, loader, device, n_classes: int, class_names: list[str]
) -> tuple[float, float, dict[str, float], np.ndarray, np.ndarray]:
    """One pass: (cross-entropy, overall accuracy, per-class recall,
    y_true, y_pred). Per-class recall matches _per_class_accuracy in
    pi_multik.py (NaN for an absent class)."""
    model.eval()
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    total_loss, count = 0.0, 0
    y_true_parts, y_pred_parts = [], []
    for seq, n_scalar, y in loader:
        seq, n_scalar, y = seq.to(device), n_scalar.to(device), y.to(device)
        logits = model(seq, n_scalar)
        total_loss += loss_fn(logits, y).item()
        count += len(y)
        y_true_parts.append(y.cpu().numpy())
        y_pred_parts.append(logits.argmax(dim=-1).cpu().numpy())

    y_true = np.concatenate(y_true_parts)
    y_pred = np.concatenate(y_pred_parts)
    per_class = {}
    for c in range(n_classes):
        mask = y_true == c
        per_class[class_names[c]] = float((y_pred[mask] == c).mean()) if mask.any() else float("nan")
    overall = float((y_pred == y_true).mean())
    return total_loss / count, overall, per_class, y_true, y_pred


def _confusion_png(path: Path, y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> None:
    k = len(class_names)
    cm = np.zeros((k, k), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    cm_norm = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None)

    fig, ax = plt.subplots(figsize=(1.8 + 0.95 * k, 1.5 + 0.95 * k))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(k))
    ax.set_yticks(range(k))
    ax.set_xticklabels(class_names, rotation=40, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    for i in range(k):
        for j in range(k):
            ax.text(
                j, i, f"{cm[i, j]}\n{cm_norm[i, j]:.2f}", ha="center", va="center",
                fontsize=8, color="white" if cm_norm[i, j] > 0.5 else "black",
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("row-normalized confusion")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_one_seed_classify(
    seed: int,
    *,
    train_features: dict[str, np.ndarray],
    adversarial_features: dict[str, np.ndarray] | None,
    adversarial_path: Path | None,
    r_grid: np.ndarray,
    output_root: Path,
    class_names: list[str],
    n_epochs: int = 200,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    early_stopping_patience: int | None = 30,
    device_pref: str | None = None,
    results_root: Path | None = None,
    run_tag: str | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_root) / f"seed_{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    adversarial_path = Path(adversarial_path) if adversarial_path is not None else None

    torch.manual_seed(seed)
    if device_pref:
        device = device_pref
    elif torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    n_classes = len(class_names)
    targets = np.asarray(train_features["targets"], dtype=np.int64)
    classes = sorted(int(c) for c in np.unique(targets))
    if classes != list(range(n_classes)):
        raise ValueError(
            f"[vihrs classify seed={seed}] expected contiguous class labels 0..{n_classes - 1}, got {classes}."
        )

    n = len(targets)
    train_idx, val_idx, test_idx = train_val_test_indices(n, seed)

    # Every normalization statistic fit on the train split ONLY, then frozen
    # for val/test/adversarial -- same discipline as the regression path.
    n_norm = fit_log_zscore(train_features["n_points"][train_idx])
    lr_norm = fit_zscore_global(train_features["l_minus_r"][train_idx])

    n_std = apply_log_zscore(train_features["n_points"], n_norm).astype(np.float32)
    seq_all = apply_zscore_global(train_features["l_minus_r"], lr_norm).astype(np.float32)

    def _loader(idx: np.ndarray, shuffle: bool) -> DataLoader:
        ds = TensorDataset(
            torch.from_numpy(seq_all[idx]),
            torch.from_numpy(n_std[idx]),
            torch.from_numpy(targets[idx]),
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    train_loader = _loader(train_idx, True)
    val_loader = _loader(val_idx, False)
    test_loader = _loader(test_idx, False)

    model = VihrsCNN(seq_len=len(r_grid), n_targets=n_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss, best_state, best_epoch = float("inf"), None, 0
    epochs_no_improve = 0

    for epoch in range(1, n_epochs + 1):
        train_loss, train_acc = _ce_train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = _ce_evaluate(model, val_loader, device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        if val_loss < best_val_loss:
            best_val_loss, best_epoch = val_loss, epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if epoch == 1 or epoch % 25 == 0 or epoch == n_epochs:
            print(
                f"[vihrs classify seed={seed}] epoch {epoch:3d} | train {train_loss:.4f} | val {val_loss:.4f} "
                f"| train_acc {train_acc:.4f} | val_acc {val_acc:.4f}"
            )
        if early_stopping_patience is not None and epochs_no_improve >= early_stopping_patience:
            print(f"[vihrs classify seed={seed}] early stopping at epoch {epoch} (no val improvement for {early_stopping_patience})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    print(f"[vihrs classify seed={seed}] best checkpoint: epoch {best_epoch}/{n_epochs}, val_loss {best_val_loss:.4f}")

    test_loss, test_acc, test_acc_per_class, test_true, test_pred = _classify_report(
        model, test_loader, device, n_classes, class_names
    )
    print(
        f"\n[vihrs classify seed={seed}] test cross-entropy {test_loss:.4f} | test accuracy {test_acc:.4f}\n"
        f"  per-class accuracy: " + ", ".join(f"{name}={acc:.3f}" for name, acc in test_acc_per_class.items())
    )

    adversarial_loss = None
    adversarial_acc = None
    adversarial_acc_per_class = None
    if adversarial_features is not None:
        adv_targets = np.asarray(adversarial_features["targets"], dtype=np.int64)
        adv_n_std = apply_log_zscore(adversarial_features["n_points"], n_norm).astype(np.float32)
        adv_seq = apply_zscore_global(adversarial_features["l_minus_r"], lr_norm).astype(np.float32)
        adv_ds = TensorDataset(torch.from_numpy(adv_seq), torch.from_numpy(adv_n_std), torch.from_numpy(adv_targets))
        adv_loader = DataLoader(adv_ds, batch_size=batch_size, shuffle=False)
        adversarial_loss, adversarial_acc, adversarial_acc_per_class, adv_true, adv_pred = _classify_report(
            model, adv_loader, device, n_classes, class_names
        )
        print(
            f"[vihrs classify seed={seed}] adversarial cross-entropy {adversarial_loss:.4f} | "
            f"adversarial accuracy {adversarial_acc:.4f}"
        )

    # ---- save (same results.pt / results.json / model.pt schema as run_one_seed) ----
    cfg = {
        "task": "classify",
        "method": "vihrs",
        "paper": "Vihrs (2022), feature set + CNN adapted to classification",
        "batch_size": batch_size,
        "n_epochs": n_epochs,
        "lr": lr,
        "weight_decay": weight_decay,
        "early_stopping_patience": early_stopping_patience,
        "checkpoint_best": True,
        "best_epoch": best_epoch,
        "embedding_dim": model.merge_dim,
        "seed": seed,
        "r_max": float(r_grid[-1]),
        "n_r": len(r_grid),
        "class_names": list(class_names),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "output_dir": str(output_dir),
        "results_root": str(results_root) if results_root else None,
        "run_tag": run_tag,
    }
    # label_norm mirrors pi_multik.py's classification stub so save_results
    # / downstream readers see a consistent shape (no mean/std for a class index).
    label_norm = {
        "kind": "classification",
        "classes": classes,
        "mean": None,
        "std": None,
        "transforms": ["class_index"] * n_classes,
    }
    stamp = provenance_stamp(run_tag=run_tag)

    torch.save(
        {
            "model_state": best_state,
            "history": history,
            "config": cfg,
            "test_loss": test_loss,
            "test_loss_per_target": None,
            "label_names": list(class_names),
            "label_norm": label_norm,
            "label_log_mean": None,
            "label_log_std": None,
            "label_transforms": ["class_index"] * n_classes,
            "n_points_norm": n_norm,
            "lr_norm": lr_norm,
            "test_accuracy": test_acc,
            "test_accuracy_per_class": test_acc_per_class,
            "adversarial_loss": adversarial_loss,
            "adversarial_loss_per_target": None,
            "adversarial_accuracy": adversarial_acc,
            "adversarial_accuracy_per_class": adversarial_acc_per_class,
            "adversarial_path": str(adversarial_path) if adversarial_path else None,
            **stamp,
        },
        output_dir / "results.pt",
    )
    torch.save(model.cpu(), output_dir / "model.pt")
    model.to(device)

    json_payload: dict[str, Any] = {
        "task": "classify",
        "method": "vihrs",
        "test_loss": test_loss,
        "test_loss_per_target": None,
        "seed": seed,
        **stamp,
        "config": cfg,
        "class_names": list(class_names),
        "test_accuracy": test_acc,
        "test_accuracy_per_class": test_acc_per_class,
    }
    if adversarial_loss is not None:
        json_payload["adversarial_accuracy"] = adversarial_acc
        json_payload["adversarial_accuracy_per_class"] = adversarial_acc_per_class
        json_payload["adversarial_loss"] = adversarial_loss
        json_payload["adversarial_loss_per_target"] = None
        json_payload["adversarial_path"] = str(adversarial_path)
    with open(output_dir / "results.json", "w") as f:
        json.dump(json_payload, f, indent=2, default=str)

    append_ledger_entry(
        output_dir=output_dir,
        results_root=results_root,
        method="vihrs",
        seed=seed,
        test_loss=test_loss,
        adversarial_loss=adversarial_loss,
        stamp=stamp,
    )

    try:
        _confusion_png(output_dir / "plots" / "confusion_matrix.png", test_true, test_pred, class_names)
        if adversarial_features is not None:
            _confusion_png(output_dir / "plots" / "confusion_matrix_adversarial.png", adv_true, adv_pred, class_names)
    except Exception as exc:  # noqa: BLE001 -- a plotting failure must not lose a finished run
        warnings.warn(f"[vihrs classify seed={seed}] confusion-matrix plot failed: {exc!r}")

    return {
        "seed": seed,
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "test_accuracy_per_class": test_acc_per_class,
        "adversarial_loss": adversarial_loss,
        "adversarial_accuracy": adversarial_acc,
        "adversarial_accuracy_per_class": adversarial_acc_per_class,
    }


# ══════════════════════════════════════════════════════════════════════════════
# CLI (standalone debugging entry point; the primary entry point going
# forward is scripts/train.py --method vihrs)
# ══════════════════════════════════════════════════════════════════════════════

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--clouds", type=Path, default=DEFAULT_CLOUDS)
    p.add_argument("--adversarial-clouds", type=Path, default=DEFAULT_ADVERSARIAL)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    p.add_argument("--n-epochs", type=int, default=20, help="paper default: 20")
    p.add_argument("--batch-size", type=int, default=100, help="paper default: 100")
    p.add_argument("--lr", type=float, default=1e-3, help="Adam default, matches this repo's other methods")
    p.add_argument("--r-max", type=float, default=R_MAX)
    p.add_argument("--n-r", type=int, default=N_R)
    p.add_argument("--checkpoint-best", action="store_true", help="select the best-val-loss checkpoint instead of the paper's fixed-epoch recipe")
    p.add_argument(
        "--feature-cache-dir", type=Path, default=None,
        help="where to cache extracted L(r)-r features; default: alongside --clouds/--adversarial-clouds",
    )
    p.add_argument("--recompute-features", action="store_true", help="ignore any cached features and recompute")
    p.add_argument(
        "--skip-mincontrast", action="store_true",
        help="skip the bonus minimum-contrast comparison (paper section 4); can be slow on large test sets",
    )
    p.add_argument("--max-clouds", type=int, default=None, help="limit clouds per split, for quick smoke-testing")
    p.add_argument("--device", type=str, default=None, choices=["cpu", "cuda", "mps"])
    p.add_argument(
        "--task", type=str, default="params", choices=["params", "classify"],
        help="params (default): the paper's parameter-estimation recipe. classify: reuse the L(r)-r + n(x) "
             "feature set and CNN to classify the point-process family instead -- requires --diagram-paths / "
             "--k-values / --class-names (see run_one_seed_classify).",
    )
    p.add_argument("--k-values", type=int, nargs="+", default=None, help="classify: per-k diagram bundle k values (order matters)")
    p.add_argument("--diagram-paths", type=Path, nargs="+", default=None, help="classify: per-k diagrams.pkl, aligned with --k-values")
    p.add_argument("--adversarial-diagram-paths", type=Path, nargs="+", default=None, help="classify: per-k adversarial_diagrams.pkl")
    p.add_argument("--class-names", type=str, nargs="+", default=None, help="classify: ordered class names (index i == class i)")
    return p


def _main_classify(args: argparse.Namespace) -> dict[int, dict[str, Any]]:
    if not args.k_values or not args.diagram_paths:
        raise SystemExit("--task classify requires --k-values and --diagram-paths.")
    if len(args.k_values) != len(args.diagram_paths):
        raise SystemExit("--k-values and --diagram-paths must have the same length.")

    data = prepare_data_classify(
        args.clouds,
        args.adversarial_clouds,
        diagram_paths=list(args.diagram_paths),
        adversarial_diagram_paths=list(args.adversarial_diagram_paths) if args.adversarial_diagram_paths else None,
        k_values=list(args.k_values),
        label_names=tuple(args.class_names) if args.class_names else None,
        r_max=args.r_max,
        n_r=args.n_r,
        cache_dir=args.feature_cache_dir,
        force=args.recompute_features,
    )

    n_epochs = args.n_epochs if args.n_epochs != 20 else 200  # classify default differs from the paper's 20
    batch_size = args.batch_size if args.batch_size != 100 else 128
    summary: dict[int, dict[str, Any]] = {}
    for seed in args.seeds:
        print(f"\n{'#' * 90}\n### vihrs classify | seed {seed}\n{'#' * 90}")
        summary[seed] = run_one_seed_classify(
            seed,
            train_features=data["train_features"],
            adversarial_features=data["adversarial_features"],
            adversarial_path=data["adversarial_path"],
            r_grid=data["r_grid"],
            output_root=args.output_root,
            class_names=data["class_names"],
            n_epochs=n_epochs,
            batch_size=batch_size,
            lr=args.lr,
            device_pref=args.device,
        )
    print(f"\nAll {len(args.seeds)} seed(s) done. Results under {args.output_root}")
    return summary


def main(argv: list[str] | None = None) -> dict[int, dict[str, Any]]:
    args = build_arg_parser().parse_args(argv)

    if args.task == "classify":
        return _main_classify(args)

    data = prepare_data(
        args.clouds,
        args.adversarial_clouds,
        r_max=args.r_max,
        n_r=args.n_r,
        cache_dir=args.feature_cache_dir,
        force=args.recompute_features,
        max_clouds=args.max_clouds,
    )

    mc_cache: dict[Any, Any] = {}
    mc_rng = np.random.default_rng(MC_SEED)

    summary: dict[int, dict[str, Any]] = {}
    for seed in args.seeds:
        print(f"\n{'#' * 90}\n### vihrs | seed {seed}\n{'#' * 90}")
        summary[seed] = run_one_seed(
            seed,
            train_records=data["train_records"],
            train_features=data["train_features"],
            adversarial_features=data["adversarial_features"],
            adversarial_path=data["adversarial_path"],
            r_grid=data["r_grid"],
            output_root=args.output_root,
            n_epochs=args.n_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            checkpoint_best=args.checkpoint_best,
            device_pref=args.device,
            skip_mincontrast=args.skip_mincontrast,
            mc_cache=mc_cache,
            mc_rng=mc_rng,
        )

    print(f"\nAll {len(args.seeds)} seed(s) done. Results under {args.output_root}")
    return summary


if __name__ == "__main__":
    main()
