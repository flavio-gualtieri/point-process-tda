#!/usr/bin/env python3
# scripts/sweep_pi_sigma.py
"""Exploratory sweep of persistence_image's Gaussian bandwidth
(features.persistence_image.params.sigma_pixels) for pi_multik, one seed only
-- meant for eyeballing sigma sensitivity before committing to a full
multi-seed run at whichever value looks best.

sigma_pixels is a bandwidth in units of pixels, shared by both the birth and
persistence axes -- PersistenceImager derives each axis's absolute sigma from
it and that axis's own pixel width, so the swept quantity is "how much
smoothing" alone. A single absolute sigma shared between the two axes doesn't
have that property: birth_range and pers_range are calibrated independently
and typically have very different spans, so the same absolute sigma is a
different number of pixels on each axis, and sweeping it conflates smoothing
strength with that axis-distortion artifact -- the same confound this repo
already found (docs/pi_multik_report.tex Section 3.6) in the retired
per-diagram adaptive design.

Reuses the diagrams.pkl already cached per k by scripts/featurize.py, so
only the Gaussian-rasterization step (PersistenceImager.transform) is redone
per sigma value -- and reuses PIMultiKExperiment.run() verbatim (same
training loop, same save_results schema as a normal scripts/train.py run),
so results here are directly comparable to the config's baseline sigma_pixels
result already on disk.

Rasterized per-sigma image tensors are the expensive/large intermediate
(hundreds of MB per k) and are exploratory only -- written to a scratch temp
dir and deleted once that sigma's training finishes, never to data/. Only
the small final results.pt/results.json/model.pt land in
results/<process>/<filtration_tag>/pi_multik/sigma_sweep/sigma_<value>/,
one flat extra folder next to the real seed_<seed>/ dirs (deliberately not
a new ResultsPaths path-scheme level -- sigma_pixels isn't a supported
comparison axis anywhere else in this repo, and this is exploratory).

Also writes, alongside comparison_seed<seed>.json:
  sigma_sweep/training_curves_seed<seed>.png -- train_loss/val_loss curves
  for every sigma_pixels overlaid on one pair of axes (the config's baseline
  sigma_pixels drawn solid/bold, the rest dashed), so convergence speed and
  overfitting can be compared across the sweep, not just final test_loss.

Usage:
    python scripts/sweep_pi_sigma.py configs/runs/nested_thomas_pi_multik.yaml
    python scripts/sweep_pi_sigma.py configs/runs/nested_thomas_pi_multik.yaml \\
        --seed 9371 --sigma-values 0.25 0.5 1 2 4 --force
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cloudforger.config import load_config
from cloudforger.core.io import dump_pickle
from cloudforger.core.records import load_diagrams
from cloudforger.features import REGISTRY as FEATURE_REGISTRY
from cloudforger.filtration import REGISTRY as FILTRATION_REGISTRY
from cloudforger.nn.experiments.base import build_experiment
from cloudforger.nn.experiments.common import MultiSourceExperiment
from cloudforger.paths import DEFAULT_DATA_ROOT, DEFAULT_RESULTS_ROOT, DataPaths, ResultsPaths, is_done
from cloudforger.vectorizers.calibrated import build_calibrated_imager
from cloudforger.vectorizers.persistence_image import DEFAULT_SIGMA_PIXELS

# Well below 8 pixels on purpose: at resolution 64, 8px already blurs across
# an eighth of the image span. The old 1px adaptive-sigma floor turned out to
# be under-smoothed (docs/pi_multik_report.tex Section 3.6), so this grid
# starts below that too, to actually see the under- to over-smoothed transition
# rather than starting inside the already-known-bad region.
DEFAULT_SIGMA_VALUES = [0.25, 0.5, 1.0, 2.0, 4.0]

# Mirrors scripts/evaluate.py's plot styling -- no shared plotting module in
# this repo, each plotting script keeps its own copy.
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


def plot_sigma_training_curves(
    histories: dict[str, dict[str, Any]], baseline_sigma: float, seed: int, out_path: Path
) -> None:
    """One line per sigma_pixels, overlaid -- train_loss and val_loss side by
    side, so overfitting/convergence speed can be compared directly across
    the sweep instead of just the final test_loss table."""
    ordered = sorted(histories.items(), key=lambda kv: kv[1]["sigma_pixels"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True)
    for ax, split in zip(axes, ("train_loss", "val_loss")):
        for i, (tag, h) in enumerate(ordered):
            curve = h.get("history", {}).get(split)
            if not curve:
                continue
            epochs = np.arange(1, len(curve) + 1)
            is_baseline = abs(h["sigma_pixels"] - baseline_sigma) < 1e-9
            label = f"sigma_pixels={h['sigma_pixels']:g}" + (" (baseline)" if is_baseline else "")
            ax.plot(
                epochs, curve, color=PALETTE[i % len(PALETTE)],
                linewidth=2.2 if is_baseline else 1.4,
                linestyle="-" if is_baseline else "--",
                label=label,
            )
        ax.set_yscale("log")
        ax.set_xlabel("epoch")
        ax.set_title(split.replace("_", " "))
        _style_axis(ax)
    axes[0].set_ylabel("loss (log scale)")
    axes[-1].legend(loc="upper right", frameon=False, fontsize=8)
    fig.suptitle(f"pi_multik training curves by sigma_pixels (seed {seed})")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved training-curve overlay -> {out_path}")


def _entropy_by_dim(diagrams: list, homology_dims: tuple[int, ...]) -> dict[int, np.ndarray]:
    entropy_feature = FEATURE_REGISTRY.build("persistence_entropy", homology_dims=homology_dims)
    per_diagram = [entropy_feature.compute(d) for d in diagrams]
    return {dim: np.array([pd[dim] for pd in per_diagram]) for dim in homology_dims}


def _image_payload(
    diagrams: list, bundle: dict, homology_dims: tuple[int, ...], resolution: int, sigma_pixels: float
) -> dict[str, Any]:
    imager = build_calibrated_imager(
        diagrams, homology_dims=homology_dims, resolution=resolution,
        sigma_pixels=sigma_pixels, verbose=False,
    )
    images = [imager.transform(d) for d in diagrams]
    # float32: this payload is a scratch intermediate, deleted right after
    # training -- load_multik_split upcasts to float64 on load regardless.
    image_tensors = {dim: np.stack([im[dim] for im in images]).astype(np.float32) for dim in homology_dims}
    return {
        **bundle, "image_tensors": image_tensors, "homology_dims": list(homology_dims),
        "imager_params": imager.params, "persistence_entropy": _entropy_by_dim(diagrams, homology_dims),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path, help="RunConfig YAML path (must have method.name == pi_multik)")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="path.to.field=value")
    parser.add_argument(
        "--seed", type=int, default=9371,
        help="single seed to sweep on (default: 9371, the one seed already trained for pi_multik/vihrs)",
    )
    parser.add_argument(
        "--sigma-values", type=float, nargs="+", default=None,
        help=f"sigma_pixels grid to try (default: {DEFAULT_SIGMA_VALUES}, plus the config's own baseline sigma_pixels)",
    )
    parser.add_argument("--force", action="store_true", help="retrain even if a sigma's results.pt already exists")
    args = parser.parse_args(argv)

    cfg = load_config(args.config, overrides=args.overrides)
    if cfg.method is None or cfg.method.name != "pi_multik":
        raise ValueError(f"This sweep is pi_multik-specific; config's method is {cfg.method.name!r}.")

    filtrations = [FILTRATION_REGISTRY.build(f.name, **f.params) for f in cfg.filtration]
    k_values = [f.params["k"] for f in filtrations]

    feat_cfg = next((f for f in cfg.features if f.name == "persistence_image"), None)
    if feat_cfg is None:
        raise ValueError("config has no persistence_image feature block -- nothing to sweep sigma_pixels on.")
    homology_dims = tuple(feat_cfg.params.get("homology_dims", (0, 1)))
    resolution = int(feat_cfg.params.get("resolution", 64))
    baseline_sigma = float(feat_cfg.params.get("sigma_pixels", DEFAULT_SIGMA_PIXELS))

    sigma_values = sorted(set(args.sigma_values or DEFAULT_SIGMA_VALUES) | {baseline_sigma})
    print(f"Sweeping sigma_pixels over {sigma_values} (config baseline: {baseline_sigma}) on seed={args.seed}.")

    data_paths = DataPaths(cfg.process.name, root=cfg.data_root or DEFAULT_DATA_ROOT)
    results_paths = ResultsPaths(cfg.process.name, root=cfg.results_root or DEFAULT_RESULTS_ROOT)
    sweep_dir = results_paths.method_dir(filtrations, "pi_multik") / "sigma_sweep"

    print("Loading cached diagrams (run scripts/featurize.py first if this errors) ...")
    per_k_train = [load_diagrams(data_paths.diagrams([f])) for f in filtrations]

    adversarial_available = cfg.use_adversarial and data_paths.clouds(adversarial=True).exists()
    per_k_adv = None
    if adversarial_available:
        per_k_adv = []
        for f in filtrations:
            p = data_paths.diagrams([f], adversarial=True)
            if not p.exists():
                raise FileNotFoundError(
                    f"{p} missing but adversarial_clouds.pkl exists -- run "
                    f"scripts/featurize.py --set filtration.0.params.k={f.params.get('k')} first."
                )
            per_k_adv.append(load_diagrams(p))

    cfg_dict_base: dict[str, Any] = {
        "task": "params", "method": "pi_multik", "seed": args.seed,
        "use_covariates": cfg.use_covariates, **cfg.method.params, "k_values": k_values,
    }
    if cfg.target_label_names is not None:
        cfg_dict_base["target_label_names"] = cfg.target_label_names
    if cfg.log_label_names is not None:
        cfg_dict_base["log_label_names"] = cfg.log_label_names

    summary: dict[str, Any] = {
        "process": cfg.process.name, "seed": args.seed, "sigma_values": sigma_values,
        "baseline_sigma": baseline_sigma, "results": {},
    }

    for sigma in sigma_values:
        tag = f"sigma_{sigma:.3f}"
        output_dir = sweep_dir / tag / f"seed_{args.seed}"
        results_pt = output_dir / "results.pt"

        if is_done(output_dir) and not args.force:
            print(f"[{tag}] already done, skipping ({output_dir}). Pass --force to redo.")
        else:
            with tempfile.TemporaryDirectory(prefix=f"pi_sigma_{tag}_") as tmp_str:
                tmp = Path(tmp_str)
                image_paths, adv_image_paths = [], []
                for i, k in enumerate(k_values):
                    diagrams, bundle = per_k_train[i]
                    payload = _image_payload(diagrams, bundle, homology_dims, resolution, sigma)
                    p = tmp / f"dtm_k{k}_pi.pkl"
                    dump_pickle(p, payload)
                    image_paths.append(p)

                    if per_k_adv is not None:
                        adv_diagrams, adv_bundle = per_k_adv[i]
                        adv_payload = _image_payload(adv_diagrams, adv_bundle, homology_dims, resolution, sigma)
                        ap = tmp / f"dtm_k{k}_pi_adv.pkl"
                        dump_pickle(ap, adv_payload)
                        adv_image_paths.append(ap)

                dataset_paths = {"clouds": data_paths.clouds(), "images": image_paths}
                adversarial_paths = (
                    {"clouds": data_paths.clouds(adversarial=True), "images": adv_image_paths}
                    if per_k_adv is not None else None
                )

                exp = build_experiment(cfg_dict_base)
                assert isinstance(exp, MultiSourceExperiment)
                print(f"\n{'=' * 80}\n[{tag}] training pi_multik seed={args.seed} sigma_pixels={sigma}\n{'=' * 80}")
                exp.run(dataset_paths, output_dir, adversarial_paths=adversarial_paths)

        result = torch.load(results_pt, map_location="cpu", weights_only=False)
        history = result.get("history") or {}
        summary["results"][tag] = {
            "sigma_pixels": sigma,
            "test_loss": result.get("test_loss"),
            "adversarial_loss": result.get("adversarial_loss"),
            "test_loss_per_target": result.get("test_loss_per_target"),
            "history": {"train_loss": history.get("train_loss", []), "val_loss": history.get("val_loss", [])},
        }

    sweep_dir.mkdir(parents=True, exist_ok=True)
    out_path = sweep_dir / f"comparison_seed{args.seed}.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    plot_sigma_training_curves(
        summary["results"], baseline_sigma, args.seed, sweep_dir / f"training_curves_seed{args.seed}.png"
    )

    print(f"\n{'=' * 80}\n  sigma_pixels sweep -- seed {args.seed}\n{'=' * 80}")
    print(f"  {'sigma_pixels':>12}{'test_loss':>14}{'adversarial_loss':>20}")
    for tag, r in sorted(summary["results"].items(), key=lambda kv: kv[1]["sigma_pixels"]):
        marker = "  <- config baseline" if abs(r["sigma_pixels"] - baseline_sigma) < 1e-9 else ""
        adv = r["adversarial_loss"]
        adv_str = f"{adv:.4f}" if adv is not None else "n/a"
        print(f"  {r['sigma_pixels']:>12.3f}{r['test_loss']:>14.4f}{adv_str:>20}{marker}")

    best_tag = min(summary["results"], key=lambda t: summary["results"][t]["test_loss"])
    print(f"\nBest by test_loss: {best_tag} (sigma_pixels={summary['results'][best_tag]['sigma_pixels']})")
    print(f"Saved comparison -> {out_path}")


if __name__ == "__main__":
    main()
