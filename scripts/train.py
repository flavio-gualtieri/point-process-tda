#!/usr/bin/env python3
# scripts/train.py
"""Train (or fit, for classical baselines) one configured method for one or
more seeds, saving results.pt / results.json / (model.pt if trainable)
under results/<process>/<filtration_tag>/<method>/seed_<seed>/ (or .../
<method>/_runs/<run_tag>/seed_<seed>/ if --run-tag is given -- see
scripts/archive_run.py to move an existing method's results into one of
these slots before retraining, so the previous results aren't overwritten).

Every result also gets a provenance stamp (git commit, dirty-tree flag,
timestamp, run_tag) written into results.json/results.pt and appended as a
row to results/experiments.jsonl -- see cloudforger.provenance.

One method name (RunConfig's method.name) selects the estimator regardless
of whether it's a CNN (cloudforger.experiments), a classical estimator
(mincontrast/palm), or the vihrs neural baseline -- all three write the
identical output schema (cloudforger.experiments.common.save_results),
so scripts/evaluate.py has exactly one code path no matter which one
produced the numbers.

Classical baselines (mincontrast/palm) have no natural notion of "seed" the
way trained models do -- they fit per-cloud, with no train/val/test split.
To make them seed-paired-comparable with every trained method (needed for
evaluate.py's paired tests), each seed's run fits only on that seed's TEST
partition, using the exact same train_val_test_indices(n, seed) split every
trained method uses, and reports loss on the same standardized (log+zscore,
fit on that seed's TRAIN split) scale trained methods use -- not raw units,
which would make "test_loss" numbers incomparable across methods.

Usage:
    python scripts/train.py configs/runs/thomas/thomas_pi_multik_k5k10k15.yaml
    python scripts/train.py configs/runs/foo.yaml --seed 9371   # single seed, for SLURM array jobs
    python scripts/train.py configs/runs/foo.yaml --force        # retrain even if results.pt exists
    python scripts/train.py configs/runs/foo.yaml --run-tag candidate_b  # keep results.pt at a labeled path
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cloudforger import baselines
from cloudforger.config import RunConfig, load_config
from cloudforger.core.io import load_pickle
from cloudforger.core.splits import train_val_test_indices
from cloudforger.data_generation.filtration import BIFILTRATION_REGISTRY, REGISTRY as FILTRATION_REGISTRY
from cloudforger.data_generation.filtration.base import Filtration
from cloudforger.experiments.base import build_experiment
from cloudforger.experiments.common import MultiSourceExperiment, save_results
from cloudforger.paths import DEFAULT_DATA_ROOT, DEFAULT_RESULTS_ROOT, DataPaths, ExplicitTag, ResultsPaths, is_done

MULTI_K_METHODS = {
    "pi_multik", "pi_multik_fusion", "pi_multik_scaleconv", "pi_multik_towers", "pi_multik_earlyfusion",
    "pi_multik_dimsplit", "betti_multik", "vec_multik",
}
CLASSICAL_BASELINE_NAMES = {"mincontrast", "mincontrast_g", "mincontrast_nested", "mincontrast_g_nested", "palm"}
FILE_KEY_TO_FEATURE_NAME = {"pi": "persistence_image", "images": "persistence_image"}
# file_keys with no filtration dependency -- their results always live under
# the "raw" tag (paths.RAW_TAG), regardless of what's configured under
# `filtration:`, so a raw_pc run never misleadingly looks like it used
# whatever filtration happened to be in the config.
FILTRATION_INDEPENDENT_FILE_KEYS = {"raw_pc", "pairwise"}


def build_filtrations(cfg: RunConfig) -> list[Filtration]:
    return [FILTRATION_REGISTRY.build(f.name, **f.params) for f in cfg.filtration]


def build_bifiltrations(cfg: RunConfig) -> list[Filtration]:
    """Bifiltration counterpart of build_filtrations -- Bifiltration only
    implements .path_tag() of the Filtration interface, but that's all
    DataPaths/ResultsPaths (paths.py) ever call, same as ExplicitTag's
    stand-in, so these are interchangeable wherever a list[Filtration] is
    used purely for path resolution below."""
    return [BIFILTRATION_REGISTRY.build(b.name, **b.params) for b in cfg.bifiltration]


def build_method_cfg(cfg: RunConfig, seed: int) -> dict[str, Any]:
    cfg_dict: dict[str, Any] = {
        "task": "params",
        "method": cfg.method.name,
        "seed": seed,
        **cfg.method.params,
    }
    if cfg.target_label_names is not None:
        cfg_dict["target_label_names"] = cfg.target_label_names
    if cfg.log_label_names is not None:
        cfg_dict["log_label_names"] = cfg.log_label_names
    return cfg_dict


def _experiment_dataset_path(exp, data_paths: DataPaths, filtrations: list[Filtration], adversarial: bool = False) -> Path:
    if exp.file_key == "raw_pc":
        return data_paths.clouds(adversarial=adversarial)
    feature_name = FILE_KEY_TO_FEATURE_NAME.get(exp.file_key, exp.file_key)
    return data_paths.feature(filtrations, feature_name, adversarial=adversarial)


def _topo_superset_image_paths(data_paths: DataPaths, m_values: list[float], adversarial: bool) -> list[Path]:
    """topo_superset (scripts/featurize_topo_superset.py) stores persistence
    images flat, keyed by mass fraction m, not under the per-k
    DataPaths.feature() convention -- see ExplicitTag's docstring for why."""
    prefix = "adversarial_" if adversarial else ""
    superset_dir = data_paths.process_dir / "topo_superset"
    return [superset_dir / f"{prefix}dtm_m{m:.2f}_persistence_image.pkl" for m in m_values]


def _multi_source_dataset_paths(
    exp, cfg: RunConfig, data_paths: DataPaths, filtrations: list[Filtration], adversarial: bool = False
) -> dict[str, Any]:
    multi_k = cfg.method.name in MULTI_K_METHODS
    m_values = cfg.method.params.get("m_values")
    # bifiltration-based multi-source methods (mph_fusion) read mph_image.pkl,
    # not persistence_image.pkl -- same cfg.bifiltration branch build_bifiltrations
    # uses for path tagging above, kept as a presence check (not a method-name
    # allowlist) so any future bifiltration-based multi-source method picks
    # this up automatically.
    image_feature_name = "mph_image" if cfg.bifiltration else "persistence_image"
    paths: dict[str, Any] = {}
    if "clouds" in exp.file_keys:
        paths["clouds"] = data_paths.clouds(adversarial=adversarial)
    if "images" in exp.file_keys:
        if m_values is not None:
            # topo_superset pathway (matern's mass-fraction sweep): still
            # reads a precomputed, shared, whole-population-calibrated
            # image file -- the same leakage the multi_k branch below now
            # avoids, not yet fixed here (see pi_multik.py's module
            # docstring, "currently-UNFIXED" item 1).
            paths["images"] = _topo_superset_image_paths(data_paths, m_values, adversarial)
        elif multi_k:
            # pi_multik-family methods calibrate persistence images fresh
            # per training seed, on that seed's train rows only (see
            # experiments/pi_multik/pi_multik.py's module docstring) --
            # so they read cached per-k diagrams here, not a precomputed,
            # shared imaging that would bake in the old whole-population
            # calibration leak.
            paths["images"] = [data_paths.diagrams([f], adversarial=adversarial) for f in filtrations]
        else:
            paths["images"] = data_paths.feature(filtrations, image_feature_name, adversarial=adversarial)
    if "diagrams" in exp.file_keys:
        # Single-filtration methods (betti_cnn) that compute their own
        # calibration on the fly from diagrams.pkl, same reasoning as the
        # multi_k "images" branch above -- filtrations here is a single-
        # entry list (one config, one filtration), so this resolves to one
        # path, not a per-k list.
        paths["diagrams"] = data_paths.diagrams(filtrations, adversarial=adversarial)
    return paths


def run_experiment_method(
    cfg: RunConfig,
    seed: int,
    data_paths: DataPaths,
    results_paths: ResultsPaths,
    filtrations: list[Filtration],
    force: bool,
    run_tag: str | None = None,
) -> None:
    cfg_dict = build_method_cfg(cfg, seed)
    cfg_dict["results_root"] = str(results_paths.root)
    if run_tag:
        cfg_dict["run_tag"] = run_tag
    m_values = cfg.method.params.get("m_values")
    if m_values is not None:
        cfg_dict["k_values"] = list(m_values)  # k_values is just an ordered scale label list internally
    elif cfg.method.name in MULTI_K_METHODS and "k_values" not in cfg_dict:
        cfg_dict["k_values"] = [f.params.get("k") for f in filtrations]

    exp = build_experiment(cfg_dict)
    is_multi_source = isinstance(exp, MultiSourceExperiment)
    if m_values is not None:
        effective_filtrations = [ExplicitTag(f"dtm_m{m:.2f}") for m in m_values]
    else:
        effective_filtrations = (
            [] if not is_multi_source and exp.file_key in FILTRATION_INDEPENDENT_FILE_KEYS else filtrations
        )
    output_dir = results_paths.seed_dir(effective_filtrations, exp.subdir, seed, run_tag=run_tag)
    if is_done(output_dir) and not force:
        print(f"[{exp.subdir} seed={seed}] already done, skipping ({output_dir}).")
        return

    adversarial_available = cfg.use_adversarial and data_paths.clouds(adversarial=True).exists()

    if is_multi_source:
        dataset_paths = _multi_source_dataset_paths(exp, cfg, data_paths, filtrations, adversarial=False)
        adversarial_paths = (
            _multi_source_dataset_paths(exp, cfg, data_paths, filtrations, adversarial=True) if adversarial_available else None
        )
        exp.run(dataset_paths, output_dir, adversarial_paths=adversarial_paths)
    else:
        dataset_path = _experiment_dataset_path(exp, data_paths, effective_filtrations, adversarial=False)
        adversarial_path = (
            _experiment_dataset_path(exp, data_paths, effective_filtrations, adversarial=True) if adversarial_available else None
        )
        exp.run(dataset_path, output_dir, adversarial_path=adversarial_path)


def run_vihrs_classify_method(
    cfg: RunConfig,
    seed: int,
    data_paths: DataPaths,
    results_paths: ResultsPaths,
    filtrations: list[Filtration],
    force: bool,
    run_tag: str | None = None,
) -> None:
    """vihrs adapted to the classification task (method.params.task: classify).

    vihrs itself needs no filtration/diagrams -- it computes L(r)-r straight
    from clouds.pkl. The per-k diagram bundles are read ONLY for their
    `seeds`/`labels`, to reproduce pi_multik's exact train/val/test split
    (the intersection of seeds common to every k, in the first k's order),
    so the two methods train and test on precisely the same clouds. The
    config's `filtration:` block must therefore match the pi_multik classify
    config's."""
    params = cfg.method.params
    subdir = params.get("results_subdir", "vihrs")
    output_dir = results_paths.seed_dir([], subdir, seed, run_tag=run_tag)
    if is_done(output_dir) and not force:
        print(f"[{subdir} seed={seed}] already done, skipping ({output_dir}).")
        return

    if not cfg.target_label_names:
        raise ValueError(
            "classification vihrs needs `target_label_names` set to the ordered class-name list "
            "(index i == class i), matching the diagram bundle's own label_names."
        )
    if not filtrations:
        raise ValueError(
            "classification vihrs reads the per-k diagram bundles only to reproduce pi_multik's "
            "train/val/test split -- set the same `filtration:` block the pi_multik classify config uses."
        )

    label_names = tuple(cfg.target_label_names)
    k_values = [f.params.get("k") for f in filtrations]
    diagram_paths = [data_paths.diagrams([f]) for f in filtrations]
    adversarial_clouds_path = data_paths.clouds(adversarial=True)
    adversarial_diagram_paths = [data_paths.diagrams([f], adversarial=True) for f in filtrations]
    use_adv = (
        cfg.use_adversarial
        and adversarial_clouds_path.exists()
        and all(p.exists() for p in adversarial_diagram_paths)
    )

    # summary_channels: which summary functions become CNN input channels.
    # ["L"] (the default) is the published vihrs feature set; adding F/G/J
    # turns it into the "union of the standard summary functions" baseline.
    # F and G are only computed when something other than L is asked for, so
    # an L-only run touches neither the new code path nor a new cache.
    summary_channels = baselines.summstats.normalize_channels(params.get("summary_channels"))
    fg_r_max = (
        None if summary_channels == ("L",)
        else float(params.get("fg_r_max", baselines.vihrs.R_MAX))
    )

    data = baselines.vihrs.prepare_data_classify(
        data_paths.clouds(),
        adversarial_clouds_path if use_adv else None,
        diagram_paths=diagram_paths,
        adversarial_diagram_paths=adversarial_diagram_paths if use_adv else None,
        k_values=k_values,
        label_names=label_names,
        r_max=params.get("r_max", baselines.vihrs.R_MAX),
        n_r=params.get("n_r", baselines.vihrs.N_R),
        force=bool(params.get("recompute_features", False)),
        fg_r_max=fg_r_max,
    )
    baselines.vihrs.run_one_seed_classify(
        seed,
        train_features=data["train_features"],
        adversarial_features=data["adversarial_features"],
        adversarial_path=data["adversarial_path"],
        r_grid=data["r_grid"],
        output_root=output_dir.parent,
        class_names=data["class_names"],
        n_epochs=params.get("n_epochs", 200),
        batch_size=params.get("batch_size", 128),
        lr=params.get("lr", 1e-3),
        weight_decay=params.get("weight_decay", 0.0),
        early_stopping_patience=params.get("early_stopping_patience", 30),
        results_root=results_paths.root,
        run_tag=run_tag,
        summary_channels=summary_channels,
    )


def run_vihrs_method(
    cfg: RunConfig,
    seed: int,
    data_paths: DataPaths,
    results_paths: ResultsPaths,
    filtrations: list[Filtration],
    force: bool,
    run_tag: str | None = None,
) -> None:
    params = cfg.method.params
    if str(params.get("task", "params")) == "classify":
        run_vihrs_classify_method(cfg, seed, data_paths, results_paths, filtrations, force, run_tag=run_tag)
        return
    checkpoint_best = bool(params.get("checkpoint_best", False))
    subdir = "vihrs_checkpointed" if checkpoint_best else "vihrs"
    # Explicit override for off-paper exploratory runs (e.g. a longer epoch
    # budget) -- keeps "vihrs"/"vihrs_checkpointed" reserved for the
    # paper-faithful/fair-comparison variants so results dirs stay unambiguous.
    subdir = params.get("results_subdir", subdir)
    output_dir = results_paths.seed_dir([], subdir, seed, run_tag=run_tag)
    if is_done(output_dir) and not force:
        print(f"[{subdir} seed={seed}] already done, skipping ({output_dir}).")
        return

    # None (no target_label_names in the YAML) lets prepare_data adapt to
    # whatever labels this process's clouds actually carry, instead of
    # assuming the original paper's fixed 3-parameter Thomas set.
    label_names = tuple(cfg.target_label_names) if cfg.target_label_names else None
    adversarial_clouds_path = data_paths.clouds(adversarial=True)
    adversarial_path = adversarial_clouds_path if cfg.use_adversarial and adversarial_clouds_path.exists() else None

    # See run_vihrs_classify_method: ["L"] (the default) is the published
    # feature set and touches neither the F/G code path nor a new cache.
    summary_channels = baselines.summstats.normalize_channels(params.get("summary_channels"))
    fg_r_max = (
        None if summary_channels == ("L",)
        else float(params.get("fg_r_max", baselines.vihrs.R_MAX))
    )

    data = baselines.vihrs.prepare_data(
        data_paths.clouds(), adversarial_path, label_names=label_names,
        r_max=params.get("r_max", baselines.vihrs.R_MAX), n_r=params.get("n_r", baselines.vihrs.N_R),
        fg_r_max=fg_r_max,
    )
    label_names = tuple(data["label_names"])
    baselines.vihrs.run_one_seed(
        seed,
        train_records=data["train_records"],
        train_features=data["train_features"],
        adversarial_features=data["adversarial_features"],
        adversarial_path=data["adversarial_path"],
        r_grid=data["r_grid"],
        output_root=output_dir.parent,
        n_epochs=params.get("n_epochs", 20),
        batch_size=params.get("batch_size", 100),
        lr=params.get("lr", 1e-3),
        early_stopping_patience=params.get("early_stopping_patience"),
        label_names=label_names,
        checkpoint_best=checkpoint_best,
        skip_mincontrast=params.get("skip_mincontrast", False),
        results_root=results_paths.root,
        run_tag=run_tag,
        summary_channels=summary_channels,
    )


def run_classical_baseline(
    method_name: str,
    cfg: RunConfig,
    seed: int,
    data_paths: DataPaths,
    results_paths: ResultsPaths,
    force: bool,
    run_tag: str | None = None,
) -> None:
    module = getattr(baselines, method_name)
    output_dir = results_paths.seed_dir([], method_name, seed, run_tag=run_tag)
    if is_done(output_dir) and not force:
        print(f"[{method_name} seed={seed}] already done, skipping ({output_dir}).")
        return

    clouds = load_pickle(data_paths.clouds())
    n = len(clouds)
    label_names = list(cfg.target_label_names) if cfg.target_label_names else list(clouds[0]["params"].keys())

    all_targets = np.array([[c["params"][name] for name in label_names] for c in clouds], dtype=float)
    train_idx, _, test_idx = train_val_test_indices(n, seed)
    label_norm = baselines.vihrs.fit_log_zscore(all_targets[train_idx])

    n_starts = int(cfg.method.params.get("n_starts", 10))
    rng = np.random.default_rng(seed)

    # Optional contrast-exponent override, threaded through the same way
    # n_starts is -- absent by default, so every existing mincontrast*/palm
    # config keeps using fit_multistart's own hardcoded c (0.25 for the
    # K-based fits, 1.0 for the g-based ones). Lets a c-sweep (e.g. the
    # nested-Thomas g fit's outstanding "re-run with c swept" experiment)
    # run via `--set method.params.c=<value> --run-tag <label>` with no new
    # config file, instead of only being reachable through each module's
    # standalone __main__ CLI.
    fit_kwargs = {}
    if "c" in cfg.method.params:
        fit_kwargs["c"] = float(cfg.method.params["c"])

    raw_predictions = np.full((len(test_idx), len(label_names)), np.nan)
    n_fit = 0
    for row, i in enumerate(test_idx):
        rec = clouds[i]
        points = np.asarray(rec["points"], dtype=float)
        if len(points) < 5:
            continue
        reg = rec.get("region", {})
        low = np.asarray(reg.get("low", [0.0, 0.0]), dtype=float)
        high = np.asarray(reg.get("high", [1.0, 1.0]), dtype=float)
        points_unit = module.crop_and_rescale(points, low, high, rng=rng)
        result = module.fit_multistart(points_unit, n_starts=n_starts, rng=rng, **fit_kwargs)
        raw_predictions[row] = [result.get(name, np.nan) for name in label_names]
        n_fit += 1
    print(f"[{method_name} seed={seed}] fit {n_fit}/{len(test_idx)} test clouds.")

    raw_truth = all_targets[test_idx]
    valid = np.isfinite(raw_predictions).all(axis=1) & (raw_predictions > 0).all(axis=1)
    pred_std = np.full_like(raw_predictions, np.nan)
    pred_std[valid] = baselines.vihrs.apply_log_zscore(raw_predictions[valid], label_norm)
    truth_std = baselines.vihrs.apply_log_zscore(raw_truth, label_norm)

    per_target_mse = np.nanmean((pred_std - truth_std) ** 2, axis=0)
    test_loss = float(np.nanmean(per_target_mse))
    test_loss_per_target = dict(zip(label_names, per_target_mse.tolist()))
    print(f"[{method_name} seed={seed}] test loss (standardized) {test_loss:.4f}")

    cfg_dict = {"task": "params", "method": method_name, "seed": seed, "n_starts": n_starts, "results_root": str(results_paths.root)}
    if run_tag:
        cfg_dict["run_tag"] = run_tag
    save_results(
        output_dir, model=None, best_state=None, history={}, cfg=cfg_dict,
        test_loss=test_loss, label_names=label_names, label_norm=label_norm,
        test_loss_per_target=test_loss_per_target,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path, help="RunConfig YAML path")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="path.to.field=value")
    parser.add_argument(
        "--seed", type=int, default=None,
        help="run only this seed (for SLURM array jobs); default: every seed in the config",
    )
    parser.add_argument("--force", action="store_true", help="retrain even if results.pt already exists")
    parser.add_argument(
        "--run-tag", default=None,
        help="write results under <method>/_runs/<run-tag>/ instead of the default path, so a labeled "
             "variant (e.g. a candidate model change) lives alongside the current results instead of "
             "overwriting them -- compare the two later with evaluate.py's method@run-tag syntax",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config, overrides=args.overrides)
    if cfg.method is None:
        raise ValueError(f"{args.config} has no `method:` section -- nothing to train.")

    data_paths = DataPaths(cfg.process.name, root=cfg.data_root or DEFAULT_DATA_ROOT)
    results_paths = ResultsPaths(cfg.process.name, root=cfg.results_root or DEFAULT_RESULTS_ROOT)
    filtrations = build_filtrations(cfg)
    bifiltrations = build_bifiltrations(cfg)
    # A config sets `filtration:` (single-parameter) or `bifiltration:`
    # (mph_*), never both -- whichever is non-empty is what dataset/results
    # paths get tagged with below. Every existing config only ever sets
    # `filtration:`, so this is a no-op (path_filtrations == filtrations)
    # for all of them; bifiltrations only becomes non-empty for configs
    # like nested_thomas_mph.yaml that couldn't reach run_experiment_method
    # at all before (no `method:` block).
    path_filtrations = bifiltrations or filtrations

    seeds = [args.seed] if args.seed is not None else cfg.seeds

    for seed in seeds:
        print(f"\n{'#' * 90}\n### {cfg.method.name} | {cfg.process.name} | seed {seed}\n{'#' * 90}")
        try:
            if cfg.method.name == "vihrs":
                run_vihrs_method(cfg, seed, data_paths, results_paths, filtrations, args.force, run_tag=args.run_tag)
            elif cfg.method.name in CLASSICAL_BASELINE_NAMES:
                run_classical_baseline(cfg.method.name, cfg, seed, data_paths, results_paths, args.force, run_tag=args.run_tag)
            else:
                run_experiment_method(cfg, seed, data_paths, results_paths, path_filtrations, args.force, run_tag=args.run_tag)
        except Exception:
            print(f"[{cfg.method.name} seed={seed}] FAILED:")
            traceback.print_exc()
            continue

    print(f"\nAll {len(seeds)} seed(s) attempted.")


if __name__ == "__main__":
    main()