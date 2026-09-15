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

DV3 (data.source: dv3 in the config; see cloudforger.evaluation.dv3):
training reads data/dv3/<train_set>/<group>/ with the train/val split fixed
at generation (no in-pool test slice), and every method is evaluated on
each of data.eval_sets (A, B, C) separately, writing one per-pattern
predictions_<set>.npz per set next to results.pt. Classical baselines fit
every pattern of every evaluation set -- the fits are deterministic (one
rng stream per case_id) and cached once under <method>/_fits/, so each seed
directory holds identical predictions, which is what a deterministic
method's seed-paired comparison should see. scripts/evaluate_regimes.py
turns the bundles into per-regime tables.

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
import zlib
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cloudforger import baselines
from cloudforger.config import RunConfig, load_config
from cloudforger.core.io import load_pickle
from cloudforger.core.splits import resolve_split, train_val_test_indices
from cloudforger.data_generation.filtration import BIFILTRATION_REGISTRY, REGISTRY as FILTRATION_REGISTRY
from cloudforger.data_generation.filtration.base import Filtration
from cloudforger.experiments.base import build_experiment
from cloudforger.experiments.common import MultiSourceExperiment, save_results
from cloudforger.evaluation import dv3
from cloudforger.paths import DEFAULT_DATA_ROOT, DEFAULT_RESULTS_ROOT, PROJECT_ROOT, DataPaths, ExplicitTag, ResultsPaths, is_done

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

# Classical estimators report their own parameter names; DV3 records carry
# the model's (docs/generation_procedure.tex, Table "Models and prior").
# Nested Thomas's `sigma` is the inner (child) scale sigma2, `sigma1` the outer.
_THOMAS_NAMES = {"parent_intensity": "kappa", "mean_offspring": "mu", "cluster_scale": "sigma"}
_NESTED_NAMES = {"parent_intensity": "kappa", "meta_offspring": "mu1", "meta_cluster_scale": "sigma1",
                 "mean_offspring": "mu2", "cluster_scale": "sigma"}
DV3_CLASSICAL_NAMES = {
    "mincontrast": _THOMAS_NAMES, "mincontrast_g": _THOMAS_NAMES, "palm": _THOMAS_NAMES,
    "mincontrast_nested": _NESTED_NAMES, "mincontrast_g_nested": _NESTED_NAMES,
}


def _dv3_root(cfg: RunConfig) -> Path:
    if cfg.data.root is None:
        return dv3.DEFAULT_DV3_ROOT
    root = Path(cfg.data.root)
    return root if root.is_absolute() else PROJECT_ROOT / root


def resolve_data_paths(cfg: RunConfig) -> tuple[DataPaths, dict[str, DataPaths]]:
    """(training-pool DataPaths, {eval set -> DataPaths}). Legacy configs get
    data/<process.name>/ and no eval sets, exactly as before."""
    if not cfg.data.is_dv3:
        return DataPaths(cfg.process.name, root=cfg.data_root or DEFAULT_DATA_ROOT), {}
    root, group = _dv3_root(cfg), cfg.data.group or cfg.process.name
    unknown = [s for s in cfg.data.eval_sets if s not in dv3.EVAL_SETS]
    if unknown:
        raise ValueError(f"data.eval_sets {unknown} are not DV3 evaluation sets {list(dv3.EVAL_SETS)}")
    train = dv3.data_paths(cfg.data.train_set, group, root)
    if not train.clouds().exists():
        raise FileNotFoundError(
            f"no DV3 training clouds at {train.clouds()} -- generate DV3 "
            f"(scripts/generation/dv3.py) or, for classification, build the merged bundle "
            f"(scripts/processing/dv3_classification_bundle.py)."
        )
    return train, {s: dv3.data_paths(s, group, root) for s in cfg.data.eval_sets}


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
    if cfg.data.is_dv3:
        cfg_dict["split"] = "dv3"
        cfg_dict["data_group"] = cfg.data.group or cfg.process.name
        cfg_dict["eval_sets"] = list(cfg.data.eval_sets)
        if cfg.data.split_reshuffle_seed is not None:
            cfg_dict["split_reshuffle_seed"] = int(cfg.data.split_reshuffle_seed)
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
    eval_data_paths: dict[str, DataPaths] | None = None,
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

    if cfg.data.is_dv3:
        # Only methods whose run() honours the DV3 split and writes the
        # per-set prediction bundles may run on DV3; anything else would
        # silently fall back to a random split of the training pool and
        # produce no regime output at all.
        if not getattr(exp, "supports_dv3", False):
            raise NotImplementedError(
                f"method {cfg.method.name!r} ({type(exp).__name__}) is not wired for data.source: dv3 yet. "
                "DV3-ready: the pi_multik family (pi_multik, _towers, _scaleconv, _earlyfusion, _dimsplit), "
                "vihrs (L / L+F+G+J), and the classical baselines. Port it by honouring cfg['split'] == "
                "'dv3' and eval_paths the way experiments/pi_multik/pi_multik.py does."
            )
        dataset_paths = _multi_source_dataset_paths(exp, cfg, data_paths, filtrations, adversarial=False)
        eval_paths = {
            set_name: _multi_source_dataset_paths(exp, cfg, dp, filtrations, adversarial=False)
            for set_name, dp in (eval_data_paths or {}).items()
        }
        exp.run(dataset_paths, output_dir, adversarial_paths=None, eval_paths=eval_paths)
    elif is_multi_source:
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
    eval_data_paths: dict[str, DataPaths] | None = None,
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
    prepare_only = bool(params.get("prepare_only", False))
    if is_done(output_dir) and not force and not prepare_only:
        print(f"[{subdir} seed={seed}] already done, skipping ({output_dir}).")
        return

    if not cfg.target_label_names:
        raise ValueError(
            "classification vihrs needs `target_label_names` set to the ordered class-name list "
            "(index i == class i), matching the diagram bundle's own label_names."
        )
    if not filtrations and not cfg.data.is_dv3:
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
        not cfg.data.is_dv3
        and cfg.use_adversarial
        and adversarial_clouds_path.exists()
        and all(p.exists() for p in adversarial_diagram_paths)
    )
    # DV3: the split is the generator's, so the diagram bundles are only
    # consulted (when a filtration is configured) to restrict every set to
    # exactly the clouds the PH methods could use; without one, all clouds.
    eval_sets_spec = None
    if cfg.data.is_dv3:
        eval_sets_spec = {
            set_name: {"clouds": dp.clouds(), "diagrams": [dp.diagrams([f]) for f in filtrations]}
            for set_name, dp in (eval_data_paths or {}).items()
        }

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
        eval_sets=eval_sets_spec,
        feature_jobs=int(params.get("feature_jobs", 1)),
    )
    if prepare_only:
        print(f"[{subdir}] prepare_only: feature caches in place, not training.")
        return
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
        use_dv3_split=cfg.data.is_dv3,
        split_reshuffle_seed=cfg.data.split_reshuffle_seed,
        eval_features=data.get("eval_features") or None,
        method_name=subdir,
    )


def run_vihrs_method(
    cfg: RunConfig,
    seed: int,
    data_paths: DataPaths,
    results_paths: ResultsPaths,
    filtrations: list[Filtration],
    force: bool,
    run_tag: str | None = None,
    eval_data_paths: dict[str, DataPaths] | None = None,
) -> None:
    params = cfg.method.params
    if str(params.get("task", "params")) == "classify":
        run_vihrs_classify_method(cfg, seed, data_paths, results_paths, filtrations, force, run_tag=run_tag,
                                  eval_data_paths=eval_data_paths)
        return
    checkpoint_best = bool(params.get("checkpoint_best", False))
    subdir = "vihrs_checkpointed" if checkpoint_best else "vihrs"
    # Explicit override for off-paper exploratory runs (e.g. a longer epoch
    # budget) -- keeps "vihrs"/"vihrs_checkpointed" reserved for the
    # paper-faithful/fair-comparison variants so results dirs stay unambiguous.
    subdir = params.get("results_subdir", subdir)
    output_dir = results_paths.seed_dir([], subdir, seed, run_tag=run_tag)
    # prepare_only: build (or verify) the feature caches and stop -- the
    # CPU-side warm-up that lets GPU array tasks start training at once.
    prepare_only = bool(params.get("prepare_only", False))
    if is_done(output_dir) and not force and not prepare_only:
        print(f"[{subdir} seed={seed}] already done, skipping ({output_dir}).")
        return

    # None (no target_label_names in the YAML) lets prepare_data adapt to
    # whatever labels this process's clouds actually carry, instead of
    # assuming the original paper's fixed 3-parameter Thomas set.
    label_names = tuple(cfg.target_label_names) if cfg.target_label_names else None
    adversarial_clouds_path = data_paths.clouds(adversarial=True)
    adversarial_path = (
        adversarial_clouds_path
        if not cfg.data.is_dv3 and cfg.use_adversarial and adversarial_clouds_path.exists() else None
    )

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
        force=bool(params.get("recompute_features", False)),
        eval_paths={s: dp.clouds() for s, dp in (eval_data_paths or {}).items()} or None,
        feature_jobs=int(params.get("feature_jobs", 1)),
    )
    if prepare_only:
        print(f"[{subdir}] prepare_only: feature caches in place, not training.")
        return
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
        # Under DV3 the in-run "bonus" minimum-contrast comparison is off: it
        # would refit every set-A cloud under the legacy parameter names.
        # The standalone mincontrast* configs cover it, with regime tables.
        skip_mincontrast=params.get("skip_mincontrast", False) or cfg.data.is_dv3,
        results_root=results_paths.root,
        run_tag=run_tag,
        summary_channels=summary_channels,
        split_labels=(np.array([r.get("split") or "" for r in data["train_records"]], dtype=object)
                      if cfg.data.is_dv3 else None),
        split_reshuffle_seed=cfg.data.split_reshuffle_seed,
        eval_features=data.get("eval_features") or None,
        eval_records=data.get("eval_records") or None,
        method_name=subdir,
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


# ---------------------------------------------------------------------------
# Classical baselines on DV3
# ---------------------------------------------------------------------------

def _case_rng(case_id: str) -> np.random.Generator:
    """One multistart stream per pattern, from its identity alone -- so a fit
    does not depend on which other patterns ran, in what order, or in which
    worker, and the cached fits are reusable by every seed."""
    return np.random.default_rng(zlib.crc32(case_id.encode()))


def _fit_one(task: tuple[str, str, np.ndarray, int, dict[str, Any]]) -> dict[str, float] | None:
    module_name, case_id, points, n_starts, fit_kwargs = task
    module = getattr(baselines, module_name)
    if len(points) < 5:
        return None
    try:
        # DV3 patterns already live on W = [0,1]^2, so they are fit as-is:
        # crop_and_rescale's sub-window (for n > 900) would return estimates in
        # the rescaled window's units -- kappa off by window^2, sigma by
        # 1/window -- which the legacy path never converted back.
        return module.fit_multistart(points, n_starts=n_starts, rng=_case_rng(case_id), **fit_kwargs)
    except Exception as exc:  # noqa: BLE001 -- a failed fit is a reportable outcome (NaN row), not a crash
        print(f"  [{module_name}] fit failed for {case_id}: {exc!r}")
        return None


def _dv3_classical_fits(
    method_name: str, clouds: list[dict[str, Any]], label_names: list[str], cache_path: Path,
    n_starts: int, fit_kwargs: dict[str, Any], jobs: int,
) -> np.ndarray:
    """(N, T) raw estimates for `clouds` in DV3 names, NaN where the fit failed
    or the estimator has no such parameter. Cached by case_id + settings."""
    name_map = DV3_CLASSICAL_NAMES.get(method_name, {})
    ids = np.array([str(c["case_id"]) for c in clouds], dtype=np.str_)
    settings = f"{method_name}|{n_starts}|{sorted(fit_kwargs.items())}|{','.join(label_names)}"
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as z:
            if np.array_equal(z["case_id"], ids) and str(z["settings"]) == settings:
                print(f"  [{method_name}] cached fits <- {cache_path}")
                return z["estimates"]
    tasks = [(method_name, str(c["case_id"]), np.asarray(c["points"], dtype=float), n_starts, fit_kwargs)
             for c in clouds]
    if jobs > 1:
        with get_context("spawn").Pool(jobs) as pool:
            results = pool.map(_fit_one, tasks, chunksize=16)
    else:
        results = [_fit_one(t) for t in tasks]
    estimates = np.full((len(clouds), len(label_names)), np.nan)
    for i, res in enumerate(results):
        if res is None:
            continue
        renamed = {name_map.get(k, k): v for k, v in res.items()}
        estimates[i] = [renamed.get(name, np.nan) for name in label_names]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, case_id=ids, estimates=estimates, settings=np.asarray(settings))
    n_ok = int(np.isfinite(estimates).all(axis=1).sum())
    print(f"  [{method_name}] fit {n_ok}/{len(clouds)} patterns -> {cache_path}")
    return estimates


def run_classical_baseline_dv3(
    method_name: str,
    cfg: RunConfig,
    seed: int,
    data_paths: DataPaths,
    eval_data_paths: dict[str, DataPaths],
    results_paths: ResultsPaths,
    force: bool,
    run_tag: str | None = None,
) -> None:
    """Fit every pattern of every DV3 evaluation set and write the same
    prediction bundles the trained methods do.

    Standardization: log + z-score with statistics fit on the DV3 TRAINING
    rows (split == "train") of the training pool -- the exact label_norm a
    trained method fits -- so the normalised loss is on one scale for every
    method. A non-positive or failed estimate stays NaN and is counted as a
    failure per regime cell rather than dropped."""
    from cloudforger.experiments.common import record_eval_set

    output_dir = results_paths.seed_dir([], method_name, seed, run_tag=run_tag)
    if is_done(output_dir) and not force:
        print(f"[{method_name} seed={seed}] already done, skipping ({output_dir}).")
        return
    if not cfg.target_label_names:
        raise ValueError(f"{method_name} on DV3 needs target_label_names (DV3 parameter names, e.g. kappa, mu, sigma).")
    label_names = list(cfg.target_label_names)
    unknown = [n for n in label_names if n not in DV3_CLASSICAL_NAMES.get(method_name, {}).values()]
    if unknown:
        raise ValueError(f"{method_name} does not estimate {unknown}; it reports "
                         f"{sorted(DV3_CLASSICAL_NAMES.get(method_name, {}).values())}.")

    train_clouds = load_pickle(data_paths.clouds())
    train_targets = np.array([[c["params"][n] for n in label_names] for c in train_clouds], dtype=float)
    train_idx, _, _ = resolve_split(
        len(train_clouds), seed, [c.get("split") or "" for c in train_clouds],
        reshuffle_seed=cfg.data.split_reshuffle_seed,
    )
    label_norm = baselines.vihrs.fit_log_zscore(train_targets[train_idx])

    params = cfg.method.params
    n_starts = int(params.get("n_starts", 10))
    fit_kwargs = {"c": float(params["c"])} if "c" in params else {}
    jobs = int(params.get("jobs", 1))
    fits_dir = results_paths.method_dir([], method_name, run_tag=run_tag) / "_fits"

    eval_sets: dict[str, dict[str, Any]] = {}
    for set_name, dp in eval_data_paths.items():
        clouds = load_pickle(dp.clouds())
        est = _dv3_classical_fits(method_name, clouds, label_names, fits_dir / f"{set_name}.npz",
                                  n_starts, fit_kwargs, jobs)
        truth_raw = np.array([[c["params"][n] for n in label_names] for c in clouds], dtype=float)
        valid = np.isfinite(est).all(axis=1) & (est > 0).all(axis=1)
        pred_std = np.full_like(est, np.nan)
        pred_std[valid] = baselines.vihrs.apply_log_zscore(est[valid], label_norm)
        eval_sets[set_name] = record_eval_set(
            output_dir, set_name, task="params", outputs=pred_std,
            truth=baselines.vihrs.apply_log_zscore(truth_raw, label_norm),
            case_id=np.array([c["case_id"] for c in clouds], dtype=object),
            family=np.array([c["process"] for c in clouds], dtype=object),
            names=label_names, label_norm=label_norm, truth_raw=truth_raw, method=method_name, seed=seed,
        )
        s_ = eval_sets[set_name]
        print(f"[{method_name} seed={seed}] eval set {set_name}: loss {s_['loss']:.4f} "
              f"({s_['n_ok']}/{s_['n']} fits usable)")

    cfg_dict = {"task": "params", "method": method_name, "seed": seed, "n_starts": n_starts,
                "results_root": str(results_paths.root), "split": "dv3", "deterministic": True, **fit_kwargs}
    if run_tag:
        cfg_dict["run_tag"] = run_tag
    a = eval_sets.get("A", {})
    save_results(
        output_dir, model=None, best_state=None, history={}, cfg=cfg_dict,
        test_loss=a.get("loss"), label_names=label_names, label_norm=label_norm,
        test_loss_per_target=a.get("loss_per_target"), eval_sets=eval_sets,
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

    data_paths, eval_data_paths = resolve_data_paths(cfg)
    if cfg.data.is_dv3:
        print(f"DV3: training pool {data_paths.process_dir}; evaluation sets "
              + ", ".join(f"{s} -> {dp.process_dir}" for s, dp in eval_data_paths.items()))
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
                run_vihrs_method(cfg, seed, data_paths, results_paths, filtrations, args.force, run_tag=args.run_tag,
                                 eval_data_paths=eval_data_paths)
            elif cfg.method.name in CLASSICAL_BASELINE_NAMES and cfg.data.is_dv3:
                run_classical_baseline_dv3(cfg.method.name, cfg, seed, data_paths, eval_data_paths, results_paths,
                                           args.force, run_tag=args.run_tag)
            elif cfg.method.name in CLASSICAL_BASELINE_NAMES:
                run_classical_baseline(cfg.method.name, cfg, seed, data_paths, results_paths, args.force, run_tag=args.run_tag)
            else:
                run_experiment_method(cfg, seed, data_paths, results_paths, path_filtrations, args.force,
                                      run_tag=args.run_tag, eval_data_paths=eval_data_paths)
        except Exception:
            print(f"[{cfg.method.name} seed={seed}] FAILED:")
            traceback.print_exc()
            continue

    print(f"\nAll {len(seeds)} seed(s) attempted.")


if __name__ == "__main__":
    main()