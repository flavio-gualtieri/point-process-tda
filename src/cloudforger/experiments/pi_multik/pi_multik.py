# src/cloudforger/experiments/pi_multik/pi_multik.py

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, Subset, TensorDataset

from cloudforger.baselines import summstats, vihrs
from cloudforger.core.diagram import PersistenceDiagram
from cloudforger.core.io import intersect_seeds
from cloudforger.core.records import load_diagrams
from cloudforger.core.splits import train_val_test_indices
from cloudforger.encoders.encoder_bank import EncoderBank
from cloudforger.encoders.scaleconv_pi import ConvFusion
from cloudforger.experiments.base import register
from cloudforger.experiments.common import (
    MultiSourceExperiment,
    apply_zscore,
    fit_zscore,
    prepare_device,
    save_results,
)
from cloudforger.models.heads.classifier import ClassificationHead
from cloudforger.models.heads.paramest import ParameterEstimator
from cloudforger.training.train import evaluate, evaluate_per_target, train_one_epoch
from cloudforger.vectorization.persistence_images.calibrated import build_calibrated_imager
from cloudforger.vectorization.persistence_images.multi_channel import MultiChannelImager
from cloudforger.vectorization.scalar_features import REGISTRY as FEATURE_REGISTRY


def _load_pickle(path: Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def _classification_n_points(
    clouds_path: Path, ref_bundle: dict[str, Any], common_seeds: np.ndarray
) -> np.ndarray:
    """Join n(x) for the classification task, where load_multik_split's plain
    {seed -> n_points} join needs care: the per-k diagram bundles carry
    GLOBALLY-unique seeds (class_index * seed_offset + original_seed, so the
    five per-process seed ranges don't collide). The sibling classification
    clouds.pkl is a {"clouds": [PointCloud], "labels": [...]} bundle that has
    shipped with EITHER seed convention -- cloud.seed already global (== the
    diagram seed), or cloud.seed the original per-process seed (unique only
    within a class, disambiguated by the parallel `labels` array). Handle
    both: direct seed lookup when clouds.pkl's seeds are globally unique,
    else a (class_index, original_seed) decode via the bundle's seed_offset."""
    payload = _load_pickle(clouds_path)
    clouds = payload["clouds"] if isinstance(payload, dict) else payload
    cloud_labels = (
        np.asarray(payload["labels"]).reshape(-1)
        if isinstance(payload, dict) and "labels" in payload else None
    )
    seed_offset = int(ref_bundle.get("config", {}).get("seed_offset", 0))

    seeds = [int(c["seed"] if isinstance(c, dict) else c.seed) for c in clouds]
    n_pts = [int(c["n_points"] if isinstance(c, dict) else c.n_points) for c in clouds]
    globally_unique = len(set(seeds)) == len(seeds)

    if globally_unique:
        by_seed = dict(zip(seeds, n_pts))
        return np.asarray([by_seed[int(s)] for s in common_seeds], dtype=np.float64)

    if cloud_labels is None or not seed_offset:
        raise KeyError(
            f"{clouds_path} has non-unique cloud seeds but no `labels`/seed_offset to "
            "disambiguate them against the globally-unique diagram seeds."
        )
    by_cls_orig = {(int(cloud_labels[i]), seeds[i]): n_pts[i] for i in range(len(seeds))}
    return np.asarray(
        [by_cls_orig[(int(s) // seed_offset, int(s) % seed_offset)] for s in common_seeds],
        dtype=np.float64,
    )


# L(r)-r side-channel radii start at 0.01, not 0: K_hat counts very few pairs
# at tiny r, so L is noisiest exactly there. Upper end is the r_max the vihrs
# baseline uses (spatstat's side/4 default for the unit square).
_LFUNC_R_MIN = 0.01
# Basenames baselines/vihrs.py caches L(r)-r under: the parameter-estimation
# path uses the first, the classification path the second.
_LFUNC_CACHE_SUFFIXES = (".lfunc_cache.npz", ".lfunc_classify_cache.npz")

# Basenames baselines/vihrs.py caches the F/G/J channels under when
# fg_r_max is set (see its prepare_data / prepare_data_classify). NNNN is
# round(fg_r_max * 1000); 0250 is the value that won the classification
# baseline sweep (slurm/summstats_classify.sh), so it is the default here.
_FG_CACHE_TEMPLATES = (".summ_fg{tag}_cache.npz", ".summ_fg{tag}_classify_cache.npz")
_FG_R_MIN = 0.01

# entropy_norms key under which build_extra stashes the curve-branch
# per-channel normalization (fit on train, applied frozen to adversarial).
_CURVE_NORM_KEY = "__curves__"


def _load_lfunc_cols(
    clouds_path: Path, common_seeds: np.ndarray, n_radii: int
) -> dict[str, np.ndarray]:
    """{column name -> (N,) array} of L(r)-r sampled at `n_radii` log-spaced
    radii, joined to `common_seeds` from the vihrs L cache.

    This is the side-channel that makes the L-reparameterized filtration
    (data_generation/filtration/lfunc.py) testable: that transform moves the
    second-order trend OUT of the diagram, so the trend has to be fed back in
    somewhere or the representation is strictly weaker. Column names are
    index-numbered (lfunc_r00..) so `sorted()` orders them by radius, giving a
    stable column order between the train and adversarial calls."""
    cache_path = None
    for suffix in _LFUNC_CACHE_SUFFIXES:
        candidate = clouds_path.parent / (clouds_path.stem + suffix)
        if candidate.exists():
            cache_path = candidate
            break
    if cache_path is None:
        raise FileNotFoundError(
            f"include_lfunc is set but no L cache next to {clouds_path} (looked for "
            + ", ".join(clouds_path.stem + s for s in _LFUNC_CACHE_SUFFIXES)
            + "). Run the vihrs baseline on this process once to build it."
        )

    z = np.load(cache_path)
    r_grid = np.asarray(z["r_grid"], dtype=np.float64)
    l_minus_r = np.asarray(z["l_minus_r"], dtype=np.float64)
    by_seed = {int(s): i for i, s in enumerate(np.asarray(z["cloud_seeds"], dtype=np.int64))}
    missing = [int(s) for s in common_seeds if int(s) not in by_seed]
    if missing:
        raise KeyError(
            f"{len(missing)}/{len(common_seeds)} seeds absent from {cache_path} "
            f"(first few: {missing[:5]}) -- L cache and diagram bundle disagree on the seed convention."
        )

    rows = np.array([by_seed[int(s)] for s in common_seeds], dtype=np.int64)
    radii = np.geomspace(_LFUNC_R_MIN, float(r_grid[-1]), n_radii)
    sampled = np.stack([np.interp(radii, r_grid, l_minus_r[i]) for i in rows])  # (N, n_radii)
    return {f"lfunc_r{j:02d}": sampled[:, j] for j in range(n_radii)}


def _load_fg_cols(
    clouds_path: Path, common_seeds: np.ndarray, n_radii: int, fg_r_max: float
) -> dict[str, np.ndarray]:
    """{column name -> (N,) array} of F(r) and G(r) sampled at `n_radii`
    log-spaced radii each, joined to `common_seeds` from the vihrs F/G/J cache.

    Exact twin of _load_lfunc_cols, one cache and two functions instead of
    one: F is the empty-space function and G the nearest-neighbour distance
    function (see baselines/summstats.py). Together with L they are the
    "union of the standard summary functions" that beat PH (+) L on 4-way
    classification, so this is what lets pi_multik ask whether persistence
    images add anything ON TOP of that union rather than on top of L alone.

    Emits 2 * n_radii columns, index-numbered per function so `sorted()`
    keeps a stable, function-grouped order between the train and adversarial
    calls -- the same contract _load_lfunc_cols relies on.
    """
    tag = f"{round(float(fg_r_max) * 1000):04d}"
    cache_path = None
    for template in _FG_CACHE_TEMPLATES:
        candidate = clouds_path.parent / (clouds_path.stem + template.format(tag=tag))
        if candidate.exists():
            cache_path = candidate
            break
    if cache_path is None:
        raise FileNotFoundError(
            f"include_fgfunc is set but no F/G cache next to {clouds_path} (looked for "
            + ", ".join(clouds_path.stem + t.format(tag=tag) for t in _FG_CACHE_TEMPLATES)
            + f"). Build it with slurm/summstats_featurize*.sh at fg_r_max={fg_r_max}."
        )

    z = np.load(cache_path)
    missing_keys = [k for k in ("fg_grid", "f_func", "g_func") if k not in z]
    if missing_keys:
        raise KeyError(f"{cache_path} is an L-only cache (missing {missing_keys}); rebuild it.")

    fg_grid = np.asarray(z["fg_grid"], dtype=np.float64)
    by_seed = {int(s): i for i, s in enumerate(np.asarray(z["cloud_seeds"], dtype=np.int64))}
    missing = [int(s) for s in common_seeds if int(s) not in by_seed]
    if missing:
        raise KeyError(
            f"{len(missing)}/{len(common_seeds)} seeds absent from {cache_path} "
            f"(first few: {missing[:5]}) -- F/G cache and diagram bundle disagree on the seed convention."
        )

    rows = np.array([by_seed[int(s)] for s in common_seeds], dtype=np.int64)
    radii = np.geomspace(_FG_R_MIN, float(fg_grid[-1]), n_radii)

    cols: dict[str, np.ndarray] = {}
    for prefix, key in (("ffunc", "f_func"), ("gfunc", "g_func")):
        curves = np.asarray(z[key], dtype=np.float64)
        sampled = np.stack([np.interp(radii, fg_grid, curves[i]) for i in rows])  # (N, n_radii)
        cols.update({f"{prefix}_r{j:02d}": sampled[:, j] for j in range(n_radii)})
    return cols


def _load_curve_stack(
    clouds_path: Path, common_seeds: np.ndarray, channels: tuple[str, ...], fg_r_max: float
) -> np.ndarray:
    """(N, C, m) float32 stack of FULL summary-function curves for the
    two-branch model's 1-D CNN branch, joined to `common_seeds`.

    Unlike _load_lfunc_cols / _load_fg_cols, nothing is subsampled: the curve
    branch reads every one of the m radii, exactly as the vihrs L+F+G baseline
    does, so "does the PH branch add anything" is a clean ablation of one
    model rather than a comparison between a coarse MLP side vector and a
    full-resolution CNN. Reads the same vihrs F/G/J cache (which also holds
    L on its own r_grid of the same length m).
    """
    tag = f"{round(float(fg_r_max) * 1000):04d}"
    cache_path = None
    for template in _FG_CACHE_TEMPLATES:
        candidate = clouds_path.parent / (clouds_path.stem + template.format(tag=tag))
        if candidate.exists():
            cache_path = candidate
            break
    if cache_path is None:
        raise FileNotFoundError(
            f"curve_channels is set but no summary-function cache next to {clouds_path} (looked for "
            + ", ".join(clouds_path.stem + t.format(tag=tag) for t in _FG_CACHE_TEMPLATES)
            + f"). Build it with slurm/summstats_featurize*.sh at fg_r_max={fg_r_max}."
        )

    z = np.load(cache_path)
    keys = [summstats.CHANNEL_KEYS[c] for c in channels]
    missing_keys = [k for k in keys if k not in z]
    if missing_keys:
        raise KeyError(f"{cache_path} lacks curve channel(s) {missing_keys}; rebuild it.")

    by_seed = {int(s): i for i, s in enumerate(np.asarray(z["cloud_seeds"], dtype=np.int64))}
    missing = [int(s) for s in common_seeds if int(s) not in by_seed]
    if missing:
        raise KeyError(
            f"{len(missing)}/{len(common_seeds)} seeds absent from {cache_path} "
            f"(first few: {missing[:5]}) -- curve cache and diagram bundle disagree on the seed convention."
        )
    rows = np.array([by_seed[int(s)] for s in common_seeds], dtype=np.int64)
    return np.stack([np.asarray(z[k], dtype=np.float32)[rows] for k in keys], axis=1)


# Reserved key under which the fitted L-curve PCA basis is stashed in the
# per-column norm dict, so it is fit on train rows and applied frozen to the
# adversarial population like every other normalization here.
_LFUNC_PCA_KEY = "__lfunc_pca__"


def _fit_lfunc_pca(train_matrix: np.ndarray, n_components: int) -> dict[str, Any]:
    """Centre + SVD + whiten the L(r)-r columns on the TRAIN rows only.

    The columns are samples of one smooth curve, so they are severely
    collinear: over the design distribution the correlation matrix has
    condition number ~5e3 and its top two eigenvalues carry ~93% of the
    variance, leaving six near-null directions. Feeding those raw to a linear
    head under weight decay is badly conditioned, and did in fact produce
    bimodal seed outcomes on classification (3 seeds ~0.88, 2 seeds ~0.80).
    Projecting onto the leading whitened components fixes the conditioning
    while keeping the interpretable content -- for a Neyman-Scott design the
    leading components are essentially clustering amplitude and clustering
    scale."""
    mu = train_matrix.mean(axis=0)
    centred = train_matrix - mu
    k = int(min(n_components, centred.shape[1], max(centred.shape[0] - 1, 1)))
    _, sv, vt = np.linalg.svd(centred, full_matrices=False)
    scale = sv[:k] / np.sqrt(max(centred.shape[0] - 1, 1))
    total = float((sv ** 2).sum())
    return {
        "mean": mu.tolist(),
        "components": vt[:k].tolist(),
        "scale": np.where(scale > 1e-12, scale, 1.0).tolist(),
        "explained_variance_ratio": ((sv[:k] ** 2) / total).tolist() if total > 0 else [],
    }


def _apply_lfunc_pca(matrix: np.ndarray, pca: dict[str, Any]) -> list[np.ndarray]:
    proj = (matrix - np.asarray(pca["mean"])) @ np.asarray(pca["components"]).T
    proj = proj / np.asarray(pca["scale"])
    return [proj[:, j].astype(np.float32) for j in range(proj.shape[1])]


def load_multik_split(
    k_values: list[int],
    diagram_paths: list[Path],
    clouds_path: Path,
    label_names: tuple[str, ...] | None,
    tag: str,
    homology_dims: tuple[int, ...] = (0, 1),
    task: str = "params",
    lfunc_n_radii: int = 0,
    fg_n_radii: int = 0,
    fg_r_max: float = 0.25,
    curve_channels: tuple[str, ...] = (),
) -> dict[str, Any] | None:
    """Load and seed-align each k's diagrams.pkl (NOT a precomputed image
    file -- see module docstring). Persistence entropy is computed here,
    directly from the diagrams (calibration-free, so no leakage concern);
    persistence IMAGES are deliberately not built here -- that's
    build_pi_tensor's job, once a train/val/test split exists to calibrate
    against.

    task="params" (default): targets are the log-normalizable regression
    columns selected from the bundle's (N, n_labels) `labels` matrix by name.
    task="classify": the bundle's `labels` is instead a 1-D integer
    class-index vector (see scripts building data/classification/), carried
    through verbatim; label_names must equal the bundle's class-name list in
    class-index order, and n(x) is joined via _classification_n_points."""
    is_classify = task == "classify"
    bundles = []
    for k, path in zip(k_values, diagram_paths):
        if not Path(path).exists():
            print(f"  [{tag}] missing {path} -- skipping split.")
            return None
        bundles.append(load_diagrams(path))  # (diagrams, bundle) per k

    seed_arrays = [np.asarray(bundle["seeds"]) for _, bundle in bundles]
    idx_per_k, common_seeds = intersect_seeds(seed_arrays)
    per_k_totals = {k: len(s) for k, s in zip(k_values, seed_arrays)}
    print(f"  [{tag}] {len(common_seeds)} clouds common to all k in {k_values} (per-k totals: {per_k_totals}).")

    # Select target columns by name -- label_names can carry deterministic
    # bookkeeping fields (e.g. edge_buffer) alongside the real targets. None
    # (no target_label_names configured) means "every label this process
    # produced" -- adapts to whatever process generated this data instead of
    # assuming a fixed target set.
    tda_label_names = list(bundles[0][1]["label_names"])
    if label_names is None:
        label_names = tuple(tda_label_names)

    if is_classify:
        # `labels` is a 1-D class-index vector; label_names is the ordered
        # class-name list (index i names class i). No column selection.
        if list(label_names) != tda_label_names:
            raise ValueError(
                f"[{tag}] classification target_label_names {list(label_names)} must equal the bundle's "
                f"class list {tda_label_names} exactly (position i == class label i)."
            )
        targets = np.asarray(bundles[0][1]["labels"]).reshape(-1)[idx_per_k[0]].astype(np.int64)
        for k, (_, bundle), idx in zip(k_values[1:], bundles[1:], idx_per_k[1:]):
            targets_k = np.asarray(bundle["labels"]).reshape(-1)[idx].astype(np.int64)
            assert np.array_equal(targets, targets_k), (
                f"[{tag}] class-label mismatch between k={k_values[0]} and k={k} after seed alignment -- alignment bug."
            )
    else:
        missing = [name for name in label_names if name not in tda_label_names]
        if missing:
            raise KeyError(f"[{tag}] label_names {tda_label_names} is missing {missing} from {label_names}")
        col_idx = [tda_label_names.index(name) for name in label_names]

        targets = np.asarray(bundles[0][1]["labels"], dtype=float)[np.ix_(idx_per_k[0], col_idx)]
        for k, (_, bundle), idx in zip(k_values[1:], bundles[1:], idx_per_k[1:]):
            targets_k = np.asarray(bundle["labels"], dtype=float)[np.ix_(idx, col_idx)]
            assert np.allclose(targets, targets_k), (
                f"[{tag}] target mismatch between k={k_values[0]} and k={k} after seed alignment -- alignment bug."
            )

    entropy_feature = FEATURE_REGISTRY.build("persistence_entropy", homology_dims=homology_dims)
    diagrams_per_k: dict[int, list[PersistenceDiagram]] = {}
    entropy_cols: dict[str, np.ndarray] = {}
    for k, (diagrams, _bundle), idx in zip(k_values, bundles, idx_per_k):
        aligned = [diagrams[i] for i in idx]
        diagrams_per_k[k] = aligned
        per_diagram_entropy = [entropy_feature.compute(d) for d in aligned]
        for dim in homology_dims:
            entropy_cols[f"entropy{dim}_k{k}"] = np.array([e[dim] for e in per_diagram_entropy], dtype=np.float64)

    # n(x) doesn't depend on k (same underlying cloud) -- joined once from
    # the sibling clouds.pkl by seed.
    if is_classify:
        n_points = _classification_n_points(clouds_path, bundles[0][1], common_seeds)
    else:
        clouds = _load_pickle(clouds_path)
        n_points_by_seed = {int(c["seed"]): c["n_points"] for c in clouds}
        n_points = np.array([n_points_by_seed[int(s)] for s in common_seeds], dtype=np.float64)

    lfunc_cols = _load_lfunc_cols(clouds_path, common_seeds, lfunc_n_radii) if lfunc_n_radii else {}
    fg_cols = (
        _load_fg_cols(clouds_path, common_seeds, fg_n_radii, fg_r_max) if fg_n_radii else {}
    )
    curves = (
        _load_curve_stack(clouds_path, common_seeds, tuple(curve_channels), fg_r_max)
        if curve_channels else None
    )

    return {
        "diagrams_per_k": diagrams_per_k,
        "entropy_cols": entropy_cols,
        "lfunc_cols": lfunc_cols,
        "fg_cols": fg_cols,
        "curves": curves,
        "n_points": n_points,
        "targets": targets,
        "seeds": common_seeds,
        "label_names": list(label_names),
    }


def build_pi_tensor(
    split: dict[str, Any],
    k_values: list[int],
    homology_dims: tuple[int, ...],
    resolution: int,
    sigma_pixels: float,
    coverage: float,
    pad: float = 1.05,
    train_idx: np.ndarray | None = None,
    imagers: list[MultiChannelImager] | None = None,
) -> tuple[np.ndarray, list[MultiChannelImager]]:
    fit = imagers is None
    if fit:
        if train_idx is None:
            raise ValueError("build_pi_tensor: train_idx is required when fitting (imagers=None).")
        imagers = []

    raw_channels: list[np.ndarray] = []
    for ki, k in enumerate(k_values):
        diagrams_k = split["diagrams_per_k"][k]
        if fit:
            calibration_diagrams = [diagrams_k[i] for i in train_idx]
            imager = build_calibrated_imager(
                calibration_diagrams, homology_dims=homology_dims, resolution=resolution,
                sigma_pixels=sigma_pixels, coverage=coverage, pad=pad, verbose=False,
            )
            imagers.append(imager)
        else:
            imager = imagers[ki]
        images = [imager.transform(d) for d in diagrams_k]
        for dim in homology_dims:
            raw_channels.append(np.stack([im[dim] for im in images]))

    # raw_channels is flat: [dim0_k0, dim1_k0, ..., dim0_k1, dim1_k1, ...]
    # -- group by k (D consecutive entries each) into (N, D, H, W), then
    # stack those groups along a new k axis.
    dims_per_k = len(raw_channels) // len(k_values)
    per_k = [np.stack(raw_channels[dims_per_k * i : dims_per_k * (i + 1)], axis=1) for i in range(len(k_values))]
    return np.stack(per_k, axis=1).astype(np.float32), imagers


def build_extra(
    split: dict[str, Any],
    train_idx: np.ndarray | None,
    n_norm: dict | None = None,
    entropy_norms: dict[str, dict[str, float]] | None = None,
    include_entropy: bool = False,
    include_log_n: bool = True,
    include_lfunc: bool = False,
    lfunc_pca: int = 0,
    include_fg: bool = False,
    include_curves: bool = False,
) -> tuple[np.ndarray, dict, dict[str, dict[str, float]]]:
    """(N, include_log_n + (n_entropy_cols if include_entropy else 0))
    side-vector, concatenated onto the per-k embeddings before the fusion
    head: [log N] by default, plus one plain z-scored column per
    split["entropy_cols"] entry (persistence entropy per (homology dim, k),
    loaded by load_multik_split) when include_entropy is set.
    include_log_n=False drops the log N(x) column entirely -- the
    logN-ablation arm needed to attribute performance to topology rather
    than point count (see the writeup's Estimation-performance section); if
    that leaves no columns at all (include_entropy also False), returns an
    (N, 0) array, a no-op under torch.cat.

    Fit-once/apply-frozen (see module docstring): pass train_idx (n_norm and
    entropy_norms left None) to fit n(x)/entropy stats on
    split[...][train_idx] only, applied to every row of split -- the main
    population's train-only fit. Pass a previously-fit n_norm (and
    entropy_norms, if include_entropy) to apply them frozen to a different
    population instead (train_idx unused, pass None) -- the adversarial
    call."""
    # Fit iff we were handed train rows to fit on. Do NOT infer this from
    # `n_norm is None`: with include_log_n=False no n_norm is ever produced, so
    # the frozen adversarial call would silently RE-FIT on the adversarial
    # population (and, in the PCA branch, index matrix[None] and produce a
    # garbage 3-D projection). Latent since include_log_n was added; harmless
    # only while no other column group was ever used without log N -- i.e. it
    # bites exactly the classification arms, which set include_log_n: false.
    fit = train_idx is not None
    if not fit and entropy_norms is None:
        entropy_norms = {}
    if include_log_n:
        if fit:
            n_norm = vihrs.fit_log_zscore(split["n_points"][train_idx])
        elif n_norm is None:
            raise ValueError("build_extra: applying frozen (train_idx=None) needs n_norm when include_log_n.")
        n_std = vihrs.apply_log_zscore(split["n_points"], n_norm).astype(np.float32)
        cols = [n_std]
    else:
        cols = []
    if fit:
        entropy_norms = {}
    # entropy_norms is the per-scalar-column norm dict, shared by every named
    # scalar column group below (entropy, then L(r)-r). Names never collide
    # (entropy<dim>_k<k> vs lfunc_r<nn>), and each group is appended in sorted
    # order, so the column layout is identical between the train fit and the
    # frozen adversarial apply.
    if include_entropy:
        for name in sorted(split.get("entropy_cols") or {}):
            raw = split["entropy_cols"][name]
            if fit:
                entropy_norms[name] = fit_zscore(raw[train_idx])
            cols.append(apply_zscore(raw, entropy_norms[name]).astype(np.float32))

    if include_lfunc:
        names = sorted(split.get("lfunc_cols") or {})
        if lfunc_pca > 0:
            # Decorrelate before the head -- see _fit_lfunc_pca's docstring.
            matrix = np.stack([split["lfunc_cols"][name] for name in names], axis=1)
            if fit:
                entropy_norms[_LFUNC_PCA_KEY] = _fit_lfunc_pca(matrix[train_idx], lfunc_pca)
            cols.extend(_apply_lfunc_pca(matrix, entropy_norms[_LFUNC_PCA_KEY]))
        else:
            for name in names:
                raw = split["lfunc_cols"][name]
                if fit:
                    entropy_norms[name] = fit_zscore(raw[train_idx])
                cols.append(apply_zscore(raw, entropy_norms[name]).astype(np.float32))

    if include_fg:
        # No PCA option here: the collinearity that motivated lfunc_pca is a
        # property of sampling ONE smooth curve at many radii, and the F and G
        # columns are two different curves whose leading directions are not
        # interchangeable.
        #
        # ONE pooled z-score PER FUNCTION (over train rows AND all of that
        # function's radii), not one per column. Per-column scaling was a bug:
        # F and G are CDFs, so at the larger radii they sit at ~1 for nearly
        # every cloud, and fit_zscore only guards std == 0 EXACTLY -- a column
        # with a tiny nonzero spread got divided by that tiny std, turning the
        # few clouds that differ into extreme inputs. That made PH (+) F/G
        # train worse than PH alone on every process (fgp__PHfg, 0/10 seeds).
        # Pooling is also what vihrs does for its curve channels
        # (fit_zscore_per_channel), so both paths now scale F and G alike.
        names = sorted(split.get("fg_cols") or {})
        for prefix in ("ffunc", "gfunc"):
            group = [name for name in names if name.startswith(prefix + "_")]
            if not group:
                continue
            key = f"__fgpool_{prefix}__"
            if fit:
                block = np.stack([split["fg_cols"][name][train_idx] for name in group], axis=1)
                entropy_norms[key] = fit_zscore(block)        # pooled: rows x radii
            for name in group:
                raw = split["fg_cols"][name]
                cols.append(apply_zscore(raw, entropy_norms[key]).astype(np.float32))

    out = (
        np.stack(cols, axis=1) if cols
        else np.zeros((len(split["n_points"]), 0), dtype=np.float32)
    )

    if include_curves:
        # The two-branch model's curve input (see PIMultiK / CurveEncoder):
        # the full (N, C, m) stack of summary-function curves, z-scored per
        # channel with one pooled mean/std over train rows x radii (vihrs's
        # fit_zscore_per_channel -- same normalization the vihrs L+F+G
        # baseline used), then FLATTENED and appended as the LAST block of
        # the side vector. Riding on the existing `extra` tensor means the
        # loaders, the shared train loop and the adversarial path are all
        # untouched; PIMultiK slices this tail back off and reshapes it to
        # (B, C, m) before the 1-D CNN. It must stay last for that to work.
        curves = split.get("curves")
        if curves is None:
            raise KeyError("include_curves is set but the split carries no `curves` "
                           "(load_multik_split was called without curve_channels).")
        if fit:
            entropy_norms[_CURVE_NORM_KEY] = vihrs.fit_zscore_per_channel(curves[train_idx])
        block = vihrs.apply_zscore_per_channel(curves, entropy_norms[_CURVE_NORM_KEY])
        out = np.concatenate(
            [out, block.reshape(block.shape[0], -1).astype(np.float32)], axis=1
        )
    return out, n_norm, entropy_norms


class CurveEncoder(nn.Module):
    """1-D CNN branch over a (B, C, m) stack of summary-function curves.

    The convolutional trunk is the vihrs baseline's EXACTLY -- Conv1d(C,64,7)
    -ReLU-MaxPool(5)-Conv1d(64,64,7)-ReLU-MaxPool(5)-Conv1d(64,64,7)-ReLU-
    Flatten (baselines/vihrs.py::VihrsCNN), so the curve-only arm of the
    two-branch model is the L+F+G baseline in all but its head. It is followed
    by Linear(flat -> out_dim)-ReLU, which is VihrsCNN's own first dense layer
    moved into the branch. That projection is what keeps the fusion balanced:
    at m = 513 the flattened trunk is 64 x 13 = 832 wide against the PI
    branch's 64-wide embedding, and concatenating those raw would let the
    curve branch dominate the head by width alone.
    """

    def __init__(self, in_channels: int, seq_len: int, out_dim: int):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7), nn.ReLU(), nn.MaxPool1d(5),
            nn.Conv1d(64, 64, kernel_size=7), nn.ReLU(), nn.MaxPool1d(5),
            nn.Conv1d(64, 64, kernel_size=7), nn.ReLU(),
        )
        with torch.no_grad():
            flat = self.trunk(torch.zeros(1, in_channels, seq_len)).flatten(1).shape[1]
        self.proj = nn.Sequential(nn.Linear(flat, out_dim), nn.ReLU())
        self.out_dim = out_dim

    def forward(self, curves: torch.Tensor) -> torch.Tensor:
        return self.proj(self.trunk(curves).flatten(start_dim=1))


class PIMultiK(nn.Module):
    """Late-fusion multi-k model, in two composable stages:

    1. EncoderBank turns each k's (H0, H1) image pair into a (B, K,
       embedding_dim) embedding sequence -- encoder_mode picks shared-weight
       (one CoordConvPIEncoder, k folded into the batch dim) vs independent
       (n_k separate CoordConvPIEncoders) -- see EncoderBank's docstring.
    2. That sequence is collapsed to one vector -- fusion_mode="concat" is a
       flat reshape (head_in grows with n_k); fusion_mode="conv" runs it
       through ConvFusion, whose fusion_pool ("avg" vs "flatten") picks
       scale-count-invariant pooling vs position-preserving flattening --
       see ConvFusion's docstring.

    Together these two axes cover what used to be three separate model
    classes (this file's old shared+concat, pi_multik_towers' independent+
    flatten) as one config-selectable combination -- e.g. independent+
    avg-pool or shared+flatten are now reachable without a new file. The
    fused vector is concatenated with the extra scalar features and passed
    through an MLP head."""

    def __init__(
        self,
        in_channels: int,
        embedding_dim: int,
        n_k: int,
        n_extra: int,
        n_targets: int,
        conv_channels: tuple[int, ...] = (32, 64, 128),
        dropout: float = 0.2,
        head_hidden_dims: tuple[int, ...] = (64, 32),
        head_dropout: float = 0.1,
        encoder_mode: str = "shared",
        fusion_mode: str = "concat",
        fusion_pool: str = "avg",
        fusion_dropout: float = 0.0,
        scale_fusion_hidden: int = 128,
        scale_fusion_out_dim: int = 128,
        scale_fusion_kernel_size: int = 3,
        scale_fusion_dropout: float = 0.0,
        pool_type: str = "max",
        use_coords: bool = True,
        task: str = "params",
        use_pi: bool = True,
        curve_channels: int = 0,
        curve_len: int = 0,
    ):
        super().__init__()
        self.n_k = n_k
        self.fusion_mode = fusion_mode

        # TWO-BRANCH MODEL. curve_channels > 0 adds a 1-D CNN branch
        # (CurveEncoder) over the full summary-function curves, which arrive
        # flattened as the LAST curve_channels * curve_len columns of `extra`
        # (build_extra's include_curves block). use_pi=False removes the
        # persistence-image branch entirely -- no encoder parameters at all --
        # so {use_pi True, False} with the same curve branch is a clean PH
        # ablation of one model. Defaults (use_pi=True, no curves) build the
        # pre-existing PI-only model unchanged.
        if not use_pi and curve_channels <= 0:
            raise ValueError("PIMultiK: use_pi=False needs a curve branch (curve_channels > 0).")
        self.use_pi = use_pi
        self.curve_channels = int(curve_channels)
        self.curve_len = int(curve_len)
        self.n_curve_cols = self.curve_channels * self.curve_len
        self.curve_encoder = (
            CurveEncoder(self.curve_channels, self.curve_len, embedding_dim)
            if self.curve_channels > 0 else None
        )
        curve_dim = self.curve_encoder.out_dim if self.curve_encoder is not None else 0
        # n_extra counts every side-vector column, curves included; the head
        # sees the curve EMBEDDING instead of the raw flattened curves.
        n_scalar = n_extra - self.n_curve_cols
        if n_scalar < 0:
            raise ValueError(
                f"PIMultiK: n_extra={n_extra} is smaller than the curve block "
                f"({self.curve_channels} x {self.curve_len})."
            )

        if not use_pi:
            self.bank = None
            self.fusion = None
            self.fusion_dropout = nn.Identity()
            fused_dim = 0
        else:
            self.bank, self.fusion, fused_dim = self._build_pi_branch(
                encoder_mode, n_k, in_channels, embedding_dim, conv_channels, dropout,
                pool_type, use_coords, fusion_mode, fusion_pool, scale_fusion_hidden,
                scale_fusion_out_dim, scale_fusion_kernel_size, scale_fusion_dropout,
            )
            # Applied after fusion, before the head, regardless of fusion_mode
            # -- previously only pi_multik_towers had this (as an always-on
            # nn.Dropout); promoted here so it's available to any combination.
            self.fusion_dropout = nn.Dropout(fusion_dropout) if fusion_dropout > 0 else nn.Identity()

        head_in = fused_dim + curve_dim + n_scalar
        # n_targets is the regression target count for task="params", or the
        # class count for task="classify" -- the fused embedding -> output MLP
        # is structurally the same either way (ParameterEstimator and
        # ClassificationHead are both plain MLPs), only the loss and the
        # output semantics differ (see PIMultiKExperiment.run).
        if task == "classify":
            self.head = ClassificationHead(
                embedding_dim=head_in, n_classes=n_targets,
                hidden_dims=head_hidden_dims, dropout=head_dropout,
            )
        else:
            self.head = ParameterEstimator(
                embedding_dim=head_in, n_params=n_targets,
                hidden_dims=head_hidden_dims, dropout=head_dropout,
            )

    @staticmethod
    def _build_pi_branch(
        encoder_mode, n_k, in_channels, embedding_dim, conv_channels, dropout,
        pool_type, use_coords, fusion_mode, fusion_pool, scale_fusion_hidden,
        scale_fusion_out_dim, scale_fusion_kernel_size, scale_fusion_dropout,
    ):
        bank = EncoderBank(
            mode=encoder_mode, n_k=n_k, in_channels=in_channels, embedding_dim=embedding_dim,
            conv_channels=conv_channels, dropout=dropout, pool_type=pool_type, use_coords=use_coords,
        )
        if fusion_mode == "concat":
            fusion = None
            fused_dim = n_k * embedding_dim
        elif fusion_mode == "conv":
            fusion = ConvFusion(
                n_k=n_k, embedding_dim=embedding_dim, hidden=scale_fusion_hidden,
                out_dim=scale_fusion_out_dim, kernel_size=scale_fusion_kernel_size,
                dropout=scale_fusion_dropout, pool=fusion_pool,
            )
            fused_dim = fusion.out_dim
        else:
            raise ValueError(f"PIMultiK: fusion_mode must be 'concat' or 'conv', got {fusion_mode!r}.")
        return bank, fusion, fused_dim

    def forward(self, pi_imgs, extra):
        parts = []
        if self.bank is not None:
            seq = self.bank(pi_imgs)  # (B, K, C)
            if self.fusion_mode == "concat":
                fused = seq.reshape(seq.shape[0], -1)
            else:
                fused = self.fusion(seq)
            parts.append(self.fusion_dropout(fused))
        if self.curve_encoder is not None:
            # The curve block is the LAST n_curve_cols columns of extra (see
            # build_extra's include_curves); slice it off, restore (B, C, m).
            curves = extra[:, -self.n_curve_cols:].reshape(-1, self.curve_channels, self.curve_len)
            extra = extra[:, :-self.n_curve_cols]
            parts.append(self.curve_encoder(curves))
        parts.append(extra)
        return self.head(torch.cat(parts, dim=1))


def _resolve_fusion_kwargs(cfg: dict) -> dict[str, Any]:
    """Translates method.params' encoder_mode/fusion_mode/fusion_pool
    overrides -- and the legacy use_fusion bool (matern_pi_multik.yaml's
    method: pi_multik still sets it: True reproduces what pi_multik_scaleconv
    defaults to, conv fusion with avg pooling; False reproduces plain
    concat) -- into PIMultiK constructor kwargs.

    Deliberately omits a key entirely when cfg doesn't set it, so each
    sibling experiment's own _build_model default (via kwargs.setdefault)
    still applies. Precedence: explicit encoder_mode/fusion_mode/fusion_pool
    cfg keys, then use_fusion, then the registered method's own default."""
    kwargs: dict[str, Any] = {}
    if "use_fusion" in cfg and "fusion_mode" not in cfg:
        kwargs["fusion_mode"] = "conv" if cfg["use_fusion"] else "concat"
        kwargs["fusion_pool"] = "avg"
    for key in ("encoder_mode", "fusion_mode", "fusion_pool"):
        if key in cfg:
            kwargs[key] = cfg[key]
    return kwargs


@torch.no_grad()
def _per_class_accuracy(
    model: nn.Module, loader: DataLoader, device: str, n_classes: int, class_names: tuple[str, ...]
) -> dict[str, float]:
    """Recall per class over `loader` (batches are the (pi_imgs, extra,
    labels) 3-tuples this experiment builds). Classification-only companion
    to training.train.evaluate_per_target, which is MSE-shaped and does not
    apply here."""
    model.eval()
    correct = np.zeros(n_classes, dtype=np.int64)
    total = np.zeros(n_classes, dtype=np.int64)
    for pi_imgs, extra, labels in loader:
        preds = model(pi_imgs.to(device), extra.to(device)).argmax(dim=-1).cpu().numpy()
        labels = labels.cpu().numpy()
        for c in range(n_classes):
            mask = labels == c
            total[c] += int(mask.sum())
            correct[c] += int((preds[mask] == c).sum())
    return {
        class_names[c]: (float(correct[c] / total[c]) if total[c] else float("nan"))
        for c in range(n_classes)
    }


@register("pi_multik")
class PIMultiKExperiment(MultiSourceExperiment):
    file_keys = ("clouds", "images")

    @property
    def subdir(self) -> str:
        return "pi_multik"

    def _build_model(self, **kwargs) -> PIMultiK:
        """Shared-weight-encoder, flat-concat baseline -- the reference
        pi_multik design. Siblings (PIMultiKScaleConvExperiment,
        PIMultiKTowersExperiment) override just the encoder_mode/fusion_mode/
        fusion_pool defaults below -- everything else in run() (data
        loading, training loop, save_results) is shared verbatim."""
        kwargs.setdefault("encoder_mode", "shared")
        kwargs.setdefault("fusion_mode", "concat")
        return PIMultiK(**kwargs)

    def run(
        self,
        dataset_paths: dict[str, Any],
        output_dir: Path,
        adversarial_paths: dict[str, Any] | None = None,
    ) -> dict:
        # fetch configs
        task = str(self.cfg.get("task", "params"))
        is_classify = task == "classify"
        label_names = tuple(self.cfg.get("target_label_names"))
        k_values = list(self.cfg["k_values"])
        homology_dims = tuple(self.cfg.get("homology_dims", (0, 1)))
        include_entropy = bool(self.cfg.get("include_entropy", False))
        # n(x) side-channel: on by default for parameter estimation (Vihrs
        # 2022), off by default for classification -- a topology-only
        # baseline. Override with method.params.include_log_n either way.
        include_log_n = bool(self.cfg.get("include_log_n", not is_classify))
        # include_lfunc: N > 0 appends N log-spaced L(r)-r samples to the scalar
        # side-vector. Feeds the second-order trend back in alongside an
        # L-reparameterized filtration (l_dtm / l_rips), which normalizes that
        # trend out of the diagram -- see data_generation/filtration/lfunc.py.
        lfunc_n_radii = int(self.cfg.get("include_lfunc", 0) or 0)
        include_lfunc = lfunc_n_radii > 0
        # lfunc_pca: k > 0 replaces the k raw (heavily collinear) L columns
        # with k whitened principal components fit on the train rows.
        lfunc_pca = int(self.cfg.get("lfunc_pca", 0) or 0)
        # include_fgfunc: N > 0 appends N log-spaced samples EACH of the
        # empty-space F(r) and nearest-neighbour G(r) functions (2N columns),
        # read from the vihrs F/G/J cache at fg_r_max. Together with
        # include_lfunc this makes the scalar side-vector the full "union of
        # the standard summary functions" -- the baseline that beat PH (+) L
        # on classification -- so this arm tests whether persistence images
        # add anything on top of that union. See baselines/summstats.py.
        fg_n_radii = int(self.cfg.get("include_fgfunc", 0) or 0)
        include_fg = fg_n_radii > 0
        fg_r_max = float(self.cfg.get("fg_r_max", 0.25))
        # TWO-BRANCH MODEL. curve_channels (e.g. [L, F, G]) adds a 1-D CNN
        # branch over the FULL summary-function curves (CurveEncoder, the
        # vihrs conv trunk); use_pi: false drops the persistence-image branch.
        # Same model with and without the PH branch is the PH ablation --
        # see slurm/twobranch_*.sh. Both default to the pre-existing model.
        curve_channels = (
            summstats.normalize_channels(self.cfg.get("curve_channels"))
            if self.cfg.get("curve_channels") else ()
        )
        include_curves = bool(curve_channels)
        use_pi = bool(self.cfg.get("use_pi", True))
        resolution = int(self.cfg.get("resolution", 64))
        sigma_pixels = float(self.cfg.get("sigma_pixels", 0.5))
        coverage = float(self.cfg.get("pd_calibration_coverage", 0.95))
        pad = float(self.cfg.get("pad", 1.05))
        seed = self.cfg["seed"]
        # init_offset: perturb weight init / dropout / batch order for a
        # RESTART without touching the data split. train_val_test_indices
        # below draws from its own local torch.Generator(seed), so it is
        # unaffected by the global RNG prepare_device sets here -- every
        # restart of a seed therefore shares that seed's exact
        # train/val/test partition, and selecting among restarts on
        # val_loss stays honest. Default 0 reproduces existing runs.
        init_offset = int(self.cfg.get("init_offset", 0) or 0)
        device = prepare_device(seed + init_offset)

        # load and align diagrams and clouds by seed
        train_split = load_multik_split(
            k_values, list(dataset_paths["images"]), Path(dataset_paths["clouds"]), label_names, tag="train_test",
            homology_dims=homology_dims, task=task, lfunc_n_radii=lfunc_n_radii,
            fg_n_radii=fg_n_radii, fg_r_max=fg_r_max, curve_channels=curve_channels,
        )
        if train_split is None:
            raise FileNotFoundError(f"diagrams missing for some k in {k_values} under {dataset_paths['images']}.")

        if self.cfg.get("shuffle_labels", False):
            # Leakage/sanity check: permute targets against every other
            # per-cloud field (diagrams, n_points, ...) BEFORE the
            # train/val/test split, so each split stays internally
            # consistent (every row still has some target row from the
            # same population) but the true (input, target) correspondence
            # is destroyed everywhere. Seeded off this run's own seed so
            # different seeds get different (but each reproducible)
            # permutations, matching every other seed-dependent choice in
            # this method. Expected result: no exploitable signal survives,
            # so test_loss collapses to ~1.0 -- the "predicts nothing beyond
            # the design-distribution mean" reference point (see the
            # writeup's loss-definition section).
            perm = np.random.default_rng(seed).permutation(len(train_split["targets"]))
            train_split["targets"] = train_split["targets"][perm]

        adv_split = None
        if adversarial_paths is not None:
            adv_split = load_multik_split(
                k_values, list(adversarial_paths["images"]), Path(adversarial_paths["clouds"]),
                label_names, tag="adversarial", homology_dims=homology_dims, task=task,
                lfunc_n_radii=lfunc_n_radii, fg_n_radii=fg_n_radii, fg_r_max=fg_r_max,
                curve_channels=curve_channels,
            )

        n = len(train_split["targets"])
        train_idx, val_idx, test_idx = train_val_test_indices(n, seed)

        if is_classify:
            # Class indices pass through verbatim (CrossEntropyLoss wants raw
            # int64 targets, no standardization). label_norm mirrors
            # experiments/base.py's classification stub so save_results and
            # any downstream reader see a consistent shape.
            classes = sorted(int(c) for c in np.unique(train_split["targets"]))
            if classes != list(range(len(label_names))):
                raise ValueError(
                    f"[{self.tag} seed={seed}] expected contiguous class labels 0..{len(label_names) - 1}, "
                    f"got {classes}."
                )
            label_norm = {
                "kind": "classification", "classes": classes, "mean": None, "std": None,
                "transforms": ["class_index"] * len(label_names),
            }
            targets_std = train_split["targets"].astype(np.int64)
        else:
            label_norm = vihrs.fit_log_zscore(train_split["targets"][train_idx])
            targets_std = vihrs.apply_log_zscore(train_split["targets"], label_norm).astype(np.float32)
        pi_img, imagers = build_pi_tensor(
            train_split, k_values, homology_dims=homology_dims, resolution=resolution,
            sigma_pixels=sigma_pixels, coverage=coverage, pad=pad, train_idx=train_idx,
        )
        extra, n_norm, entropy_norms = build_extra(
            train_split, train_idx, include_entropy=include_entropy, include_log_n=include_log_n,
            include_lfunc=include_lfunc, lfunc_pca=lfunc_pca, include_fg=include_fg,
            include_curves=include_curves,
        )

        full_dataset = TensorDataset(
            torch.from_numpy(pi_img), torch.from_numpy(extra), torch.from_numpy(targets_std),
        )

        def _loader(idx: np.ndarray, shuffle: bool) -> DataLoader:
            return DataLoader(Subset(full_dataset, idx), batch_size=self.cfg["batch_size"], shuffle=shuffle)

        train_loader, val_loader, test_loader = _loader(train_idx, True), _loader(val_idx, False), _loader(test_idx, False)

        # PIMultiK.forward(pi_imgs, extra) lines up exactly with
        # cloudforger.training.train's (inputs, covariates, labels) 3-tuple batch
        # convention, so the shared train loop applies as-is.
        # Only passed when the two-branch model is actually requested: sibling
        # experiments override _build_model (DimSplitPIMultiK does not accept
        # these kwargs), and at the defaults every existing method must build
        # exactly what it built before.
        two_branch_kwargs: dict[str, Any] = {}
        if include_curves or not use_pi:
            two_branch_kwargs = {
                "use_pi": use_pi,
                "curve_channels": len(curve_channels),
                "curve_len": int(train_split["curves"].shape[2]) if include_curves else 0,
            }
        model = self._build_model(
            in_channels=len(homology_dims),
            embedding_dim=self.cfg["embedding_dim"],
            n_k=len(k_values),
            n_extra=extra.shape[1],
            n_targets=len(label_names),
            conv_channels=tuple(self.cfg.get("conv_channels", (32, 64, 128))),
            dropout=self.cfg.get("dropout", 0.2),
            head_hidden_dims=tuple(self.cfg.get("head_hidden_dims", (64, 32))),
            head_dropout=self.cfg.get("head_dropout", 0.1),
            scale_fusion_hidden=self.cfg.get("scale_fusion_hidden", 128),
            scale_fusion_out_dim=self.cfg.get("scale_fusion_out_dim", 128),
            scale_fusion_kernel_size=self.cfg.get("scale_fusion_kernel_size", 3),
            scale_fusion_dropout=self.cfg.get("scale_fusion_dropout", 0.0),
            fusion_dropout=self.cfg.get("fusion_dropout", 0.0),
            pool_type=str(self.cfg.get("pool_type", "max")),
            use_coords=bool(self.cfg.get("coordconv", True)),
            task=task,
            **two_branch_kwargs,
            **_resolve_fusion_kwargs(self.cfg),
        ).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"[{self.tag} seed={seed}] branches: "
              f"PI={'on' if use_pi else 'OFF'}  curves={'+'.join(curve_channels) or 'none'}  "
              f"({n_params:,} params)")
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.cfg.get("lr", 1e-3), weight_decay=self.cfg.get("weight_decay", 1e-4))
        # optimizer = torch.optim.Adam(model.parameters(), lr=self.cfg.get("lr", 1e-3), weight_decay=self.cfg.get("weight_decay", 1e-4))
        loss_fn = nn.CrossEntropyLoss() if is_classify else nn.MSELoss()

        history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
        if is_classify:
            history["train_acc"], history["val_acc"] = [], []
        best_val_loss, best_state = float("inf"), None
        n_epochs = self.cfg["n_epochs"]
        patience = self.cfg.get("early_stopping_patience")
        epochs_no_improve = 0

        for epoch in range(1, n_epochs + 1):
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
            val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            if is_classify:
                history["train_acc"].append(train_acc)
                history["val_acc"].append(val_acc)
            # Checkpoint/early-stop on val loss for both tasks (lower is
            # better: MSE for params, cross-entropy for classify) -- one
            # code path, matching every other experiment in the repo.
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            if epoch == 1 or epoch % 25 == 0 or epoch == n_epochs:
                msg = f"[{self.tag} seed={seed}] epoch {epoch:3d} | train {train_loss:.4f} | val {val_loss:.4f}"
                if is_classify:
                    msg += f" | train_acc {train_acc:.4f} | val_acc {val_acc:.4f}"
                print(msg)
            if patience is not None and epochs_no_improve >= patience:
                print(f"[{self.tag} seed={seed}] early stopping at epoch {epoch} (no val improvement for {patience} epochs)")
                break

        model.load_state_dict(best_state)
        test_loss, test_acc = evaluate(model, test_loader, loss_fn, device)
        if is_classify:
            test_loss_per_target = None
            test_acc_per_class = _per_class_accuracy(model, test_loader, device, len(label_names), label_names)
            print(
                f"\n[{self.tag} seed={seed}] test cross-entropy {test_loss:.4f} | test accuracy {test_acc:.4f}\n"
                f"  per-class accuracy: "
                + ", ".join(f"{name}={acc:.3f}" for name, acc in test_acc_per_class.items())
            )
        else:
            test_loss_per_target = dict(zip(label_names, evaluate_per_target(model, test_loader, device).tolist()))
            test_acc_per_class = None
            print(f"\n[{self.tag} seed={seed}] test loss {test_loss:.4f}")

        adversarial_loss = None
        adversarial_loss_per_target = None
        adversarial_acc = None
        adversarial_acc_per_class = None
        if adv_split is not None:
            if is_classify:
                adv_targets_std = adv_split["targets"].astype(np.int64)
            else:
                adv_targets_std = vihrs.apply_log_zscore(adv_split["targets"], label_norm).astype(np.float32)
            adv_pi_img, _ = build_pi_tensor(
                adv_split, k_values, homology_dims=homology_dims, resolution=resolution,
                sigma_pixels=sigma_pixels, coverage=coverage, pad=pad, imagers=imagers,
            )
            adv_extra, _, _ = build_extra(
                adv_split, None, n_norm=n_norm, entropy_norms=entropy_norms, include_entropy=include_entropy,
                include_log_n=include_log_n, include_lfunc=include_lfunc, lfunc_pca=lfunc_pca,
                include_fg=include_fg, include_curves=include_curves,
            )
            adv_ds = TensorDataset(
                torch.from_numpy(adv_pi_img), torch.from_numpy(adv_extra), torch.from_numpy(adv_targets_std),
            )
            adv_loader = DataLoader(adv_ds, batch_size=self.cfg["batch_size"], shuffle=False)
            adversarial_loss, adversarial_acc = evaluate(model, adv_loader, loss_fn, device)
            if is_classify:
                adversarial_acc_per_class = _per_class_accuracy(
                    model, adv_loader, device, len(label_names), label_names
                )
                print(
                    f"[{self.tag} seed={seed}] adversarial cross-entropy {adversarial_loss:.4f} | "
                    f"adversarial accuracy {adversarial_acc:.4f}"
                )
            else:
                adversarial_loss_per_target = dict(
                    zip(label_names, evaluate_per_target(model, adv_loader, device).tolist())
                )
                print(f"[{self.tag} seed={seed}] adversarial loss {adversarial_loss:.4f}")

        cfg_meta = {
            **self.cfg,
            "channels": [f"k{k}_h{d}" for k in k_values for d in homology_dims],
            # Per-k calibrated imager params, so a seed's exact calibration
            # (birth_range/pers_range/sigma_pixels -- now seed-specific, see
            # module docstring) is recoverable from results.json alone.
            "imager_params": {k: imager.params for k, imager in zip(k_values, imagers)},
        }
        # best_val_loss is the selection statistic for restart arms
        # (slurm/strauss_restarts.sh -> scripts/collect_restarts.py) and the
        # convergence flag for the epoch-cap arms; surfaced into results.json
        # so neither ever has to load results.pt.
        extra_meta: dict[str, Any] = {
            "best_val_loss": float(best_val_loss),
            "n_epochs_run": len(history["val_loss"]),
        }
        if is_classify:
            extra_meta.update({
                "task": "classify",
                "class_names": list(label_names),
                "test_accuracy": test_acc,
                "test_accuracy_per_class": test_acc_per_class,
                "adversarial_accuracy": adversarial_acc,
                "adversarial_accuracy_per_class": adversarial_acc_per_class,
            })
        save_results(
            output_dir, model=model, best_state=best_state, history=history, cfg=cfg_meta,
            test_loss=test_loss, label_names=list(label_names), label_norm=label_norm,
            test_loss_per_target=test_loss_per_target, adversarial_loss=adversarial_loss,
            adversarial_loss_per_target=adversarial_loss_per_target,
            adversarial_path=adversarial_paths.get("clouds") if adversarial_paths else None,
            extra_meta=extra_meta,
        )

        result: dict[str, Any] = {"seed": seed, "test_loss": test_loss, "test_loss_per_target": test_loss_per_target}
        if is_classify:
            result["test_accuracy"] = test_acc
            result["test_accuracy_per_class"] = test_acc_per_class
        if adversarial_loss is not None:
            result["adversarial_loss"] = adversarial_loss
            result["adversarial_loss_per_target"] = adversarial_loss_per_target
            if is_classify:
                result["adversarial_accuracy"] = adversarial_acc
                result["adversarial_accuracy_per_class"] = adversarial_acc_per_class
        return result
