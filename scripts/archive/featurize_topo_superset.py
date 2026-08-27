#!/usr/bin/env python3
# scripts/featurize_topo_superset.py
"""Clouds -> DTM-by-mass-fraction diagrams -> calibrated persistence images,
in one pass, for the topo_superset m-sweep.

Supersedes the old two-stage precompute_topo_superset.py (clouds ->
diagrams) + featurize_topo_superset.py (diagrams -> images) split. Merged
because the point clouds get regenerated from time to time (new process
params, more clouds, a generator fix, ...), and the two-stage split let a
stale topo_superset/*_diagrams.pkl -- computed from a since-replaced
clouds.pkl -- silently sit there and get fed straight into imaging as if it
were current. This script always recomputes and overwrites both the raw
diagrams and the images from the CURRENT clouds.pkl (no skip-if-exists
guard) specifically so that failure mode can't happen again: every run is
against today's point clouds, not whatever happened to be cached.

Per cloud, k = round(m * N) is recomputed from that cloud's own N, so the
same m means "the same relative neighborhood size" for every cloud
regardless of point count -- this is why the sweep is parameterized by mass
fraction m rather than a fixed neighbor count k in the first place (a fixed
k drifts across a very different fraction of each cloud once N varies
within the dataset). Diagrams are still saved (topo_superset/
dtm_m<value>_diagrams.pkl), not just images, so a later resigma/resolution
retune doesn't require rerunning the TDA computation -- only the imaging
half of this script.

The persistence imager is fit on TRAIN diagrams only, then applied frozen
to the adversarial diagrams for that same m (no calibration leakage into
the adversarial split).

Persistence-image resolution/sigma_pixels/homology_dims come from the
config's features.persistence_image block, same as scripts/featurize.py's
own persistence_image handler -- so these channels are directly comparable
to the existing dtm_k5/dtm_k10/dtm_k15 ones.

The m-channel subset is a live area of work -- override with --m-values
rather than editing DEFAULT_M_VALUES.

Filtration value cutoff (thresh): DTMRipsComplex's `k` only sets the DTM
*density-weighting* function -- it does NOT bound which simplices get
built. That's `max_filtration` (GUDHI's WeightedRipsComplex), and
DTMFiltration's own default is thresh=None -> max_filtration=inf, meaning
EVERY vertex/edge (and by clique expansion, every triangle up to maxdim+1)
gets built regardless of k/m. For clouds with hundreds to ~1300 points
(nested_thomas), that's an unbounded 2-skeleton -- computationally
catastrophic, confirmed directly: a real 50-shard run against nested_thomas
didn't finish a single m value's diagrams in 35+ minutes, one shard OOM'd
at 16G/CPU, and `sstat` showed >12GB RSS mid-run.

_thresh_for_m() bounds it instead, calibrated from data already on disk
(data/nested_thomas/dtm_k5/diagrams.pkl -- computed thresh=None, so its
finite death values ARE the filtration values that were actually needed,
unbounded, at k=5): for each of its 6996 diagrams, k=5 maps to an effective
m = 5/N (N varies 42-1169 there, so this covers roughly m in [0.004,
0.12]); binning by that effective m and taking the 99.9th percentile of
each bin's max finite death value gives an empirical growth curve, fit as a
power law threshold(m) = A * m^B (least squares on log-log bin medians:
A~2.6, B~0.25 -- notably a much SLOWER growth than the naive uniform-
density neighbor-radius estimate of B=0.5, consistent with a genuinely
clustered, non-uniform-density process). A THRESH_SAFETY=1.5x multiplier on
top of the already-conservative p99.9 covers the ~30x extrapolation past
m=0.12 out to the sweep's m=0.90.

Caveat, and it's real: this genuinely helps the fine/mid channels (m up to
~0.20ish), where the natural filtration values are small relative to a
generously-safe finite cutoff. It buys much less at m=0.45/0.90 -- those
channels are, by construction, trying to see near-global structure, so
their natural filtration values approach the domain's own scale regardless
of thresh; a cutoff loose enough not to truncate real coarse-scale features
is also loose enough not to prune much of the expensive tail. That's not a
flaw in the calibration, it's what "coarse scale" means -- expect m=0.45/
0.90 to still be the slowest channels, just no longer literally unbounded.

Override the whole formula with --thresh (applies one fixed value to every
m instead of the per-m curve) if you'd rather set it by hand.

Three ways to run it:

  1. Single process, does everything (fine for a quick/small run):
       python scripts/featurize_topo_superset.py configs/runs/matern/matern_pi_multik_scale.yaml

  2. Sharded, for parallel SLURM array jobs (diagram computation is the
     expensive, embarrassingly-per-cloud part; calibration/imaging needs the
     FULL set at once, so it can't be sharded -- see slurm/featurize_topo_superset_*.sh):
       # once, fast, before spending the array's compute budget:
       python scripts/featurize_topo_superset.py <config> --sanity-only
       # N array tasks, index 0..N-1, diagrams only, writes to topo_superset/_shards/:
       python scripts/featurize_topo_superset.py <config> --shard $INDEX $TOTAL
       # once, after all N shard tasks finish: merge, calibrate, image:
       python scripts/featurize_topo_superset.py <config> --merge $TOTAL
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cloudforger.config import load_config
from cloudforger.core.cloud import PointCloud
from cloudforger.core.diagram import PersistenceDiagram
from cloudforger.core.io import dump_pickle, load_pickle
from cloudforger.core.records import diagram_to_record, record_to_diagram, to_pointcloud
from cloudforger.data_generation.filtration import DTMFiltration
from cloudforger.paths import DEFAULT_DATA_ROOT, DataPaths
from cloudforger.vectorization.persistence_images.calibrated import build_calibrated_imager
from cloudforger.vectorization.scalar_features import REGISTRY as FEATURE_REGISTRY

# Even log-spaced coverage from fine (matches the k=5,10,15 regime) to
# coarse (previously-unsampled parent-scale territory).
DEFAULT_M_VALUES = [0.01, 0.02, 0.04, 0.10, 0.20, 0.45, 0.90]

DTM_Q = 2.0
SANITY_N_CLOUDS = 5
SANITY_SEED = 0
SHARDS_SUBDIR = "_shards"

# Filtration-value cutoff, thresh(m) = THRESH_FIT_A * m**THRESH_FIT_B *
# THRESH_SAFETY -- see the module docstring's "Filtration value cutoff"
# section for the calibration this came from (nested_thomas's own
# dtm_k5/diagrams.pkl, thresh=None, binned by effective m = 5/N, p99.9 of
# max finite death per bin, power-law fit).
THRESH_FIT_A = 2.6
THRESH_FIT_B = 0.25
THRESH_SAFETY = 1.5


def _thresh_for_m(m: float) -> float:
    return THRESH_SAFETY * THRESH_FIT_A * m**THRESH_FIT_B


# ---------------------------------------------------------------------------
# DTM by mass fraction
# ---------------------------------------------------------------------------


def _k_from_m(n_points: int, m: float) -> int:
    return int(np.clip(round(m * n_points), 1, max(1, n_points - 1)))


def _dtm_diagram_for_cloud(cloud: PointCloud, m: float, *, q: float = DTM_Q, maxdim: int, thresh: float | None) -> PersistenceDiagram:
    k = _k_from_m(cloud.n_points, m)
    filtration = DTMFiltration(maxdim=maxdim, k=k, q=q, thresh=thresh)
    diagram = filtration.compute(cloud)
    diagram.filtration_name = "dtm_mfrac"
    diagram.filtration_params = {**diagram.filtration_params, "m": m, "k": k, "thresh": thresh}
    return diagram


def _compute_dtm_for_records(records: list[dict], m: float, maxdim: int, tag: str, thresh: float | None) -> list[PersistenceDiagram]:
    diagrams = []
    for i, rec in enumerate(records):
        cloud = to_pointcloud(rec)
        diagrams.append(_dtm_diagram_for_cloud(cloud, m, maxdim=maxdim, thresh=thresh))
        if (i + 1) % 500 == 0 or i + 1 == len(records):
            print(f"  [{tag}] {i + 1}/{len(records)} diagrams computed", flush=True)
    return diagrams


# ---------------------------------------------------------------------------
# Degeneracy sanity check
# ---------------------------------------------------------------------------


def _is_degenerate(diagram: PersistenceDiagram, maxdim: int, eps: float = 1e-9) -> bool:
    """True if every homology dim's finite pairs are either absent or have
    ~zero persistence (birth == death), i.e. empty or fully collapsed."""
    for dim in range(maxdim + 1):
        finite = diagram.finite_pairs(dim)
        if finite.size == 0:
            continue
        if np.any((finite[:, 1] - finite[:, 0]) > eps):
            return False
    return True


def _sanity_check_dtm(records: list[dict], m_values: list[float], maxdim: int, thresh_fn) -> None:
    """Cheap (SANITY_N_CLOUDS clouds x 2 extreme m values) pre-flight check,
    meant to run before the expensive full sweep -- see --sanity-only."""
    rng = np.random.default_rng(SANITY_SEED)
    idx = rng.choice(len(records), size=min(SANITY_N_CLOUDS, len(records)), replace=False)
    extremes = (min(m_values), max(m_values))
    print(f"[sanity] checking m={extremes} (thresh={[round(thresh_fn(m), 3) for m in extremes]}) on {len(idx)} sampled clouds ...")

    degenerate_count = {m: 0 for m in extremes}
    for i in idx:
        cloud = to_pointcloud(records[int(i)])
        for m in extremes:
            diagram = _dtm_diagram_for_cloud(cloud, m, maxdim=maxdim, thresh=thresh_fn(m))
            n0 = diagram.finite_pairs(0).shape[0]
            n1 = diagram.finite_pairs(1).shape[0]
            degenerate = _is_degenerate(diagram, maxdim)
            flag = " DEGENERATE" if degenerate else ""
            print(
                f"  [sanity] cloud#{i} (N={cloud.n_points}) m={m:.2f} k={diagram.filtration_params['k']} "
                f"H0_finite={n0} H1_finite={n1}{flag}"
            )
            if degenerate:
                degenerate_count[m] += 1

    for m in extremes:
        if degenerate_count[m] == len(idx):
            raise RuntimeError(
                f"[sanity] every sampled cloud produced a degenerate (empty/collapsed) DTM diagram at "
                f"m={m:.2f}. Aborting rather than caching garbage -- adjust --m-values."
            )
    print("[sanity] OK -- extreme m values are not degenerate.")


# ---------------------------------------------------------------------------
# Bundle assembly / imaging
# ---------------------------------------------------------------------------


def _build_bundle(diagrams: list[PersistenceDiagram], records: list[dict], m: float) -> dict:
    label_names = list(records[0]["params"].keys())
    labels = np.array([[r["params"][k] for k in label_names] for r in records], dtype=float)
    return {
        "diagrams": [diagram_to_record(d) for d in diagrams],
        "labels": labels,
        "label_names": label_names,
        "seeds": [r["seed"] for r in records],
        "params": [r["params"] for r in records],
        "n_points": [r["n_points"] for r in records],
        "process": records[0].get("process", ""),
        "m": m,
        "filtration_name": "dtm_mfrac",
    }


def _entropy_by_dim(diagrams: list[PersistenceDiagram], homology_dims: tuple[int, ...]) -> dict[int, np.ndarray]:
    entropy_feature = FEATURE_REGISTRY.build("persistence_entropy", homology_dims=homology_dims)
    per_diagram = [entropy_feature.compute(d) for d in diagrams]
    return {dim: np.array([pd[dim] for pd in per_diagram]) for dim in homology_dims}


def _image_bundle(diagrams: list[PersistenceDiagram], bundle: dict, imager, homology_dims: tuple[int, ...]) -> dict:
    images = [imager.transform(d) for d in diagrams]
    image_tensors = {dim: np.stack([im[dim] for im in images]) for dim in homology_dims}
    return {
        **bundle, "image_tensors": image_tensors, "homology_dims": list(homology_dims),
        "imager_params": imager.params, "persistence_entropy": _entropy_by_dim(diagrams, homology_dims),
    }


# ---------------------------------------------------------------------------
# Sharding (diagrams only -- calibration/imaging needs the full merged set)
# ---------------------------------------------------------------------------


def _shard_bounds(n: int, index: int, total: int) -> tuple[int, int]:
    """Balanced contiguous [start, end) for shard `index` of `total` over `n`
    items -- the first `n % total` shards get one extra item."""
    base, rem = divmod(n, total)
    start = index * base + min(index, rem)
    end = start + base + (1 if index < rem else 0)
    return start, end


def _shard_path(shards_dir: Path, prefix: str, m: float, index: int, total: int) -> Path:
    return shards_dir / f"{prefix}dtm_m{m:.2f}_diagrams.shard{index:03d}of{total:03d}.pkl"


def run_shard(
    train_records: list[dict], adv_records: list[dict] | None, m_values: list[float], maxdim: int,
    shards_dir: Path, index: int, total: int, thresh_fn,
) -> None:
    if not (0 <= index < total):
        raise ValueError(f"--shard index must be in [0, {total}), got {index}.")
    shards_dir.mkdir(parents=True, exist_ok=True)

    start, end = _shard_bounds(len(train_records), index, total)
    train_shard = train_records[start:end]
    print(f"[shard {index}/{total}] train clouds [{start}:{end}) -> {len(train_shard)} clouds")
    adv_shard = None
    if adv_records is not None:
        astart, aend = _shard_bounds(len(adv_records), index, total)
        adv_shard = adv_records[astart:aend]
        print(f"[shard {index}/{total}] adversarial clouds [{astart}:{aend}) -> {len(adv_shard)} clouds")

    for m in m_values:
        thresh = thresh_fn(m)
        tag = f"shard {index}/{total} m={m:.2f} thresh={thresh:.3f}"
        diagrams = _compute_dtm_for_records(train_shard, m, maxdim, f"train {tag}", thresh)
        dump_pickle(_shard_path(shards_dir, "", m, index, total), _build_bundle(diagrams, train_shard, m))
        if adv_shard is not None:
            adv_diagrams = _compute_dtm_for_records(adv_shard, m, maxdim, f"adversarial {tag}", thresh)
            dump_pickle(_shard_path(shards_dir, "adversarial_", m, index, total), _build_bundle(adv_diagrams, adv_shard, m))

    print(f"[shard {index}/{total}] done.")


def _load_shard_bundle(shards_dir: Path, prefix: str, m: float, index: int, total: int) -> dict:
    path = _shard_path(shards_dir, prefix, m, index, total)
    if not path.exists():
        raise FileNotFoundError(f"missing shard file {path} -- did shard {index}/{total} finish (--shard {index} {total})?")
    return load_pickle(path)


def _merge_shard_bundles(bundles: list[dict], m: float) -> tuple[list[PersistenceDiagram], dict]:
    """Concatenates shard bundles (in shard order) into one full bundle,
    reconstructing live PersistenceDiagram objects from their serialized
    records for calibration/imaging."""
    diagrams = [record_to_diagram(r) for b in bundles for r in b["diagrams"]]
    bundle = {
        "diagrams": [diagram_to_record(d) for d in diagrams],
        "labels": np.concatenate([b["labels"] for b in bundles], axis=0),
        "label_names": bundles[0]["label_names"],
        "seeds": [s for b in bundles for s in b["seeds"]],
        "params": [p for b in bundles for p in b["params"]],
        "n_points": [n for b in bundles for n in b["n_points"]],
        "process": bundles[0]["process"],
        "m": m,
        "filtration_name": "dtm_mfrac",
    }
    return diagrams, bundle


def run_merge(
    out_dir: Path, shards_dir: Path, m_values: list[float], homology_dims: tuple[int, ...],
    resolution: int, sigma_pixels: float, total: int, expected_train_n: int, expected_adv_n: int | None,
) -> None:
    for m in m_values:
        tag = f"m={m:.2f}"
        train_bundles = [_load_shard_bundle(shards_dir, "", m, i, total) for i in range(total)]
        train_diagrams, train_bundle = _merge_shard_bundles(train_bundles, m)
        if len(train_diagrams) != expected_train_n:
            raise RuntimeError(
                f"[{tag}] merged {len(train_diagrams)} train diagrams, expected {expected_train_n} "
                f"(from clouds.pkl) -- a shard is incomplete or --shard was run with a different --m-values/total."
            )
        diagrams_path = out_dir / f"dtm_m{m:.2f}_diagrams.pkl"
        images_path = out_dir / f"dtm_m{m:.2f}_persistence_image.pkl"
        dump_pickle(diagrams_path, train_bundle)
        print(f"  [{tag}] merged {len(train_diagrams)} train diagrams -> {diagrams_path}")

        imager = build_calibrated_imager(train_diagrams, homology_dims=homology_dims, resolution=resolution, sigma_pixels=sigma_pixels)
        dump_pickle(images_path, _image_bundle(train_diagrams, train_bundle, imager, homology_dims))
        print(f"  [{tag}] saved -> {images_path}")

        if expected_adv_n is not None:
            adv_bundles = [_load_shard_bundle(shards_dir, "adversarial_", m, i, total) for i in range(total)]
            adv_diagrams, adv_bundle = _merge_shard_bundles(adv_bundles, m)
            if len(adv_diagrams) != expected_adv_n:
                raise RuntimeError(f"[{tag}] merged {len(adv_diagrams)} adversarial diagrams, expected {expected_adv_n}.")
            adv_diagrams_path = out_dir / f"adversarial_dtm_m{m:.2f}_diagrams.pkl"
            adv_images_path = out_dir / f"adversarial_dtm_m{m:.2f}_persistence_image.pkl"
            dump_pickle(adv_diagrams_path, adv_bundle)
            print(f"  [{tag}] merged {len(adv_diagrams)} adversarial diagrams -> {adv_diagrams_path}")
            # Same imager fit above on TRAIN diagrams only, applied frozen here.
            dump_pickle(adv_images_path, _image_bundle(adv_diagrams, adv_bundle, imager, homology_dims))
            print(f"  [{tag}] saved -> {adv_images_path}")

    print(f"\nDone. topo_superset refreshed under {out_dir}")
    print(f"Shard fragments left in place under {shards_dir} -- safe to delete once you've spot-checked the merge.")


# ---------------------------------------------------------------------------
# Single-process full sweep (default, no sharding)
# ---------------------------------------------------------------------------


def run_full(
    out_dir: Path, train_records: list[dict], adv_records: list[dict] | None, m_values: list[float],
    maxdim: int, homology_dims: tuple[int, ...], resolution: int, sigma_pixels: float, thresh_fn,
) -> None:
    for m in m_values:
        thresh = thresh_fn(m)
        tag = f"m={m:.2f} thresh={thresh:.3f}"
        train_diagrams = _compute_dtm_for_records(train_records, m, maxdim, f"train {tag}", thresh)
        train_bundle = _build_bundle(train_diagrams, train_records, m)
        diagrams_path = out_dir / f"dtm_m{m:.2f}_diagrams.pkl"
        images_path = out_dir / f"dtm_m{m:.2f}_persistence_image.pkl"
        dump_pickle(diagrams_path, train_bundle)
        print(f"  [{tag}] saved -> {diagrams_path}")

        imager = build_calibrated_imager(train_diagrams, homology_dims=homology_dims, resolution=resolution, sigma_pixels=sigma_pixels)
        dump_pickle(images_path, _image_bundle(train_diagrams, train_bundle, imager, homology_dims))
        print(f"  [{tag}] saved -> {images_path}")

        if adv_records is not None:
            adv_diagrams = _compute_dtm_for_records(adv_records, m, maxdim, f"adversarial {tag}", thresh)
            adv_bundle = _build_bundle(adv_diagrams, adv_records, m)
            adv_diagrams_path = out_dir / f"adversarial_dtm_m{m:.2f}_diagrams.pkl"
            adv_images_path = out_dir / f"adversarial_dtm_m{m:.2f}_persistence_image.pkl"
            dump_pickle(adv_diagrams_path, adv_bundle)
            print(f"  [{tag}] saved -> {adv_diagrams_path}")
            # Same imager fit above on TRAIN diagrams only, applied frozen here.
            dump_pickle(adv_images_path, _image_bundle(adv_diagrams, adv_bundle, imager, homology_dims))
            print(f"  [{tag}] saved -> {adv_images_path}")

    print(f"\nDone. topo_superset refreshed under {out_dir}")


# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path, help="RunConfig YAML path (process name / data_root / feature params)")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="path.to.field=value")
    parser.add_argument(
        "--m-values", type=float, nargs="+", default=DEFAULT_M_VALUES,
        help=f"mass-fraction channels to compute (default: {DEFAULT_M_VALUES}). Must be identical across "
             "--sanity-only/--shard/--merge calls for the same run.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--sanity-only", action="store_true", help="run just the degeneracy pre-flight check and exit")
    mode.add_argument(
        "--shard", type=int, nargs=2, metavar=("INDEX", "TOTAL"),
        help="compute diagrams only, for shard INDEX of TOTAL (0-indexed) -- for a parallel SLURM array; "
             "follow with --merge TOTAL once every shard has finished",
    )
    mode.add_argument(
        "--merge", type=int, metavar="TOTAL",
        help="merge TOTAL shards (written by --shard) into the full diagrams + calibrated images",
    )
    parser.add_argument(
        "--thresh", type=float, default=None,
        help="fixed filtration-value cutoff applied to every m (overrides the default per-m "
             f"thresh(m) = {THRESH_SAFETY}*{THRESH_FIT_A}*m^{THRESH_FIT_B} curve -- see the module docstring's "
             "'Filtration value cutoff' section). Must be identical across --sanity-only/--shard/--merge calls.",
    )
    args = parser.parse_args(argv)
    thresh_fn = (lambda m: args.thresh) if args.thresh is not None else _thresh_for_m

    cfg = load_config(args.config, overrides=args.overrides)
    data_paths = DataPaths(cfg.process.name, root=cfg.data_root or DEFAULT_DATA_ROOT)
    out_dir = data_paths.process_dir / "topo_superset"
    out_dir.mkdir(parents=True, exist_ok=True)
    shards_dir = out_dir / SHARDS_SUBDIR

    pi_cfg = next((f for f in cfg.features if f.name == "persistence_image"), None)
    if pi_cfg is None:
        raise ValueError(f"{args.config} has no features: persistence_image entry -- nothing to size the imager from.")
    homology_dims = tuple(pi_cfg.params.get("homology_dims", (0, 1)))
    resolution = int(pi_cfg.params.get("resolution", 64))
    sigma_pixels = float(pi_cfg.params.get("sigma_pixels", 2.0))
    maxdim = max(homology_dims)

    m_values = args.m_values
    print(f"m grid: {m_values}")
    print(f"thresh(m): {[round(thresh_fn(m), 3) for m in m_values]}"
          + (" (fixed --thresh override)" if args.thresh is not None else " (default per-m curve)"))
    print(f"imager params: resolution={resolution} sigma_pixels={sigma_pixels} homology_dims={homology_dims}")

    clouds_path = data_paths.clouds()
    if not clouds_path.exists():
        raise FileNotFoundError(f"{clouds_path} not found -- run scripts/generate.py first.")
    train_records = load_pickle(clouds_path)
    print(f"[train] {len(train_records)} clouds from {clouds_path}")

    adv_clouds_path = data_paths.clouds(adversarial=True)
    adv_records = load_pickle(adv_clouds_path) if adv_clouds_path.exists() else None
    if adv_records is not None:
        print(f"[adversarial] {len(adv_records)} clouds from {adv_clouds_path}")

    if args.sanity_only:
        _sanity_check_dtm(train_records, m_values, maxdim, thresh_fn)
        return

    if args.shard is not None:
        index, total = args.shard
        run_shard(train_records, adv_records, m_values, maxdim, shards_dir, index, total, thresh_fn)
        return

    if args.merge is not None:
        run_merge(
            out_dir, shards_dir, m_values, homology_dims, resolution, sigma_pixels,
            args.merge, expected_train_n=len(train_records),
            expected_adv_n=(len(adv_records) if adv_records is not None else None),
        )
        return

    # Default: single-process, does everything (sanity check + full sweep).
    _sanity_check_dtm(train_records, m_values, maxdim, thresh_fn)
    run_full(out_dir, train_records, adv_records, m_values, maxdim, homology_dims, resolution, sigma_pixels, thresh_fn)


if __name__ == "__main__":
    main()
