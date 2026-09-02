#!/usr/bin/env python3
# scripts/processing/diagrams_shard.py
"""Shard-parallel persistence-diagram computation + stitch-back, for the
current cloudforger pipeline (scripts/featurize.py's diagram stage, spread
across a SLURM array).

Computing a persistence diagram is independent per point cloud -- there is
no cross-cloud coupling at this stage (persistence-image calibration
happens later, per seed, inside the experiments). So each
(process, filtration) pair is split into N contiguous shards, one array
task computes one shard, and a final merge concatenates the shards -- in
shard order -- into exactly the bundle scripts/featurize.py would have
written:

    data/<process>/<filtration_tag>/diagrams.pkl
    data/<process>/<filtration_tag>/adversarial_diagrams.pkl

The merged file is byte-for-byte the same shape featurize.py's own
`_build_bundle` produces (keys: diagrams, labels, label_names, seeds,
params, process), so scripts/train.py and a later scripts/featurize.py run
consume it transparently -- featurize.py will load these cached diagrams
and only add persistence_image.pkl on top, no recompute.

Modes
-----
    compute  one shard  -> data/<p>/<tag>/_shards/{,adversarial_}diagrams.shard<I>of<N>.pkl
    merge    N shards    -> data/<p>/<tag>/{,adversarial_}diagrams.pkl
                            (validates stitched count against clouds.pkl;
                             refuses to write a truncated bundle)

Both splits (clouds.pkl and adversarial_clouds.pkl) are handled in one
call, sharded independently with the same contiguous-balanced partition.

Filtrations (all maxdim=1, DTM q=2.0 -- matches every configs/runs/*
filtration: block):

    rips   dtm_k5   dtm_k10   dtm_k15

Usage
-----
    # one array task, shard I of N, for one (process, filtration):
    python scripts/processing/diagrams_shard.py compute \
        --process thomas --filtration dtm_k5 --n-shards 15 --shard-index 3

    # once every compute task for that (process, filtration) has finished:
    python scripts/processing/diagrams_shard.py merge \
        --process thomas --filtration dtm_k5 --n-shards 15

compute is resumable: a shard whose output already exists is skipped
(pass --force to recompute). merge fails loudly, naming the missing shard
indices, if any partial is absent.
"""

from __future__ import annotations

import os

# Single-threaded numeric backends, set before numpy is imported anywhere
# in the chain below. Each array task is already pinned to one CPU
# (--cpus-per-task=1); GUDHI's DTM/weighted-Rips backend is internally
# multi-threaded, so without this a single task oversubscribes its node.
# Mirrors scripts/featurize.py's own preamble.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from cloudforger.core.io import dump_pickle, load_pickle
from cloudforger.core.records import diagram_to_record, to_pointcloud
from cloudforger.data_generation.filtration import REGISTRY as FILTRATION_REGISTRY
from cloudforger.paths import DEFAULT_DATA_ROOT, DataPaths

# name -> (registry key, kwargs). The name is also the on-disk path_tag()
# (asserted in build_filtration), so it matches data/<process>/<name>/.
FILTRATIONS: dict[str, tuple[str, dict]] = {
    "rips":    ("rips", {"maxdim": 1}),
    "dtm_k5":  ("dtm",  {"maxdim": 1, "k": 5,  "q": 2.0}),
    "dtm_k10": ("dtm",  {"maxdim": 1, "k": 10, "q": 2.0}),
    "dtm_k15": ("dtm",  {"maxdim": 1, "k": 15, "q": 2.0}),
}

SHARDS_SUBDIR = "_shards"

# GUDHI's DTM / weighted-Rips backend leaks memory per call -- unbounded
# growth (~80 MB/diagram, measured on job 25035368), not a fixed import
# cost. There is no in-process fix: a ProcessPoolExecutor worker forked
# after numpy/GUDHI import deadlocks (job 25038922), and `spawn` re-imports
# the whole stack every recycle. So instead the shards are kept SMALL
# (see NSHARDS in slurm/diagrams_compute.sh: ~200 diagrams/task) and each
# array task is its own fresh OS process -- the leak resets at every task
# boundary and never gets near the mem cap. Plain sequential loop below.


def build_filtration(name: str):
    reg_name, params = FILTRATIONS[name]
    filt = FILTRATION_REGISTRY.build(reg_name, **params)
    assert filt.path_tag() == name, (
        f"path_tag() {filt.path_tag()!r} != cli name {name!r}; the data/<process>/<tag>/ "
        "layout would not line up with scripts/featurize.py."
    )
    return filt


def shard_bounds(n: int, index: int, total: int) -> tuple[int, int]:
    """Balanced contiguous [start, end) for shard `index` of `total` over
    `n` items -- the first `n % total` shards get one extra item.
    Contiguous (not strided) so merging in ascending shard order
    reconstructs the original cloud order exactly."""
    base, rem = divmod(n, total)
    start = index * base + min(index, rem)
    end = start + base + (1 if index < rem else 0)
    return start, end


def _shard_path(shards_dir: Path, adversarial: bool, index: int, total: int) -> Path:
    prefix = "adversarial_" if adversarial else ""
    return shards_dir / f"{prefix}diagrams.shard{index:03d}of{total:03d}.pkl"


def _final_bundle(diagram_records: list[dict], clouds: list[dict]) -> dict:
    """Exactly scripts/featurize.py::_build_bundle's output shape."""
    label_names = list(clouds[0]["params"].keys())
    labels = np.array([[c["params"][k] for k in label_names] for c in clouds], dtype=float)
    return {
        "diagrams": diagram_records,
        "labels": labels,
        "label_names": label_names,
        "seeds": [c["seed"] for c in clouds],
        "params": [c["params"] for c in clouds],
        "process": clouds[0].get("process", ""),
    }


# ---------------------------------------------------------------------------
# compute
# ---------------------------------------------------------------------------


def run_compute(data_paths: DataPaths, filt_name: str, index: int, total: int, force: bool) -> None:
    filt = build_filtration(filt_name)
    shards_dir = data_paths.filtration_dir([filt]) / SHARDS_SUBDIR
    shards_dir.mkdir(parents=True, exist_ok=True)

    for adversarial in (False, True):
        split = "adversarial" if adversarial else "train/test"
        clouds_path = data_paths.clouds(adversarial=adversarial)
        if not clouds_path.exists():
            print(f"[{filt_name} shard {index}/{total}] {clouds_path} missing -- skipping {split} split.")
            continue

        out_path = _shard_path(shards_dir, adversarial, index, total)
        if out_path.exists() and not force:
            print(f"[{filt_name} shard {index}/{total}] {out_path.name} exists -- skipping {split} (pass --force).")
            continue

        clouds = load_pickle(clouds_path)
        n = len(clouds)
        start, end = shard_bounds(n, index, total)
        chunk = clouds[start:end]
        print(
            f"[{filt_name} shard {index}/{total}] {split}: clouds [{start}:{end}) of {n} "
            f"-> {len(chunk)} clouds",
            flush=True,
        )

        records: list[dict] = []
        for i, rec in enumerate(chunk):
            records.append(diagram_to_record(filt.compute(to_pointcloud(rec))))
            if (i + 1) % 50 == 0 or i + 1 == len(chunk):
                print(f"  [{filt_name} shard {index}/{total}] {split} {i + 1}/{len(chunk)}", flush=True)

        payload = {
            "diagrams": records,
            "shard_index": index,
            "shard_total": total,
            "start": start,
            "end": end,
            "n_total": n,
            "seeds": [c.get("seed") for c in chunk],
            "process": chunk[0].get("process", "") if chunk else "",
        }
        dump_pickle(out_path, payload)
        print(f"[{filt_name} shard {index}/{total}] {split}: wrote {out_path}")


# ---------------------------------------------------------------------------
# merge
# ---------------------------------------------------------------------------


def run_merge(data_paths: DataPaths, filt_name: str, total: int, force: bool, keep_shards: bool) -> None:
    filt = build_filtration(filt_name)
    filt_dir = data_paths.filtration_dir([filt])
    shards_dir = filt_dir / SHARDS_SUBDIR
    wrote_something = False

    for adversarial in (False, True):
        split = "adversarial" if adversarial else "train/test"
        clouds_path = data_paths.clouds(adversarial=adversarial)
        if not clouds_path.exists():
            print(f"[{filt_name} merge] {clouds_path} missing -- skipping {split} split.")
            continue

        final_path = data_paths.diagrams([filt], adversarial=adversarial)
        if final_path.exists() and not force:
            print(f"[{filt_name} merge] {final_path} exists -- skipping {split} (pass --force).")
            continue

        clouds = load_pickle(clouds_path)
        n = len(clouds)

        records: list[dict] = []
        missing: list[int] = []
        expected_start = 0
        for i in range(total):
            sp = _shard_path(shards_dir, adversarial, i, total)
            if not sp.exists():
                missing.append(i)
                continue
            part = load_pickle(sp)
            if part.get("n_total") != n:
                raise SystemExit(
                    f"[{filt_name} merge] {sp.name} was computed against n_total={part.get('n_total')}, "
                    f"but {clouds_path.name} now has {n} clouds -- the shards are stale. "
                    f"Recompute (diagrams_shard.py compute ... --force) then merge again."
                )
            if part.get("start") != expected_start:
                raise SystemExit(
                    f"[{filt_name} merge] {sp.name} start={part.get('start')} != expected {expected_start} "
                    f"-- shard files were written with a different --n-shards. Recompute with a consistent value."
                )
            records.extend(part["diagrams"])
            expected_start = part.get("end")

        if missing:
            raise SystemExit(
                f"[{filt_name} merge] {split}: missing shard(s) {missing} of {total} under {shards_dir}. "
                f"Did every compute task finish? compute is resumable -- rerun just those indices "
                f"(sbatch --array=<ids> slurm/diagrams_compute.sh), then merge again."
            )
        if len(records) != n:
            raise SystemExit(
                f"[{filt_name} merge] {split}: stitched {len(records)} diagrams but {clouds_path.name} "
                f"has {n} clouds -- refusing to write a truncated bundle."
            )

        dump_pickle(final_path, _final_bundle(records, clouds))
        print(f"[{filt_name} merge] {split}: {len(records)} diagrams -> {final_path}")
        wrote_something = True

    if wrote_something and not keep_shards:
        removed = 0
        for adversarial in (False, True):
            for i in range(total):
                sp = _shard_path(shards_dir, adversarial, i, total)
                if sp.exists():
                    sp.unlink()
                    removed += 1
        try:
            shards_dir.rmdir()
        except OSError:
            pass  # not empty (a different --n-shards left fragments) -- leave it
        print(f"[{filt_name} merge] removed {removed} shard fragment(s) under {shards_dir}")


# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("mode", choices=["compute", "merge"])
    parser.add_argument("--process", required=True, help="e.g. thomas / nested_thomas / matern_cluster")
    parser.add_argument("--filtration", required=True, choices=sorted(FILTRATIONS))
    parser.add_argument("--n-shards", type=int, required=True)
    parser.add_argument("--shard-index", type=int, default=None, help="compute mode: which shard (0-indexed)")
    parser.add_argument("--data-root", type=Path, default=None, help=f"default: {DEFAULT_DATA_ROOT}")
    parser.add_argument("--force", action="store_true", help="recompute / rewrite outputs that already exist")
    parser.add_argument(
        "--keep-shards", action="store_true",
        help="merge mode: keep the _shards/ fragments instead of deleting them after a successful merge",
    )
    args = parser.parse_args(argv)

    if args.n_shards < 1:
        raise SystemExit("--n-shards must be >= 1")

    data_paths = DataPaths(args.process, root=args.data_root or DEFAULT_DATA_ROOT)

    if args.mode == "compute":
        if args.shard_index is None:
            raise SystemExit("--shard-index is required in compute mode (pass $SLURM_ARRAY_TASK_ID-derived value).")
        if not (0 <= args.shard_index < args.n_shards):
            raise SystemExit(f"--shard-index {args.shard_index} out of range for --n-shards {args.n_shards}.")
        run_compute(data_paths, args.filtration, args.shard_index, args.n_shards, args.force)
    else:
        run_merge(data_paths, args.filtration, args.n_shards, args.force, args.keep_shards)

    print("Done.")


if __name__ == "__main__":
    main()
