# scripts/processing/params/diagrams_slurm.py
"""
Parallelize the `compute_diagrams` stage across a SLURM array.

Model
-----
Computing a persistence diagram is independent per point cloud (there is no
cross-cloud coupling in this stage -- calibration only happens later, in
`vectorize_diagrams`). So we can split the clouds of each split into N
contiguous groups, hand one group to each array task, and later concatenate
the per-group diagrams back into the single canonical file. The merged result
is bit-identical to a serial `compute_diagrams` run.

Two modes
---------
    --mode compute   (default)  one array task -> diagrams for its group only,
                                written to  <base>/_chunks/<name>.group<NNNN>.pkl
    --mode merge                combine every group's partial into the final
                                <base>/<name>.pkl (e.g. diagrams.pkl)

Typical flow: submit the array (compute) job, then submit the merge job with a
Slurm dependency on the array (see the accompanying .sbatch files).

Where the format lives
----------------------
Everything format-specific is in the "RECORD FORMAT ADAPTER" block below. It
assumes each .pkl is one of:
    * a dict  { cloud_id: value }                (matches `formats: dict`)
    * a list  [ value, ... ]
    * a 2-tuple (data, meta) where data is one of the above
which is what `records.load_diagrams` returning `(container, meta)` implies.
If your on-disk objects are wrapped differently, adapt `_peel` / `_load` /
`_dump` in that one block -- nothing else needs to change.
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np

# NOTE: heavy project imports (pipeline_lib, cloudforger, ...) are done lazily
# inside build_pipeline() / run_compute() so this file's pure helpers can be
# imported and unit-tested without the full runtime environment.


# ======================================================================
# RECORD FORMAT ADAPTER  --  the only format-dependent code in this file
# ======================================================================

def _load(path: Path) -> Any:
    with open(path, "rb") as fh:
        return pickle.load(fh)


def _dump(obj: Any, path: Path) -> None:
    """Atomic write: a preempted/timed-out task never leaves a truncated .pkl."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def _peel(obj: Any) -> tuple[Any, Any, Callable[[Any], Any]]:
    """
    Return (data, meta, rebuild) where `data` is the dict/list payload and
    `rebuild(new_data)` reconstructs the original top-level shape.
    `meta` is None unless the object is a (data, meta) tuple.
    """
    if isinstance(obj, tuple) and len(obj) == 2 and isinstance(obj[0], (dict, list)):
        meta = obj[1]
        return obj[0], meta, (lambda nd: (nd, meta))
    if isinstance(obj, (dict, list)):
        return obj, None, (lambda nd: nd)
    raise TypeError(
        f"Unsupported record container: {type(obj).__name__}. "
        "Edit the RECORD FORMAT ADAPTER (_peel) to teach the splitter about "
        "your on-disk structure."
    )


def _is_diagram_bundle(obj: Any) -> bool:
    """True for compute_diagrams_from_clouds's output shape: a single dict per
    group holding parallel per-cloud fields (diagrams/labels/params/seeds)
    alongside shared scalar metadata -- NOT a {cloud_id: value} mapping."""
    return isinstance(obj, dict) and "diagrams" in obj


def container_len(obj: Any) -> int:
    if _is_diagram_bundle(obj):
        return len(obj["diagrams"])
    data, _meta, _rebuild = _peel(obj)
    return len(data)


def subset_container(obj: Any, start: int, stop: int) -> Any:
    """Contiguous slice [start:stop) preserving the top-level shape and order."""
    data, _meta, rebuild = _peel(obj)
    if isinstance(data, dict):
        return rebuild(dict(list(data.items())[start:stop]))
    return rebuild(data[start:stop])


def merge_diagram_bundles(objs: list[dict[str, Any]]) -> dict[str, Any]:
    """Concatenate per-group diagram bundles (compute_diagrams_from_clouds's
    output shape) into one, preserving group order. Per-cloud fields
    (diagrams/labels/params/seeds) are concatenated; shared scalar fields
    (process/filtration_params/label_names) are taken from group 0."""
    first = objs[0]
    for key in ("process", "filtration_params", "label_names"):
        if any(o.get(key) != first.get(key) for o in objs[1:]):
            print(
                f"  WARNING: per-group '{key}' differs across partials; keeping the "
                "value from group 0. Inspect if downstream stages rely on it.",
                file=sys.stderr,
            )

    return {
        "diagrams": [d for o in objs for d in o["diagrams"]],
        "labels": np.concatenate([np.asarray(o["labels"]) for o in objs], axis=0),
        "label_names": first["label_names"],
        "params": [p for o in objs for p in o["params"]],
        "seeds": [s for o in objs for s in o["seeds"]],
        "process": first["process"],
        "filtration_params": first["filtration_params"],
    }


def merge_containers(objs: list[Any]) -> Any:
    """Union dicts / concat lists, preserving group order. Metas must agree."""
    if objs and all(_is_diagram_bundle(o) for o in objs):
        return merge_diagram_bundles(objs)

    peeled = [_peel(o) for o in objs]
    datas = [d for d, _m, _r in peeled]
    metas = [m for _d, m, _r in peeled]
    rebuild = peeled[0][2]

    if any(m != metas[0] for m in metas[1:]):
        print(
            "  WARNING: per-group metadata differs across partials; keeping the "
            "metadata from group 0. Inspect if downstream stages rely on it.",
            file=sys.stderr,
        )

    if all(isinstance(d, dict) for d in datas):
        out: dict[Any, Any] = {}
        for d in datas:
            overlap = out.keys() & d.keys()
            if overlap:
                raise ValueError(
                    f"Overlapping keys while merging groups: {sorted(overlap)[:5]}... "
                    "This usually means --n-groups differs between compute and merge."
                )
            out.update(d)
        return rebuild(out)

    if all(isinstance(d, list) for d in datas):
        out_list: list[Any] = []
        for d in datas:
            out_list.extend(d)
        return rebuild(out_list)

    raise TypeError("Mixed container types across group partials; cannot merge.")


# ======================================================================
# Chunking arithmetic (pure)
# ======================================================================

def split_bounds(n: int, n_groups: int) -> list[tuple[int, int]]:
    """
    Balanced *contiguous* partition of range(n) into n_groups blocks, like
    numpy.array_split. Contiguous (not strided) so a list-based merge in
    ascending group order reconstructs the original ordering exactly.
    """
    if n_groups < 1:
        raise ValueError("n_groups must be >= 1")
    base, rem = divmod(n, n_groups)
    bounds: list[tuple[int, int]] = []
    start = 0
    for i in range(n_groups):
        size = base + (1 if i < rem else 0)
        bounds.append((start, start + size))
        start += size
    return bounds


def grouped_name(name: str, group_id: int) -> str:
    """'diagrams.pkl', 7 -> 'diagrams.group0007.pkl'."""
    p = Path(name)
    return f"{p.stem}.group{group_id:04d}{p.suffix}"


# ======================================================================
# Pipeline wiring
# ======================================================================

def build_pipeline(config_path: str, process: str | None, dimension: int):
    """Load the config, scope it to one dimension and the compute_diagrams
    stage, then hand it to the existing ParamsPipeline for path resolution."""
    from pipeline import resolve_execution
    from pipeline_lib.config import load_pipeline_config
    from pipeline_lib.orchestrator import ParamsPipeline

    config = load_pipeline_config(config_path)
    if process:
        config["process"] = process
    config["dimensions"] = [int(dimension)]
    config.pop("dimension_indices", None)  # avoid the length-mismatch check
    config["stages"] = ["compute_diagrams"]

    resolve_execution(config)          # parity with diagrams_dtm.py
    return ParamsPipeline(config)


def run_split(
    pl,
    dim: int,
    splits: list[str],
    n_groups: int,
    overwrite: bool,
) -> None:
    """One-time, single-node pre-split. Loads each full clouds file ONCE and
    writes n_groups contiguous chunk files under _chunks/. After this, each
    compute task reads only its own ~1/n_groups slice instead of the whole
    file -- this is what keeps the array tasks under their memory limit."""
    chunks_dir = pl.base_dir(dim) / "_chunks"

    for split in splits:
        clouds_path = pl.split_path(dim, split, "clouds")
        if not clouds_path.exists():
            print(f"  {split}: missing clouds, skipping: {clouds_path}")
            continue

        clouds_obj = _load(clouds_path)          # the only full-file load, once
        n = container_len(clouds_obj)
        clouds_name = pl.splits[split]["clouds"]
        bounds = split_bounds(n, n_groups)

        for g, (start, stop) in enumerate(bounds):
            chunk = chunks_dir / grouped_name(clouds_name, g)
            if chunk.exists() and not overwrite:
                continue
            _dump(subset_container(clouds_obj, start, stop), chunk)
        print(f"  {split}: {n} clouds -> {n_groups} chunks in {chunks_dir}")

        del clouds_obj  # free before the next (possibly large) split


def run_compute(
    pl,
    dim: int,
    splits: list[str],
    n_groups: int,
    group_id: int,
    overwrite: bool,
    keep_cloud_chunks: bool,
) -> None:
    from pipeline_lib import diagrams
    from pipeline_lib.io import require_format

    if not (0 <= group_id < n_groups):
        raise SystemExit(f"--group-id {group_id} out of range for --n-groups {n_groups}")

    filtration = diagrams.build_filtration(pl.config)
    diagram_format = require_format(
        pl.config.get("formats", {}).get("diagrams", "dict"), "formats.diagrams"
    )
    chunks_dir = pl.base_dir(dim) / "_chunks"

    for split in splits:
        clouds_path = pl.split_path(dim, split, "clouds")
        if not clouds_path.exists():
            print(f"  [group {group_id}] {split}: missing clouds, skipping: {clouds_path}")
            continue

        diag_name = pl.splits[split]["diagrams"]
        partial = chunks_dir / grouped_name(diag_name, group_id)
        if partial.exists() and not overwrite:
            print(f"  [group {group_id}] {split}: partial exists, skipping: {partial.name}")
            continue

        chunk = chunks_dir / grouped_name(pl.splits[split]["clouds"], group_id)

        if chunk.exists():
            # Pre-split path: hand the chunk straight to the compute function.
            # This process never loads the clouds container itself -- peak
            # memory is one slice's worth, set by --n-groups.
            print(f"  [group {group_id}] {split}: chunk {chunk.name} -> {partial.name}")
        else:
            # Fallback (no --mode split was run): load the whole file and slice.
            # Convenient for small data, but this is the memory-heavy path.
            print(f"  [group {group_id}] {split}: no pre-split chunk; loading full file")
            clouds_obj = _load(clouds_path)
            n = container_len(clouds_obj)
            if n == 0:
                print(f"  [group {group_id}] {split}: 0 clouds, skipping")
                continue
            start, stop = split_bounds(n, n_groups)[group_id]
            _dump(subset_container(clouds_obj, start, stop), chunk)
            del clouds_obj

        diagrams.compute_diagrams_from_clouds(
            chunk, partial, filtration, pl.process, diagram_format
        )

        # The cloud slice is single-use (only this group reads it); drop it.
        # Keep the diagram partial -- the merge step needs it.
        if not keep_cloud_chunks:
            chunk.unlink(missing_ok=True)


def run_merge(
    pl,
    dim: int,
    splits: list[str],
    n_groups: int,
    overwrite: bool,
    clean_chunks: bool,
) -> None:
    chunks_dir = pl.base_dir(dim) / "_chunks"

    for split in splits:
        final = pl.split_path(dim, split, "diagrams")
        if final.exists() and not overwrite:
            print(f"  {split}: final exists, skipping (use --overwrite): {final}")
            continue

        diag_name = pl.splits[split]["diagrams"]
        objs, missing = [], []
        for g in range(n_groups):
            p = chunks_dir / grouped_name(diag_name, g)
            if p.exists():
                objs.append(_load(p))
            else:
                missing.append(g)

        if missing:
            raise SystemExit(
                f"  {split}: missing partial(s) for groups {missing} in {chunks_dir}. "
                "Did every array task finish? Re-run the failed group ids "
                "(compute mode is resumable), then merge again."
            )

        merged = merge_containers(objs)
        _dump(merged, final)
        print(f"  {split}: merged {n_groups} groups -> {final}  ({container_len(merged)} items)")

    if clean_chunks:
        for split in splits:
            diag_name = pl.splits[split]["diagrams"]
            for g in range(n_groups):
                (chunks_dir / grouped_name(diag_name, g)).unlink(missing_ok=True)
        try:
            chunks_dir.rmdir()
            print(f"  removed {chunks_dir}")
        except OSError:
            pass  # not empty (other splits / dims), leave it


# ======================================================================
# CLI
# ======================================================================

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Parallel compute_diagrams over a SLURM array.")
    p.add_argument("--config", required=True)
    p.add_argument("--process", default=None, help="Override config['process'].")
    p.add_argument("--dimension", type=int, required=True)
    p.add_argument("--n-groups", type=int, required=True)
    p.add_argument("--group-id", type=int, default=None,
                   help="Which chunk this task handles (compute mode). Use $SLURM_ARRAY_TASK_ID.")
    p.add_argument("--splits", nargs="+", required=True)
    p.add_argument("--mode", choices=["split", "compute", "merge"], default="compute")
    p.add_argument("--overwrite", action="store_true",
                   help="compute: recompute existing partials. merge: rewrite existing final files.")
    p.add_argument("--keep-cloud-chunks", action="store_true",
                   help="Keep the temporary per-group clouds slices (debugging).")
    p.add_argument("--clean-chunks", action="store_true",
                   help="merge: delete _chunks partials after a successful merge.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    pl = build_pipeline(args.config, args.process, args.dimension)

    unknown = set(args.splits) - set(pl.splits)
    if unknown:
        raise SystemExit(f"Unknown split(s): {sorted(unknown)}. Config has {sorted(pl.splits)}.")

    dim = int(args.dimension)
    print(f"[{args.mode}] process={pl.process} dim={dim} "
          f"n_groups={args.n_groups} splits={args.splits}")
    print(f"  base dir: {pl.base_dir(dim)}")

    if args.mode == "split":
        run_split(pl, dim, args.splits, args.n_groups, args.overwrite)
    elif args.mode == "compute":
        if args.group_id is None:
            raise SystemExit("--group-id is required in compute mode (pass $SLURM_ARRAY_TASK_ID).")
        run_compute(pl, dim, args.splits, args.n_groups, args.group_id,
                    args.overwrite, args.keep_cloud_chunks)
    else:  # merge
        run_merge(pl, dim, args.splits, args.n_groups, args.overwrite, args.clean_chunks)

    print("Done.")


if __name__ == "__main__":
    main()