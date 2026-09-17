# src/cloudforger/generation/pipeline.py
"""The four DV3 jobs (generation.tex, "How the generation runs"):

    make_cells   B cells and C ladders -> dv3_cells.csv (reviewed at D1; committed)
    make_plan    dv3.yaml + null tables + cells -> plan.csv (+ plan.json with checksums)
    run_shard    one (set, family, shard) of the plan -> shard files; idempotent
    merge        all shards -> per-(set, family) outputs + dataset card;
                 refuses if any case is missing or any shard is stale
    regen_case   rebuild one case from its id and compare with what is stored
"""

from __future__ import annotations

import math
import platform
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np
import scipy
import yaml

from scipy.spatial import cKDTree

from ..core.io import dump_pickle
from .design import build_cells, cell_shape, cells_bytes, read_cells
from .plan import build_plan, draw_from_row
from .prior import build_priors, draw, fixed
from .samplers import WINDOW, points_sha1, simulate
from .seeding import PARAMS, PATTERN, case_rng, key_str, parse_case_id
from .spec import PRIOR_SETS, Spec, load_spec
from .store import (
    MANIFEST_COLUMNS, PLAN_COLUMNS, DV3Paths, atomic_write_bytes, csv_bytes, load_points,
    points_npz_bytes, read_csv, read_json, sha256_bytes, sha256_file, write_json,
)

REPO = Path(__file__).resolve().parents[3]


class PipelineError(RuntimeError):
    pass


def git_state() -> tuple[str | None, bool | None]:
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=True).stdout
    try:
        return git("rev-parse", "HEAD").strip(), bool(git("status", "--porcelain").strip())
    except (OSError, subprocess.CalledProcessError):
        return None, None


def _null_sha(spec: Spec) -> str:
    if not spec.null_tables_path.exists():
        raise PipelineError(f"no null tables at {spec.null_tables_path}; run `nulls` first")
    return sha256_file(spec.null_tables_path)


# --- B/C cells ----------------------------------------------------------------------

def make_cells(spec: Spec, force: bool = False) -> list[dict[str, Any]]:
    path = spec.cells_path
    if path is None:
        raise PipelineError(f"{spec.path} has no `cells:` entry")
    data = cells_bytes(build_cells(spec, load_tables(spec.null_tables_path)))
    if path.exists() and path.read_bytes() != data and not force:
        raise PipelineError(f"{path} exists and differs; pass --force to replace it (plans made from it go stale)")
    atomic_write_bytes(path, data)
    return read_cells(path)


def _cells_sha(spec: Spec) -> str | None:
    if spec.cells_path is None:
        return None
    if not spec.cells_path.exists():
        raise PipelineError(f"no cells file at {spec.cells_path}; run `cells` first")
    return sha256_file(spec.cells_path)


# --- plan ----------------------------------------------------------------------

def make_plan(spec: Spec, paths: DV3Paths, force: bool = False, jobs: int = 1) -> dict[str, Any]:
    null_sha = _null_sha(spec)
    cells_sha = _cells_sha(spec)
    data = csv_bytes(PLAN_COLUMNS, build_plan(spec, jobs=jobs))
    plan_sha = sha256_bytes(data)

    if paths.plan_meta.exists() and not force:
        old = read_json(paths.plan_meta)
        if old["plan_sha256"] == plan_sha:
            return old
        raise PipelineError(
            f"{paths.plan_csv} exists and differs from the plan this spec produces "
            f"(was {old['plan_sha256'][:12]}, now {plan_sha[:12]}). Pass --force to replace it; "
            f"existing shards will then be treated as stale."
        )

    commit, dirty = git_state()
    meta = {
        "dv": 3,
        "spec_path": str(spec.path),
        "spec_sha256": spec.sha256,
        "null_tables_sha256": null_sha,
        "cells_sha256": cells_sha,
        "plan_sha256": plan_sha,
        "n_cases": data.count(b"\n") - 1,
        "git_commit": commit,
        "git_dirty": dirty,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    atomic_write_bytes(paths.plan_csv, data)
    write_json(paths.plan_meta, meta)
    return meta


def load_plan(spec: Spec, paths: DV3Paths) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not paths.plan_meta.exists():
        raise PipelineError(f"no plan at {paths.plan_meta}; run `plan` first")
    meta = read_json(paths.plan_meta)
    if meta["spec_sha256"] != spec.sha256:
        raise PipelineError(f"{spec.path} changed since the plan was made; re-run `plan --force`")
    if meta["null_tables_sha256"] != _null_sha(spec):
        raise PipelineError(f"{spec.null_tables_path} changed since the plan was made; re-run `plan --force`")
    if meta.get("cells_sha256") != _cells_sha(spec):
        raise PipelineError(f"{spec.cells_path} changed since the plan was made; re-run `plan --force`")
    if sha256_file(paths.plan_csv) != meta["plan_sha256"]:
        raise PipelineError(f"{paths.plan_csv} does not match the checksum in {paths.plan_meta}")
    return read_csv(paths.plan_csv), meta


def list_shards(rows: list[dict[str, Any]]) -> list[tuple[str, str, int]]:
    """Every (set, family, shard) in plan order; the position is the SLURM array task id."""
    return list(dict.fromkeys((r["set"], r["family"], r["shard"]) for r in rows))


# --- run_shard -----------------------------------------------------------------

def shard_is_valid(paths: DV3Paths, set_: str, family: str, shard: int, plan_sha: str) -> bool:
    npz, csv_, done = paths.shard_files(set_, family, shard)
    if not (npz.exists() and csv_.exists() and done.exists()):
        return False
    stamp = read_json(done)
    return (stamp.get("plan_sha256") == plan_sha
            and stamp.get("npz_sha256") == sha256_file(npz)
            and stamp.get("csv_sha256") == sha256_file(csv_))


def simulate_row(spec: Spec, row: dict[str, Any], prior) -> tuple[np.ndarray, dict[str, Any]]:
    rng = case_rng(spec.root, row["set"], row["family"], row["index"], PATTERN)
    sim = simulate(row["family"], draw_from_row(row, prior), rng, n_min=spec.n_min)
    return sim.points, {**row, "n": len(sim.points), **sim.diagnostics}


def run_shard(spec: Spec, paths: DV3Paths, set_: str, family: str, shard: int, force: bool = False) -> str:
    rows, meta = load_plan(spec, paths)
    if not force and shard_is_valid(paths, set_, family, shard, meta["plan_sha256"]):
        return "skipped (already complete)"

    mine = [r for r in rows if (r["set"], r["family"], r["shard"]) == (set_, family, shard)]
    if not mine:
        raise PipelineError(f"plan has no cases for set={set_} family={family} shard={shard}")

    prior = build_priors(spec)[family]
    clouds, manifest = [], []
    for row in mine:
        points, mrow = simulate_row(spec, row, prior)
        clouds.append(points)
        manifest.append(mrow)

    npz_data = points_npz_bytes(clouds, [r["index"] for r in mine])
    csv_data = csv_bytes(MANIFEST_COLUMNS, manifest)
    npz, csv_, done = paths.shard_files(set_, family, shard)
    atomic_write_bytes(npz, npz_data)
    atomic_write_bytes(csv_, csv_data)
    # Written last: its presence, with matching hashes, is what "complete" means.
    write_json(done, {
        "plan_sha256": meta["plan_sha256"],
        "npz_sha256": sha256_bytes(npz_data),
        "csv_sha256": sha256_bytes(csv_data),
        "n_cases": len(mine),
    })
    return f"wrote {len(mine)} cases"


def _shard_task(args: tuple[str, str, str, str, int, bool]) -> tuple[str, str, int, str]:
    spec_path, root, set_, family, shard, force = args
    return set_, family, shard, run_shard(load_spec(spec_path), DV3Paths(root), set_, family, shard, force)


def run_all_shards(spec: Spec, paths: DV3Paths, jobs: int = 1, force: bool = False, log=print) -> None:
    shards = list_shards(load_plan(spec, paths)[0])
    tasks = [(str(spec.path), str(paths.root), *s, force) for s in shards]
    if jobs > 1:
        with get_context("spawn").Pool(jobs) as pool:
            for i, (set_, family, shard, status) in enumerate(pool.imap_unordered(_shard_task, tasks)):
                log(f"[{i + 1}/{len(tasks)}] {set_}/{family}/shard {shard}: {status}")
    else:
        for i, t in enumerate(tasks):
            _, _, shard, status = _shard_task(t)
            log(f"[{i + 1}/{len(tasks)}] {t[2]}/{t[3]}/shard {shard}: {status}")


# --- merge ----------------------------------------------------------------------

def v0_counts(manifest: list[dict[str, Any]]) -> dict[str, Any]:
    """Check V0 on a prior-drawn set: mean n / nbar within max(3 s.e., 1%) of 1."""
    ratio = np.array([r["n"] / r["nbar"] for r in manifest])
    mean, se = float(ratio.mean()), float(ratio.std(ddof=1) / math.sqrt(len(ratio)))
    return {"mean_n_over_nbar": mean, "se": se, "pass": abs(mean - 1.0) <= max(3.0 * se, 0.01)}


def v0_cells(manifest: list[dict[str, Any]], alpha: float = 0.01) -> dict[str, Any]:
    """Check V0 per B/C cell, with the pre-registered Holm correction over the
    (set, family) battery at family-wise alpha: a cell fails if |mean n / nbar - 1|
    exceeds 1% and Holm rejects mean = 1. `n_flagged` counts cells beyond the
    uncorrected max(3 s.e., 1%) rule, for reference."""
    from scipy.stats import norm

    groups: dict[tuple, list[float]] = defaultdict(list)
    for r in manifest:
        groups[(r["cell_id"], r["level_id"])].append(r["n"] / r["nbar"])
    dev, z, flagged = [], [], 0
    for ratios in groups.values():
        x = np.array(ratios)
        se = x.std(ddof=1) / math.sqrt(len(x))
        dev.append(abs(x.mean() - 1.0))
        z.append(dev[-1] / se if se > 0 else 0.0)
        flagged += int(dev[-1] > max(3.0 * se, 0.01))
    p = 2.0 * norm.sf(np.array(z))
    rejected, m = np.zeros(len(p), bool), len(p)
    for j, i in enumerate(np.argsort(p)):          # Holm step-down
        if p[i] > alpha / (m - j):
            break
        rejected[i] = True
    fail = int(np.sum(rejected & (np.array(dev) > 0.01)))
    return {"n_cells": m, "n_flagged": flagged, "n_fail": fail, "max_z": float(max(z)),
            "min_p": float(p.min()), "pass": fail == 0}


def v3_hard_core(clouds: list[np.ndarray], manifest: list[dict[str, Any]]) -> dict[str, Any]:
    """Check V3 (Matern II): every pattern's minimum inter-point distance is >= R."""
    worst = min((float(cKDTree(p).query(p, k=2)[0][:, 1].min()) / r["R"] for p, r in zip(clouds, manifest)
                 if len(p) >= 2), default=float("inf"))
    return {"min_nn_over_R": worst, "pass": worst >= 1.0}


def v7_embedding(manifest: list[dict[str, Any]]) -> dict[str, Any]:
    """Check V7 (LGCP, per draw): min / max eigenvalue of the embedding >= -1e-10."""
    worst = min(r["min_eig"] for r in manifest)
    grids = sorted({r["grid_M"] for r in manifest})
    return {"min_eig_ratio": worst, "grid_M_used": {str(M): sum(r["grid_M"] == M for r in manifest) for M in grids},
            "pad_P_used": sorted({r["pad_P"] for r in manifest}), "pass": worst >= -1e-10}


def delta_summary(manifest: list[dict[str, Any]]) -> dict[str, Any]:
    d = np.array([r["delta_tilde"] for r in manifest])
    return {"frac_le_1": float((d <= 1).mean()), "q05": float(np.quantile(d, 0.05)),
            "median": float(np.median(d)), "q95": float(np.quantile(d, 0.95))}


def compat_record(row: dict[str, Any], points: np.ndarray, prior) -> dict[str, Any]:
    """The legacy cloud record (core/records.py) so featurize/diagrams run unchanged.
    `seed` is the per-(set, family) index: the legacy pipeline uses it as a cloud id."""
    params = {"nbar": row["nbar"], **{k: row[k] for k in prior.design_keys + prior.model_keys}}
    params.update({k: row[k] for k in ("tau_K", "tau_K2", "delta_tilde") if row[k] is not None})
    return {
        "points": points,
        "params": params,
        "process": row["family"],
        "seed": row["index"],
        "n_points": len(points),
        "dimension": 2,
        "region": {"low": WINDOW.low, "high": WINDOW.high},
        "case_id": row["case_id"],
        "split": row["split"],
        "cell_id": row["cell_id"],
        "level_id": row["level_id"],
        "rep": row["rep"],
        "dv": 3,
    }


def merge(spec: Spec, paths: DV3Paths, allow_dirty: bool = False) -> dict[str, Any]:
    commit, dirty = git_state()
    if dirty and not allow_dirty:
        raise PipelineError("git tree is dirty; commit first (the dataset card must name a clean commit) "
                            "or pass --allow-dirty for a throwaway build")

    rows, meta = load_plan(spec, paths)
    priors = build_priors(spec)
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[(r["set"], r["family"])].append(r)

    missing = [s for s in list_shards(rows) if not shard_is_valid(paths, *s, meta["plan_sha256"])]
    if missing:
        shown = ", ".join(f"{s}/{f}/{k}" for s, f, k in missing[:10])
        raise PipelineError(f"{len(missing)} shard(s) missing, incomplete or stale: {shown}")

    outputs = {}
    for (set_, family), expected in groups.items():
        clouds, manifest = [], []
        for shard in sorted({r["shard"] for r in expected}):
            npz, csv_, _ = paths.shard_files(set_, family, shard)
            pts, _ = load_points(npz)
            clouds += pts
            manifest += read_csv(csv_)

        got = [r["case_id"] for r in manifest]
        if got != [r["case_id"] for r in expected]:
            raise PipelineError(f"{set_}/{family}: merged case ids do not match the plan")
        for pts, r in zip(clouds, manifest):
            if points_sha1(pts) != r["sha1"] or len(pts) != r["n"]:
                raise PipelineError(f"{r['case_id']}: stored points do not match the manifest row")

        atomic_write_bytes(paths.points(set_, family), points_npz_bytes(clouds, [r["index"] for r in manifest]))
        atomic_write_bytes(paths.manifest(set_, family), csv_bytes(MANIFEST_COLUMNS, manifest))
        dump_pickle(paths.clouds(set_, family),
                    [compat_record(r, p, priors[family]) for r, p in zip(manifest, clouds)])

        n = np.array([r["n"] for r in manifest])
        outputs[f"{set_}/{family}"] = {
            "n_cases": len(manifest),
            "splits": {s: sum(r["split"] == s for r in manifest) for s in ("train", "val", "test")},
            "n_points": {"total": int(n.sum()), "min": int(n.min()), "max": int(n.max())},
            "points_sha256": sha256_file(paths.points(set_, family)),
            "delta_tilde": delta_summary(manifest),
            "V0": v0_counts(manifest) if set_ in PRIOR_SETS else v0_cells(manifest),
            "pattern_redraws": int(sum(r["pattern_tries"] - 1 for r in manifest)),
        }
        if family == "matern2":
            outputs[f"{set_}/{family}"]["V3"] = v3_hard_core(clouds, manifest)
        if family == "lgcp":
            outputs[f"{set_}/{family}"]["V7"] = v7_embedding(manifest)

    card = {
        "dataset": "DV3",
        "spec_path": str(spec.path),
        "spec_sha256": spec.sha256,
        "null_tables_sha256": meta["null_tables_sha256"],
        "cells_sha256": meta.get("cells_sha256"),
        "plan_sha256": meta["plan_sha256"],
        "n_min": spec.n_min,
        "git_commit": commit,
        "git_dirty": dirty,
        "merged": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "versions": {
            "python": sys.version.split()[0], "numpy": np.__version__,
            "scipy": scipy.__version__, "pyyaml": yaml.__version__, "platform": platform.platform(),
        },
        "outputs": outputs,
    }
    write_json(paths.card, card)
    return card


# --- regen_case ------------------------------------------------------------------

def regen_case(spec: Spec, paths: DV3Paths, cid: str) -> dict[str, Any]:
    """Rebuild one case alone and compare it with the stored copy: theta
    bit-for-bit (prior-drawn sets), points bit-for-bit on the same build."""
    set_, family, index = parse_case_id(cid)
    manifest = read_csv(paths.manifest(set_, family))
    pos = next((i for i, r in enumerate(manifest) if r["case_id"] == cid), None)
    row = manifest[pos] if pos is not None else None
    if row is None:
        raise PipelineError(f"{cid} not found in {paths.manifest(set_, family)}")
    if row["spawn_key"] != key_str(set_, family, index):
        raise PipelineError(f"{cid}: stored spawn key {row['spawn_key']} != rebuilt {key_str(set_, family, index)}")

    prior = build_priors(spec)[family]
    report: dict[str, Any] = {"case_id": cid, "spawn_key": row["spawn_key"]}

    stored = draw_from_row(row, prior)
    if set_ in PRIOR_SETS:
        d = draw(prior, case_rng(spec.root, set_, family, index, PARAMS), spec.nbar_low, spec.nbar_high)
    else:
        cell = next(c for c in read_cells(spec.cells_path)
                    if (c["set"], c["family"], c["cell_id"], c["level_id"]) == (set_, family, row["cell_id"], row["level_id"]))
        d = fixed(prior, cell["nbar"], cell_shape(cell, prior))
    report["theta_identical"] = (d.nbar == stored.nbar and d.design == stored.design and d.model == stored.model)

    points, mrow = simulate_row(spec, row, prior)
    stored_points, _ = load_points(paths.points(set_, family))
    report["n_stored"], report["n_regenerated"] = row["n"], mrow["n"]
    report["points_identical"] = (mrow["sha1"] == row["sha1"] == points_sha1(stored_points[pos])
                                  and np.array_equal(points, stored_points[pos]))
    return report
