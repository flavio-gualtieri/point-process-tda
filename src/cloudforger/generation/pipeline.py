# src/cloudforger/generation/pipeline.py
"""The four DV3 jobs (generation.tex, "How the generation runs"):

    make_plan    dv3.yaml -> plan.csv (+ plan.json with checksums)
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
from pathlib import Path
from typing import Any

import numpy as np
import scipy
import yaml

from ..core.io import dump_pickle
from .plan import build_plan, draw_from_row
from .prior import build_priors, draw
from .samplers import WINDOW, points_sha1, simulate
from .seeding import PARAMS, PATTERN, case_rng, key_str, parse_case_id
from .spec import PRIOR_SETS, Spec
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


# --- plan ----------------------------------------------------------------------

def make_plan(spec: Spec, paths: DV3Paths, force: bool = False) -> dict[str, Any]:
    data = csv_bytes(PLAN_COLUMNS, build_plan(spec))
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
    sim = simulate(row["family"], draw_from_row(row, prior), rng)
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


# --- merge ----------------------------------------------------------------------

def v0_counts(manifest: list[dict[str, Any]]) -> dict[str, Any]:
    """Check V0 on a prior-drawn set: mean n / nbar within max(3 s.e., 1%) of 1."""
    ratio = np.array([r["n"] / r["nbar"] for r in manifest])
    mean, se = float(ratio.mean()), float(ratio.std(ddof=1) / math.sqrt(len(ratio)))
    return {"mean_n_over_nbar": mean, "se": se, "pass": abs(mean - 1.0) <= max(3.0 * se, 0.01)}


def compat_record(row: dict[str, Any], points: np.ndarray, prior) -> dict[str, Any]:
    """The legacy cloud record (core/records.py) so featurize/diagrams run unchanged.
    `seed` is the per-(set, family) index: the legacy pipeline uses it as a cloud id."""
    params = {"nbar": row["nbar"], **{k: row[k] for k in prior.design_keys + prior.model_keys}}
    params.update({k: row[k] for k in ("tau_K", "delta_tilde") if row[k] is not None})
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
            "V0": v0_counts(manifest),
        }

    card = {
        "dataset": "DV3",
        "spec_path": str(spec.path),
        "spec_sha256": spec.sha256,
        "plan_sha256": meta["plan_sha256"],
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

    if set_ in PRIOR_SETS:
        d = draw(prior, case_rng(spec.root, set_, family, index, PARAMS), spec.nbar_low, spec.nbar_high)
        stored = draw_from_row(row, prior)
        report["theta_identical"] = (d.nbar == stored.nbar and d.design == stored.design and d.model == stored.model)

    points, mrow = simulate_row(spec, row, prior)
    stored_points, _ = load_points(paths.points(set_, family))
    report["n_stored"], report["n_regenerated"] = row["n"], mrow["n"]
    report["points_identical"] = (mrow["sha1"] == row["sha1"] == points_sha1(stored_points[pos])
                                  and np.array_equal(points, stored_points[pos]))
    return report
