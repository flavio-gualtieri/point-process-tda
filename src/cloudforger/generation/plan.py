# src/cloudforger/generation/plan.py
"""Enumerate every prior-drawn case (training, A) into plan rows, drawing
theta from each case's PARAMS stream and computing everything that follows
from theta by arithmetic alone: delta-tilde (from the null tables), tau, and
the LGCP grid (steps 1-3 of the pipeline figure).

The plan is a pure function of dv3.yaml and the null tables: same inputs,
same rows, byte for byte, whatever `jobs` is. Simulation (run_shard) reads
theta back from the plan instead of redrawing it.
"""

from __future__ import annotations

from multiprocessing import get_context
from typing import Any

from .nulls import R_GRID, delta_tilde, load_tables
from .prior import FamilyPrior, PriorDraw, build_priors, draw
from .seeding import DV, PARAMS, case_id, case_rng, key_str
from .spec import PRIOR_SETS, Spec, load_spec


def split_for(spec: Spec, set_: str, index: int) -> str:
    # Draws are i.i.d., so the last n_val indices are as random a validation
    # set as any permutation would give -- and the rule needs no extra stream.
    if set_ == "train":
        return "val" if index >= spec.set_size(set_) - spec.n_val(set_) else "train"
    return "test"


def complete(prior: FamilyPrior, d: PriorDraw, tabs: dict[str, Any]) -> PriorDraw:
    """Add the table-dependent quantities: delta_tilde and per-case numerics."""
    regime = {**d.regime, "delta_tilde": delta_tilde(prior.excess(R_GRID, d.nbar, d.design), d.nbar, tabs)}
    return PriorDraw(d.nbar, d.design, d.model, regime, d.tries, prior.numerics(d.nbar, d.design, tabs))


def plan_row(spec: Spec, set_: str, family: str, index: int, d: PriorDraw) -> dict[str, Any]:
    return {
        "case_id": case_id(set_, family, index),
        "dv": DV,
        "set": set_,
        "family": family,
        "index": index,
        "spawn_key": key_str(set_, family, index),
        "split": split_for(spec, set_, index),
        "shard": index // spec.shard_size,
        "nbar": d.nbar,
        **d.design,
        **d.model,
        **d.numerics,
        **d.regime,
        "prior_tries": d.tries,
    }


def draw_from_row(row: dict[str, Any], prior: FamilyPrior) -> PriorDraw:
    return PriorDraw(
        nbar=row["nbar"],
        design={k: row[k] for k in prior.design_keys},
        model={k: row[k] for k in prior.model_keys},
        regime={k: row[k] for k in ("delta_tilde", "tau_K", "tau_K2")},
        tries=row["prior_tries"],
        numerics={k: row[k] for k in ("grid_M",) if row.get(k) is not None},
    )


def _chunk_rows(args: tuple[str, str, str, int, int]) -> list[dict[str, Any]]:
    spec_path, set_, family, start, stop = args
    spec = load_spec(spec_path)
    prior = build_priors(spec)[family]
    tabs = load_tables(spec.null_tables_path)
    rows = []
    for index in range(start, stop):
        rng = case_rng(spec.root, set_, family, index, PARAMS)
        d = complete(prior, draw(prior, rng, spec.nbar_low, spec.nbar_high), tabs)
        rows.append(plan_row(spec, set_, family, index, d))
    return rows


def build_plan(spec: Spec, jobs: int = 1, chunk: int = 500) -> list[dict[str, Any]]:
    tasks = [(str(spec.path), set_, family, start, min(start + chunk, spec.set_size(set_)))
             for set_ in PRIOR_SETS if set_ in spec.sets
             for family in spec.families
             for start in range(0, spec.set_size(set_), chunk)]
    if jobs > 1:
        with get_context("spawn").Pool(jobs) as pool:
            parts = pool.map(_chunk_rows, tasks, chunksize=1)
    else:
        parts = [_chunk_rows(t) for t in tasks]
    return [row for part in parts for row in part]
