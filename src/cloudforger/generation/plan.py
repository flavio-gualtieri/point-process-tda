# src/cloudforger/generation/plan.py
"""Enumerate every case into plan rows (steps 1-3 of the pipeline figure).

Training and A draw theta from each case's PARAMS stream; B and C take theta
from dv3_cells.csv (feasible cells only), `reps` replicates per cell. Then
everything that follows from theta by arithmetic alone is added:
delta-tilde (null tables), tau, and the LGCP grid.

The plan is a pure function of dv3.yaml, the null tables and the cells file:
same inputs, same rows, byte for byte, whatever `jobs` is. Simulation
(run_shard) reads theta back from the plan instead of redrawing it.
"""

from __future__ import annotations

from multiprocessing import get_context
from typing import Any

from .design import cell_shape, read_cells
from .nulls import R_GRID, delta_tilde, load_tables
from .prior import FamilyPrior, PriorDraw, build_priors, draw, fixed
from .seeding import DV, PARAMS, case_id, case_rng, index_B, index_C, key_str
from .spec import FIXED_SETS, PRIOR_SETS, Spec, load_spec


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


def plan_row(spec: Spec, set_: str, family: str, index: int, d: PriorDraw,
             cell_id: int | None = None, level_id: int | None = None, rep: int | None = None) -> dict[str, Any]:
    return {
        "case_id": case_id(set_, family, index),
        "dv": DV,
        "set": set_,
        "family": family,
        "index": index,
        "cell_id": cell_id,
        "level_id": level_id,
        "rep": rep,
        "spawn_key": key_str(set_, family, index),
        "split": split_for(spec, set_, index),
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


def _cell_rows(args: tuple[str, str, str]) -> list[dict[str, Any]]:
    """Every replicate of every feasible cell of one (fixed set, family)."""
    spec_path, set_, family = args
    spec = load_spec(spec_path)
    prior = build_priors(spec)[family]
    tabs = load_tables(spec.null_tables_path)
    rows = []
    for cell in read_cells(spec.cells_path):
        if (cell["set"], cell["family"]) != (set_, family) or not cell["feasible"]:
            continue
        d = complete(prior, fixed(prior, cell["nbar"], cell_shape(cell, prior)), tabs)
        for rep in range(cell["reps"]):
            index = (index_B(cell["cell_id"], rep) if set_ == "B"
                     else index_C(cell["cell_id"], cell["level_id"], rep))
            rows.append(plan_row(spec, set_, family, index, d, cell["cell_id"], cell["level_id"], rep))
    return rows


def build_plan(spec: Spec, jobs: int = 1, chunk: int = 500) -> list[dict[str, Any]]:
    tasks = [(_chunk_rows, (str(spec.path), set_, family, start, min(start + chunk, spec.set_size(set_))))
             for set_ in PRIOR_SETS if set_ in spec.sets
             for family in spec.families
             for start in range(0, spec.set_size(set_), chunk)]
    if spec.cells_path is not None:
        tasks += [(_cell_rows, (str(spec.path), set_, family)) for set_ in FIXED_SETS for family in spec.families]
    if jobs > 1:
        with get_context("spawn").Pool(jobs) as pool:
            parts = pool.starmap(_call, tasks, chunksize=1)
    else:
        parts = [_call(f, a) for f, a in tasks]
    rows = [row for part in parts for row in part]

    # Shards: consecutive cases of one (set, family), shard_size each.
    position: dict[tuple[str, str], int] = {}
    for row in rows:
        key = (row["set"], row["family"])
        row["shard"] = position.get(key, 0) // spec.shard_size
        position[key] = position.get(key, 0) + 1
    return rows


def _call(f, args):
    return f(args)
