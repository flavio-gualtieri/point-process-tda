# src/cloudforger/generation/plan.py
"""Enumerate every prior-drawn case (training, A) into plan rows, drawing
theta from each case's PARAMS stream (steps 1-3 of the pipeline figure).

The plan is a pure function of dv3.yaml: same spec, same rows, byte for byte.
Simulation (run_shard) reads theta back from the plan instead of redrawing it.
"""

from __future__ import annotations

from typing import Any

from .prior import FamilyPrior, PriorDraw, build_priors, draw
from .seeding import DV, PARAMS, case_id, case_rng, key_str
from .spec import PRIOR_SETS, Spec


def split_for(spec: Spec, set_: str, index: int) -> str:
    # Draws are i.i.d., so the last n_val indices are as random a validation
    # set as any permutation would give -- and the rule needs no extra stream.
    if set_ == "train":
        return "val" if index >= spec.set_size(set_) - spec.n_val(set_) else "train"
    return "test"


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
        **d.regime,
        "prior_tries": d.tries,
    }


def draw_from_row(row: dict[str, Any], prior: FamilyPrior) -> PriorDraw:
    return PriorDraw(
        nbar=row["nbar"],
        design={k: row[k] for k in prior.design_keys},
        model={k: row[k] for k in prior.model_keys},
        regime={"tau_K": row["tau_K"], "delta_tilde": row["delta_tilde"]},
        tries=row["prior_tries"],
    )


def build_plan(spec: Spec) -> list[dict[str, Any]]:
    priors = build_priors(spec)
    rows = []
    for set_ in PRIOR_SETS:
        if set_ not in spec.sets:
            continue
        for family, prior in priors.items():
            for index in range(spec.set_size(set_)):
                rng = case_rng(spec.root, set_, family, index, PARAMS)
                rows.append(plan_row(spec, set_, family, index, draw(prior, rng, spec.nbar_low, spec.nbar_high)))
    return rows
