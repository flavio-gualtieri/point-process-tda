# src/cloudforger/generation/design.py
"""Fixed-theta test sets (generation.tex, "Training pool and test sets").

B (regime grid): per family, a factorial over nbar x a scale level x a target
delta-tilde; the amplitude parameter is solved from the target by a 1-D
bisection (delta-tilde is monotone in the amplitude). Matern II is laid out in
tau directly. A cell is kept only if theta meets the constraints and lies in
the central 80% of every log-uniform marginal (the interior rule).

C (ladders to CSR): per family, one fixed shape at two nbar; 16 amplitude
levels log-spaced over the amplitude's interior range, capped by the
constraints and clipped to delta-tilde in [0.1, 10] (Matern II: tau over its
interior, unclipped). Plus exact Poisson anchors at three nbar.

Cell ids enumerate the full factorial, feasible or not, so dropping a cell
never renumbers (and so never reseeds) the others. The result is written to
dv3_cells.csv, which is reviewed (gate D1) and committed.
"""

from __future__ import annotations

import csv
import io
import math
from typing import Any

import numpy as np

from .nulls import R_GRID, delta_tilde
from .prior import FamilyPrior, build_priors
from .spec import Spec
from .store import DESIGN

SHAPE_KEYS = [k for k in DESIGN if k != "nbar"]
CELL_COLUMNS = ["set", "family", "cell_id", "level_id", "nbar", *SHAPE_KEYS,
                "target_delta", "delta_tilde", "feasible", "why", "reps"]
_INT = {"cell_id", "level_id", "reps"}
_STR = {"set", "family", "why"}


def interior(prior: FamilyPrior, key: str, frac: float) -> tuple[float, float]:
    lo, hi = prior.ranges[key]
    return lo * (hi / lo) ** frac, lo * (hi / lo) ** (1.0 - frac)


def check(prior: FamilyPrior, nbar: float, shape: dict[str, float], frac: float) -> str:
    """'' if theta is feasible for a fixed-theta set, else the reason."""
    if not prior.accepts(nbar, shape):
        return "constraint"
    for key in prior.design_keys:
        lo, hi = interior(prior, key, frac)
        if not lo * (1 - 1e-12) <= shape[key] <= hi * (1 + 1e-12):
            return f"{key} outside interior"
    return ""


def delta_of(prior: FamilyPrior, nbar: float, shape: dict[str, float], tabs: dict[str, Any]) -> float:
    return delta_tilde(prior.excess(R_GRID, nbar, shape), nbar, tabs)


def solve_amplitude(prior, nbar, shape, amp, target, tabs, lo, hi, iters=60) -> float | None:
    """Amplitude in [lo, hi] with delta-tilde = target (bisection in log), or None if unreachable."""
    f = lambda a: delta_of(prior, nbar, {**shape, amp: a}, tabs) - target
    if f(lo) > 0 or f(hi) < 0:
        return None
    a, b = math.log(lo), math.log(hi)
    for _ in range(iters):
        m = 0.5 * (a + b)
        a, b = (m, b) if f(math.exp(m)) < 0 else (a, m)
    return math.exp(0.5 * (a + b))


def _cap(prior, nbar, shape, amp, lo, hi, iters=60) -> float:
    """Largest amplitude in [lo, hi] the constraints accept (they are monotone in it)."""
    ok = lambda a: prior.accepts(nbar, {**shape, amp: a})
    if ok(hi):
        return hi
    a, b = math.log(lo), math.log(hi)
    for _ in range(iters):
        m = 0.5 * (a + b)
        a, b = (m, b) if ok(math.exp(m)) else (a, m)
    return math.exp(a)


def _row(set_, family, cell_id, level_id, nbar, shape, target, delta, why, reps) -> dict[str, Any]:
    return {"set": set_, "family": family, "cell_id": cell_id, "level_id": level_id, "nbar": float(nbar),
            **shape, "target_delta": target, "delta_tilde": delta, "feasible": int(not why), "why": why, "reps": reps}


def b_cells(spec: Spec, priors: dict[str, FamilyPrior], tabs: dict[str, Any]) -> list[dict[str, Any]]:
    cfg, frac = spec.test_sets["B"], spec.test_sets["interior"]
    amps = {f: lad["amplitude"] for f, lad in spec.test_sets["C"]["ladders"].items()}
    rows = []
    for family in spec.families:
        if family not in cfg["shapes"] and not (family == "matern2" and "matern2_tau" in cfg):
            continue
        prior = priors[family]
        nbars = cfg["nbar"].get(family, cfg["nbar"]["default"])
        if family == "matern2":
            combos = [(nb, {"tau": float(t)}, None) for nb in nbars for t in cfg["matern2_tau"]]
        else:
            combos = [(nb, dict(sh), float(dt)) for nb in nbars for sh in cfg["shapes"][family] for dt in cfg["deltas"]]
        for cell_id, (nb, shape, target) in enumerate(combos):
            if target is not None:
                lo, hi = prior.ranges[amps[family]]
                a = solve_amplitude(prior, nb, shape, amps[family], target, tabs, lo / 100, hi * 100)
                if a is None:
                    rows.append(_row("B", family, cell_id, None, nb, shape, target, None, "unreachable", cfg["reps"]))
                    continue
                shape = {**shape, amps[family]: a}
            why = check(prior, nb, shape, frac)
            rows.append(_row("B", family, cell_id, None, nb, shape, target, delta_of(prior, nb, shape, tabs),
                             why, cfg["reps"]))
    return rows


def c_ladders(spec: Spec, priors: dict[str, FamilyPrior], tabs: dict[str, Any]) -> list[dict[str, Any]]:
    cfg, frac = spec.test_sets["C"], spec.test_sets["interior"]
    dmin, dmax = (float(x) for x in cfg["delta_range"])
    rows = []
    if "poisson" in spec.families:
        for ladder, nb in enumerate(cfg["poisson"]["nbar"]):
            rows.append(_row("C", "poisson", ladder, 0, nb, {}, None, 0.0, "", cfg["poisson"]["reps"]))
    for family, lad in cfg["ladders"].items():
        if family not in spec.families:
            continue
        prior, amp, shape = priors[family], lad["amplitude"], dict(lad["shape"])
        for ladder, nb in enumerate(cfg["nbar"].get(family, cfg["nbar"]["default"])):
            lo, hi = interior(prior, amp, frac)
            hi = _cap(prior, nb, shape, amp, lo, hi)
            if family != "matern2":   # Matern II: levels in tau, not clipped in delta
                if delta_of(prior, nb, {**shape, amp: lo}, tabs) < dmin:
                    lo = solve_amplitude(prior, nb, shape, amp, dmin, tabs, lo, hi)
                if delta_of(prior, nb, {**shape, amp: hi}, tabs) > dmax:
                    hi = solve_amplitude(prior, nb, shape, amp, dmax, tabs, lo, hi)
            for level, a in enumerate(np.geomspace(lo, hi, int(cfg["levels"]))):
                full = {**shape, amp: float(a)}
                rows.append(_row("C", family, ladder, level, nb, full, None, delta_of(prior, nb, full, tabs),
                                 check(prior, nb, full, frac), cfg["reps"]))
    return rows


def build_cells(spec: Spec, tabs: dict[str, Any]) -> list[dict[str, Any]]:
    priors = build_priors(spec)
    return b_cells(spec, priors, tabs) + c_ladders(spec, priors, tabs)


def cells_bytes(rows: list[dict[str, Any]]) -> bytes:
    from .store import fmt
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(CELL_COLUMNS)
    for r in rows:
        w.writerow([fmt(r.get(c)) for c in CELL_COLUMNS])
    return buf.getvalue().encode()


def read_cells(path) -> list[dict[str, Any]]:
    def parse(k, v):
        if v == "":
            return None
        return v if k in _STR else int(v) if k in _INT or k == "feasible" else float(v)
    with open(path, newline="") as f:
        return [{k: parse(k, v) for k, v in row.items()} for row in csv.DictReader(f)]


def cell_shape(row: dict[str, Any], prior: FamilyPrior) -> dict[str, float]:
    return {k: row[k] for k in prior.design_keys}
