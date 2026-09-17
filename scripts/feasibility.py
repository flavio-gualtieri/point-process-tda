"""Largest delta-tilde each family can reach under the validity rules, across nbar.

For every valid shape on a log grid, the amplitude is pushed to its limit
(cv_max, or the Matern fill bound) and delta-tilde is evaluated there.
Reports the share of valid shapes that can reach each target delta.
"""

from __future__ import annotations

import argparse
import itertools
import math

import numpy as np
import yaml

from cloudforger.departure.tables import Tables
from cloudforger.simulation.families import CONFIG, FAMILIES, Rules, amp_max, delta

TARGETS = (1, 3, 10, 30)


def shapes(fam, nbar, per_axis):
    box = fam.shape_box(nbar)
    axes = [np.geomspace(lo, hi, per_axis) for lo, hi in box.values()]
    for values in itertools.product(*axes) if axes else [()]:
        shape = dict(zip(box, map(float, values)))
        if fam.valid_shape(nbar, shape):
            yield shape


def monotone(fam, tables, nbar, shape, points=25):
    lo, hi = fam.amp_bounds(nbar, shape)[0], amp_max(fam, nbar, shape)
    d = [delta(fam, tables, nbar, shape, a) for a in np.geomspace(max(lo, hi * 1e-4), hi, points)]
    return bool(np.all(np.diff(d) >= -1e-9))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nbar", type=int, default=6)
    p.add_argument("--per-axis", type=int, default=9)
    args = p.parse_args()

    tables = Tables()
    rules = Rules.load(tables)
    cfg = yaml.safe_load(CONFIG.read_text())["nbar"]
    print(f"rules: {rules}\n")
    print(f"{'family':8s} {'nbar':>5s} {'shapes':>6s} {'max delta':>9s}  " + "  ".join(f"P(reach {t:>2d})" for t in TARGETS) + "  monotone")
    for name in ("thomas", "nested", "lgcp", "matern2"):
        fam = FAMILIES[name](rules)
        for nbar in np.geomspace(cfg["low"], cfg["high"], args.nbar):
            grid = list(shapes(fam, nbar, args.per_axis))
            dmax = np.array([delta(fam, tables, nbar, s, amp_max(fam, nbar, s)) for s in grid])
            mono = all(monotone(fam, tables, nbar, s) for s in grid[:: max(1, len(grid) // 5)])
            reach = "  ".join(f"{np.mean(dmax >= t):11.2f}" for t in TARGETS)
            print(f"{name:8s} {nbar:5.0f} {len(grid):6d} {dmax.max():9.1f}  {reach}  {mono}", flush=True)
        print()


if __name__ == "__main__":
    main()
