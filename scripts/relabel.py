#!/usr/bin/env python3
"""Recompute delta-tilde for every simulated theta under the CURRENT null tables.

    python scripts/relabel.py            # writes one column per reduction into every manifest
    python scripts/relabel.py --check    # recompute and report, write nothing

One column per calibrated reduction (departure.tables.REDUCTIONS): `delta_tilde` for the default
sup and `delta_tilde_<name>` for the rest. They are alternative coordinates on the same bank, each
calibrated so that 1 is its own 5% rejection boundary, and they are NOT interchangeable as numbers
-- pick one with regimes.py --delta-column and report the others as a sensitivity check.

Poisson is not special-cased: its closed-form curve is exactly zero, and under the extremum
reductions a zero curve scores the CSR noise floor (a negative number), not 0. Only the `sup`
column is 0 there.

delta-tilde is a label, not a design coordinate: the bank is drawn on physical parameters and
knows nothing about departure. It is a deterministic function of (theta, nbar) and the null
tables, so refitting the tables, or adding a reduction, is a relabelling and never a
re-simulation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cloudforger.classical.lfunction import RADII, l_minus_r_from_excess   # noqa: E402
from cloudforger.departure.tables import DEFAULT, Tables                   # noqa: E402
from cloudforger.simulation.families import BANK_FAMILIES, FAMILIES, Rules                # noqa: E402
from cloudforger.simulation.bank import DATA as BANK                # noqa: E402

CHUNK = 500


def column(name: str) -> str:
    """The default reduction keeps the bare name, so nothing downstream has to know about the rest."""
    return "delta_tilde" if name == DEFAULT else f"delta_tilde_{name}"


def delta_tilde_of(family: str, thetas: pd.DataFrame, tables: Tables) -> dict[str, np.ndarray]:
    """One delta-tilde per row of `thetas` (one row per theta) per reduction, from the closed-form
    L at nbar. The curves are the expensive part and are shared across the reductions."""
    names = list(tables.coef_c)
    fam = FAMILIES[family](Rules.load())
    out = {n: np.empty(len(thetas)) for n in names}
    for i in range(0, len(thetas), CHUNK):
        rows = thetas.iloc[i:i + CHUNK]
        curves = np.stack([
            l_minus_r_from_excess(fam.excess(RADII, {k: r[k] for k in fam.params}))
            for _, r in rows.iterrows()
        ])
        nbar = rows.nbar.to_numpy(float)
        for n in names:
            out[n][i:i + CHUNK] = tables.delta_tilde(curves, nbar, n)
    return out


def relabel(family: str, tables: Tables, write: bool) -> list[dict]:
    path = BANK / family / "manifest.csv"
    man = pd.read_csv(path)
    thetas = man.drop_duplicates("theta")
    labels = delta_tilde_of(family, thetas, tables)
    fresh = {column(name) for name in labels}
    man = man.drop(columns=[c for c in man.columns                    # reductions that no longer
                            if c.startswith("delta_tilde") and c not in fresh])   # exist
    for name, values in labels.items():
        d = pd.Series(values, index=thetas.theta.to_numpy())
        man[column(name)] = d.reindex(man.theta.to_numpy()).to_numpy()
    if write:
        tmp = path.with_suffix(".csv.tmp")
        man.to_csv(tmp, index=False)
        tmp.replace(path)
    out = []
    for name, values in labels.items():
        q = np.quantile(values, [0.05, 0.5, 0.95])
        out.append({"family": family, "reduction": name, "thetas": len(thetas), "min": values.min(),
                    "max": values.max(), "q05": q[0], "q50": q[1], "q95": q[2],
                    "below_1": float((values < 1).mean())})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="recompute and report without writing")
    args = ap.parse_args()
    tables = Tables()
    print(f"tables: min_pairs = {tables.min_pairs:g}, reductions = {', '.join(tables.coef_c)}\n")
    print(f"{'family':<9}{'reduction':<11}{'thetas':>7}{'min':>8}{'max':>8}{'q05':>8}{'q50':>8}{'q95':>8}{'<1':>8}")
    columns = []
    for family in BANK_FAMILIES:
        for r in relabel(family, tables, write=not args.check):
            columns.append(column(r["reduction"]))
            print(f"{r['family']:<9}{r['reduction']:<11}{r['thetas']:>7}{r['min']:>8.3f}{r['max']:>8.2f}"
                  f"{r['q05']:>8.3f}{r['q50']:>8.3f}{r['q95']:>8.2f}{r['below_1']:>8.1%}")
    cols = ", ".join(f"`{c}`" for c in dict.fromkeys(columns))
    print(f"\nwrote {cols} into every manifest" if not args.check else "\n--check: nothing written")


if __name__ == "__main__":
    main()
