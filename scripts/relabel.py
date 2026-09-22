#!/usr/bin/env python3
"""Recompute delta-tilde for every simulated theta under the CURRENT null tables.

    python scripts/relabel.py            # writes the `delta_tilde` column into every manifest
    python scripts/relabel.py --check    # recompute and report, write nothing

delta-tilde is a label, not a design coordinate: the bank is drawn on physical parameters and
knows nothing about departure. It is a deterministic function of (theta, nbar) and the null
tables, so refitting the tables is a relabelling, never a re-simulation.
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
from cloudforger.departure.tables import Tables                            # noqa: E402
from cloudforger.simulation.families import FAMILIES, Rules                # noqa: E402
from cloudforger.simulation.bank import DATA as BANK                # noqa: E402

CHUNK = 500


def delta_tilde_of(family: str, thetas: pd.DataFrame, tables: Tables) -> np.ndarray:
    """One delta-tilde per row of `thetas` (one row per theta), from the closed-form L at nbar."""
    fam = FAMILIES[family](Rules.load())
    if family == "poisson":
        return np.zeros(len(thetas))
    out = np.empty(len(thetas))
    for i in range(0, len(thetas), CHUNK):
        rows = thetas.iloc[i:i + CHUNK]
        curves = np.stack([
            l_minus_r_from_excess(fam.excess(RADII, {k: r[k] for k in fam.params}))
            for _, r in rows.iterrows()
        ])
        out[i:i + CHUNK] = tables.delta_tilde(curves, rows.nbar.to_numpy(float))
    return out


def relabel(family: str, tables: Tables, write: bool) -> dict:
    path = BANK / family / "manifest.csv"
    man = pd.read_csv(path)
    thetas = man.drop_duplicates("theta")
    d = pd.Series(delta_tilde_of(family, thetas, tables), index=thetas.theta.to_numpy())
    man["delta_tilde"] = d.reindex(man.theta.to_numpy()).to_numpy()
    if write:
        tmp = path.with_suffix(".csv.tmp")
        man.to_csv(tmp, index=False)
        tmp.replace(path)
    q = np.quantile(man.delta_tilde, [0.05, 0.5, 0.95])
    return {"family": family, "thetas": len(thetas), "min": man.delta_tilde.min(), "max": man.delta_tilde.max(),
            "q05": q[0], "q50": q[1], "q95": q[2]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="recompute and report without writing")
    args = ap.parse_args()
    tables = Tables()
    print(f"tables: min_pairs = {tables.min_pairs:g}\n")
    print(f"{'family':<9}{'thetas':>7}{'min':>8}{'max':>8}{'q05':>8}{'q50':>8}{'q95':>8}")
    for family in FAMILIES:
        r = relabel(family, tables, write=not args.check)
        print(f"{r['family']:<9}{r['thetas']:>7}{r['min']:>8.3f}{r['max']:>8.2f}"
              f"{r['q05']:>8.3f}{r['q50']:>8.3f}{r['q95']:>8.2f}")
    print("\nwrote `delta_tilde` into every manifest" if not args.check else "\n--check: nothing written")


if __name__ == "__main__":
    main()
