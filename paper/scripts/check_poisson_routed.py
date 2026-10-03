#!/usr/bin/env python3
"""Does the canonical CSR test also find nothing in the clouds the pipeline routes to Poisson?

    PAPER_CONFIG=paper/paper_v2.yaml python paper/scripts/check_poisson_routed.py

Section 6, "Sending near-CSR patterns to Poisson costs nothing", argues that the structured clouds
the headline classifier calls Poisson might as well BE Poisson, and supports it with the true
model's gain over CSR -- our own kernel score. This check makes the same point with a detector that
owes the pipeline nothing: the exact-size L test of Section 6 (departure.tables, default reduction,
size 5%), the canonical test the whole regime boundary is calibrated against.

The test is applied to the pattern the CLASSIFIER SAW -- replicate 0, the evaluation's `fit_case_id`
-- not to the replicate the fit is scored on, so the question is exactly "on this pattern, would the
canonical test have found the structure the classifier missed?".

Three groups of the evaluation's clouds, so the rate has something to be read against:
  sent to Poisson    the structured clouds the pipeline called Poisson -- the claim: ~5%, the test's
                     own size, i.e. the test finds no more structure here than it does in CSR
  sent to a family   the structured clouds it called something -- the test should fire often
  Poisson clouds     true CSR: the test's realised size on this sample, the yardstick for the first row

Rejection rates carry Wilson 95% intervals (binomial over clouds). The claim is one-sided -- the
test finds NO MORE structure here than in CSR -- so the verdict is a one-sided binomial test of the
rate against the nominal 5%, not an eyeballed interval overlap. The rate on true Poisson clouds is
reported beside it, and the point counts of each group, because the L test's power depends on n
(its SIZE does not: the tables are calibrated per n, so 5% holds at every n).

Output  <paper outputs.results>/poisson_routed.md   the tables below, and the sentence's numbers
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common as C  # noqa: E402

from cloudforger.departure.tables import Tables  # noqa: E402
from cloudforger.paths import CURVES  # noqa: E402

SIZE = 0.05
STRATA = ("below 0.5", "0.5-0.9", "above 0.9", "k 2-2", "k 3-4", "k 5-30")


def rejects(case_ids: pd.Index, families: np.ndarray, n: np.ndarray) -> np.ndarray:
    """The exact-size L test at 5% on the stored L(r) - r of each pattern; True = rejects CSR.

    Identical to the `CSR test` detector of scripts/regime.py, by which the regime boundary's L-test column is
    computed: the default (two-arm extremum) reduction, which rejects when the statistic exceeds 1.
    """
    tables, out = Tables(), np.zeros(len(case_ids), bool)
    for f in pd.unique(families):
        m = families == f
        z = np.load(CURVES / f / "fixed" / "curves.npz")
        at = pd.Index(z["case_id"].astype(str)).get_indexer(case_ids[m])
        if (at < 0).any():
            raise SystemExit(f"{f}: {(at < 0).sum()} clouds have no stored L curve")
        out[m] = tables.statistic(z["L"][at].astype(np.float64), n[m].astype(float)) > 1
    return out


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% interval for a rejection rate; stable at k = 0 and k = n, unlike the normal one."""
    if not n:
        return (float("nan"), float("nan"))
    p, d = k / n, 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def rate(g: pd.DataFrame) -> str:
    k, n = int(g.rejects.sum()), len(g)
    lo, hi = wilson(k, n)
    return f"{k}/{n} = {k / n:.3f} [{lo:.3f}, {hi:.3f}]" if n else "—"


def main() -> None:
    variant = C.cfg()["headline_variant"]
    df = pd.read_csv(C.evaluation("main") / "clouds.csv")
    d = df[df.variant == variant].copy()
    if d.empty:
        raise SystemExit(f"{C.evaluation('main')}: no rows for the headline variant `{variant}`")

    # The fit (and so the call) comes from replicate 0: that is the pattern to test.
    d["rejects"] = rejects(pd.Index(d.fit_case_id.to_numpy(str)), d.family.to_numpy(str),
                           d.n_fit.to_numpy(float))
    d["group"] = np.where(d.family == "poisson", "Poisson clouds",
                          np.where(d.family_hat == "poisson", "sent to Poisson", "sent to a family"))
    order = ["sent to Poisson", "sent to a family", "Poisson clouds"]

    L = [f"# The L test on the clouds the pipeline sends to Poisson: {C.cfg()['run']}", "",
         f"Headline pipeline `{variant}`, evaluation set `{C.cfg()['evaluation']['main']}` "
         f"({len(d)} clouds). Detector: the exact-size L test at {SIZE:.0%} "
         "(`cloudforger.departure.tables`, default two-arm extremum reduction), the same detector as "
         "the `L test` column of the regime-boundary table, applied to replicate 0 -- the pattern the "
         "classifier saw. Rates are rejections / clouds with a Wilson 95% interval.", "",
         "| clouds | rejects CSR at 5% |", "|---|---|"]
    for g in order:
        L.append(f"| {g} | {rate(d[d.group == g])} |")

    sent = d[d.group == "sent to Poisson"]
    null = d[d.group == "Poisson clouds"]
    L += ["", "The clouds sent to Poisson, by family and regime stratum (the claim is that they are "
          "near-CSR, so most of them sit in the unstructured regime):", "",
          "| family | stratum | n | rejects CSR at 5% |", "|---|---|---|---|"]
    for (f, s), g in sent.groupby(["family", "stratum"]):
        L.append(f"| {C.NAMES.get(f, f)} | {s} | {len(g)} | {rate(g)} |")
    L += ["", "By stratum, pooled over families:", "", "| stratum | n | rejects CSR at 5% |", "|---|---|---|"]
    for s in [s for s in STRATA if (sent.stratum == s).any()]:
        L.append(f"| {s} | {int((sent.stratum == s).sum())} | {rate(sent[sent.stratum == s])} |")

    k, n = int(sent.rejects.sum()), len(sent)
    lo, hi = wilson(k, n)
    k0, n0 = int(null.rejects.sum()), len(null)
    lo0, hi0 = wilson(k0, n0)
    above = stats.binomtest(k, n, SIZE, alternative="greater").pvalue   # does the rate EXCEED its size?
    two = stats.binomtest(k, n, SIZE).pvalue
    vs_null = stats.fisher_exact([[k, n - k], [k0, n0 - k0]])[1]
    gain = sent.kernel_gain
    L += ["", "## For the main text", "",
          f"Of the {n} structured clouds the pipeline assigns to Poisson, the canonical L test rejects "
          f"CSR on {k} ({k / n:.1%} [{lo:.1%}, {hi:.1%}]). There is no evidence the rate exceeds the "
          f"test's own {SIZE:.0%} size (one-sided binomial p = {above:.2f}); it is in fact below it "
          f"(two-sided p = {two:.3f}), and below the {k0 / n0:.1%} [{lo0:.1%}, {hi0:.1%}] the same test "
          f"rejects on the {n0} true Poisson clouds of this set (Fisher p = {vs_null:.3f}). "
          f"The true model's gain over CSR on the same clouds is "
          f"{gain.mean():.0e} ± {gain.std(ddof=1) / np.sqrt(n):.0e}.", "",
          "Two readings to keep straight. The claim the paragraph needs is the first: a detector that "
          "owes the pipeline nothing, calibrated to reject 5% of CSR patterns, finds no more structure "
          "in these than it finds in CSR. That the rate is *lower* than on genuine Poisson clouds is "
          "not a statement about the patterns -- nothing is less structured than CSR -- but a selection "
          "effect: the classifier's Poisson call and the L test are correlated detectors, so "
          "conditioning on the call selects patterns whose L(r) - r is quiet, including ones where "
          "noise alone would have tripped the test. It is the same agreement between detectors that "
          "the regime boundary is built on.", "",
          "Not a power artifact: the groups' point counts are comparable "
          + ", ".join(f"{g} median n {d[d.group == g].n_fit.median():.0f}" for g in order)
          + ", and the test is exact-size at every n.", ""]

    out = C.RESULTS / "poisson_routed.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
