"""End-to-end score of fitted pipelines: does the fitted model reproduce the pattern?

Both scores are proper scoring rules, lower is better, and compare MODELS ON THE SAME PATTERN:
    kernel  primary. kernel.Component(x, R, kind, h, grid, centres), built ONCE per cloud and reused
            for every model (its kernel width is fixed from x, which is what makes the models' scores
            comparable), then .scores(sims)[tau_mult]. Strictly proper for the law of radius-R local
            configurations.
    dss     secondary. Dawid-Sebastiani on dss.statistics, restricted to the NO_PH groups so no
            persistent-homology statistic overlaps with the PH features being scored.

Per cloud, three kinds of model are simulated (simulate.simulate; never the bank's seeds):
    oracle   the true family at the true theta          -- the floor
    csr      poisson at nbar = n                        -- no structure
    fit      each pipeline's (family_hat, theta_hat). Identical fits share one simulation set; a
             poisson fit at nbar = n IS csr.

    regret = S(fit) - S(oracle)          >= 0 in expectation; 0 = as good as the truth
    gain   = S(csr) - S(oracle)          how much structure there is to find (same for every pipeline)
    skill  = 1 - sum(regret) / sum(gain) pooled over clouds, only where gain is resolved
             (mean > min_z standard errors): 1 = as good as the truth, 0 = no better than CSR
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import dss
from .kernel import Component
from .simulate import fit_key, rng_for, sampler_kwargs, simulate


def score_cloud(job: dict) -> list[dict]:
    """job: case_id, x (the scored pattern), oracle (family, kwargs), variants [{variant, family_hat,
    theta_hat}], ev {kernel, dss, seed}. One row per variant."""
    ev, cid, x = job["ev"], job["case_id"], job["x"]
    kc, dc = ev["kernel"], ev["dss"]
    rng = rng_for(ev["seed"], cid, "component")
    comp = Component(x, kc["R"], kc["kind"], kc["h"], kc["grid"], kc["centres"], rng)
    cols = dss.columns(getattr(dss, dc["groups"].upper()) if isinstance(dc["groups"], str) else tuple(dc["groups"]))
    t_x = dss.statistics(x)[cols] if dc["enabled"] else None
    k = max(kc["sims"], dc["sims"] if dc["enabled"] else 0)

    def score(key, family, kwargs):
        sims = simulate(family, kwargs, k, rng_for(ev["seed"], cid, key))
        out = {"kernel": comp.scores(sims[:kc["sims"]], rng, kc["pool"])[kc["tau_mult"]]}
        if dc["enabled"]:
            T = np.stack([dss.statistics(s) for s in sims[:dc["sims"]]])[:, cols]
            out["dss"] = dss.score(t_x, T)
        return out

    csr_key = fit_key("poisson", {"nbar": float(len(x))})
    S = {"oracle": score("oracle", *job["oracle"]), csr_key: score("csr", "poisson", {"nbar": float(len(x))})}
    rows = []
    for v in job["variants"]:
        # a poisson fit at nbar = len(x) IS csr and shares its simulations; a poisson fit estimated
        # from another replicate (nbar_hat = its n) is a model of its own
        key = fit_key(v["family_hat"], v["theta_hat"])
        if key not in S:
            # a fit the sampler cannot realise (no pattern with n in the bank's range, or an LGCP
            # covariance the embedding rejects) is a result about the estimator: recorded, not raised
            try:
                S[key] = score(key, v["family_hat"], sampler_kwargs(v["family_hat"], v["theta_hat"]))
            except (RuntimeError, ValueError) as e:
                S[key] = str(e)
        failed = isinstance(S[key], str)
        row = {"case_id": cid, "variant": v["variant"], "family_hat": v["family_hat"],
               "ended": "poisson" if v["family_hat"] == "poisson" else "family",
               "failed": failed, "fail_reason": S[key] if failed else ""}
        for s in S["oracle"]:
            fit = np.nan if failed else S[key][s]
            row |= {f"{s}_fit": fit, f"{s}_oracle": S["oracle"][s], f"{s}_csr": S[csr_key][s],
                    f"{s}_regret": fit - S["oracle"][s], f"{s}_gain": S[csr_key][s] - S["oracle"][s]}
        rows.append(row)
    return rows


def summarize(df: pd.DataFrame, scores: list[str], min_z: float) -> dict:
    """Per variant: regret, gain and skill overall, on structured clouds, by family, by family x
    stratum, by where the pipeline ended (poisson | family), and by whether the family was identified."""
    def se(v):
        return float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else float("inf")

    def cell(g):
        # a fit the sampler could not realise falls back to CSR: its regret is the cloud's gain, so every
        # pipeline stays scored on the same clouds; skill_ok is the skill over realisable fits alone
        failed = g["failed"].to_numpy(bool) if "failed" in g else np.zeros(len(g), bool)
        out = {"n": int(len(g)), "failed": int(failed.sum())}
        for s in scores:
            gain = g[f"{s}_gain"]
            reg = g[f"{s}_regret"].where(~failed, gain)
            resolved = gain.mean() > min_z * se(gain)
            ok = ~failed
            out[s] = {"regret": float(reg.mean()), "regret_se": se(reg), "gain": float(gain.mean()),
                      "gain_se": se(gain), "skill": float(1 - reg.sum() / gain.sum()) if resolved else None,
                      "skill_ok": float(1 - reg[ok].sum() / gain[ok].sum()) if resolved and ok.any() else None,
                      "fit_beats_csr": float((g[f"{s}_fit"][ok] < g[f"{s}_csr"][ok]).mean()) if ok.any() else None}
        return out

    out = {}
    for variant, g in df.groupby("variant"):
        structured = g[g.family != "poisson"]
        out[variant] = {
            "overall": cell(g),
            "structured": cell(structured),
            "by_family": {f: cell(h) for f, h in g.groupby("family")},
            "by_family_stratum": {f"{f} | {s}": cell(h) for (f, s), h in g.groupby(["family", "stratum"])},
            "structured_by_end": {e: cell(h) for e, h in structured.groupby("ended")},
            "identified": {str(k): cell(h) for k, h in structured.groupby(structured.family_hat == structured.family)},
        }
    return out
