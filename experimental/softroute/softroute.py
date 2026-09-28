#!/usr/bin/env python3
"""Soft routing: does simulating from the classifier's mixture beat simulating from its argmax?

    python experimental/softroute/softroute.py [--config ...] [--workers N] [--limit N]

The paper's pipeline routes each cloud to argmax_f P(f | x) and simulates from that family at its
estimate. Soft routing simulates from sum_f w_f P_f(theta_hat_f) instead, with w = P(f | x) or a
variant of it (config). Everything else is the paper's: its classifier and estimators (their stored
predictions, frozen in the config), its clouds (fit on replicate 0, scored on replicate 1), and its
kernel score with the same settings. Nothing here writes outside experimental/softroute/results/;
the paper run is only read (config `paper_run`).

EXACT MIXTURE SCORE. The kernel score is quadratic in the model's mean embedding, so for a mixture

    S(sum_f w_f P_f, x) = 1/2 sum_{f,g} w_f w_g K_fg  -  sum_f w_f c_f
    K_fg = E k(Phi_f, Phi_g')  (configurations of simulations of f and of g; f = g: different simulations)
    c_f  = mean_i E k(phi_i, Phi_f)

With w one-hot this is scores.kernel.Component.scores exactly. So every family's model is simulated
once per cloud (config kernel.sims patterns, as the paper), and every weighting is scored from the
same simulations: differences between weightings carry no simulation noise of their own. A family
whose estimate the sampler cannot realise is dropped from the mixture and the rest renormalised
(recorded as `lost`).

Output  experimental/softroute/results/<name>/{clouds.csv, mixture_terms.npz, report.json, summary.md}
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.spatial.distance import cdist

from cloudforger.paths import ROOT
from cloudforger.pipeline.core import TARGETS, log, rows as bank_rows, write_json
from cloudforger.scores.kernel import Component
from cloudforger.scores.simulate import observed, rng_for, sampler_kwargs, simulate, true_model

HERE = Path(__file__).resolve().parent
FAMILIES = ["poisson", "thomas", "nested", "lgcp", "matern2", "ring", "matern1", "cell"]


# ------------------------------------------------------------------------- the paper run (read only)

def paper(cfg: dict) -> Path:
    """The pipeline run whose stored predictions and evaluation clouds are used (config paper_run)."""
    return ROOT / cfg["paper_run"]


def paper_classifier(cfg: dict, model: str) -> pd.DataFrame:
    """Class posteriors (val + test rows), one column per family."""
    z = np.load(paper(cfg) / "classify" / model / "predictions.npz")
    return pd.DataFrame(z["posterior"], columns=[str(c) for c in z["classes"]],
                        index=pd.Index(z["case_id"], name="case_id"))


def paper_estimator(cfg: dict, family: str, model: str) -> pd.DataFrame:
    """theta_hat from `model` for `family`, on every val + test row (any family)."""
    z = np.load(paper(cfg) / "estimate" / family / model / "predictions.npz")
    return pd.DataFrame(z["theta_hat"], columns=[str(c) for c in z["targets"]],
                        index=pd.Index(z["case_id"], name="case_id"))


# ------------------------------------------------------------------------------------- clouds

def clouds(cfg: dict) -> pd.DataFrame:
    """fit_case_id-indexed: scored_case_id, family, stratum, sets (the paper evaluations it is in)."""
    parts = []
    for name, rel in cfg["clouds"].items():
        d = pd.read_csv(paper(cfg) / rel / "clouds.csv")
        d = d.drop_duplicates("fit_case_id")[["fit_case_id", "scored_case_id", "family", "stratum"]]
        parts.append(d.assign(sets=name))
    d = pd.concat(parts)
    sets = d.groupby("fit_case_id").sets.agg(lambda s: ",".join(sorted(set(s))))
    return d.drop_duplicates("fit_case_id").set_index("fit_case_id").drop(columns="sets").join(sets)


def weightings(p: np.ndarray, true_idx: int, temps: list[float]) -> dict[str, np.ndarray]:
    one = lambda i: np.eye(len(p))[i]
    top2 = np.where(p >= np.sort(p)[-2], p, 0.0)
    out = {"hard": one(int(np.argmax(p))), "soft": p / p.sum(), "top2": top2 / top2.sum(), "true": one(true_idx),
           "uniform": np.full(len(p), 1.0 / len(p))}
    for t in temps:
        q = p ** (1.0 / t)
        out[f"temp_{t:g}"] = q / q.sum()
    return out


# -------------------------------------------------------------------------------- per-cloud work

def score_cloud(job: dict) -> list[dict]:
    kc, seed, sid, x = job["kernel"], job["seed"], job["scored_case_id"], job["x"]
    rng = rng_for(seed, sid, "component")
    comp = Component(x, kc["R"], kc["kind"], kc["h"], kc["grid"], kc["centres"], rng)
    t2 = 2 * (float(kc["tau_mult"]) * comp.tau0) ** 2
    per = max(1, kc["pool"] // kc["sims"])

    E, W, failed = {}, {}, {}
    for key, (family, kw) in job["models"].items():
        try:
            if key.startswith("fam:"):
                kw = sampler_kwargs(family, kw)
            sims = simulate(family, kw, kc["sims"], rng_for(seed, sid, key))
        except (RuntimeError, ValueError) as e:
            failed[key] = str(e)
            continue
        erng = rng_for(seed, sid, "embed:" + key)
        embs = [comp.embed(s, erng, per) for s in sims]
        E[key] = np.concatenate(embs)
        W[key] = np.concatenate([np.full(len(e), i) for i, e in enumerate(embs)])

    def k_mean(a: str, b: str) -> float:
        k = np.exp(-cdist(E[a], E[b], "sqeuclidean") / t2)
        return float(k[W[a][:, None] != W[b][None, :]].mean()) if a == b else float(k.mean())

    def c_mean(a: str) -> float:
        return float(np.exp(-cdist(comp.ex, E[a], "sqeuclidean") / t2).mean())

    base = {"scored_case_id": sid, **job["meta"], "failed": ",".join(failed)}
    if any(k not in E or len(E[k]) == 0 for k in ("oracle", "csr")) or len(comp.ex) == 0:
        return [base | {"variant": "none", "regret": np.nan, "gain": np.nan}], None
    s_oracle = 0.5 * k_mean("oracle", "oracle") - c_mean("oracle")
    s_csr = 0.5 * k_mean("csr", "csr") - c_mean("csr")

    fams = [f for f in FAMILIES if f"fam:{f}" in E and len(E[f"fam:{f}"])]
    keys = [f"fam:{f}" for f in fams]
    K = np.array([[k_mean(a, b) if j >= i else 0.0 for j, b in enumerate(keys)] for i, a in enumerate(keys)])
    K = np.triu(K) + np.triu(K, 1).T
    c = np.array([c_mean(a) for a in keys])
    at = [FAMILIES.index(f) for f in fams]

    Kf = np.full((len(FAMILIES), len(FAMILIES)), np.nan)
    cf = np.full(len(FAMILIES), np.nan)
    Kf[np.ix_(at, at)], cf[at] = K, c
    mix = {"fit_case_id": job["meta"]["fit_case_id"], "K": Kf, "c": cf, "oracle": s_oracle, "csr": s_csr, "p": job["p"]}
    out = []
    for name, w_full in weightings(job["p"], FAMILIES.index(job["meta"]["family"]), job["temps"]).items():
        w = w_full[at]
        lost = float(w_full.sum() - w.sum())
        if w.sum() <= 0:
            out.append(base | {"variant": name, "regret": np.nan, "gain": s_csr - s_oracle, "lost": lost})
            continue
        w = w / w.sum()
        s = 0.5 * w @ K @ w - w @ c
        out.append(base | {"variant": name, "score": s, "oracle": s_oracle, "csr": s_csr,
                           "regret": s - s_oracle, "gain": s_csr - s_oracle, "lost": lost,
                           "n_components": int((w > 1e-3).sum())})
    return out, mix


# ------------------------------------------------------------------------------------ summary

def summarize(df: pd.DataFrame, cfg: dict) -> tuple[dict, str]:
    rng = np.random.default_rng(cfg["seed"])
    variants = [v for v in df.variant.unique() if v != "none"]
    wide = df.pivot_table(index="fit_case_id", columns="variant", values="regret")
    gain = df.drop_duplicates("fit_case_id").set_index("fit_case_id").gain
    info = df.drop_duplicates("fit_case_id").set_index("fit_case_id")

    def cell(ids) -> dict:
        ids = [i for i in ids if i in wide.index]
        g = gain.loc[ids]
        se = g.std(ddof=1) / np.sqrt(len(g)) if len(g) > 1 else np.inf
        resolved = g.mean() > cfg["skill_min_z"] * se
        out = {"n": len(ids), "gain": float(g.mean())}
        B = rng.integers(0, len(ids), (cfg["bootstrap"], len(ids))) if ids else None
        for v in variants:
            r = wide.loc[ids, v]
            d = (wide.loc[ids, "hard"] - r).to_numpy()          # > 0: v beats hard
            ok = ~np.isnan(d)
            ci = np.nanpercentile(np.nanmean(d[B], 1), [2.5, 97.5]) if ok.sum() > 1 else [np.nan, np.nan]
            out[v] = {"regret": float(r.mean()), "skill": float(1 - r.sum() / g.sum()) if resolved else None,
                      "better_than_hard": float(np.nanmean(d)), "ci": [float(ci[0]), float(ci[1])],
                      "skill_gain": float(np.nansum(d) / g.sum()) if resolved else None}
        return out

    groups = {"all": list(wide.index)}
    for s in ["paper", "strong"]:
        groups[f"set {s}"] = list(info.index[info.sets.str.contains(s)])
    groups["structured"] = list(info.index[info.family != "poisson"])
    for f in FAMILIES:
        groups[f"family {f}"] = list(info.index[info.family == f])
    conf = pd.cut(info.p_max, [0, 0.5, 0.8, 1.0001], labels=["max P < 0.5", "0.5-0.8", "> 0.8"])
    for lab in conf.cat.categories:
        groups[f"confidence {lab}"] = list(info.index[conf == lab])
    groups["argmax right"] = list(info.index[info.argmax == info.family])
    groups["argmax wrong"] = list(info.index[info.argmax != info.family])
    rep = {k: cell(v) for k, v in groups.items() if len(v)}

    order = ["hard", "soft", *[v for v in variants if v.startswith("temp")], "top2", "uniform", "true"]
    order = [v for v in order if v in variants]
    L = ["# Soft routing: " + cfg["name"], "",
         "Kernel score of the paper pipeline's fits (classifier " + cfg["classifier"] + ", frozen estimators), "
         "fit on replicate 0, scored on replicate 1. Every weighting is scored from the same simulations. "
         "Skill = 1 − Σregret/Σgain (shown where the gain is resolved). **Δ skill vs hard** = Σ(regret_hard − "
         "regret_v)/Σgain: the skill points a weighting adds over the paper's argmax routing; the interval is a "
         "bootstrap over clouds of the mean regret difference, in regret units (> 0 means better than hard).", "",
         "`true` routes every cloud to its true family (with that family's estimate): the ceiling on what any "
         "routing can gain. `uniform` ignores the classifier: equal weight on every family's fit.", "",
         "| clouds | n | " + " | ".join(f"skill {v}" for v in order) + " | " +
         " | ".join(f"Δ skill {v}" for v in order if v != "hard") + " |",
         "|---|---|" + "---|" * (2 * len(order) - 1)]
    for k, c in rep.items():
        sk = lambda v: "—" if c[v]["skill"] is None else f"{c[v]['skill']:.3f}"
        dk = lambda v: ("—" if c[v]["skill_gain"] is None else f"{c[v]['skill_gain']:+.3f}") + \
            f" [{c[v]['ci'][0]:+.1e}, {c[v]['ci'][1]:+.1e}]"
        L.append(f"| {k} | {c['n']} | " + " | ".join(sk(v) for v in order) + " | "
                 + " | ".join(dk(v) for v in order if v != "hard") + " |")
    lost = df[df.variant == "soft"].lost
    L += ["", f"Mixture mass lost to unrealisable estimates (soft): mean {lost.mean():.4f}, "
          f"max {lost.max():.3f}; clouds with a failed family: {(df.drop_duplicates('fit_case_id').failed != '').sum()}.", ""]
    return rep, "\n".join(L) + "\n"


# ----------------------------------------------------------------------------------------- main

def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(HERE / "config.yaml"))
    p.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    p.add_argument("--limit", type=int, help="score only N clouds (smoke test)")
    args = p.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text())
    out = HERE / "results" / (cfg["name"] + (f"_limit{args.limit}" if args.limit else ""))
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(Path(args.config).read_text())
    t0 = time.time()

    ch = clouds(cfg)
    if args.limit:
        ch = ch.sample(args.limit, random_state=0)
    r = bank_rows({"families": FAMILIES})
    post = paper_classifier(cfg, cfg["classifier"]).loc[ch.index, FAMILIES]
    est = {f: paper_estimator(cfg, f, m).loc[ch.index, TARGETS[f]] for f, m in cfg["estimators"].items()}
    xs = observed(r.loc[ch.scored_case_id])
    log(f"{len(ch)} clouds ({ch.family.value_counts().to_dict()}), {args.workers} workers")

    jobs = []
    for cid, c in ch.iterrows():
        pv = post.loc[cid].to_numpy(float)
        models = {"oracle": true_model(r.loc[cid]), "csr": ("poisson", {"nbar": float(len(xs[c.scored_case_id]))}),
                  "fam:poisson": ("poisson", {"nbar": float(r.n[cid])})}
        models |= {f"fam:{f}": (f, {k: float(v) for k, v in est[f].loc[cid].items()}) for f in est}
        am = FAMILIES[int(np.argmax(pv))]
        meta = {"fit_case_id": cid, "family": c.family, "stratum": c.stratum, "sets": c.sets, "argmax": am,
                "p_max": float(pv.max()), "p_true": float(pv[FAMILIES.index(c.family)]),
                "entropy": float(-(pv * np.log(np.clip(pv, 1e-12, 1))).sum())}
        jobs.append({"scored_case_id": c.scored_case_id, "x": xs[c.scored_case_id].astype(float), "models": models,
                     "p": pv, "meta": meta, "kernel": cfg["kernel"], "seed": cfg["seed"],
                     "temps": cfg["temperatures"]})

    recs, mixes = [], []
    with mp.get_context("fork").Pool(args.workers) as pool:
        for i, (rr, mix) in enumerate(pool.imap_unordered(score_cloud, jobs), 1):
            recs += rr
            mixes += [mix] if mix else []
            if i % 50 == 0 or i == len(jobs):
                log(f"  {i}/{len(jobs)} clouds")
    df = pd.DataFrame(recs)
    df.to_csv(out / "clouds.csv", index=False)
    # everything a weighting needs, per cloud: any w scores offline as 1/2 w'Kw - w'c (families in FAMILIES order)
    np.savez(out / "mixture_terms.npz", fit_case_id=np.array([m["fit_case_id"] for m in mixes]),
             K=np.stack([m["K"] for m in mixes]), c=np.stack([m["c"] for m in mixes]),
             oracle=np.array([m["oracle"] for m in mixes]), csr=np.array([m["csr"] for m in mixes]),
             p=np.stack([m["p"] for m in mixes]), families=np.array(FAMILIES))
    rep, md = summarize(df, cfg)
    write_json(out / "report.json", rep)
    (out / "summary.md").write_text(md)
    print(md)
    log(f"-> {out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
