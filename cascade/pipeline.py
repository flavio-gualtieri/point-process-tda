#!/usr/bin/env python3
"""Put the separately trained components together, and summarise the tau sweep.

    python cascade/pipeline.py [--config ...] [--clouds]      # after components; sbatch cascade/assemble.sh

For every model m (the same model at stages 2 and 3) and every stage-2 tau, on the TEST split:

    stage 1 (results/<name>/stage1)   poisson    -> poisson, nbar_hat = n            (stage3.poisson: mle)
                                      clustered  -> clustered_m: thomas | nested | lgcp | reject
                                      repulsive  -> repulsive_m: matern2 | reject    (matern2 if no model)
                                      reject     -> poisson, nbar_hat = n
    stage 3                           theta_hat from the routed family's regressor, trained at
                                      stage-3 tau = the stage-2 tau and/or 0 (pipeline.stage3_taus)

Nothing is retrained here: it is a lookup into each component's predictions.npz.

Scores, all on clouds as they actually arrive (report.json per assembly, sweep/*.csv overall):
  stage2        on test clouds stage 1 routes to a branch: balanced accuracy over the branch's
                families (reject = wrong), share of poisson impostors rejected, share of the
                branch's own clouds rejected (false rejects)
  stage3        on test clouds that end up at their TRUE family: RMSE(log)/s.d. per family, for the
                filtered regressor (stage-3 tau = stage-2 tau) and the unfiltered one (tau 0) on
                exactly the same clouds -- the test of "train stage 3 on clean clouds"
  end_to_end    family accuracy under the balanced-groups prior (dominated by stage 1; see README)

Output  cascade/results/<run>/assembled/tau_<tau>/<model>/report.json   (+ clouds_s3tau<t>.csv with --clouds)
        cascade/results/<run>/sweep/stage2.csv, stage3.csv, end_to_end.csv, sweep.png
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from common import DEFAULT_CONFIG, REGIME, RESULTS, TARGETS, load_config, load_predictions, manifest, sample_weights, write_json
from components import REJECT, out_dir

INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
BRANCH = {"clustered": ["thomas", "nested", "lgcp"], "repulsive": ["matern2"]}


def load(cfg, tau, comp, model):
    path = out_dir(cfg, tau, comp, model) / "predictions.npz"
    return np.load(path) if path.exists() else None


def calls(z, ids: pd.Index) -> pd.Series:
    return pd.Series(np.array(z["classes"])[z["posterior"].argmax(1)], index=z["case_id"]).loc[ids]


def theta(z, ids: pd.Index) -> pd.DataFrame:
    return pd.DataFrame(z["theta_hat"], index=z["case_id"], columns=list(z["targets"])).loc[ids]


def assemble(cfg, tau, model, s1: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame | None:
    """Per test cloud: family, group_hat, family_hat (None if a needed component is missing)."""
    clu, rep = load(cfg, tau, "clustered", model), load(cfg, tau, "repulsive", model)
    if clu is None:
        return None
    r = rows.loc[s1.index, ["family", "n"]].copy()
    r["group_hat"] = s1.pred
    r["stage2_call"] = None
    r["family_hat"] = "poisson"
    for branch, z in (("clustered", clu), ("repulsive", rep)):
        ids = r.index[r.group_hat == branch]
        c = calls(z, ids) if z is not None else pd.Series("matern2", index=ids)
        r.loc[ids, "stage2_call"] = c.to_numpy()
        r.loc[ids, "family_hat"] = c.replace({REJECT: "poisson"}).to_numpy()
    return r


def stage2_scores(r: pd.DataFrame) -> dict:
    out = {}
    for branch, fams in BRANCH.items():
        g = r[r.group_hat == branch]
        own = g[g.family.isin(fams)]
        out[branch] = {
            "n_routed": int(len(g)),
            "balanced_accuracy": float(np.mean([(own.stage2_call[own.family == f] == f).mean() for f in fams])),
            "impostors_rejected": float((g[g.family == "poisson"].stage2_call == REJECT).mean()),
            "own_rejected": float((own.stage2_call == REJECT).mean()),
            "impostor_share": float((~g.family.isin(fams)).mean())}
    return out


def stage3_scores(cfg, tau, model, r: pd.DataFrame, rows: pd.DataFrame) -> dict:
    """On clouds that end up at their true family: filtered (tau) vs unfiltered (0) regressors."""
    out = {}
    for f in ("thomas", "nested", "matern2", "lgcp"):
        ids = r.index[(r.family == f) & (r.family_hat == f)]
        if not len(ids):
            continue
        y = np.log(rows.loc[ids, TARGETS[f]].to_numpy(float))
        entry = {"n": int(len(ids))}
        for label, t in (("filtered", tau), ("unfiltered", 0)):
            z = load(cfg, t, f, model)
            if z is None:
                continue
            rmse = np.sqrt(((np.log(theta(z, ids).to_numpy()) - y) ** 2).mean(0)) / y.std(0)
            entry[label] = float(rmse.mean())
            entry[f"{label}_per_target"] = dict(zip(TARGETS[f], rmse.round(4).tolist()))
        out[f] = entry
    return out


def stage3_model(model, family: str) -> str:
    """A stage-3 model is one name for every family, or a mapping family -> name."""
    return model[family] if isinstance(model, dict) else model


def clouds(cfg, r: pd.DataFrame, tau3, model) -> pd.DataFrame:
    """evaluate.py's input: theta_hat per cloud from the stage-3 regressors at tau3. `model` is one
    name or a mapping family -> name (e.g. networks for the clustered families, PersLay for matern2)."""
    th = {c: {"nbar": float(n)} for c, n in r.n[r.family_hat == "poisson"].items()}
    for f in ("thomas", "nested", "matern2", "lgcp"):
        ids = r.index[r.family_hat == f]
        z = load(cfg, tau3, f, stage3_model(model, f))
        th |= {c: {k: float(v) for k, v in row.items()} for c, row in theta(z, ids).iterrows()}
    return r.assign(theta_hat=[json.dumps(th[c]) for c in r.index])


def plot(s2: pd.DataFrame, s3: pd.DataFrame, e2e: pd.DataFrame, models: list[str], path) -> None:
    import matplotlib.pyplot as plt
    color = dict(zip(models, SERIES))
    fig, axes = plt.subplots(2, 4, figsize=(19, 8.5), facecolor=SURFACE)
    panels = [(s2[s2.branch == "clustered"], "balanced_accuracy", "stage 2 clustered: balanced accuracy (routed)"),
              (s2[s2.branch == "clustered"], "impostors_rejected", "stage 2 clustered: poisson impostors rejected"),
              (s2[s2.branch == "clustered"], "own_rejected", "stage 2 clustered: own clouds rejected (lower = better)"),
              (e2e, "family_balanced_accuracy", "end to end: family accuracy")]
    for ax, (df, col, title) in zip(axes[0], panels):
        for m in models:
            g = df[df.model == m].sort_values("tau")
            if len(g):
                ax.plot(g.tau, g[col], color=color[m], linewidth=2, marker="o", markersize=4, label=m)
        ax.set_title(title, fontsize=9, color=INK, loc="left")
    for ax, f in zip(axes[1], ["thomas", "nested", "matern2", "lgcp"]):
        for m in models:
            g = s3[(s3.model == m) & (s3.family == f)].sort_values("tau")
            if len(g):
                ax.plot(g.tau, g.gain, color=color[m], linewidth=2, marker="o", markersize=4, label=m)
        ax.axhline(0, color=INK2, linewidth=1, linestyle="--")
        ax.set_title(f"stage 3 {f}: gain from filtering\n(unfiltered − filtered RMSE/s.d.; > 0 = filtering helps)",
                     fontsize=9, color=INK, loc="left")
    for ax in axes.flat:
        ax.set_xlabel("τ", fontsize=8.5, color=INK2)
        ax.set_facecolor(SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=INK2, labelsize=8)
        ax.grid(color=GRID, linewidth=0.6)
    axes[0, 0].legend(frameon=False, fontsize=7.5)
    fig.suptitle("Component-wise cascade with re-routing: effect of the regime cutoff τ, per model (test, as routed)",
                 fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--clouds", action="store_true", help="also write per-cloud CSVs (evaluate.py's input)")
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    rows = manifest()
    s1 = load_predictions(RESULTS / cfg["name"] / "stage1" / "predictions.npz")
    s1 = s1[s1.split == "test"]
    models = [m for m in cfg["stage2"]["models"] if m in cfg["stage3"]["models"]]

    rec2, rec3, rece = [], [], []
    for model in models:
        for tau in cfg["regime"]["taus"]:
            r = assemble(cfg, tau, model, s1, rows)
            if r is None:
                continue
            w = sample_weights(r.family)
            rep = {"model": model, "tau": tau,
                   "end_to_end": {"family_balanced_accuracy": float(np.average(r.family_hat == r.family, weights=w)),
                                  "calls": {f: g.family_hat.value_counts(normalize=True).round(4).to_dict()
                                            for f, g in r.groupby("family")}},
                   "stage2": stage2_scores(r), "stage3": stage3_scores(cfg, tau, model, r, rows)}
            out = RESULTS / cfg["name"] / "assembled" / f"tau_{tau:g}" / model
            out.mkdir(parents=True, exist_ok=True)
            write_json(out / "report.json", rep)
            if args.clouds:
                for t3 in {tau if s == "same" else 0 for s in cfg["pipeline"]["stage3_taus"]}:
                    if load(cfg, t3, "thomas", model) is not None:
                        clouds(cfg, r, t3, model).to_csv(out / f"clouds_s3tau{t3:g}.csv")
            rece.append({"model": model, "tau": tau, **rep["end_to_end"] | {"calls": None}})
            rec2 += [{"model": model, "tau": tau, "branch": b, **v} for b, v in rep["stage2"].items()]
            rec3 += [{"model": model, "tau": tau, "family": f, "n": v["n"], "filtered": v.get("filtered"),
                      "unfiltered": v.get("unfiltered"),
                      "gain": (v["unfiltered"] - v["filtered"]) if "filtered" in v and "unfiltered" in v else None}
                     for f, v in rep["stage3"].items()]
            c = rep["stage2"]["clustered"]
            print(f"{model:24s} tau {tau:<4g} e2e {rep['end_to_end']['family_balanced_accuracy']:.4f} | clustered "
                  f"acc {c['balanced_accuracy']:.3f} impostors rej {c['impostors_rejected']:.2f} own rej "
                  f"{c['own_rejected']:.2f} | s3 filt/unfilt " + " ".join(
                      f"{f}:{v.get('filtered', float('nan')):.3f}/{v.get('unfiltered', float('nan')):.3f}"
                      for f, v in rep["stage3"].items()), flush=True)

    sweep = RESULTS / cfg["name"] / "sweep"
    sweep.mkdir(parents=True, exist_ok=True)
    s2, s3, e2e = pd.DataFrame(rec2), pd.DataFrame(rec3), pd.DataFrame(rece).drop(columns="calls")
    s2.to_csv(sweep / "stage2.csv", index=False)
    s3.to_csv(sweep / "stage3.csv", index=False)
    e2e.to_csv(sweep / "end_to_end.csv", index=False)
    if len(s2):
        plot(s2, s3, e2e, models, sweep / "sweep.png")
    print(f"-> {sweep}")


if __name__ == "__main__":
    main()
