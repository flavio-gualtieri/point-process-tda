#!/usr/bin/env python3
"""Put the separately trained components together, and summarise the tau sweep.

    python cascade/pipeline.py [--config ...]           # after components.py; sbatch cascade/assemble.sh

For every tau in regime.taus and every (stage-2 model, stage-3 model) pair whose components exist:

    stage 1 (results/<name>/stage1)   poisson    -> family poisson, nbar_hat = n  (stage3.poisson: mle)
                                      repulsive  -> family matern2               (one family, no model)
                                      clustered  -> stage 2's argmax over thomas | nested | lgcp
    stage 3                           theta_hat from the routed family's regressor

Nothing is retrained here: it is a lookup into each component's predictions.npz on the TEST split.

Output  cascade/results/<run>/assembled/tau_<tau>/<s2>_<s3>/clouds.csv    one row per test cloud:
            family, family_hat, group_hat, n, delta_tilde, theta_hat (json) -- evaluate.py's input
        cascade/results/<run>/assembled/tau_<tau>/<s2>_<s3>/report.json   end-to-end scores
        cascade/results/<run>/sweep/components.csv, assembled.csv, sweep.png
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from common import (DEFAULT_CONFIG, RESULTS, TARGETS, load_config, load_predictions, manifest,
                    sample_weights, write_json)
from components import COMPONENTS, out_dir

FAMILY_COLOR = {"thomas": "#2a78d6", "nested": "#eb6834", "matern2": "#1baf7a", "lgcp": "#eda100"}
MODEL_STYLE = {"hgb": "-", "nn": "--"}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def stage1_test(cfg: dict) -> pd.DataFrame:
    s1 = load_predictions(RESULTS / cfg["name"] / "stage1" / "predictions.npz")
    return s1[s1.split == "test"]


def assemble(cfg: dict, tau: float, s2: str, s3: str, s1: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame | None:
    def component(name, model):
        path = out_dir(cfg, tau, name, model) / "predictions.npz"
        return np.load(path) if path.exists() else None

    c2 = component("clustered", s2)
    c3 = {f: component(f, s3) for f in ("thomas", "nested", "matern2", "lgcp")}
    if c2 is None or any(v is None for v in c3.values()):
        return None
    r = rows.loc[s1.index, ["family", "n", "delta_tilde"]].copy()
    r["group_hat"] = s1.pred
    post = pd.DataFrame(c2["posterior"], index=c2["case_id"], columns=list(c2["classes"]))
    r["family_hat"] = r.group_hat.map({"poisson": "poisson", "repulsive": "matern2"})
    clu = r.index[r.group_hat == "clustered"]
    r.loc[clu, "family_hat"] = post.loc[clu].idxmax(axis=1).to_numpy()

    theta = {c: {"nbar": float(n)} for c, n in r.n[r.family_hat == "poisson"].items()}
    for f, z in c3.items():
        want = r.index[r.family_hat == f]
        th = pd.DataFrame(z["theta_hat"], index=z["case_id"], columns=list(z["targets"])).loc[want]
        theta |= {c: {k: float(v) for k, v in row.items()} for c, row in th.iterrows()}
    r["theta_hat"] = [json.dumps(theta[c]) for c in r.index]
    return r


def report(r: pd.DataFrame, rows: pd.DataFrame) -> dict:
    w = sample_weights(r.family)
    rep = {"n": int(len(r)),
           "family_balanced_accuracy": float(np.average(r.family_hat == r.family, weights=w)),
           "calls": {f: g.family_hat.value_counts(normalize=True).round(4).to_dict() for f, g in r.groupby("family")},
           "stage3_on_correctly_routed": {}}
    for f in ("thomas", "nested", "matern2", "lgcp"):
        g = r[(r.family == f) & (r.family_hat == f)]
        if not len(g):
            continue
        targets = TARGETS[f]
        y = np.log(rows.loc[g.index, targets].to_numpy(float))
        est = np.log(np.array([[json.loads(t)[k] for k in targets] for t in g.theta_hat]))
        rmse = np.sqrt(((est - y) ** 2).mean(0))
        rep["stage3_on_correctly_routed"][f] = {"n": int(len(g)),
                                                "rmse_over_sd": dict(zip(targets, (rmse / y.std(0)).round(4).tolist()))}
    return rep


def component_table(cfg: dict) -> pd.DataFrame:
    recs = []
    for path in sorted((RESULTS / cfg["name"] / "components").glob("tau_*/*/report.json")):
        rep = json.loads(path.read_text())
        for subset in ("in_regime", "all"):
            s = rep[subset]
            value = s["balanced_accuracy"] if rep["kind"] == "classify" else float(np.mean(list(s["rmse_over_sd"].values())))
            recs.append({"tau": rep["tau"], "component": rep["component"], "model": rep["model"], "subset": subset,
                         "metric": "balanced_accuracy" if rep["kind"] == "classify" else "mean_rmse_over_sd",
                         "value": value, "n_test": s["n"], "n_train": rep["n_train"]})
    return pd.DataFrame(recs)


def plot(comp: pd.DataFrame, asm: pd.DataFrame, path) -> None:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), facecolor=SURFACE)
    ax = axes[0]
    for model, g in comp[(comp.component == "clustered") & (comp.subset == "all")].groupby("model"):
        g = g.sort_values("tau")
        ax.plot(g.tau, g.value, MODEL_STYLE[model], color=FAMILY_COLOR["thomas"], linewidth=2, marker="o",
                markersize=5, label=model)
    ax.set_title("stage 2 (clustered): balanced accuracy, all test clouds", fontsize=9.5, color=INK, loc="left")
    ax = axes[1]
    for (f, model), g in comp[(comp.component != "clustered") & (comp.subset == "all")].groupby(["component", "model"]):
        g = g.sort_values("tau")
        ax.plot(g.tau, g.value, MODEL_STYLE[model], color=FAMILY_COLOR[f], linewidth=2, marker="o", markersize=5,
                label=f"{f} {model}")
    ax.set_title("stage 3: mean RMSE(log θ) / s.d., all test clouds  (lower is better)", fontsize=9.5, color=INK,
                 loc="left")
    ax = axes[2]
    for (s2, s3), g in asm.groupby(["stage2", "stage3"]):
        g = g.sort_values("tau")
        ax.plot(g.tau, g.family_balanced_accuracy, MODEL_STYLE[s3], linewidth=2, marker="o", markersize=5,
                color=FAMILY_COLOR["lgcp"] if s2 == "hgb" else FAMILY_COLOR["nested"], label=f"stage 2 {s2}, stage 3 {s3}")
    ax.set_title("assembled: family accuracy (balanced groups), test", fontsize=9.5, color=INK, loc="left")
    for ax in axes:
        ax.set_xlabel("τ (training cutoff)", fontsize=8.5, color=INK2)
        ax.set_facecolor(SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=INK2, labelsize=8)
        ax.grid(color=GRID, linewidth=0.6)
        ax.legend(frameon=False, fontsize=7.5)
    fig.suptitle("Component-wise training: effect of the regime cutoff τ  (τ = 0: no filtering)", fontsize=11,
                 color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    rows = manifest()
    s1 = stage1_test(cfg)

    recs = []
    for tau in cfg["regime"]["taus"]:
        for s2 in cfg["stage2"]["models"]:
            for s3 in cfg["stage3"]["models"]:
                r = assemble(cfg, tau, s2, s3, s1, rows)
                if r is None:
                    continue
                out = RESULTS / cfg["name"] / "assembled" / f"tau_{tau:g}" / f"{s2}_{s3}"
                out.mkdir(parents=True, exist_ok=True)
                r.to_csv(out / "clouds.csv")
                rep = report(r, rows)
                write_json(out / "report.json", rep)
                recs.append({"tau": tau, "stage2": s2, "stage3": s3,
                             "family_balanced_accuracy": rep["family_balanced_accuracy"]})
                print(f"tau {tau:<4g} stage2 {s2:3s} stage3 {s3:3s}  family acc {rep['family_balanced_accuracy']:.4f}  "
                      + "  ".join(f"{f}:{np.mean(list(v['rmse_over_sd'].values())):.3f}"
                                  for f, v in rep["stage3_on_correctly_routed"].items()), flush=True)

    sweep = RESULTS / cfg["name"] / "sweep"
    sweep.mkdir(parents=True, exist_ok=True)
    comp, asm = component_table(cfg), pd.DataFrame(recs)
    comp.to_csv(sweep / "components.csv", index=False)
    asm.to_csv(sweep / "assembled.csv", index=False)
    if len(comp) and len(asm):
        plot(comp, asm, sweep / "sweep.png")
    print(f"-> {sweep}")


if __name__ == "__main__":
    main()
