#!/usr/bin/env python3
"""Stages 2 and 3 of the cascade: every component trained on its own, on in-regime bank clouds.

    python cascade/components.py --model hgb --tau 0.5              # every component, one tau
    python cascade/components.py --model perslay_z_dtm_k10 --tau-index 2
    python cascade/components.py --nn-grid-index 7                  # (nn model, tau) pair no. 7
    sbatch cascade/components.sh                                    # hgb, CPU array over regime.taus
    sbatch cascade/components_nn.sh                                 # every nn model x tau, GPU array

Components
    clustered   stage 2: thomas | nested | lgcp | reject
    repulsive   stage 2: matern2 | reject          (only exists with reject: one family otherwise)
    thomas, nested, matern2, lgcp
                stage 3: regressors of the family's TARGETS, squared error on log(target)
Training rows, train split only, no stage-1 prediction involved:
    family classes  the branch's families' clouds IN REGIME at tau (regime.Cutoffs, from
                    results/<regime.source_run>/regime_analysis/cutoffs.json)
    reject          config stage2.reject.sources: every poisson cloud, the branch's own clouds
                    outside the regime, the other group's clouds
    regressors      the family's clouds in regime at tau
Classifiers weight every class (reject included) equally. Networks early-stop on val rows chosen
by the same rules. A reject at assembly time re-routes to poisson (pipeline.py).

Models: any name under the config's `models:` -- `kind: hgb` (trees on cascade/features.py, CPU)
or `kind: nn` (cloudforger's PHNet on the entry's input via cascade/nn.py, GPU). An existing
component output is skipped unless --force, so adding a model or a tau only trains what is new.

Predictions are written for EVERY val/test cloud of every family, so the assembled pipeline can
look up whatever stage 1 routes to a component.

Scores (report.json), test split:
  classifier  balanced accuracy over the branch's families (a reject counts as wrong), on
              in_regime clouds and on all clouds of those families; and reject rates: of test
              poisson clouds, of the other group's clouds, of in-regime family clouds (false
              rejects), of out-of-regime family clouds
  regressor   RMSE of log(target) per target, and that over the target's s.d. on the same set
              (1 = no better than the mean), on in_regime clouds and on all of the family's clouds

Output  cascade/results/<run>/components/tau_<tau>/<component>_<model>/
            predictions.npz   case_id, split, and posterior + classes | theta_hat + targets
            report.json, model.joblib | model.pt
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from common import (DEFAULT_CONFIG, REGIME, RESULTS, TARGETS, Data, fit, load_config, make_classifier,
                    make_regressor, manifest, write_json)
from regime import Cutoffs

REJECT = "reject"
COMPONENTS = {"clustered": ("classify", ["thomas", "nested", "lgcp"]),
              "repulsive": ("classify", ["matern2"]),
              **{f: ("regress", [f]) for f in ("thomas", "nested", "matern2", "lgcp")}}


def out_dir(cfg: dict, tau: float, component: str, model: str) -> Path:
    return RESULTS / cfg["name"] / "components" / f"tau_{tau:g}" / f"{component}_{model}"


def in_regime(rows: pd.DataFrame, families: list[str], cut: Cutoffs, tau: float) -> np.ndarray:
    """Rows of `families` past the cutoff at tau; everything else False."""
    out = np.zeros(len(rows), bool)
    for f in families:
        m = (rows.family == f).to_numpy()
        out[m] = cut.mask(rows[m], f, tau)
    return out


def classes_of(families: list[str], cfg: dict) -> list[str]:
    return families + ([REJECT] if cfg["stage2"]["reject"]["enabled"] else [])


def labels(rows: pd.DataFrame, families: list[str], cut: Cutoffs, tau: float, cfg: dict) -> np.ndarray:
    """Class index per row for a classifier component; -1 = not a training example."""
    fam = rows.family.to_numpy()
    inreg = in_regime(rows, families, cut, tau)
    y = np.full(len(rows), -1)
    for i, f in enumerate(families):
        y[(fam == f) & inreg] = i
    rj = cfg["stage2"]["reject"]
    if rj["enabled"]:
        group = REGIME[families[0]]
        src = np.zeros(len(rows), bool)
        if "poisson" in rj["sources"]:
            src |= fam == "poisson"
        if "below_cutoff" in rj["sources"]:
            src |= np.isin(fam, families) & ~inreg
        if "other_group" in rj["sources"]:
            src |= np.array([REGIME[f] not in ("poisson", group) for f in fam])
        y[src] = len(families)
    return y


def class_weights(y: np.ndarray) -> np.ndarray:
    counts = np.bincount(y)
    w = 1.0 / counts[y]
    return w * len(w) / w.sum()


# ------------------------------------------------------------------------------------- scoring

def score(kind: str, families: list[str], rows: pd.DataFrame, pred: dict, inreg: np.ndarray) -> dict:
    """rows/pred/inreg aligned over held-out clouds; scored on the test split."""
    test = (rows.split == "test").to_numpy()
    fam = rows.family.to_numpy()
    mine = test & np.isin(fam, families)
    out = {}
    if kind == "classify":
        called = np.array(pred["classes"])[pred["posterior"].argmax(1)]
        for subset, m in (("in_regime", mine & inreg), ("all", mine)):
            recall = {f: float((called[m & (fam == f)] == f).mean()) for f in families}
            out[subset] = {"n": int(m.sum()), "balanced_accuracy": float(np.mean(list(recall.values()))),
                           "recall": recall}
        if REJECT in pred["classes"]:
            group = REGIME[families[0]]
            sets = {"poisson": test & (fam == "poisson"),
                    "other_group": test & np.array([REGIME[f] not in ("poisson", group) for f in fam]),
                    "in_regime": mine & inreg, "out_of_regime": mine & ~inreg}
            out["reject_rate"] = {k: (float((called[m] == REJECT).mean()) if m.any() else None)
                                  for k, m in sets.items()}
        return out
    for subset, m in (("in_regime", mine & inreg), ("all", mine)):
        r = rows[m]
        y = np.log(r[pred["targets"]].to_numpy(float))
        rmse = np.sqrt(((np.log(pred["theta_hat"][m]) - y) ** 2).mean(0))
        out[subset] = {"n": int(m.sum()),
                       "rmse_log": dict(zip(pred["targets"], rmse.round(4).tolist())),
                       "rmse_over_sd": dict(zip(pred["targets"], (rmse / y.std(0)).round(4).tolist()))}
    return out


# ------------------------------------------------------------------------------------- backends

def run_hgb(kind, families, d: Data, cut, tau, cfg):
    held = d.split != "train"
    if kind == "classify":
        classes = classes_of(families, cfg)
        y = labels(d.rows, families, cut, tau, cfg)
        tr = (d.split == "train") & (y >= 0)
        model = fit(make_classifier("hgb"), d.X[tr], y[tr], class_weights(y[tr]))
        post = np.zeros((held.sum(), len(classes)))
        post[:, model.classes_] = model.predict_proba(d.X[held])
        pred = {"posterior": post, "classes": classes}
    else:
        targets = TARGETS[families[0]]
        tr = (d.split == "train") & in_regime(d.rows, families, cut, tau)
        Y = np.log(d.rows[targets].to_numpy(float))
        model, theta = [], np.zeros((held.sum(), len(targets)))
        for j in range(len(targets)):
            model.append(fit(make_regressor("hgb"), d.X[tr], Y[tr, j], None))
            theta[:, j] = model[-1].predict(d.X[held])
        if cfg["stage3"]["clip_to_train_range"]:
            theta = np.clip(theta, Y[tr].min(0), Y[tr].max(0))
        pred = {"theta_hat": np.exp(theta), "targets": targets}
    return d.case_id[held], d.split[held], pred, model, int(tr.sum())


def run_nn(kind, families, ds, cut, tau, cfg, s: dict, tag: str):
    import torch.nn as tnn

    import nn
    from cloudforger.training import data as D
    dataset, n_tags, rows = ds
    split = rows.split.to_numpy()
    held = np.flatnonzero(split != "train")
    if kind == "classify":
        classes = classes_of(families, cfg)
        y = labels(rows, families, cut, tau, cfg)
        tr, va = (np.flatnonzero((split == sp) & (y >= 0)) for sp in ("train", "val"))
        yy = np.maximum(y, 0).astype(np.int64)                    # -1 rows are never indexed
        model, info = nn.train(dataset, n_tags, s, yy, tr, va, nn.weighted_ce(y[tr], len(classes)),
                               len(classes), tag)
        pred = {"posterior": nn.softmax(nn.predict(model, dataset, s, held)), "classes": classes}
        saved = {"state_dict": model.state_dict()}
    else:
        targets = TARGETS[families[0]]
        inreg = in_regime(rows, families, cut, tau)
        tr, va = (np.flatnonzero((split == sp) & inreg) for sp in ("train", "val"))
        yz, norm = D.targets(dataset.manifest, targets, tr)
        model, info = nn.train(dataset, n_tags, s, np.nan_to_num(yz), tr, va, tnn.MSELoss(), len(targets), tag)
        logs = np.log(D.invert_targets(nn.predict(model, dataset, s, held), norm))
        if cfg["stage3"]["clip_to_train_range"]:
            Y = np.log(rows[targets].to_numpy(float)[tr])
            logs = np.clip(logs, Y.min(0), Y.max(0))
        pred = {"theta_hat": np.exp(logs), "targets": targets}
        saved = {"state_dict": model.state_dict(), "target_norm": norm}
    saved["fit"] = {k: info[k] for k in ("best_epoch", "epochs_run", "best_val_loss")}
    return rows.index.to_numpy(str)[held], split[held], pred, saved, len(tr)


# ----------------------------------------------------------------------------------------- main

def nn_grid(cfg: dict) -> list[tuple[str, float]]:
    """Every (nn model, tau) pair the config asks for, in a fixed order (the GPU array's index)."""
    names = list(dict.fromkeys(cfg["stage2"]["models"] + cfg["stage3"]["models"]))
    return [(m, t) for m in names if cfg["models"][m]["kind"] == "nn" for t in cfg["regime"]["taus"]]


def wanted(comp: str, model: str, cfg: dict) -> bool:
    kind, families = COMPONENTS[comp]
    if kind == "classify":
        return model in cfg["stage2"]["models"] and len(classes_of(families, cfg)) > 1
    return model in cfg["stage3"]["models"]


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--model", help="a name under `models:`")
    p.add_argument("--tau", type=float, help="default: every regime.taus")
    p.add_argument("--tau-index", type=int, help="index into regime.taus (for SLURM arrays)")
    p.add_argument("--nn-grid-index", type=int, help="index into nn_grid(): picks model AND tau")
    p.add_argument("--only", nargs="+", choices=list(COMPONENTS))
    p.add_argument("--force", action="store_true", help="retrain components whose output exists")
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    taus = cfg["regime"]["taus"]
    if args.nn_grid_index is not None:
        model_name, tau = nn_grid(cfg)[args.nn_grid_index]
        taus = [tau]
    else:
        model_name = args.model
        taus = [args.tau] if args.tau is not None else [taus[args.tau_index]] if args.tau_index is not None else taus
    kind_of_model = cfg["models"][model_name]["kind"]
    cut = Cutoffs(RESULTS / cfg["regime"]["source_run"] / "regime_analysis" / "cutoffs.json")
    todo = [(t, c) for t in taus for c in (args.only or COMPONENTS)
            if wanted(c, model_name, cfg) and (args.force or not (out_dir(cfg, t, c, model_name) / "report.json").exists())]
    if not todo:
        print(f"{model_name} tau {taus}: every component already exists (--force to retrain)")
        return

    t0 = time.time()
    rows_all = manifest()
    if kind_of_model == "hgb":
        data = Data.load()
    else:
        import nn
        s = nn.spec(cfg, model_name)
        dataset, n_tags, arm = nn.build(s)
        rows_nn = rows_all.loc[dataset.manifest["case_id"].to_numpy(str)].assign(
            split=dataset.manifest["split"].to_numpy(str))
        print(f"{model_name}: {arm}, {len(rows_nn)} patterns ({time.time() - t0:.0f}s)", flush=True)

    for tau, comp in todo:
        kind, families = COMPONENTS[comp]
        t1 = time.time()
        if kind_of_model == "hgb":
            case_id, split, pred, model, n_train = run_hgb(kind, families, data, cut, tau, cfg)
        else:
            case_id, split, pred, model, n_train = run_nn(kind, families, (dataset, n_tags, rows_nn), cut, tau,
                                                          cfg, s, f"{model_name}/tau{tau:g}/{comp}")
        out = out_dir(cfg, tau, comp, model_name)
        out.mkdir(parents=True, exist_ok=True)
        np.savez(out / "predictions.npz", case_id=case_id, split=split, **{k: np.asarray(v) for k, v in pred.items()})
        if kind_of_model == "hgb":
            import joblib
            joblib.dump(model, out / "model.joblib")
        else:
            import torch
            torch.save(model, out / "model.pt")

        r = rows_all.loc[case_id].assign(split=split)
        rep = {"component": comp, "kind": kind, "families": families, "model": model_name, "tau": tau,
               "rule": {f: cut.describe(f, tau) for f in families}, "n_train": n_train,
               "seconds": round(time.time() - t1),
               **score(kind, families, r, pred, in_regime(r, families, cut, tau))}
        if kind_of_model == "nn":
            rep["fit"] = model["fit"]
        write_json(out / "report.json", rep)
        if kind == "classify":
            head = (f"bal acc {rep['in_regime']['balanced_accuracy']:.4f} in regime, {rep['all']['balanced_accuracy']:.4f} all"
                    + ("" if "reject_rate" not in rep else "  | rejects " + " ".join(
                        f"{k}:{'-' if v is None else f'{v:.2f}'}" for k, v in rep["reject_rate"].items())))
        else:
            head = ("rmse/sd in regime " + " ".join(f"{k}:{v:.3f}" for k, v in rep["in_regime"]["rmse_over_sd"].items())
                    + " | all " + " ".join(f"{k}:{v:.3f}" for k, v in rep["all"]["rmse_over_sd"].items()))
        print(f"tau {tau:<4g} {comp:9s} {model_name:24s} n_train {n_train:6d}  {head}  ({rep['seconds']}s)", flush=True)


if __name__ == "__main__":
    main()
