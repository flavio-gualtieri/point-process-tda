#!/usr/bin/env python3
"""Train one unit: a classifier (one model over every family) or an estimator (one model, one family).

    python oneshot/train.py list [--kind cpu|gpu]           # the config's units, in array order
    python oneshot/train.py unit --kind cpu --index 3        # unit no. 3 of that kind (a SLURM array task)
    python oneshot/train.py classify --model hgb_classical
    python oneshot/train.py estimate --model nn_curves --family ring

Units come from the config: every `classify` model, and every `estimate` model x every family but
poisson. A unit is CPU or GPU by its learner (learners.GPU). A unit whose report.json exists is
skipped unless --force, so adding a model to the config and resubmitting trains only the new units.

Training rows are the train split; networks early-stop on val. Classifiers weight rows by the prior
(config `prior`). The regime's reference classifier also writes out-of-fold train predictions
(`crossfit_folds`, folds grouped by theta) when its learner supports it.

Scores (report.json), test split:
  classifier  accuracy under the prior, recall per family, NLL, and the detection rate per family
              (share NOT called poisson -- the CSR filter)
  estimator   RMSE of log(target) and that over the target's s.d. (1 = no better than the mean), on
              the family's test clouds

Output  oneshot/results/<run>/classify/<model>/  or  .../estimate/<family>/<model>/
            predictions.npz (core.py's contract), report.json, config.yaml, model.joblib | model.pt
"""

from __future__ import annotations

import argparse
import time

import numpy as np

import learners
from core import (TARGETS, config_arg, estimated_families, load_config, rows as load_rows, sample_weights,
                  save_config, save_predictions, unit_dir, write_json)


def units(cfg: dict, kind: str | None = None) -> list[tuple[str, str, str | None]]:
    """(task, model, family) in a fixed order; kind filters by cpu | gpu."""
    out = [("classify", m, None) for m in cfg["classify"]]
    out += [("estimate", m, f) for m in cfg["estimate"] for f in estimated_families(cfg)]
    if kind:
        out = [u for u in out if learners.needs_gpu(cfg, u[1]) == (kind == "gpu")]
    return out


def done(cfg: dict, task: str, model: str, family: str | None) -> bool:
    return (unit_dir(cfg, task, model, family) / "report.json").exists()


def classify(cfg: dict, config_path: str, model: str) -> None:
    t0 = time.time()
    r = load_rows(cfg)
    fam, split = r.family.to_numpy(), r.split.to_numpy()
    classes = cfg["families"]
    y = np.array([classes.index(f) for f in fam])
    w = sample_weights(fam, cfg)
    train, val = np.flatnonzero(split == "train"), np.flatnonzero(split == "val")
    held = np.flatnonzero(split != "train")
    learner = learners.make(cfg, model, r)
    print(f"classify {model}: {len(train)} train rows, input ready ({time.time() - t0:.0f}s)", flush=True)
    post = learner.classify(y, w, train, val, held)
    idx, post_all = held, post
    if model == cfg["regime"]["classifier"]:
        oof = learner.crossfit(y, w, train, r.family.to_numpy()[train] + ":" + r.theta.to_numpy()[train].astype(str),
                               cfg["crossfit_folds"])
        if oof is not None:
            idx, post_all = np.concatenate([train, held]), np.vstack([oof, post])
            print(f"classify {model}: out-of-fold train predictions ({time.time() - t0:.0f}s)", flush=True)
    out = unit_dir(cfg, "classify", model)
    save_predictions(out / "predictions.npz", case_id=r.index.to_numpy(str)[idx], split=split[idx],
                     posterior=post_all.astype(np.float32), classes=np.array(classes))
    test = split[held] == "test"
    report = {"task": "classify", "model": model, "spec": cfg["models"][model], "n_train": int(len(train)),
              "seconds": round(time.time() - t0), **score_classifier(post[test], fam[held][test], classes, cfg)}
    finish(cfg, config_path, out, learner, report)


def score_classifier(post: np.ndarray, fam: np.ndarray, classes: list[str], cfg: dict) -> dict:
    call = np.array(classes)[post.argmax(1)]
    y = np.array([classes.index(f) for f in fam])
    w = sample_weights(fam, cfg)
    p_true = np.clip(post[np.arange(len(y)), y], 1e-12, None)
    return {"accuracy": float(np.average(call == fam, weights=w)),
            "nll": float(np.average(-np.log(p_true), weights=w)),
            "recall": {f: float((call[fam == f] == f).mean()) for f in classes},
            "detected": {f: float((call[fam == f] != "poisson").mean()) for f in classes}}


def estimate(cfg: dict, config_path: str, model: str, family: str) -> None:
    t0 = time.time()
    r = load_rows(cfg)
    split, fam = r.split.to_numpy(), r.family.to_numpy()
    targets = TARGETS[family]
    train, val = (np.flatnonzero((split == s) & (fam == family)) for s in ("train", "val"))
    held = np.flatnonzero(split != "train")                      # every family: the pipeline may route here
    learner = learners.make(cfg, model, r)
    print(f"estimate {family} {model}: {len(train)} train rows, input ready ({time.time() - t0:.0f}s)", flush=True)
    logs = learner.regress(targets, train, val, held)
    if cfg["clip_to_train_range"]:
        Y = np.log(r[targets].to_numpy(float)[train])
        logs = np.clip(logs, Y.min(0), Y.max(0))
    out = unit_dir(cfg, "estimate", model, family)
    save_predictions(out / "predictions.npz", case_id=r.index.to_numpy(str)[held], split=split[held],
                     theta_hat=np.exp(logs), targets=np.array(targets), family=np.array(family))
    mine = (split[held] == "test") & (fam[held] == family)
    y = np.log(r[targets].to_numpy(float)[held][mine])
    rmse = np.sqrt(((logs[mine] - y) ** 2).mean(0))
    report = {"task": "estimate", "model": model, "family": family, "spec": cfg["models"][model],
              "n_train": int(len(train)), "seconds": round(time.time() - t0), "n_test": int(mine.sum()),
              "rmse_log": dict(zip(targets, rmse.round(4).tolist())),
              "rmse_over_sd": dict(zip(targets, (rmse / y.std(0)).round(4).tolist())),
              "mean_rmse_over_sd": float((rmse / y.std(0)).mean())}
    finish(cfg, config_path, out, learner, report)


def finish(cfg, config_path, out, learner, report) -> None:
    for fname, obj in learner.artifacts().items():
        if fname.endswith(".pt"):
            import torch
            torch.save(obj, out / fname)
        else:
            import joblib
            joblib.dump(obj, out / fname, compress=3)
    save_config(config_path, out)
    write_json(out / "report.json", report)                     # last: its existence marks the unit done
    head = report.get("accuracy", report.get("mean_rmse_over_sd"))
    print(f"-> {out}  ({report['seconds']}s, {'accuracy' if 'accuracy' in report else 'mean RMSE/sd'} {head:.4f})", flush=True)


def run(cfg, config_path, task, model, family, force) -> None:
    if not force and done(cfg, task, model, family):
        print(f"{task} {model} {family or ''}: done (--force to retrain)")
        return
    if task == "classify":
        classify(cfg, config_path, model)
    else:
        estimate(cfg, config_path, model, family)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    config_arg(p)
    sub = p.add_subparsers(dest="cmd", required=True)
    ls = sub.add_parser("list")
    ls.add_argument("--kind", choices=["cpu", "gpu"])
    ls.add_argument("--todo", action="store_true", help="only units without a report.json")
    u = sub.add_parser("unit")
    u.add_argument("--kind", choices=["cpu", "gpu"], required=True)
    u.add_argument("--index", type=int, required=True)
    c = sub.add_parser("classify")
    c.add_argument("--model", required=True)
    e = sub.add_parser("estimate")
    e.add_argument("--model", required=True)
    e.add_argument("--family", required=True)
    for s in (u, c, e):
        s.add_argument("--force", action="store_true")
    args = p.parse_args(argv)
    cfg = load_config(args.config)

    if args.cmd == "list":
        for i, (task, model, family) in enumerate(units(cfg, args.kind)):
            if not (args.todo and done(cfg, task, model, family)):
                print(i, task, model, family or "")
        return
    if args.cmd == "unit":
        task, model, family = units(cfg, args.kind)[args.index]
    else:
        task, model, family = args.cmd, args.model, getattr(args, "family", None)
        if model not in cfg["models"]:
            raise SystemExit(f"unknown model `{model}` (config models: {', '.join(cfg['models'])})")
    run(cfg, args.config, task, model, family, args.force)


if __name__ == "__main__":
    main()
