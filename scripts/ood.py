#!/usr/bin/env python3
"""Apply the trained classifiers to families the run never trained on: the out-of-distribution test.

    python scripts/ood.py [--config ...] [--family matern1] [--model M ...]

A family is out of distribution when the bank has it and the config's `families` do not (in configs/v2,
Matern I). Every classifier of the config is applied to the family's test clouds through exactly the
statistics it was trained under: a table learner's fitted model on the same feature tables; a network's
saved weights on its input built with the family appended as `unseen` (training.data), so imagers and
z-scores are the training ones. Nothing is retrained; a classifier not trained yet is skipped.

The question is the title's: what does the pipeline call a pattern from a family it does not know,
and -- once estimators are in -- does the fit it then makes reproduce the pattern? This script does
the first half.

Output  <results>/<run>/ood/<family>/classify/<model>/predictions.npz   case_id, split, posterior, classes
        <results>/<run>/ood/<family>/summary.md                         the calls, overall and along the
                                                                        family's regime coordinate
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from cloudforger.paths import BANK
from cloudforger.pipeline import inputs
from cloudforger.pipeline.core import config_arg, load_config, log, rows as load_rows, save_predictions
from cloudforger.pipeline.regime import coordinate
from cloudforger.pipeline.units import run_dir, unit_dir


def unseen_families(cfg: dict) -> list[str]:
    return sorted(p.parent.name for p in BANK.glob("*/manifest.csv") if p.parent.name not in cfg["families"])


def table_posterior(cfg: dict, model: str, family: str, r: pd.DataFrame) -> np.ndarray:
    import joblib
    saved = joblib.load(unit_dir(cfg, "classify", model) / "model.joblib")
    X, columns = inputs.load({**cfg, "families": [family]}, cfg["models"][model]["inputs"], r)
    if columns != saved["columns"]:
        raise SystemExit(f"{model}: the feature tables' columns differ from the ones it was trained on")
    clf = saved["models"]["classifier"]
    out = np.zeros((len(r), len(cfg["families"])), np.float32)
    out[:, clf.classes_] = clf.predict_proba(X)                 # a class absent from training stays 0
    return out


def network_posterior(cfg: dict, model: str, family: str, r: pd.DataFrame) -> np.ndarray:
    import torch
    from cloudforger.pipeline import nn
    saved = torch.load(unit_dir(cfg, "classify", model) / "model.pt", map_location="cpu", weights_only=False)
    dataset, n_tags, _ = nn.build(saved["spec"], cfg["families"], unseen=[family])
    pos = pd.Index(dataset.manifest["case_id"].to_numpy(str)).get_indexer(r.index)
    if (pos < 0).any():
        raise SystemExit(f"{model}: {(pos < 0).sum()} {family} clouds have no network input")
    net = nn.load(saved, dataset, n_tags, len(cfg["families"]))
    return nn.softmax(nn.predict(net, dataset, saved["spec"], pos)).astype(np.float32)


def summary(cfg: dict, family: str, r: pd.DataFrame, calls: dict[str, np.ndarray]) -> str:
    fams = cfg["families"]
    L = [f"# {family} through a pipeline that never saw it: {cfg['name']}", "",
         f"Test clouds: {len(r)}. Share of clouds called each family (argmax posterior).", "",
         "| classifier | " + " | ".join(fams) + " |", "|---|" + "---|" * len(fams)]
    for m, call in calls.items():
        L.append(f"| {m} | " + " | ".join(f"{np.mean(call == f):.2f}" for f in fams) + " |")
    coord = cfg["regime"]["coordinates"].get(family)
    if coord:
        x = coordinate(r, family, coord)
        edges = np.quantile(x, np.linspace(0, 1, 6))
        bins = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, 4)
        L += ["", f"Along {coord} (quintiles of the test clouds; low = near CSR):", "",
              "| classifier | " + coord + " | n | " + " | ".join(fams) + " |", "|---|---|---|" + "---|" * len(fams)]
        for m, call in calls.items():
            for b in range(5):
                k = bins == b
                L.append(f"| {m} | {edges[b]:.3g}-{edges[b + 1]:.3g} | {k.sum()} | "
                         + " | ".join(f"{np.mean(call[k] == f):.2f}" for f in fams) + " |")
    return "\n".join(L) + "\n"


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    config_arg(p)
    p.add_argument("--family", action="append", help="default: every bank family the config does not train on")
    p.add_argument("--model", action="append", help="default: every classifier of the config")
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    for family in args.family or unseen_families(cfg):
        if family in cfg["families"]:
            raise SystemExit(f"{family} is one of the run's families: not out of distribution")
        r = load_rows({**cfg, "families": [family]})
        r = r[r.split == "test"]
        calls = {}
        for model in args.model or cfg["classify"]:
            if not (unit_dir(cfg, "classify", model) / "report.json").exists():
                log(f"{family}: {model} not trained -- skipped")
                continue
            log(f"{family}: {model} on {len(r)} test clouds")
            fn = network_posterior if cfg["models"][model]["learner"] == "nn" else table_posterior
            post = fn(cfg, model, family, r)
            save_predictions(run_dir(cfg, "ood", family, "classify", model) / "predictions.npz",
                             case_id=r.index.to_numpy(str), split=r.split.to_numpy(str), posterior=post,
                             classes=np.array(cfg["families"]))
            calls[model] = np.array(cfg["families"])[post.argmax(1)]
        out = run_dir(cfg, "ood", family) / "summary.md"
        out.write_text(summary(cfg, family, r, calls))
        log(f"-> {out}")


if __name__ == "__main__":
    main()
