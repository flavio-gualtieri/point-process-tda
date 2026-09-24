#!/usr/bin/env python3
"""Stage 2 of the cascade: which family, within the group stage 1 routed the cloud to?

    python cascade/stage2.py [--config ...]           # after stage1.py train

One classifier per non-poisson group (clustered: thomas | nested | lgcp, repulsive: matern2), each
with a `reject` class when config stage2.reject is true.

Training set (config stage2.training):
  cascade  train clouds that stage 1 ROUTES to this group, by its out-of-fold predictions. So the
           training set holds what the group's classifier will really receive: the group's own
           families minus whatever stage 1 called poisson (the near-CSR ones), plus misrouted
           clouds from elsewhere (mostly near-CSR poisson), which are labelled `reject`.
  oracle   train clouds whose TRUE family is in the group, however stage 1 routed them.
  regime   ALL train clouds of the group's families past stage 1's regime boundary, plus reject
           candidates weighted by how often stage 1 sends clouds like them here. Needs only
           held-out stage-1 predictions, so it also works with the neural stage 1. See regime.py.
Sample weights are the prior's, computed over the whole train split BEFORE routing, so the routed
set keeps the mix that stage 1 actually sends (re-balancing after routing would undo the filter).

Predictions are written for every cloud routed to the group, in every split: train rows out-of-
fold (they are the training set, and stage 3's cascade training routes on them), val/test rows
and routed train rows outside the training set from the full fit.

Output  cascade/results/<run>/stage2/<group>.npz   case_id, split, posterior, classes
        cascade/results/<run>/stage2/report.json    per group: test accuracy on routed clouds, calls
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from common import (CLASSES, GROUPS, REGIME, REJECT, Data, config_arg, crossfit, fit, load_config,
                    make_classifier, proba, run_dir, sample_weights, save_predictions, write_json)
from pipeline import route_group, stage_dir
from regime import EPS, RoutingCurve


def train_group(group: str, d: Data, routed: np.ndarray, w_all: np.ndarray, cfg: dict,
                curve: RoutingCurve | None) -> tuple:
    c2 = cfg["stage2"]
    classes = GROUPS[group] + ([REJECT] if c2["reject"] else [])
    fam = d.family.to_numpy()
    in_group = np.array([REGIME[f] == group for f in fam])
    tr = d.split == "train"
    w = w_all.copy()
    label_as_family = in_group
    if c2["training"] == "cascade":
        fit_rows = tr & routed & (in_group | c2["reject"])
    elif c2["training"] == "oracle":
        fit_rows = tr & in_group
    elif c2["training"] == "regime":
        resolved = in_group & (curve.own(fam, d.rows) >= cfg["regime"]["threshold"])
        label_as_family = resolved
        fit_rows = tr & resolved
        if c2["reject"]:
            pi = curve.pi(fam, d.rows, group)
            if cfg["regime"]["reject_weight"] == "routing_probability":
                w = np.where(resolved, w, w * pi)
            elif cfg["regime"]["reject_weight"] != "uniform":
                raise SystemExit(f"unknown regime.reject_weight `{cfg['regime']['reject_weight']}`")
            fit_rows |= tr & ~resolved & (pi >= EPS)
    else:
        raise SystemExit(f"unknown stage2.training `{c2['training']}`")
    y = np.where(label_as_family, [classes.index(f) if f in classes else -1 for f in fam],
                 classes.index(REJECT) if c2["reject"] else -1)

    posterior = np.zeros((len(fam), len(classes)))
    if len(classes) == 1:                         # single family and no reject: nothing to learn
        posterior[:] = 1.0
    else:
        model = fit(make_classifier(c2["model"]), d.X[fit_rows], y[fit_rows], w[fit_rows])
        posterior[routed] = proba(model, d.X[routed], len(classes))
        if c2["training"] == "cascade":           # out-of-fold train rows, for stage 3's cascade mode
            posterior[fit_rows] = crossfit(c2["model"], d.X[fit_rows], y[fit_rows], w[fit_rows],
                                           d.rows.theta.to_numpy()[fit_rows], c2["crossfit_folds"])
    reject_share = float(w[fit_rows][y[fit_rows] == classes.index(REJECT)].sum() / w[fit_rows].sum()) \
        if c2["reject"] else 0.0
    return classes, posterior[routed], int(fit_rows.sum()), reject_share


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    config_arg(p)
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    out = run_dir(cfg, "stage2", args.config)

    d = Data.load()
    group_hat = route_group(cfg).reindex(d.case_id).group_hat.to_numpy()   # NaN: no stage-1 row
    curve = (RoutingCurve(cfg, stage_dir(cfg, "stage1") / "predictions.npz", d.rows)
             if cfg["stage2"]["training"] == "regime" else None)
    w_all = np.zeros(len(d.case_id))
    for split in ("train", "val", "test"):          # the prior's weights, per split, BEFORE routing
        s = d.split == split
        w_all[s] = sample_weights(d.family[s])

    report = {}
    for group in CLASSES[1:]:
        t0 = time.time()
        routed = group_hat == group
        classes, posterior, n_fit, reject_share = train_group(group, d, routed, w_all, cfg, curve)
        save_predictions(out / f"{group}.npz", d.case_id[routed], d.split[routed], posterior, classes)

        te = d.split[routed] == "test"
        fam = d.family.to_numpy()[routed][te]
        truth = np.array([f if REGIME[f] == group else REJECT for f in fam])
        pred = np.array(classes)[posterior[te].argmax(1)]
        w = w_all[routed][te]
        report[group] = {
            "classes": classes, "n_fit": n_fit, "reject_weight_share": reject_share,
            "n_test_routed": int(te.sum()),
            "test_accuracy": float(np.average(pred == truth, weights=w)),
            "calls": {f: pd.Series(pred[fam == f]).value_counts(normalize=True).round(4).to_dict()
                      for f in np.unique(fam)}}
        print(f"{group:10s} fit on {n_fit} (reject {reject_share:.0%} of weight), {te.sum()} test clouds routed, "
              f"acc {report[group]['test_accuracy']:.4f}  ({time.time() - t0:.0f}s)", flush=True)
        for f, calls in report[group]["calls"].items():
            print(f"   {f:8s} " + "  ".join(f"{k}:{v:.3f}" for k, v in sorted(calls.items())))
    if curve is not None:
        report["regime_boundaries"] = curve.boundaries()
        for f, b in report["regime_boundaries"].items():
            print(f"   boundary {f:8s} {b['coordinate']} >= {b['boundary']}")
    write_json(out / "report.json", report)


if __name__ == "__main__":
    main()
