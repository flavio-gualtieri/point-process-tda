#!/usr/bin/env python3
"""Stage 3 of the cascade: the routed family's parameters.

    python cascade/stage3.py [--config ...]           # after stage2.py

One regressor per (family, target) on log(target), for every non-poisson family; poisson is the
MLE nbar_hat = n (pipeline.theta_hat), no model. The loss is squared error on log(target), i.e.
log-MSE, the same objective as the PH arm's params task. Wasserstein never enters training; it is
only the end-to-end score (evaluate.py).

Training set (config stage3.training):
  cascade  train clouds of the family that the pipeline routes to that family (out-of-fold
           routing at both earlier stages), so near-CSR clouds that stage 1 hands to poisson are
           not in it -- the filtering argument of the cascade, applied to inference.
  regime   every train cloud of the family past stage 1's regime boundary (regime.py), whether
           or not that particular cloud was routed right: the cutoff is a property of the regime.
  all      every train cloud of the family.
Misrouted clouds are never training data: they have no true value of this family's parameters.

Predictions are written for every val/test cloud ROUTED to the family, whatever its true family,
clipped to the training targets' range when config stage3.clip_to_train_range is set (the
simulator needs an in-prior theta).

Output  cascade/results/<run>/stage3/<family>.npz   case_id, split, theta_hat (P, T), targets
        cascade/results/<run>/stage3/report.json     RMSE in log units on correctly routed test
                                                     clouds, next to the targets' own s.d.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from common import TARGETS, Data, config_arg, fit, load_config, make_regressor, run_dir, write_json
from pipeline import route_family, stage_dir
from regime import RoutingCurve


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    config_arg(p)
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    c3 = cfg["stage3"]
    out = run_dir(cfg, "stage3", args.config)

    d = Data.load()
    family_hat = route_family(cfg).reindex(d.case_id).family_hat.to_numpy()
    fam = d.family.to_numpy()
    tr, held = d.split == "train", d.split != "train"
    own = (RoutingCurve(cfg, stage_dir(cfg, "stage1") / "predictions.npz", d.rows).own(fam, d.rows)
           if c3["training"] == "regime" else None)

    report = {}
    for family, targets in TARGETS.items():
        if family == "poisson":
            continue
        t0 = time.time()
        if c3["training"] == "cascade":
            fit_rows = tr & (fam == family) & (family_hat == family)
        elif c3["training"] == "regime":
            fit_rows = tr & (fam == family) & (own >= cfg["regime"]["threshold"])
        elif c3["training"] == "all":
            fit_rows = tr & (fam == family)
        else:
            raise SystemExit(f"unknown stage3.training `{c3['training']}`")
        pred_rows = held & (family_hat == family)

        Y = np.log(d.rows.loc[:, targets].to_numpy(float))
        theta = np.zeros((pred_rows.sum(), len(targets)))
        for j in range(len(targets)):
            model = fit(make_regressor(c3["model"]), d.X[fit_rows], Y[fit_rows, j], None)
            theta[:, j] = model.predict(d.X[pred_rows])
        if c3["clip_to_train_range"]:
            theta = np.clip(theta, Y[fit_rows].min(0), Y[fit_rows].max(0))
        np.savez(out / f"{family}.npz", case_id=d.case_id[pred_rows], split=d.split[pred_rows],
                 theta_hat=np.exp(theta), targets=np.array(targets))

        ok = (d.split[pred_rows] == "test") & (fam[pred_rows] == family)
        err = theta[ok] - Y[pred_rows][ok]
        report[family] = {
            "n_fit": int(fit_rows.sum()), "n_routed_held_out": int(pred_rows.sum()),
            "n_test_correct": int(ok.sum()),
            "rmse_log": dict(zip(targets, np.sqrt((err ** 2).mean(0)).round(4).tolist())),
            "target_sd_log": dict(zip(targets, Y[pred_rows][ok].std(0).round(4).tolist()))}
        r = report[family]
        print(f"{family:8s} fit on {r['n_fit']}, {r['n_test_correct']} correctly routed test clouds "
              f"({time.time() - t0:.0f}s)", flush=True)
        for t in targets:
            print(f"   {t:7s} rmse_log {r['rmse_log'][t]:.3f}   (target sd {r['target_sd_log'][t]:.3f})")
    write_json(out / "report.json", report)


if __name__ == "__main__":
    main()
