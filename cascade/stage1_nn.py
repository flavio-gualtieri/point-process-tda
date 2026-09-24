#!/usr/bin/env python3
"""Stage 1 with cloudforger's network (cascade/nn.py), trained directly on the three groups.

    CONFIG=cascade/configs/nn_curves.yaml sbatch cascade/stage1_nn.sh
    python cascade/stage1_nn.py --config cascade/configs/nn_curves.yaml

Model: the `models:` entry named by stage1.nn_model (default `nn`, the curves arm L+F+G+J on the
fixed grid). Labels are
the family GROUP; the cross-entropy is weighted to the balanced-groups prior (the bank has 1:3:1
patterns per group, so `clustered` gets weight 1/3). Output is stage1.py's predictions.npz for VAL
and TEST rows only -- no out-of-fold train predictions, so this stage 1 is for comparison
(stage1.py compare), not for fitting the regime cutoffs.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn as tnn

import nn
import stage1
from common import CLASSES, GROUPS, REGIME, config_arg, load_config, manifest, run_dir, save_predictions, write_json


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    config_arg(p)
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    nn_cfg = nn.spec(cfg, cfg["stage1"]["nn_model"])
    out = run_dir(cfg, "stage1", args.config)

    t0 = time.time()
    dataset, n_tags, arm = nn.build(nn_cfg)
    m = dataset.manifest
    y = m["family"].map(lambda f: CLASSES.index(REGIME[f])).to_numpy(np.int64)
    weight = torch.tensor([1.0 / len(GROUPS[c]) for c in CLASSES], dtype=torch.float32)
    loss_fn = tnn.CrossEntropyLoss(weight=(weight / weight.mean()).to(nn.DEVICE))
    tr, va, te = (dataset.index(s) for s in ("train", "val", "test"))
    model, fit = nn.train(dataset, n_tags, nn_cfg, y, tr, va, loss_fn, len(CLASSES), f"{cfg['name']}/stage1_nn")

    held = np.concatenate([va, te])
    posterior = nn.softmax(nn.predict(model, dataset, nn_cfg, held))
    case_id, split = m["case_id"].to_numpy(str)[held], m["split"].to_numpy(str)[held]

    rows = manifest()
    rep = stage1.evaluate(case_id[split == "test"], posterior[split == "test"], rows)
    rep |= {"val_balanced_accuracy": stage1.evaluate(case_id[split == "val"], posterior[split == "val"],
                                                     rows)["balanced_accuracy"],
            "model": "nn", "input": arm, "nn": nn_cfg, "best_epoch": fit["best_epoch"],
            "epochs_run": fit["epochs_run"], "n_train": int(len(tr)), "seconds": round(time.time() - t0)}
    stage1.show(f"{cfg['name']}/stage1 (nn {arm})", rep)
    save_predictions(out / "predictions.npz", case_id, split, posterior, CLASSES)
    torch.save(model.state_dict(), out / "model.pt")
    write_json(out / "report.json", rep)


if __name__ == "__main__":
    main()
