#!/usr/bin/env python3
"""Stage 1 with a neural network: the project's PHNet, trained directly on the three groups.

    sbatch cascade/stage1_nn.sh                                     # GPU; config cascade/configs/nn.yaml
    python cascade/stage1_nn.py --config cascade/configs/nn.yaml    # the same, by hand

The network, the diagram loader and the training loop are cloudforger's, imported read-only and
used exactly as scripts/train.py uses them (same vectorization, split, optimizer, early stopping).
Only two things differ: the labels are the family GROUP, and the cross-entropy is class-weighted
to the balanced-groups prior (the bank has 1:3:1 patterns per group, so `clustered` gets weight 1/3).

Output is the same predictions.npz as stage1.py, but for VAL and TEST rows only: there are no
out-of-fold train predictions (K more GPU fits), so downstream stages must use
`training: regime`, which needs only held-out stage-1 predictions. See stage2.py.
Config keys: stage1.nn.{filtration, dims, perslay | curves, seed, epochs, patience, batch_size, lr}.
`curves` (e.g. "L@fixed,F@fixed,G@fixed,J@fixed") selects the classical arm, as scripts/train.py
--curves: the curves from data/classical/ through the same PHNet with a 1-D CNN encoder, and
overrides filtration/dims/perslay.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from common import CLASSES, GROUPS, REGIME, config_arg, load_config, manifest, run_dir, save_predictions, write_json
from cloudforger.training import data as D, train as T
from cloudforger.training.model import PHNet
from cloudforger.vectorization.persistence_images import Scaling
from cloudforger.vectorization.perslay import PersLay

import stage1


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    config_arg(p)
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    nn_cfg = cfg["stage1"]["nn"]
    out = run_dir(cfg, "stage1", args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    scaling = Scaling(coords="sqrt_n", density=True)
    if nn_cfg.get("curves"):                      # the classical arm: one pass, as train.py --curves
        dataset = D.build_curves(list(D.FAMILIES), D.parse_curves(nn_cfg["curves"], "sqrtn_u2"))
        tags, arm = ["curves"], nn_cfg["curves"]
    else:
        tags, arm = nn_cfg["filtration"].split(","), f"{nn_cfg['filtration']} h{nn_cfg['dims']}"
        dims = [int(d) for d in str(nn_cfg["dims"]).split(",")]
        build = D.build_diagrams if nn_cfg["perslay"] else D.build
        dataset = build(list(D.FAMILIES), tags, dims, scaling=scaling)
    m = dataset.manifest
    y = m["family"].map(lambda f: CLASSES.index(REGIME[f])).to_numpy(np.int64)
    weight = torch.tensor([1.0 / len(GROUPS[c]) for c in CLASSES], dtype=torch.float32)
    loss_fn = nn.CrossEntropyLoss(weight=(weight / weight.mean()).to(device))
    tr, va, te = (dataset.index(s) for s in ("train", "val", "test"))

    def loader(index, shuffle):
        return DataLoader(D.Rows(dataset, y, index), batch_size=nn_cfg["batch_size"], shuffle=shuffle)

    keys = sorted(dataset.images)
    torch.manual_seed(nn_cfg["seed"])
    model = PHNet(ranks=[dataset.images[k].ndim - 2 for k in keys], n_tags=len(tags),
                  channels=[dataset.images[k].shape[1] // len(tags) for k in keys],
                  n_covariates=dataset.covariates.shape[1], n_outputs=len(CLASSES),
                  encoder=(lambda rank, ch: PersLay(in_channels=ch))
                  if nn_cfg.get("perslay") and not nn_cfg.get("curves") else None
                  ).to(device)
    t0 = time.time()
    fit = T.fit(model, loader(tr, True), loader(va, False), loss_fn, device, lr=nn_cfg["lr"],
                epochs=nn_cfg["epochs"], patience=nn_cfg["patience"], tag=f"{cfg['name']}/stage1_nn")

    held = np.concatenate([va, te])
    logits = T.predict(model, loader(held, False), device)
    posterior = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    case_id, split = m["case_id"].to_numpy(str)[held], m["split"].to_numpy(str)[held]

    rows = manifest()
    rep = stage1.evaluate(case_id[split == "test"], posterior[split == "test"], rows)
    rep |= {"val_balanced_accuracy": stage1.evaluate(case_id[split == "val"], posterior[split == "val"],
                                                     rows)["balanced_accuracy"],
            "model": "nn", "nn": nn_cfg, "best_epoch": fit["best_epoch"], "epochs_run": fit["epochs_run"],
            "n_train": int(len(tr)), "seconds": round(time.time() - t0)}
    stage1.show(f"{cfg['name']}/stage1 (nn {arm})", rep)
    save_predictions(out / "predictions.npz", case_id, split, posterior, CLASSES)
    torch.save(model.state_dict(), out / "model.pt")
    write_json(out / "report.json", rep)


if __name__ == "__main__":
    main()
