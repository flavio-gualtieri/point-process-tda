#!/usr/bin/env python3
"""Train one model on the simulated sweep and save its per-pattern test predictions.

    # 5-way family classification, DTM k=10, H0 + H1
    python scripts/train.py --task classify --filtration dtm_k10 --dims 0,1 --seed 1

    # parameter estimation for nested Thomas, rips H0 (a 1-D image: rips births are all 0)
    python scripts/train.py --task params --family nested --filtration rips --dims 0 --seed 1

    # the multi-k arm: several filtrations through one shared encoder
    python scripts/train.py --task classify --filtration dtm_k5,dtm_k10,dtm_k15 --dims 0,1 --seed 1

Writes results/<task>/<group>/<filtrations>/h<dims>/seed_<seed>/:

    predictions.npz  case_id, y_true, y_pred (+ posterior when classifying) for every TEST pattern
    run.json         the arguments, imager parameters, target transform, losses, git stamp
    model.pt         the best-validation weights

Everything per regime is computed afterwards by joining predictions.npz to the manifest on case_id;
no regime enters training. The split is fixed by theta (cloudforger.simulation.split), the same for
every seed, task and filtration, so runs are paired pattern by pattern.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cloudforger.provenance import provenance_stamp                      # noqa: E402
from cloudforger.training import data as D, train as T                   # noqa: E402
from cloudforger.training.model import PHNet                             # noqa: E402
from cloudforger.vectorization.persistence_images import Scaling         # noqa: E402

RESULTS = ROOT / "results"


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", choices=["classify", "params"], required=True)
    p.add_argument("--family", help="params: which family to estimate for (classify uses all five)")
    p.add_argument("--filtration", required=True,
                   help="comma-separated tags as under data/featurization/<family>/ (rips, alpha, dtm_k5...)")
    p.add_argument("--dims", default="0,1", help="homology dimensions, comma-separated")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--targets", help="params: comma-separated manifest columns (default: the family's own)")
    p.add_argument("--resolution", type=int, default=64)
    p.add_argument("--sigma-pixels", type=float, default=1.0)
    p.add_argument("--coverage", type=float, default=0.99)
    p.add_argument("--raw-coords", action="store_true", help="do not rescale diagrams by sqrt(n)")
    p.add_argument("--raw-mass", action="store_true", help="do not divide images by n")
    p.add_argument("--embedding-dim", type=int, default=64)
    p.add_argument("--conv-channels", default="32,64,128")
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=3e-4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", type=Path, help="output directory (default: the results/... path above)")
    p.add_argument("--force", action="store_true", help="retrain even if run.json exists")
    args = p.parse_args(argv)
    if args.task == "params" and not args.family:
        p.error("--task params needs --family")
    if args.task == "classify" and args.family:
        p.error("--task classify uses every family; drop --family")
    return args


def output_dir(args) -> Path:
    if args.out:
        return args.out
    group = args.family if args.task == "params" else "all"
    dims = "".join(args.dims.split(","))
    return RESULTS / args.task / group / args.filtration.replace(",", "+") / f"h{dims}" / f"seed_{args.seed}"


def main(argv=None) -> None:
    args = parse_args(argv)
    out = output_dir(args)
    if (out / "run.json").exists() and not args.force:
        print(f"{out}/run.json exists; --force to retrain")
        return
    torch.manual_seed(args.seed)

    families = list(D.FAMILIES) if args.task == "classify" else [args.family]
    tags = args.filtration.split(",")
    dims = [int(d) for d in args.dims.split(",")]
    scaling = Scaling(coords="none" if args.raw_coords else "sqrt_n", density=not args.raw_mass)
    tag = f"{args.task}/{'+'.join(families) if len(families) == 1 else 'all'}/{args.filtration}/h{args.dims}/s{args.seed}"

    print(f"[{tag}] building images", flush=True)
    dataset = D.build(families, tags, dims, resolution=args.resolution, sigma_pixels=args.sigma_pixels,
                      coverage=args.coverage, scaling=scaling)
    train_idx, val_idx, test_idx = (dataset.index(s) for s in ("train", "val", "test"))

    manifest = dataset.manifest
    if args.task == "classify":
        labels = list(D.FAMILIES)
        y = manifest["family"].map(labels.index).to_numpy(np.int64)
        target_norm = {"classes": labels}
        loss_fn, n_outputs = nn.CrossEntropyLoss(), len(labels)
    else:
        labels = None
        columns = args.targets.split(",") if args.targets else D.TARGETS[args.family]
        y, target_norm = D.targets(manifest, columns, train_idx)
        loss_fn, n_outputs = nn.MSELoss(), len(columns)
    print(f"[{tag}] {len(manifest)} patterns: {len(train_idx)} train, {len(val_idx)} val, "
          f"{len(test_idx)} test; {n_outputs} outputs", flush=True)

    def loader(index, shuffle):
        return DataLoader(D.Rows(dataset, y, index), batch_size=args.batch_size, shuffle=shuffle)

    model = PHNet(
        ranks=[dataset.images[dim].ndim - 2 for dim in sorted(dataset.images)],
        n_tags=len(tags), n_covariates=dataset.covariates.shape[1], n_outputs=n_outputs,
        embedding_dim=args.embedding_dim,
        conv_channels=tuple(int(c) for c in args.conv_channels.split(",")),
        dropout=args.dropout,
    ).to(args.device)

    fit = T.fit(model, loader(train_idx, True), loader(val_idx, False), loss_fn, args.device,
                lr=args.lr, weight_decay=args.weight_decay, epochs=args.epochs,
                patience=args.patience, tag=tag)

    test_loader = loader(test_idx, False)
    test_loss, test_acc = T.evaluate(model, test_loader, loss_fn, args.device)
    outputs = T.predict(model, test_loader, args.device)

    saved = {"case_id": manifest["case_id"].to_numpy(str)[test_idx]}
    if args.task == "classify":
        posterior = torch.softmax(torch.from_numpy(outputs), dim=1).numpy()
        saved |= {"y_true": y[test_idx], "y_pred": outputs.argmax(axis=1), "posterior": posterior}
        print(f"[{tag}] test loss {test_loss:.4f}  accuracy {test_acc:.4f}", flush=True)
    else:
        saved |= {"y_true": D.invert_targets(y[test_idx], target_norm),
                  "y_pred": D.invert_targets(outputs, target_norm),
                  "y_true_std": y[test_idx], "y_pred_std": outputs}
        print(f"[{tag}] test loss {test_loss:.4f} (standardized MSE)", flush=True)

    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "predictions.npz", **saved)
    torch.save(model.state_dict(), out / "model.pt")
    (out / "run.json").write_text(json.dumps({
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "families": families, "tags": tags, "dims": dims,
        "n_train": len(train_idx), "n_val": len(val_idx), "n_test": len(test_idx),
        "imagers": {f"{t}_h{d}": im.params for (t, d), im in dataset.imagers.items()},
        "targets": target_norm, "labels": labels,
        "best_val_loss": fit["best_val_loss"], "best_epoch": fit["best_epoch"],
        "epochs_run": fit["epochs_run"], "test_loss": test_loss,
        "test_accuracy": test_acc if args.task == "classify" else None,
        "history": fit["history"], "provenance": provenance_stamp(),
    }, indent=2, default=float))
    print(f"[{tag}] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
