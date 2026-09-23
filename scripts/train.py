#!/usr/bin/env python3
"""Train one model on the simulated sweep and save its per-pattern test predictions.

    # 5-way family classification, DTM k=10, H0 + H1
    python scripts/train.py --task classify --filtration dtm_k10 --dims 0,1 --seed 1

    # the classical arm: summary-function curves instead of diagrams, same everything else
    python scripts/train.py --task classify --curves L,F,G,J --grid sqrtn_u2 --seed 1

    # per-function grids: L on the literature's r axis, F/G/J on the sqrt(n) axis they vary over
    python scripts/train.py --task classify --curves L@fixed,F,G,J --grid sqrtn_u2 --seed 1

    # parameter estimation for nested Thomas, rips H0 (a 1-D image: rips births are all 0)
    python scripts/train.py --task params --family nested --filtration rips --dims 0 --seed 1

    # the multi-k arm: several filtrations through one shared encoder
    python scripts/train.py --task classify --filtration dtm_k5,dtm_k10,dtm_k15 --dims 0,1 --seed 1

    # the PersLay arm: the same diagrams, vectorized by the network instead of rasterized
    python scripts/train.py --task classify --filtration dtm_k10 --dims 0,1 --perslay --seed 1

Writes results/<task>/<group>/<filtrations>/h<dims>/seed_<seed>/ (perslay_h<dims> for that arm,
so the two vectorizations of one filtration sit side by side):

    predictions.npz  case_id, y_true, y_pred (+ posterior when classifying) for every TEST pattern
    run.json         the arguments, the fitted vectorizer's parameters, target transform, losses,
                     git stamp
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
from cloudforger.vectorization.perslay import OPS, PersLay               # noqa: E402

RESULTS = ROOT / "results"


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", choices=["classify", "params"], required=True)
    p.add_argument("--family", help="params: which family to estimate for (classify uses all five)")
    p.add_argument("--filtration",
                   help="PH arm: comma-separated tags as under data/featurization/<family>/ (rips, dtm_k5...)")
    p.add_argument("--dims", default="0,1", help="PH arm: homology dimensions, comma-separated")
    p.add_argument("--curves", help="classical arm: comma-separated functions, each optionally with "
                                    "its own grid (L@fixed,F,G,J)")
    p.add_argument("--grid", default="sqrtn_u2", help="classical arm: grid for curves that name none")
    p.add_argument("--curves-separate", action="store_true",
                   help="classical arm: one encoder per curve even when they share a grid "
                        "(curves on different grids always get one each)")
    p.add_argument("--seed", default="1", help="one seed, or several ('1,2,3'): the features are "
                                                "built once and every seed trained from them")
    p.add_argument("--targets", help="params: comma-separated manifest columns (default: the family's own)")
    p.add_argument("--perslay", action="store_true",
                   help="PH arm: learn the vectorization (PersLay) instead of rasterizing into "
                        "persistence images; --resolution/--sigma-pixels/--image-transform do not apply")
    p.add_argument("--perslay-points", type=int, default=64,
                   help="PersLay: number of learned point transformations (the vectorization's width)")
    p.add_argument("--perslay-op", choices=list(OPS), default="sum",
                   help="PersLay: permutation-invariant pooling over a diagram's points")
    p.add_argument("--max-points", type=int, default=1024,
                   help="PersLay: hard cap on a padded diagram's length, whatever --coverage asks for")
    p.add_argument("--resolution", type=int, default=64)
    p.add_argument("--sigma-pixels", type=float, default=1.0)
    p.add_argument("--coverage", type=float, default=0.99)
    p.add_argument("--raw-coords", action="store_true", help="do not rescale diagrams by sqrt(n)")
    p.add_argument("--raw-mass", action="store_true", help="do not divide images by n")
    p.add_argument("--image-transform", choices=["sqrt", "none"], default="sqrt",
                   help="variance-stabilizing transform applied before the per-channel z-score")
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
    if bool(args.filtration) == bool(args.curves):
        p.error("pass exactly one of --filtration (PH arm) or --curves (classical arm)")
    if args.perslay and not args.filtration:
        p.error("--perslay vectorizes diagrams; it needs --filtration, not --curves")
    return args


def run_id(args) -> tuple[str, str, str]:
    """(group, features, variant): `features` is what was fed in and `variant` how, whichever arm
    produced it -- dtm_k10/h01 for the PH arm, L+F+G+J/sqrtn_u2 for the classical one."""
    group = args.family if args.task == "params" else "all"
    if args.curves:
        curves = D.parse_curves(args.curves, args.grid)
        grids = [grid for _, grid in curves]
        # variant carries the grids in curve order, so features + variant reconstruct the spec
        return (group, "+".join(name for name, _ in curves),
                grids[0] if len(set(grids)) == 1 else "+".join(grids))
    dims = "h" + "".join(args.dims.split(","))
    return group, args.filtration.replace(",", "+"), f"perslay_{dims}" if args.perslay else dims


def output_dir(args, seed: int) -> Path:
    """results/<task>/<group>/<features>/<variant>/seed_<n>."""
    if args.out:
        return args.out
    return RESULTS.joinpath(args.task, *run_id(args), f"seed_{seed}")


def main(argv=None) -> None:
    args = parse_args(argv)
    seeds = [int(s) for s in str(args.seed).split(",")]
    todo = [s for s in seeds if args.force or not (output_dir(args, s) / "run.json").exists()]
    for seed in seeds:
        if seed not in todo:
            print(f"{output_dir(args, seed)}/run.json exists; --force to retrain")
    if not todo:
        return

    families = list(D.FAMILIES) if args.task == "classify" else [args.family]
    tags = args.filtration.split(",") if args.filtration else []
    dims = [int(d) for d in args.dims.split(",")]
    scaling = Scaling(coords="none" if args.raw_coords else "sqrt_n", density=not args.raw_mass)
    run = f"{args.task}/{'/'.join(run_id(args))}"

    print(f"[{run}] building features once for seed(s) {','.join(map(str, todo))}", flush=True)
    if args.curves:
        curves = D.parse_curves(args.curves, args.grid)
        dataset = D.build_curves(families, curves, stack=False if args.curves_separate else None)
    elif args.perslay:
        dataset = D.build_diagrams(families, tags, dims, coverage=args.coverage,
                                   max_points=args.max_points, scaling=scaling)
    else:
        dataset = D.build(families, tags, dims, resolution=args.resolution, sigma_pixels=args.sigma_pixels,
                          coverage=args.coverage, scaling=scaling, transform=args.image_transform)
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
    print(f"[{run}] {len(manifest)} patterns: {len(train_idx)} train, {len(val_idx)} val, "
          f"{len(test_idx)} test; {n_outputs} outputs", flush=True)

    def loader(index, shuffle):
        return DataLoader(D.Rows(dataset, y, index), batch_size=args.batch_size, shuffle=shuffle)

    keys = sorted(dataset.images)
    n_tags = len(tags) if args.filtration else 1   # PH: one pass per filtration; curves: one pass

    def perslay_encoder(rank, channels):
        return PersLay(embedding_dim=args.embedding_dim, n_transforms=args.perslay_points,
                       op=args.perslay_op, dropout=args.dropout, in_channels=channels)

    # Features are seed-independent (the split is fixed by theta, and every fit -- imager box, pixel
    # z-score, target transform -- uses train rows only), so they are built once above and each seed
    # only re-initializes and retrains the network.
    for seed in todo:
        tag = f"{run}/s{seed}"
        out = output_dir(args, seed)
        torch.manual_seed(seed)
        model = PHNet(
            ranks=[dataset.images[k].ndim - 2 for k in keys],
            n_tags=n_tags,
            channels=[dataset.images[k].shape[1] // n_tags for k in keys],
            n_covariates=dataset.covariates.shape[1], n_outputs=n_outputs,
            embedding_dim=args.embedding_dim,
            conv_channels=tuple(int(c) for c in args.conv_channels.split(",")),
            dropout=args.dropout,
            encoder=perslay_encoder if args.perslay else None,
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
                "args": {**{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
                     "seed": seed},
            "families": families, "tags": tags, "dims": dims if not args.curves else [],
            "curves": D.parse_curves(args.curves, args.grid) if args.curves else [],
            "curves_stacked": bool(args.curves) and len(dataset.images) == 1 and len(args.curves.split(",")) > 1,
            "n_train": len(train_idx), "n_val": len(val_idx), "n_test": len(test_idx),
            "imagers": {f"{t}_h{d}": im.params for (t, d), im in dataset.imagers.items()},
            "image_transform": args.image_transform,
            "targets": target_norm, "labels": labels,
            "best_val_loss": fit["best_val_loss"], "best_epoch": fit["best_epoch"],
            "epochs_run": fit["epochs_run"], "test_loss": test_loss,
            "test_accuracy": test_acc if args.task == "classify" else None,
            "history": fit["history"], "provenance": provenance_stamp(),
        }, indent=2, default=float))
        print(f"[{tag}] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
