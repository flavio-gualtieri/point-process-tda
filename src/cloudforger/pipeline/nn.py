"""cloudforger's PHNet as a pipeline learner: input builders, training and prediction.

Everything network-side is cloudforger.training's: the input builders (curves or diagrams,
statistics fitted on train rows), PHNet, PersLay (including the z-scored variant, calibrated once on
training rows before fitting), the AdamW + early-stopping loop. What this adds is only WHICH rows
train and which labels or targets they carry.

A model spec is the config's `nn:` defaults overridden by one entry of `models:`, e.g.
    {learner: nn, curves: "L@fixed,F@fixed,G@fixed,J@fixed"}                     the curves arm
    {learner: nn, filtration: dtm_k10, dims: "0,1", perslay: true, perslay_norm: zscore}
    {learner: nn, curves: "...", filtration: alpha_diameter, dims: "0,1"}         curves + PH, fused

The input is built once over every configured family, so one fitted network can predict for any
cloud the pipeline routes to it. Its normalisation statistics therefore use every family's train
rows (a feature scaling, never a label), while the network itself sees only its own rows.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as tnn
from torch.utils.data import DataLoader

from ..training import data as D, train as T
from ..training.model import PHNet
from ..vectorization.persistence_images import Scaling
from ..vectorization.perslay import PersLay, calibrate

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CALIBRATION_ROWS = 8192          # training rows the PersLay z-score is calibrated on


def build(s: dict, families: list[str]):
    """(dataset over `families` in manifest order, n_tags, input description).

    `curves` alone is the curves arm, `filtration` alone the diagram arm; both together fuse them:
    the persistence images and the curves each get their own encoder and meet at the head, as H0 and
    H1 do. Fusion takes one filtration (PHNet passes every key through its encoder n_tags times) and
    persistence images, not PersLay (train() gives every key the same kind of encoder)."""
    curves = D.build_curves(families, D.parse_curves(s["curves"], "sqrtn_u2")) if s.get("curves") else None
    if not s.get("filtration"):
        return curves, 1, s["curves"]
    tags = s["filtration"].split(",")
    dims = [int(d) for d in str(s["dims"]).split(",")]
    make = D.build_diagrams if s.get("perslay") else D.build
    arm = f"{s['filtration']} h{s['dims']}" + (f" perslay({s.get('perslay_norm', 'none')})" if s.get("perslay") else "")
    if curves is not None and (len(tags) != 1 or s.get("perslay")):
        raise ValueError(f"fusion needs one filtration and persistence images, got {arm}")
    data = make(families, tags, dims, scaling=Scaling(coords="sqrt_n", density=True))
    if curves is None:
        return data, len(tags), arm
    # string keys throughout: the image and curve blocks are ordered by sorted(key) in train() and Rows
    data.images = {f"ph_h{d}": v for d, v in data.images.items()} | curves.images
    return data, 1, f"{arm} + {s['curves']}"


def _loader(dataset, y, index, s, shuffle):
    return DataLoader(D.Rows(dataset, y, index), batch_size=s["batch_size"], shuffle=shuffle)


def train(dataset, n_tags: int, s: dict, y: np.ndarray, train_idx, val_idx, loss_fn,
          n_outputs: int, tag: str) -> tuple[tnn.Module, dict]:
    keys = sorted(dataset.images)
    torch.manual_seed(s["seed"])
    perslay = s.get("perslay") and not s.get("curves")
    norm = s.get("perslay_norm", "none")
    model = PHNet(ranks=[dataset.images[k].ndim - 2 for k in keys], n_tags=n_tags,
                  channels=[dataset.images[k].shape[1] // n_tags for k in keys],
                  n_covariates=dataset.covariates.shape[1], n_outputs=n_outputs,
                  encoder=(lambda rank, ch: PersLay(in_channels=ch, norm=norm)) if perslay else None).to(DEVICE)
    if perslay and norm == "zscore":
        n_cal = min(CALIBRATION_ROWS, len(train_idx))
        cal = np.asarray(train_idx)[np.linspace(0, len(train_idx) - 1, n_cal).round().astype(int)]
        calibrate(model, lambda: T.predict(model, _loader(dataset, y, cal, s, False), DEVICE))
    fit = T.fit(model, _loader(dataset, y, train_idx, s, True), _loader(dataset, y, val_idx, s, False),
                loss_fn, DEVICE, lr=s["lr"], epochs=s["epochs"], patience=s["patience"], tag=tag)
    return model, fit


def predict(model, dataset, s: dict, index: np.ndarray) -> np.ndarray:
    """Raw outputs (logits / standardized targets) for dataset rows `index`."""
    dummy = np.zeros(len(dataset.manifest), np.float32)
    return T.predict(model, _loader(dataset, dummy, index, s, False), DEVICE)


def softmax(logits: np.ndarray) -> np.ndarray:
    return torch.softmax(torch.from_numpy(logits), dim=1).numpy()
