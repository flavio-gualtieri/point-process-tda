"""cloudforger's PHNet as a cascade model, for any subset of bank rows.

Everything network-side is cloudforger's, imported read-only and used as scripts/train.py uses it:
the input builders (curves or diagrams, statistics fitted on train rows), PHNet, PersLay (including
the z-scored variant, calibrated once on training rows before fitting, as --perslay-norm zscore),
the AdamW + early-stopping loop. What the cascade adds is only WHICH rows train (a component's
in-regime clouds, plus reject examples) and which labels or targets they carry.

A model spec is the config's `nn:` defaults overridden by one entry of `models:`, e.g.
    {kind: nn, curves: "L@fixed,F@fixed,G@fixed,J@fixed"}                       the curves arm
    {kind: nn, filtration: dtm_k10, dims: "0,1", perslay: true, perslay_norm: zscore}

The input is built once over all five families, so one fitted network can predict for any cloud
the assembled pipeline routes to it. Its normalisation statistics therefore use every family's
train rows (a feature scaling, never a label), while the network itself sees only its own rows.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as tnn
from torch.utils.data import DataLoader

import common  # noqa: F401  (puts src/ on the path)
from cloudforger.training import data as D, train as T
from cloudforger.training.model import PHNet
from cloudforger.vectorization.persistence_images import Scaling
from cloudforger.vectorization.perslay import PersLay, calibrate

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CALIBRATION_ROWS = 8192          # scripts/train.py --calibration-rows default


def spec(cfg: dict, name: str) -> dict:
    """The config's `nn:` defaults overridden by models[name]."""
    return {**cfg["nn"], **cfg["models"][name]}


def build(s: dict):
    """(dataset over all five families in cloudforger's manifest order, n_tags, input description)."""
    families = list(D.FAMILIES)
    if s.get("curves"):
        return D.build_curves(families, D.parse_curves(s["curves"], "sqrtn_u2")), 1, s["curves"]
    tags = s["filtration"].split(",")
    dims = [int(d) for d in str(s["dims"]).split(",")]
    make = D.build_diagrams if s.get("perslay") else D.build
    arm = f"{s['filtration']} h{s['dims']}" + (f" perslay({s.get('perslay_norm', 'none')})" if s.get("perslay") else "")
    return make(families, tags, dims, scaling=Scaling(coords="sqrt_n", density=True)), len(tags), arm


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


def weighted_ce(y_train: np.ndarray, n_classes: int) -> tnn.Module:
    """Cross-entropy weighted so every class has equal total weight on the training rows."""
    counts = np.bincount(y_train, minlength=n_classes).astype(float)
    w = torch.tensor(1.0 / np.maximum(counts, 1), dtype=torch.float32)
    return tnn.CrossEntropyLoss(weight=(w / w.mean()).to(DEVICE))


def softmax(logits: np.ndarray) -> np.ndarray:
    return torch.softmax(torch.from_numpy(logits), dim=1).numpy()
