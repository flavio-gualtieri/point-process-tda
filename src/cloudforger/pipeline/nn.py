"""cloudforger's PHNet as a pipeline learner: input builders, training and prediction.

Everything network-side is cloudforger.training's: the input builders (curves or diagrams,
statistics fitted on train rows), PHNet, PersLay (including the z-scored variant, calibrated once on
training rows before fitting), the AdamW + early-stopping loop. What this adds is only WHICH rows
train and which labels or targets they carry.

A model spec is the config's `nn:` defaults overridden by one entry of `models:`, e.g.
    {learner: nn, curves: "L@fixed,F@fixed,G@fixed,J@fixed"}                     the curves arm
    {learner: nn, filtration: dtm_k10, dims: "0,1", perslay: true, perslay_norm: zscore}
    {learner: nn, curves: "...", filtration: alpha_diameter, dims: "0,1"}         curves + PH, fused
    {learner: nn, curves: "...", filtration: "alpha_diameter,dtm_k10", dims: "0,1"}   ... on two filtrations

The input is built once over every configured family, so one fitted network can predict for any
cloud the pipeline routes to it. Its normalisation statistics therefore use every family's train
rows (a feature scaling, never a label), while the network itself sees only its own rows.
"""

from __future__ import annotations

import hashlib
import json
import time

import numpy as np
import torch
import torch.nn as tnn
from torch.utils.data import DataLoader

from ..paths import DATA
from ..training import data as D, train as T
from ..training.model import PHNet
from ..vectorization.persistence_images import Scaling
from ..vectorization.perslay import PersLay, calibrate

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CALIBRATION_ROWS = 8192          # training rows the PersLay z-score is calibrated on


# A built input is a deterministic function of these spec keys, the families and the data root. Building
# it (rasterizing every diagram) can take most of a GPU job's hour, so `prebuild` writes it once and
# build() reads it back: <data>/nn_inputs/<key>/. Nothing reads a cache that was not asked for.
INPUT_KEYS = ("curves", "filtration", "dims", "perslay", "perslay_norm")
CACHE = DATA / "nn_inputs"


def cache_dir(s: dict, families: list[str], unseen: list[str] = ()):
    key = json.dumps({"spec": {k: s.get(k) for k in INPUT_KEYS}, "families": list(families),
                      "unseen": list(unseen)}, sort_keys=True)
    return CACHE / hashlib.sha256(key.encode()).hexdigest()[:16]


def prebuild(s: dict, families: list[str], unseen: list[str] = ()):
    """Build the input and write it to its cache directory; build() then reads it. Image blocks are raw
    .npy (fast to read back); the rest -- manifest, covariates, fitted imagers, norms -- one joblib."""
    import joblib
    out = cache_dir(s, families, unseen)
    if (out / "meta.joblib").exists():
        return out
    t0 = time.time()
    dataset, n_tags, arm = build(s, families, unseen, cache=False)
    tmp = out.with_name(out.name + ".tmp")
    tmp.mkdir(parents=True, exist_ok=True)
    keys = sorted(dataset.images)
    for i, k in enumerate(keys):
        np.save(tmp / f"image_{i}.npy", dataset.images[k])
    joblib.dump({"keys": keys, "manifest": dataset.manifest, "covariates": dataset.covariates,
                 "imagers": dataset.imagers, "norm": dataset.norm, "n_tags": n_tags, "arm": arm,
                 "spec": {k: s.get(k) for k in INPUT_KEYS}, "families": list(families), "unseen": list(unseen)},
                tmp / "meta.joblib")                            # last: its existence marks the cache complete
    tmp.rename(out)
    print(f"  [cache] {arm} -> {out} ({time.time() - t0:.0f}s)", flush=True)
    return out


def _read_cache(path):
    import joblib
    meta = joblib.load(path / "meta.joblib")
    images = {k: np.load(path / f"image_{i}.npy") for i, k in enumerate(meta["keys"])}
    dataset = D.Dataset(meta["manifest"], images, meta["covariates"], meta["imagers"], meta["norm"])
    print(f"  [cache] {meta['arm']} <- {path}", flush=True)
    return dataset, meta["n_tags"], meta["arm"]


def build(s: dict, families: list[str], unseen: list[str] = (), cache: bool = True):
    """(dataset over `families` in manifest order, n_tags, input description). `unseen` families are
    appended after them with no train rows (training.data), for applying a fitted network to them.
    A prebuilt input (prebuild) is read instead of built, unless cache=False.

    `curves` alone is the curves arm, `filtration` alone the diagram arm; both together fuse them:
    the curves and every (filtration, dim) persistence image get their own encoder and meet at the
    head, as H0 and H1 do. Fused filtrations are built one at a time, not stacked as n_tags, since
    their images need not share a shape (alpha H0 is 1-D, DTM H0 2-D). Fusion takes persistence
    images, not PersLay (train() gives every key the same kind of encoder)."""
    if cache and (cache_dir(s, families, unseen) / "meta.joblib").exists():
        return _read_cache(cache_dir(s, families, unseen))
    curves = (D.build_curves(families, D.parse_curves(s["curves"], "sqrtn_u2"), unseen=unseen)
              if s.get("curves") else None)
    if not s.get("filtration"):
        return curves, 1, s["curves"]
    tags = s["filtration"].split(",")
    dims = [int(d) for d in str(s["dims"]).split(",")]
    scaling = Scaling(coords="sqrt_n", density=True)
    arm = f"{s['filtration']} h{s['dims']}" + (f" perslay({s.get('perslay_norm', 'none')})" if s.get("perslay") else "")
    if curves is None:
        make = D.build_diagrams if s.get("perslay") else D.build
        return make(families, tags, dims, scaling=scaling, unseen=unseen), len(tags), arm
    if s.get("perslay"):
        raise ValueError(f"fusion takes persistence images, not PersLay: got {arm}")
    # string keys throughout: the image and curve blocks are ordered by sorted(key) in train() and Rows
    for tag in tags:
        data = D.build(families, [tag], dims, scaling=scaling, unseen=unseen)
        curves.images |= {f"{tag}_h{d}": v for d, v in data.images.items()}
        curves.norm |= {f"{tag}_h{d}": v for d, v in data.norm.items()}
        curves.imagers |= data.imagers
    return curves, 1, f"{arm} + {s['curves']}"


def _loader(dataset, y, index, s, shuffle):
    return DataLoader(D.Rows(dataset, y, index), batch_size=s["batch_size"], shuffle=shuffle)


def network(dataset, n_tags: int, s: dict, n_outputs: int) -> tnn.Module:
    """The untrained PHNet for this input: one encoder per channel key, in sorted(key) order."""
    keys = sorted(dataset.images)
    norm = s.get("perslay_norm", "none")
    perslay = s.get("perslay") and not s.get("curves")
    return PHNet(ranks=[dataset.images[k].ndim - 2 for k in keys], n_tags=n_tags,
                 channels=[dataset.images[k].shape[1] // n_tags for k in keys],
                 n_covariates=dataset.covariates.shape[1], n_outputs=n_outputs,
                 encoder=(lambda rank, ch: PersLay(in_channels=ch, norm=norm)) if perslay else None).to(DEVICE)


def load(saved: dict, dataset, n_tags: int, n_outputs: int) -> tnn.Module:
    """A fitted network from its model.pt (pipeline.learners.NN._keep), on an input built as it was."""
    model = network(dataset, n_tags, saved["spec"], n_outputs)
    model.load_state_dict(saved["state_dict"])
    return model.eval()


def train(dataset, n_tags: int, s: dict, y: np.ndarray, train_idx, val_idx, loss_fn,
          n_outputs: int, tag: str) -> tuple[tnn.Module, dict]:
    torch.manual_seed(s["seed"])
    perslay = s.get("perslay") and not s.get("curves")
    norm = s.get("perslay_norm", "none")
    model = network(dataset, n_tags, s, n_outputs)
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
