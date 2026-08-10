# src/cloudforger/experiments/base.py

from __future__ import annotations

import json
import pickle
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from cloudforger.core.splits import train_val_test_indices
from cloudforger.training.splits import train_val_test_split
from cloudforger.training.train import train_one_epoch, evaluate, evaluate_per_target
from cloudforger.models.heads.paramest import ParameterEstimator
from cloudforger.models.heads.classifier import ClassificationHead
from cloudforger.models.single_modal import SingleModalModel
from cloudforger.experiments.common import (
    MultiSourceExperiment,
    prepare_device,
    select_labels,
    fit_label_norm,
    apply_label_norm,
    save_results,
)

# Matches dimensioned methods like "pi_1" / "betti_0" / "betti_cnn_1".
_PERSIST_TOKEN = re.compile(r"^(pi|betti_cnn_weighted|betti_cnn|betti)_(\d+)$")


class CovariateHeadDataset(Dataset):
    """Wrap a base dataset so each sample returns (x, covariate_mean, y)."""

    def __init__(self, base: Dataset, covariates: np.ndarray, dtype=torch.float32):
        self.base = base
        self.covariates = torch.as_tensor(covariates, dtype=dtype)

    def __len__(self):
        return len(self.base)

    def __getattr__(self, name):
        # Preserve attributes like input_dim used by existing experiments.
        return getattr(self.base, name)

    def __getitem__(self, idx):
        x, y = self.base[idx]
        return x, self.covariates[idx], y


def _extract_covariate_means(payload: Any) -> np.ndarray:
    if isinstance(payload, dict) and "covariates" in payload:
        covariates = payload["covariates"]
    elif isinstance(payload, dict) and "clouds" in payload:
        covariates = [c.get("covariates") if isinstance(c, dict) else getattr(c, "covariates", None)
                      for c in payload["clouds"]]
    elif isinstance(payload, list):
        covariates = [c.get("covariates") if isinstance(c, dict) else getattr(c, "covariates", None)
                      for c in payload]
    else:
        raise KeyError(
            "use_covariates=True requires covariates either at payload['covariates'] "
            "or inside each cloud record."
        )

    means = []
    for i, cov in enumerate(covariates):
        if cov is None:
            raise ValueError(f"Missing covariates at sample index {i}")

        arr = np.asarray(cov, dtype=float)
        if arr.ndim != 2:
            raise ValueError(f"Expected covariates with shape (n_points, n_covariates), got {arr.shape}")

        if arr.shape[0] == 0:
            means.append(np.zeros(arr.shape[1], dtype=float))
        else:
            means.append(arr.mean(axis=0))

    return np.asarray(means, dtype=np.float32)


def _combine_head_extra(parts: list[np.ndarray]) -> np.ndarray:
    """Concatenate one or more per-sample (N,) or (N,k) feature arrays into a
    single (N, total_k) array for the encoder-embedding/head merge point."""
    parts = [p if p.ndim > 1 else p[:, None] for p in parts]
    return parts[0] if len(parts) == 1 else np.concatenate(parts, axis=1)


# _select_labels / _normalize_labels_by_name moved to
# cloudforger.experiments.common (as select_labels / normalize_labels_by_name)
# so MultiSourceExperiment subclasses (fusion, pi_multik, pi_multik_fusion)
# can share them too.


# ---------------------------------------------------------------------------
# CLASSIFICATION HELPERS
# ---------------------------------------------------------------------------


def _as_1d_integer_labels(labels: np.ndarray) -> np.ndarray:
    """Validate classification labels and return shape (N,) int64 labels."""
    labels = np.asarray(labels)

    # Permit payloads that store a single target as shape (N, 1).
    if labels.ndim == 2 and labels.shape[1] == 1:
        labels = labels[:, 0]

    if labels.ndim != 1:
        raise ValueError(
            "Classification expects one class label per sample, with shape "
            f"(N,) or (N, 1); got {labels.shape}"
        )

    if not np.all(np.isfinite(labels)):
        raise ValueError("Classification labels contain NaN or infinity")

    # This permits whole-valued floats such as 2.0, but rejects 2.5.
    if not np.all(labels == np.floor(labels)):
        raise ValueError("Classification labels must be integer-valued")

    return labels.astype(np.int64)


def _encode_classification_labels(
    labels: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Fit a contiguous class-index mapping on the training payload."""
    raw_labels = _as_1d_integer_labels(labels)
    classes = np.unique(raw_labels)

    class_to_index = {
        int(raw_class): index
        for index, raw_class in enumerate(classes)
    }

    encoded = np.asarray(
        [class_to_index[int(value)] for value in raw_labels],
        dtype=np.int64,
    )

    return encoded, {
        "kind": "classification",
        "classes": [int(value) for value in classes],
        # Retain these keys temporarily because _save currently expects them.
        "mean": None,
        "std": None,
        "transforms": ["class_index"],
    }


def _apply_classification_encoding(
    labels: np.ndarray,
    label_meta: dict,
) -> np.ndarray:
    """Apply the training class mapping to another payload."""
    raw_labels = _as_1d_integer_labels(labels)

    class_to_index = {
        int(raw_class): index
        for index, raw_class in enumerate(label_meta["classes"])
    }

    unknown = sorted(set(map(int, raw_labels)) - set(class_to_index))
    if unknown:
        raise ValueError(
            f"Labels contain classes not seen during training: {unknown}"
        )

    return np.asarray(
        [class_to_index[int(value)] for value in raw_labels],
        dtype=np.int64,
    )


# ---------------------------------------------------------------------------
# Shared n(x) side-channel (Vihrs 2022 feeds n(x) alongside the curve because
# the curve alone can't recover intensity-related parameters). One
# implementation shared by every experiment that wants it, so all methods in
# a comparison have equal access to it rather than some getting it and
# others not.
# ---------------------------------------------------------------------------

def _resolve_sibling_clouds_path(dataset_path: Path) -> Path:
    """Resolve the clouds.pkl (or adversarial_clouds.pkl) a features/betti/
    images file was derived from, using the "adversarial_" filename prefix
    convention every split already follows, rather than pattern-matching on
    the file's own base name. Two layouts are supported: the flat one
    (clouds.pkl next to the feature file, same directory -- still used by
    not-yet-migrated callers) and cloudforger.paths' nested one
    (data/<process>/<filtration_tag>/<feature>.pkl, with clouds.pkl one
    level up in data/<process>/) -- whichever actually exists wins, checked
    in that order."""
    name = "adversarial_clouds.pkl" if dataset_path.name.startswith("adversarial_") else "clouds.pkl"
    same_dir = dataset_path.parent / name
    if same_dir.exists():
        return same_dir
    return dataset_path.parent.parent / name


def zscore_fit_once(experiment: Any, values: np.ndarray, attr: str) -> np.ndarray:
    """Same fit-once/freeze convention as log_zscore_fit_once, but no log
    step -- use this for values that can legitimately be zero or negative
    (persistence entropy can be exactly 0), where log() would be undefined."""
    if not hasattr(experiment, attr):
        std = float(values.std())
        setattr(experiment, attr, {"mean": float(values.mean()), "std": std if std else 1.0})
    norm = getattr(experiment, attr)
    return ((values - norm["mean"]) / norm["std"]).astype(np.float32)


def log_zscore_fit_once(experiment: Any, values: np.ndarray, attr: str) -> np.ndarray:
    """Log+zscore `values`, fitting mean/std on the first call and freezing
    them on `experiment` for later calls (e.g. the adversarial payload) --
    the same fit-once/apply-frozen convention _normalize_labels_by_name uses
    for the targets."""
    log_v = np.log(values)
    if not hasattr(experiment, attr):
        std = float(log_v.std())
        setattr(experiment, attr, {"mean": float(log_v.mean()), "std": std if std else 1.0})
    norm = getattr(experiment, attr)
    return ((log_v - norm["mean"]) / norm["std"]).astype(np.float32)


def n_points_head_extra(experiment: Any, payload: Any, dataset_path: Path) -> np.ndarray:
    """Log+zscore n(x), joined from the sibling clouds.pkl by the shared
    'seed' field -- for any experiment whose own payload (betti.pkl,
    images.pkl, features.pkl, ...) carries a "seeds" list but not n_points
    itself. Call from Experiment.extract_head_extra as:
        return n_points_head_extra(self, payload, dataset_path)
    """
    clouds_path = _resolve_sibling_clouds_path(dataset_path)
    with open(clouds_path, "rb") as f:
        clouds = pickle.load(f)
    n_points_by_seed = {c["seed"]: c["n_points"] for c in clouds}

    seeds = payload["seeds"]
    n_points = np.array([n_points_by_seed[int(s)] for s in seeds], dtype=np.float64)
    return log_zscore_fit_once(experiment, n_points, attr="_n_points_norm")


def persistence_entropy_head_extra(
    experiment: Any, payload: Any, dims: tuple[int, ...], n_x: np.ndarray
) -> np.ndarray:
    """Join per-dim persistence entropy (payload["persistence_entropy"] is
    always keyed by a single int homology dim -- see
    dtm_experiment/compute_features.py's persistence_entropy) onto n(x), one
    extra z-scored column per dim in `dims`. Must be called with every dim
    the experiment actually trains on: a single `self.hom_dim` lookup breaks
    for tuple/None hom_dim (the "01"-fused and ph_combined methods), since
    the entropy dict is never keyed by a tuple or None -- it would silently
    fall back to n_x-only with no entropy feature at all."""
    entropy_by_dim = payload.get("persistence_entropy", {})
    entropies = [entropy_by_dim.get(d) for d in dims]
    if any(e is None for e in entropies):
        return n_x

    entropy_cols = [
        zscore_fit_once(experiment, np.asarray(e, dtype=np.float64), attr=f"_entropy_norm_{d}")
        for d, e in zip(dims, entropies)
    ]
    return np.stack([n_x, *entropy_cols], axis=1)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

REGISTRY: dict[str, type["Experiment"] | type[MultiSourceExperiment]] = {}


def register(*names: str):
    """Class decorator registering an Experiment (or MultiSourceExperiment)
    under one or more base names -- both live in the same registry, so
    build_experiment()/the train CLI can dispatch on --method without
    caring which shape a given method happens to be."""
    def deco(cls):
        for name in names:
            REGISTRY[name] = cls
        return cls
    return deco


def _parse_method(method: str) -> tuple[str, int | tuple[int, ...] | None]:
    m = _PERSIST_TOKEN.match(method)
    if m:
        digits = m.group(2)
        hom_dim = int(digits) if len(digits) == 1 else tuple(int(c) for c in digits)
        return m.group(1), hom_dim
    return method, None


def build_experiment(cfg: dict) -> "Experiment | MultiSourceExperiment":
    """Construct the Experiment/MultiSourceExperiment for a single concrete
    method string. MultiSourceExperiment subclasses (fusion, pi_multik,
    pi_multik_fusion) take only cfg -- they have no hom_dim concept, unlike
    the persistence-image/Betti-curve family's "pi_1"/"betti_0" methods."""
    base, hom_dim = _parse_method(cfg["method"])
    try:
        cls = REGISTRY[base]
    except KeyError:
        raise ValueError(
            f"Unknown method '{cfg['method']}'. Registered: {sorted(REGISTRY)}; "
            "plus pi_<dim>, betti_<dim>."
        )
    if issubclass(cls, MultiSourceExperiment):
        return cls(cfg)
    return cls(cfg, hom_dim)


# ---------------------------------------------------------------------------
# Shared machinery
# ---------------------------------------------------------------------------

def _load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _normalize_labels(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    log_labels = np.log(labels)
    log_mean = log_labels.mean(axis=0)
    log_std = log_labels.std(axis=0)
    log_std = np.where(log_std == 0, 1.0, log_std)
    return (log_labels - log_mean) / log_std, log_mean, log_std


def _make_loaders(dataset: Dataset, cfg: dict, collate_fn=None):
    train_ds, val_ds, test_ds = train_val_test_split(dataset, seed=cfg["seed"])
    kw = dict(batch_size=cfg["batch_size"], collate_fn=collate_fn)
    return (
        DataLoader(train_ds, **kw, shuffle=True),
        DataLoader(val_ds, **kw),
        DataLoader(test_ds, **kw),
    )


def _train_and_eval(model, loaders, cfg: dict, device: str, tag: str):
    train_loader, val_loader, test_loader = loaders
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    history = {"train_loss": [], "val_loss": []}
    best_val_loss, best_state = float("inf"), None

    for epoch in range(1, cfg["n_epochs"] + 1):
        train_loss, _ = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss, _ = evaluate(model, val_loader, loss_fn, device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(f"[{tag}] Epoch {epoch:3d} | train loss {train_loss:.4f} | val loss {val_loss:.4f}")

    model.load_state_dict(best_state)
    test_loss, _ = evaluate(model, test_loader, loss_fn, device)
    print(f"\n[{tag}] Test loss: {test_loss:.4f}")
    return history, best_state, test_loss


# ---------------------------------------------------------------------------
# The abstraction
# ---------------------------------------------------------------------------

class Experiment(ABC):

    file_key: str
    collate_fn = None  # override on subclasses whose samples have variable size

    def __init__(self, cfg: dict, hom_dim: int | None = None, task_type: "params" | "classification" = "params"):
        self.cfg = cfg
        self.hom_dim = hom_dim  # used by dimensioned experiments; ignored otherwise
        self.task_type = task_type

    # -- identity / metadata ------------------------------------------------

    @property
    @abstractmethod
    def subdir(self) -> str:
        """Result subdirectory and default log tag."""

    @property
    def tag(self) -> str:
        return self.subdir

    @property
    def extra_meta(self) -> dict:
        """Extra fields written into results.json."""
        return {}

    @property
    def encoder_output_dim(self) -> int:
        """Width of the vector build_encoder's module emits, before any
        head-extra features (covariates, n_points, ...) are concatenated on.
        Default: cfg['embedding_dim'], matching every single-encoder
        experiment. Override for encoders that fuse several sub-embeddings
        (e.g. a multi-modal fusion encoder over several data files) into a
        wider vector, so the shared head is sized correctly."""
        return self.cfg["embedding_dim"]

    # -- per-experiment hooks ----------------------------------------------

    def extract_labels(self, payload: Any) -> tuple[np.ndarray, list[str]]:
        """Default: payloads carry their own ``labels`` / ``label_names``."""
        return np.asarray(payload["labels"], dtype=float), list(payload["label_names"])

    def extract_head_extra(self, payload: Any, dataset_path: Path) -> np.ndarray | None:
        """Optional per-sample feature(s) concatenated onto the encoder's
        embedding before the shared regression head — e.g. point count n(x),
        as in Vihrs (2022). Default: None (no extra feature; existing
        experiments that don't override this are unaffected)."""
        return None

    @abstractmethod
    def build_dataset(self, payload: Any, labels: np.ndarray) -> Dataset:
        """Turn the loaded payload + normalized labels into a Dataset."""

    @abstractmethod
    def build_encoder(self, dataset: Dataset) -> nn.Module:
        """Build the encoder that feeds the shared regression head."""

    # -- shared orchestration ----------------------------------------------

    def run(self, dataset_path: Path, output_dir: Path, adversarial_path: Path | None = None) -> dict:
        device = prepare_device(self.cfg["seed"])

        self._dataset_path = Path(dataset_path)
        payload = _load_pickle(self._dataset_path)

        labels, label_names = self.extract_labels(payload)
        labels, label_names = select_labels(
            labels,
            label_names,
            self.cfg.get("target_label_names"),
        )
        if self.task_type == "classification":
            labels, label_norm = _encode_classification_labels(labels)
            n_outputs = len(label_norm["classes"])
        else:
            # Fit mean/std on the TRAIN rows only (same seed/fractions
            # _make_loaders' train_val_test_split will later carve the
            # dataset with, so the indices line up), then apply those frozen
            # stats to every row before building the dataset -- fitting on
            # the full train+val+test pool would leak test statistics into
            # every normalized training label.
            train_idx, _val_idx, _test_idx = train_val_test_indices(
                len(labels), seed=self.cfg["seed"]
            )
            label_norm = fit_label_norm(labels[train_idx], label_names, self.cfg.get("log_label_names"))
            labels = apply_label_norm(labels, label_names, label_norm)
            n_outputs = labels.shape[1]

        dataset = self.build_dataset(payload, labels)

        head_extra_parts: list[np.ndarray] = []
        if self.cfg.get("use_covariates", False):
            covariate_means = _extract_covariate_means(payload)
            if len(covariate_means) != len(dataset):
                raise ValueError(f"Covariate/data length mismatch: {len(covariate_means)} vs {len(dataset)}")
            head_extra_parts.append(covariate_means)

        extra = self.extract_head_extra(payload, Path(dataset_path))
        if extra is not None:
            extra = np.asarray(extra, dtype=np.float32)
            if len(extra) != len(dataset):
                raise ValueError(f"head-extra feature/data length mismatch: {len(extra)} vs {len(dataset)}")
            head_extra_parts.append(extra)

        head_extra_dim = 0
        if head_extra_parts:
            combined_extra = _combine_head_extra(head_extra_parts)
            dataset = CovariateHeadDataset(dataset, combined_extra)
            head_extra_dim = combined_extra.shape[1]

        loaders = _make_loaders(dataset, self.cfg, collate_fn=self.collate_fn)

        if self.task_type == "params":
            model = SingleModalModel(
                encoder=self.build_encoder(dataset),
                head=ParameterEstimator(embedding_dim=self.encoder_output_dim + head_extra_dim, n_params=labels.shape[1])
            ).to(device)
        else:
            model = SingleModalModel(
                encoder=self.build_encoder(dataset),
                head=ClassificationHead(embedding_dim=self.encoder_output_dim + head_extra_dim, n_classes=labels.shape[1])
            ).to(device)

        history, best_state, test_loss = _train_and_eval(
            model, loaders, self.cfg, device, self.tag
        )
        test_loss_per_target = dict(zip(
            label_names, evaluate_per_target(model, loaders[2], device).tolist()
        ))

        adversarial_loss = None
        adversarial_loss_per_target = None

        if adversarial_path is not None and Path(adversarial_path).exists():
            self._dataset_path = Path(adversarial_path)
            adversarial_payload = _load_pickle(self._dataset_path)
            adversarial_labels, adversarial_label_names = self.extract_labels(adversarial_payload)
            adversarial_labels, _ = select_labels(
                adversarial_labels,
                adversarial_label_names,
                self.cfg.get("target_label_names"),
            )

            if self.task_type == "classification":
                # Pre-existing gap, not touched here: label_norm["mean"]/["std"]
                # are None for classification (see _encode_classification_labels),
                # so this path needs _apply_classification_encoding instead of
                # apply_label_norm. Left as-is to keep this change scoped to the
                # train/test normalization leak.
                adv_transformed = adversarial_labels.astype(float).copy()
                for j, transform in enumerate(label_norm["transforms"]):
                    if transform == "log":
                        if np.any(adv_transformed[:, j] <= 0):
                            raise ValueError(f"Cannot log-transform adversarial label {label_names[j]!r}")
                        adv_transformed[:, j] = np.log(adv_transformed[:, j])
                adversarial_labels = (adv_transformed - label_norm["mean"]) / label_norm["std"]
            else:
                adversarial_labels = apply_label_norm(adversarial_labels, label_names, label_norm)

            adversarial_dataset = self.build_dataset(adversarial_payload, adversarial_labels)

            adv_extra_parts: list[np.ndarray] = []
            if self.cfg.get("use_covariates", False):
                adversarial_covariate_means = _extract_covariate_means(adversarial_payload)
                adv_extra_parts.append(adversarial_covariate_means)

            adv_extra = self.extract_head_extra(adversarial_payload, Path(adversarial_path))
            if adv_extra is not None:
                adv_extra_parts.append(np.asarray(adv_extra, dtype=np.float32))

            if adv_extra_parts:
                adversarial_dataset = CovariateHeadDataset(adversarial_dataset, _combine_head_extra(adv_extra_parts))

            adversarial_loader = DataLoader(
                adversarial_dataset,
                batch_size=self.cfg["batch_size"],
                shuffle=False,
                collate_fn=self.collate_fn,
            )

            loss_fn = nn.MSELoss() if self.task_type == "params" else nn.CrossEntropyLoss()
            adversarial_loss, _ = evaluate(model, adversarial_loader, loss_fn, device)
            adversarial_loss_per_target = dict(zip(
                label_names, evaluate_per_target(model, adversarial_loader, device).tolist()
            ))

            print(f"\n[{self.tag}] Adversarial test loss: {adversarial_loss:.4f}")

        save_results(
            Path(output_dir),
            model=model,
            best_state=best_state,
            history=history,
            cfg=self.cfg,
            test_loss=test_loss,
            label_names=label_names,
            label_norm=label_norm,
            adversarial_loss=adversarial_loss,
            adversarial_path=adversarial_path,
            test_loss_per_target=test_loss_per_target,
            adversarial_loss_per_target=adversarial_loss_per_target,
            extra_meta=self.extra_meta,
        )

        result = {
            "history": history,
            "test_loss": test_loss,
            "test_loss_per_target": test_loss_per_target,
        }

        if adversarial_loss is not None:
            result["adversarial_loss"] = adversarial_loss
            result["adversarial_loss_per_target"] = adversarial_loss_per_target

        return result