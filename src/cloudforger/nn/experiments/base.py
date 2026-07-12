# src/cloudforger/nn/experiments/base.py

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

from cloudforger.nn.splits import train_val_test_split
from cloudforger.nn.train import train_one_epoch, evaluate
from cloudforger.nn.heads.paramest import ParameterEstimator
from cloudforger.nn.models.single_modal import SingleModalModel

# Number of parameters predicted per generating process.
N_PARAMS = {"thomas": 3, "matern": 2}

# Matches dimensioned methods like "pi_1" / "betti_0" / "betti_cnn_1".
_PERSIST_TOKEN = re.compile(r"^(pi|betti_cnn|betti)_(\d+)$")


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


def _select_labels(
    labels: np.ndarray,
    label_names: list[str],
    target_label_names: list[str] | None,
) -> tuple[np.ndarray, list[str]]:
    if not target_label_names:
        return labels, label_names

    idx = []
    for name in target_label_names:
        if name not in label_names:
            raise KeyError(f"Requested label {name!r}, available labels are {label_names}")
        idx.append(label_names.index(name))

    return labels[:, idx], list(target_label_names)


def _normalize_labels_by_name(
    labels: np.ndarray,
    label_names: list[str],
    log_label_names: list[str] | None,
) -> tuple[np.ndarray, dict]:
    log_set = set(log_label_names or label_names)

    transformed = labels.astype(float).copy()
    transforms = []

    for j, name in enumerate(label_names):
        if name in log_set:
            if np.any(transformed[:, j] <= 0):
                raise ValueError(f"Cannot log-transform non-positive label {name!r}")
            transformed[:, j] = np.log(transformed[:, j])
            transforms.append("log")
        else:
            transforms.append("identity")

    mean = transformed.mean(axis=0)
    std = transformed.std(axis=0)
    std = np.where(std == 0, 1.0, std)

    return (transformed - mean) / std, {
        "mean": mean,
        "std": std,
        "transforms": transforms,
    }


# ---------------------------------------------------------------------------
# Shared n(x) side-channel (Vihrs 2022 feeds n(x) alongside the curve because
# the curve alone can't recover intensity-related parameters). One
# implementation shared by every experiment that wants it, so all methods in
# a comparison have equal access to it rather than some getting it and
# others not.
# ---------------------------------------------------------------------------

def _resolve_sibling_clouds_path(dataset_path: Path) -> Path:
    """Resolve the sibling clouds.pkl (or adversarial_clouds.pkl) living next
    to a features/betti/images file, using the "adversarial_" filename
    prefix convention every split already follows (see DEFAULT_SPLITS in
    pipeline_lib/config.py), rather than pattern-matching on the file's own
    base name. Robust to any naming scheme (betti.pkl, betti_dtm_k5.pkl,
    images.pkl, features.pkl, ...) as long as it lives in the same directory
    as clouds.pkl, which every split does by construction."""
    name = "adversarial_clouds.pkl" if dataset_path.name.startswith("adversarial_") else "clouds.pkl"
    return dataset_path.parent / name


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


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

REGISTRY: dict[str, type["Experiment"]] = {}


def register(*names: str):
    """Class decorator registering an Experiment under one or more base names."""
    def deco(cls: type["Experiment"]) -> type["Experiment"]:
        for name in names:
            REGISTRY[name] = cls
        return cls
    return deco


def _parse_method(method: str) -> tuple[str, int | None]:
    m = _PERSIST_TOKEN.match(method)
    if m:
        return m.group(1), int(m.group(2))
    return method, None


def build_experiment(cfg: dict) -> "Experiment":
    """Construct the Experiment for a single concrete method string."""
    base, hom_dim = _parse_method(cfg["method"])
    try:
        cls = REGISTRY[base]
    except KeyError:
        raise ValueError(
            f"Unknown method '{cfg['method']}'. Registered: {sorted(REGISTRY)}; "
            "plus pi_<dim>, betti_<dim>."
        )
    return cls(cfg, hom_dim)


# ---------------------------------------------------------------------------
# Shared machinery
# ---------------------------------------------------------------------------

def _load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _prepare_device(cfg: dict) -> str:
    torch.manual_seed(cfg["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg["seed"])
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


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

    def __init__(self, cfg: dict, hom_dim: int | None = None):
        self.cfg = cfg
        self.hom_dim = hom_dim  # used by dimensioned experiments; ignored otherwise

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
        device = _prepare_device(self.cfg)

        payload = _load_pickle(Path(dataset_path))

        labels, label_names = self.extract_labels(payload)
        labels, label_names = _select_labels(
            labels,
            label_names,
            self.cfg.get("target_label_names"),
        )
        labels, label_norm = _normalize_labels_by_name(
            labels,
            label_names,
            self.cfg.get("log_label_names"),
        )

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

        model = SingleModalModel(
            encoder=self.build_encoder(dataset),
            head=ParameterEstimator(embedding_dim=self.cfg["embedding_dim"] + head_extra_dim, n_params=labels.shape[1])
        ).to(device)

        history, best_state, test_loss = _train_and_eval(
            model, loaders, self.cfg, device, self.tag
        )

        adversarial_loss = None

        if adversarial_path is not None and Path(adversarial_path).exists():
            adversarial_payload = _load_pickle(Path(adversarial_path))
            adversarial_labels, adversarial_label_names = self.extract_labels(adversarial_payload)
            adversarial_labels, _ = _select_labels(
                adversarial_labels,
                adversarial_label_names,
                self.cfg.get("target_label_names"),
            )

            adv_transformed = adversarial_labels.astype(float).copy()
            for j, transform in enumerate(label_norm["transforms"]):
                if transform == "log":
                    if np.any(adv_transformed[:, j] <= 0):
                        raise ValueError(f"Cannot log-transform adversarial label {label_names[j]!r}")
                    adv_transformed[:, j] = np.log(adv_transformed[:, j])

            adversarial_labels = (adv_transformed - label_norm["mean"]) / label_norm["std"]

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

            loss_fn = nn.MSELoss()
            adversarial_loss, _ = evaluate(model, adversarial_loader, loss_fn, device)

            print(f"\n[{self.tag}] Adversarial test loss: {adversarial_loss:.4f}")

        self._save(
            Path(output_dir),
            model,
            best_state,
            history,
            test_loss,
            label_names,
            label_norm,
            adversarial_loss=adversarial_loss,
            adversarial_path=adversarial_path,
        )

        result = {"history": history, "test_loss": test_loss}

        if adversarial_loss is not None:
            result["adversarial_loss"] = adversarial_loss

        return result

    def _save(
        self,
        output_dir: Path,
        model: nn.Module,
        best_state,
        history,
        test_loss,
        label_names,
        label_norm,
        adversarial_loss: float | None = None,
        adversarial_path: Path | None = None,
    ):
        output_dir.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "model_state": best_state,
                "history": history,
                "config": self.cfg,
                "test_loss": test_loss,
                "label_names": label_names,
                "label_norm": label_norm,
                "label_log_mean": label_norm["mean"],
                "label_log_std": label_norm["std"],
                "label_transforms": label_norm["transforms"],
                "adversarial_loss": adversarial_loss,
                "adversarial_path": adversarial_path,
            },
            output_dir / "results.pt",
        )

        torch.save(model.cpu(), output_dir / "model.pt")

        json_payload = {
            "task": self.cfg["task"],
            "method": self.cfg["method"],
            "test_loss": test_loss,
            "seed": self.cfg["seed"],
            **self.extra_meta,
        }

        if adversarial_loss is not None:
            json_payload["adversarial_loss"] = adversarial_loss
            json_payload["adversarial_path"] = str(adversarial_path)

        with open(output_dir / "results.json", "w") as f:
            json.dump(json_payload, f, indent=2)