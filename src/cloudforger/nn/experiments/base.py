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
N_PARAMS = {"thomas": 3}

# Matches dimensioned methods like "pi_1" / "betti_0".
_PERSIST_TOKEN = re.compile(r"^(pi|betti)_(\d+)$")


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


def _make_head(cfg: dict) -> ParameterEstimator:
    process = cfg["process"]
    if process not in N_PARAMS:
        raise ValueError(f"Unknown process '{process}'. Available: {sorted(N_PARAMS)}")
    return ParameterEstimator(embedding_dim=cfg["embedding_dim"], n_params=N_PARAMS[process])


def _make_loaders(dataset: Dataset, cfg: dict):
    train_ds, val_ds, test_ds = train_val_test_split(dataset, seed=cfg["seed"])
    kw = dict(batch_size=cfg["batch_size"])
    return (
        DataLoader(train_ds, **kw, shuffle=True),
        DataLoader(val_ds, **kw),
        DataLoader(test_ds, **kw),
    )


def _train_and_eval(model, loaders, cfg: dict, device: str, tag: str):
    train_loader, val_loader, test_loader = loaders
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
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
    """One end-to-end parameter-estimation run.

    Subclasses declare ``file_key`` (semantic dataset key) and ``subdir``
    (result location), then implement ``build_dataset`` and ``build_encoder``.
    :meth:`run` ties the shared pipeline together.
    """

    #: semantic dataset key; the runner maps it to a concrete filename.
    file_key: str

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

    @abstractmethod
    def build_dataset(self, payload: Any, labels: np.ndarray) -> Dataset:
        """Turn the loaded payload + normalized labels into a Dataset."""

    @abstractmethod
    def build_encoder(self, dataset: Dataset) -> nn.Module:
        """Build the encoder that feeds the shared regression head."""

    # -- shared orchestration ----------------------------------------------

    def run(self, dataset_path: Path, output_dir: Path) -> dict:
        device = _prepare_device(self.cfg)

        payload = _load_pickle(Path(dataset_path))
        labels, label_names = self.extract_labels(payload)
        labels, log_mean, log_std = _normalize_labels(labels)

        dataset = self.build_dataset(payload, labels)
        loaders = _make_loaders(dataset, self.cfg)

        model = SingleModalModel(
            encoder=self.build_encoder(dataset),
            head=_make_head(self.cfg),
        ).to(device)

        history, best_state, test_loss = _train_and_eval(
            model, loaders, self.cfg, device, self.tag
        )

        self._save(Path(output_dir), best_state, history, test_loss,
                   label_names, log_mean, log_std)
        return {"history": history, "test_loss": test_loss}

    def _save(self, output_dir: Path, best_state, history, test_loss,
              label_names, log_mean, log_std):
        output_dir.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "model_state": best_state,
                "history": history,
                "config": self.cfg,
                "test_loss": test_loss,
                "label_names": label_names,
                "label_log_mean": log_mean,
                "label_log_std": log_std,
            },
            output_dir / "results.pt",
        )

        with open(output_dir / "results.json", "w") as f:
            json.dump(
                {
                    "task": self.cfg["task"],
                    "method": self.cfg["method"],
                    "test_loss": test_loss,
                    "seed": self.cfg["seed"],
                    **self.extra_meta,
                },
                f,
                indent=2,
            )