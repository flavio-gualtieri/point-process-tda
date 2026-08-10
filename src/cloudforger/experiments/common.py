# src/cloudforger/experiments/common.py
"""Shared machinery for both Experiment (single dataset_path) and
MultiSourceExperiment (several dataset files joined by seed -- fusion,
pi_multik, pi_multik_fusion) subclasses, so every method -- regardless of
how many source files it reads -- writes the identical results.pt /
results.json / model.pt schema and shares the same device-selection /
label-handling / zscore utilities instead of each hand-rolling its own copy."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from cloudforger.provenance import append_ledger_entry, provenance_stamp


def prepare_device(seed: int) -> str:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def fit_zscore(values: np.ndarray) -> dict[str, float]:
    """Plain (non-log) fit/apply-frozen zscore -- for values that can
    legitimately be zero or negative (persistence entropy, PI pixels),
    where a log-based normalization would be undefined."""
    std = float(values.std())
    return {"mean": float(values.mean()), "std": std if std else 1.0}


def apply_zscore(values: np.ndarray, norm: dict[str, float]) -> np.ndarray:
    return (values - norm["mean"]) / norm["std"]


def select_labels(
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


def _log_transform_by_name(
    labels: np.ndarray,
    label_names: list[str],
    log_label_names: list[str] | None,
) -> tuple[np.ndarray, list[str]]:
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

    return transformed, transforms


def fit_label_norm(
    labels: np.ndarray,
    label_names: list[str],
    log_label_names: list[str] | None,
) -> dict:
    """Fit log+zscore label stats on `labels` -- pass ONLY the training rows
    here (see apply_label_norm to transform other splits with the result),
    the same fit-on-train/apply-frozen convention as fit_zscore /
    vihrs.fit_log_zscore. Fitting on train+val+test combined would leak test
    statistics into every normalized training label."""
    transformed, transforms = _log_transform_by_name(labels, label_names, log_label_names)

    mean = transformed.mean(axis=0)
    std = transformed.std(axis=0)
    std = np.where(std == 0, 1.0, std)

    return {"mean": mean, "std": std, "transforms": transforms}


def apply_label_norm(
    labels: np.ndarray,
    label_names: list[str],
    label_norm: dict,
) -> np.ndarray:
    """Apply an already-fit label_norm (see fit_label_norm) to `labels` --
    the frozen-stats half of the fit/apply split, used for val/test rows and
    for the adversarial payload."""
    # label_norm["transforms"] is positional ("log"/"identity" per column,
    # written by fit_label_norm); _log_transform_by_name wants a name set,
    # so translate positionally rather than re-deriving it from log_set.
    log_names = [name for name, t in zip(label_names, label_norm["transforms"]) if t == "log"]
    transformed, _ = _log_transform_by_name(labels, label_names, log_names)
    return (transformed - label_norm["mean"]) / label_norm["std"]


def normalize_labels_by_name(
    labels: np.ndarray,
    label_names: list[str],
    log_label_names: list[str] | None,
) -> tuple[np.ndarray, dict]:
    """Fit AND apply in one call -- kept for callers that don't need a
    separate train-only fit (e.g. a one-off script normalizing a single
    array). Prefer fit_label_norm/apply_label_norm for a train/val/test
    split so val and test are transformed with frozen, train-only stats."""
    label_norm = fit_label_norm(labels, label_names, log_label_names)
    return apply_label_norm(labels, label_names, label_norm), label_norm


def save_results(
    output_dir: Path,
    *,
    model: nn.Module | None,
    best_state: dict | None,
    history: dict,
    cfg: dict,
    test_loss: float,
    label_names: list[str],
    label_norm: dict,
    test_loss_per_target: dict[str, float] | None = None,
    adversarial_loss: float | None = None,
    adversarial_loss_per_target: dict[str, float] | None = None,
    adversarial_path: Path | None = None,
    extra_meta: dict | None = None,
) -> None:
    """The one place every method writes results.pt / (optionally) model.pt /
    results.json, regardless of whether it's a single-source Experiment, a
    multi-source one, or a classical baseline with no trained model at all
    (model=None) -- this is what lets evaluate.py have exactly one code path
    no matter which method produced the numbers."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Provenance: exact commit + dirty-tree flag + when + which run_tag
    # archive this belongs to (cfg["run_tag"]/cfg["results_root"] are set by
    # scripts/train.py, not part of the method's own hyperparameters) -- see
    # cloudforger.provenance for why this needs to be recoverable from
    # results.json alone, without loading the torch file.
    stamp = provenance_stamp(run_tag=cfg.get("run_tag"))

    torch.save(
        {
            "model_state": best_state,
            "history": history,
            "config": cfg,
            "test_loss": test_loss,
            "test_loss_per_target": test_loss_per_target,
            "label_names": list(label_names),
            "label_norm": label_norm,
            "label_log_mean": label_norm["mean"],
            "label_log_std": label_norm["std"],
            "label_transforms": label_norm.get("transforms", ["log"] * len(label_names)),
            "adversarial_loss": adversarial_loss,
            "adversarial_loss_per_target": adversarial_loss_per_target,
            "adversarial_path": adversarial_path,
            **stamp,
        },
        output_dir / "results.pt",
    )
    if model is not None:
        torch.save(model.cpu(), output_dir / "model.pt")

    json_payload: dict[str, Any] = {
        "task": cfg.get("task", "params"),
        "method": cfg["method"],
        "test_loss": test_loss,
        "test_loss_per_target": test_loss_per_target,
        "seed": cfg["seed"],
        **stamp,
        "config": cfg,
        **(extra_meta or {}),
    }
    if adversarial_loss is not None:
        json_payload["adversarial_loss"] = adversarial_loss
        json_payload["adversarial_loss_per_target"] = adversarial_loss_per_target
        json_payload["adversarial_path"] = str(adversarial_path)

    with open(output_dir / "results.json", "w") as f:
        json.dump(json_payload, f, indent=2, default=str)

    append_ledger_entry(
        output_dir=output_dir,
        results_root=cfg.get("results_root"),
        method=cfg["method"],
        seed=cfg["seed"],
        test_loss=test_loss,
        adversarial_loss=adversarial_loss,
        stamp=stamp,
    )


class MultiSourceExperiment(ABC):
    """Sibling to Experiment for methods that need several dataset files
    joined by seed (fusion: vihrs + images + betti; pi_multik: images at
    several DTM k; pi_multik_fusion: vihrs + pi_multik) -- Experiment.run()'s
    single dataset_path: Path contract has no hook for that join. Same
    registry (register under the same name via nn.experiments.base.register),
    same output schema (via save_results above); different run() shape.

    dataset_paths values are Path or list[Path] (list for pi_multik's
    per-k image files), keyed from a small fixed vocabulary the concrete
    subclass documents (e.g. "clouds", "images", "betti")."""

    file_keys: tuple[str, ...]  # declares which dataset_paths keys this method needs, for CLI/config wiring

    def __init__(self, cfg: dict):
        self.cfg = cfg

    @property
    @abstractmethod
    def subdir(self) -> str:
        """Result subdirectory and default log tag."""

    @property
    def tag(self) -> str:
        return self.subdir

    @abstractmethod
    def run(
        self,
        dataset_paths: dict[str, Any],
        output_dir: Path,
        adversarial_paths: dict[str, Any] | None = None,
    ) -> dict:
        """Train, evaluate, and save results for one seed (self.cfg["seed"]).
        adversarial_paths uses the same keys as dataset_paths, or is None to
        skip adversarial evaluation. Mirrors Experiment.run()'s
        (dataset_path, output_dir, adversarial_path) shape, pluralized."""
