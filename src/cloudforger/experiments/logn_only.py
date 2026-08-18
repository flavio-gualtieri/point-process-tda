# src/cloudforger/experiments/logn_only.py
"""[none (log N only)] control -- predicts targets from log N(x) (the point
count) alone, with no topology and no filtration of any kind: the cheapest
possible baseline, needed to attribute any of pi_multik's performance to
the persistence images rather than to counting points (see the writeup's
Estimation-performance section -- this arm and pi_multik's
include_log_n=False arm are a matched pair: "topology alone" isn't
quantifiable without both).

Reads clouds.pkl directly (raw_pc.py's _cloud_list/_labels_from_clouds --
reused verbatim so the target values/order are byte-identical to raw_pc's,
which are themselves the same params every diagram-based method's labels
trace back to). No diagrams, no persistence images, no filtration config --
file_keys = ("clouds",) and an empty `filtration: []` in the YAML routes
results under results/<process>/raw/logn_only/seed_<seed>/ (see
paths.combined_filtration_tag: an empty filtration list resolves to the
"raw" tag), the same convention raw_pc uses for the same reason.

Normalization matches pi_multik.py exactly (vihrs.fit_log_zscore on
targets, plain z-score on log N(x)), not common.py's fit_label_norm/
apply_label_norm, so loss numbers are on the identical scale and directly
comparable to every pi_multik arm."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, TensorDataset

from cloudforger.baselines import vihrs
from cloudforger.core.splits import train_val_test_indices
from cloudforger.experiments.base import register
from cloudforger.experiments.common import (
    MultiSourceExperiment,
    prepare_device,
    save_results,
    select_labels,
)
from cloudforger.experiments.raw_pc import _cloud_list, _labels_from_clouds
from cloudforger.models.heads.paramest import ParameterEstimator
from cloudforger.training.train import evaluate, evaluate_per_target, train_one_epoch


def _load_pickle(path: Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


class _LogNHead(nn.Module):
    """Thin (inputs, covariates) -> logits wrapper around ParameterEstimator
    so train_one_epoch/evaluate's model(inputs, covariates) calling
    convention (training/train.py's _unpack_batch, which every batch here
    goes through as a 2-tuple -> covariates=None) works even though this
    model has no second, covariate input."""

    def __init__(self, head: ParameterEstimator):
        super().__init__()
        self.head = head

    def forward(self, x: torch.Tensor, covariates: torch.Tensor | None = None) -> torch.Tensor:
        return self.head(x)


@register("logn_only")
class LogNOnlyExperiment(MultiSourceExperiment):
    """MLP(log N(x)) -> targets. No CNN, no persistence-image branch --
    ParameterEstimator's hidden layers are the entire model."""

    file_keys = ("clouds",)

    @property
    def subdir(self) -> str:
        return "logn_only"

    def _load_split(self, clouds_path: Path) -> dict[str, Any]:
        clouds = _cloud_list(_load_pickle(clouds_path))
        n_points = np.array([c["n_points"] for c in clouds], dtype=np.float64)
        labels, label_names = _labels_from_clouds(clouds)
        target_label_names = self.cfg.get("target_label_names")
        labels, label_names = select_labels(labels, label_names, target_label_names)
        return {"n_points": n_points, "targets": labels, "label_names": label_names}

    def run(
        self,
        dataset_paths: dict[str, Any],
        output_dir: Path,
        adversarial_paths: dict[str, Any] | None = None,
    ) -> dict:
        seed = self.cfg["seed"]
        device = prepare_device(seed)

        train_split = self._load_split(Path(dataset_paths["clouds"]))
        label_names = train_split["label_names"]

        adv_split = None
        if adversarial_paths is not None:
            adv_split = self._load_split(Path(adversarial_paths["clouds"]))

        n = len(train_split["targets"])
        train_idx, val_idx, test_idx = train_val_test_indices(n, seed)

        label_norm = vihrs.fit_log_zscore(train_split["targets"][train_idx])
        targets_std = vihrs.apply_log_zscore(train_split["targets"], label_norm).astype(np.float32)

        # Identical log-then-zscore treatment pi_multik.py's build_extra gives
        # its own log N(x) column, so this baseline and pi_multik's
        # include_log_n=False/True arms are directly comparable.
        n_norm = vihrs.fit_log_zscore(train_split["n_points"][train_idx])
        n_std = vihrs.apply_log_zscore(train_split["n_points"], n_norm).astype(np.float32).reshape(-1, 1)

        full_dataset = TensorDataset(torch.from_numpy(n_std), torch.from_numpy(targets_std))

        def _loader(idx: np.ndarray, shuffle: bool) -> DataLoader:
            return DataLoader(Subset(full_dataset, idx), batch_size=self.cfg["batch_size"], shuffle=shuffle)

        train_loader, val_loader, test_loader = _loader(train_idx, True), _loader(val_idx, False), _loader(test_idx, False)

        model = _LogNHead(ParameterEstimator(
            embedding_dim=1, n_params=len(label_names),
            hidden_dims=tuple(self.cfg.get("head_hidden_dims", (16, 8))),
            dropout=self.cfg.get("head_dropout", 0.1),
        )).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.cfg.get("lr", 1e-3), weight_decay=self.cfg.get("weight_decay", 1e-4),
        )
        loss_fn = nn.MSELoss()

        history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
        best_val_loss, best_state = float("inf"), None
        n_epochs = self.cfg["n_epochs"]
        patience = self.cfg.get("early_stopping_patience")
        epochs_no_improve = 0

        for epoch in range(1, n_epochs + 1):
            train_loss, _ = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
            val_loss, _ = evaluate(model, val_loader, loss_fn, device)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            if epoch == 1 or epoch % 25 == 0 or epoch == n_epochs:
                print(f"[{self.tag} seed={seed}] epoch {epoch:3d} | train {train_loss:.4f} | val {val_loss:.4f}")
            if patience is not None and epochs_no_improve >= patience:
                print(f"[{self.tag} seed={seed}] early stopping at epoch {epoch} (no val improvement for {patience} epochs)")
                break

        model.load_state_dict(best_state)
        test_loss, _ = evaluate(model, test_loader, loss_fn, device)
        test_loss_per_target = dict(zip(label_names, evaluate_per_target(model, test_loader, device).tolist()))
        print(f"\n[{self.tag} seed={seed}] test loss {test_loss:.4f}")

        adversarial_loss = None
        adversarial_loss_per_target = None
        if adv_split is not None:
            adv_targets_std = vihrs.apply_log_zscore(adv_split["targets"], label_norm).astype(np.float32)
            adv_n_std = vihrs.apply_log_zscore(adv_split["n_points"], n_norm).astype(np.float32).reshape(-1, 1)
            adv_ds = TensorDataset(torch.from_numpy(adv_n_std), torch.from_numpy(adv_targets_std))
            adv_loader = DataLoader(adv_ds, batch_size=self.cfg["batch_size"], shuffle=False)
            adversarial_loss, _ = evaluate(model, adv_loader, loss_fn, device)
            adversarial_loss_per_target = dict(
                zip(label_names, evaluate_per_target(model, adv_loader, device).tolist())
            )
            print(f"[{self.tag} seed={seed}] adversarial loss {adversarial_loss:.4f}")

        save_results(
            output_dir, model=model, best_state=best_state, history=history, cfg=dict(self.cfg),
            test_loss=test_loss, label_names=list(label_names), label_norm=label_norm,
            test_loss_per_target=test_loss_per_target, adversarial_loss=adversarial_loss,
            adversarial_loss_per_target=adversarial_loss_per_target,
            adversarial_path=adversarial_paths.get("clouds") if adversarial_paths else None,
        )
