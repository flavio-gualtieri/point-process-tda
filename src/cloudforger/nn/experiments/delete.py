# src/cloudforger/nn/experiments/pi_multik.py

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, TensorDataset

from cloudforger.baselines import vihrs
from cloudforger.core.io import intersect_seeds
from cloudforger.core.splits import train_val_test_indices
from cloudforger.nn.encoders.persistence_image import PIEncoder
from cloudforger.nn.experiments.base import register
from cloudforger.nn.experiments.common import (
    MultiSourceExperiment,
    apply_zscore,
    fit_zscore,
    prepare_device,
    save_results,
)
from cloudforger.nn.heads.paramest import ParameterEstimator
from cloudforger.nn.models.single_modal import SingleModalModel
from cloudforger.nn.train import evaluate, evaluate_per_target, train_one_epoch


def _load_pickle(path: Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def load_multik_split(
    k_values: list[int],
    image_paths: list[Path],
    clouds_path: Path,
    label_names: tuple[str, ...],
    tag: str,
) -> dict[str, Any] | None:
    payloads = []
    for k, path in zip(k_values, image_paths):
        if not Path(path).exists():
            print(f"  [{tag}] missing {path} -- skipping split.")
            return None
        payloads.append(_load_pickle(path))

    seed_arrays = [np.asarray(p["seeds"]) for p in payloads]
    idx_per_k, common_seeds = intersect_seeds(seed_arrays)
    per_k_totals = {k: len(s) for k, s in zip(k_values, seed_arrays)}
    print(f"  [{tag}] {len(common_seeds)} clouds common to all k in {k_values} (per-k totals: {per_k_totals}).")

    # Select target columns by name -- label_names can carry deterministic
    # bookkeeping fields (e.g. edge_buffer) alongside the real targets.
    tda_label_names = list(payloads[0]["label_names"])
    missing = [name for name in label_names if name not in tda_label_names]
    if missing:
        raise KeyError(f"[{tag}] label_names {tda_label_names} is missing {missing} from {label_names}")
    col_idx = [tda_label_names.index(name) for name in label_names]

    targets = np.asarray(payloads[0]["labels"], dtype=float)[np.ix_(idx_per_k[0], col_idx)]
    for k, payload, idx in zip(k_values[1:], payloads[1:], idx_per_k[1:]):
        targets_k = np.asarray(payload["labels"], dtype=float)[np.ix_(idx, col_idx)]
        assert np.allclose(targets, targets_k), (
            f"[{tag}] target mismatch between k={k_values[0]} and k={k} after seed alignment -- alignment bug."
        )

    # Channel order: for k in k_values, [pi0(k), pi1(k)] -- see module docstring.
    pi_channels: list[np.ndarray] = []
    entropy_cols: dict[str, np.ndarray] = {}
    for k, payload, idx in zip(k_values, payloads, idx_per_k):
        pi_channels.append(np.asarray(payload["image_tensors"][0], dtype=np.float64)[idx])
        pi_channels.append(np.asarray(payload["image_tensors"][1], dtype=np.float64)[idx])
        entropy_cols[f"entropy0_k{k}"] = np.asarray(payload["persistence_entropy"][0], dtype=np.float64)[idx]
        entropy_cols[f"entropy1_k{k}"] = np.asarray(payload["persistence_entropy"][1], dtype=np.float64)[idx]

    # n(x) doesn't depend on k (same underlying cloud) -- joined once from
    # the sibling clouds.pkl by seed.
    clouds = _load_pickle(clouds_path)
    n_points_by_seed = {int(c["seed"]): c["n_points"] for c in clouds}
    n_points = np.array([n_points_by_seed[int(s)] for s in common_seeds], dtype=np.float64)

    return {
        "pi_channels": pi_channels,
        "entropy_cols": entropy_cols,
        "n_points": n_points,
        "targets": targets,
        "seeds": common_seeds,
    }


def build_pi_tensor(
    split: dict[str, Any], channel_norms: list[dict[str, float]] | None
) -> tuple[np.ndarray, list[dict[str, float]]]:
    """(N, 2*len(k_values), H, W) float32 tensor, one independent z-score per
    channel -- PI pixels are raw Gaussian-kernel sums with a huge dynamic
    range, and different k's have different raw magnitude scales on top of
    that, so per-channel (not global) normalization matters here more than
    it would for a single k."""
    fit = channel_norms is None
    if fit:
        channel_norms = []
    channels = []
    for i, raw in enumerate(split["pi_channels"]):
        norm = fit_zscore(raw) if fit else channel_norms[i]
        if fit:
            channel_norms.append(norm)
        channels.append(apply_zscore(raw, norm))
    return np.stack(channels, axis=1).astype(np.float32), channel_norms


def build_extra(
    split: dict[str, Any], n_norm: dict, entropy_norms: dict[str, dict] | None, k_values: list[int]
) -> tuple[np.ndarray, dict[str, dict]]:
    fit = entropy_norms is None
    if fit:
        entropy_norms = {}
    n_std = vihrs.apply_log_zscore(split["n_points"], n_norm).astype(np.float32)
    cols = [n_std]
    for k in k_values:
        for dim in (0, 1):
            name = f"entropy{dim}_k{k}"
            values = split["entropy_cols"][name]
            norm = fit_zscore(values) if fit else entropy_norms[name]
            if fit:
                entropy_norms[name] = norm
            cols.append(apply_zscore(values, norm).astype(np.float32))
    return np.stack(cols, axis=1), entropy_norms


@register("pi_multik")
class PIMultiKExperiment(MultiSourceExperiment):
    file_keys = ("clouds", "images")

    @property
    def subdir(self) -> str:
        return "pi_multik"

    def run(
        self,
        dataset_paths: dict[str, Any],
        output_dir: Path,
        adversarial_paths: dict[str, Any] | None = None,
    ) -> dict:
        label_names = tuple(self.cfg.get("label_names", vihrs.DEFAULT_LABEL_NAMES))
        k_values = list(self.cfg["k_values"])
        seed = self.cfg["seed"]
        device = prepare_device(seed)

        train_split = load_multik_split(
            k_values, list(dataset_paths["images"]), Path(dataset_paths["clouds"]), label_names, tag="train_test",
        )
        if train_split is None:
            raise FileNotFoundError(f"images missing for some k in {k_values} under {dataset_paths['images']}.")

        adv_split = None
        if adversarial_paths is not None:
            adv_split = load_multik_split(
                k_values, list(adversarial_paths["images"]), Path(adversarial_paths["clouds"]),
                label_names, tag="adversarial",
            )

        n = len(train_split["targets"])
        label_norm = vihrs.fit_log_zscore(train_split["targets"])
        n_norm = vihrs.fit_log_zscore(train_split["n_points"])

        targets_std = vihrs.apply_log_zscore(train_split["targets"], label_norm).astype(np.float32)
        pi_img, channel_norms = build_pi_tensor(train_split, channel_norms=None)
        extra, entropy_norms = build_extra(train_split, n_norm, entropy_norms=None, k_values=k_values)

        full_dataset = TensorDataset(
            torch.from_numpy(pi_img), torch.from_numpy(extra), torch.from_numpy(targets_std),
        )
        train_idx, val_idx, test_idx = train_val_test_indices(n, seed)

        def _loader(idx: np.ndarray, shuffle: bool) -> DataLoader:
            return DataLoader(Subset(full_dataset, idx), batch_size=self.cfg["batch_size"], shuffle=shuffle)

        train_loader, val_loader, test_loader = _loader(train_idx, True), _loader(val_idx, False), _loader(test_idx, False)

        # SingleModalModel(encoder, head).forward(x, covariates) lines up
        # exactly with cloudforger.nn.train's 3-tuple batch convention
        # (inputs, covariates, labels), so the shared train loop applies as-is.
        model = SingleModalModel(
            encoder=PIEncoder(in_channels=pi_img.shape[1], embedding_dim=self.cfg["embedding_dim"]),
            head=ParameterEstimator(embedding_dim=self.cfg["embedding_dim"] + extra.shape[1], n_params=len(label_names)),
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.cfg["lr"], weight_decay=1e-4)
        loss_fn = nn.MSELoss()

        history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
        best_val_loss, best_state = float("inf"), None
        n_epochs = self.cfg["n_epochs"]

        for epoch in range(1, n_epochs + 1):
            train_loss, _ = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
            val_loss, _ = evaluate(model, val_loader, loss_fn, device)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if epoch == 1 or epoch % 25 == 0 or epoch == n_epochs:
                print(f"[pi_multik seed={seed}] epoch {epoch:3d} | train {train_loss:.4f} | val {val_loss:.4f}")

        model.load_state_dict(best_state)
        test_loss, _ = evaluate(model, test_loader, loss_fn, device)
        test_loss_per_target = dict(zip(label_names, evaluate_per_target(model, test_loader, device).tolist()))
        print(f"\n[pi_multik seed={seed}] test loss {test_loss:.4f}")

        adversarial_loss = None
        adversarial_loss_per_target = None
        if adv_split is not None:
            adv_targets_std = vihrs.apply_log_zscore(adv_split["targets"], label_norm).astype(np.float32)
            adv_pi_img, _ = build_pi_tensor(adv_split, channel_norms=channel_norms)
            adv_extra, _ = build_extra(adv_split, n_norm, entropy_norms=entropy_norms, k_values=k_values)
            adv_ds = TensorDataset(
                torch.from_numpy(adv_pi_img), torch.from_numpy(adv_extra), torch.from_numpy(adv_targets_std),
            )
            adv_loader = DataLoader(adv_ds, batch_size=self.cfg["batch_size"], shuffle=False)
            adversarial_loss, _ = evaluate(model, adv_loader, loss_fn, device)
            adversarial_loss_per_target = dict(
                zip(label_names, evaluate_per_target(model, adv_loader, device).tolist())
            )
            print(f"[pi_multik seed={seed}] adversarial loss {adversarial_loss:.4f}")

        cfg_meta = {
            **self.cfg,
            "method": "pi_multik",
            "channels": [f"k{k}_h{d}" for k in k_values for d in (0, 1)],
        }
        save_results(
            output_dir, model=model, best_state=best_state, history=history, cfg=cfg_meta,
            test_loss=test_loss, label_names=list(label_names), label_norm=label_norm,
            test_loss_per_target=test_loss_per_target, adversarial_loss=adversarial_loss,
            adversarial_loss_per_target=adversarial_loss_per_target,
            adversarial_path=adversarial_paths.get("clouds") if adversarial_paths else None,
        )

        result: dict[str, Any] = {"seed": seed, "test_loss": test_loss, "test_loss_per_target": test_loss_per_target}
        if adversarial_loss is not None:
            result["adversarial_loss"] = adversarial_loss
            result["adversarial_loss_per_target"] = adversarial_loss_per_target
        return result
