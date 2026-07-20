# src/cloudforger/nn/experiments/fusion.py
"""L(r)-r + TDA fusion: feeds vihrs's L(r)-r/n(x) representation and one or
more TDA branches (persistence images, Betti curves, persistence entropy)
into ONE model, late-fused by concatenating branch embeddings before a
shared regression head. Absorbed from dtm_experiment/fusion_model.py.

Not an Experiment subclass: vihrs's L(r)-r population (every cloud) and the
DTM-filtered TDA population are different sizes and must be joined by seed
(cloudforger.core.io.align_seeds), which Experiment.run()'s single
dataset_path contract has no hook for -- see MultiSourceExperiment.

cfg["variant"] selects which TDA arm(s) are fused in, alongside the
always-on L(r)-r + n(x) + entropy branch:
  full    -- L(r)-r + persistence images + Betti curves (the complete model)
  pi      -- L(r)-r + persistence images only
  betti   -- L(r)-r + Betti curves only
  scalars -- L(r)-r only (n(x) + persistence entropy scalars, no image/curve branch)
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, TensorDataset

from cloudforger.baselines import vihrs
from cloudforger.core.io import align_seeds
from cloudforger.core.splits import train_val_test_indices
from cloudforger.nn.encoders.persistence_image import PIEncoder
from cloudforger.nn.encoders.sequence_cnn import SequenceCNNEncoder
from cloudforger.nn.experiments.base import register
from cloudforger.nn.experiments.common import (
    MultiSourceExperiment,
    apply_zscore,
    fit_zscore,
    prepare_device,
    save_results,
)

VARIANTS = ("full", "pi", "betti", "scalars")


def _load_pickle(path: Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def load_fusion_split(
    images_path: Path,
    betti_path: Path,
    lr_features: dict[str, np.ndarray],
    label_names: tuple[str, ...],
    tag: str,
) -> dict[str, np.ndarray] | None:
    if not images_path.exists() or not betti_path.exists():
        print(f"  [{tag}] missing {images_path if not images_path.exists() else betti_path} -- skipping split.")
        return None

    images = _load_pickle(images_path)
    betti = _load_pickle(betti_path)

    assert list(images["seeds"]) == list(betti["seeds"]), (
        f"[{tag}] images/betti seed order mismatch -- both should come from the same "
        "featurize run and share identical 'seeds' order."
    )

    # Select target columns by name rather than assuming an exact column
    # count/order: the payload's "params" can carry deterministic bookkeeping
    # fields (e.g. edge_buffer) alongside the real targets.
    tda_label_names = list(images["label_names"])
    missing = [name for name in label_names if name not in tda_label_names]
    if missing:
        raise KeyError(f"[{tag}] TDA label_names {tda_label_names} is missing {missing} from {label_names}")
    col_idx = [tda_label_names.index(name) for name in label_names]

    tda_idx, lr_idx = align_seeds(np.asarray(images["seeds"]), lr_features["cloud_seeds"])
    print(f"  [{tag}] {len(tda_idx)}/{len(images['seeds'])} TDA clouds matched to L(r)-r clouds.")

    # Sanity check: independently-loaded targets must agree at the aligned indices.
    tda_targets = np.asarray(images["labels"], dtype=float)[np.ix_(tda_idx, col_idx)]
    lr_targets = lr_features["targets"][lr_idx]
    assert np.allclose(tda_targets, lr_targets), f"[{tag}] target mismatch after seed alignment -- alignment bug."

    return {
        "pi0": np.asarray(images["image_tensors"][0], dtype=np.float64)[tda_idx],
        "pi1": np.asarray(images["image_tensors"][1], dtype=np.float64)[tda_idx],
        "b0": np.asarray(betti["betti0_matrix"], dtype=np.float64)[tda_idx],
        "b1": np.asarray(betti["betti1_matrix"], dtype=np.float64)[tda_idx],
        "entropy0": np.asarray(images["persistence_entropy"][0], dtype=np.float64)[tda_idx],
        "entropy1": np.asarray(images["persistence_entropy"][1], dtype=np.float64)[tda_idx],
        "lr_seq": lr_features["l_minus_r"][lr_idx],
        "n_points": lr_features["n_points"][lr_idx],
        "targets": tda_targets,
    }


class FusionCNN(nn.Module):
    def __init__(
        self,
        lr_seq_len: int,
        betti_seq_len: int,
        embedding_dim: int,
        n_extra: int,
        n_targets: int,
        include_pi: bool = True,
        include_betti: bool = True,
    ):
        super().__init__()
        self.include_pi = include_pi
        self.include_betti = include_betti

        # L(r)-r branch -- identical architecture to VihrsCNN's conv stack.
        self.lr_conv = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7), nn.ReLU(), nn.MaxPool1d(5),
            nn.Conv1d(64, 64, kernel_size=7), nn.ReLU(), nn.MaxPool1d(5),
            nn.Conv1d(64, 64, kernel_size=7), nn.ReLU(),
        )
        with torch.no_grad():
            lr_flat_dim = self.lr_conv(torch.zeros(1, 1, lr_seq_len)).flatten(1).shape[1]
        self.lr_head = nn.Linear(lr_flat_dim, embedding_dim)

        merge_dim = embedding_dim + n_extra
        if include_pi:
            self.pi_encoder = PIEncoder(in_channels=2, embedding_dim=embedding_dim)
            merge_dim += embedding_dim
        if include_betti:
            self.betti_encoder = SequenceCNNEncoder(input_dim=betti_seq_len, embedding_dim=embedding_dim, pool_size=5)
            merge_dim += embedding_dim

        self.head = nn.Sequential(
            nn.Linear(merge_dim, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(32, n_targets),
        )

    def forward(
        self, lr_seq: torch.Tensor, pi_img: torch.Tensor, betti_seq: torch.Tensor, extra: torch.Tensor
    ) -> torch.Tensor:
        parts = [self.lr_head(self.lr_conv(lr_seq.unsqueeze(1)).flatten(start_dim=1))]
        if self.include_pi:
            parts.append(self.pi_encoder(pi_img))
        if self.include_betti:
            parts.append(self.betti_encoder(betti_seq))
        parts.append(extra)
        return self.head(torch.cat(parts, dim=1))


def _train_one_epoch(model, loader, optimizer, loss_fn, device) -> float:
    model.train()
    total, count = 0.0, 0
    for lr_seq, pi_img, betti_seq, extra, y in loader:
        lr_seq, pi_img, betti_seq, extra, y = (t.to(device) for t in (lr_seq, pi_img, betti_seq, extra, y))
        optimizer.zero_grad()
        pred = model(lr_seq, pi_img, betti_seq, extra)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(y)
        count += len(y)
    return total / count


@torch.no_grad()
def _evaluate_loss(model, loader, loss_fn, device) -> float:
    model.eval()
    total, count = 0.0, 0
    for lr_seq, pi_img, betti_seq, extra, y in loader:
        lr_seq, pi_img, betti_seq, extra, y = (t.to(device) for t in (lr_seq, pi_img, betti_seq, extra, y))
        pred = model(lr_seq, pi_img, betti_seq, extra)
        loss = loss_fn(pred, y)
        total += loss.item() * len(y)
        count += len(y)
    return total / count


@torch.no_grad()
def _evaluate_per_target(model, loader, device) -> np.ndarray:
    model.eval()
    total_sq_err, count = None, 0
    for lr_seq, pi_img, betti_seq, extra, y in loader:
        lr_seq, pi_img, betti_seq, extra, y = (t.to(device) for t in (lr_seq, pi_img, betti_seq, extra, y))
        pred = model(lr_seq, pi_img, betti_seq, extra)
        sq_err = (pred - y).pow(2).sum(dim=0)
        total_sq_err = sq_err if total_sq_err is None else total_sq_err + sq_err
        count += len(y)
    return (total_sq_err / count).cpu().numpy()


def _build_extra(split: dict[str, np.ndarray], n_norm: dict, entropy0_norm: dict, entropy1_norm: dict) -> np.ndarray:
    n_std = vihrs.apply_log_zscore(split["n_points"], n_norm).astype(np.float32)
    e0_std = apply_zscore(split["entropy0"], entropy0_norm).astype(np.float32)
    e1_std = apply_zscore(split["entropy1"], entropy1_norm).astype(np.float32)
    return np.stack([n_std, e0_std, e1_std], axis=1)


@register("fusion")
class FusionExperiment(MultiSourceExperiment):
    file_keys = ("clouds", "images", "betti")

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        variant = cfg.get("variant", "full")
        if variant not in VARIANTS:
            raise ValueError(f"Unknown fusion variant {variant!r}; expected one of {VARIANTS}")
        self.variant = variant
        self.include_pi = variant in ("full", "pi")
        self.include_betti = variant in ("full", "betti")

    @property
    def subdir(self) -> str:
        return "fusion" if self.variant == "full" else f"fusion_{self.variant}"

    def run(
        self,
        dataset_paths: dict[str, Any],
        output_dir: Path,
        adversarial_paths: dict[str, Any] | None = None,
    ) -> dict:
        label_names = tuple(self.cfg.get("label_names", vihrs.DEFAULT_LABEL_NAMES))
        seed = self.cfg["seed"]
        device = prepare_device(seed)

        lr_data = vihrs.prepare_data(
            Path(dataset_paths["clouds"]),
            Path(adversarial_paths["clouds"]) if adversarial_paths else None,
            label_names=label_names,
        )
        train_split = load_fusion_split(
            Path(dataset_paths["images"]), Path(dataset_paths["betti"]),
            lr_data["train_features"], label_names, tag="train_test",
        )
        if train_split is None:
            raise FileNotFoundError(f"{dataset_paths['images']} / {dataset_paths['betti']} missing.")

        adv_split = None
        if adversarial_paths is not None and lr_data["adversarial_features"] is not None:
            adv_split = load_fusion_split(
                Path(adversarial_paths["images"]), Path(adversarial_paths["betti"]),
                lr_data["adversarial_features"], label_names, tag="adversarial",
            )

        n = len(train_split["targets"])
        label_norm = vihrs.fit_log_zscore(train_split["targets"])
        n_norm = vihrs.fit_log_zscore(train_split["n_points"])
        entropy0_norm = fit_zscore(train_split["entropy0"])
        entropy1_norm = fit_zscore(train_split["entropy1"])

        targets_std = vihrs.apply_log_zscore(train_split["targets"], label_norm).astype(np.float32)
        lr_seq = train_split["lr_seq"].astype(np.float32)
        pi_img = np.stack([train_split["pi0"], train_split["pi1"]], axis=1).astype(np.float32)
        betti_seq = np.concatenate([train_split["b0"], train_split["b1"]], axis=1).astype(np.float32)
        extra = _build_extra(train_split, n_norm, entropy0_norm, entropy1_norm)

        # One shared dataset over the whole split, sliced lazily via Subset --
        # not per-split fancy-indexed copies, which would double the PI
        # tensor's memory footprint.
        full_dataset = TensorDataset(
            torch.from_numpy(lr_seq), torch.from_numpy(pi_img),
            torch.from_numpy(betti_seq), torch.from_numpy(extra),
            torch.from_numpy(targets_std),
        )
        train_idx, val_idx, test_idx = train_val_test_indices(n, seed)

        def _loader(idx: np.ndarray, shuffle: bool) -> DataLoader:
            return DataLoader(Subset(full_dataset, idx), batch_size=self.cfg["batch_size"], shuffle=shuffle)

        train_loader, val_loader, test_loader = _loader(train_idx, True), _loader(val_idx, False), _loader(test_idx, False)

        model = FusionCNN(
            lr_seq_len=lr_seq.shape[1], betti_seq_len=betti_seq.shape[1],
            embedding_dim=self.cfg["embedding_dim"], n_extra=extra.shape[1], n_targets=len(label_names),
            include_pi=self.include_pi, include_betti=self.include_betti,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.cfg["lr"], weight_decay=1e-4)
        loss_fn = nn.MSELoss()

        history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
        best_val_loss, best_state = float("inf"), None
        n_epochs = self.cfg["n_epochs"]

        for epoch in range(1, n_epochs + 1):
            train_loss = _train_one_epoch(model, train_loader, optimizer, loss_fn, device)
            val_loss = _evaluate_loss(model, val_loader, loss_fn, device)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if epoch == 1 or epoch % 25 == 0 or epoch == n_epochs:
                print(f"[{self.subdir} seed={seed}] epoch {epoch:3d} | train {train_loss:.4f} | val {val_loss:.4f}")

        model.load_state_dict(best_state)
        test_loss = _evaluate_loss(model, test_loader, loss_fn, device)
        test_loss_per_target = dict(zip(label_names, _evaluate_per_target(model, test_loader, device).tolist()))
        print(f"\n[{self.subdir} seed={seed}] test loss {test_loss:.4f}")

        adversarial_loss = None
        adversarial_loss_per_target = None
        if adv_split is not None:
            adv_targets_std = vihrs.apply_log_zscore(adv_split["targets"], label_norm).astype(np.float32)
            adv_lr_seq = adv_split["lr_seq"].astype(np.float32)
            adv_pi_img = np.stack([adv_split["pi0"], adv_split["pi1"]], axis=1).astype(np.float32)
            adv_betti_seq = np.concatenate([adv_split["b0"], adv_split["b1"]], axis=1).astype(np.float32)
            adv_extra = _build_extra(adv_split, n_norm, entropy0_norm, entropy1_norm)
            adv_ds = TensorDataset(
                torch.from_numpy(adv_lr_seq), torch.from_numpy(adv_pi_img),
                torch.from_numpy(adv_betti_seq), torch.from_numpy(adv_extra),
                torch.from_numpy(adv_targets_std),
            )
            adv_loader = DataLoader(adv_ds, batch_size=self.cfg["batch_size"], shuffle=False)
            adversarial_loss = _evaluate_loss(model, adv_loader, loss_fn, device)
            adversarial_loss_per_target = dict(
                zip(label_names, _evaluate_per_target(model, adv_loader, device).tolist())
            )
            print(f"[{self.subdir} seed={seed}] adversarial loss {adversarial_loss:.4f}")

        cfg_meta = {
            **self.cfg,
            "method": self.subdir,
            "channels": (
                ["l_minus_r", "n_points", "entropy_0", "entropy_1"]
                + (["pi_0", "pi_1"] if self.include_pi else [])
                + (["betti_0", "betti_1"] if self.include_betti else [])
            ),
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
