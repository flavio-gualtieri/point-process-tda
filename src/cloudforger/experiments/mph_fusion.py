# src/cloudforger/experiments/mph_fusion.py
"""Fusion of vihrs's L(r)-r + n(x) branch with mph_pi's CoordConv
multiparameter-persistence-image branch, late-fused by concatenating both
branch embeddings before a shared regression head -- same "concatenate
before the head" pattern pi_multik_fusion.py/fusion.py use, minus the k-axis
(mph_pi has no k/scale axis to fold into the batch dim, see mph_pi.py's
docstring: one CoordConvPIEncoder call directly on the (B, D, R, R) image
stack, not K of them).

Motivation: a preliminary 3-seed comparison (results/nested_thomas/mph_dtm0.05/
mph_pi vs results/nested_thomas/raw/vihrs_checkpointed) found the two methods
win on DIFFERENT targets -- mph_pi better at intensity/count parameters
(parent_intensity, meta_offspring), vihrs better at spatial-scale parameters
(cluster_scale, meta_cluster_scale) -- consistent with what each feature
actually encodes (L(r)-r targets characteristic clustering radii directly;
the DTM-bifiltration image is a density/topology signal). This fusion tests
whether one model can get both.

Reuses, rather than reimplements:
  - mph_pi.load_mph_split for the image side (single bifiltration file, no
    k-axis to intersect -- pi_multik_fusion.py's analogous reuse of
    pi_multik.load_multik_split has that extra intersection step because
    pi_multik joins several per-k files first).
  - mph_pi.build_mph_tensor for per-channel image z-scoring.
  - cloudforger.core.io.align_seeds for the join against vihrs's L(r)-r
    population (every cloud, never filtered) -- same join fusion.py/
    pi_multik_fusion.py need, for the same reason (different, in-principle
    filterable population sizes; mph_image happens to cover every cloud
    too, but this doesn't assume that).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, TensorDataset

from cloudforger.baselines import vihrs
from cloudforger.core.io import align_seeds
from cloudforger.core.splits import train_val_test_indices
from cloudforger.encoders.coordconv_pi import CoordConvPIEncoder
from cloudforger.experiments import mph_pi
from cloudforger.experiments.base import register
from cloudforger.experiments.common import MultiSourceExperiment, prepare_device, save_results
from cloudforger.models.heads.paramest import ParameterEstimator


def load_fusion_split(
    image_path: Path,
    clouds_path: Path,
    lr_features: dict[str, np.ndarray],
    label_names: tuple[str, ...],
    tag: str,
    homology_dims: tuple[int, ...] | None = None,
) -> dict[str, Any] | None:
    mph_split = mph_pi.load_mph_split(image_path, clouds_path, label_names, tag=tag, homology_dims=homology_dims)
    if mph_split is None:
        return None

    mph_idx, lr_idx = align_seeds(mph_split["seeds"], lr_features["cloud_seeds"])
    print(f"  [{tag}] {len(mph_idx)}/{len(mph_split['seeds'])} mph clouds matched to L(r)-r clouds.")

    targets_mph = mph_split["targets"][mph_idx]
    targets_lr = lr_features["targets"][lr_idx]
    assert np.allclose(targets_mph, targets_lr), f"[{tag}] target mismatch after seed alignment -- alignment bug."

    return {
        "image_tensors": mph_split["image_tensors"][mph_idx],
        "dims": mph_split["dims"],
        "n_points": mph_split["n_points"][mph_idx],
        "targets": targets_mph,
        "lr_seq": lr_features["l_minus_r"][lr_idx],
    }


class VihrsMPHFusion(nn.Module):
    def __init__(
        self,
        lr_seq_len: int,
        in_channels: int,
        embedding_dim: int,
        n_extra: int,
        n_targets: int,
        conv_channels: tuple[int, ...] = (32, 64, 128),
        dropout: float = 0.2,
        pool_type: str = "max",
        head_hidden_dims: tuple[int, ...] = (64, 32),
        head_dropout: float = 0.1,
    ):
        super().__init__()

        # L(r)-r branch -- identical architecture to VihrsCNN's/FusionCNN's own conv stack.
        self.lr_conv = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7), nn.ReLU(), nn.MaxPool1d(5),
            nn.Conv1d(64, 64, kernel_size=7), nn.ReLU(), nn.MaxPool1d(5),
            nn.Conv1d(64, 64, kernel_size=7), nn.ReLU(),
        )
        with torch.no_grad():
            lr_flat_dim = self.lr_conv(torch.zeros(1, 1, lr_seq_len)).flatten(1).shape[1]
        self.lr_head = nn.Linear(lr_flat_dim, embedding_dim)

        # mph_pi branch -- identical to mph_pi's own CoordConvPIEncoder, one
        # call directly on the (B, D, R, R) stack (no k-axis to fold).
        self.pi_encoder = CoordConvPIEncoder(
            in_channels=in_channels, embedding_dim=embedding_dim,
            conv_channels=conv_channels, dropout=dropout, pool_type=pool_type,
        )

        merge_dim = embedding_dim + embedding_dim + n_extra
        self.head = ParameterEstimator(
            embedding_dim=merge_dim, n_params=n_targets,
            hidden_dims=head_hidden_dims, dropout=head_dropout,
        )

    def forward(self, lr_seq: torch.Tensor, pi_img: torch.Tensor, extra: torch.Tensor) -> torch.Tensor:
        lr_emb = self.lr_head(self.lr_conv(lr_seq.unsqueeze(1)).flatten(start_dim=1))
        pi_emb = self.pi_encoder(pi_img)
        return self.head(torch.cat([lr_emb, pi_emb, extra], dim=1))


def _train_one_epoch(model, loader, optimizer, loss_fn, device) -> float:
    model.train()
    total, count = 0.0, 0
    for lr_seq, pi_img, extra, y in loader:
        lr_seq, pi_img, extra, y = (t.to(device) for t in (lr_seq, pi_img, extra, y))
        optimizer.zero_grad()
        pred = model(lr_seq, pi_img, extra)
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
    for lr_seq, pi_img, extra, y in loader:
        lr_seq, pi_img, extra, y = (t.to(device) for t in (lr_seq, pi_img, extra, y))
        pred = model(lr_seq, pi_img, extra)
        loss = loss_fn(pred, y)
        total += loss.item() * len(y)
        count += len(y)
    return total / count


@torch.no_grad()
def _evaluate_per_target(model, loader, device) -> np.ndarray:
    model.eval()
    total_sq_err, count = None, 0
    for lr_seq, pi_img, extra, y in loader:
        lr_seq, pi_img, extra, y = (t.to(device) for t in (lr_seq, pi_img, extra, y))
        pred = model(lr_seq, pi_img, extra)
        sq_err = (pred - y).pow(2).sum(dim=0)
        total_sq_err = sq_err if total_sq_err is None else total_sq_err + sq_err
        count += len(y)
    return (total_sq_err / count).cpu().numpy()


@register("mph_fusion")
class MPHFusionExperiment(MultiSourceExperiment):
    file_keys = ("clouds", "images")

    @property
    def subdir(self) -> str:
        return "mph_fusion"

    def run(
        self,
        dataset_paths: dict[str, Any],
        output_dir: Path,
        adversarial_paths: dict[str, Any] | None = None,
    ) -> dict:
        # None (no target_label_names in the YAML) lets prepare_data adapt
        # to every label this process's clouds actually carry.
        target_label_names = self.cfg.get("target_label_names")
        label_names = tuple(target_label_names) if target_label_names else None
        homology_dims_cfg = self.cfg.get("homology_dims")
        homology_dims = tuple(homology_dims_cfg) if homology_dims_cfg else None
        seed = self.cfg["seed"]
        device = prepare_device(seed)

        lr_data = vihrs.prepare_data(
            Path(dataset_paths["clouds"]),
            Path(adversarial_paths["clouds"]) if adversarial_paths else None,
            label_names=label_names,
        )
        label_names = tuple(lr_data["label_names"])
        train_split = load_fusion_split(
            Path(dataset_paths["images"]), Path(dataset_paths["clouds"]),
            lr_data["train_features"], label_names, tag="train_test", homology_dims=homology_dims,
        )
        if train_split is None:
            raise FileNotFoundError(f"mph_image missing under {dataset_paths['images']}.")

        adv_split = None
        if adversarial_paths is not None and lr_data["adversarial_features"] is not None:
            adv_split = load_fusion_split(
                Path(adversarial_paths["images"]), Path(adversarial_paths["clouds"]),
                lr_data["adversarial_features"], label_names, tag="adversarial", homology_dims=homology_dims,
            )

        n = len(train_split["targets"])
        label_norm = vihrs.fit_log_zscore(train_split["targets"])
        n_norm = vihrs.fit_log_zscore(train_split["n_points"])
        lr_norm = vihrs.fit_zscore_global(train_split["lr_seq"])

        targets_std = vihrs.apply_log_zscore(train_split["targets"], label_norm).astype(np.float32)
        lr_seq = vihrs.apply_zscore_global(train_split["lr_seq"], lr_norm).astype(np.float32)
        pi_img, channel_norms = mph_pi.build_mph_tensor(train_split["image_tensors"], channel_norms=None)
        extra = vihrs.apply_log_zscore(train_split["n_points"], n_norm).astype(np.float32)[:, None]

        full_dataset = TensorDataset(
            torch.from_numpy(lr_seq), torch.from_numpy(pi_img), torch.from_numpy(extra), torch.from_numpy(targets_std),
        )
        train_idx, val_idx, test_idx = train_val_test_indices(n, seed)

        def _loader(idx: np.ndarray, shuffle: bool) -> DataLoader:
            return DataLoader(Subset(full_dataset, idx), batch_size=self.cfg["batch_size"], shuffle=shuffle)

        train_loader, val_loader, test_loader = _loader(train_idx, True), _loader(val_idx, False), _loader(test_idx, False)

        model = VihrsMPHFusion(
            lr_seq_len=lr_seq.shape[1], in_channels=pi_img.shape[1], embedding_dim=self.cfg["embedding_dim"],
            n_extra=extra.shape[1], n_targets=len(label_names),
            conv_channels=tuple(self.cfg.get("conv_channels", (32, 64, 128))),
            dropout=self.cfg.get("dropout", 0.2),
            pool_type=str(self.cfg.get("pool_type", "max")),
            head_hidden_dims=tuple(self.cfg.get("head_hidden_dims", (64, 32))),
            head_dropout=self.cfg.get("head_dropout", 0.1),
        ).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=self.cfg["lr"], weight_decay=self.cfg.get("weight_decay", 1e-4),
        )
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
                print(f"[mph_fusion seed={seed}] epoch {epoch:3d} | train {train_loss:.4f} | val {val_loss:.4f}")

        model.load_state_dict(best_state)
        test_loss = _evaluate_loss(model, test_loader, loss_fn, device)
        test_loss_per_target = dict(zip(label_names, _evaluate_per_target(model, test_loader, device).tolist()))
        print(f"\n[mph_fusion seed={seed}] test loss {test_loss:.4f}")

        adversarial_loss = None
        adversarial_loss_per_target = None
        if adv_split is not None:
            adv_targets_std = vihrs.apply_log_zscore(adv_split["targets"], label_norm).astype(np.float32)
            adv_lr_seq = vihrs.apply_zscore_global(adv_split["lr_seq"], lr_norm).astype(np.float32)
            adv_pi_img, _ = mph_pi.build_mph_tensor(adv_split["image_tensors"], channel_norms=channel_norms)
            adv_extra = vihrs.apply_log_zscore(adv_split["n_points"], n_norm).astype(np.float32)[:, None]
            adv_ds = TensorDataset(
                torch.from_numpy(adv_lr_seq), torch.from_numpy(adv_pi_img),
                torch.from_numpy(adv_extra), torch.from_numpy(adv_targets_std),
            )
            adv_loader = DataLoader(adv_ds, batch_size=self.cfg["batch_size"], shuffle=False)
            adversarial_loss = _evaluate_loss(model, adv_loader, loss_fn, device)
            adversarial_loss_per_target = dict(
                zip(label_names, _evaluate_per_target(model, adv_loader, device).tolist())
            )
            print(f"[mph_fusion seed={seed}] adversarial loss {adversarial_loss:.4f}")

        cfg_meta = {
            **self.cfg,
            "method": "mph_fusion",
            "channels": ["l_minus_r", "n_points"] + [f"mph_h{d}" for d in train_split["dims"]],
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
