# src/cloudforger/nn/experiments/pi_multik_fusion.py
"""Fusion of vihrs's L(r)-r + n(x) branch with pi_multik's stacked
multi-k persistence-image branch, late-fused by concatenating both branch
embeddings before a shared regression head -- same "concatenate before the
head" pattern fusion.py's own (lr + pi + betti) FusionCNN uses, minus the
betti branch. Absorbed from dtm_experiment/pi_multik_fusion_model.py.

Reuses, rather than reimplements:
  - pi_multik.load_multik_split for the image side, so it inherits that
    function's k-dependent seed-intersection handling.
  - cloudforger.core.io.align_seeds for the join against vihrs's L(r)-r
    population (every cloud, never DTM-filtered) -- the same join fusion.py
    itself needs, for the same reason (different, filtered population sizes).
  - pi_multik.build_pi_tensor / build_extra for per-channel image z-scoring
    and the n(x)+entropy scalar side-channel -- both operate on any dict
    with "pi_channels"/"entropy_cols"/"n_points" keys, which is exactly the
    schema load_fusion_split below returns.
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
from cloudforger.nn.encoders.persistence_image import PIEncoder
from cloudforger.nn.experiments import pi_multik
from cloudforger.nn.experiments.base import register
from cloudforger.nn.experiments.common import MultiSourceExperiment, prepare_device, save_results


def load_fusion_split(
    k_values: list[int],
    image_paths: list[Path],
    clouds_path: Path,
    lr_features: dict[str, np.ndarray],
    label_names: tuple[str, ...],
    tag: str,
) -> dict[str, Any] | None:
    multik_split = pi_multik.load_multik_split(k_values, image_paths, clouds_path, label_names, tag=tag)
    if multik_split is None:
        return None

    multik_idx, lr_idx = align_seeds(multik_split["seeds"], lr_features["cloud_seeds"])
    print(f"  [{tag}] {len(multik_idx)}/{len(multik_split['seeds'])} pi_multik clouds matched to L(r)-r clouds.")

    targets_multik = multik_split["targets"][multik_idx]
    targets_lr = lr_features["targets"][lr_idx]
    assert np.allclose(targets_multik, targets_lr), f"[{tag}] target mismatch after seed alignment -- alignment bug."

    return {
        "pi_channels": [ch[multik_idx] for ch in multik_split["pi_channels"]],
        "entropy_cols": {name: values[multik_idx] for name, values in multik_split["entropy_cols"].items()},
        "n_points": multik_split["n_points"][multik_idx],
        "targets": targets_multik,
        "lr_seq": lr_features["l_minus_r"][lr_idx],
    }


class VihrsPIMultiKFusion(nn.Module):
    def __init__(self, lr_seq_len: int, in_channels: int, embedding_dim: int, n_extra: int, n_targets: int):
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

        # Multi-k persistence-image branch -- identical to pi_multik's own.
        self.pi_encoder = PIEncoder(in_channels=in_channels, embedding_dim=embedding_dim)

        merge_dim = 2 * embedding_dim + n_extra
        self.head = nn.Sequential(
            nn.Linear(merge_dim, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(32, n_targets),
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


@register("pi_multik_fusion")
class PIMultiKFusionExperiment(MultiSourceExperiment):
    file_keys = ("clouds", "images")

    @property
    def subdir(self) -> str:
        return "pi_multik_fusion"

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

        lr_data = vihrs.prepare_data(
            Path(dataset_paths["clouds"]),
            Path(adversarial_paths["clouds"]) if adversarial_paths else None,
            label_names=label_names,
        )
        train_split = load_fusion_split(
            k_values, list(dataset_paths["images"]), Path(dataset_paths["clouds"]),
            lr_data["train_features"], label_names, tag="train_test",
        )
        if train_split is None:
            raise FileNotFoundError(f"images missing for some k in {k_values} under {dataset_paths['images']}.")

        adv_split = None
        if adversarial_paths is not None and lr_data["adversarial_features"] is not None:
            adv_split = load_fusion_split(
                k_values, list(adversarial_paths["images"]), Path(adversarial_paths["clouds"]),
                lr_data["adversarial_features"], label_names, tag="adversarial",
            )

        n = len(train_split["targets"])
        label_norm = vihrs.fit_log_zscore(train_split["targets"])
        n_norm = vihrs.fit_log_zscore(train_split["n_points"])
        lr_norm = vihrs.fit_zscore_global(train_split["lr_seq"])

        targets_std = vihrs.apply_log_zscore(train_split["targets"], label_norm).astype(np.float32)
        lr_seq = vihrs.apply_zscore_global(train_split["lr_seq"], lr_norm).astype(np.float32)
        pi_img, channel_norms = pi_multik.build_pi_tensor(train_split, channel_norms=None)
        extra, entropy_norms = pi_multik.build_extra(train_split, n_norm, entropy_norms=None, k_values=k_values)

        full_dataset = TensorDataset(
            torch.from_numpy(lr_seq), torch.from_numpy(pi_img), torch.from_numpy(extra), torch.from_numpy(targets_std),
        )
        train_idx, val_idx, test_idx = train_val_test_indices(n, seed)

        def _loader(idx: np.ndarray, shuffle: bool) -> DataLoader:
            return DataLoader(Subset(full_dataset, idx), batch_size=self.cfg["batch_size"], shuffle=shuffle)

        train_loader, val_loader, test_loader = _loader(train_idx, True), _loader(val_idx, False), _loader(test_idx, False)

        model = VihrsPIMultiKFusion(
            lr_seq_len=lr_seq.shape[1], in_channels=pi_img.shape[1], embedding_dim=self.cfg["embedding_dim"],
            n_extra=extra.shape[1], n_targets=len(label_names),
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
                print(f"[pi_multik_fusion seed={seed}] epoch {epoch:3d} | train {train_loss:.4f} | val {val_loss:.4f}")

        model.load_state_dict(best_state)
        test_loss = _evaluate_loss(model, test_loader, loss_fn, device)
        test_loss_per_target = dict(zip(label_names, _evaluate_per_target(model, test_loader, device).tolist()))
        print(f"\n[pi_multik_fusion seed={seed}] test loss {test_loss:.4f}")

        adversarial_loss = None
        adversarial_loss_per_target = None
        if adv_split is not None:
            adv_targets_std = vihrs.apply_log_zscore(adv_split["targets"], label_norm).astype(np.float32)
            adv_lr_seq = vihrs.apply_zscore_global(adv_split["lr_seq"], lr_norm).astype(np.float32)
            adv_pi_img, _ = pi_multik.build_pi_tensor(adv_split, channel_norms=channel_norms)
            adv_extra, _ = pi_multik.build_extra(adv_split, n_norm, entropy_norms=entropy_norms, k_values=k_values)
            adv_ds = TensorDataset(
                torch.from_numpy(adv_lr_seq), torch.from_numpy(adv_pi_img),
                torch.from_numpy(adv_extra), torch.from_numpy(adv_targets_std),
            )
            adv_loader = DataLoader(adv_ds, batch_size=self.cfg["batch_size"], shuffle=False)
            adversarial_loss = _evaluate_loss(model, adv_loader, loss_fn, device)
            adversarial_loss_per_target = dict(
                zip(label_names, _evaluate_per_target(model, adv_loader, device).tolist())
            )
            print(f"[pi_multik_fusion seed={seed}] adversarial loss {adversarial_loss:.4f}")

        cfg_meta = {
            **self.cfg,
            "method": "pi_multik_fusion",
            "channels": ["l_minus_r", "n_points"] + [f"k{k}_h{d}" for k in k_values for d in (0, 1)],
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
