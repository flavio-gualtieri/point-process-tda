# src/cloudforger/nn/experiments/pi_multik.py
"""Multi-k persistence-image model: per k in cfg["k_values"], the (H0, H1)
persistence-image pair is fed through a SHARED-WEIGHT CoordConv CNN branch
(cloudforger.nn.encoders.coordconv_pi.CoordConvPIEncoder) -- one encoder
instance, called once per k (folded into the batch dimension so it's a
single conv-stack invocation, not K separate ones) -- producing one
embedding f_k per k. The f_k's are late-fused by concatenation with
[log N, ...] extra scalar features, then passed through an MLP regression
head. This supersedes the old pi_multik design (all k's stacked into one
wide-channel tensor, seen jointly by a single conv from layer 1 -- i.e.
EARLY fusion across k) with a late-fusion, Siamese-style alternative; the
old implementation is preserved verbatim in experiments/delete.py.

The flat concat above is order-blind to the k axis. pi_multik_scaleconv.py
registers a scale-aware sibling experiment that reuses everything here
(PIMultiKExperiment.run, load_multik_split, build_pi_tensor/build_extra)
but overrides _build_model to swap in ScaleConvFusion
(cloudforger.nn.encoders.scaleconv_pi), a small Conv1d block over the
ordered k axis, before the head -- see PIMultiK(use_fusion=...).

Channel order per k (documented here since nothing in the tensor itself
labels it): [pi_d(k) for d in homology_dims], homology_dims configurable via
cfg["homology_dims"] (default (0, 1)). The per-k image tensor built by
build_pi_tensor has shape (N, K, D, R, R), K = len(k_values), D =
len(homology_dims), e.g. for k_values = [5, 10, 15], homology_dims = (0, 1):
    slice [:, 0] = (H0, H1) at k=5     slice [:, 1] = (H0, H1) at k=10
    slice [:, 2] = (H0, H1) at k=15

Not an Experiment subclass, for the same reason fusion.py isn't: each k's
images_dtm_k<k>.pkl survives the empty-diagram filter with a DIFFERENT
subset of seeds, so the files must be intersected by seed
(cloudforger.core.io.intersect_seeds) before their channels can be stacked,
which Experiment.run()'s single dataset_path contract has no hook for.
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
from cloudforger.core.io import intersect_seeds
from cloudforger.core.splits import train_val_test_indices
from cloudforger.nn.encoders.coordconv_pi import CoordConvPIEncoder
from cloudforger.nn.encoders.scaleconv_pi import ScaleConvFusion
from cloudforger.nn.experiments.base import register
from cloudforger.nn.experiments.common import (
    MultiSourceExperiment,
    apply_zscore,
    fit_zscore,
    prepare_device,
    save_results,
)
from cloudforger.nn.heads.paramest import ParameterEstimator
from cloudforger.nn.train import evaluate, evaluate_per_target, train_one_epoch


def _load_pickle(path: Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def load_multik_split(
    k_values: list[int],
    image_paths: list[Path],
    clouds_path: Path,
    label_names: tuple[str, ...] | None,
    tag: str,
    homology_dims: tuple[int, ...] = (0, 1),
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
    # bookkeeping fields (e.g. edge_buffer) alongside the real targets. None
    # (no target_label_names configured) means "every label this process
    # produced" -- adapts to whatever process generated this data instead of
    # assuming a fixed target set.
    tda_label_names = list(payloads[0]["label_names"])
    if label_names is None:
        label_names = tuple(tda_label_names)
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

    # Channel order: for k in k_values, [pi_d(k) for d in homology_dims] --
    # see module docstring. homology_dims must be a subset of what was
    # actually computed at featurize time (payload["image_tensors"].keys()).
    pi_channels: list[np.ndarray] = []
    entropy_cols: dict[str, np.ndarray] = {}
    for k, payload, idx in zip(k_values, payloads, idx_per_k):
        available = set(payload["image_tensors"].keys())
        missing_dims = [d for d in homology_dims if d not in available]
        if missing_dims:
            raise KeyError(
                f"[{tag}] homology_dims {homology_dims} requests dims {missing_dims} not present in "
                f"image_tensors for k={k} (available: {sorted(available)}) -- these must have been computed "
                f"by the featurize step (features.persistence_image.params.homology_dims)."
            )
        for dim in homology_dims:
            pi_channels.append(np.asarray(payload["image_tensors"][dim], dtype=np.float64)[idx])
            entropy_cols[f"entropy{dim}_k{k}"] = np.asarray(payload["persistence_entropy"][dim], dtype=np.float64)[idx]

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
        "label_names": list(label_names),
    }


def build_pi_tensor(
    split: dict[str, Any], k_values: list[int], channel_norms: list[dict[str, float]] | None
) -> tuple[np.ndarray, list[dict[str, float]]]:
    """(N, K, D, H, W) float32 tensor, K = len(k_values), D = number of
    homology dims selected (split["pi_channels"] has K*D entries, D per k,
    in homology_dims order -- see load_multik_split) -- unlike the old
    pi_multik's (N, D*K, H, W) layout (all k's flattened into one channel
    axis), k is kept as its own axis here so the shared-weight CoordConv
    branch can fold it into the batch dimension for a single conv-stack call
    per forward pass. Still one independent z-score per raw (k, dim)
    channel -- PI pixels are raw Gaussian-kernel sums with a huge dynamic
    range, and different k's have different raw magnitude scales on top of
    that, so per-channel (not global) normalization matters here more than
    it would for a single k."""
    fit = channel_norms is None
    if fit:
        channel_norms = []
    normed_channels = []
    for i, raw in enumerate(split["pi_channels"]):
        norm = fit_zscore(raw) if fit else channel_norms[i]
        if fit:
            channel_norms.append(norm)
        normed_channels.append(apply_zscore(raw, norm))
    # normed_channels is flat: [dim0_k0, dim1_k0, ..., dim0_k1, dim1_k1, ...]
    # -- group by k (D consecutive entries each) into (N, D, H, W), then
    # stack those groups along a new k axis.
    dims_per_k = len(normed_channels) // len(k_values)
    per_k = [np.stack(normed_channels[dims_per_k * i : dims_per_k * (i + 1)], axis=1) for i in range(len(k_values))]
    return np.stack(per_k, axis=1).astype(np.float32), channel_norms


def build_extra(
    split: dict[str, Any],
    n_norm: dict,
    entropy_norms: dict[str, dict[str, float]] | None = None,
    include_entropy: bool = False,
) -> tuple[np.ndarray, dict[str, dict[str, float]]]:
    """(N, 1 + (n_entropy_cols if include_entropy else 0)) side-vector,
    concatenated onto the per-k embeddings before the fusion head: [log N]
    always, plus one plain z-scored column per split["entropy_cols"] entry
    (persistence entropy per (homology dim, k), loaded by load_multik_split)
    when include_entropy is set. entropy_norms is None to fit (train split),
    or the train split's returned norms to apply frozen (val/test/adversarial)."""
    n_std = vihrs.apply_log_zscore(split["n_points"], n_norm).astype(np.float32)
    cols = [n_std]
    fit = entropy_norms is None
    if fit:
        entropy_norms = {}
    if include_entropy:
        for name in sorted(split["entropy_cols"]):
            raw = split["entropy_cols"][name]
            norm = fit_zscore(raw) if fit else entropy_norms[name]
            if fit:
                entropy_norms[name] = norm
            cols.append(apply_zscore(raw, norm).astype(np.float32))
    return np.stack(cols, axis=1), entropy_norms


class PIMultiK(nn.Module):
    """SHARED-WEIGHT CoordConv branch: one CoordConvPIEncoder instance,
    applied independently to each k's (H0, H1) image pair (k folded into
    the batch dimension, so it's one conv-stack call per forward, not K),
    producing f_k5, f_k10, f_k15, ... Those are late-fused by concatenation
    with the extra scalar features, then passed through an MLP fusion head.
    """

    def __init__(
        self,
        in_channels: int,
        embedding_dim: int,
        n_k: int,
        n_extra: int,
        n_targets: int,
        conv_channels: tuple[int, ...] = (32, 64, 128),
        dropout: float = 0.2,
        head_hidden_dims: tuple[int, ...] = (64, 32),
        head_dropout: float = 0.1,
        use_fusion: bool = False,
        scale_fusion_hidden: int = 128,
        scale_fusion_out_dim: int = 128,
        scale_fusion_kernel_size: int = 3,
        pool_type: str = "max",
    ):
        super().__init__()
        self.n_k = n_k
        self.use_fusion = use_fusion
        self.encoder = CoordConvPIEncoder(
            in_channels=in_channels, embedding_dim=embedding_dim,
            conv_channels=conv_channels, dropout=dropout,
            pool_type=pool_type,
        )
        if self.use_fusion:
            self.scale_fusion = ScaleConvFusion(
                embedding_dim, hidden=scale_fusion_hidden, out_dim=scale_fusion_out_dim,
                kernel_size=scale_fusion_kernel_size,
            )
            head_in = self.scale_fusion.out_dim + n_extra
        else:
            head_in = n_k * embedding_dim + n_extra

        self.head = ParameterEstimator(
            embedding_dim=head_in, n_params=n_targets,
            hidden_dims=head_hidden_dims, dropout=head_dropout,
        )

    def forward(self, pi_imgs, extra):
        b = pi_imgs.shape[0]
        flat = pi_imgs.reshape(b * self.n_k, *pi_imgs.shape[2:])
        emb = self.encoder(flat)                                   # (B*K, C)
        if self.use_fusion:
            seq = emb.reshape(b, self.n_k, self.encoder.embedding_dim)  # (B, K, C)
            pooled = self.scale_fusion(seq)                        # (B, out_dim)
        else:
            pooled = emb.reshape(b, self.n_k * self.encoder.embedding_dim)
        return self.head(torch.cat([pooled, extra], dim=1))


@register("pi_multik")
class PIMultiKExperiment(MultiSourceExperiment):
    file_keys = ("clouds", "images")

    @property
    def subdir(self) -> str:
        return "pi_multik"

    def _build_model(self, **kwargs) -> PIMultiK:
        """Flat-concat baseline. Overridden by PIMultiKScaleConvExperiment
        to swap in ScaleConvFusion instead -- everything else in run()
        (data loading, training loop, save_results) is shared verbatim."""
        return PIMultiK(use_fusion=False, **kwargs)

    def run(
        self,
        dataset_paths: dict[str, Any],
        output_dir: Path,
        adversarial_paths: dict[str, Any] | None = None,
    ) -> dict:
        # None (no target_label_names in the YAML) lets load_multik_split
        # adapt to every label the process's data actually carries.
        target_label_names = self.cfg.get("target_label_names")
        label_names = tuple(target_label_names) if target_label_names else None
        k_values = list(self.cfg["k_values"])
        # Channel selection: which persistence homology dims to stack per k/m
        # (must be a subset of what features.persistence_image.params.homology_dims
        # actually computed at featurize time); include_entropy additionally
        # appends per-(dim, k) persistence entropy scalars to the extra side-vector.
        homology_dims = tuple(self.cfg.get("homology_dims", (0, 1)))
        include_entropy = bool(self.cfg.get("include_entropy", False))
        seed = self.cfg["seed"]
        device = prepare_device(seed)

        train_split = load_multik_split(
            k_values, list(dataset_paths["images"]), Path(dataset_paths["clouds"]), label_names, tag="train_test",
            homology_dims=homology_dims,
        )
        if train_split is None:
            raise FileNotFoundError(f"images missing for some k in {k_values} under {dataset_paths['images']}.")
        label_names = tuple(train_split["label_names"])

        adv_split = None
        if adversarial_paths is not None:
            adv_split = load_multik_split(
                k_values, list(adversarial_paths["images"]), Path(adversarial_paths["clouds"]),
                label_names, tag="adversarial", homology_dims=homology_dims,
            )

        n = len(train_split["targets"])
        label_norm = vihrs.fit_log_zscore(train_split["targets"])
        n_norm = vihrs.fit_log_zscore(train_split["n_points"])

        targets_std = vihrs.apply_log_zscore(train_split["targets"], label_norm).astype(np.float32)
        pi_img, channel_norms = build_pi_tensor(train_split, k_values, channel_norms=None)
        extra, entropy_norms = build_extra(train_split, n_norm, include_entropy=include_entropy)

        full_dataset = TensorDataset(
            torch.from_numpy(pi_img), torch.from_numpy(extra), torch.from_numpy(targets_std),
        )
        train_idx, val_idx, test_idx = train_val_test_indices(n, seed)

        def _loader(idx: np.ndarray, shuffle: bool) -> DataLoader:
            return DataLoader(Subset(full_dataset, idx), batch_size=self.cfg["batch_size"], shuffle=shuffle)

        train_loader, val_loader, test_loader = _loader(train_idx, True), _loader(val_idx, False), _loader(test_idx, False)

        # PIMultiK.forward(pi_imgs, extra) lines up exactly with
        # cloudforger.nn.train's (inputs, covariates, labels) 3-tuple batch
        # convention, so the shared train loop applies as-is.
        model = self._build_model(
            in_channels=len(homology_dims),
            embedding_dim=self.cfg["embedding_dim"],
            n_k=len(k_values),
            n_extra=extra.shape[1],
            n_targets=len(label_names),
            conv_channels=tuple(self.cfg.get("conv_channels", (32, 64, 128))),
            dropout=self.cfg.get("dropout", 0.2),
            head_hidden_dims=tuple(self.cfg.get("head_hidden_dims", (64, 32))),
            head_dropout=self.cfg.get("head_dropout", 0.1),
            scale_fusion_hidden=self.cfg.get("scale_fusion_hidden", 128),
            scale_fusion_out_dim=self.cfg.get("scale_fusion_out_dim", 128),
            scale_fusion_kernel_size=self.cfg.get("scale_fusion_kernel_size", 3),
            pool_type=str(self.cfg.get("pool_type", "max"))
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.cfg.get("lr", 1e-3), weight_decay=self.cfg.get("weight_decay", 1e-4))
        # optimizer = torch.optim.Adam(model.parameters(), lr=self.cfg.get("lr", 1e-3), weight_decay=self.cfg.get("weight_decay", 1e-4))
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
            adv_pi_img, _ = build_pi_tensor(adv_split, k_values, channel_norms=channel_norms)
            adv_extra, _ = build_extra(adv_split, n_norm, entropy_norms=entropy_norms, include_entropy=include_entropy)
            adv_ds = TensorDataset(
                torch.from_numpy(adv_pi_img), torch.from_numpy(adv_extra), torch.from_numpy(adv_targets_std),
            )
            adv_loader = DataLoader(adv_ds, batch_size=self.cfg["batch_size"], shuffle=False)
            adversarial_loss, _ = evaluate(model, adv_loader, loss_fn, device)
            adversarial_loss_per_target = dict(
                zip(label_names, evaluate_per_target(model, adv_loader, device).tolist())
            )
            print(f"[{self.tag} seed={seed}] adversarial loss {adversarial_loss:.4f}")

        cfg_meta = {
            **self.cfg,
            "channels": [f"k{k}_h{d}" for k in k_values for d in homology_dims],
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
