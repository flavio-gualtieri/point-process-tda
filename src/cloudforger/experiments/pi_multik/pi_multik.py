# src/cloudforger/experiments/pi_multik/pi_multik.py

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, Subset, TensorDataset

from cloudforger.baselines import vihrs
from cloudforger.core.diagram import PersistenceDiagram
from cloudforger.core.io import intersect_seeds
from cloudforger.core.records import load_diagrams
from cloudforger.core.splits import train_val_test_indices
from cloudforger.encoders.encoder_bank import EncoderBank
from cloudforger.encoders.scaleconv_pi import ConvFusion
from cloudforger.experiments.base import register
from cloudforger.experiments.common import (
    MultiSourceExperiment,
    apply_zscore,
    fit_zscore,
    prepare_device,
    save_results,
)
from cloudforger.models.heads.paramest import ParameterEstimator
from cloudforger.training.train import evaluate, evaluate_per_target, train_one_epoch
from cloudforger.vectorization.persistence_images.calibrated import build_calibrated_imager
from cloudforger.vectorization.persistence_images.multi_channel import MultiChannelImager
from cloudforger.vectorization.scalar_features import REGISTRY as FEATURE_REGISTRY


def _load_pickle(path: Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def load_multik_split(
    k_values: list[int],
    diagram_paths: list[Path],
    clouds_path: Path,
    label_names: tuple[str, ...] | None,
    tag: str,
    homology_dims: tuple[int, ...] = (0, 1),
) -> dict[str, Any] | None:
    """Load and seed-align each k's diagrams.pkl (NOT a precomputed image
    file -- see module docstring). Persistence entropy is computed here,
    directly from the diagrams (calibration-free, so no leakage concern);
    persistence IMAGES are deliberately not built here -- that's
    build_pi_tensor's job, once a train/val/test split exists to calibrate
    against."""
    bundles = []
    for k, path in zip(k_values, diagram_paths):
        if not Path(path).exists():
            print(f"  [{tag}] missing {path} -- skipping split.")
            return None
        bundles.append(load_diagrams(path))  # (diagrams, bundle) per k

    seed_arrays = [np.asarray(bundle["seeds"]) for _, bundle in bundles]
    idx_per_k, common_seeds = intersect_seeds(seed_arrays)
    per_k_totals = {k: len(s) for k, s in zip(k_values, seed_arrays)}
    print(f"  [{tag}] {len(common_seeds)} clouds common to all k in {k_values} (per-k totals: {per_k_totals}).")

    # Select target columns by name -- label_names can carry deterministic
    # bookkeeping fields (e.g. edge_buffer) alongside the real targets. None
    # (no target_label_names configured) means "every label this process
    # produced" -- adapts to whatever process generated this data instead of
    # assuming a fixed target set.
    tda_label_names = list(bundles[0][1]["label_names"])
    if label_names is None:
        label_names = tuple(tda_label_names)
    missing = [name for name in label_names if name not in tda_label_names]
    if missing:
        raise KeyError(f"[{tag}] label_names {tda_label_names} is missing {missing} from {label_names}")
    col_idx = [tda_label_names.index(name) for name in label_names]

    targets = np.asarray(bundles[0][1]["labels"], dtype=float)[np.ix_(idx_per_k[0], col_idx)]
    for k, (_, bundle), idx in zip(k_values[1:], bundles[1:], idx_per_k[1:]):
        targets_k = np.asarray(bundle["labels"], dtype=float)[np.ix_(idx, col_idx)]
        assert np.allclose(targets, targets_k), (
            f"[{tag}] target mismatch between k={k_values[0]} and k={k} after seed alignment -- alignment bug."
        )

    entropy_feature = FEATURE_REGISTRY.build("persistence_entropy", homology_dims=homology_dims)
    diagrams_per_k: dict[int, list[PersistenceDiagram]] = {}
    entropy_cols: dict[str, np.ndarray] = {}
    for k, (diagrams, _bundle), idx in zip(k_values, bundles, idx_per_k):
        aligned = [diagrams[i] for i in idx]
        diagrams_per_k[k] = aligned
        per_diagram_entropy = [entropy_feature.compute(d) for d in aligned]
        for dim in homology_dims:
            entropy_cols[f"entropy{dim}_k{k}"] = np.array([e[dim] for e in per_diagram_entropy], dtype=np.float64)

    # n(x) doesn't depend on k (same underlying cloud) -- joined once from
    # the sibling clouds.pkl by seed.
    clouds = _load_pickle(clouds_path)
    n_points_by_seed = {int(c["seed"]): c["n_points"] for c in clouds}
    n_points = np.array([n_points_by_seed[int(s)] for s in common_seeds], dtype=np.float64)

    return {
        "diagrams_per_k": diagrams_per_k,
        "entropy_cols": entropy_cols,
        "n_points": n_points,
        "targets": targets,
        "seeds": common_seeds,
        "label_names": list(label_names),
    }


def build_pi_tensor(
    split: dict[str, Any],
    k_values: list[int],
    homology_dims: tuple[int, ...],
    resolution: int,
    sigma_pixels: float,
    coverage: float,
    train_idx: np.ndarray | None = None,
    imagers: list[MultiChannelImager] | None = None,
) -> tuple[np.ndarray, list[MultiChannelImager]]:
    fit = imagers is None
    if fit:
        if train_idx is None:
            raise ValueError("build_pi_tensor: train_idx is required when fitting (imagers=None).")
        imagers = []

    raw_channels: list[np.ndarray] = []
    for ki, k in enumerate(k_values):
        diagrams_k = split["diagrams_per_k"][k]
        if fit:
            calibration_diagrams = [diagrams_k[i] for i in train_idx]
            imager = build_calibrated_imager(
                calibration_diagrams, homology_dims=homology_dims, resolution=resolution,
                sigma_pixels=sigma_pixels, coverage=coverage, verbose=False,
            )
            imagers.append(imager)
        else:
            imager = imagers[ki]
        images = [imager.transform(d) for d in diagrams_k]
        for dim in homology_dims:
            raw_channels.append(np.stack([im[dim] for im in images]))

    # raw_channels is flat: [dim0_k0, dim1_k0, ..., dim0_k1, dim1_k1, ...]
    # -- group by k (D consecutive entries each) into (N, D, H, W), then
    # stack those groups along a new k axis.
    dims_per_k = len(raw_channels) // len(k_values)
    per_k = [np.stack(raw_channels[dims_per_k * i : dims_per_k * (i + 1)], axis=1) for i in range(len(k_values))]
    return np.stack(per_k, axis=1).astype(np.float32), imagers


def build_extra(
    split: dict[str, Any],
    train_idx: np.ndarray | None,
    n_norm: dict | None = None,
    entropy_norms: dict[str, dict[str, float]] | None = None,
    include_entropy: bool = False,
) -> tuple[np.ndarray, dict, dict[str, dict[str, float]]]:
    """(N, 1 + (n_entropy_cols if include_entropy else 0)) side-vector,
    concatenated onto the per-k embeddings before the fusion head: [log N]
    always, plus one plain z-scored column per split["entropy_cols"] entry
    (persistence entropy per (homology dim, k), loaded by load_multik_split)
    when include_entropy is set.

    Fit-once/apply-frozen (see module docstring): pass train_idx (n_norm and
    entropy_norms left None) to fit n(x)/entropy stats on
    split[...][train_idx] only, applied to every row of split -- the main
    population's train-only fit. Pass a previously-fit n_norm (and
    entropy_norms, if include_entropy) to apply them frozen to a different
    population instead (train_idx unused, pass None) -- the adversarial
    call."""
    fit = n_norm is None
    if fit:
        if train_idx is None:
            raise ValueError("build_extra: train_idx is required when fitting (n_norm=None).")
        n_norm = vihrs.fit_log_zscore(split["n_points"][train_idx])
    n_std = vihrs.apply_log_zscore(split["n_points"], n_norm).astype(np.float32)
    cols = [n_std]
    if fit:
        entropy_norms = {}
    if include_entropy:
        for name in sorted(split["entropy_cols"]):
            raw = split["entropy_cols"][name]
            if fit:
                norm = fit_zscore(raw[train_idx])
                entropy_norms[name] = norm
            else:
                norm = entropy_norms[name]
            cols.append(apply_zscore(raw, norm).astype(np.float32))
    return np.stack(cols, axis=1), n_norm, entropy_norms


class PIMultiK(nn.Module):
    """Late-fusion multi-k model, in two composable stages:

    1. EncoderBank turns each k's (H0, H1) image pair into a (B, K,
       embedding_dim) embedding sequence -- encoder_mode picks shared-weight
       (one CoordConvPIEncoder, k folded into the batch dim) vs independent
       (n_k separate CoordConvPIEncoders) -- see EncoderBank's docstring.
    2. That sequence is collapsed to one vector -- fusion_mode="concat" is a
       flat reshape (head_in grows with n_k); fusion_mode="conv" runs it
       through ConvFusion, whose fusion_pool ("avg" vs "flatten") picks
       scale-count-invariant pooling vs position-preserving flattening --
       see ConvFusion's docstring.

    Together these two axes cover what used to be three separate model
    classes (this file's old shared+concat, pi_multik_towers' independent+
    flatten) as one config-selectable combination -- e.g. independent+
    avg-pool or shared+flatten are now reachable without a new file. The
    fused vector is concatenated with the extra scalar features and passed
    through an MLP head."""

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
        encoder_mode: str = "shared",
        fusion_mode: str = "concat",
        fusion_pool: str = "avg",
        fusion_dropout: float = 0.0,
        scale_fusion_hidden: int = 128,
        scale_fusion_out_dim: int = 128,
        scale_fusion_kernel_size: int = 3,
        scale_fusion_dropout: float = 0.0,
        pool_type: str = "max",
    ):
        super().__init__()
        self.n_k = n_k
        self.fusion_mode = fusion_mode
        self.bank = EncoderBank(
            mode=encoder_mode, n_k=n_k, in_channels=in_channels, embedding_dim=embedding_dim,
            conv_channels=conv_channels, dropout=dropout, pool_type=pool_type,
        )
        if fusion_mode == "concat":
            self.fusion = None
            fused_dim = n_k * embedding_dim
        elif fusion_mode == "conv":
            self.fusion = ConvFusion(
                n_k=n_k, embedding_dim=embedding_dim, hidden=scale_fusion_hidden,
                out_dim=scale_fusion_out_dim, kernel_size=scale_fusion_kernel_size,
                dropout=scale_fusion_dropout, pool=fusion_pool,
            )
            fused_dim = self.fusion.out_dim
        else:
            raise ValueError(f"PIMultiK: fusion_mode must be 'concat' or 'conv', got {fusion_mode!r}.")
        # Applied after fusion, before the head, regardless of fusion_mode --
        # previously only pi_multik_towers had this (as an always-on
        # nn.Dropout); promoted here so it's available to any combination.
        self.fusion_dropout = nn.Dropout(fusion_dropout) if fusion_dropout > 0 else nn.Identity()

        self.head = ParameterEstimator(
            embedding_dim=fused_dim + n_extra, n_params=n_targets,
            hidden_dims=head_hidden_dims, dropout=head_dropout,
        )

    def forward(self, pi_imgs, extra):
        seq = self.bank(pi_imgs)  # (B, K, C)
        if self.fusion_mode == "concat":
            fused = seq.reshape(seq.shape[0], -1)
        else:
            fused = self.fusion(seq)
        fused = self.fusion_dropout(fused)
        return self.head(torch.cat([fused, extra], dim=1))


def _resolve_fusion_kwargs(cfg: dict) -> dict[str, Any]:
    """Translates method.params' encoder_mode/fusion_mode/fusion_pool
    overrides -- and the legacy use_fusion bool (matern_pi_multik.yaml's
    method: pi_multik still sets it: True reproduces what pi_multik_scaleconv
    defaults to, conv fusion with avg pooling; False reproduces plain
    concat) -- into PIMultiK constructor kwargs.

    Deliberately omits a key entirely when cfg doesn't set it, so each
    sibling experiment's own _build_model default (via kwargs.setdefault)
    still applies. Precedence: explicit encoder_mode/fusion_mode/fusion_pool
    cfg keys, then use_fusion, then the registered method's own default."""
    kwargs: dict[str, Any] = {}
    if "use_fusion" in cfg and "fusion_mode" not in cfg:
        kwargs["fusion_mode"] = "conv" if cfg["use_fusion"] else "concat"
        kwargs["fusion_pool"] = "avg"
    for key in ("encoder_mode", "fusion_mode", "fusion_pool"):
        if key in cfg:
            kwargs[key] = cfg[key]
    return kwargs


@register("pi_multik")
class PIMultiKExperiment(MultiSourceExperiment):
    file_keys = ("clouds", "images")

    @property
    def subdir(self) -> str:
        return "pi_multik"

    def _build_model(self, **kwargs) -> PIMultiK:
        """Shared-weight-encoder, flat-concat baseline -- the reference
        pi_multik design. Siblings (PIMultiKScaleConvExperiment,
        PIMultiKTowersExperiment) override just the encoder_mode/fusion_mode/
        fusion_pool defaults below -- everything else in run() (data
        loading, training loop, save_results) is shared verbatim."""
        kwargs.setdefault("encoder_mode", "shared")
        kwargs.setdefault("fusion_mode", "concat")
        return PIMultiK(**kwargs)

    def run(
        self,
        dataset_paths: dict[str, Any],
        output_dir: Path,
        adversarial_paths: dict[str, Any] | None = None,
    ) -> dict:
        # fetch configs
        label_names = tuple(self.cfg.get("target_label_names"))
        k_values = list(self.cfg["k_values"])
        homology_dims = tuple(self.cfg.get("homology_dims", (0, 1)))
        include_entropy = bool(self.cfg.get("include_entropy", False))
        resolution = int(self.cfg.get("resolution", 64))
        sigma_pixels = float(self.cfg.get("sigma_pixels", 2.0))
        coverage = float(self.cfg.get("pd_calibration_coverage", 0.99))
        seed = self.cfg["seed"]
        device = prepare_device(seed)

        # load and align diagrams and clouds by seed
        train_split = load_multik_split(
            k_values, list(dataset_paths["images"]), Path(dataset_paths["clouds"]), label_names, tag="train_test",
            homology_dims=homology_dims,
        )
        if train_split is None:
            raise FileNotFoundError(f"diagrams missing for some k in {k_values} under {dataset_paths['images']}.")

        adv_split = None
        if adversarial_paths is not None:
            adv_split = load_multik_split(
                k_values, list(adversarial_paths["images"]), Path(adversarial_paths["clouds"]),
                label_names, tag="adversarial", homology_dims=homology_dims,
            )

        n = len(train_split["targets"])
        train_idx, val_idx, test_idx = train_val_test_indices(n, seed)

        label_norm = vihrs.fit_log_zscore(train_split["targets"][train_idx])
        targets_std = vihrs.apply_log_zscore(train_split["targets"], label_norm).astype(np.float32)
        pi_img, imagers = build_pi_tensor(
            train_split, k_values, homology_dims=homology_dims, resolution=resolution,
            sigma_pixels=sigma_pixels, coverage=coverage, train_idx=train_idx,
        )
        extra, n_norm, entropy_norms = build_extra(train_split, train_idx, include_entropy=include_entropy)

        full_dataset = TensorDataset(
            torch.from_numpy(pi_img), torch.from_numpy(extra), torch.from_numpy(targets_std),
        )

        def _loader(idx: np.ndarray, shuffle: bool) -> DataLoader:
            return DataLoader(Subset(full_dataset, idx), batch_size=self.cfg["batch_size"], shuffle=shuffle)

        train_loader, val_loader, test_loader = _loader(train_idx, True), _loader(val_idx, False), _loader(test_idx, False)

        # PIMultiK.forward(pi_imgs, extra) lines up exactly with
        # cloudforger.training.train's (inputs, covariates, labels) 3-tuple batch
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
            scale_fusion_dropout=self.cfg.get("scale_fusion_dropout", 0.0),
            fusion_dropout=self.cfg.get("fusion_dropout", 0.0),
            pool_type=str(self.cfg.get("pool_type", "max")),
            **_resolve_fusion_kwargs(self.cfg),
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
            adv_pi_img, _ = build_pi_tensor(
                adv_split, k_values, homology_dims=homology_dims, resolution=resolution,
                sigma_pixels=sigma_pixels, coverage=coverage, imagers=imagers,
            )
            adv_extra, _, _ = build_extra(
                adv_split, None, n_norm=n_norm, entropy_norms=entropy_norms, include_entropy=include_entropy,
            )
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
            # Per-k calibrated imager params, so a seed's exact calibration
            # (birth_range/pers_range/sigma_pixels -- now seed-specific, see
            # module docstring) is recoverable from results.json alone.
            "imager_params": {k: imager.params for k, imager in zip(k_values, imagers)},
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
