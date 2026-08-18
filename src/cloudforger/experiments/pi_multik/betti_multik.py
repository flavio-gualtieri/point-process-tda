# src/cloudforger/experiments/pi_multik/betti_multik.py
"""Late-fusion multi-k model over Betti curves + the derived Euler-
characteristic curve -- the direct sibling of pi_multik.py, swapping
persistence images (a 2D, per-k CoordConv branch) for 1D curves (a per-k
Vihrs-style Conv1D branch), so the two are directly comparable: same
k_values/seeds/splits, same train_idx-fit/apply-frozen calibration
discipline, same late-fusion-across-k design, same shared regression head.

Per k, per homology dim d in homology_dims, one Betti curve beta_d(t) is
computed on a shared, per-k-calibrated filtration-value grid (see
build_calibrated_betti -- ONE grid_range across every d, not one per d,
which is what makes the Euler-characteristic channel below well-defined);
the Euler characteristic curve chi(t) = sum_d (-1)^d * beta_d(t) is then a
literal pointwise combination of the others, not a separate calibration.
Channel order per k: [beta_{d} for d in homology_dims] + ([chi] if
include_euler else []) -- e.g. for homology_dims=(0, 1), include_euler=True
(both defaults): [beta_0, beta_1, chi]. method.params.include_euler: false
drops that last channel (method.params.homology_dims also controls which
beta_d channels exist at all -- e.g. homology_dims=[0] alone for an H0-only
run); in_channels is sized accordingly, so the conv stack's first layer
always matches whatever channel count this run actually produces. That
(grid_size,)-per-channel bundle is run through SequenceEncoderBank's per-k
Vihrs-style Conv1D stack
(cloudforger.encoders.sequence_bank -- Conv1d(in_channels,64,7)-Pool-x2 +
Conv1d(64,64,7), matching Vihrs (2022)'s own L(r)-r architecture) BEFORE
reaching the shared regression head -- replacing betti.py's old design,
which fed the raw curve vector straight into a dense StatsEncoder MLP with
no conv stage at all. See sequence_bank.py's docstring for the one
deliberate hyperparameter deviation from the paper (pool_size).

REPLACES, ENTIRELY, the old betti/betti_cnn/ph_combined/fusion(betti
variant) pipeline:
  - betti.py (StatsEncoder, no conv) / betti_cnn.py (SequenceCNNEncoder,
    single k, single filtration) computed Betti-curve calibration
    (build_calibrated_betti_curves, since renamed/replaced -- see
    vectorization/scalar_features/calibrated.py) against the ENTIRE
    train_test population in scripts/featurize.py, before any train/val/
    test split existed -- the exact persistence-image leakage bug
    pi_multik.py's own module docstring documents, just never fixed on the
    Betti side. Both experiment classes, and the featurize.py handler that
    fed them (a precomputed, shared betti_curve.pkl), are deleted; nothing
    reads that pathway anymore.
  - ph_combined.py (PI + Betti fusion) and fusion.py's old "betti"/"full"
    variants depended on that same precomputed, leaky betti_curve.pkl.
    ph_combined.py is deleted outright (its entire premise was PI+Betti
    fusion); fusion.py drops its betti branch (see that file).
This file computes Betti/Euler curves on the fly from the same cached
per-k diagrams.pkl pi_multik.py already reads (load_multik_split, reused
unchanged below), calibrated fresh per seed on that seed's train_idx rows
only -- there is no separate betti-specific data file to precompute or go
stale.

Reuses, rather than reimplements (same convention pi_multik_fusion.py
already follows for the same reason):
  - pi_multik.load_multik_split for the diagram side -- k-dependent seed
    intersection, target alignment, and n(x) join are all filtration/
    vectorization-agnostic.
  - pi_multik.build_extra for the [log N(x), optional per-(dim,k) entropy]
    side-vector -- likewise agnostic to what the main per-k tensor holds.
Only the per-k tensor itself (build_betti_tensor here vs build_pi_tensor
there) and the model (BettiMultiK here vs PIMultiK there) differ."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, Subset, TensorDataset

from cloudforger.baselines import vihrs
from cloudforger.core.splits import train_val_test_indices
from cloudforger.encoders.scaleconv_pi import ConvFusion
from cloudforger.encoders.sequence_bank import SequenceEncoderBank
from cloudforger.experiments.base import register
from cloudforger.experiments.common import MultiSourceExperiment, prepare_device, save_results
from cloudforger.experiments.pi_multik import pi_multik
from cloudforger.models.heads.paramest import ParameterEstimator
from cloudforger.training.train import evaluate, evaluate_per_target, train_one_epoch
from cloudforger.vectorization.scalar_features.betti_curve import BettiCurve
from cloudforger.vectorization.scalar_features.calibrated import build_calibrated_betti, euler_characteristic_curve


def build_betti_tensor(
    split: dict[str, Any],
    k_values: list[int],
    homology_dims: tuple[int, ...],
    grid_size: int,
    coverage: float,
    range_pad: float = 1.1,
    include_euler: bool = True,
    train_idx: np.ndarray | None = None,
    betti_features: list[BettiCurve] | None = None,
) -> tuple[np.ndarray, list[BettiCurve]]:
    """(N, K, C, grid_size) float32 tensor, K = len(k_values), C =
    len(homology_dims) Betti channels plus one Euler-characteristic channel
    if include_euler (else just the len(homology_dims) Betti channels) --
    channel order [beta_{d} for d in homology_dims] + ([euler] if
    include_euler else []) per k (see module docstring). The PI counterpart's
    per-channel pixel z-score
    (build_pi_tensor) has no analogue here: a Betti curve's values are
    already bounded, small counts (0..a few dozen simultaneously-alive
    features), not PI's raw Gaussian-kernel-sum pixel magnitudes, so the
    conv stack's own BatchNorm-free but bias-carrying first layer absorbs
    the scale directly -- consistent with how build_extra's z-scored extras
    are the only explicit normalization on this branch, same as pi_multik.

    Fit-once/apply-frozen, same convention as build_pi_tensor:
      - Fit (betti_features=None): train_idx selects which rows of
        split["diagrams_per_k"][k] calibrate each k's BettiCurve
        (build_calibrated_betti -- ONE shared grid_range across every dim,
        not per-dim, see that function's docstring) -- the main
        population's train-only fit. Every row of split (train AND
        val/test) is still transformed and returned, just not used to fit.
      - Apply (betti_features=list[BettiCurve], train_idx unused/None):
        every row of split is transformed through the given, already-fit
        BettiCurve per k -- the adversarial-population call."""
    fit = betti_features is None
    if fit:
        if train_idx is None:
            raise ValueError("build_betti_tensor: train_idx is required when fitting (betti_features=None).")
        betti_features = []

    per_k: list[np.ndarray] = []
    for ki, k in enumerate(k_values):
        diagrams_k = split["diagrams_per_k"][k]
        if fit:
            calibration_diagrams = [diagrams_k[i] for i in train_idx]
            betti = build_calibrated_betti(
                calibration_diagrams, homology_dims=homology_dims, grid_size=grid_size,
                coverage=coverage, range_pad=range_pad,
            )
            betti_features.append(betti)
        else:
            betti = betti_features[ki]

        per_diagram_curves = [betti.compute(d).curves for d in diagrams_k]  # list[{dim: (grid_size,)}]
        stacked = {dim: np.stack([c[dim] for c in per_diagram_curves]) for dim in homology_dims}  # {dim: (N, grid_size)}
        channel_list = [stacked[dim] for dim in homology_dims]
        if include_euler:
            channel_list.append(euler_characteristic_curve(stacked))  # (N, grid_size)
        channels = np.stack(channel_list, axis=1)  # (N, C, grid_size)
        per_k.append(channels)

    return np.stack(per_k, axis=1).astype(np.float32), betti_features  # (N, K, C, grid_size)


class BettiMultiK(nn.Module):
    """Late-fusion multi-k model, structurally identical to pi_multik.py's
    PIMultiK -- same two composable fusion axes (encoder_mode: shared- vs
    independent-weight per-k encoder; fusion_mode/fusion_pool: flat concat
    vs ConvFusion's avg-pool/flatten Conv1d-over-k) -- with the per-k
    encoder swapped from EncoderBank (Conv2D over persistence images) to
    SequenceEncoderBank (Conv1D over Betti/Euler curves, see
    encoders/sequence_bank.py). ConvFusion itself is reused verbatim: it
    only ever consumes an already-embedded (B, K, embedding_dim) sequence,
    so the k-axis fusion stage doesn't need to know or care whether those
    embeddings came from a 2D image encoder or a 1D curve encoder."""

    def __init__(
        self,
        in_channels: int,
        grid_size: int,
        embedding_dim: int,
        n_k: int,
        n_extra: int,
        n_targets: int,
        n_filters: int = 64,
        kernel_size: int = 7,
        pool_size: int = 2,
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
    ):
        super().__init__()
        self.n_k = n_k
        self.fusion_mode = fusion_mode
        self.bank = SequenceEncoderBank(
            mode=encoder_mode, n_k=n_k, in_channels=in_channels, input_dim=grid_size, embedding_dim=embedding_dim,
            n_filters=n_filters, kernel_size=kernel_size, pool_size=pool_size, dropout=dropout,
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
            raise ValueError(f"BettiMultiK: fusion_mode must be 'concat' or 'conv', got {fusion_mode!r}.")
        self.fusion_dropout = nn.Dropout(fusion_dropout) if fusion_dropout > 0 else nn.Identity()

        self.head = ParameterEstimator(
            embedding_dim=fused_dim + n_extra, n_params=n_targets,
            hidden_dims=head_hidden_dims, dropout=head_dropout,
        )

    def forward(self, betti_seqs, extra):
        seq = self.bank(betti_seqs)  # (B, K, C)
        if self.fusion_mode == "concat":
            fused = seq.reshape(seq.shape[0], -1)
        else:
            fused = self.fusion(seq)
        fused = self.fusion_dropout(fused)
        return self.head(torch.cat([fused, extra], dim=1))


@register("betti_multik")
class BettiMultiKExperiment(MultiSourceExperiment):
    file_keys = ("clouds", "images")  # "images" is diagrams.pkl per k -- see pi_multik.py's module docstring

    @property
    def subdir(self) -> str:
        # "betti_multik" (unchanged) for the both-dims case -- every earlier
        # betti_multik config used homology_dims=(0, 1), so this keeps their
        # results paths stable/overwritable rather than silently forking a
        # new directory. H0-only and H1-only runs are NEW variants that
        # would otherwise collide with each other (and with the both-dims
        # results) if they all landed in "betti_multik/" too, so they get
        # their own results subdir instead.
        dims = tuple(self.cfg.get("homology_dims", (0, 1)))
        if dims == (0,):
            return "betti_multik_h0"
        if dims == (1,):
            return "betti_multik_h1"
        return "betti_multik"

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
        # Euler characteristic is on by default (every existing
        # betti_multik config predates this flag and relied on it always
        # being there); set method.params.include_euler: false to drop it.
        include_euler = bool(self.cfg.get("include_euler", True))
        grid_size = int(self.cfg.get("grid_size", 128))
        coverage = float(self.cfg.get("pd_calibration_coverage", 0.99))
        range_pad = float(self.cfg.get("range_pad", 1.1))
        seed = self.cfg["seed"]
        device = prepare_device(seed)

        # load and align diagrams and clouds by seed -- identical to
        # pi_multik.py's own load, reused rather than reimplemented (see
        # module docstring).
        train_split = pi_multik.load_multik_split(
            k_values, list(dataset_paths["images"]), Path(dataset_paths["clouds"]), label_names, tag="train_test",
            homology_dims=homology_dims,
        )
        if train_split is None:
            raise FileNotFoundError(f"diagrams missing for some k in {k_values} under {dataset_paths['images']}.")

        adv_split = None
        if adversarial_paths is not None:
            adv_split = pi_multik.load_multik_split(
                k_values, list(adversarial_paths["images"]), Path(adversarial_paths["clouds"]),
                label_names, tag="adversarial", homology_dims=homology_dims,
            )

        n = len(train_split["targets"])
        train_idx, val_idx, test_idx = train_val_test_indices(n, seed)

        label_norm = vihrs.fit_log_zscore(train_split["targets"][train_idx])
        targets_std = vihrs.apply_log_zscore(train_split["targets"], label_norm).astype(np.float32)
        betti_seq, betti_features = build_betti_tensor(
            train_split, k_values, homology_dims=homology_dims, grid_size=grid_size,
            coverage=coverage, range_pad=range_pad, include_euler=include_euler, train_idx=train_idx,
        )
        extra, n_norm, entropy_norms = pi_multik.build_extra(train_split, train_idx, include_entropy=include_entropy)

        full_dataset = TensorDataset(
            torch.from_numpy(betti_seq), torch.from_numpy(extra), torch.from_numpy(targets_std),
        )

        def _loader(idx: np.ndarray, shuffle: bool) -> DataLoader:
            return DataLoader(Subset(full_dataset, idx), batch_size=self.cfg["batch_size"], shuffle=shuffle)

        train_loader, val_loader, test_loader = _loader(train_idx, True), _loader(val_idx, False), _loader(test_idx, False)

        # BettiMultiK.forward(betti_seqs, extra) lines up exactly with
        # cloudforger.training.train's (inputs, covariates, labels) 3-tuple
        # batch convention, so the shared train loop applies as-is.
        model = BettiMultiK(
            in_channels=len(homology_dims) + (1 if include_euler else 0),  # betas [+ euler characteristic]
            grid_size=grid_size,
            embedding_dim=self.cfg["embedding_dim"],
            n_k=len(k_values),
            n_extra=extra.shape[1],
            n_targets=len(label_names),
            n_filters=int(self.cfg.get("n_filters", 64)),
            kernel_size=int(self.cfg.get("kernel_size", 7)),
            pool_size=int(self.cfg.get("pool_size", 2)),
            dropout=float(self.cfg.get("dropout", 0.2)),
            head_hidden_dims=tuple(self.cfg.get("head_hidden_dims", (64, 32))),
            head_dropout=self.cfg.get("head_dropout", 0.1),
            encoder_mode=str(self.cfg.get("encoder_mode", "shared")),
            fusion_mode=str(self.cfg.get("fusion_mode", "concat")),
            fusion_pool=str(self.cfg.get("fusion_pool", "avg")),
            fusion_dropout=float(self.cfg.get("fusion_dropout", 0.0)),
            scale_fusion_hidden=int(self.cfg.get("scale_fusion_hidden", 128)),
            scale_fusion_out_dim=int(self.cfg.get("scale_fusion_out_dim", 128)),
            scale_fusion_kernel_size=int(self.cfg.get("scale_fusion_kernel_size", 3)),
            scale_fusion_dropout=float(self.cfg.get("scale_fusion_dropout", 0.0)),
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.cfg.get("lr", 1e-3), weight_decay=self.cfg.get("weight_decay", 1e-4))
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
            adv_betti_seq, _ = build_betti_tensor(
                adv_split, k_values, homology_dims=homology_dims, grid_size=grid_size,
                coverage=coverage, range_pad=range_pad, include_euler=include_euler, betti_features=betti_features,
            )
            adv_extra, _, _ = pi_multik.build_extra(
                adv_split, None, n_norm=n_norm, entropy_norms=entropy_norms, include_entropy=include_entropy,
            )
            adv_ds = TensorDataset(
                torch.from_numpy(adv_betti_seq), torch.from_numpy(adv_extra), torch.from_numpy(adv_targets_std),
            )
            adv_loader = DataLoader(adv_ds, batch_size=self.cfg["batch_size"], shuffle=False)
            adversarial_loss, _ = evaluate(model, adv_loader, loss_fn, device)
            adversarial_loss_per_target = dict(
                zip(label_names, evaluate_per_target(model, adv_loader, device).tolist())
            )
            print(f"[{self.tag} seed={seed}] adversarial loss {adversarial_loss:.4f}")

        cfg_meta = {
            **self.cfg,
            "channels": [
                f"k{k}_{name}"
                for k in k_values
                for name in [f"beta{d}" for d in homology_dims] + (["euler"] if include_euler else [])
            ],
            # Per-k calibrated grid_range (and the rest of BettiCurve.params),
            # so a seed's exact calibration is recoverable from results.json
            # alone -- mirrors pi_multik.py's imager_params.
            "betti_params": {k: betti.params for k, betti in zip(k_values, betti_features)},
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
