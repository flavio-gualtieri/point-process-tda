# src/cloudforger/experiments/pi_multik/pi_multik_dimsplit.py
"""Per-homology-dimension encoder ablation for pi_multik.

The canonical PIMultiK stacks the H0 and H1 persistence images as the two
input channels of ONE CoordConvPIEncoder, so they are mixed by the first
3x3 convolution. But H0 and H1 images are calibrated to different
birth/persistence ranges (build_pi_tensor fits a separate imager per
homology dim), so a given pixel does not mean the same thing in the two
channels, and the shared CoordConv coordinate channels impose a single
absolute-position frame on both.

DimSplitPIMultiK gives each homology dimension its own EncoderBank
(in_channels=1) and concatenates the per-(dim, k) embeddings before the
head. Everything downstream of the encoder -- the flat concat over k, the
[fused, extra] merge, the MLP head, the whole PIMultiKExperiment.run()
training/eval/save path -- is byte-for-byte the shared pi_multik code, so
this is a clean one-axis ablation: shared-channel encoder vs per-dim
encoder, holding calibration, splits, seeds, optimiser, loss and eval
fixed.

Deliberately NOT a point in PIMultiK's encoder_mode/fusion_mode space: the
k-fusion knobs (fusion_mode / fusion_pool / scale_fusion_*) are dropped and
k-fusion is fixed to flat concat. encoder_mode (shared- vs
independent-weight CoordConvPIEncoder across k, via EncoderBank) is still
honoured.

Capacity: two full-width per-dim encoders roughly double the encoder
parameter count, and -- at the same embedding_dim -- double the vector fed
to the head (n_dims * n_k * embedding_dim vs n_k * embedding_dim). The
ablation configs (configs/runs/*/dimsplit_k5.yaml + slurm/dimsplit_*.sh)
bracket this with an embedding_dim=32 arm (head width matched to the
embedding_dim=64 shared baseline), an embedding_dim=64 arm (unconstrained),
a narrowed-conv arm (total conv params ~ matched), and a widened shared
arm (params added to the shared wiring) so a win can be attributed to the
wiring rather than the parameter budget.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from cloudforger.encoders.encoder_bank import EncoderBank
from cloudforger.experiments.base import register
from cloudforger.experiments.pi_multik.pi_multik import PIMultiKExperiment
from cloudforger.models.heads.classifier import ClassificationHead
from cloudforger.models.heads.paramest import ParameterEstimator


class DimSplitPIMultiK(nn.Module):
    """One EncoderBank per homology dimension (in_channels=1), embeddings
    concatenated over (dim, k) then merged with the scalar extras and passed
    through an MLP head -- see module docstring.

    forward(pi_imgs, extra) matches PIMultiK's signature exactly (and hence
    cloudforger.training.train's (inputs, covariates, labels) batch
    convention), so the shared train loop applies unchanged."""

    def __init__(
        self,
        *,
        n_dims: int,
        n_k: int,
        embedding_dim: int,
        n_extra: int,
        n_targets: int,
        conv_channels: tuple[int, ...] = (32, 64, 128),
        dropout: float = 0.2,
        pool_type: str = "avg",
        use_coords: bool = True,
        encoder_mode: str = "shared",
        head_hidden_dims: tuple[int, ...] = (64, 32),
        head_dropout: float = 0.1,
        task: str = "params",
    ):
        super().__init__()
        self.n_dims = n_dims
        self.n_k = n_k
        # One bank per homology dim; each bank is the same shared/independent
        # -across-k CoordConvPIEncoder machinery pi_multik uses, only over a
        # single-channel image instead of the stacked (H0, H1) pair.
        self.banks = nn.ModuleList(
            [
                EncoderBank(
                    mode=encoder_mode,
                    n_k=n_k,
                    in_channels=1,
                    embedding_dim=embedding_dim,
                    conv_channels=conv_channels,
                    dropout=dropout,
                    pool_type=pool_type,
                    use_coords=use_coords,
                )
                for _ in range(n_dims)
            ]
        )
        fused_dim = n_dims * n_k * embedding_dim
        head_in = fused_dim + n_extra
        if task == "classify":
            self.head = ClassificationHead(
                embedding_dim=head_in,
                n_classes=n_targets,
                hidden_dims=head_hidden_dims,
                dropout=head_dropout,
            )
        else:
            self.head = ParameterEstimator(
                embedding_dim=head_in,
                n_params=n_targets,
                hidden_dims=head_hidden_dims,
                dropout=head_dropout,
            )

    def forward(self, pi_imgs: torch.Tensor, extra: torch.Tensor) -> torch.Tensor:
        # pi_imgs: (B, K, D, H, W) from build_pi_tensor. Slice dim d keeping a
        # singleton channel axis so each bank sees (B, K, 1, H, W).
        per_dim = [self.banks[d](pi_imgs[:, :, d : d + 1]) for d in range(self.n_dims)]
        seq = torch.cat(per_dim, dim=-1)          # (B, K, D * embedding_dim)
        fused = seq.reshape(seq.shape[0], -1)     # flat concat over k, as PIMultiK
        return self.head(torch.cat([fused, extra], dim=1))


@register("pi_multik_dimsplit")
class DimSplitPIMultiKExperiment(PIMultiKExperiment):
    """pi_multik with a per-homology-dimension encoder. Thin _build_model
    override on PIMultiKExperiment -- data loading, per-seed train-split
    calibration, the training loop, per-target/per-class eval and
    save_results are all inherited verbatim, so results.json compares
    directly against a method: pi_multik run on the same config."""

    # K-fusion knobs PIMultiKExperiment.run() forwards to _build_model that
    # DimSplitPIMultiK has no use for (k-fusion is fixed to flat concat here).
    _DROPPED_FUSION_KWARGS = (
        "fusion_mode",
        "fusion_pool",
        "fusion_dropout",
        "scale_fusion_hidden",
        "scale_fusion_out_dim",
        "scale_fusion_kernel_size",
        "scale_fusion_dropout",
    )

    @property
    def subdir(self) -> str:
        return "pi_multik_dimsplit"

    def _build_model(self, **kwargs) -> DimSplitPIMultiK:
        kwargs["n_dims"] = kwargs.pop("in_channels")  # len(homology_dims)
        kwargs.setdefault("encoder_mode", "shared")
        for drop in self._DROPPED_FUSION_KWARGS:
            kwargs.pop(drop, None)
        return DimSplitPIMultiK(**kwargs)
