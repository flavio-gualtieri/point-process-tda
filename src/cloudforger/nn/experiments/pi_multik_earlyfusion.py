# src/cloudforger/nn/experiments/pi_multik_earlyfusion.py
"""EARLY-fusion sibling of pi_multik/pi_multik_scaleconv/pi_multik_towers:
all k's (H0, H1) channels are stacked into one (n_k * in_channels)-channel
image BEFORE the first conv layer, so a single CoordConvPIEncoder sees every
scale jointly from layer 1 -- no per-k embedding, no late-fusion pooling
step. This is the pre-a83b74e pi_multik design (see that commit's parent
revision, and the module docstring in pi_multik.py), reinstated here as an
explicit sibling method instead of the default, so it stays directly
comparable against the late-fusion family on the same channel grid.

Reuses everything in pi_multik.py (load_multik_split, build_pi_tensor,
build_extra, PIMultiKExperiment.run) -- only _build_model differs. The (N,
K, D, H, W) tensor build_pi_tensor produces is reshaped to (N, K*D, H, W) in
PIMultiKEarlyFusion.forward, in [k0_d0, k0_d1, k1_d0, k1_d1, ...] order
(matching the original pi_multik's fixed channel order)."""

from __future__ import annotations

import torch
import torch.nn as nn

from cloudforger.nn.encoders.coordconv_pi import CoordConvPIEncoder
from cloudforger.nn.experiments.base import register
from cloudforger.nn.experiments.pi_multik import PIMultiKExperiment
from cloudforger.nn.heads.paramest import ParameterEstimator


class PIMultiKEarlyFusion(nn.Module):
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
        **_scale_fusion_kwargs,  # accepted-and-ignored: scale_fusion_* only apply to the late-fusion siblings
    ):
        super().__init__()
        self.n_k = n_k
        self.encoder = CoordConvPIEncoder(
            in_channels=in_channels * n_k, embedding_dim=embedding_dim,
            conv_channels=conv_channels, dropout=dropout,
        )
        self.head = ParameterEstimator(
            embedding_dim=embedding_dim + n_extra, n_params=n_targets,
            hidden_dims=head_hidden_dims, dropout=head_dropout,
        )

    def forward(self, pi_imgs: torch.Tensor, extra: torch.Tensor) -> torch.Tensor:
        b = pi_imgs.shape[0]
        wide = pi_imgs.reshape(b, -1, *pi_imgs.shape[3:])  # (B, K, D, H, W) -> (B, K*D, H, W)
        emb = self.encoder(wide)
        return self.head(torch.cat([emb, extra], dim=1))


@register("pi_multik_earlyfusion")
class PIMultiKEarlyFusionExperiment(PIMultiKExperiment):
    @property
    def subdir(self) -> str:
        return "pi_multik_earlyfusion"

    def _build_model(self, **kwargs) -> PIMultiKEarlyFusion:
        return PIMultiKEarlyFusion(**kwargs)
