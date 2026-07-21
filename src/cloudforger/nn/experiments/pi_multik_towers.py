# src/cloudforger/nn/experiments/pi_multik_towers.py

from __future__ import annotations

import torch
import torch.nn as nn

from cloudforger.nn.encoders.coordconv_pi import CoordConvPIEncoder
from cloudforger.nn.encoders.towerconv_pi import TowerConv
from cloudforger.nn.experiments.base import register
from cloudforger.nn.heads.paramest import ParameterEstimator
from cloudforger.nn.experiments.pi_multik import PIMultiKExperiment


class PIMultiKTowers(nn.Module):

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
        scale_fusion_hidden: int = 128,
        scale_fusion_out_dim: int = 128,
        scale_fusion_kernel_size: int = 3,
        fusion_dropout: float = 0.1,
        scale_fusion_dropout: float = 0.0,
    ):
        super().__init__()
        self.n_k = n_k
        self.encoders = nn.ModuleList([
            CoordConvPIEncoder(in_channels=in_channels, embedding_dim=embedding_dim,
                               conv_channels=conv_channels, dropout=dropout)
            for _ in range(n_k)
        ])
        self.tower_fusion = TowerConv(
            n_k=n_k, embedding_dim=embedding_dim, hidden=scale_fusion_hidden,
            out_dim=scale_fusion_out_dim,
            kernel_size=scale_fusion_kernel_size,
            dropout=scale_fusion_dropout,
        )
        head_in = self.tower_fusion.out_dim + n_extra

        self.head = ParameterEstimator(
            embedding_dim=head_in, n_params=n_targets,
            hidden_dims=head_hidden_dims, dropout=head_dropout,
        )
        self.fusion_dropout = nn.Dropout(fusion_dropout)

    def forward(self, pi_imgs, extra):
        embs = [enc(pi_imgs[:, i]) for i, enc in enumerate(self.encoders)]
        seq = torch.stack(embs, dim=1)       # (B, K, C)
        fused = self.tower_fusion(seq)       # (B, out_dim*K)
        fused = self.fusion_dropout(fused)

        return self.head(torch.cat([fused, extra], dim=1))
    

@register("pi_multik_towers")
class PIMultiKTowersExperiment(PIMultiKExperiment):
    @property
    def subdir(self) -> str:
        return "pi_multik_towers"

    def _build_model(self, **kwargs) -> PIMultiKTowers:
        kwargs["fusion_dropout"] = self.cfg.get("fusion_dropout", 0.1)
        kwargs["dropout_fusion_conv"] = self.cfg.get("scale_fusion_dropout", 0.0)  # if you also did step 1
        return PIMultiKTowers(**kwargs)

