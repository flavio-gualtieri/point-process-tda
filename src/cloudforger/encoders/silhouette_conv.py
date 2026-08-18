# src/cloudforger/encoders/silhouette_conv.py
"""1-D CoordConv mirror of coordconv_pi.py's CoordConvPIEncoder, for
persistence silhouettes (G,)-per-dim curves -- the native encoder path for
vectorization="silhouette" (see experiments/pi_multik/vectorized_multik.py).
Same coordinate-augmentation idea (append a position channel so the conv
stack can tell where along the filtration-value grid it is, not just what
shape of tent it's looking at), same three-block (32, 64, 128) channel
progression as CoordConvPIEncoder's own conv_channels default -- just 1-D
instead of 2-D, and one coordinate channel instead of two."""

from __future__ import annotations

import torch
import torch.nn as nn

from .base import Encoder


class SilhouetteConv1DEncoder(Encoder):
    def __init__(
        self,
        in_channels: int,
        embedding_dim: int = 128,
        conv_channels: tuple[int, ...] = (32, 64, 128),
        kernel_size: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__(embedding_dim=embedding_dim)
        layers: list[nn.Module] = []
        prev = in_channels + 1  # +1 for the coordinate ramp channel appended in forward()
        padding = kernel_size // 2
        for i, ch in enumerate(conv_channels):
            layers.append(nn.Conv1d(prev, ch, kernel_size=kernel_size, padding=padding))
            layers.append(nn.ReLU())
            if i < len(conv_channels) - 1:  # pool + spatial dropout between blocks, not after the last
                layers.append(nn.MaxPool1d(2))
                layers.append(nn.Dropout1d(dropout))
            prev = ch
        layers.append(nn.AdaptiveAvgPool1d(1))
        self.conv = nn.Sequential(*layers)
        self.fc = nn.Linear(prev, embedding_dim)

    @property
    def input_modality(self) -> str:
        return "silhouette_coordconv1d"

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, in_channels, G)
        ramp = self._coord_channel(x.shape[0], x.shape[-1], x.device, x.dtype)
        x = torch.cat([x, ramp], dim=1)
        h = self.conv(x).flatten(1)
        return self.fc(h)

    @staticmethod
    def _coord_channel(batch_size: int, length: int, device, dtype) -> torch.Tensor:
        ramp = torch.linspace(-1, 1, length, device=device, dtype=dtype)
        return ramp.reshape(1, 1, length).expand(batch_size, 1, -1)
