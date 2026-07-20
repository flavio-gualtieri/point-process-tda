# src/cloudforger/nn/encoders/coordconv_pi.py

from __future__ import annotations

import torch
import torch.nn as nn

from .base import Encoder


class CoordConvPIEncoder(Encoder):
    def __init__(
        self,
        in_channels: int = 2,
        embedding_dim: int = 64,
        conv_channels: tuple[int, ...] = (32, 64, 128),
        dropout: float = 0.2,
    ):
        super().__init__(embedding_dim=embedding_dim)
        layers: list[nn.Module] = []
        prev = in_channels + 2  # +2 for the coordinate channels appended in forward()
        for i, ch in enumerate(conv_channels):
            layers.append(nn.Conv2d(prev, ch, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            if i < len(conv_channels) - 1:
                layers.append(nn.MaxPool2d(2))
                layers.append(nn.Dropout2d(dropout))
            prev = ch
        layers.append(nn.AdaptiveAvgPool2d(1))
        self.conv = nn.Sequential(*layers)
        self.fc = nn.Linear(prev, embedding_dim)

    @property
    def input_modality(self) -> str:
        return "persistence_image_coordconv"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        coords = self._coord_channels(x.shape[0], x.shape[-1], x.device, x.dtype)
        x = torch.cat([x, coords], dim=1)
        h = self.conv(x).flatten(1)
        return self.fc(h)

    @staticmethod
    def _coord_channels(batch_size: int, resolution: int, device, dtype) -> torch.Tensor:
        coords = torch.linspace(-1, 1, resolution, device=device, dtype=dtype)
        y, x = torch.meshgrid(coords, coords, indexing="ij")
        grid = torch.stack([x, y], dim=0)
        return grid.unsqueeze(0).expand(batch_size, -1, -1, -1)
