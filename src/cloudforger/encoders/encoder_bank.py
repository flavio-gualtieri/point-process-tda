# src/cloudforger/nn/encoders/encoder_bank.py

from __future__ import annotations

import torch
import torch.nn as nn

from .base import Encoder
from .coordconv_pi import CoordConvPIEncoder


class EncoderBank(Encoder):
    """Turns (B, K, C, H, W) per-k persistence images into a (B, K,
    embedding_dim) embedding sequence, via one of two per-k CoordConv
    strategies -- the one axis of the pi_multik design space that used to be
    hardcoded per-file (PIMultiK.forward shared-weight folding vs
    PIMultiKTowers.forward's independent nn.ModuleList):

    mode="shared": one CoordConvPIEncoder instance, applied to every k by
    folding k into the batch dimension (one conv-stack call per forward, not
    K) -- weight-tied across scales.

    mode="independent": n_k separate CoordConvPIEncoder instances (a
    nn.ModuleList), one call per k -- no weight sharing across scales.

    Either way, dropout/pool_type/conv_channels apply uniformly per k; the
    fusion step that combines the resulting (B, K, embedding_dim) sequence
    into a single vector is a separate, orthogonal concern (see
    cloudforger.encoders.scaleconv_pi.ConvFusion / ConcatFusion in
    pi_multik.py)."""

    def __init__(
        self,
        mode: str,
        n_k: int,
        in_channels: int,
        embedding_dim: int,
        conv_channels: tuple[int, ...] = (32, 64, 128),
        dropout: float = 0.2,
        pool_type: str = "max",
    ):
        super().__init__(embedding_dim=embedding_dim)
        if mode not in ("shared", "independent"):
            raise ValueError(f"EncoderBank: mode must be 'shared' or 'independent', got {mode!r}.")
        self.mode = mode
        self.n_k = n_k

        def _make_encoder() -> CoordConvPIEncoder:
            return CoordConvPIEncoder(
                in_channels=in_channels, embedding_dim=embedding_dim,
                conv_channels=conv_channels, dropout=dropout, pool_type=pool_type,
            )

        if mode == "shared":
            self.encoder = _make_encoder()
        else:
            self.encoders = nn.ModuleList([_make_encoder() for _ in range(n_k)])

    @property
    def input_modality(self) -> str:
        return "multiscale_persistence_image_bank"

    def forward(self, pi_imgs: torch.Tensor) -> torch.Tensor:  # (B, K, C, H, W)
        if self.mode == "shared":
            b = pi_imgs.shape[0]
            flat = pi_imgs.reshape(b * self.n_k, *pi_imgs.shape[2:])
            emb = self.encoder(flat)                             # (B*K, C)
            return emb.reshape(b, self.n_k, self.embedding_dim)  # (B, K, C)
        embs = [enc(pi_imgs[:, i]) for i, enc in enumerate(self.encoders)]
        return torch.stack(embs, dim=1)  # (B, K, C)
