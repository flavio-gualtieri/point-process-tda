# src/cloudforger/encoders/sequence_bank.py
"""Conv1D counterpart of encoder_bank.py's EncoderBank/CoordConvPIEncoder
pair, for per-k 1D curves (Betti curves, the derived Euler-characteristic
curve, ...) instead of per-k 2D persistence images -- see
cloudforger.experiments.pi_multik.betti_multik for the multi-k experiment
this backs.

VihrsConv1DEncoder(in_channels, 7)-Pool(2) x2 + Conv1d(*, *, 7)-Flatten-
Linear is Vihrs (2022)'s own L(r)-r architecture
(cloudforger.baselines.vihrs.VihrsCNN.conv: n_filters=64, kernel_size=7),
generalized from a hardcoded single input channel to `in_channels` so
several already-aligned curves can be read as one multi-channel sequence.
One deliberate deviation from the paper's own pool_size=5: that value was
tuned for Vihrs's 513-sample L(r)-r grid (N_R in baselines/vihrs.py); at
betti_multik's 128-sample grid (grid_size, see build_calibrated_betti),
two rounds of /5 pooling collapse the sequence to length 3 before the
third (kernel_size=7) conv layer, which cannot run on an input shorter
than its own kernel. pool_size therefore defaults to 2 here (matching
sequence_cnn.py's SequenceCNNEncoder's own general-purpose default, not
the Vihrs-specific override betti_cnn.py used to pass) -- still a 3-layer
Conv1d-then-pool stack ahead of the shared head, i.e. the same *shape* of
architecture the paper uses, just re-tuned for a shorter curve. Both
pool_size and kernel_size stay overridable via method.params."""

from __future__ import annotations

import torch
import torch.nn as nn

from .base import Encoder


class VihrsConv1DEncoder(Encoder):
    """Conv1d(in_channels, n_filters, kernel_size)-ReLU-MaxPool1d(pool_size)
    -Dropout1d, twice, then Conv1d(n_filters, n_filters, kernel_size)-ReLU-
    Flatten-Linear(embedding_dim) -- see module docstring for how this
    relates to Vihrs (2022)'s own L(r)-r conv stack. Unlike
    cloudforger.encoders.sequence_cnn.SequenceCNNEncoder (which always
    unsqueezes its input to a single channel), this encoder expects an
    already-multi-channel (B, in_channels, input_dim) tensor -- e.g. [beta_0,
    beta_1, euler_characteristic] stacked along dim 1, all sharing one
    filtration-value grid (see build_calibrated_betti)."""

    def __init__(
        self,
        in_channels: int,
        input_dim: int,
        embedding_dim: int = 128,
        n_filters: int = 64,
        kernel_size: int = 7,
        pool_size: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__(embedding_dim=embedding_dim)
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, n_filters, kernel_size=kernel_size), nn.ReLU(),
            nn.MaxPool1d(pool_size), nn.Dropout1d(dropout),
            nn.Conv1d(n_filters, n_filters, kernel_size=kernel_size), nn.ReLU(),
            nn.MaxPool1d(pool_size), nn.Dropout1d(dropout),
            nn.Conv1d(n_filters, n_filters, kernel_size=kernel_size), nn.ReLU(),
        )
        with torch.no_grad():
            flat_dim = self.conv(torch.zeros(1, in_channels, input_dim)).flatten(1).shape[1]
        self.head = nn.Linear(flat_dim, embedding_dim)
        self.flat_dropout = nn.Dropout(dropout)

    @property
    def input_modality(self) -> str:
        return "multichannel_curve"

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, in_channels, input_dim)
        h = self.conv(x).flatten(1)
        h = self.flat_dropout(h)
        return self.head(h)


class SequenceEncoderBank(Encoder):
    """Turns (B, K, C, L) per-k multi-channel curves into a (B, K,
    embedding_dim) embedding sequence, via one of two per-k
    VihrsConv1DEncoder strategies -- the direct Conv1D analogue of
    encoder_bank.py's EncoderBank (see that class's docstring; every design
    choice there applies here unchanged, just swapping the per-k Conv2D
    image encoder for a per-k Conv1D curve encoder):

    mode="shared": one VihrsConv1DEncoder instance, applied to every k by
    folding k into the batch dimension -- weight-tied across scales.
    mode="independent": n_k separate VihrsConv1DEncoder instances (a
    nn.ModuleList), one call per k -- no weight sharing across scales.

    The fusion step that combines the resulting (B, K, embedding_dim)
    sequence into one vector is a separate, orthogonal concern --
    cloudforger.encoders.scaleconv_pi.ConvFusion is reused as-is for it (it
    only ever sees an already-embedded per-k sequence, so the same class
    fuses both EncoderBank's image embeddings and this bank's curve
    embeddings)."""

    def __init__(
        self,
        mode: str,
        n_k: int,
        in_channels: int,
        input_dim: int,
        embedding_dim: int,
        n_filters: int = 64,
        kernel_size: int = 7,
        pool_size: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__(embedding_dim=embedding_dim)
        if mode not in ("shared", "independent"):
            raise ValueError(f"SequenceEncoderBank: mode must be 'shared' or 'independent', got {mode!r}.")
        self.mode = mode
        self.n_k = n_k

        def _make_encoder() -> VihrsConv1DEncoder:
            return VihrsConv1DEncoder(
                in_channels=in_channels, input_dim=input_dim, embedding_dim=embedding_dim,
                n_filters=n_filters, kernel_size=kernel_size, pool_size=pool_size, dropout=dropout,
            )

        if mode == "shared":
            self.encoder = _make_encoder()
        else:
            self.encoders = nn.ModuleList([_make_encoder() for _ in range(n_k)])

    @property
    def input_modality(self) -> str:
        return "multiscale_curve_bank"

    def forward(self, seqs: torch.Tensor) -> torch.Tensor:  # (B, K, C, L)
        if self.mode == "shared":
            b = seqs.shape[0]
            flat = seqs.reshape(b * self.n_k, *seqs.shape[2:])
            emb = self.encoder(flat)                             # (B*K, C)
            return emb.reshape(b, self.n_k, self.embedding_dim)  # (B, K, C)
        embs = [enc(seqs[:, i]) for i, enc in enumerate(self.encoders)]
        return torch.stack(embs, dim=1)  # (B, K, C)
