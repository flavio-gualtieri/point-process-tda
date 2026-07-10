import torch
import torch.nn as nn

from .base import Encoder


class SequenceCNNEncoder(Encoder):
    """1D CNN encoder for curve-shaped inputs (e.g. Betti curves): three
    Conv1D layers with two max-pooling stages in between, following the
    Conv1D(64,7)-Pool-Conv1D(64,7)-Pool-Conv1D(64,7)-Flatten branch that
    Vihrs (2022) uses for the Ripley L(r)-r curve (see
    scripts/runners/params/run_new_feature.py, the paper-replication baseline
    this was benchmarked against).

    pool_size defaults to 2 rather than the paper's 5: that value was sized
    for their ~513-point r-grid, whereas this repo's Betti curves have length
    128 (data/params/2d/thomas/betti.pkl), and pool=5 twice would shrink the
    sequence below the final kernel size before the third convolution.
    """

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int = 128,
        n_filters: int = 64,
        kernel_size: int = 7,
        pool_size: int = 2,
    ):
        super().__init__(embedding_dim=embedding_dim)
        self.conv = nn.Sequential(
            nn.Conv1d(1, n_filters, kernel_size=kernel_size), nn.ReLU(), nn.MaxPool1d(pool_size),
            nn.Conv1d(n_filters, n_filters, kernel_size=kernel_size), nn.ReLU(), nn.MaxPool1d(pool_size),
            nn.Conv1d(n_filters, n_filters, kernel_size=kernel_size), nn.ReLU(),
        )
        with torch.no_grad():
            flat_dim = self.conv(torch.zeros(1, 1, input_dim)).flatten(1).shape[1]
        self.head = nn.Linear(flat_dim, embedding_dim)

    @property
    def input_modality(self) -> str:
        return "sequence"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x.unsqueeze(1))
        h = h.flatten(1)
        return self.head(h)
