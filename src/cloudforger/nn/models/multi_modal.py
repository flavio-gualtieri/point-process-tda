# src/cloudforger/nn/models/multi_modal.py
import torch
import torch.nn as nn
from ..encoders.base import Encoder

class MultiModalModel(nn.Module):
    """Multiple encoders → concatenated embeddings → head.

    Forward takes a dict mapping modality names to input tensors.
    """

    def __init__(
        self,
        encoders: dict[str, Encoder],
        head: nn.Module,
    ):
        super().__init__()
        self.encoders = nn.ModuleDict(encoders)
        self.head = head

    @property
    def total_embedding_dim(self) -> int:
        return sum(enc.embedding_dim for enc in self.encoders.values())

    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        # Compute each encoder's embedding
        embeddings = []
        for name, encoder in self.encoders.items():
            if name not in inputs:
                raise KeyError(f"missing input for modality '{name}'")
            embeddings.append(encoder(inputs[name]))
        # Concatenate along feature dimension
        fused = torch.cat(embeddings, dim=-1)
        return self.head(fused)