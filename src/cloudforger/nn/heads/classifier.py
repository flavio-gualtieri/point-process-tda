# src/pointforge/nn/heads/classifier.py
import torch
import torch.nn as nn

class ClassificationHead(nn.Module):
    """MLP from embedding to class logits."""

    def __init__(
        self,
        embedding_dim: int,
        n_classes: int,
        hidden_dims: tuple[int, ...] = (128,),
        dropout: float = 0.1,
    ):
        super().__init__()
        layers = []
        prev = embedding_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.mlp = nn.Sequential(*layers)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.mlp(embedding)