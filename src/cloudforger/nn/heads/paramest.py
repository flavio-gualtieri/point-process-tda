import torch
import torch.nn as nn

class ParameterEstimator(nn.Module):
    
    def __init__(
        self,
        embedding_dim: int,
        n_params: int,
        hidden_dims: tuple[int, ...],
        dropout: float = 0.1,
    ):
        super().__init__()
        layers = []
        prev = embedding_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, n_params))
        self.mlp = nn.Sequential(*layers)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.mlp(embedding)