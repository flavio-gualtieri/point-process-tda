import torch
import torch.nn as nn
from .base import Encoder

class PointNetEncoder(Encoder):
    def __init__(self, input_dim: int, embedding_dim: int, hidden_dims: tuple[int, ...] = (64, 128, 256)):
        super().__init__(embedding_dim=embedding_dim)
        
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        self.point_mlp = nn.Sequential(*layers)

        self.head = nn.Linear(hidden_dims[-1], embedding_dim)

    @property
    def input_modality(self) -> str:
        return "point_cloud"
    
    def forward(self, x: torch.Tensor | dict)  -> torch.Tensor:
        if isinstance(x, dict):
            points, mask = x["points"], x["mask"]
        else:
            points, mask = x, None

        features = self.point_mlp(points)

        if mask is not None:
            # Padded points are excluded from the max-pool: real features are
            # >=0 (last point_mlp layer is a ReLU), so -inf never wins.
            features = features.masked_fill(~mask.unsqueeze(-1), float("-inf"))

        pooled, _ = features.max(dim=1)
        return self.head(pooled)
