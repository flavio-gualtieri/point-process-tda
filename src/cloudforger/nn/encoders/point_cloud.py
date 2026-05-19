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
    
    def forward(self, x: torch.Tensor)  -> torch.Tensor:
        features = self.point_mlp(x)

        pooled, _ = features.max(dim=1)
        return self.head(pooled)
