import torch
import torch.nn as nn
from .base import Encoder

class PIEncoder(Encoder):
    def __init__(self, in_channels: int = 2, embedding_dim: int = 128):
        super().__init__(embedding_dim=embedding_dim)
        self.in_channels = in_channels 

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(128, embedding_dim)

    @property
    def input_modality(self) -> str:
        return "persistence_image"
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        h = h.flatten(1)
        return self.fc(h)