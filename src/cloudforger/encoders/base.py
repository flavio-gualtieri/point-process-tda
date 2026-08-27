from abc import abstractmethod
import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...

    @property
    @abstractmethod
    def input_modality(self) -> str:
        ...