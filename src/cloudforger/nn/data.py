import torch
import numpy as np
from torch.utils.data import Dataset
from ..core.cloud import PointCloud


class PersistenceImageDataset(Dataset):
    """Wraps persistence images (one per homology dimension) for PyTorch training.

    Each item returns a dict {"h0": (1,R,R), "h1": (1,R,R), ...} and a label.
    Keys match the encoder names expected by MultiModalModel.
    """

    def __init__(self, images: list[dict], labels):
        self.images = images
        self.labels = torch.as_tensor(labels, dtype=torch.long)
        if len(self.images) != len(self.labels):
            raise ValueError("images and labels must have the same length")
        self.dims = sorted(images[0].keys())

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        img_dict = self.images[idx]
        tensors = {
            f"h{d}": torch.from_numpy(np.asarray(img_dict[d], dtype=np.float32)).unsqueeze(0)
            for d in self.dims
        }
        return tensors, self.labels[idx]

class PointCloudDataset(Dataset):
    """Wraps a list of PointCloud objects for PyTorch training."""

    def __init__(self, clouds: list[PointCloud], labels, n_points: int | None = None):
        self.clouds = clouds
        self.labels = torch.as_tensor(labels, dtype=torch.long)
        self.n_points = n_points
        if len(self.clouds) != len(self.labels):
            raise ValueError("clouds and labels must have same length")

    def __len__(self) -> int:
        return len(self.clouds)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        cloud = self.clouds[idx]
        points = cloud.points  # (n, dim)
        if self.n_points is not None:
            n = len(points)
            chosen = np.random.choice(n, self.n_points, replace=n < self.n_points)
            points = points[chosen]
        return torch.from_numpy(points).float(), self.labels[idx]