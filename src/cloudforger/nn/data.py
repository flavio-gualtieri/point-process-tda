import torch
import numpy as np
from torch.utils.data import Dataset
from ..core.cloud import PointCloud
from ..core.features import CorrelationFeatures

class PersistenceImageDataset(Dataset):
    def __init__(self, images: list[dict], labels: np.array):
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
    def __init__(self, clouds: list[PointCloud], labels: np.array, n_points: int | None = None, dtype: torch.dtype = torch.long):
        self.clouds = clouds
        self.labels = torch.as_tensor(labels, dtype=dtype)
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


class CorrelationFeatureDataset(Dataset):
    def __init__(
        self,
        features: list[CorrelationFeatures],
        labels: np.ndarray,
        statistic_names: list[str] | None = None,
    ):
        if len(features) != len(labels):
            raise ValueError("features and labels must have same length")
        self.features = features
        self.labels = torch.as_tensor(labels, dtype=torch.long)
        self.statistic_names = statistic_names

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        vec = self.features[idx].vector(self.statistic_names)
        x = torch.as_tensor(vec, dtype=torch.float32)
        return x, self.labels[idx]

    @property
    def input_dim(self) -> int:
        """Length of the concatenated feature vector — pass to MLPEncoder."""
        return len(self.features[0].vector(self.statistic_names))
    

