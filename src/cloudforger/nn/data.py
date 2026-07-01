# src/cloudforger/nn/data.py

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from ..core.cloud import PointCloud
from ..core.features import CorrelationFeatures
from ..core.betti import BettiCurveFeature


def _parse_dim_key(key: Any) -> int | None:
    if isinstance(key, int):
        return key

    if isinstance(key, np.integer):
        return int(key)

    if not isinstance(key, str):
        return None

    if key.isdigit():
        return int(key)

    if key.startswith("h") and key[1:].isdigit():
        return int(key[1:])

    if key.startswith("betti_") and key[len("betti_"):].isdigit():
        return int(key[len("betti_"):])

    return None


def _lookup_dim(mapping: Mapping, dim: int) -> Any:
    candidates = (dim, str(dim), f"h{dim}", f"betti_{dim}")

    for key in candidates:
        if key in mapping:
            return mapping[key]

    for key, value in mapping.items():
        if _parse_dim_key(key) == dim:
            return value

    raise KeyError(f"Could not find homology dimension {dim} in keys {list(mapping.keys())}")


def _infer_dims(mapping: Mapping) -> list[int]:
    dims = sorted(
        dim for dim in (_parse_dim_key(key) for key in mapping.keys())
        if dim is not None
    )

    if not dims:
        raise ValueError(f"Could not infer homology dimensions from keys {list(mapping.keys())}")

    return dims


def _normalize_dims(homology_dims: list[int] | tuple[int, ...] | None, source: Mapping) -> list[int]:
    if homology_dims is not None:
        return [int(d) for d in homology_dims]

    return _infer_dims(source)


def _as_label_tensor(labels: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
    return torch.as_tensor(labels, dtype=dtype)


def _as_image_tensor(image: Any) -> torch.Tensor:
    arr = np.asarray(image, dtype=np.float32)

    if arr.ndim == 2:
        arr = arr[None, :, :]
    elif arr.ndim == 3 and arr.shape[0] != 1:
        pass
    elif arr.ndim != 3:
        raise ValueError(f"Expected image with shape (H, W) or (C, H, W), got {arr.shape}")

    return torch.from_numpy(arr)


def _as_vector_tensor(values: Any) -> torch.Tensor:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    return torch.from_numpy(arr)


def _is_stacked_dim_mapping(source: Any, labels_len: int) -> bool:
    if not isinstance(source, Mapping):
        return False

    dims = [_parse_dim_key(key) for key in source.keys()]
    if not any(dim is not None for dim in dims):
        return False

    for key, value in source.items():
        if _parse_dim_key(key) is None:
            continue

        arr = np.asarray(value)
        if arr.ndim < 2 or len(arr) != labels_len:
            return False

    return True


def _extract_betti_matrix_source(source: Mapping) -> dict[int, Any]:
    matrices: dict[int, Any] = {}

    for key, value in source.items():
        if not isinstance(key, str):
            continue

        if not key.startswith("betti") or not key.endswith("_matrix"):
            continue

        middle = key[len("betti"):-len("_matrix")]
        if middle.isdigit():
            matrices[int(middle)] = value

    return matrices


class PersistenceImageDataset(Dataset):
    def __init__(
        self,
        images: Any,
        labels: np.ndarray,
        homology_dims: list[int] | tuple[int, ...] | None = None,
        return_dict: bool | None = None,
        dtype: torch.dtype = torch.float32,
    ):
        if isinstance(images, Mapping) and "image_tensors" in images:
            images = images["image_tensors"]
        elif isinstance(images, Mapping) and "images" in images:
            images = images["images"]

        self.images = images
        self.labels = _as_label_tensor(labels, dtype)
        self.stacked = _is_stacked_dim_mapping(images, len(self.labels))

        if self.stacked:
            self.dims = _normalize_dims(homology_dims, images)
            n_images = len(np.asarray(_lookup_dim(images, self.dims[0])))
        else:
            if len(images) == 0:
                raise ValueError("images cannot be empty")
            self.dims = _normalize_dims(homology_dims, images[0])
            n_images = len(images)

        if n_images != len(self.labels):
            raise ValueError("images and labels must have the same length")

        self.return_dict = len(self.dims) > 1 if return_dict is None else bool(return_dict)

    def __len__(self) -> int:
        return len(self.labels)

    def _item_mapping(self, idx: int) -> dict[int, Any]:
        if self.stacked:
            return {
                dim: np.asarray(_lookup_dim(self.images, dim))[idx]
                for dim in self.dims
            }

        item = self.images[idx]

        if not isinstance(item, Mapping):
            if len(self.dims) != 1:
                raise ValueError("Non-mapping image items require exactly one homology dimension")
            return {self.dims[0]: item}

        return {
            dim: _lookup_dim(item, dim)
            for dim in self.dims
        }

    def __getitem__(self, idx: int):
        item = self._item_mapping(idx)
        tensors = {
            f"h{dim}": _as_image_tensor(item[dim])
            for dim in self.dims
        }

        if self.return_dict:
            return tensors, self.labels[idx]

        if len(self.dims) == 1:
            return tensors[f"h{self.dims[0]}"], self.labels[idx]

        x = torch.cat([tensors[f"h{dim}"] for dim in self.dims], dim=0)
        return x, self.labels[idx]

    @property
    def input_shape(self) -> tuple[int, ...]:
        x, _ = self[0]
        if isinstance(x, dict):
            first = x[f"h{self.dims[0]}"]
            return tuple(first.shape)
        return tuple(x.shape)


class BettiCurveDataset(Dataset):
    def __init__(
        self,
        curves: Any,
        labels: np.ndarray,
        homology_dims: list[int] | tuple[int, ...] | None = None,
        dtype: torch.dtype = torch.float32,
    ):
        if isinstance(curves, Mapping) and "betti_curves" in curves:
            curves = curves["betti_curves"]
        elif isinstance(curves, Mapping) and "curves" in curves:
            curves = curves["curves"]
        elif isinstance(curves, Mapping):
            matrix_source = _extract_betti_matrix_source(curves)
            if matrix_source:
                curves = matrix_source

        self.curves = curves
        self.labels = _as_label_tensor(labels, dtype)
        self.stacked = _is_stacked_dim_mapping(curves, len(self.labels))

        if self.stacked:
            self.dims = _normalize_dims(homology_dims, curves)
            n_curves = len(np.asarray(_lookup_dim(curves, self.dims[0])))
        else:
            if len(curves) == 0:
                raise ValueError("curves cannot be empty")
            first = self._curve_mapping(curves[0])
            self.dims = _normalize_dims(homology_dims, first)
            n_curves = len(curves)

        if n_curves != len(self.labels):
            raise ValueError("curves and labels must have the same length")

    def __len__(self) -> int:
        return len(self.labels)

    @staticmethod
    def _curve_mapping(item: Any) -> Mapping:
        if isinstance(item, BettiCurveFeature):
            return item.curves

        if hasattr(item, "curves") and isinstance(item.curves, Mapping):
            return item.curves

        if isinstance(item, Mapping) and "curves" in item:
            return item["curves"]

        if isinstance(item, Mapping):
            return item

        raise TypeError(f"Cannot extract Betti curves from object of type {type(item)}")

    def _item_mapping(self, idx: int) -> Mapping:
        if self.stacked:
            return {
                dim: np.asarray(_lookup_dim(self.curves, dim))[idx]
                for dim in self.dims
            }

        return self._curve_mapping(self.curves[idx])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        item = self._item_mapping(idx)
        vectors = [_as_vector_tensor(_lookup_dim(item, dim)) for dim in self.dims]
        x = vectors[0] if len(vectors) == 1 else torch.cat(vectors, dim=0)
        return x, self.labels[idx]

    @property
    def input_dim(self) -> int:
        x, _ = self[0]
        return int(x.numel())


class PointCloudDataset(Dataset):
    def __init__(
        self,
        clouds: list[PointCloud],
        labels: np.ndarray,
        n_points: int | None = None,
        dtype: torch.dtype = torch.float32,
    ):
        self.clouds = clouds
        self.labels = _as_label_tensor(labels, dtype)
        self.n_points = n_points

        if len(self.clouds) != len(self.labels):
            raise ValueError("clouds and labels must have same length")

    def __len__(self) -> int:
        return len(self.clouds)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        cloud = self.clouds[idx]
        points = cloud.points

        if self.n_points is not None:
            n = len(points)
            chosen = np.random.choice(n, self.n_points, replace=n < self.n_points)
            points = points[chosen]

        return torch.from_numpy(np.asarray(points, dtype=np.float32)), self.labels[idx]


class CorrelationFeatureDataset(Dataset):
    def __init__(
        self,
        features: Any,
        labels: np.ndarray,
        statistic_names: list[str] | None = None,
        dtype: torch.dtype = torch.float32,
    ):
        if isinstance(features, Mapping) and "feature_matrix" in features and statistic_names is None:
            features = features["feature_matrix"]
        elif isinstance(features, Mapping) and "features" in features:
            features = features["features"]

        self.labels = _as_label_tensor(labels, dtype)
        self.statistic_names = statistic_names
        self.feature_matrix = None

        if isinstance(features, np.ndarray):
            if len(features) != len(self.labels):
                raise ValueError("features and labels must have same length")
            self.feature_matrix = np.asarray(features, dtype=np.float32)
            self.features = None
        else:
            if len(features) != len(self.labels):
                raise ValueError("features and labels must have same length")
            self.features = [self._coerce(f) for f in features]

    @staticmethod
    def _coerce(feature: CorrelationFeatures | Mapping) -> CorrelationFeatures:
        if isinstance(feature, CorrelationFeatures):
            return feature

        return CorrelationFeatures(
            features=feature["features"],
            generator_name=feature.get("process", ""),
            generator_params=feature.get("params", {}),
            seed=feature.get("seed"),
            statistic_params=feature.get("statistic_params", {}),
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.feature_matrix is not None:
            x = torch.as_tensor(self.feature_matrix[idx], dtype=torch.float32)
        else:
            vec = self.features[idx].vector(self.statistic_names)
            x = torch.as_tensor(vec, dtype=torch.float32)

        return x, self.labels[idx]

    @property
    def input_dim(self) -> int:
        if self.feature_matrix is not None:
            return int(self.feature_matrix.shape[1])

        return len(self.features[0].vector(self.statistic_names))