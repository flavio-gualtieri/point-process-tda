# src/cloudforger/training/__init__.py
"""Training mechanics shared across experiments: Dataset wrappers
(data.py), the train/eval loop (train.py), and train/val/test splitting
(splits.py) -- kept separate from encoders/ (architecture) and models/
(composition) since these are orthogonal to what's being trained.

Re-exports the same names cloudforger.nn's __init__.py used to (this
package absorbed nn/data.py, nn/splits.py, nn/train.py verbatim)."""

from .data import (
    PersistenceImageDataset,
    BettiCurveDataset,
    PointCloudDataset,
    CorrelationFeatureDataset,
)
from .splits import train_val_test_split
from .train import train_one_epoch, evaluate, evaluate_per_target

__all__ = [
    "PersistenceImageDataset",
    "BettiCurveDataset",
    "PointCloudDataset",
    "CorrelationFeatureDataset",
    "train_val_test_split",
    "train_one_epoch",
    "evaluate",
    "evaluate_per_target",
]
