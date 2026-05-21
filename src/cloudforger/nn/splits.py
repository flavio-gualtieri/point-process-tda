# src/cloudforger/nn/splits.py
import torch
from torch.utils.data import Dataset, Subset, random_split

def train_val_test_split(
    dataset: Dataset,
    fractions: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 0,
) -> tuple[Subset, Subset, Subset]:
    """Split dataset into train, validation, and test subsets.

    Args:
        dataset:   any PyTorch Dataset.
        fractions: (train, val, test) proportions. Must sum to 1.
        seed:      for reproducibility.

    Returns:
        (train_ds, val_ds, test_ds) as Subset objects.
    """
    if abs(sum(fractions) - 1.0) > 1e-6:
        raise ValueError(f"fractions must sum to 1, got {sum(fractions)}")

    n = len(dataset)
    n_train = int(fractions[0] * n)
    n_val = int(fractions[1] * n)
    n_test = n - n_train - n_val   # remainder avoids rounding gaps

    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [n_train, n_val, n_test], generator=generator)