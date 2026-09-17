"""Train/val/test by theta index, the same for every family, method and seed.

Theta i is drawn from its own PARAMS stream, so indices are iid prior draws and contiguous blocks
are a random split. Both replicates of a theta share its split. Thetas past the test block (a
larger sweep) go to train, so the test set never changes.
"""

from __future__ import annotations

import numpy as np

TRAIN_END, VAL_END, TEST_END = 7000, 8000, 10000


def split_of(theta: np.ndarray | int) -> np.ndarray:
    theta = np.asarray(theta)
    return np.select([theta < TRAIN_END, theta < VAL_END, theta < TEST_END], ["train", "val", "test"], "train")
