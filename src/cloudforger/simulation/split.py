"""Train/val/test by theta index, the same for every family, method and seed.

Theta i is drawn from its own PARAMS stream, so indices are iid prior draws and contiguous blocks
are a random split. Both replicates of a theta share its split. Thetas past the test block (a
larger sweep) go to train, so the test set never changes.

The test block is 12000 thetas because the evidence near CSR comes from slices of the test set
(regime strata, per-family bootstraps over test thetas), and a narrow block leaves the near-CSR
slices with a handful of thetas each -- exactly where the argument is. Widening is the only lever
that adds evidence there WITHOUT selecting on a label: delta-tilde and the regime are applied
afterwards, so a split that referred to them would move every time they were refit.
"""

from __future__ import annotations

import numpy as np

TRAIN_END, VAL_END, TEST_END = 7000, 8000, 20000


def split_of(theta: np.ndarray | int) -> np.ndarray:
    theta = np.asarray(theta)
    return np.select([theta < TRAIN_END, theta < VAL_END, theta < TEST_END], ["train", "val", "test"], "train")
