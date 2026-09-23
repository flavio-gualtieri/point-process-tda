"""Train/val/test by theta index, the same for every family, method and seed.

Theta i is drawn from its own PARAMS stream, so indices are iid prior draws and contiguous blocks
are a random split. Both replicates of a theta share its split. Thetas past the test block (a
larger sweep) go to train, so the test set never changes.

The test block is 12000 thetas because the paper's unit of evidence is a regime CELL, not the
marginal: the bootstrap in scripts/regimes.py resamples test thetas, and a cell is a
(family, delta bin, nbar bin) slice of them. At the old 2000 the near-CSR bins held 12-19 thetas
per family, which is where the argument is and where the intervals were widest. Widening is the
only lever that adds evidence there WITHOUT selecting on delta-tilde: delta-tilde is a label
applied afterwards by scripts/relabel.py, so a split that referred to it would move every time the
null tables are refit or a different reduction is read.
"""

from __future__ import annotations

import numpy as np

TRAIN_END, VAL_END, TEST_END = 7000, 8000, 20000


def split_of(theta: np.ndarray | int) -> np.ndarray:
    theta = np.asarray(theta)
    return np.select([theta < TRAIN_END, theta < VAL_END, theta < TEST_END], ["train", "val", "test"], "train")
