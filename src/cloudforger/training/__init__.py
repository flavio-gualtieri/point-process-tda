"""Stage 3: diagrams -> trained model.

    data.py   diagrams + manifest -> images, covariates, targets, splits
    model.py  one CNN per homology dimension + log n -> MLP head
    train.py  the training loop
"""

from . import data, model, train

__all__ = ["data", "model", "train"]
