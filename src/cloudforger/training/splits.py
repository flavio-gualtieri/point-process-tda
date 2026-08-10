# Deprecated: moved to cloudforger.core.splits (not nn-specific -- classical
# baselines need the same per-seed partition). Shim for not-yet-migrated callers.

from ..core.splits import train_val_test_split, train_val_test_indices

__all__ = ["train_val_test_split", "train_val_test_indices"]
