# src/cloudforger/baselines/__init__.py
"""Non-TDA comparison baselines: classical point-process estimators
(mincontrast, palm) and the vihrs neural baseline. Each module exposes a
prepare-once/fit-or-run-per-seed pair (mincontrast.estimate / palm.estimate
for classical per-cloud fits, vihrs.prepare_data + vihrs.run_one_seed for
the trained-per-seed CNN) -- deliberately not force-unified into a single
class shape here, since the two families have genuinely different execution
models (see cloudforger's train CLI for how both get dispatched from one
--method flag)."""

from . import mincontrast
from . import palm
from . import vihrs

__all__ = ["mincontrast", "palm", "vihrs"]
