# src/cloudforger/baselines/__init__.py
"""Non-TDA comparison baselines: classical point-process estimators
(mincontrast, mincontrast_g, palm) and the vihrs neural baseline. Each
module exposes a prepare-once/fit-or-run-per-seed pair (mincontrast.estimate
/ mincontrast_g.estimate / palm.estimate for classical per-cloud fits,
vihrs.prepare_data + vihrs.run_one_seed for the trained-per-seed CNN) --
deliberately not force-unified into a single class shape here, since the
two families have genuinely different execution models (see cloudforger's
train CLI for how both get dispatched from one --method flag).
mincontrast_g mirrors mincontrast's shape exactly (same crop_and_rescale/
fit/fit_multistart/estimate contract) so scripts/train.py's
run_classical_baseline (`module = getattr(baselines, method_name)`) picks
either up identically -- see mincontrast_g.py's module docstring for the
K-vs-G distinction."""

from . import mincontrast
from . import mincontrast_g
from . import palm
from . import vihrs

__all__ = ["mincontrast", "mincontrast_g", "palm", "vihrs"]
