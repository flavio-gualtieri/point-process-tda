# src/cloudforger/baselines/__init__.py
"""Non-TDA comparison baselines: classical point-process estimators
(mincontrast, mincontrast_g, mincontrast_nested, mincontrast_g_nested, palm)
and the vihrs neural baseline. Each module exposes a prepare-once/fit-or-
run-per-seed pair (mincontrast.estimate / mincontrast_g.estimate /
mincontrast_nested.estimate / mincontrast_g_nested.estimate / palm.estimate
for classical per-cloud fits, vihrs.prepare_data + vihrs.run_one_seed for
the trained-per-seed CNN) -- deliberately not force-unified into a single
class shape here, since the two families have genuinely different
execution models (see cloudforger's train CLI for how both get dispatched
from one --method flag).
mincontrast_g mirrors mincontrast's shape exactly (same crop_and_rescale/
fit/fit_multistart/estimate contract) so scripts/train.py's
run_classical_baseline (`module = getattr(baselines, method_name)`) picks
either up identically -- see mincontrast_g.py's module docstring for the
K-vs-G distinction. mincontrast_nested/mincontrast_g_nested are the same
K/g split applied to the two-level Nested Thomas process's closed form
(writeup's \\eqref{eq:g-nested}) instead of the single-level Thomas one --
see mincontrast_nested.py's module docstring."""

from . import mincontrast
from . import mincontrast_g
from . import mincontrast_nested
from . import mincontrast_g_nested
from . import palm
from . import vihrs

__all__ = ["mincontrast", "mincontrast_g", "mincontrast_nested", "mincontrast_g_nested", "palm", "vihrs"]
