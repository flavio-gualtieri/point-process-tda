# src/cloudforger/experiments/pi_multik/__init__.py
"""The reference k=5,10,15 experiment (docs/architecture.md) and its
sibling variants, consolidated into one subpackage. All five modules were
flat files directly under nn/experiments/ before the pipeline-housekeeping
refactor; filenames were kept identical during the move (pure relocation,
see docs/refactor_inventory.md) so e.g. pi_multik.py's own content is
untouched other than import-path fixes.

  pi_multik.py             -- PIMultiK (late-fusion, shared-weight CoordConv
                               branch per k) + PIMultiKExperiment, the
                               canonical method. Also home to the shared
                               load_multik_split/build_pi_tensor/build_extra
                               helpers every sibling below reuses.
  pi_multik_scaleconv.py   -- swaps in ScaleConvFusion (Conv1d over the
                               ordered k axis) instead of flat concat.
  pi_multik_towers.py      -- per-k encoder towers + TowerConv fusion.
  pi_multik_earlyfusion.py -- the pre-late-fusion design (all k's channels
                               stacked before the first conv layer), kept as
                               an explicit comparison sibling.
  pi_multik_fusion.py      -- fuses vihrs's L(r)-r branch with pi_multik's
                               multi-k branch.
"""

from .pi_multik import (
    PIMultiK,
    PIMultiKExperiment,
    build_extra,
    build_pi_tensor,
    load_multik_split,
)
from .pi_multik_scaleconv import PIMultiKScaleConvExperiment
from .pi_multik_towers import PIMultiKTowers, PIMultiKTowersExperiment
from .pi_multik_earlyfusion import PIMultiKEarlyFusion, PIMultiKEarlyFusionExperiment
from .pi_multik_fusion import VihrsPIMultiKFusion, PIMultiKFusionExperiment

__all__ = [
    "PIMultiK",
    "PIMultiKExperiment",
    "build_extra",
    "build_pi_tensor",
    "load_multik_split",
    "PIMultiKScaleConvExperiment",
    "PIMultiKTowers",
    "PIMultiKTowersExperiment",
    "PIMultiKEarlyFusion",
    "PIMultiKEarlyFusionExperiment",
    "VihrsPIMultiKFusion",
    "PIMultiKFusionExperiment",
]
