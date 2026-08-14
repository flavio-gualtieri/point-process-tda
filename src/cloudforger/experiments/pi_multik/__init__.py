# src/cloudforger/experiments/pi_multik/__init__.py
"""The reference k=5,10,15 experiment (docs/architecture.md) and its
sibling variants, consolidated into one subpackage. All five modules were
flat files directly under nn/experiments/ before the pipeline-housekeeping
refactor; filenames were kept identical during the move (pure relocation,
see docs/refactor_inventory.md) so e.g. pi_multik.py's own content is
untouched other than import-path fixes.

  pi_multik.py             -- PIMultiK + PIMultiKExperiment, the canonical
                               method. PIMultiK composes two independent,
                               config-selectable axes: encoder_mode (shared-
                               vs independent-weight CoordConvPIEncoder per
                               k, via cloudforger.encoders.EncoderBank) and
                               fusion_mode/fusion_pool (flat concat vs
                               ConvFusion's avg-pool/flatten Conv1d-over-k --
                               cloudforger.encoders.scaleconv_pi.ConvFusion).
                               Also home to the shared load_multik_split/
                               build_pi_tensor/build_extra helpers every
                               sibling below reuses.
  pi_multik_scaleconv.py   -- preset: shared encoders + ConvFusion(avg).
  pi_multik_towers.py      -- preset: independent encoders + ConvFusion
                               (flatten). Both presets are thin _build_model
                               overrides on PIMultiKExperiment/PIMultiK --
                               every default stays overridable per-run via
                               method.params (e.g. encoder_mode,
                               fusion_pool), so new encoder/fusion
                               combinations are a config change, not a new
                               file.
  pi_multik_earlyfusion.py -- the pre-late-fusion design (all k's channels
                               stacked before the first conv layer), kept as
                               an explicit comparison sibling -- a different
                               fusion *stage* (pixel-level, pre-encoder),
                               not a point in the encoder_mode/fusion_mode
                               space above.
  pi_multik_fusion.py      -- fuses vihrs's L(r)-r branch with pi_multik's
                               multi-k branch -- a different, cross-modal
                               fusion problem, likewise not a point in that
                               space.
"""

from .pi_multik import (
    PIMultiK,
    PIMultiKExperiment,
    build_extra,
    build_pi_tensor,
    load_multik_split,
)
from .pi_multik_scaleconv import PIMultiKScaleConvExperiment
from .pi_multik_towers import PIMultiKTowersExperiment
from .pi_multik_earlyfusion import PIMultiKEarlyFusion, PIMultiKEarlyFusionExperiment
from .pi_multik_fusion import VihrsPIMultiKFusion, PIMultiKFusionExperiment

__all__ = [
    "PIMultiK",
    "PIMultiKExperiment",
    "build_extra",
    "build_pi_tensor",
    "load_multik_split",
    "PIMultiKScaleConvExperiment",
    "PIMultiKTowersExperiment",
    "PIMultiKEarlyFusion",
    "PIMultiKEarlyFusionExperiment",
    "VihrsPIMultiKFusion",
    "PIMultiKFusionExperiment",
]
