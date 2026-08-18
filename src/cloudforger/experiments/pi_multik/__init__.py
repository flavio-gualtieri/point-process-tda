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
  betti_multik.py           -- the Betti-curve/Euler-characteristic sibling
                               of pi_multik.py: same k_values/seeds/splits/
                               train_idx-fit-apply-frozen calibration
                               discipline and late-fusion-across-k design,
                               with EncoderBank's per-k Conv2D image encoder
                               swapped for SequenceEncoderBank's per-k
                               Vihrs-style Conv1D curve encoder. Reuses
                               load_multik_split/build_extra from
                               pi_multik.py; see its own module docstring
                               for what it replaces (betti.py/betti_cnn.py/
                               ph_combined.py, all deleted).
  vectorized_multik.py       -- generalizes pi_multik.py's design across a
                               `vectorization` axis (persistence_image /
                               landscape / silhouette / persistence_
                               statistics) and an `encoder_path` axis
                               (native CNN / shared flatten-MLP), registered
                               as method: vec_multik. vectorization=
                               persistence_image, encoder_path=native
                               delegates straight to PIMultiKExperiment
                               (byte-identical to method: pi_multik) rather
                               than reimplementing it -- see that module's
                               own docstring.
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
from .betti_multik import BettiMultiK, BettiMultiKExperiment, build_betti_tensor
from .vectorized_multik import (
    VectorizedMultiK,
    VectorizedMultiKExperiment,
    build_landscape_tensor,
    build_silhouette_tensor,
    build_stats_tensor,
)

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
    "BettiMultiK",
    "BettiMultiKExperiment",
    "build_betti_tensor",
    "VectorizedMultiK",
    "VectorizedMultiKExperiment",
    "build_landscape_tensor",
    "build_silhouette_tensor",
    "build_stats_tensor",
]
