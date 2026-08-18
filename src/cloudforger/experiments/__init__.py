# src/cloudforger/experiments/__init__.py

from .base import Experiment, register, build_experiment, REGISTRY
from .common import MultiSourceExperiment
from .raw_pc import RawPointCloudExperiment
from .pairwise import PairwiseExperiment
from .persistence_image import PersistenceImageExperiment
from .mph_pi import MPHImageExperiment
from .mph_fusion import MPHFusionExperiment
from .fusion import FusionExperiment
from .pi_multik import (
    PIMultiKExperiment,
    PIMultiKFusionExperiment,
    PIMultiKScaleConvExperiment,
    PIMultiKTowersExperiment,
    PIMultiKEarlyFusionExperiment,
    BettiMultiKExperiment,
    VectorizedMultiKExperiment,
)
# betti_cnn imports cloudforger.experiments.pi_multik.pi_multik (for
# build_extra) -- must come after the .pi_multik import block above so that
# subpackage is already fully initialized rather than importing it early,
# mid-way through this file's own execution.
from .betti_cnn import BettiCNNExperiment


__all__ = [
    "Experiment",
    "MultiSourceExperiment",
    "register",
    "build_experiment",
    "REGISTRY",
    "RawPointCloudExperiment",
    "PairwiseExperiment",
    "PersistenceImageExperiment",
    "MPHImageExperiment",
    "MPHFusionExperiment",
    "FusionExperiment",
    "PIMultiKExperiment",
    "PIMultiKFusionExperiment",
    "PIMultiKScaleConvExperiment",
    "PIMultiKTowersExperiment",
    "PIMultiKEarlyFusionExperiment",
    "BettiMultiKExperiment",
    "BettiCNNExperiment",
    "VectorizedMultiKExperiment",
]