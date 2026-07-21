# src/cloudforger/nn/experiments/__init__.py

from .base import Experiment, register, build_experiment, REGISTRY
from .common import MultiSourceExperiment
from .raw_pc import RawPointCloudExperiment
from .pairwise import PairwiseExperiment
from .persistence_image import PersistenceImageExperiment
from .betti import BettiCurveExperiment
from .betti_cnn import BettiCurveCNNExperiment
from .ph_combined import PHCombinedExperiment
from .fusion import FusionExperiment
from .pi_multik import PIMultiKExperiment
from .pi_multik_fusion import PIMultiKFusionExperiment
from .pi_multik_scaleconv import PIMultiKScaleConvExperiment

__all__ = [
    "Experiment",
    "MultiSourceExperiment",
    "register",
    "build_experiment",
    "REGISTRY",
    "RawPointCloudExperiment",
    "PairwiseExperiment",
    "PersistenceImageExperiment",
    "BettiCurveExperiment",
    "BettiCurveCNNExperiment",
    "PHCombinedExperiment",
    "FusionExperiment",
    "PIMultiKExperiment",
    "PIMultiKFusionExperiment",
    "PIMultiKScaleConvExperiment",
]