# src/cloudforger/experiments/__init__.py

from .base import Experiment, register, build_experiment, REGISTRY
from .common import MultiSourceExperiment
from .raw_pc import RawPointCloudExperiment
from .pairwise import PairwiseExperiment
from .persistence_image import PersistenceImageExperiment
from .fusion import FusionExperiment
from .logn_only import LogNOnlyExperiment


__all__ = [
    "Experiment",
    "MultiSourceExperiment",
    "register",
    "build_experiment",
    "REGISTRY",
    "RawPointCloudExperiment",
    "PairwiseExperiment",
    "PersistenceImageExperiment",
    "FusionExperiment",
    "LogNOnlyExperiment",
]