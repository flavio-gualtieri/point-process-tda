from ...core.registry import Registry
from .base import DiagramFeature
from .result import BettiCurveFeature
from .betti_curve import BettiCurve
from .persistence_entropy import PersistenceEntropy
from .persistence_statistics import PersistenceStatistics

__all__ = [
    "REGISTRY",
    "DiagramFeature",
    "BettiCurveFeature",
    "BettiCurve",
    "PersistenceEntropy",
    "PersistenceStatistics",
]

REGISTRY: Registry[DiagramFeature] = Registry("feature")
REGISTRY.register("betti_curve")(BettiCurve)
REGISTRY.register("persistence_entropy")(PersistenceEntropy)
REGISTRY.register("persistence_statistics")(PersistenceStatistics)
