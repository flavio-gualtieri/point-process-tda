from ...core.registry import Registry
from .base import DiagramFeature
from .result import BettiCurveFeature
from .betti_curve import BettiCurve
from .persistence_entropy import PersistenceEntropy

__all__ = [
    "REGISTRY",
    "DiagramFeature",
    "BettiCurveFeature",
    "BettiCurve",
    "PersistenceEntropy",
]

REGISTRY: Registry[DiagramFeature] = Registry("feature")
REGISTRY.register("betti_curve")(BettiCurve)
REGISTRY.register("persistence_entropy")(PersistenceEntropy)
