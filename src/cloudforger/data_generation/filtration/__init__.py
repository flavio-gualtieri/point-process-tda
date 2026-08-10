from ...core.registry import Registry
from .base import Filtration
from .rips import RipsFiltration
from .dtm import DTMFiltration
from .bifiltration import Bifiltration, DtmRipsBifiltration

__all__ = [
    "REGISTRY",
    "BIFILTRATION_REGISTRY",
    "Filtration",
    "RipsFiltration",
    "DTMFiltration",
    "Bifiltration",
    "DtmRipsBifiltration",
]

REGISTRY: Registry[Filtration] = Registry("filtration")
REGISTRY.register("rips")(RipsFiltration)
REGISTRY.register("dtm")(DTMFiltration)

BIFILTRATION_REGISTRY: Registry[Bifiltration] = Registry("bifiltration")
BIFILTRATION_REGISTRY.register("mph_dtm")(DtmRipsBifiltration)
