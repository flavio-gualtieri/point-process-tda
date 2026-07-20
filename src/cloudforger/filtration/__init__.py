from ..core.registry import Registry
from .base import Filtration
from .rips import RipsFiltration
from .dtm import DTMFiltration

__all__ = [
    "REGISTRY",
    "Filtration",
    "RipsFiltration",
    "DTMFiltration",
]

REGISTRY: Registry[Filtration] = Registry("filtration")
REGISTRY.register("rips")(RipsFiltration)
REGISTRY.register("dtm")(DTMFiltration)
