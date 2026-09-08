from ...core.registry import Registry
from .base import Filtration
from .rips import RipsFiltration
from .dtm import DTMFiltration
from .lfunc import (
    LDTMFiltration,
    LFuncFiltration,
    LRipsFiltration,
    apply_l_transform,
    monotone_l,
)
from .bifiltration import Bifiltration, DtmRipsBifiltration

__all__ = [
    "REGISTRY",
    "BIFILTRATION_REGISTRY",
    "Filtration",
    "RipsFiltration",
    "DTMFiltration",
    "LFuncFiltration",
    "LRipsFiltration",
    "LDTMFiltration",
    "apply_l_transform",
    "monotone_l",
    "Bifiltration",
    "DtmRipsBifiltration",
]

REGISTRY: Registry[Filtration] = Registry("filtration")
REGISTRY.register("rips")(RipsFiltration)
REGISTRY.register("dtm")(DTMFiltration)
# L-reparameterized variants: (b, d) -> (L(b), L(d)); see lfunc.py.
# Path tags are l_rips / l_dtm_k<k>, so their diagrams and results live
# alongside (never overwrite) the untransformed ones.
REGISTRY.register("l_rips")(LRipsFiltration)
REGISTRY.register("l_dtm")(LDTMFiltration)

BIFILTRATION_REGISTRY: Registry[Bifiltration] = Registry("bifiltration")
BIFILTRATION_REGISTRY.register("mph_dtm")(DtmRipsBifiltration)
