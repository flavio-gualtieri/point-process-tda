from ...core.registry import Registry
from ...core.base import PointProcess
from .poisson import PoissonProcess
from .matern import MaternHardCoreProcess
from .neyman_scott import NeymanScottProcess
from .thomas import ThomasProcess
from .nested_thomas import NestedThomasProcess
from .lgcp_ce import CirculantLGCPProcess
from .gibbs import StraussProcess
from .kernels import Kernel, GaussianKernel

__all__ = [
    "REGISTRY",
    "PoissonProcess",
    "MaternHardCoreProcess",
    "NeymanScottProcess",
    "ThomasProcess",
    "NestedThomasProcess",
    "CirculantLGCPProcess",
    "StraussProcess",
    "Kernel",
    "GaussianKernel",
]

REGISTRY: Registry[PointProcess] = Registry("process")
REGISTRY.register("poisson")(PoissonProcess)
REGISTRY.register("matern")(MaternHardCoreProcess)
REGISTRY.register("neyman_scott")(NeymanScottProcess)
REGISTRY.register("thomas")(ThomasProcess)
REGISTRY.register("nested_thomas")(NestedThomasProcess)
REGISTRY.register("lgcp")(CirculantLGCPProcess)
REGISTRY.register("strauss")(StraussProcess)
