from ...core.registry import Registry
from ...core.base import PointProcess
from .poisson import PoissonProcess
from .matern import MaternHardCoreProcess
from .matern_cluster import MaternClusterProcess
from .neyman_scott import NeymanScottProcess
from .thomas import ThomasProcess
from .nested_thomas import NestedThomasProcess
from .inhom_thomas import InhomThomas
from .kernels import Kernel, GaussianKernel, BallKernel

__all__ = [
    "REGISTRY",
    "PoissonProcess",
    "MaternHardCoreProcess",
    "MaternClusterProcess",
    "NeymanScottProcess",
    "ThomasProcess",
    "NestedThomasProcess",
    "InhomThomas",
    "Kernel",
    "GaussianKernel",
    "BallKernel",
]

REGISTRY: Registry[PointProcess] = Registry("process")
REGISTRY.register("poisson")(PoissonProcess)
REGISTRY.register("matern")(MaternHardCoreProcess)
REGISTRY.register("matern_cluster")(MaternClusterProcess)
REGISTRY.register("neyman_scott")(NeymanScottProcess)
REGISTRY.register("thomas")(ThomasProcess)
REGISTRY.register("nested_thomas")(NestedThomasProcess)
REGISTRY.register("inhom_thomas")(InhomThomas)
