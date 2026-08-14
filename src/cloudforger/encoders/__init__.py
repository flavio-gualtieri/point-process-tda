from ..core.registry import Registry

from .base import Encoder
from .persistence_image import PIEncoder
from .point_cloud import PointNetEncoder
from .sequence_cnn import SequenceCNNEncoder
from .stats import StatsEncoder
from .coordconv_pi import CoordConvPIEncoder
from .encoder_bank import EncoderBank
from .scaleconv_pi import ConvFusion

__all__ = [
    "REGISTRY",
    "Encoder",
    "PIEncoder",
    "PointNetEncoder",
    "SequenceCNNEncoder",
    "StatsEncoder",
    "CoordConvPIEncoder",
    "EncoderBank",
    "ConvFusion",
]

# New in the pipeline-housekeeping refactor (docs/architecture.md): every
# experiment still imports its encoder directly by module path (e.g.
# `from cloudforger.encoders.coordconv_pi import CoordConvPIEncoder`) --
# nothing existing was changed to build through this registry, so this is
# purely additive. It closes the one pluggable axis in the pipeline that
# didn't already have a Registry[T] (point processes, filtrations,
# bifiltrations, and scalar features all did): an encoder is now also
# nameable/buildable generically, e.g. REGISTRY.build("coordconv_pi", **kwargs).
REGISTRY: Registry[Encoder] = Registry("encoder")
REGISTRY.register("persistence_image")(PIEncoder)
REGISTRY.register("point_cloud")(PointNetEncoder)
REGISTRY.register("sequence_cnn")(SequenceCNNEncoder)
REGISTRY.register("stats")(StatsEncoder)
REGISTRY.register("coordconv_pi")(CoordConvPIEncoder)
REGISTRY.register("encoder_bank")(EncoderBank)
REGISTRY.register("scaleconv_pi")(ConvFusion)
