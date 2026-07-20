# Deprecated: moved to cloudforger.features.result (it's a feature-computation
# output, not a core domain type). Shim for not-yet-migrated callers.

from ..features.result import BettiCurveFeature

__all__ = ["BettiCurveFeature"]
