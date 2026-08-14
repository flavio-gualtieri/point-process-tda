# src/cloudforger/experiments/pi_multik/pi_multik_scaleconv.py

from __future__ import annotations

from cloudforger.experiments.base import register
from cloudforger.experiments.pi_multik.pi_multik import PIMultiK, PIMultiKExperiment


@register("pi_multik_scaleconv")
class PIMultiKScaleConvExperiment(PIMultiKExperiment):
    @property
    def subdir(self) -> str:
        return "pi_multik_scaleconv"

    def _build_model(self, **kwargs) -> PIMultiK:
        """Shared-weight encoder + ConvFusion(pool="avg") -- Conv1d over the
        ordered k axis, then averaged over k into a scale-count-invariant
        vector. encoder_mode/fusion_pool stay overridable via method.params
        (e.g. encoder_mode: independent, or fusion_pool: flatten) without
        needing a new file/class -- see PIMultiK's docstring."""
        kwargs.setdefault("encoder_mode", "shared")
        kwargs.setdefault("fusion_mode", "conv")
        kwargs.setdefault("fusion_pool", "avg")
        return PIMultiK(**kwargs)
