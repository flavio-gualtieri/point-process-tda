# src/cloudforger/nn/experiments/pi_multik_scaleconv.py

from __future__ import annotations

from cloudforger.nn.experiments.base import register
from cloudforger.nn.experiments.pi_multik import PIMultiK, PIMultiKExperiment


@register("pi_multik_scaleconv")
class PIMultiKScaleConvExperiment(PIMultiKExperiment):
    @property
    def subdir(self) -> str:
        return "pi_multik_scaleconv"

    def _build_model(self, **kwargs) -> PIMultiK:
        return PIMultiK(use_fusion=True, **kwargs)
