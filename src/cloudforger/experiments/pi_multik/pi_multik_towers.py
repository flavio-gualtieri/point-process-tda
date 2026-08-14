# src/cloudforger/experiments/pi_multik/pi_multik_towers.py

from __future__ import annotations

from cloudforger.experiments.base import register
from cloudforger.experiments.pi_multik.pi_multik import PIMultiK, PIMultiKExperiment


@register("pi_multik_towers")
class PIMultiKTowersExperiment(PIMultiKExperiment):
    @property
    def subdir(self) -> str:
        return "pi_multik_towers"

    def _build_model(self, **kwargs) -> PIMultiK:
        """Independent per-k encoders (no weight sharing) + ConvFusion(pool=
        "flatten") -- Conv1d over the ordered k axis, keeping every k
        position instead of pooling them away. Used to be a separate
        PIMultiKTowers model class duplicating PIMultiK's forward pass and
        TowerConv duplicating ScaleConvFusion; both folded into PIMultiK/
        ConvFusion (see their docstrings) once encoder-sharing and K-fusion
        became independent, config-selectable axes instead of one-per-file."""
        kwargs.setdefault("encoder_mode", "independent")
        kwargs.setdefault("fusion_mode", "conv")
        kwargs.setdefault("fusion_pool", "flatten")
        return PIMultiK(**kwargs)
