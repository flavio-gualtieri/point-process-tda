# src/cloudforger/nn/experiments/pi_multik_scaleconv.py
"""Scale-aware sibling of pi_multik: identical data loading and training
loop (inherited from PIMultiKExperiment verbatim), the only difference is
_build_model swapping the flat concat-across-k fusion for ScaleConvFusion
(cloudforger.nn.encoders.scaleconv_pi) -- a small Conv1d block over the
ordered k axis, so the head sees k-adjacency instead of an order-blind bag
of per-k embeddings.

Registered under its own name (not a use_fusion flag on pi_multik) so it
gets its own results/<process>/<filtration_tag>/pi_multik_scaleconv/
tree, mirroring how pi_multik_fusion is a sibling experiment of pi_multik
rather than a flag -- keeps the flat-concat baseline in pi_multik/ from
ever being overwritten by a scale-aware run, and avoids colliding with
"fusion" already meaning vihrs late-fusion elsewhere in this repo."""

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
