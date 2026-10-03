"""The pipeline over the v2 library (configs/v2): every family it trains on is estimable, has a regime
coordinate, round-trips to its sampler, and the minimum-contrast baseline covers it."""

import inspect

import numpy as np
import pandas as pd
import pytest
import yaml

from cloudforger.baselines.mincontrast import FALLBACK, FITTABLE, models
from cloudforger.paths import CONFIGS, _merge
from cloudforger.pipeline.core import TARGETS
from cloudforger.pipeline.regime import COORDINATES
from cloudforger.scores.simulate import sampler_kwargs, true_model
from cloudforger.simulation.processes import SAMPLERS


V1 = yaml.safe_load((CONFIGS / "pipeline.yaml").read_text())
V2 = _merge(V1, yaml.safe_load((CONFIGS / "v2" / "pipeline.yaml").read_text()))


def test_v2_trains_on_strauss_and_not_matern1():
    assert "strauss" in V2["families"] and "matern1" not in V2["families"]
    assert V2["regime"]["coordinates"]["strauss"] == "zeta"
    assert V2["regime"]["coordinates"]["thomas"] == "omega"           # merged into the base, not replacing it


@pytest.mark.parametrize("cfg", [V1, V2], ids=["v1", "v2"])
def test_every_family_is_estimable_and_has_a_regime(cfg):
    for f in cfg["families"]:
        assert f in TARGETS
        coord = cfg["regime"]["coordinates"].get(f)
        if f not in ("poisson", "cell"):
            assert coord in COORDINATES[f]


@pytest.mark.parametrize("cfg", [V1, V2], ids=["v1", "v2"])
def test_mincontrast_covers_every_family(cfg):
    """Each estimated family is either fitted by K or given the fallback, never both, never neither."""
    fitted = models(cfg["families"])
    for f in cfg["families"]:
        if f != "poisson":
            assert (f in fitted) != (f in FALLBACK)
    assert models(V1["families"]) == list(FITTABLE)                     # v1: the original six, in order


def test_fallback_scales_lengths_with_n():
    train = pd.DataFrame({"q": [0.2, 0.5, 0.8], "R": [0.01, 0.02, 0.03], "nbar": [400.0, 100.0, 44.4]})
    theta = FALLBACK["strauss"](train, np.array([100.0, 400.0]))
    assert np.allclose(theta[:, 0], [100, 400])
    assert np.allclose(theta[:, 2] * np.sqrt([100, 400]), 0.2)          # the median core, R sqrt(n)


@pytest.mark.parametrize("family", ["strauss", "lgcp", "thomas"])
def test_true_model_takes_only_the_models_arguments(family):
    """Sampler arguments with a default (lgcp's root_lam, strauss's sweeps) are not model parameters."""
    required = [a for a, p in inspect.signature(SAMPLERS[family]).parameters.items()
                if a != "rng" and p.default is inspect.Parameter.empty]
    row = pd.Series({"family": family, **{a: 1.0 for a in required}, "sweeps": np.nan, "root_lam": np.nan})
    _, kw = true_model(row)
    assert sorted(kw) == sorted(required)


def test_strauss_targets_are_its_sampler_arguments():
    theta = {t: 1.0 for t in TARGETS["strauss"]}
    kw = sampler_kwargs("strauss", theta)
    assert set(kw) <= set(inspect.signature(SAMPLERS["strauss"]).parameters)
