import math

import numpy as np
import pandas as pd
import pytest

from cloudforger.departure.tables import Tables
from cloudforger.simulation.bank import DATA, Config, draw_theta, sample_patterns
from cloudforger.simulation.families import FAMILIES, Rules, cv

TABLES = Tables()
RULES = Rules.load()
CFG = Config.load()
STRUCTURED = ["thomas", "nested", "matern2", "lgcp"]


@pytest.mark.parametrize("name", STRUCTURED)
def test_draw_respects_the_rules(name):
    fam = FAMILIES[name](RULES)
    for i in range(8):
        t = draw_theta(fam, TABLES, CFG, i)
        p, nbar = t["model"], t["nbar"]
        assert fam.nbar(p) == pytest.approx(nbar, rel=1e-9)     # the draw really hits its nbar
        assert cv(fam, p) <= RULES.cv_max
        assert set(fam.params) <= set(p)
        if name in ("thomas", "nested"):
            assert p["kappa"] >= RULES.kappa_min
            sigma = p["sigma"] if name == "thomas" else p["sigma1"]
            assert RULES.omega[0] <= sigma * math.sqrt(p["kappa"]) <= RULES.omega[1]
        if name == "thomas":
            assert p["mu"] >= RULES.mu_min
        if name == "nested":
            assert p["mu2"] >= RULES.mu2_min and p["mu1"] * p["mu2"] >= RULES.meta_min
            assert RULES.rho[0] <= p["sigma1"] / p["sigma2"] <= RULES.rho[1]
        if name == "matern2":
            assert p["R"] * math.sqrt(nbar) >= RULES.core_min
            assert math.pi * p["R"] ** 2 * nbar <= RULES.matern_fill


@pytest.mark.parametrize("name", STRUCTURED)
def test_richness_and_smearing_are_independent(name):
    """The two dials must both move: a bank drawn on one of them is the failure the design fixes."""
    fam = FAMILIES[name](RULES)
    drawn = [draw_theta(fam, TABLES, CFG, i)["model"] for i in range(60)]
    for key in {"thomas": ("mu", "sigma"), "nested": ("mu2", "sigma1"),
                "matern2": ("R",), "lgcp": ("sigma2", "s")}[name]:
        v = np.array([p[key] for p in drawn])
        assert v.max() / v.min() > 3


@pytest.mark.skipif(not (DATA / "thomas" / "manifest.csv").exists(), reason="no bank on disk")
@pytest.mark.parametrize("name", ["poisson", "matern2", "lgcp"])
def test_stored_pattern_regenerates(name):
    m = pd.read_csv(DATA / name / "manifest.csv")
    z = np.load(DATA / name / "points.npz")
    k = CFG.reps * 123 + 1
    theta = draw_theta(FAMILIES[name](RULES), TABLES, CFG, 123)
    _, pts, _ = list(sample_patterns(name, theta, CFG, 123))[1]
    assert m.case_id[k] == f"{name}-00123-1"
    np.testing.assert_array_equal(pts, z["points"][z["offsets"][k]:z["offsets"][k + 1]])


def test_split_by_theta():
    from cloudforger.simulation.split import split_of

    from cloudforger.simulation.split import TEST_END, TRAIN_END, VAL_END

    s = split_of(np.arange(TEST_END + 2000))
    counts = [(s[:TEST_END] == k).sum() for k in ("train", "val", "test")]
    assert counts == [TRAIN_END, VAL_END - TRAIN_END, TEST_END - VAL_END]
    assert (s[TEST_END:] == "train").all()          # a larger sweep only ever grows train
