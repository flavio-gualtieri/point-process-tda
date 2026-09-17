import numpy as np
import pandas as pd
import pytest

from cloudforger.departure.tables import Tables
from cloudforger.simulation.families import FAMILIES, Rules, cv, delta
from cloudforger.simulation.sweep import DATA, Config, draw_theta, sample_patterns

TABLES = Tables()
RULES = Rules.load(TABLES)
CFG = Config.load()


@pytest.mark.parametrize("name", ["thomas", "nested", "lgcp", "matern2"])
def test_theta_hits_target_within_rules(name):
    fam = FAMILIES[name](RULES)
    for i in range(5):
        t = draw_theta(fam, TABLES, CFG, i)
        assert fam.valid_shape(t["nbar"], t["shape"])
        assert delta(fam, TABLES, t["nbar"], t["shape"], t["amp"]) == pytest.approx(t["delta"], rel=1e-6)
        assert cv(fam, t["nbar"], t["shape"], t["amp"]) <= RULES.cv_max


def test_rules_reproduce_old_bounds():
    assert FAMILIES["thomas"](RULES).shape_box(400)["sigma"][1] == pytest.approx(0.0985, abs=1e-4)
    assert FAMILIES["lgcp"](RULES).shape_box(400)["s"][1] == pytest.approx(0.1553, abs=1e-4)
    assert FAMILIES["nested"](RULES).rho_min() == pytest.approx(2.49, abs=0.01)


@pytest.mark.skipif(not (DATA / "thomas" / "manifest.csv").exists(), reason="no simulated data")
@pytest.mark.parametrize("name", ["poisson", "matern2", "lgcp"])
def test_stored_pattern_regenerates(name):
    m = pd.read_csv(DATA / name / "manifest.csv")
    z = np.load(DATA / name / "points.npz")
    k = 2 * 123 + 1
    theta = draw_theta(FAMILIES[name](RULES), TABLES, CFG, 123)
    rep, pts, _ = list(sample_patterns(name, theta, CFG, 123))[1]
    assert m.case_id[k] == f"{name}-00123-1"
    np.testing.assert_array_equal(pts, z["points"][z["offsets"][k]:z["offsets"][k + 1]])
