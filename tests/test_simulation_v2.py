"""The v2 bank's rules (configs/v2/simulation.yaml over configs/simulation.yaml) and the Strauss sampler."""

import math

import numpy as np
import pytest
import yaml
from scipy.spatial import cKDTree

from cloudforger.departure.tables import Tables
from cloudforger.paths import CONFIGS, _merge
from cloudforger.simulation.bank import Config, draw_theta, sample_patterns
from cloudforger.simulation.families import FAMILIES, Rules, cv
from cloudforger.simulation.processes import strauss, strauss_sweeps

_CFG = _merge(yaml.safe_load((CONFIGS / "simulation.yaml").read_text()),
              yaml.safe_load((CONFIGS / "v2" / "simulation.yaml").read_text()))
RULES = Rules(**{k: tuple(v) if isinstance(v, list) else v for k, v in _CFG["rules"].items()})
CFG = Config(int(_CFG["root"]), (_CFG["nbar"]["low"], _CFG["nbar"]["high"]), int(_CFG["thetas"]), int(_CFG["reps"]),
             (_CFG["n"]["low"], _CFG["n"]["high"]), int(_CFG["shard_size"]), tuple(_CFG["families"]), 1, _CFG["set"])
TABLES = Tables()


def test_cv_limit_is_a_count_of_clusters():
    assert Rules.load(CONFIGS / "simulation.yaml").cv_limit(100.0) == 0.30         # the first bank: a constant
    assert RULES.cv_limit(100.0) == pytest.approx(math.sqrt(1 / 100 + 1 / RULES.kappa_eff_min))


@pytest.mark.parametrize("name", ["thomas", "nested", "lgcp", "ring", "matern2", "matern1", "cell", "strauss"])
def test_v2_draw_respects_the_rules(name):
    fam = FAMILIES[name](RULES)
    for i in range(6):
        t = draw_theta(fam, TABLES, CFG, i, set_=CFG.set)
        p, nbar = t["model"], t["nbar"]
        assert CFG.nbar[0] <= nbar <= CFG.nbar[1]
        assert cv(fam, p) <= RULES.cv_limit(fam.nbar(p)) + 1e-9
        if name in ("thomas", "nested", "ring"):
            assert p["kappa"] >= RULES.kappa_min
        if name == "strauss":
            assert p["nbar"] == round(nbar) and RULES.strauss_q[0] <= p["q"] <= RULES.strauss_q[1]
            assert RULES.core_min <= p["R"] * math.sqrt(p["nbar"]) <= RULES.strauss_core_max


def test_v2_streams_are_not_the_first_banks():
    fam = FAMILIES["thomas"](RULES)
    assert draw_theta(fam, TABLES, CFG, 0, set_="bank_v2")["nbar"] != draw_theta(fam, TABLES, CFG, 0, set_="bank")["nbar"]


def _pairs(x, R):
    return len(cKDTree(x, boxsize=1.0).query_pairs(R))


def test_strauss_count_is_exact_and_reproducible():
    fam = FAMILIES["strauss"](RULES)
    theta = draw_theta(fam, TABLES, CFG, 3, set_=CFG.set)
    a = [pts for _, pts, _ in sample_patterns("strauss", theta, CFG, 3, set_=CFG.set)]
    b = [pts for _, pts, _ in sample_patterns("strauss", theta, CFG, 3, set_=CFG.set)]
    assert all(len(x) == theta["model"]["nbar"] for x in a)
    assert all(np.array_equal(x, y) for x, y in zip(a, b)) and not np.array_equal(a[0], a[1])


def test_strauss_hard_core_has_no_close_pair():
    n = 150
    for c in (0.5, 0.8):                                         # packing 0.2 and 0.5
        R = c / math.sqrt(n)
        x = strauss(np.random.default_rng(0), n, 1.0, R)
        assert x.shape == (n, 2) and ((x >= 0) & (x < 1)).all() and _pairs(x, R) == 0


def test_strauss_interpolates_between_binomial_and_hard_core():
    n = 150
    R = 0.5 / math.sqrt(n)
    binomial = n * (n - 1) / 2 * math.pi * R**2                  # expected close pairs at q = 0
    mean = lambda q: np.mean([_pairs(strauss(np.random.default_rng(s), n, q, R), R) for s in range(12)])
    weak, mid = mean(0.01), mean(0.7)
    assert weak == pytest.approx(binomial, rel=0.15)
    assert 0.15 * binomial < mid < 0.6 * binomial


def test_strauss_has_converged_at_its_own_sweeps():
    """Twice the sweeps changes neither the close-pair count nor the nearest-neighbour distance."""
    n, q = 150, 0.9
    R = 0.7 / math.sqrt(n)
    base = strauss_sweeps(q, R, n)
    def stats(sweeps):
        xs = [strauss(np.random.default_rng(s), n, q, R, sweeps=sweeps) for s in range(16)]
        nn = [cKDTree(x, boxsize=1.0).query(x, 2)[0][:, 1].mean() for x in xs]
        return np.mean([_pairs(x, R) for x in xs]), np.mean(nn), np.std(nn) / 4
    (p1, d1, se), (p2, d2, _) = stats(base), stats(2 * base)
    assert abs(d1 - d2) < 4 * se and abs(p1 - p2) < max(3.0, 0.3 * p2)
