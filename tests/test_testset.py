"""The stratified test set: balance, disjointness, determinism, semantics."""

from __future__ import annotations

import numpy as np
import pytest

from cloudforger.evaluation import testset as T


def test_allocate_is_even_when_capacity_allows():
    assert T.allocate([100, 100, 100], 30) == [10, 10, 10]


def test_allocate_spills_off_a_full_bin():
    take = T.allocate([5, 100, 100], 30)
    assert take[0] == 5 and sum(take) == 30


def test_allocate_reports_short_rather_than_overdrawing():
    take = T.allocate([2, 3], 30)
    assert take == [2, 3] and sum(take) < 30


def test_stratum_edges_anchor_on_the_detection_threshold():
    # delta <= 1 is exactly "classical 5% test does not reject"; the first
    # stratum boundary must sit there and nowhere else.
    assert T.STRATUM_EDGES[1] == 1.0
    idx = T.stratum_index(np.array([0.0, 0.999, 1.0, 1.5, 2.0, 3.9, 4.0, 1e9]))
    assert [T.STRATUM_NAMES[i] for i in idx] == [
        "below", "below", "weak", "weak", "moderate", "moderate", "strong", "strong"]


@pytest.fixture(scope="module")
def built():
    try:
        return T.build(per_cell=50, seed=1)
    except FileNotFoundError as exc:
        pytest.skip(f"generated data not present: {exc}")


def test_every_cell_is_balanced(built):
    ts, _ = built
    counts = ts.counts()
    assert set(counts) == set(T.FAMILIES)
    for fam, row in counts.items():
        assert set(row) == set(T.STRATUM_NAMES), fam
        assert set(row.values()) == {50}, fam


def test_cases_are_never_reused(built):
    ts, _ = built
    ids = [r["case_id"] for r in ts.rows]
    assert len(ids) == len(set(ids))


def test_null_family_is_spread_over_every_stratum(built):
    """CSR has delta = 0, so a delta cut alone would put all of it in `below`.
    It is held at a constant share instead, to keep the class prior uniform."""
    ts, _ = built
    nulls = [r for r in ts.rows if r["is_null"]]
    per = {}
    for r in nulls:
        per[r["stratum"]] = per.get(r["stratum"], 0) + 1
    assert per == {s: 50 for s in T.STRATUM_NAMES}
    assert all(r["delta_tilde"] == 0.0 for r in nulls)


def test_structured_rows_sit_in_the_stratum_their_delta_implies(built):
    ts, _ = built
    for r in ts.rows:
        if r["is_null"]:
            continue
        expected = T.STRATUM_NAMES[int(T.stratum_index(np.array([r["delta_tilde"]]))[0])]
        assert r["stratum"] == expected, r


def test_build_is_deterministic():
    try:
        a, _ = T.build(per_cell=25, seed=7)
        b, _ = T.build(per_cell=25, seed=7)
    except FileNotFoundError as exc:
        pytest.skip(f"generated data not present: {exc}")
    assert [r["case_id"] for r in a.rows] == [r["case_id"] for r in b.rows]


def test_theta_id_separates_replicates_from_prior_draws():
    # fixed-theta rows share an id across replicates; prior rows do not.
    assert T.theta_id("B", "thomas", 3, -1, 30001) == T.theta_id("B", "thomas", 3, -1, 30002)
    assert T.theta_id("A", "thomas", -1, -1, 1) != T.theta_id("A", "thomas", -1, -1, 2)


def test_roundtrip_csv(tmp_path, built):
    ts, _ = built
    path = T.write_csv(ts, tmp_path / "testset.csv")
    back = T.load(path)
    assert len(back) == len(ts)
    assert [r["case_id"] for r in back.rows] == [r["case_id"] for r in ts.rows]
    assert [r["stratum"] for r in back.rows] == [r["stratum"] for r in ts.rows]
