# tests/test_registries.py
"""Smoke tests: every registry can build its registered classes and run
them end to end on a tiny synthetic example -- adding a new process/
filtration/feature should be verifiable by extending these, not by writing
a new bespoke script."""

from __future__ import annotations

import numpy as np
import pytest

from cloudforger.core.region import Box
from cloudforger.vectorization.scalar_features import REGISTRY as FEATURE_REGISTRY
from cloudforger.data_generation.filtration import REGISTRY as FILTRATION_REGISTRY
from cloudforger.data_generation.point_processes import REGISTRY as PROCESS_REGISTRY


def _sample_cloud(process_name: str, **params):
    proc = PROCESS_REGISTRY.build(process_name, **params)
    region = Box(low=np.zeros(2), high=np.ones(2))
    return proc.sample(region, seed=0)


def test_process_registry_covers_expected_names():
    assert {
        "poisson", "matern", "matern_cluster", "thomas", "nested_thomas", "inhom_thomas", "neyman_scott",
        "aniso_thomas", "trend_thomas", "lgcp", "strauss", "lgcp_strauss",
    } <= set(PROCESS_REGISTRY.names())


def test_lgcp_process_builds_and_samples():
    cloud = _sample_cloud("lgcp", mu=5.0, sigma2=1.0, s=0.1)
    assert cloud.n_points >= 0
    assert cloud.points.shape[1] == 2


def test_strauss_process_builds_and_samples():
    cloud = _sample_cloud("strauss", beta=250.0, gamma=0.3, radius=0.05, n_steps=3000)
    assert cloud.n_points >= 0
    assert cloud.points.shape[1] == 2


def test_lgcp_strauss_process_builds_and_samples():
    cloud = _sample_cloud("lgcp_strauss", mu=5.0, sigma2=1.0, s=0.1, gamma=0.3, radius=0.05, n_steps=3000)
    assert cloud.n_points >= 0
    assert cloud.points.shape[1] == 2


def test_thomas_process_builds_and_samples():
    cloud = _sample_cloud("thomas", parent_intensity=50.0, mean_offspring=8.0, cluster_scale=0.03, edge_buffer=0.1)
    assert cloud.n_points >= 0
    assert cloud.points.shape[1] == 2


def test_aniso_thomas_builds_and_samples():
    cloud = _sample_cloud(
        "aniso_thomas",
        parent_intensity=60.0, mean_offspring=10.0, cluster_scale=0.03,
        cluster_aspect=3.0, cluster_theta=0.7,
    )
    assert cloud.n_points >= 0
    assert cloud.points.shape[1] == 2


def test_aniso_thomas_reduces_to_thomas_labels_when_aspect_one():
    p = PROCESS_REGISTRY.build(
        "aniso_thomas", parent_intensity=60.0, mean_offspring=10.0, cluster_scale=0.03
    ).params
    assert p["cluster_aspect"] == 1.0
    assert p["cluster_sigma_1"] == p["cluster_sigma_2"] == 0.03
    # same overlap-index diagnostic as ThomasProcess
    assert abs(p["c1"] - 2.0 * 0.03 * 60.0 ** 0.5) < 1e-12


def test_aniso_thomas_geometric_mean_scale_is_fixed():
    p = PROCESS_REGISTRY.build(
        "aniso_thomas", parent_intensity=60.0, mean_offspring=10.0, cluster_scale=0.03,
        cluster_aspect=4.0, cluster_theta=1.0,
    ).params
    assert abs(p["cluster_sigma_1"] * p["cluster_sigma_2"] - 0.03 ** 2) < 1e-12
    assert p["cluster_sigma_1"] > p["cluster_sigma_2"]


def test_trend_thomas_isotropic_builds_and_samples():
    cloud = _sample_cloud(
        "trend_thomas",
        parent_intensity=80.0, mean_offspring=8.0, cluster_scale=0.03, beta=[1.5, -1.0],
    )
    assert cloud.n_points >= 0
    assert cloud.points.shape[1] == 2


def test_trend_thomas_scalar_beta_fallback_matches_vector():
    # beta=[b0, b1] and the scalar beta_0/beta_1 stopgap must build the same process
    kw = dict(parent_intensity=80.0, mean_offspring=8.0, cluster_scale=0.03)
    from_vec = PROCESS_REGISTRY.build("trend_thomas", beta=[1.5, -1.0], **kw).params
    from_scalars = PROCESS_REGISTRY.build("trend_thomas", beta_0=1.5, beta_1=-1.0, **kw).params
    assert from_vec == from_scalars
    assert from_vec["beta_0"] == 1.5 and from_vec["beta_1"] == -1.0


def test_trend_thomas_anisotropic_preserves_geometric_mean_scale():
    p = PROCESS_REGISTRY.build(
        "trend_thomas",
        parent_intensity=80.0, mean_offspring=8.0, cluster_scale=0.03,
        beta=[1.0, 0.0], cluster_aspect=4.0, cluster_theta=0.5,
    ).params
    assert abs(p["cluster_sigma_1"] * p["cluster_sigma_2"] - 0.03 ** 2) < 1e-12
    assert p["cluster_sigma_1"] > p["cluster_sigma_2"]
    assert p["edge_buffer"] > 0.03  # buffer follows the long axis


def test_trend_thomas_zero_beta_builds_and_samples():
    cloud = _sample_cloud(
        "trend_thomas",
        parent_intensity=80.0, mean_offspring=8.0, cluster_scale=0.03, beta=[0.0, 0.0],
    )
    assert cloud.points.shape[1] == 2


def test_anisotropic_gaussian_kernel_shape_orientation_and_2d_only():
    from cloudforger.data_generation.point_processes.kernels import AnisotropicGaussianKernel

    rng = np.random.default_rng(0)
    k = AnisotropicGaussianKernel(sigma_1=0.10, sigma_2=0.02, theta=0.0)
    x = k.sample(20_000, 2, rng)
    assert x.shape == (20_000, 2)
    assert abs(x[:, 0].std() - 0.10) < 0.01
    assert abs(x[:, 1].std() - 0.02) < 0.005

    # theta = pi/2 swaps which ambient axis carries the long spread
    xr = AnisotropicGaussianKernel(0.10, 0.02, theta=np.pi / 2).sample(20_000, 2, rng)
    assert abs(xr[:, 0].std() - 0.02) < 0.005
    assert abs(xr[:, 1].std() - 0.10) < 0.01

    with pytest.raises(ValueError):
        k.sample(10, 3, rng)


def test_matern_cluster_process_builds_and_samples():
    cloud = _sample_cloud("matern_cluster", parent_intensity=50.0, mean_offspring=8.0, cluster_radius=0.05)
    assert cloud.n_points >= 0
    assert cloud.points.shape[1] == 2


def test_matern_cluster_edge_buffer_is_exact_radius():
    # BallKernel.support_radius is compact -- no eps tail-mass margin, unlike
    # GaussianKernel's -- so the derived edge_buffer must equal cluster_radius exactly.
    proc = PROCESS_REGISTRY.build(
        "matern_cluster", parent_intensity=50.0, mean_offspring=8.0, cluster_radius=0.05
    )
    assert proc.params["edge_buffer"] == 0.05
    assert proc.params["cluster_radius"] == 0.05


def test_filtration_registry_and_path_tag():
    assert {"rips", "dtm"} <= set(FILTRATION_REGISTRY.names())
    dtm = FILTRATION_REGISTRY.build("dtm", k=5)
    assert dtm.path_tag() == "dtm_k5"
    rips = FILTRATION_REGISTRY.build("rips")
    assert rips.path_tag() == "rips"


def test_dtm_filtration_end_to_end():
    cloud = _sample_cloud("thomas", parent_intensity=50.0, mean_offspring=8.0, cluster_scale=0.03, edge_buffer=0.1)
    diagram = FILTRATION_REGISTRY.build("dtm", k=5).compute(cloud)
    assert set(diagram.dimensions()) <= {0, 1}


def test_feature_registry_betti_and_entropy():
    assert {"betti_curve", "persistence_entropy"} <= set(FEATURE_REGISTRY.names())
    cloud = _sample_cloud("thomas", parent_intensity=50.0, mean_offspring=8.0, cluster_scale=0.03, edge_buffer=0.1)
    diagram = FILTRATION_REGISTRY.build("dtm", k=5).compute(cloud)

    betti = FEATURE_REGISTRY.build("betti_curve", homology_dims=(0, 1), grid_size=64)
    bc = betti.compute(diagram)
    assert bc.vector().shape == (128,)

    entropy = FEATURE_REGISTRY.build("persistence_entropy", homology_dims=(0, 1))
    ent = entropy.compute(diagram)
    assert set(ent.keys()) == {0, 1}


def test_calibrated_imager():
    from cloudforger.vectorization.persistence_images.calibrated import build_calibrated_imager

    clouds = [_sample_cloud("thomas", parent_intensity=50.0, mean_offspring=8.0, cluster_scale=0.03, edge_buffer=0.1) for _ in range(3)]
    dtm = FILTRATION_REGISTRY.build("dtm", k=5)
    diagrams = [dtm.compute(c) for c in clouds]

    imager = build_calibrated_imager(diagrams, homology_dims=(0, 1), resolution=16, verbose=False)
    images = imager.transform(diagrams[0])
    assert images[0].shape == (16, 16)
    assert images[1].shape == (16, 16)
