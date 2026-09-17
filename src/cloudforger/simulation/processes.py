"""Samplers on W = [0,1]^2: model parameters + rng -> (n, 2) points."""

from __future__ import annotations

import math

import numpy as np
import scipy.fft
from scipy.spatial import cKDTree

BUFFER = 5.0   # parents simulated on W expanded by 5 displacement s.d.s (missed mass exp(-12.5))


def _crop(x):
    return x[np.all((x >= 0) & (x <= 1), axis=1)]


def _uniform(intensity, b, rng):
    side = 1 + 2 * b
    return rng.random((rng.poisson(intensity * side**2), 2)) * side - b


def _offspring(parents, mean, sigma, rng):
    k = rng.poisson(mean, len(parents))
    return np.repeat(parents, k, axis=0) + rng.normal(0.0, sigma, (k.sum(), 2))


def poisson(rng, nbar):
    return rng.random((rng.poisson(nbar), 2))


def thomas(rng, kappa, mu, sigma):
    return _crop(_offspring(_uniform(kappa, BUFFER * sigma, rng), mu, sigma, rng))


def nested(rng, kappa, mu1, mu2, sigma1, sigma2):
    meta = _uniform(kappa, BUFFER * math.hypot(sigma1, sigma2), rng)
    return _crop(_offspring(_offspring(meta, mu1, sigma1, rng), mu2, sigma2, rng))


def matern2(rng, R, lam_p):
    x = _uniform(lam_p, R, rng)
    marks = rng.random(len(x))
    i, j = cKDTree(x).query_pairs(R, output_type="ndarray").T
    dead = np.zeros(len(x), bool)
    dead[np.where(marks[i] > marks[j], i, j)] = True
    return _crop(x[~dead])


def lgcp_eigenvalues(sigma2, s, M):
    """Eigenvalues of the exponential covariance on the (2M)^2 torus with spacing 1/M."""
    N = 2 * M
    d = (np.minimum(np.arange(N), N - np.arange(N)) / M).astype(np.float32)
    lam = scipy.fft.fft2(sigma2 * np.exp(-np.hypot(d[:, None], d[None, :]) / s)).real
    if lam.min() < -1e-6 * lam.max():
        raise ValueError(f"circulant embedding not nonnegative (sigma2={sigma2}, s={s}, M={M})")
    return np.sqrt(np.clip(lam, 0, None))


def lgcp(rng, mu_log, sigma2, s, M, root_lam=None):
    root_lam = lgcp_eigenvalues(sigma2, s, M) if root_lam is None else root_lam
    w = rng.standard_normal(root_lam.shape, dtype=np.float32)
    y = scipy.fft.ifft2(root_lam * scipy.fft.fft2(w)).real[:M, :M]
    counts = rng.poisson(np.exp(mu_log + y.astype(np.float64)) / M**2)
    cells = np.repeat(np.argwhere(counts > 0), counts[counts > 0], axis=0)
    return (cells + rng.random((len(cells), 2))) / M


SAMPLERS = {"poisson": poisson, "thomas": thomas, "nested": nested, "matern2": matern2, "lgcp": lgcp}
