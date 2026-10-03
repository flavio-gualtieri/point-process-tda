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


def ring(rng, kappa, mu, rho, sigma):
    parents = _uniform(kappa, rho + BUFFER * sigma, rng)
    k = rng.poisson(mu, len(parents))
    phi = rng.uniform(0.0, 2 * math.pi, k.sum())
    circle = rho * np.column_stack([np.cos(phi), np.sin(phi)])
    return _crop(np.repeat(parents, k, axis=0) + circle + rng.normal(0.0, sigma, (k.sum(), 2)))


def matern1(rng, R, lam_p):
    x = _uniform(lam_p, R, rng)
    i, j = cKDTree(x).query_pairs(R, output_type="ndarray").T
    dead = np.zeros(len(x), bool)
    dead[i] = dead[j] = True
    return _crop(x[~dead])


def cell(rng, nbar, k):
    """Cells of side c = nbar^-1/2 on a uniformly shifted grid covering W, each with 0 | 1 | k points."""
    k = int(k)                                                   # a count, even when read from a float column
    c = 1 / math.sqrt(nbar)
    m = math.ceil(1 / c) + 1
    corners = np.stack(np.meshgrid(np.arange(m), np.arange(m), indexing="ij"), -1).reshape(-1, 2) * c - rng.random(2) * c
    counts = rng.choice([0, 1, k], size=len(corners), p=[1 / k, 1 - 1 / (k - 1), 1 / (k * (k - 1))])
    return _crop(np.repeat(corners, counts, axis=0) + rng.random((counts.sum(), 2)) * c)


def strauss_sweeps(q, R, n):
    """Metropolis sweeps before the state is returned: 200, growing to 1000 for a hard core at
    packing 0.5, where points can only shuffle into place (tests/test_simulation.py checks that
    doubling this changes nothing)."""
    return math.ceil(200 + 1600 * q * math.pi * R**2 * n / 4)


def strauss(rng, nbar, q, R, sweeps=None):
    """Strauss given n = nbar points on the unit torus: density proportional to (1 - q)^(pairs closer
    than R). Metropolis from a binomial start, one point per step, half the proposals uniform on the
    torus and half within R of the point. Pair counts come from a grid of cells no smaller than R."""
    n, gamma = int(round(nbar)), 1.0 - q
    m = int(min(1 / R, math.sqrt(n)))                            # cells per side, side >= R
    if m < 3:
        raise ValueError(f"strauss: R={R} too large for n={n} (the 3x3 cell neighbourhood would wrap)")
    sweeps = strauss_sweeps(q, R, n) if sweeps is None else sweeps
    xs, ys = rng.random(n).tolist(), rng.random(n).tolist()
    around = [[((a + da) % m) * m + (b + db) % m for da in (-1, 0, 1) for db in (-1, 0, 1)]
              for a in range(m) for b in range(m)]
    home = [int(x * m) % m * m + int(y * m) % m for x, y in zip(xs, ys)]
    cells = [[] for _ in range(m * m)]
    for i, c in enumerate(home):
        cells[c].append(i)
    R2 = R * R

    def close(px, py, c, skip):
        k = 0
        for cc in around[c]:
            for j in cells[cc]:
                if j != skip:
                    dx, dy = abs(px - xs[j]), abs(py - ys[j])
                    if dx > 0.5:
                        dx = 1.0 - dx
                    if dy > 0.5:
                        dy = 1.0 - dy
                    if dx * dx + dy * dy < R2:
                        k += 1
        return k

    for _ in range(sweeps):
        who = rng.integers(0, n, n).tolist()
        for i, (kind, u, v, a) in zip(who, rng.random((n, 4)).tolist()):
            if kind < 0.5:
                px, py = u, v
            else:
                px, py = (xs[i] + (u - 0.5) * 2 * R) % 1.0, (ys[i] + (v - 0.5) * 2 * R) % 1.0
            c = int(px * m) % m * m + int(py * m) % m
            d = close(px, py, c, i) - close(xs[i], ys[i], home[i], i)
            if d <= 0 or a < gamma**d:
                xs[i], ys[i] = px, py
                if c != home[i]:
                    cells[home[i]].remove(i)
                    cells[c].append(i)
                    home[i] = c
    return np.column_stack([xs, ys])


SAMPLERS = {"poisson": poisson, "thomas": thomas, "nested": nested, "matern2": matern2, "lgcp": lgcp,
            "ring": ring, "matern1": matern1, "cell": cell, "strauss": strauss}
