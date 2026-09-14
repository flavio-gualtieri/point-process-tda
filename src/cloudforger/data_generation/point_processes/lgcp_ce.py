# src/cloudforger/data_generation/point_processes/lgcp_ce.py
"""Log-Gaussian Cox process on a grid, with the Gaussian field drawn exactly by
circulant embedding (Wood & Chan 1994; Dietrich & Newsam 1997).

The window (a square Box) is cut into M x M cells of side D = side / M. The
field Y ~ GP(0, sigma2 exp(-r / s)) is drawn exactly at the cell centres:
embed the node grid in a torus of (P M)^2 nodes, whose covariance matrix is
circulant and so diagonalised by the 2-D FFT; its eigenvalues are the FFT of
the covariance at wrapped node distances. If none is below -eig_tol times the
largest, scaling complex white noise by sqrt(eigenvalue / #nodes) and
transforming gives a field whose real part has exactly the target covariance
on the torus, hence on the embedded window. Otherwise P doubles. Padding to
P >= 2 keeps the wrap-around from correlating opposite edges of the window.

Then N_ij ~ Poisson(D^2 exp(mu + Y_ij)) points uniform in cell (i, j): exactly
the grid LGCP whose intensity is constant on each cell, with
E n = exp(mu + sigma2 / 2) |W| and no thinning bound. How close the grid
process is to the continuous one is the caller's choice of M
(generation/lgcp_grid.py).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.fft

from ...core.base import PointProcess
from ...core.region import Box, Region


def torus_eigenvalues(sigma2: float, s: float, M: int, P: int, side: float = 1.0) -> np.ndarray:
    """Eigenvalues of the (P M)^2 circulant covariance of exp(-r/s) on a torus
    with node spacing side / M."""
    N = P * M
    i = np.arange(N)
    d = (side / M) * np.minimum(i, N - i)
    c = sigma2 * np.exp(-np.hypot(d[:, None], d[None, :]) / s)
    return scipy.fft.fft2(c).real


class CirculantLGCPProcess(PointProcess):

    def __init__(
        self,
        mu: float,
        sigma2: float,
        s: float,
        grid_M: int,
        pad_P: int = 2,
        eig_tol: float = 1e-10,
        max_P: int = 16,
    ):
        if sigma2 <= 0 or s <= 0:
            raise ValueError("sigma2 and s must be positive")
        if grid_M < 2 or pad_P < 2:
            raise ValueError("need grid_M >= 2 and pad_P >= 2")
        self.mu = float(mu)
        self.sigma2 = float(sigma2)
        self.s = float(s)
        self.grid_M = int(grid_M)
        self.pad_P = int(pad_P)
        self.eig_tol = float(eig_tol)
        self.max_P = int(max_P)

    @property
    def name(self) -> str:
        return "lgcp"

    @property
    def params(self) -> dict[str, Any]:
        return {"mu": self.mu, "sigma2": self.sigma2, "s": self.s, "grid_M": self.grid_M}

    def embedding(self, side: float) -> tuple[np.ndarray, int, float]:
        """(eigenvalues, P, min/max eigenvalue ratio) for the smallest valid P.
        Deterministic: no random numbers are used before P is fixed."""
        P = self.pad_P
        while True:
            lam = torus_eigenvalues(self.sigma2, self.s, self.grid_M, P, side)
            ratio = float(lam.min() / lam.max())
            if ratio >= -self.eig_tol:
                return np.clip(lam, 0.0, None), P, ratio
            P *= 2
            if P > self.max_P:
                raise RuntimeError(f"circulant embedding not nonnegative up to P={self.max_P}")

    def _sample_points(self, n, region: Region, rng: np.random.Generator) -> np.ndarray:
        if not isinstance(region, Box) or region.dimension != 2:
            raise ValueError("CirculantLGCPProcess needs a 2-D Box region")
        side = float(region.high[0] - region.low[0])
        if not np.isclose(side, float(region.high[1] - region.low[1])):
            raise ValueError("CirculantLGCPProcess needs a square window")

        M = self.grid_M
        lam, P, ratio = self.embedding(side)
        self.last_pad_P, self.last_min_eig = P, ratio   # per-draw diagnostics (DV3 manifest)

        N = P * M
        z = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
        y = scipy.fft.fft2(np.sqrt(lam / N**2) * z).real[:M, :M]

        delta = side / M
        counts = rng.poisson(delta**2 * np.exp(self.mu + y))
        ij = np.argwhere(counts > 0)
        cells = np.repeat(ij, counts[counts > 0], axis=0)
        return region.low + (cells + rng.random((len(cells), 2))) * delta
