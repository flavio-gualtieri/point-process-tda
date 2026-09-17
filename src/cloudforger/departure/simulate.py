"""Binomial (CSR given n) patterns in W and their L - r curves, stored per n."""

from __future__ import annotations

import os
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from ..classical.lfunction import l_minus_r
from ..simulation.seeding import PARAMS, PATTERN, case_rng
from .config import DATA, Config

_REP_MAX = 100_000   # grid index = n * _REP_MAX + rep; validation uses indices below _REP_MAX


def grid_curve(root: int, n: int, rep: int) -> np.ndarray:
    rng = case_rng(root, "pilot", "poisson", n * _REP_MAX + rep, PATTERN)
    return l_minus_r(rng.random((n, 2)))


def validation_n(root: int, i: int, low: int, high: int) -> int:
    u = case_rng(root, "pilot", "poisson", i, PARAMS).random()
    return int(round(low * (high / low) ** u))


def validation_curve(root: int, i: int, low: int, high: int) -> np.ndarray:
    n = validation_n(root, i, low, high)
    return l_minus_r(case_rng(root, "pilot", "poisson", i, PATTERN).random((n, 2)))


def curves_path(n: int) -> Path:
    return DATA / "curves" / f"n_{n:05d}.npy"


def validation_path() -> Path:
    return DATA / "validation.npz"


def _save(path: Path, write) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        write(f)
    os.replace(tmp, path)


def _grid_task(args):
    return grid_curve(*args)


def _validation_task(args):
    return validation_curve(*args)


def simulate_n(cfg: Config, n: int, jobs: int = 1) -> Path:
    path = curves_path(n)
    if not path.exists():
        tasks = [(cfg.root, n, rep) for rep in range(cfg.reps)]
        curves = np.stack(_map(_grid_task, tasks, jobs)).astype(np.float32)
        _save(path, lambda f: np.save(f, curves))
    return path


def simulate_validation(cfg: Config, jobs: int = 1) -> Path:
    path = validation_path()
    if not path.exists():
        n = np.array([validation_n(cfg.root, i, cfg.n_low, cfg.n_high) for i in range(cfg.validation)])
        tasks = [(cfg.root, i, cfg.n_low, cfg.n_high) for i in range(cfg.validation)]
        curves = np.stack(_map(_validation_task, tasks, jobs)).astype(np.float32)
        _save(path, lambda f: np.savez(f, n=n, curves=curves))
    return path


def _map(f, tasks, jobs):
    if jobs == 1:
        return [f(t) for t in tasks]
    with get_context("spawn").Pool(jobs) as pool:
        return pool.map(f, tasks, chunksize=max(1, len(tasks) // (8 * jobs)))
