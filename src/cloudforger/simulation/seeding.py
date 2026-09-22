"""One addressable random stream per (set, family, index, role):

    rng = Generator(PCG64DXSM(SeedSequence(root, spawn_key=(DV, set, family, index, role))))

Any stream can be rebuilt alone. PARAMS draws theta, PATTERN drives the sampler.
"""

from __future__ import annotations

from numpy.random import PCG64DXSM, Generator, SeedSequence

DV = 3
SET_ID = {"pilot": 8, "bank": 11}
FAMILY_ID = {"poisson": 0, "thomas": 1, "nested": 2, "matern2": 3, "lgcp": 5}
PARAMS, PATTERN = 0, 1


def case_rng(root: int, set_: str, family: str, index: int, role: int) -> Generator:
    if not 0 <= root < 2**128 or role not in (PARAMS, PATTERN) or index < 0:
        raise ValueError(f"bad stream address: root={root}, index={index}, role={role}")
    key = (DV, SET_ID[set_], FAMILY_ID[family], int(index), role)
    return Generator(PCG64DXSM(SeedSequence(root, spawn_key=key)))
