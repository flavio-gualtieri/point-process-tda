# src/cloudforger/generation/seeding.py
"""Per-case random streams (generation.tex, "Random numbers").

Every case owns two streams, addressed -- not spawned in sequence -- by

    key = (DV, set_id, family_id, index, role)
    rng = Generator(PCG64DXSM(SeedSequence(ROOT, spawn_key=key)))

SeedSequence hashes (ROOT, key) into the generator state, so any case can be
rebuilt alone and in any order, neighbouring keys give independent streams,
and no two (set, family) pairs can share a stream. Role PARAMS draws theta,
role PATTERN drives the sampler: changing a sampler never changes theta.
"""

from __future__ import annotations

import numpy as np
from numpy.random import PCG64DXSM, Generator, SeedSequence

DV = 3

SET_ID = {"train": 0, "A": 1, "B": 2, "C": 3, "pilot": 8, "validation": 9}
FAMILY_ID = {"poisson": 0, "thomas": 1, "nested": 2, "matern2": 3, "strauss": 4, "lgcp": 5}

PARAMS = 0
PATTERN = 1

# Index layouts for the fixed-theta sets. The bounds keep the packed index
# injective: rep 10_000 of cell 3 would otherwise alias rep 0 of cell 4.
_REP_MAX = 10_000
_LEVEL_MAX = 100


def check_root(root: int) -> int:
    if isinstance(root, bool) or not isinstance(root, int) or not 0 <= root < 2**128:
        raise ValueError(f"root must be a 128-bit non-negative int, got {root!r}")
    return root


def spawn_key(set_: str, family: str, index: int, role: int) -> tuple[int, ...]:
    if set_ not in SET_ID:
        raise ValueError(f"unknown set {set_!r}; expected one of {sorted(SET_ID)}")
    if family not in FAMILY_ID:
        raise ValueError(f"unknown family {family!r}; expected one of {sorted(FAMILY_ID)}")
    if role not in (PARAMS, PATTERN):
        raise ValueError(f"role must be PARAMS (0) or PATTERN (1), got {role!r}")
    if int(index) != index or index < 0:
        raise ValueError(f"index must be a non-negative int, got {index!r}")
    return (DV, SET_ID[set_], FAMILY_ID[family], int(index), role)


def case_rng(root: int, set_: str, family: str, index: int, role: int) -> Generator:
    ss = SeedSequence(check_root(root), spawn_key=spawn_key(set_, family, index, role))
    return Generator(PCG64DXSM(ss))


def r_seed(root: int, set_: str, family: str, index: int, role: int = PATTERN) -> int:
    """Seed for R's set.seed (Strauss CFTP route): a positive int32 derived from the same key."""
    ss = SeedSequence(check_root(root), spawn_key=spawn_key(set_, family, index, role))
    return int(ss.generate_state(1, dtype=np.uint32)[0]) % (2**31 - 1) + 1


def index_B(cell: int, rep: int) -> int:
    if not 0 <= rep < _REP_MAX:
        raise ValueError(f"B rep must be in [0, {_REP_MAX}), got {rep}")
    return _REP_MAX * cell + rep


def index_C(ladder: int, level: int, rep: int) -> int:
    if not 0 <= rep < _REP_MAX:
        raise ValueError(f"C rep must be in [0, {_REP_MAX}), got {rep}")
    if not 0 <= level < _LEVEL_MAX:
        raise ValueError(f"C level must be in [0, {_LEVEL_MAX}), got {level}")
    return _REP_MAX * _LEVEL_MAX * ladder + _REP_MAX * level + rep


def key_str(set_: str, family: str, index: int) -> str:
    """The role-free part of the spawn key, as stored in plan/manifest rows."""
    return ":".join(str(k) for k in spawn_key(set_, family, index, PARAMS)[:4])


def case_id(set_: str, family: str, index: int) -> str:
    spawn_key(set_, family, index, PARAMS)  # validates
    return f"dv{DV}-{set_}-{family}-{index:06d}"


def parse_case_id(cid: str) -> tuple[str, str, int]:
    parts = cid.split("-")
    if len(parts) != 4 or parts[0] != f"dv{DV}":
        raise ValueError(f"not a DV{DV} case id: {cid!r}")
    _, set_, family, index = parts
    spawn_key(set_, family, int(index), PARAMS)  # validates
    return set_, family, int(index)
