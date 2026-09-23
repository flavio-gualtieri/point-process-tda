"""Size of the alpha test on fresh CSR patterns at random, mostly off-grid, n -- one per reduction."""

from __future__ import annotations

import numpy as np

from .config import Config
from .simulate import validation_path
from .tables import REDUCTIONS, Tables

BANDS = 6


def _rate(reject: np.ndarray, alpha: float) -> dict:
    se = np.sqrt(alpha * (1 - alpha) / len(reject))
    rate = float(reject.mean())
    return {"patterns": int(len(reject)), "size": rate, "z": (rate - alpha) / se, "pass": bool(abs(rate - alpha) <= 3 * se)}


def size(cfg: Config, tables: Tables) -> dict:
    z = np.load(validation_path())
    n, curves = z["n"], z["curves"]
    edges = np.geomspace(cfg.n_low, cfg.n_high, BANDS + 1)
    band = np.clip(np.searchsorted(edges, n, side="right") - 1, 0, BANDS - 1)
    out = {}
    for name in REDUCTIONS:
        s = np.concatenate([tables.statistic(curves[i:i + 5000].astype(np.float64), n[i:i + 5000], name)
                            for i in range(0, len(n), 5000)])
        reject = s > 1
        out[name] = {
            "overall": _rate(reject, cfg.alpha),
            "by_n": [{"n": [int(np.ceil(edges[k])), int(edges[k + 1])], **_rate(reject[band == k], cfg.alpha)}
                     for k in range(BANDS)],
        }
    return out
