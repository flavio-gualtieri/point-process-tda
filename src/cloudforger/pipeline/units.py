"""The units a config defines and where each one lives. No numerical imports, so slurm/plan.py can
size the training arrays without paying numpy/pandas start-up on a slow filesystem.

A unit is one trained model: (classify, model, None) over every family, or (estimate, model, family)
for every family but poisson. It is CPU or GPU by its learner, and done once its report.json exists.
"""

from __future__ import annotations

from pathlib import Path

from ..paths import RESULTS

GPU_LEARNERS = {"nn"}


def run_dir(cfg: dict, *parts: str) -> Path:
    return RESULTS.joinpath(cfg["name"], *parts)


def unit_dir(cfg: dict, task: str, model: str, family: str | None = None) -> Path:
    """<results>/<run>/classify/<model>/ or .../estimate/<family>/<model>/."""
    return run_dir(cfg, task, *([family] if family else []), model)


def estimated_families(cfg: dict) -> list[str]:
    """Families with a learned estimator (everything but poisson)."""
    return [f for f in cfg["families"] if f != "poisson"]


def needs_gpu(cfg: dict, model: str) -> bool:
    return cfg["models"][model]["learner"] in GPU_LEARNERS


def units(cfg: dict, kind: str | None = None) -> list[tuple[str, str, str | None]]:
    """(task, model, family) in a fixed order -- the SLURM array order; kind filters by cpu | gpu."""
    out = [("classify", m, None) for m in cfg["classify"]]
    out += [("estimate", m, f) for m in cfg["estimate"] for f in estimated_families(cfg)]
    if kind:
        out = [u for u in out if needs_gpu(cfg, u[1]) == (kind == "gpu")]
    return out


def done(cfg: dict, task: str, model: str, family: str | None) -> bool:
    return (unit_dir(cfg, task, model, family) / "report.json").exists()
