# src/cloudforger/paths.py
"""One place deriving every data/results path from a process/filtration/
method description -- deterministic, so results directories can never drift
out of sync with the code that produces them (unlike
dtm_experiment/train_k5.py's hand-maintained output_dir_for_variant, which
referenced a fusion_model.py shape that had already changed underneath it --
see the refactor plan's Context for the exact bug that motivated this).

Dimension is deliberately absent from every path: this project is 2D-only
for now (see cloudforger.config.RunConfig)."""

from __future__ import annotations

import re
from pathlib import Path

from .data_generation.filtration import Filtration

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "results"

# Path segment for methods with no filtration dependency (raw point cloud,
# vihrs) -- always an explicit segment, never conditionally omitted. This is
# the structural fix for the class of path-drift bug in the Context above.
RAW_TAG = "raw"

# Leading-underscore reserved sibling of every real method subdir, so
# evaluate.py's aggregate output can never collide with a method name.
COMPARE_DIR_NAME = "_compare"

# Leading-underscore reserved child of every real method dir, holding named
# archives of that method's results (see ResultsPaths.method_dir's run_tag).
# Same collision-avoidance reasoning as COMPARE_DIR_NAME: no method or run
# tag can ever be named "_runs".
RUN_ARCHIVE_DIR_NAME = "_runs"


class ExplicitTag:
    """Minimal Filtration stand-in (only .path_tag() is required by
    combined_filtration_tag()/ResultsPaths) for methods whose diagrams
    weren't computed via the Filtration registry -- e.g. topo_superset's
    mass-fraction DTM sweep (scripts/precompute_topo_superset.py), which
    needs a per-cloud k = round(m*N) the registry's fixed-k contract has no
    hook for, so it bypasses DataPaths/the registry entirely."""

    def __init__(self, tag: str):
        self._tag = tag

    def path_tag(self) -> str:
        return self._tag


# Splits a path_tag() into (family_prefix, trailing_numeric_value), e.g.
# "dtm_m0.01" -> ("dtm_m", "0.01"), "dtm_k10" -> ("dtm_k", "10"). Tags with
# no trailing number (e.g. "raw") don't match and stay ungrouped.
_TAG_FAMILY_RE = re.compile(r"^(.*?)(\d[\d.]*)$")


def combined_filtration_tag(filtrations: list[Filtration] | None) -> str:
    """Path segment for zero, one, or several Filtration instances (more
    than one for multi-channel sweeps like pi_multik/pi_multik_scaleconv).
    Each filtration contributes its own path_tag() (e.g. DTM's "dtm_k5" or
    a mass-fraction sweep's "dtm_m0.01"). Consecutive tags sharing the same
    family prefix are fused into one segment with their values joined by
    "+" (e.g. "dtm_k5", "dtm_k10", "dtm_k15" -> "dtm_k5+10+15") instead of
    repeating the prefix per value -- the old "-".join() blew up into
    unreadable names like "dtm_m0.01-dtm_m0.02-dtm_m0.04-...-dtm_m0.90" for
    a 7-channel combo. Value order is preserved (it's semantically
    meaningful for e.g. ScaleConvFusion's ordered-channel-axis convolution),
    never sorted. Distinct families (or tags with no trailing number) are
    joined with "-", as before."""
    if not filtrations:
        return RAW_TAG
    groups: list[tuple[str, list[str]]] = []
    for f in filtrations:
        tag = f.path_tag()
        match = _TAG_FAMILY_RE.match(tag)
        prefix, value = match.groups() if match else (None, tag)
        if groups and groups[-1][0] == prefix and prefix is not None:
            groups[-1][1].append(value)
        else:
            groups.append((prefix, [value]))
    return "-".join((prefix or "") + "+".join(values) for prefix, values in groups)


class DataPaths:
    """data/<process>/... layout."""

    def __init__(self, process: str, root: Path | str = DEFAULT_DATA_ROOT):
        self.process = process
        self.root = Path(root)

    @property
    def process_dir(self) -> Path:
        return self.root / self.process

    def clouds(self, adversarial: bool = False) -> Path:
        name = "adversarial_clouds.pkl" if adversarial else "clouds.pkl"
        return self.process_dir / name

    def manifest(self) -> Path:
        return self.process_dir / "cloud_generation_manifest.yaml"

    def filtration_dir(self, filtrations: list[Filtration] | None) -> Path:
        return self.process_dir / combined_filtration_tag(filtrations)

    def diagrams(self, filtrations: list[Filtration] | None, adversarial: bool = False) -> Path:
        prefix = "adversarial_" if adversarial else ""
        return self.filtration_dir(filtrations) / f"{prefix}diagrams.pkl"

    def signed_measures(self, bifiltrations, adversarial: bool = False) -> Path:
        prefix = "adversarial_" if adversarial else ""
        return self.filtration_dir(bifiltrations) / f"{prefix}signed_measures.pkl"

    def feature(self, filtrations: list[Filtration] | None, feature_name: str, adversarial: bool = False) -> Path:
        prefix = "adversarial_" if adversarial else ""
        return self.filtration_dir(filtrations) / f"{prefix}{feature_name}.pkl"


class ResultsPaths:
    """results/<process>/<filtration_tag>/<method>/seed_<seed>/... layout,
    or results/<process>/<filtration_tag>/<method>/_runs/<run_tag>/seed_<seed>/...
    when run_tag is given -- a named, non-overwritten archive of that
    method's results living alongside the current (untagged) ones, so
    comparing "old vs new" after a code change never requires manually
    copying a whole results tree aside (see scripts/archive_run.py, which
    moves a method's current results into one of these slots)."""

    def __init__(self, process: str, root: Path | str = DEFAULT_RESULTS_ROOT):
        self.process = process
        self.root = Path(root)

    def method_dir(self, filtrations: list[Filtration] | None, method: str, run_tag: str | None = None) -> Path:
        base = self.root / self.process / combined_filtration_tag(filtrations) / method
        return base / RUN_ARCHIVE_DIR_NAME / run_tag if run_tag else base

    def seed_dir(
        self, filtrations: list[Filtration] | None, method: str, seed: int, run_tag: str | None = None
    ) -> Path:
        return self.method_dir(filtrations, method, run_tag) / f"seed_{seed}"

    def compare_dir(self, filtrations: list[Filtration] | None, name: str) -> Path:
        return self.root / self.process / combined_filtration_tag(filtrations) / COMPARE_DIR_NAME / name


def is_done(seed_dir: Path) -> bool:
    """The one skip-if-exists resumability check every method/seed uses,
    matching the convention dtm_experiment's train scripts already followed
    (results.pt existing means that (method, seed) is complete)."""
    return (Path(seed_dir) / "results.pt").exists()
