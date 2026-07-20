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

from pathlib import Path

from .filtration import Filtration

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


def combined_filtration_tag(filtrations: list[Filtration] | None) -> str:
    """Path segment for zero, one, or several Filtration instances (more
    than one for multi-k sweeps like pi_multik). Each filtration contributes
    its own path_tag() (e.g. DTM's "dtm_k5"); several are joined with "-"."""
    if not filtrations:
        return RAW_TAG
    return "-".join(f.path_tag() for f in filtrations)


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

    def feature(self, filtrations: list[Filtration] | None, feature_name: str, adversarial: bool = False) -> Path:
        prefix = "adversarial_" if adversarial else ""
        return self.filtration_dir(filtrations) / f"{prefix}{feature_name}.pkl"


class ResultsPaths:
    """results/<process>/<filtration_tag>/<method>/seed_<seed>/... layout."""

    def __init__(self, process: str, root: Path | str = DEFAULT_RESULTS_ROOT):
        self.process = process
        self.root = Path(root)

    def method_dir(self, filtrations: list[Filtration] | None, method: str) -> Path:
        return self.root / self.process / combined_filtration_tag(filtrations) / method

    def seed_dir(self, filtrations: list[Filtration] | None, method: str, seed: int) -> Path:
        return self.method_dir(filtrations, method) / f"seed_{seed}"

    def compare_dir(self, filtrations: list[Filtration] | None, name: str) -> Path:
        return self.root / self.process / combined_filtration_tag(filtrations) / COMPARE_DIR_NAME / name


def is_done(seed_dir: Path) -> bool:
    """The one skip-if-exists resumability check every method/seed uses,
    matching the convention dtm_experiment's train scripts already followed
    (results.pt existing means that (method, seed) is complete)."""
    return (Path(seed_dir) / "results.pt").exists()
