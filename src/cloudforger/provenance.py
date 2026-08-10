# src/cloudforger/provenance.py
"""Per-run provenance stamping + a flat experiment ledger.

Shared by every place that writes results.pt/results.json --
cloudforger.experiments.common.save_results and baselines.vihrs.run_one_seed
(which duplicates that schema by hand instead of calling save_results) -- so
"what commit/config produced this?" is answerable from results.json alone
without loading a torch file, and "what have I run, and how did it do?" is
answerable from one results/experiments.jsonl file instead of walking the
results/ tree. See paths.py's RUN_ARCHIVE_DIR_NAME for the companion
mechanism that keeps an old run's directory around instead of overwriting it
in place."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .paths import DEFAULT_RESULTS_ROOT, PROJECT_ROOT

LEDGER_FILENAME = "experiments.jsonl"


def _git_head() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True,
        ).stdout.strip())
        return {"git_commit": commit, "git_dirty": dirty}
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return {"git_commit": None, "git_dirty": None}


def provenance_stamp(run_tag: str | None = None) -> dict[str, Any]:
    """Fields every saved result gets stamped with: the exact commit
    (git_dirty=True means the working tree had uncommitted changes on top of
    it, so that commit alone doesn't fully reproduce the run), when it ran,
    and which named archive (if any -- see paths.RUN_ARCHIVE_DIR_NAME) it
    belongs to."""
    stamp = _git_head()
    stamp["timestamp"] = datetime.now(timezone.utc).isoformat()
    stamp["run_tag"] = run_tag
    return stamp


def _process_and_filtration_tag(output_dir: Path, results_root: Path) -> tuple[str | None, str | None]:
    """Best-effort (process, filtration_tag) recovered from output_dir's
    position under results_root -- both are always parts[0]/parts[1] of that
    relative path regardless of run_tag (paths.ResultsPaths inserts the
    RUN_ARCHIVE_DIR_NAME/run_tag segment after the method, not before the
    process/filtration_tag). Returns (None, None) if output_dir isn't
    actually under results_root (e.g. a caller passed a mismatched root)."""
    try:
        parts = output_dir.resolve().relative_to(Path(results_root).resolve()).parts
    except ValueError:
        return None, None
    process = parts[0] if len(parts) > 0 else None
    filtration_tag = parts[1] if len(parts) > 1 else None
    return process, filtration_tag


def append_ledger_entry(
    *,
    output_dir: Path,
    results_root: Path | str | None,
    method: str,
    seed: int,
    test_loss: float | None,
    adversarial_loss: float | None = None,
    stamp: dict[str, Any] | None = None,
) -> None:
    """Append one row to <results_root>/experiments.jsonl. Append-only, so
    it's always the complete run history regardless of what results/
    currently looks like on disk -- archived, overwritten, or deleted
    directories don't erase their row here."""
    root = Path(results_root) if results_root else DEFAULT_RESULTS_ROOT
    output_dir = Path(output_dir)
    process, filtration_tag = _process_and_filtration_tag(output_dir, root)
    stamp = stamp if stamp is not None else provenance_stamp()
    entry = {
        **stamp,
        "process": process,
        "filtration_tag": filtration_tag,
        "method": method,
        "seed": seed,
        "test_loss": test_loss,
        "adversarial_loss": adversarial_loss,
        "output_dir": str(output_dir),
    }
    root.mkdir(parents=True, exist_ok=True)
    with open(root / LEDGER_FILENAME, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=str) + "\n")
