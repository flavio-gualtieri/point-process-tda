# src/cloudforger/provenance.py
"""Per-run provenance stamping.

Every run.json written by scripts/train.py carries one of these stamps, so
"what commit produced this?" is answerable from run.json alone without
loading a torch file."""

from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]


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
    it, so that commit alone doesn't fully reproduce the run), and when it
    ran."""
    stamp = _git_head()
    stamp["timestamp"] = datetime.now(timezone.utc).isoformat()
    stamp["run_tag"] = run_tag
    return stamp
