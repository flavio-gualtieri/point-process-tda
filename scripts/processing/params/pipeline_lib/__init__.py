# scripts/processing/params/pipeline_lib/__init__.py

from __future__ import annotations

import sys
from pathlib import Path

# scripts/processing/params/pipeline_lib/<this file> -> parents[4] == project root
PROJECT_ROOT = Path(__file__).resolve().parents[4]
_SRC = PROJECT_ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
