"""Standalone JEPO training package."""

from __future__ import annotations

import sys
from pathlib import Path

_JEPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_VERL_ROOT = Path("/project/peilab/qjl/2026/wmrl/verl")
for _path in (_JEPO_ROOT, _JEPO_ROOT / "verl", _DEFAULT_VERL_ROOT):
    if _path.exists() and str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
