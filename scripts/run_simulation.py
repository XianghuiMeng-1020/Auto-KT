#!/usr/bin/env python3
"""Entry point for the controlled signal-decoupling simulation ladder.

Forwards all arguments to the underlying simulation runner.

Usage:
    python scripts/run_simulation.py [--seeds ...] [--conditions ...]
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "simulation"))

if __name__ == "__main__":
    sys.argv[0] = str(ROOT / "src" / "simulation" / "run_synthetic_alignment_ladder.py")
    runpy.run_path(sys.argv[0], run_name="__main__")
