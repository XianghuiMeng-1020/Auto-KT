#!/usr/bin/env python3
"""Entry point for response-limited knowledge-tracing training/evaluation.

Defaults to the GRU backbone (original protocol). Pass ``--backbone SAKT``
to run the SAKT backbone variant instead.

Usage:
    python scripts/run_response_limited_kt.py [--dataset ...] [--exposure ...] [--condition ...] [--seed ...]
    python scripts/run_response_limited_kt.py --backbone SAKT [...]
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    if "--backbone" in sys.argv:
        idx = sys.argv.index("--backbone")
        backbone = sys.argv[idx + 1]
        del sys.argv[idx : idx + 2]
        if backbone.upper() == "SAKT":
            target = ROOT / "src" / "kt" / "run_sakt_response_limited_kt.py"
        else:
            target = ROOT / "src" / "kt" / "run_limited_kt.py"
    else:
        target = ROOT / "src" / "kt" / "run_limited_kt.py"
    sys.path.insert(0, str(ROOT / "src" / "kt"))
    sys.argv[0] = str(target)
    runpy.run_path(str(target), run_name="__main__")
