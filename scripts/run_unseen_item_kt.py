#!/usr/bin/env python3
"""Entry point for genuine unseen-item (5-fold item holdout) knowledge tracing.

Forwards all arguments to the underlying runner (GRU + SAKT, both backbones
share the same item folds, UNK-item representation, and item-ID dropout
policy). Use ``src/kt/run_unseen_item_kt_parallel.py`` to launch one process
per item fold.

Usage:
    python scripts/run_unseen_item_kt.py --dataset xes3g5m [--fold N] [--backbone GRU|SAKT]
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "kt"))

if __name__ == "__main__":
    target = ROOT / "src" / "kt" / "run_unseen_item_kt.py"
    sys.argv[0] = str(target)
    runpy.run_path(str(target), run_name="__main__")
