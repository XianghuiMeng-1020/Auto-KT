#!/usr/bin/env python3
"""Aggregate raw run registries into the compact result tables under results/.

Run this after the response-limited and/or unseen-item KT experiments have
produced entries in runs/response_limited_kt/RUN_REGISTRY.csv,
runs/sakt_response_limited_kt/RUN_REGISTRY.csv, and/or
runs/unseen_item_kt/RUN_REGISTRY.csv.

Usage:
    python scripts/build_results.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "kt"))
sys.path.insert(0, str(ROOT / "src" / "llm_scoring"))


def main() -> int:
    import analyze_limited_kt
    import generate_limited_kt_figures
    import analyze_unseen_item_kt

    for mod, label in [
        (analyze_limited_kt, "response-limited KT result tables"),
        (generate_limited_kt_figures, "response-limited KT figures"),
        (analyze_unseen_item_kt, "genuine unseen-item KT result tables"),
    ]:
        print(f"--- {label} ---")
        try:
            mod.main()
        except FileNotFoundError as exc:
            print(f"  skipped ({exc})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
