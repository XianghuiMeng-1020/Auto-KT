#!/usr/bin/env python3
"""Entry point for the authentic learner-response analyses.

Runs, in order: Rasch estimation, held-out construct-validity analyses
(including orientation-corrected references), RQ1 visible-feature
association, and incremental-validity / sensitivity result tables.

Usage:
    python scripts/run_authentic_analysis.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "measurement"))


def main() -> int:
    import build_authentic_difficulty_references
    import run_authentic_construct_validity
    import build_orientation_corrected_references
    import run_rq1_feature_association
    import incremental_robustness_analysis
    import sensitivity_tables
    import kt_summary_tables
    import generate_measurement_figures
    import plot_authentic_validity_figure

    for mod, label in [
        (build_authentic_difficulty_references, "held-out difficulty references (incl. Rasch)"),
        (run_authentic_construct_validity, "authentic construct validity"),
        (build_orientation_corrected_references, "orientation-corrected references"),
        (run_rq1_feature_association, "RQ1 feature association"),
        (incremental_robustness_analysis, "incremental robustness"),
        (sensitivity_tables, "sensitivity tables"),
        (kt_summary_tables, "KT summary tables"),
        (generate_measurement_figures, "measurement figures"),
        (plot_authentic_validity_figure, "authentic validity figure"),
    ]:
        print(f"--- {label} ---")
        mod.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
