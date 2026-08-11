#!/usr/bin/env python3
"""Plot LLM-score/character-length association against held-out learner error.

Plots item character-length association (orange squares) alongside held-out
learner-error correlation (blue circles, 95% CI) from the frozen result tables.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "results"
OUT = ROOT / "expansion_revision" / "figures" / "figure2_authentic_validity.pdf"

ORANGE = "#d55e00"
BLUE = "#0072b2"

# Row order (bottom to top), matching the published figure.
ROWS = [
    ("xes3g5m", "gpt-4o-mini", "XES3G5M / GPT-4o-mini"),
    ("xes3g5m", "gpt-5.4", "XES3G5M / GPT-5.4"),
    ("junyi", "gpt-4o-mini", "Junyi / GPT-4o-mini"),
    ("junyi", "gpt-5.4", "Junyi / GPT-5.4"),
]


def main() -> int:
    corr = pd.read_csv(TABLES / "AUTHENTIC_VALIDITY_CORRELATIONS.csv")
    corr = corr[corr["reference"] == "test_smoothed_error"]
    biv = pd.read_csv(TABLES / "RQ1_FEATURE_BIVARIATE.csv")
    biv = biv[biv["feature"] == "char_length"]

    err, err_lo, err_hi, char, labels = [], [], [], [], []
    for ds, mdl, lab in ROWS:
        c = corr[(corr["dataset"] == ds) & (corr["model"] == mdl)].iloc[0]
        b = biv[(biv["dataset"] == ds) & (biv["model"] == mdl)].iloc[0]
        # Held-out error correspondence is reported as absolute magnitude,
        # matching the published figure; CI bounds are mapped accordingly.
        rho = abs(c["spearman_rho"])
        lo, hi = sorted([abs(c["spearman_ci_lo"]), abs(c["spearman_ci_hi"])])
        err.append(rho)
        err_lo.append(rho - lo)
        err_hi.append(hi - rho)
        char.append(abs(b["spearman_rho"]))
        labels.append(lab)

    y = list(range(len(ROWS)))
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.axvline(0.0, color="0.55", lw=1.0, ls="--", zorder=0)
    for gx in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        ax.axvline(gx, color="0.9", lw=0.8, zorder=0)

    ax.errorbar(err, y, xerr=[err_lo, err_hi], fmt="o", color=BLUE,
                markersize=9, capsize=5, lw=1.6, zorder=3,
                label="Held-out learner error (95% CI)")
    ax.scatter(char, y, marker="s", s=90, color=ORANGE, zorder=4,
               label="Item character length (point estimate)")

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_ylim(-0.6, len(ROWS) - 0.4)
    ax.set_xlim(-0.02, 0.72)
    ax.set_xlabel("Spearman rank correlation (absolute magnitude)")
    ax.set_title("LLM difficulty estimates: weak criterion correspondence\n"
                 "vs. stronger visible-feature association")
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    # Legend in the empty upper-right region (no markers there).
    ax.legend(loc="upper right", frameon=False, fontsize=9,
              bbox_to_anchor=(1.0, 0.90))
    fig.tight_layout()
    fig.savefig(OUT)
    fig.savefig(OUT.with_suffix(".png"), dpi=200)
    plt.close(fig)
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
