#!/usr/bin/env python3
"""Generate limited KT figures."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "figures" / "kt"
TABLE_DIR = ROOT / "results"


def main() -> int:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    trends = pd.read_csv(TABLE_DIR / "LIMITED_KT_EXPOSURE_TRENDS.csv")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, ds in zip(axes, ["xes3g5m", "junyi"]):
        sub = trends[trends["dataset"] == ds]
        for cond in sub["condition"].unique():
            c = sub[sub["condition"] == cond]
            ax.plot(c["exposure"].astype(str), c["log_loss_mean"], marker="o", label=cond)
        ax.set_title(ds)
        ax.set_xlabel("Exposure level")
        ax.set_ylabel("Log loss (mean)")
        ax.tick_params(axis="x", rotation=45)
    axes[0].legend(fontsize=6, loc="best")
    fig.suptitle("Figure 1: Predictive performance vs exposure")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_limited_kt_predictive.png", dpi=200)
    fig.savefig(FIG_DIR / "fig_limited_kt_predictive.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, ds in zip(axes, ["xes3g5m", "junyi"]):
        sub = trends[trends["dataset"] == ds]
        for cond in sub["condition"].unique():
            c = sub[sub["condition"] == cond]
            ax.plot(c["exposure"].astype(str), c["brier_mean"], marker="o", label=cond)
        ax.set_title(ds)
        ax.set_xlabel("Exposure level")
        ax.set_ylabel("Brier (mean)")
    fig.suptitle("Figure 2: Calibration vs exposure")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_limited_kt_calibration.png", dpi=200)
    fig.savefig(FIG_DIR / "fig_limited_kt_calibration.pdf")
    plt.close(fig)

    print(f"Figures -> {FIG_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
