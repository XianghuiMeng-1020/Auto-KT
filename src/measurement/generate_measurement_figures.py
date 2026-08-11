#!/usr/bin/env python3
"""Generate Phase F measurement validity figures."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "measurement"))

from measurement_common import FIGURE_DIR, TABLE_DIR  # noqa: E402

FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def fig_synthetic_ladder():
    df = pd.read_csv(TABLE_DIR / "SYNTHETIC_ALIGNMENT_VALIDITY.csv")
    agg = df.groupby("condition")["rho_d_llm_sim_error"].agg(["mean", "std"]).reindex(
        ["S0", "S1", "S2", "S3", "S4", "S5"]
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(agg))
    ax.errorbar(x, agg["mean"], yerr=agg["std"], fmt="o-", capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(agg.index)
    ax.set_xlabel("Synthetic alignment condition")
    ax.set_ylabel("Spearman rho (d_llm vs simulated error)")
    ax.set_title("Figure A: Synthetic Alignment Ladder")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig_synthetic_alignment_ladder.png", dpi=200)
    fig.savefig(FIGURE_DIR / "fig_synthetic_alignment_ladder.pdf")
    plt.close(fig)


def fig_authentic_validity():
    df = pd.read_csv(TABLE_DIR / "AUTHENTIC_VALIDITY_CORRELATIONS.csv")
    sub = df[df["reference"] == "test_smoothed_error"].copy()
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = [f"{r['dataset']}\n{r['model']}" for _, r in sub.iterrows()]
    y = np.arange(len(sub))
    ax.barh(y, sub["spearman_rho"], xerr=[
        sub["spearman_rho"] - sub["spearman_ci_lo"],
        sub["spearman_ci_hi"] - sub["spearman_rho"],
    ], capsize=3)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Spearman rho (LLM vs held-out smoothed error)")
    ax.set_title("Figure B: Authentic Construct Validity")
    ax.axvline(0, color="k", lw=0.5)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig_authentic_validity.png", dpi=200)
    fig.savefig(FIGURE_DIR / "fig_authentic_validity.pdf")
    plt.close(fig)


def fig_confound_comparison():
    df = pd.read_csv(TABLE_DIR / "AUTHENTIC_CONFOUND_DIAGNOSTICS.csv")
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(df))
    w = 0.35
    ax.bar(x - w / 2, df["authentic_spearman"], w, label="Authentic difficulty")
    ax.bar(x + w / 2, df["strongest_surface_spearman"].abs(), w, label="|Strongest surface|")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r['dataset']}\n{r['model']}" for _, r in df.iterrows()], fontsize=8)
    ax.set_ylabel("Spearman rho with LLM difficulty")
    ax.set_title("Figure C: Surface vs Authentic Association")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig_surface_vs_authentic.png", dpi=200)
    fig.savefig(FIGURE_DIR / "fig_surface_vs_authentic.pdf")
    plt.close(fig)


def fig_incremental_validity():
    df = pd.read_csv(TABLE_DIR / "AUTHENTIC_INCREMENTAL_VALIDITY.csv")
    sub = df[df["outcome"] == "test_smoothed_error"].copy()
    fig, ax = plt.subplots(figsize=(8, 4))
    for dataset in sub["dataset"].unique():
        d = sub[sub["dataset"] == dataset]
        ax.plot(d["model_spec"], d["oof_r2"], marker="o", label=dataset)
    ax.set_xticklabels(ax.get_xticks(), rotation=30, ha="right")
    ax.set_ylabel("Out-of-fold R²")
    ax.set_title("Figure D: Incremental Validity")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig_incremental_validity.png", dpi=200)
    fig.savefig(FIGURE_DIR / "fig_incremental_validity.pdf")
    plt.close(fig)


def fig_bucket_monotonicity():
    df = pd.read_csv(TABLE_DIR / "AUTHENTIC_BUCKET_ANALYSIS.csv")
    sub = df[df["scheme"] == "easy_medium_hard"]
    fig, ax = plt.subplots(figsize=(7, 4))
    for (dataset, model), g in sub.groupby(["dataset", "model"]):
        ax.plot(g["bucket"], g["mean_held_out_error"], marker="o", label=f"{dataset} {model}")
    ax.set_ylabel("Mean held-out error")
    ax.set_title("Figure E: Difficulty Bucket Monotonicity")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig_bucket_monotonicity.png", dpi=200)
    fig.savefig(FIGURE_DIR / "fig_bucket_monotonicity.pdf")
    plt.close(fig)


def main() -> int:
    fig_synthetic_ladder()
    fig_authentic_validity()
    fig_confound_comparison()
    fig_incremental_validity()
    fig_bucket_monotonicity()
    print(f"Figures written to {FIGURE_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
