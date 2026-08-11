#!/usr/bin/env python3
"""Phase F2: synthetic alignment ladder on GSM8K-IRT."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "measurement"))
sys.path.insert(0, str(ROOT / "src" / "simulation"))

from measurement_common import (  # noqa: E402
    REPORT_DIR,
    TABLE_DIR,
    holm_correction,
    load_config,
    rank_independent,
    utc_now,
)
from synthetic_alignment_common import (  # noqa: E402
    build_surface_confounded_difficulty,
    generate_responses,
    load_gsm8k_items,
    normalize_z,
    rank_agreement,
    review_budget_metrics,
)


def _condition_difficulty(df: pd.DataFrame, cond: str, spec: dict, rng: np.random.Generator) -> np.ndarray:
    d_llm = df["d_llm"].values
    if spec.get("type") == "surface_confounded":
        return build_surface_confounded_difficulty(df, rng)
    d_ind = df["d_ind"].values
    w_llm = spec.get("d_llm_weight", 0.0)
    w_ind = spec.get("d_ind_weight", 0.0)
    d_gen = w_llm * d_llm + w_ind * d_ind
    return normalize_z(d_gen)


def run_ladder(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    syn = cfg["synthetic"]
    items = load_gsm8k_items(syn["item_subset"])
    d_llm = normalize_z(items["difficulty"].values)
    rng_master = np.random.default_rng(cfg["rasch"]["seed"])
    d_ind = rng_master.permutation(d_llm)
    # enforce rank independence
    if rank_independent(d_llm, d_ind) > 0.15:
        d_ind = rng_master.normal(size=len(d_llm))
        d_ind = normalize_z(d_ind)
    items = items.copy()
    items["d_llm"] = d_llm
    items["d_ind"] = d_ind

    cond_rows = []
    for code, spec in syn["conditions"].items():
        cond_rows.append({"condition": code, **spec})
    cond_df = pd.DataFrame(cond_rows)
    cond_df.to_csv(TABLE_DIR / "SYNTHETIC_ALIGNMENT_CONDITIONS.csv", index=False)

    validity_rows = []
    seed_rows = []
    utility_rows = []

    for seed in syn["seeds"]:
        rng = np.random.default_rng(seed)
        for code, spec in syn["conditions"].items():
            d_gen = _condition_difficulty(items, code, spec, rng)
            sim = generate_responses(items, d_gen, seed, syn)
            err = 1 - sim.groupby("item_id")["correct"].mean()
            err = err.reindex(items["item_id"]).values
            rho_llm_gen = float(stats.spearmanr(d_llm, d_gen).correlation)
            rho_llm_err = float(stats.spearmanr(d_llm, err).correlation)
            rho_gen_err = float(stats.spearmanr(d_gen, err).correlation)
            util = review_budget_metrics(d_llm, err)
            row = {
                "seed": seed,
                "condition": code,
                "rho_d_llm_d_gen": rho_llm_gen,
                "rho_d_llm_sim_error": rho_llm_err,
                "rho_d_gen_sim_error": rho_gen_err,
                "rank_agreement_llm_gen": rank_agreement(d_llm, d_gen),
                "mean_sim_error": float(np.mean(err)),
                **util,
            }
            validity_rows.append(row)
            seed_rows.append({**row, "n_students": syn["num_students"], "n_items": syn["item_subset"]})
            if code == "S5":
                utility_rows.append(row)

    validity_df = pd.DataFrame(validity_rows)
    seed_df = pd.DataFrame(seed_rows)
    utility_df = pd.DataFrame(utility_rows)

    trend_rows = []
    for metric in ["rho_d_llm_sim_error", "rho_d_llm_d_gen"]:
        sub = validity_df.groupby("condition")[metric].agg(["mean", "std"]).reset_index()
        order = ["S0", "S1", "S2", "S3", "S4", "S5"]
        sub["condition"] = pd.Categorical(sub["condition"], categories=order, ordered=True)
        sub = sub.sort_values("condition")
        x = np.arange(len(sub))
        slope, intercept, r, p, _ = stats.linregress(x[:5], sub["mean"].values[:5])
        trend_rows.append({
            "metric": metric,
            "linear_slope_S0_S4": float(slope),
            "linear_intercept": float(intercept),
            "linear_r": float(r),
            "linear_p": float(p),
            "isotonic_monotone_S0_S4": bool(np.all(np.diff(sub["mean"].values[:5]) <= 1e-9)),
        })
    trend_df = pd.DataFrame(trend_rows)
    pvals = trend_df["linear_p"].tolist()
    trend_df["linear_p_holm"] = holm_correction(pvals)
    return validity_df, seed_df, trend_df, utility_df


def build_claim_ledger(validity_df: pd.DataFrame, trend_df: pd.DataFrame) -> str:
    s0 = validity_df[validity_df["condition"] == "S0"]["rho_d_llm_sim_error"].mean()
    s4 = validity_df[validity_df["condition"] == "S4"]["rho_d_llm_sim_error"].mean()
    s5 = validity_df[validity_df["condition"] == "S5"]["rho_d_llm_sim_error"].mean()
    slope = trend_df[trend_df["metric"] == "rho_d_llm_sim_error"]["linear_slope_S0_S4"].iloc[0]
    lines = [
        "# Synthetic Alignment Claim Ledger",
        "",
        f"**Generated:** {utc_now()}",
        "",
        "| # | Claim | Status | Evidence |",
        "|---|-------|--------|----------|",
        f"| 1 | S0 circular alignment near-perfect validity | {'SUPPORTED' if s0 > 0.8 else 'PARTIALLY_SUPPORTED'} | mean rho={s0:.3f} |",
        f"| 2 | Validity declines as alignment weakens | {'SUPPORTED' if slope < 0 else 'NOT_SUPPORTED'} | slope={slope:.4f} |",
        f"| 3 | Independent latent removes most validity | {'SUPPORTED' if abs(s4) < 0.2 else 'PARTIALLY_SUPPORTED'} | S4 mean rho={s4:.3f} |",
        f"| 4 | Surface confounding preserves apparent validity | {'PARTIALLY_SUPPORTED' if abs(s5) > abs(s4) else 'NOT_SUPPORTED'} | S5 rho={s5:.3f} |",
        f"| 5 | Synthetic KT utility declines with alignment | NOT_TESTABLE | KT-light not frozen in protocol |",
        f"| 6 | Conference synthetic result partly design-aligned | SUPPORTED | S0 circular condition documented |",
    ]
    return "\n".join(lines)


def main() -> int:
    cfg = load_config()
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    validity_df, seed_df, trend_df, utility_df = run_ladder(cfg)
    validity_df.to_csv(TABLE_DIR / "SYNTHETIC_ALIGNMENT_VALIDITY.csv", index=False)
    seed_df.to_csv(TABLE_DIR / "SYNTHETIC_ALIGNMENT_SEED_SUMMARY.csv", index=False)
    trend_df.to_csv(TABLE_DIR / "SYNTHETIC_ALIGNMENT_TREND_TESTS.csv", index=False)
    utility_df.to_csv(TABLE_DIR / "SYNTHETIC_ALIGNMENT_DECISION_UTILITY.csv", index=False)

    report = [
        "# Synthetic Alignment Ladder Report",
        "",
        f"**Generated:** {utc_now()}",
        "",
        "Primary outcome: Spearman correlation between d_llm and simulated item error.",
        "",
        "## Condition means (rho d_llm vs simulated error)",
        "",
    ]
    means = validity_df.groupby("condition")["rho_d_llm_sim_error"].agg(["mean", "std"])
    for cond, r in means.iterrows():
        report.append(f"- **{cond}**: {r['mean']:.3f} ± {r['std']:.3f}")
    (REPORT_DIR / "SYNTHETIC_ALIGNMENT_LADDER_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    (REPORT_DIR / "SYNTHETIC_ALIGNMENT_CLAIM_LEDGER.md").write_text(
        build_claim_ledger(validity_df, trend_df), encoding="utf-8"
    )
    print("Synthetic alignment ladder complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
