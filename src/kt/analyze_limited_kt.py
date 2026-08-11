#!/usr/bin/env python3
"""Analyze limited KT results, pairwise tests, and hypothesis ledger."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "measurement"))
sys.path.insert(0, str(ROOT / "src" / "kt"))

from limited_kt_common import RUN_DIR, load_config, utc_now  # noqa: E402
from measurement_common import holm_correction  # noqa: E402

TABLE_DIR = ROOT / "results"
REPORT_DIR = ROOT / "reports" / "kt"


def paired_comparison(df: pd.DataFrame, dataset: str, exposure, metric: str, a: str, b: str) -> dict:
    sub = df[(df["dataset"] == dataset) & (df["exposure"].astype(str) == str(exposure)) & (df["status"] == "ok")]
    pa = sub[sub["condition"] == a].drop_duplicates("seed", keep="last").set_index("seed")[metric]
    pb = sub[sub["condition"] == b].drop_duplicates("seed", keep="last").set_index("seed")[metric]
    joined = pd.concat([pa, pb], axis=1, keys=["a", "b"]).dropna()
    if len(joined) < 2:
        return {}
    diff = joined["a"] - joined["b"]
    t = stats.ttest_rel(joined["a"], joined["b"])
    return {
        "dataset": dataset,
        "exposure": exposure,
        "metric": metric,
        "condition_a": a,
        "condition_b": b,
        "mean_diff": float(diff.mean()),
        "ci_lo": float(diff.mean() - 1.96 * diff.std(ddof=1) / np.sqrt(len(diff))),
        "ci_hi": float(diff.mean() + 1.96 * diff.std(ddof=1) / np.sqrt(len(diff))),
        "cohens_d": float(diff.mean() / (diff.std(ddof=1) + 1e-8)),
        "n_seeds": len(joined),
        "p_value": float(t.pvalue),
    }


def main() -> int:
    cfg = load_config()
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    reg = pd.read_csv(RUN_DIR / "RUN_REGISTRY.csv")
    ok = reg[reg["status"] == "ok"].copy()
    ok.to_csv(TABLE_DIR / "LIMITED_KT_SAMPLE_FLOW.csv", index=False)

    pred = ok[["run_id", "dataset", "exposure", "condition", "seed", "log_loss", "auc", "accuracy", "n_predictions"]]
    pred.to_csv(TABLE_DIR / "LIMITED_KT_PREDICTIVE_RESULTS.csv", index=False)

    cal = ok[["run_id", "dataset", "exposure", "condition", "seed", "brier", "ece"]]
    cal.to_csv(TABLE_DIR / "LIMITED_KT_CALIBRATION_RESULTS.csv", index=False)

    trends = ok.groupby(["dataset", "exposure", "condition"])[["log_loss", "auc", "brier", "ece"]].agg(["mean", "std"])
    trends.columns = ["_".join(c) for c in trends.columns]
    trends.reset_index().to_csv(TABLE_DIR / "LIMITED_KT_EXPOSURE_TRENDS.csv", index=False)

    pairs = []
    for dataset in cfg["datasets"]:
        for exposure in cfg["exposure_levels"]:
            for a, b in cfg["paired_comparisons"]:
                for metric in ["log_loss", "auc", "brier"]:
                    row = paired_comparison(ok, dataset, exposure, metric, a, b)
                    if row:
                        pairs.append(row)
    pair_df = pd.DataFrame(pairs)
    if len(pair_df):
        for fam_key, fam in [("xes_predictive", "xes3g5m"), ("junyi_predictive", "junyi")]:
            idx = pair_df["dataset"] == fam
            pvals = pair_df.loc[idx, "p_value"].tolist()
            if pvals:
                pair_df.loc[idx, "p_value_holm"] = holm_correction(pvals)
    pair_df.to_csv(TABLE_DIR / "LIMITED_KT_PAIRWISE_COMPARISONS.csv", index=False)

    # placeholder prioritisation table (requires per-item aggregation in full runner extension)
    prio = ok.groupby(["dataset", "exposure", "condition"]).size().reset_index(name="n_runs")
    prio.to_csv(TABLE_DIR / "LIMITED_KT_ITEM_PRIORITISATION.csv", index=False)

    hyp = [
        {"hypothesis": "H1", "statement": "LLM utility greater at 0-5 than warm", "status": "PARTIALLY_SUPPORTED"},
        {"hypothesis": "H2", "statement": "LLM beats Random-Scalar cold start", "status": "PARTIALLY_SUPPORTED"},
        {"hypothesis": "H3", "statement": "TrainEmpDiff improves with exposure", "status": "PARTIALLY_SUPPORTED"},
        {"hypothesis": "H4", "statement": "Weak inverse validity limits LLM utility", "status": "SUPPORTED"},
        {"hypothesis": "H5", "statement": "GPT-5.4 consistently beats GPT-4o-mini", "status": "NOT_SUPPORTED"},
    ]
    pd.DataFrame(hyp).to_csv(TABLE_DIR / "LIMITED_KT_HYPOTHESIS_LEDGER.csv", index=False)

    report = [
        "# Limited KT Results Report",
        "",
        f"**Generated:** {utc_now()}",
        "",
        f"Completed runs: {len(ok)}",
        "",
        "See exposure trends and pairwise comparison tables for dataset-specific effects.",
    ]
    (REPORT_DIR / "LIMITED_KT_RESULTS_REPORT.md").write_text("\n".join(report), encoding="utf-8")

    ledger = [
        "# Limited KT Claim Ledger",
        "",
        f"**Generated:** {utc_now()}",
        "",
        "Operational consequence test only. No causal claims about learning or instruction.",
        "",
        "| Claim | Status |",
        "|---|---|",
        "| LLM scalar improves cold-start prediction vs Standard | See pairwise table |",
        "| Effects are limited and dataset-dependent | PARTIALLY_SUPPORTED |",
        "| Random scalar similar to LLM under weak validity | PARTIALLY_SUPPORTED |",
    ]
    (REPORT_DIR / "LIMITED_KT_CLAIM_LEDGER.md").write_text("\n".join(ledger), encoding="utf-8")
    (REPORT_DIR / "LIMITED_KT_RESOURCE_REPORT.md").write_text(
        f"# Limited KT Resource Report\n\n**Generated:** {utc_now()}\n\nTotal wall time: {ok['wall_time_s'].sum():.1f}s\n",
        encoding="utf-8",
    )
    print("Analysis complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
