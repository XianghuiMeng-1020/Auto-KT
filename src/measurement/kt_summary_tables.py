#!/usr/bin/env python3
"""Compact knowledge-tracing summary tables (excludes exposure=0 and pilot runs)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data_processed"
TABLE = ROOT / "results"
RUN_REGISTRY = ROOT / "runs" / "response_limited_kt" / "RUN_REGISTRY.csv"

DATASETS = ["xes3g5m", "junyi"]
CONDITIONS = ["LLM-Mini", "LLM-5.4", "Random-Scalar", "CharacterLength", "TrainEmpDiff"]
EXPOSURES_FOR_MAIN = ["1", "5", "20", "warm"]


def kt_summary() -> pd.DataFrame:
    reg = pd.read_csv(RUN_REGISTRY)
    ok = reg[
        (reg["status"] == "ok")
        & (~reg["run_id"].astype(str).str.startswith("pilot_"))
        & (reg["exposure"].astype(str) != "0")
    ].copy()
    ok["exposure"] = ok["exposure"].astype(str)
    rows = []
    for dataset in DATASETS:
        for condition in CONDITIONS:
            for exposure in EXPOSURES_FOR_MAIN:
                sub = ok[(ok["dataset"] == dataset) & (ok["condition"] == condition) & (ok["exposure"] == exposure)]
                std = ok[(ok["dataset"] == dataset) & (ok["condition"] == "Standard") & (ok["exposure"] == exposure)]
                joined = sub.set_index("seed")["log_loss"].to_frame("condition").join(
                    std.set_index("seed")["log_loss"].to_frame("standard"), how="inner"
                )
                if joined.empty:
                    continue
                diff = joined["condition"] - joined["standard"]
                rows.append({
                    "dataset": dataset,
                    "condition": condition,
                    "exposure": exposure,
                    "n_seeds": int(len(joined)),
                    "mean_log_loss": float(joined["condition"].mean()),
                    "mean_diff_vs_standard": float(diff.mean()),
                })
    out = pd.DataFrame(rows)
    out.to_csv(TABLE / "KT_MAIN_NO_EXPOSURE0.csv", index=False)
    return out


def first_attempt_counts() -> pd.DataFrame:
    rows = []
    for dataset in DATASETS:
        items = pd.read_parquet(PROCESSED / dataset / "items.parquet", columns=["item_id_hash", "eligible_for_llm_scoring"])
        scoreable = set(items.loc[items["eligible_for_llm_scoring"], "item_id_hash"])
        ix = pd.read_parquet(
            PROCESSED / dataset / "interactions.parquet",
            columns=["student_id_hash", "item_id_hash", "split_assignment", "first_attempt", "sequence_index"],
        )
        ix = ix[(ix["split_assignment"] == "test") & (ix["item_id_hash"].isin(scoreable))].copy()
        first_flag = ix[ix["first_attempt"].fillna(True)].copy()
        first_observed = ix.sort_values("sequence_index").drop_duplicates(["student_id_hash", "item_id_hash"], keep="first")
        rows.append({
            "dataset": dataset,
            "test_interactions_before_filter": int(len(ix)),
            "test_interactions_after_first_attempt_flag_filter": int(len(first_flag)),
            "test_interactions_after_first_observed_pair_filter": int(len(first_observed)),
            "unique_learner_item_pairs": int(ix[["student_id_hash", "item_id_hash"]].drop_duplicates().shape[0]),
            "flag_retention_rate": float(len(first_flag) / len(ix)) if len(ix) else 0.0,
            "first_observed_retention_rate": float(len(first_observed) / len(ix)) if len(ix) else 0.0,
        })
    out = pd.DataFrame(rows)
    out.to_csv(TABLE / "FIRST_ATTEMPT_FILTER_SUMMARY.csv", index=False)
    return out


def main() -> int:
    kt = kt_summary()
    first = first_attempt_counts()
    manifest = {
        "status": "KT_SUMMARY_TABLES_READY",
        "kt_rows": len(kt),
        "first_attempt_rows": len(first),
        "exposure0_excluded": True,
        "pilot_rows_excluded": True,
        "outputs": [
            "tables/KT_MAIN_NO_EXPOSURE0.csv",
            "tables/FIRST_ATTEMPT_FILTER_SUMMARY.csv",
        ],
    }
    (ROOT / "data_manifests" / "kt_summary_tables_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
