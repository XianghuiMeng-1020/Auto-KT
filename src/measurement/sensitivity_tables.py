#!/usr/bin/env python3
"""Criterion and surface-feature sensitivity tables.

These tables are descriptive add-ons and do not overwrite the primary analysis tables.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
TABLE = ROOT / "results"
PROCESSED = ROOT / "data_processed"
LLM_FEATURES = ROOT / "artifacts" / "scores" / "llm_item_scores.parquet"

DATASETS = ["xes3g5m", "junyi"]
MODELS = ["gpt-4o-mini", "gpt-5.4"]
PRIMARY_THRESHOLD = 20
BOOT_N = 500
BOOT_SEED = 2024


def bootstrap_pair(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray | None,
    fn: Callable[[np.ndarray, np.ndarray, np.ndarray | None], float],
) -> tuple[float, float, float]:
    rng = np.random.default_rng(BOOT_SEED)
    n = len(a)
    point = float(fn(a, b, c))
    vals: list[float] = []
    for _ in range(BOOT_N):
        idx = rng.integers(0, n, n)
        try:
            v = float(fn(a[idx], b[idx], None if c is None else c[idx]))
            if np.isfinite(v):
                vals.append(v)
        except Exception:
            continue
    lo, hi = np.quantile(vals, [0.025, 0.975]) if vals else (np.nan, np.nan)
    return point, float(lo), float(hi)


def spearman(a: np.ndarray, b: np.ndarray, _: np.ndarray | None = None) -> float:
    return float(stats.spearmanr(a, b).correlation)


def diff_len_minus_error(a: np.ndarray, b: np.ndarray, c: np.ndarray | None) -> float:
    assert c is not None
    return float(stats.spearmanr(a, b).correlation - stats.spearmanr(a, c).correlation)


def scoreable_ids(dataset: str) -> set[str]:
    items = pd.read_parquet(PROCESSED / dataset / "items.parquet", columns=["item_id_hash", "eligible_for_llm_scoring"])
    return set(items.loc[items["eligible_for_llm_scoring"], "item_id_hash"])


def build_criterion_sensitivity() -> pd.DataFrame:
    llm = pd.read_parquet(LLM_FEATURES, columns=["dataset", "item_id_hash", "model_identifier", "scalar_difficulty"])
    rows = []
    for ds in DATASETS:
        ids = scoreable_ids(ds)
        ix = pd.read_parquet(
            PROCESSED / ds / "interactions.parquet",
            columns=["student_id_hash", "item_id_hash", "correct", "split_assignment", "first_attempt", "sequence_index"],
        )
        ix = ix[(ix["split_assignment"] == "test") & (ix["item_id_hash"].isin(ids))].copy()
        ix["incorrect"] = 1 - ix["correct"].astype(int)

        response = ix.groupby("item_id_hash", as_index=False).agg(
            n_responses=("incorrect", "size"),
            response_incorrect=("incorrect", "sum"),
        )
        response["response_error"] = (response["response_incorrect"] + 1) / (response["n_responses"] + 2)
        first_observed_ix = ix.sort_values("sequence_index").drop_duplicates(["student_id_hash", "item_id_hash"], keep="first")
        first = first_observed_ix.groupby("item_id_hash", as_index=False).agg(
            n_first_attempts=("incorrect", "size"),
            first_incorrect=("incorrect", "sum"),
        )
        first["first_attempt_error"] = (first["first_incorrect"] + 1) / (first["n_first_attempts"] + 2)
        student_item = ix.groupby(["item_id_hash", "student_id_hash"], as_index=False).agg(
            student_item_error=("incorrect", "mean")
        )
        student_weighted = student_item.groupby("item_id_hash", as_index=False).agg(
            n_students=("student_id_hash", "nunique"),
            student_weighted_error=("student_item_error", "mean"),
        )
        student_weighted["student_weighted_error"] = (
            student_weighted["student_weighted_error"] * student_weighted["n_students"] + 1
        ) / (student_weighted["n_students"] + 2)
        crit = response.merge(first, on="item_id_hash", how="left").merge(student_weighted, on="item_id_hash", how="left")
        crit = crit[crit["n_responses"] >= PRIMARY_THRESHOLD].copy()
        for model in MODELS:
            l = llm[(llm["dataset"] == ds) & (llm["model_identifier"] == model)][["item_id_hash", "scalar_difficulty"]]
            df = crit.merge(l, on="item_id_hash", how="inner")
            for ref, min_col, value_col in [
                ("response_level_error", "n_responses", "response_error"),
                ("first_attempt_error", "n_first_attempts", "first_attempt_error"),
                ("student_weighted_error", "n_students", "student_weighted_error"),
            ]:
                sub = df.dropna(subset=["scalar_difficulty", value_col])
                if len(sub) < 5:
                    continue
                x = sub["scalar_difficulty"].to_numpy()
                y = sub[value_col].to_numpy()
                point, lo, hi = bootstrap_pair(x, y, None, spearman)
                rows.append({
                    "dataset": ds,
                    "model": model,
                    "criterion": ref,
                    "n_items": int(len(sub)),
                    "min_item_records": int(sub[min_col].min()),
                    "median_item_records": float(sub[min_col].median()),
                    "spearman_rho": point,
                    "spearman_ci_lo": lo,
                    "spearman_ci_hi": hi,
                })
    out = pd.DataFrame(rows)
    out.to_csv(TABLE / "CRITERION_SENSITIVITY.csv", index=False)
    return out


def build_surface_ci() -> pd.DataFrame:
    llm = pd.read_parquet(LLM_FEATURES, columns=["dataset", "item_id_hash", "model_identifier", "scalar_difficulty"])
    ref = pd.read_csv(TABLE / "AUTHENTIC_DIFFICULTY_REFERENCES_V2_ORIENTATION_CORRECTED.csv")
    surface = pd.read_csv(TABLE / "AUTHENTIC_ITEM_SURFACE_FEATURES.csv")
    rows = []
    for ds in DATASETS:
        held = ref[
            (ref["dataset"] == ds)
            & (ref["reference_scope"] == "held_out_test")
            & (ref["heldout_response_count"] >= PRIMARY_THRESHOLD)
        ][["dataset", "item_id_hash", "smoothed_error_beta_1_1"]]
        surf = surface[surface["dataset"] == ds][["dataset", "item_id_hash", "char_length"]]
        for model in MODELS:
            l = llm[(llm["dataset"] == ds) & (llm["model_identifier"] == model)][["item_id_hash", "scalar_difficulty"]]
            df = held.merge(surf, on=["dataset", "item_id_hash"], how="inner").merge(l, on="item_id_hash", how="inner")
            x = df["scalar_difficulty"].to_numpy()
            y_len = df["char_length"].to_numpy()
            y_err = df["smoothed_error_beta_1_1"].to_numpy()
            len_p, len_lo, len_hi = bootstrap_pair(x, y_len, None, spearman)
            err_p, err_lo, err_hi = bootstrap_pair(x, y_err, None, spearman)
            diff_p, diff_lo, diff_hi = bootstrap_pair(x, y_len, y_err, diff_len_minus_error)
            rows.extend([
                {
                    "dataset": ds,
                    "model": model,
                    "association": "LLM_score_vs_character_length",
                    "n_items": int(len(df)),
                    "spearman_rho": len_p,
                    "spearman_ci_lo": len_lo,
                    "spearman_ci_hi": len_hi,
                },
                {
                    "dataset": ds,
                    "model": model,
                    "association": "LLM_score_vs_held_out_error",
                    "n_items": int(len(df)),
                    "spearman_rho": err_p,
                    "spearman_ci_lo": err_lo,
                    "spearman_ci_hi": err_hi,
                },
                {
                    "dataset": ds,
                    "model": model,
                    "association": "character_length_minus_held_out_error",
                    "n_items": int(len(df)),
                    "spearman_rho": diff_p,
                    "spearman_ci_lo": diff_lo,
                    "spearman_ci_hi": diff_hi,
                },
            ])
    out = pd.DataFrame(rows)
    out.to_csv(TABLE / "SURFACE_ASSOCIATION_BOOTSTRAP.csv", index=False)
    return out


def main() -> int:
    TABLE.mkdir(parents=True, exist_ok=True)
    crit = build_criterion_sensitivity()
    surf = build_surface_ci()
    manifest = {
        "status": "SENSITIVITY_TABLES_READY",
        "criterion_sensitivity_rows": len(crit),
        "surface_bootstrap_rows": len(surf),
        "bootstrap_draws": BOOT_N,
        "bootstrap_seed": BOOT_SEED,
        "primary_threshold": PRIMARY_THRESHOLD,
        "outputs": [
            "tables/CRITERION_SENSITIVITY.csv",
            "tables/SURFACE_ASSOCIATION_BOOTSTRAP.csv",
        ],
    }
    (ROOT / "data_manifests" / "sensitivity_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
