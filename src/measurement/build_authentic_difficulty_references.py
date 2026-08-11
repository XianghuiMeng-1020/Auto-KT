#!/usr/bin/env python3
"""Build leakage-safe authentic item difficulty references for Phase F1."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "measurement"))

from measurement_common import (  # noqa: E402
    DATASETS,
    REPORT_DIR,
    TABLE_DIR,
    beta_smooth,
    compute_error_rates,
    compute_exposure,
    empirical_bayes_prior,
    extract_surface_features,
    load_config,
    load_scoreable_items,
    load_interactions,
    utc_now,
)
from rasch_estimator import fit_rasch_1pl  # noqa: E402


def _error_reference_table(
    dataset: str,
    interactions: pd.DataFrame,
    split: str,
    reference_scope: str,
) -> pd.DataFrame:
    raw = compute_error_rates(interactions, split)
    raw_first = compute_error_rates(interactions, split, first_attempt_only=True)
    eb_alpha, eb_beta = empirical_bayes_prior(raw["n_correct"], raw["n_responses"])
    first_map = raw_first.set_index("item_id_hash")["raw_error_rate"]
    return pd.DataFrame({
        "dataset": dataset,
        "item_id_hash": raw["item_id_hash"],
        "reference_scope": reference_scope,
        "split_source": split,
        "raw_error_rate": raw["raw_error_rate"],
        "smoothed_error_beta_1_1": beta_smooth(raw["n_correct"], raw["n_responses"], 1, 1),
        "smoothed_error_beta_2_2": beta_smooth(raw["n_correct"], raw["n_responses"], 2, 2),
        "smoothed_error_eb": beta_smooth(raw["n_correct"], raw["n_responses"], eb_alpha, eb_beta),
        "n_responses": raw["n_responses"],
        "n_students": raw["n_students"],
        "first_attempt_error_rate": raw["item_id_hash"].map(first_map),
        "eb_prior_alpha": eb_alpha,
        "eb_prior_beta": eb_beta,
    })


def _oracle_error_reference(dataset: str, interactions: pd.DataFrame) -> pd.DataFrame:
    agg = interactions.groupby("item_id_hash", as_index=False).agg(
        n_responses=("correct", "size"),
        n_correct=("correct", "sum"),
        n_students=("student_id_hash", "nunique"),
    )
    agg["raw_error_rate"] = 1 - agg["n_correct"] / agg["n_responses"]
    eb_alpha, eb_beta = empirical_bayes_prior(agg["n_correct"], agg["n_responses"])
    return pd.DataFrame({
        "dataset": dataset,
        "item_id_hash": agg["item_id_hash"],
        "reference_scope": "oracle_diagnostic",
        "split_source": "all",
        "raw_error_rate": agg["raw_error_rate"],
        "smoothed_error_beta_1_1": beta_smooth(agg["n_correct"], agg["n_responses"], 1, 1),
        "smoothed_error_beta_2_2": beta_smooth(agg["n_correct"], agg["n_responses"], 2, 2),
        "smoothed_error_eb": beta_smooth(agg["n_correct"], agg["n_responses"], eb_alpha, eb_beta),
        "n_responses": agg["n_responses"],
        "n_students": agg["n_students"],
        "first_attempt_error_rate": float("nan"),
        "eb_prior_alpha": eb_alpha,
        "eb_prior_beta": eb_beta,
    })


def _attach_rasch(refs: pd.DataFrame, interactions: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    result = fit_rasch_1pl(
        interactions[["student_id_hash", "item_id_hash", "correct"]],
        max_iter=cfg["rasch"]["max_iter"],
        tol=cfg["rasch"]["tol"],
        seed=cfg["rasch"]["seed"],
        min_responses=cfg["rasch"]["min_responses_for_fit"],
    )
    rasch = result.item_difficulties.set_index("item_id_hash")
    refs = refs.copy()
    refs["rasch_item_difficulty"] = refs["item_id_hash"].map(rasch["rasch_difficulty"])
    refs["rasch_se"] = refs["item_id_hash"].map(rasch["rasch_se"])
    refs["identifiable"] = refs["item_id_hash"].map(rasch["identifiable"])
    refs["perfect_score"] = refs["item_id_hash"].map(rasch["perfect_score"])
    refs["zero_score"] = refs["item_id_hash"].map(rasch["zero_score"])
    return refs, result.convergence


def main() -> int:
    cfg = load_config()
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    all_refs = []
    rasch_rows = []
    surface_rows = []

    for dataset in DATASETS:
        print(f"Processing {dataset}...", flush=True)
        items = load_scoreable_items(dataset)
        scoreable_ids = set(items["item_id_hash"])
        expected = cfg["llm_scoreable_counts"][dataset]
        if len(items) != expected:
            raise SystemExit(f"{dataset}: expected {expected} scoreable items, got {len(items)}")

        cols = ["student_id_hash", "item_id_hash", "correct", "split_assignment", "first_attempt"]
        interactions = load_interactions(dataset, columns=cols)
        interactions = interactions[interactions["item_id_hash"].isin(scoreable_ids)].copy()

        exposure = compute_exposure(interactions.assign(dataset=dataset), dataset)
        surface_rows.append(extract_surface_features(items.assign(dataset=dataset), exposure))

        scopes = [
            ("train", "deployable_train"),
            ("test", "held_out_test"),
        ]
        for split, scope in scopes:
            split_ix = interactions[interactions["split_assignment"] == split]
            refs = _error_reference_table(dataset, split_ix, split, scope)
            refs, conv = _attach_rasch(refs, split_ix, cfg)
            conv.update({"dataset": dataset, "reference_scope": scope, "split_source": split})
            rasch_rows.append(conv)
            all_refs.append(refs)

        oracle_ix = interactions
        refs = _oracle_error_reference(dataset, oracle_ix)
        refs, conv = _attach_rasch(refs, oracle_ix, cfg)
        conv.update({"dataset": dataset, "reference_scope": "oracle_diagnostic", "split_source": "all"})
        rasch_rows.append(conv)
        all_refs.append(refs)

    refs_df = pd.concat(all_refs, ignore_index=True)
    refs_df.to_csv(TABLE_DIR / "AUTHENTIC_DIFFICULTY_REFERENCES.csv", index=False)
    pd.concat(surface_rows, ignore_index=True).to_csv(
        TABLE_DIR / "AUTHENTIC_ITEM_SURFACE_FEATURES.csv", index=False
    )
    pd.DataFrame(rasch_rows).to_csv(TABLE_DIR / "RASCH_CONVERGENCE_SUMMARY.csv", index=False)
    refs_df[
        ["dataset", "item_id_hash", "reference_scope", "split_source",
         "rasch_item_difficulty", "rasch_se", "identifiable"]
    ].drop_duplicates().to_csv(TABLE_DIR / "RASCH_ITEM_DIFFICULTIES.csv", index=False)

    report = [
        "# Rasch Estimation Report",
        "",
        f"**Generated:** {utc_now()}",
        f"**Estimator:** `{cfg['rasch']['estimator_version']}`",
        "",
        "Larger `rasch_item_difficulty` means harder items.",
        "",
        "Perfect-score and zero-score items are flagged non-identifiable.",
        "",
    ]
    for r in rasch_rows:
        report.append(
            f"- **{r['dataset']} / {r['reference_scope']}**: converged={r['converged']}, "
            f"iter={r['iterations']}, identifiable={r['n_identifiable_items']}, "
            f"extreme={r['n_extreme_items']}, n_obs={r['n_observations']}"
        )
    (REPORT_DIR / "RASCH_ESTIMATION_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Wrote {len(refs_df)} reference rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
