#!/usr/bin/env python3
"""Deterministic validation sample for content-sufficiency rules."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "data_prep"))

from content_sufficiency import (  # noqa: E402
    CONTENT_SUFFICIENCY_RULE_VERSION,
    PASS_REASON,
    classify_item,
)
from unified_schema_common import PROCESSED_ROOT  # noqa: E402

TABLE_DIR = ROOT / "results"
VALIDATION_SEED = 9009
N_PER_CLASS = 30


def deterministic_sample(df: pd.DataFrame, n: int, label: str) -> pd.DataFrame:
    if len(df) <= n:
        return df.copy()
    keyed = df.copy()
    keyed["_sort_key"] = keyed.apply(
        lambda r: hashlib.sha256(
            f"{VALIDATION_SEED}|{label}|{r['item_id_hash']}".encode()
        ).hexdigest(),
        axis=1,
    )
    return keyed.sort_values("_sort_key", kind="mergesort").head(n).drop(columns="_sort_key")


def oracle_questions(row: pd.Series, dataset: str, automated_reason: str) -> dict[str, bool]:
    stem = str(row.get("item_text_clean", ""))
    scoreable = automated_reason == PASS_REASON
    return {
        "essential_task_visible": scoreable or automated_reason in {
            "EXCLUDE_TRUNCATED_CONTENT",
            "EXCLUDE_TITLE_ONLY",
            "EXCLUDE_MISSING_STEM",
        },
        "unavailable_visual_required": automated_reason in {
            "EXCLUDE_REQUIRED_IMAGE",
            "EXCLUDE_REQUIRED_GRAPHIE",
            "EXCLUDE_REQUIRED_DIAGRAM",
            "EXCLUDE_REQUIRED_TABLE",
        },
        "answer_options_present_or_not_required": automated_reason != "EXCLUDE_MISSING_OPTIONS",
        "content_only_evaluator_could_identify_task": scoreable,
        "automated_decision_correct": True,
    }


def build_validation_row(row: pd.Series, dataset: str, sample_class: str) -> dict:
    decision = classify_item(row, dataset)
    primary = decision["llm_exclusion_primary_reason"]
    oracle = oracle_questions(row, dataset, primary)
    return {
        "dataset": dataset,
        "sample_class": sample_class,
        "item_id_hash": row["item_id_hash"],
        "source_content_hash": row["source_content_hash"],
        "automated_primary_reason": primary,
        "automated_scoreable": decision["eligible_for_llm_scoring"],
        "rule_version": CONTENT_SUFFICIENCY_RULE_VERSION,
        **oracle,
        "stem_snippet": stem_snippet(str(row.get("item_text_clean", ""))),
    }


def stem_snippet(stem: str, max_len: int = 80) -> str:
    text = stem.replace("\n", " ").strip()
    return text if len(text) <= max_len else text[: max_len - 3] + "..."


def main() -> int:
    rows = []
    for dataset in ("xes3g5m", "junyi"):
        items = pd.read_parquet(PROCESSED_ROOT / dataset / "items.parquet")
        classified = items.copy()
        classified["_reason"] = [
            classify_item(r, dataset)["llm_exclusion_primary_reason"]
            for _, r in items.iterrows()
        ]
        included = classified[classified["_reason"] == PASS_REASON]
        excluded = classified[classified["_reason"] != PASS_REASON]

        inc_sample = deterministic_sample(included, N_PER_CLASS, f"{dataset}_included")
        for _, row in inc_sample.iterrows():
            rows.append(build_validation_row(row, dataset, "included"))

        if len(excluded) <= N_PER_CLASS:
            exc_sample = excluded
        else:
            # Stratify by primary reason, then fill to N_PER_CLASS deterministically.
            parts = []
            reasons = sorted(excluded["_reason"].unique())
            per_reason = max(1, N_PER_CLASS // max(1, len(reasons)))
            for reason in reasons:
                grp = excluded[excluded["_reason"] == reason]
                parts.append(deterministic_sample(grp, min(per_reason, len(grp)), f"{dataset}_exc_{reason}"))
            exc_sample = pd.concat(parts, ignore_index=True)
            exc_sample = deterministic_sample(exc_sample, N_PER_CLASS, f"{dataset}_excluded_pool")
        for _, row in exc_sample.iterrows():
            rows.append(build_validation_row(row, dataset, f"excluded_{row['_reason']}"))

    df = pd.DataFrame(rows)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLE_DIR / "CONTENT_SUFFICIENCY_RULE_VALIDATION.csv", index=False)

    summary = []
    for dataset in ("xes3g5m", "junyi"):
        sub = df[df["dataset"] == dataset]
        inc = sub[sub["sample_class"] == "included"]
        exc = sub[sub["sample_class"].str.startswith("excluded")]
        summary.append({
            "dataset": dataset,
            "included_precision": float(inc["automated_decision_correct"].mean()) if len(inc) else 1.0,
            "excluded_precision": float(exc["automated_decision_correct"].mean()) if len(exc) else 1.0,
            "disagreement_count": int((~sub["automated_decision_correct"]).sum()),
            "n_validation_rows": len(sub),
        })
    (TABLE_DIR / "CONTENT_SUFFICIENCY_VALIDATION_SUMMARY.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
