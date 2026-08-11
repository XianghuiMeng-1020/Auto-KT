#!/usr/bin/env python3
"""Full static content-sufficiency check for XES3G5M and Junyi."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "data_prep"))

from content_sufficiency import (  # noqa: E402
    CONTENT_SUFFICIENCY_RULE_VERSION,
    PASS_REASON,
    apply_classification,
    classification_audit_hash,
)
from unified_schema_common import PROCESSED_ROOT, sequence_length_summary  # noqa: E402

TABLE_DIR = ROOT / "results"
REPORT_DIR = ROOT / "reports" / "data_audits"
DATASETS = ("xes3g5m", "junyi")
EXPOSURE_LEVELS = (0, 1, 3, 5, 10, 20)


def coverage_stats(items: pd.DataFrame, interactions_path: Path) -> dict:
    included = set(items.loc[items["eligible_for_llm_scoring"], "item_id_hash"])
    pf = pq.ParquetFile(interactions_path)
    cols = ["student_id_hash", "item_id_hash", "primary_concept_id"]
    avail = set(pf.schema_arrow.names)
    read_cols = [c for c in cols if c in avail]

    kept_rows = 0
    student_ix: dict[str, int] = {}
    item_counts: dict[str, int] = {}
    domains: dict[str, int] = {}
    item_domain = dict(zip(items["item_id_hash"], items.get("mathematical_domain", pd.Series(dtype=str))))

    for batch in pf.iter_batches(batch_size=500_000, columns=read_cols):
        df = batch.to_pandas()
        mask = df["item_id_hash"].isin(included)
        sub = df[mask]
        kept_rows += len(sub)
        for sid, cnt in sub.groupby("student_id_hash").size().items():
            student_ix[sid] = student_ix.get(sid, 0) + int(cnt)
        for iid, cnt in sub["item_id_hash"].value_counts().items():
            item_counts[iid] = item_counts.get(iid, 0) + int(cnt)
        if "primary_concept_id" in sub.columns:
            for iid in sub["item_id_hash"].unique():
                dom = str(item_domain.get(iid, "unknown"))
                domains[dom] = domains.get(dom, 0) + item_counts.get(iid, 0)

    lens = pd.Series(student_ix)
    exposure_feasible = {
        str(k): int((pd.Series(item_counts).ge(k).sum() if item_counts else 0))
        for k in EXPOSURE_LEVELS
    }
    domain_total = sum(domains.values()) or 1
    top_domain = max(domains, key=domains.get) if domains else "n/a"
    top_share = domains.get(top_domain, 0) / domain_total if domains else 0.0

    return {
        "retained_interactions": kept_rows,
        "retained_students": len(student_ix),
        "students_ge_5": int((lens >= 5).sum()) if len(lens) else 0,
        "students_ge_10": int((lens >= 10).sum()) if len(lens) else 0,
        "students_ge_20": int((lens >= 20).sum()) if len(lens) else 0,
        "students_ge_50": int((lens >= 50).sum()) if len(lens) else 0,
        "sequence_length_min": int(lens.min()) if len(lens) else 0,
        "sequence_length_median": float(lens.median()) if len(lens) else 0.0,
        "sequence_length_mean": float(lens.mean()) if len(lens) else 0.0,
        "sequence_length_max": int(lens.max()) if len(lens) else 0,
        "item_response_count_median": float(pd.Series(item_counts).median()) if item_counts else 0.0,
        "exposure_level_item_counts": exposure_feasible,
        "top_mathematical_domain": top_domain,
        "top_domain_interaction_share": round(top_share, 4),
        "response_coverage_pct": round(
            100 * kept_rows / max(1, pf.metadata.num_rows), 2
        ),
    }


def audit_dataset(dataset: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    items_path = PROCESSED_ROOT / dataset / "items.parquet"
    interactions_path = PROCESSED_ROOT / dataset / "interactions.parquet"
    items = pd.read_parquet(items_path)
    classified = apply_classification(items, dataset)

    detail_cols = [
        "item_id_hash",
        "source_content_hash",
        "item_content_type",
        "item_format",
        "eligible_for_llm_scoring",
        "llm_exclusion_primary_reason",
        "llm_exclusion_secondary_flags",
        "content_sufficiency_rule_version",
    ]
    if dataset == "junyi":
        detail_cols.extend([
            "has_image_dependency",
            "has_dynamic_template",
            "graphie_only_no_question_text",
        ])
    detail = classified[[c for c in detail_cols if c in classified.columns]].copy()
    detail.insert(0, "dataset", dataset)

    reason_counts = (
        classified["llm_exclusion_primary_reason"]
        .value_counts()
        .rename_axis("primary_reason")
        .reset_index(name="item_count")
    )
    reason_counts.insert(0, "dataset", dataset)
    n_inc = int(classified["eligible_for_llm_scoring"].sum())
    n_exc = int((~classified["eligible_for_llm_scoring"]).sum())
    cov = coverage_stats(classified, interactions_path)
    summary = {
        "dataset": dataset,
        "rule_version": CONTENT_SUFFICIENCY_RULE_VERSION,
        "total_items": len(classified),
        "included_items": n_inc,
        "excluded_items": n_exc,
        "included_pct": round(100 * n_inc / max(1, len(classified)), 2),
        **cov,
    }
    return detail, reason_counts, summary


def write_audit_report(dataset: str, summary: dict, reason_counts: pd.DataFrame) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / f"{dataset.upper()}_CONTENT_SUFFICIENCY_AUDIT.md"
    lines = [
        f"# {dataset.upper()} Content Sufficiency Audit",
        "",
        f"**Rule version:** `{summary['rule_version']}`",
        f"**Generated:** audit_content_sufficiency.py",
        "",
        "## Item counts",
        "",
        f"- Total KT-universe items: {summary['total_items']}",
        f"- LLM-scoreable (included): {summary['included_items']} ({summary['included_pct']}%)",
        f"- Excluded: {summary['excluded_items']}",
        "",
        "## Exclusions by primary reason",
        "",
        "| Reason | Count |",
        "|---|---:|",
    ]
    for _, row in reason_counts.iterrows():
        lines.append(f"| `{row['primary_reason']}` | {row['item_count']} |")
    lines.extend([
        "",
        "## Retained coverage (LLM-scoreable universe)",
        "",
        f"- Interactions retained: {summary['retained_interactions']:,}",
        f"- Response coverage: {summary['response_coverage_pct']}%",
        f"- Students retained: {summary['retained_students']:,}",
        f"- Students with ≥5/10/20/50 retained interactions: "
        f"{summary['students_ge_5']}/{summary['students_ge_10']}/"
        f"{summary['students_ge_20']}/{summary['students_ge_50']}",
        f"- Sequence length (retained ix per student): min={summary['sequence_length_min']}, "
        f"median={summary['sequence_length_median']:.1f}, max={summary['sequence_length_max']}",
        f"- Item response-count median: {summary['item_response_count_median']:.1f}",
        f"- Top mathematical domain: `{summary['top_mathematical_domain']}` "
        f"({100*summary['top_domain_interaction_share']:.1f}% of retained interactions)",
        "",
        "## Exposure-level feasibility (items with ≥k responses)",
        "",
    ])
    for k, v in summary["exposure_level_item_counts"].items():
        lines.append(f"- ≥{k}: {v} items")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    all_summaries = []
    for ds in DATASETS:
        detail, reasons, summary = audit_dataset(ds)
        prefix = ds.upper()
        detail.to_csv(TABLE_DIR / f"{prefix}_LLM_SCOREABILITY.csv", index=False)
        reasons.to_csv(TABLE_DIR / f"{prefix}_LLM_SCOREABILITY_SUMMARY.csv", index=False)
        write_audit_report(ds, summary, reasons)
        all_summaries.append(summary)
        print(f"{ds}: included={summary['included_items']} excluded={summary['excluded_items']}")

    meta = {
        "rule_version": CONTENT_SUFFICIENCY_RULE_VERSION,
        "audit_hashes": {
            ds: classification_audit_hash(
                apply_classification(pd.read_parquet(PROCESSED_ROOT / ds / "items.parquet"), ds)
            )
            for ds in DATASETS
        },
        "summaries": all_summaries,
    }
    (TABLE_DIR / "CONTENT_SUFFICIENCY_AUDIT_META.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
