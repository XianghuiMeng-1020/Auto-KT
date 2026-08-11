#!/usr/bin/env python3
"""Validate unified schema outputs and emit audit reports."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "data_prep"))

from unified_schema_common import (  # noqa: E402
    LLM_PROMPT_ALLOWLIST,
    LLM_PROMPT_DENYLIST,
    PROCESSED_ROOT,
    UnifiedSchemaConfig,
    sha256_file,
)

DATASETS = ("xes3g5m", "junyi")
FROZEN_COUNTS = {"xes3g5m": 7618, "junyi": 666}


def load_dataset(dataset: str) -> dict[str, pd.DataFrame]:
    d = PROCESSED_ROOT / dataset
    return {
        "interactions": pd.read_parquet(d / "interactions.parquet"),
        "items": pd.read_parquet(d / "items.parquet"),
        "splits": pd.read_parquet(d / "splits.parquet"),
        "concept_edges": pd.read_parquet(d / "concept_edges.parquet"),
        "llm_prompt_items": pd.read_parquet(d / "llm_prompt_items.parquet"),
    }


def check_no_raw_ids(columns: set[str]) -> bool:
    banned = {"user_id", "uid", "student_id", "exercise", "question_id", "item_id"}
    return banned.isdisjoint(columns)


def check_split_disjoint(splits: pd.DataFrame) -> bool:
    train = set(splits.loc[splits["split_assignment"] == "train", "student_id_hash"])
    val = set(splits.loc[splits["split_assignment"] == "val", "student_id_hash"])
    test = set(splits.loc[splits["split_assignment"] == "test", "student_id_hash"])
    return not (train & val or train & test or val & test)


def check_interaction_splits_match_path(inter_path: Path, splits: pd.DataFrame) -> bool:
    import pyarrow.parquet as pq

    mapping = dict(zip(splits["student_id_hash"], splits["split_assignment"]))
    pf = pq.ParquetFile(inter_path)
    for batch in pf.iter_batches(
        batch_size=500_000, columns=["student_id_hash", "split_assignment"]
    ):
        df = batch.to_pandas()
        expected = df["student_id_hash"].map(mapping)
        if not (expected == df["split_assignment"]).all():
            return False
    return True


def check_interaction_splits_match(interactions: pd.DataFrame, splits: pd.DataFrame) -> bool:
    mapping = dict(zip(splits["student_id_hash"], splits["split_assignment"]))
    expected = interactions["student_id_hash"].map(mapping)
    return (expected == interactions["split_assignment"]).all()


def check_item_join_path(inter_path: Path, item_set: set[str]) -> bool:
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(inter_path)
    for batch in pf.iter_batches(batch_size=500_000, columns=["item_id_hash"]):
        if not batch.to_pandas()["item_id_hash"].isin(item_set).all():
            return False
    return True


def check_sequence_monotonic_path(inter_path: Path) -> bool:
    import pyarrow.parquet as pq
    from collections import defaultdict

    stats: dict[str, dict[str, int]] = defaultdict(lambda: {"min": 10**9, "max": -1, "cnt": 0})
    pf = pq.ParquetFile(inter_path)
    for batch in pf.iter_batches(batch_size=500_000, columns=["student_id_hash", "sequence_index"]):
        df = batch.to_pandas()
        grouped = df.groupby("student_id_hash")["sequence_index"].agg(["min", "max", "count"])
        for sid, row in grouped.iterrows():
            st = stats[sid]
            st["min"] = min(st["min"], int(row["min"]))
            st["max"] = max(st["max"], int(row["max"]))
            st["cnt"] += int(row["count"])
    for st in stats.values():
        if st["min"] != 0 or st["max"] + 1 != st["cnt"]:
            return False
    return True


def check_llm_prompt_safe(llm: pd.DataFrame, items: pd.DataFrame) -> tuple[bool, list[str]]:
    errors = []
    if set(llm.columns) != set(LLM_PROMPT_ALLOWLIST):
        errors.append(f"columns mismatch: {set(llm.columns)} vs {LLM_PROMPT_ALLOWLIST}")
    for col in LLM_PROMPT_DENYLIST:
        if col in llm.columns:
            errors.append(f"denied column present: {col}")
    if "correct_answer_separate" in items.columns:
        if "correct_answer_separate" in llm.columns:
            errors.append("answer leaked into llm export")
    return len(errors) == 0, errors


def check_sequence_monotonic(interactions: pd.DataFrame | Path) -> bool:
    if isinstance(interactions, Path):
        return check_sequence_monotonic_path(interactions)
    df = interactions[["student_id_hash", "sequence_index"]]
    for _, grp in interactions.groupby("student_id_hash"):
        ts = grp["timestamp_or_order"].tolist()
        if ts != sorted(ts):
            return False
    for _, grp in df.groupby("student_id_hash", sort=False):
        seq = grp["sequence_index"].tolist()
        if seq != list(range(len(seq))):
            return False
    return True


def check_concept_edge_provenance(edges: pd.DataFrame) -> bool:
    train_edges = edges[edges["edge_source"] == "training_response_cooccurrence"]
    if not (train_edges["permitted_split"] == "train").all():
        return False
    official = edges[edges["edge_source"] == "official_metadata"]
    return (official["permitted_split"] == "all").all()


def validate_dataset(dataset: str) -> dict:
    import pyarrow.parquet as pq

    d = PROCESSED_ROOT / dataset
    inter_path = d / "interactions.parquet"
    pf = pq.ParquetFile(inter_path)
    inter_columns = set(pf.schema_arrow.names)
    n_inter = pf.metadata.num_rows
    items = pd.read_parquet(d / "items.parquet")
    splits = pd.read_parquet(d / "splits.parquet")
    edges = pd.read_parquet(d / "concept_edges.parquet")
    llm = pd.read_parquet(d / "llm_prompt_items.parquet")
    llm_ok, llm_errors = check_llm_prompt_safe(llm, items)
    item_set = set(items["item_id_hash"])
    join_ok = check_item_join_path(inter_path, item_set)
    count_ok = len(items) == FROZEN_COUNTS[dataset]
    checks = {
        "no_raw_ids": check_no_raw_ids(inter_columns),
        "split_disjoint": check_split_disjoint(splits),
        "interaction_split_match": check_interaction_splits_match_path(inter_path, splits),
        "llm_prompt_safe": llm_ok,
        "eligible_count": count_ok,
        "interaction_item_join": join_ok,
        "stable_content_hash": items["source_content_hash"].notna().all(),
        "sequence_monotonic": check_sequence_monotonic_path(inter_path),
        "concept_edge_provenance": check_concept_edge_provenance(edges),
    }
    if not llm_ok:
        checks["llm_errors"] = llm_errors
    status = "PASS" if all(v for k, v in checks.items() if k != "llm_errors") else "FAIL"
    return {
        "dataset": dataset,
        "status": status,
        "eligible_items": len(items),
        "interactions": n_inter,
        "students": splits["student_id_hash"].nunique(),
        "llm_prompt_items": len(llm),
        "checks": checks,
        "file_hashes": {
            name: sha256_file(PROCESSED_ROOT / dataset / f"{name}.parquet")
            for name in ("interactions", "items", "splits", "concept_edges", "llm_prompt_items")
        },
    }


def write_llm_audit(results: list[dict]) -> None:
    lines = [
        "# LLM Prompt Input Audit",
        "",
        f"**Generated from commit validation**",
        "",
    ]
    for r in results:
        ds = r["dataset"]
        d = PROCESSED_ROOT / ds
        llm = pd.read_parquet(d / "llm_prompt_items.parquet")
        items = pd.read_parquet(d / "items.parquet")
        lines += [
            f"## {ds}",
            "",
            f"| Metric | Value |",
            f"|---|---:|",
            f"| Item count (eligible) | {len(items)} |",
            f"| LLM prompt rows | {len(llm)} |",
            f"| Allowed fields | {', '.join(sorted(LLM_PROMPT_ALLOWLIST))} |",
            f"| Missing text count | {int((llm['item_text_clean'].astype(str).str.len() < 5).sum())} |",
            f"| Prompt-eligible count | {len(llm)} |",
            f"| Checks passed | {r['checks'].get('llm_prompt_safe')} |",
            "",
        ]
    path = ROOT / "reports" / "data_audits" / "LLM_PROMPT_INPUT_AUDIT.md"
    path.write_text("\n".join(lines), encoding="utf-8")


def write_unified_report(results: list[dict], cfg: UnifiedSchemaConfig) -> str:
    all_pass = all(r["status"] == "PASS" for r in results)
    if all_pass:
        overall = "SCHEMA_BUILD_PASS"
    else:
        failed = [r for r in results if r["status"] != "PASS"]
        if any(not r["checks"].get("eligible_count") for r in failed):
            overall = "SCHEMA_BUILD_COUNT_MISMATCH"
        elif any(not r["checks"].get("interaction_item_join") for r in failed):
            overall = "SCHEMA_BUILD_JOIN_FAIL"
        elif any(not r["checks"].get("split_disjoint") or not r["checks"].get("interaction_split_match") for r in failed):
            overall = "SCHEMA_BUILD_SPLIT_FAIL"
        elif any(not r["checks"].get("llm_prompt_safe") for r in failed):
            overall = "SCHEMA_BUILD_LEAKAGE_FAIL"
        else:
            overall = "SCHEMA_BUILD_LEAKAGE_FAIL"

    lines = [
        "# Unified Schema Build Report",
        "",
        f"**Overall status:** `{overall}`",
        f"**Config hash:** `{cfg.config_hash()}`",
        "",
    ]
    for r in results:
        lines += [
            f"## {r['dataset']}",
            "",
            f"- Eligible items: {r['eligible_items']} (frozen: {FROZEN_COUNTS[r['dataset']]})",
            f"- Interactions: {r['interactions']}",
            f"- Students: {r['students']}",
            f"- Status: {r['status']}",
            "",
            "### Checks",
            "",
        ]
        for k, v in r["checks"].items():
            if k == "llm_errors":
                continue
            lines.append(f"- {k}: {'PASS' if v else 'FAIL'}")
        lines.append("")

    path = ROOT / "reports" / "data_audits" / "UNIFIED_SCHEMA_REPORT.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return overall


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.bool_, np.generic)):
        return obj.item()
    return obj


def write_unified_manifest(results: list[dict], overall: str, cfg: UnifiedSchemaConfig) -> None:
    manifest = {
        "manifest_version": "1.0.0",
        "overall_status": overall,
        "config_hash": cfg.config_hash(),
        "frozen_eligible_counts": FROZEN_COUNTS,
        "datasets": {r["dataset"]: _json_safe(r) for r in results},
    }
    path = ROOT / "data_manifests" / "_unified_schema_manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def write_combined_tables() -> None:
    """Merge per-dataset audit tables into protocol-named combined CSVs."""
    table_dir = ROOT / "results"
    samples, splits, flows = [], [], []
    for ds in DATASETS:
        sample_path = table_dir / f"{ds}_SCHEMA_SAMPLE.csv"
        split_path = table_dir / f"{ds}_SPLIT_COUNTS.csv"
        flow_path = table_dir / f"{ds}_EXCLUSION_FLOW.csv"
        if sample_path.exists():
            df = pd.read_csv(sample_path)
            if "dataset" not in df.columns:
                df.insert(0, "dataset", ds)
            samples.append(df)
        if split_path.exists():
            df = pd.read_csv(split_path)
            if "dataset" not in df.columns:
                df.insert(0, "dataset", ds)
            splits.append(df)
        if flow_path.exists():
            df = pd.read_csv(flow_path)
            if "dataset" not in df.columns:
                df.insert(0, "dataset", ds)
            flows.append(df)
    if samples:
        pd.concat(samples, ignore_index=True).to_csv(
            table_dir / "_SCHEMA_SAMPLE.csv", index=False
        )
    if splits:
        pd.concat(splits, ignore_index=True).to_csv(
            table_dir / "_SPLIT_COUNTS.csv", index=False
        )
    if flows:
        pd.concat(flows, ignore_index=True).to_csv(
            table_dir / "_EXCLUSION_FLOW.csv", index=False
        )


def main() -> int:
    cfg = UnifiedSchemaConfig.load()
    results = [validate_dataset(ds) for ds in DATASETS]
    write_llm_audit(results)
    overall = write_unified_report(results, cfg)
    write_unified_manifest(results, overall, cfg)
    write_combined_tables()
    print(overall)
    return 0 if overall == "SCHEMA_BUILD_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
