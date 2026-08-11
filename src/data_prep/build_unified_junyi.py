#!/usr/bin/env python3
"""Build unified schema outputs for Junyi Academy (partition-rescan, low disk)."""

from __future__ import annotations

import gc
import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "data_prep"))

from reconcile_junyi_html_coverage import RAW_BASE, extract_question_text, extract_title  # noqa: E402
from content_sufficiency import apply_classification  # noqa: E402
from unified_schema_common import (  # noqa: E402
    PROCESSED_ROOT,
    UnifiedSchemaConfig,
    build_response_cooccurrence_edges,
    build_splits_parquet,
    file_record,
    git_commit,
    hash_id,
    make_llm_prompt_items,
    protocol_hash,
    sequence_length_summary,
    sha256_file,
    sha256_text,
    verify_eligible_count,
    write_parquet,
)

DATASET = "junyi"
LOG_CSV = ROOT / "data_raw" / "junyi" / "extracted" / "junyi_ProblemLog_original.csv"
EX_CSV = ROOT / "data_raw" / "junyi" / "extracted" / "junyi_Exercise_table.csv"
RECON_CSV = ROOT / "results" / "JUNYI_EXERCISE_HTML_RECONCILIATION.csv"
HTML_DIR = ROOT / "data_raw" / "junyi" / "exercises_html"
OUT_DIR = PROCESSED_ROOT / DATASET
TABLE_DIR = ROOT / "results"
REPORT_DIR = ROOT / "reports" / "data_audits"
CHUNK = 500_000
N_PARTITIONS = 64


def load_frozen_eligible_slugs() -> pd.DataFrame:
    recon = pd.read_csv(RECON_CSV)
    return recon[recon["eligible_for_llm"] == True].copy()  # noqa: E712


def build_items_from_reconciliation(eligible: pd.DataFrame, ex: pd.DataFrame, cfg: UnifiedSchemaConfig) -> pd.DataFrame:
    ex_meta = ex.set_index("name")
    rows = []
    for _, row in eligible.iterrows():
        slug = row["slug"]
        meta = ex_meta.loc[slug] if slug in ex_meta.index else None
        topic = str(meta["topic"]) if meta is not None else ""
        area = str(meta["area"]) if meta is not None else "mathematics"
        html_path = HTML_DIR / f"{slug}.html"
        html = html_path.read_text(encoding="utf-8", errors="replace") if html_path.exists() else ""
        stem = extract_question_text(html) or extract_title(html)
        rows.append({
            "dataset": DATASET,
            "item_id_hash": hash_id(DATASET, "item", slug, cfg.hash_salt),
            "source_item_id_hash": hash_id(DATASET, "source_item", slug, cfg.hash_salt),
            "item_text_raw_reference": f"{RAW_BASE}/{slug}.html",
            "item_text_clean": stem,
            "item_content_type": (
                "html_question_div" if row["stem_source"] == "question_div" else "html_title_fallback"
            ),
            "language": "zh",
            "mathematical_domain": area,
            "educational_level": "k12",
            "item_format": "fill_in_or_graphie",
            "answer_options": None,
            "correct_answer_separate": None,
            "concept_ids": topic,
            "primary_concept_id": topic,
            "has_image_dependency": bool(row.get("has_image_dep", False)),
            "has_dynamic_template": bool(row.get("has_dynamic_vars", False)),
            "graphie_only_no_question_text": bool(row.get("graphie_only_no_question_text", False)),
            "html_source_file_hash": row.get("html_sha256"),
            "slug_to_html_status": "one_to_one",
            "eligible_for_llm": True,
            "eligible_for_kt": True,
            "exclusion_reason": None,
            "source_content_hash": sha256_text(stem),
        })
    return pd.DataFrame(rows)


def student_counts(eligible_slugs: set[str], cfg: UnifiedSchemaConfig) -> tuple[dict[int, int], int, int]:
    counts: dict[int, int] = defaultdict(int)
    raw_total = kept = 0
    for chunk in pd.read_csv(LOG_CSV, usecols=["user_id", "exercise"], chunksize=CHUNK, low_memory=False):
        raw_total += len(chunk)
        chunk = chunk[chunk["exercise"].isin(eligible_slugs)]
        kept += len(chunk)
        for uid, n in chunk.groupby("user_id").size().items():
            counts[int(uid)] += int(n)
    return counts, raw_total, kept


def build_interactions_streaming(
    eligible_slugs: set[str],
    keep_users: set[int],
    items: pd.DataFrame,
    splits: pd.DataFrame,
    cfg: UnifiedSchemaConfig,
    out_path: Path,
) -> int:
    concept_map = dict(zip(items["item_id_hash"], items["primary_concept_id"]))
    concept_ids_map = dict(zip(items["item_id_hash"], items["concept_ids"]))
    split_map = dict(zip(splits["student_id_hash"], splits["split_assignment"]))
    usecols = ["user_id", "exercise", "correct", "time_done", "count_attempts", "hint_used", "time_taken"]
    writer = None
    total = 0

    for part in range(N_PARTITIONS):
        parts: list[pd.DataFrame] = []
        for chunk in pd.read_csv(LOG_CSV, usecols=usecols, chunksize=CHUNK, low_memory=False):
            chunk = chunk[chunk["user_id"] % N_PARTITIONS == part]
            chunk = chunk[chunk["exercise"].isin(eligible_slugs)]
            chunk = chunk[chunk["user_id"].isin(keep_users)]
            if not chunk.empty:
                parts.append(chunk)
        if not parts:
            continue
        df = pd.concat(parts, ignore_index=True)
        del parts
        gc.collect()
        df = df.sort_values(["user_id", "time_done", "exercise"])
        df["sequence_index"] = df.groupby("user_id").cumcount()
        sid_raw = df["user_id"].astype(str)
        item_raw = df["exercise"].astype(str)
        out = pd.DataFrame({
            "dataset": DATASET,
            "student_id_hash": sid_raw.map(lambda s: hash_id(DATASET, "student", s, cfg.hash_salt)),
            "item_id_hash": item_raw.map(lambda s: hash_id(DATASET, "item", s, cfg.hash_salt)),
            "timestamp_or_order": df["time_done"].astype("int64"),
            "sequence_index": df["sequence_index"].astype("int32"),
            "correct": df["correct"].astype("int8"),
            "attempt_index": df["count_attempts"].fillna(1).astype("int32"),
            "first_attempt": df["count_attempts"].fillna(1).astype("int32") == 1,
            "hint_used": df["hint_used"].fillna(0).astype(bool),
            "response_time": df["time_taken"],
        })
        out["interaction_id_hash"] = [
            hash_id(DATASET, "interaction", k, cfg.hash_salt)
            for k in (
                sid_raw + "|" + item_raw + "|"
                + df["time_done"].astype(str) + "|"
                + df["sequence_index"].astype(str)
            )
        ]
        out["source_row_hash"] = [
            sha256_text(f"{u}|{e}|{t}|{i}")
            for u, e, t, i in zip(sid_raw, item_raw, df["time_done"], df["sequence_index"])
        ]
        out["concept_ids"] = out["item_id_hash"].map(concept_ids_map).fillna("")
        out["primary_concept_id"] = out["item_id_hash"].map(concept_map).fillna("")
        out["split_assignment"] = out["student_id_hash"].map(split_map)
        table = pa.Table.from_pandas(out, preserve_index=False)
        writer = writer or pq.ParquetWriter(
            out_path, table.schema, compression="zstd", compression_level=3
        )
        writer.write_table(table)
        total += len(out)
        del df, out, table
        gc.collect()
        print(f"  partition {part + 1}/{N_PARTITIONS} done, total rows={total}", flush=True)

    if writer:
        writer.close()
    return total


def build_official_concept_edges(items: pd.DataFrame, eligible_df: pd.DataFrame, ex: pd.DataFrame, cfg: UnifiedSchemaConfig) -> pd.DataFrame:
    slug_by_hash = {hash_id(DATASET, "item", slug, cfg.hash_salt): slug for slug in eligible_df["slug"]}
    ex_meta = ex.set_index("name")
    rows = []
    seen: set[tuple[str, str]] = set()
    for ihash in items["item_id_hash"]:
        slug = slug_by_hash.get(ihash)
        if not slug or slug not in ex_meta.index:
            continue
        meta = ex_meta.loc[slug]
        area, topic = str(meta["area"]), str(meta["topic"])
        if area and topic and area != topic:
            src_h = hash_id(DATASET, "concept", area, cfg.hash_salt)
            tgt_h = hash_id(DATASET, "concept", topic, cfg.hash_salt)
            if (src_h, tgt_h) not in seen:
                seen.add((src_h, tgt_h))
                rows.append({
                    "dataset": DATASET,
                    "source_concept_id_hash": src_h,
                    "target_concept_id_hash": tgt_h,
                    "edge_type": "area_topic_hierarchy",
                    "edge_weight": 1.0,
                    "edge_source": "official_metadata",
                    "permitted_split": "all",
                    "source_hash": sha256_text(f"{area}|{topic}"),
                })
    return pd.DataFrame(rows)


def main() -> int:
    cfg = UnifiedSchemaConfig.load()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    eligible_df = load_frozen_eligible_slugs()
    eligible_slugs = set(eligible_df["slug"].tolist())
    status = verify_eligible_count(DATASET, len(eligible_slugs), cfg)
    if status == "SCHEMA_BUILD_COUNT_MISMATCH":
        (REPORT_DIR / f"{DATASET}_SCHEMA_BUILD_BLOCKER.md").write_text(
            f"Expected {cfg.frozen_eligible_item_counts[DATASET]}, got {len(eligible_slugs)}\n",
            encoding="utf-8",
        )
        return 1

    ex = pd.read_csv(EX_CSV, low_memory=False)
    items = build_items_from_reconciliation(eligible_df, ex, cfg)
    items = apply_classification(items, DATASET)
    source_hashes = {
        "junyi_ProblemLog_original.csv": sha256_file(LOG_CSV),
        "junyi_Exercise_table.csv": sha256_file(EX_CSV),
        "JUNYI_EXERCISE_HTML_RECONCILIATION.csv": sha256_file(RECON_CSV),
    }

    counts, raw_total, kept_on_eligible = student_counts(eligible_slugs, cfg)
    keep_users = {u for u, n in counts.items() if n >= cfg.min_student_interactions}
    retained = sum(counts[u] for u in keep_users)
    exclusion_flow = pd.DataFrame([
        {"stage": "raw_interactions", "count": raw_total},
        {"stage": "interactions_on_eligible_items", "count": kept_on_eligible},
        {"stage": "retained_interactions", "count": retained},
        {"stage": "retained_students", "count": len(keep_users)},
    ])

    splits = build_splits_parquet(DATASET, {str(u) for u in keep_users}, cfg)
    interactions_path = OUT_DIR / "interactions.parquet"
    print("Building interactions (partition-rescan)...", flush=True)
    n_inter = build_interactions_streaming(
        eligible_slugs, keep_users, items, splits, cfg, interactions_path
    )

    official_edges = build_official_concept_edges(items, eligible_df, ex, cfg)
    train_edges = build_response_cooccurrence_edges(interactions_path, DATASET, cfg)
    concept_edges = pd.concat([official_edges, train_edges], ignore_index=True)
    if len(concept_edges):
        concept_edges = concept_edges.sort_values(
            ["edge_source", "source_concept_id_hash", "target_concept_id_hash"],
            kind="mergesort",
        ).reset_index(drop=True)
    llm_items = make_llm_prompt_items(items[items["eligible_for_llm_scoring"]])

    write_parquet(items, OUT_DIR / "items.parquet")
    write_parquet(splits, OUT_DIR / "splits.parquet")
    write_parquet(concept_edges, OUT_DIR / "concept_edges.parquet")
    write_parquet(llm_items, OUT_DIR / "llm_prompt_items.parquet")

    interactions = pd.read_parquet(
        interactions_path,
        columns=["split_assignment", "student_id_hash", "item_id_hash", "interaction_id_hash"],
    )

    manifests = {}
    for name, path in (
        ("interactions", interactions_path),
        ("items", OUT_DIR / "items.parquet"),
        ("splits", OUT_DIR / "splits.parquet"),
        ("concept_edges", OUT_DIR / "concept_edges.parquet"),
        ("llm_prompt_items", OUT_DIR / "llm_prompt_items.parquet"),
    ):
        rec = file_record(path, source_hashes)
        rec["row_count"] = n_inter if name == "interactions" else pq.ParquetFile(path).metadata.num_rows
        rec["columns"] = list(pq.ParquetFile(path).schema_arrow.names)
        rec["code_commit"] = git_commit()
        rec["config_hash"] = cfg.config_hash()
        rec["protocol_hash"] = protocol_hash()
        manifests[name] = rec

    items.head(5).to_csv(TABLE_DIR / f"{DATASET}_SCHEMA_SAMPLE.csv", index=False)
    interactions.groupby("split_assignment").agg(
        interactions=("interaction_id_hash", "count"),
        students=("student_id_hash", "nunique"),
        items=("item_id_hash", "nunique"),
    ).reset_index().to_csv(TABLE_DIR / f"{DATASET}_SPLIT_COUNTS.csv", index=False)
    exclusion_flow.to_csv(TABLE_DIR / f"{DATASET}_EXCLUSION_FLOW.csv", index=False)

    seq_df = pd.read_parquet(interactions_path, columns=["student_id_hash"])
    build_meta = {
        "dataset": DATASET,
        "status": status,
        "eligible_items": len(items),
        "students": splits["student_id_hash"].nunique(),
        "interactions": n_inter,
        "raw_interactions": raw_total,
        "response_coverage_pct": round(n_inter / raw_total * 100, 2),
        "extraction_question_div": int((items["item_content_type"] == "html_question_div").sum()),
        "extraction_title_fallback": int((items["item_content_type"] == "html_title_fallback").sum()),
        "has_image_dependency": int(items["has_image_dependency"].sum()),
        "has_dynamic_template": int(items["has_dynamic_template"].sum()),
        "outputs": manifests,
        "sequence_length_summary": sequence_length_summary(seq_df).to_dict(orient="records")[0],
    }
    (OUT_DIR / "build_metadata.json").write_text(json.dumps(build_meta, indent=2), encoding="utf-8")

    print(f"{DATASET}: {status}")
    print(f"  items={len(items)} interactions={n_inter} students={splits.shape[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
