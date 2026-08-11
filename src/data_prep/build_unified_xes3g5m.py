#!/usr/bin/env python3
"""Build unified schema outputs for XES3G5M."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "data_prep"))

from unified_schema_common import (  # noqa: E402
    CONFIG_PATH,
    PROCESSED_ROOT,
    UnifiedSchemaConfig,
    assign_splits_to_interactions,
    build_hierarchy_edges_from_path,
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
from content_sufficiency import apply_classification  # noqa: E402

DATASET = "xes3g5m"
DATA_DIR = ROOT / "data" / "xes3g5m" / "XES3G5M"
OUT_DIR = PROCESSED_ROOT / DATASET
TABLE_DIR = ROOT / "results"
REPORT_DIR = ROOT / "reports" / "data_audits"


def load_questions() -> dict:
    path = DATA_DIR / "metadata" / "questions.json"
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def expand_interactions(questions: dict, cfg: UnifiedSchemaConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (retained_interactions, exclusion_flow)."""
    df = pd.read_csv(DATA_DIR / "question_level" / "train_valid_sequences_quelevel.csv")
    rows: list[dict] = []
    excluded_mask = 0
    excluded_no_content = 0

    for _, row in df.iterrows():
        uid = row["uid"]
        q_list = [int(x) for x in str(row["questions"]).split(",")]
        r_list = [int(x) for x in str(row["responses"]).split(",")]
        m_list = [int(x) for x in str(row["selectmasks"]).split(",")]
        ts_list = [int(x) for x in str(row["timestamps"]).split(",")]
        concepts_raw = str(row["concepts"]).split(",")
        concepts: list[int] = []
        for c in concepts_raw:
            first = c.split("_")[0]
            try:
                concepts.append(int(first))
            except ValueError:
                concepts.append(0)

        for pos, (q, r, m, ts, c) in enumerate(zip(q_list, r_list, m_list, ts_list, concepts)):
            if m != 1:
                excluded_mask += 1
                continue
            qdata = questions.get(str(q), {})
            content = str(qdata.get("content", "")).strip()
            if not content:
                excluded_no_content += 1
                continue
            rows.append({
                "student_id_raw": str(uid),
                "item_id_raw": str(q),
                "correct_raw": int(r),
                "timestamp_or_order": int(ts),
                "position_in_seq": pos,
                "concept_id_raw": str(c),
                "source_row_hash": sha256_text(f"{uid}|{q}|{ts}|{pos}|{r}|{c}"),
            })

    inter = pd.DataFrame(rows)
    eligible_items = set(inter["item_id_raw"].unique())
    sc = inter.groupby("student_id_raw").size()
    keep_students = set(sc[sc >= cfg.min_student_interactions].index)
    before = len(inter)
    inter = inter[inter["student_id_raw"].isin(keep_students)]
    excluded_students = before - len(inter)

    inter = inter.sort_values(
        ["student_id_raw", "timestamp_or_order", "position_in_seq", "item_id_raw"]
    )
    inter["sequence_index"] = inter.groupby("student_id_raw").cumcount()

    flow = pd.DataFrame([
        {"stage": "raw_sequence_rows", "count": len(df)},
        {"stage": "excluded_selectmask_0", "count": excluded_mask},
        {"stage": "excluded_no_content", "count": excluded_no_content},
        {"stage": "valid_interactions_with_content", "count": len(rows)},
        {"stage": "unique_items_in_valid_interactions", "count": len(eligible_items)},
        {"stage": "excluded_student_lt_min_interactions", "count": excluded_students},
        {"stage": "retained_interactions", "count": len(inter)},
        {"stage": "retained_students", "count": inter["student_id_raw"].nunique()},
    ])
    return inter, flow


def build_items(questions: dict, eligible_ids: set[str], cfg: UnifiedSchemaConfig) -> pd.DataFrame:
    rows = []
    for qid in sorted(eligible_ids, key=int):
        q = questions.get(str(qid), questions.get(int(qid), {}))
        content = str(q.get("content", "")).strip()
        qtype_raw = q.get("type", 0)
        if isinstance(qtype_raw, int):
            qtype = qtype_raw
        elif str(qtype_raw) in ("1", "multiple_choice"):
            qtype = 1
        else:
            qtype = 0
        options = q.get("options")
        item_format = "multiple_choice" if qtype == 1 else "fill_in"
        kc_routes = q.get("kc_routes") or []
        concept_ids = "|".join(str(x) for x in kc_routes) if kc_routes else ""
        primary = kc_routes[-1] if kc_routes else ""
        rows.append({
            "dataset": DATASET,
            "item_id_hash": hash_id(DATASET, "item", qid, cfg.hash_salt),
            "source_item_id_hash": hash_id(DATASET, "source_item", qid, cfg.hash_salt),
            "item_text_raw_reference": f"xes3g5m:metadata/questions.json:{qid}",
            "item_text_clean": content,
            "item_content_type": "plain_text_chinese",
            "language": "zh",
            "mathematical_domain": "k12_mathematics",
            "educational_level": "grade_3",
            "item_format": item_format,
            "answer_options": json.dumps(options, ensure_ascii=False) if options else None,
            "correct_answer_separate": str(q.get("answer", "")),
            "concept_ids": concept_ids,
            "primary_concept_id": primary,
            "has_image_dependency": False,
            "has_dynamic_template": False,
            "eligible_for_llm": True,
            "eligible_for_kt": True,
            "exclusion_reason": None,
            "source_content_hash": sha256_text(content),
        })
    return pd.DataFrame(rows)


def build_interactions_table(raw: pd.DataFrame, items: pd.DataFrame, cfg: UnifiedSchemaConfig) -> pd.DataFrame:
    concept_map = dict(zip(items["item_id_hash"], items["primary_concept_id"]))
    concept_ids_map = dict(zip(items["item_id_hash"], items["concept_ids"]))

    out = pd.DataFrame({
        "dataset": DATASET,
        "student_id_hash": raw["student_id_raw"].map(
            lambda s: hash_id(DATASET, "student", s, cfg.hash_salt)
        ),
        "item_id_hash": raw["item_id_raw"].map(
            lambda s: hash_id(DATASET, "item", s, cfg.hash_salt)
        ),
        "timestamp_or_order": raw["timestamp_or_order"].astype("int64"),
        "sequence_index": raw["sequence_index"].astype("int32"),
        "correct": raw["correct_raw"].astype("int8"),
        "attempt_index": 1,
        "first_attempt": True,
        "source_row_hash": raw["source_row_hash"],
    })
    out["interaction_id_hash"] = [
        hash_id(DATASET, "interaction", key, cfg.hash_salt)
        for key in (
            raw["student_id_raw"].astype(str) + "|"
            + raw["item_id_raw"].astype(str) + "|"
            + raw["timestamp_or_order"].astype(str) + "|"
            + raw["sequence_index"].astype(str)
        )
    ]
    out["concept_ids"] = out["item_id_hash"].map(concept_ids_map).fillna("")
    out["primary_concept_id"] = out["item_id_hash"].map(concept_map).fillna("")
    return out


def build_official_concept_edges(questions: dict, eligible_ids: set[str], cfg: UnifiedSchemaConfig) -> pd.DataFrame:
    seen: set[tuple[str, str]] = set()
    rows = []
    for qid in sorted(eligible_ids, key=int):
        q = questions.get(str(qid), questions.get(int(qid), {}))
        for route in q.get("kc_routes") or []:
            for edge in build_hierarchy_edges_from_path(str(route), DATASET, cfg):
                key = (edge["source_concept_id_hash"], edge["target_concept_id_hash"])
                if key in seen:
                    continue
                seen.add(key)
                rows.append(edge)
    return pd.DataFrame(rows)


def main() -> int:
    cfg = UnifiedSchemaConfig.load()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    questions = load_questions()
    source_hashes = {
        "questions.json": sha256_file(DATA_DIR / "metadata" / "questions.json"),
        "train_valid_sequences_quelevel.csv": sha256_file(
            DATA_DIR / "question_level" / "train_valid_sequences_quelevel.csv"
        ),
        "kc_routes_map.json": sha256_file(DATA_DIR / "metadata" / "kc_routes_map.json"),
    }

    raw_inter, exclusion_flow = expand_interactions(questions, cfg)
    eligible_ids = set(raw_inter["item_id_raw"].unique()) if len(raw_inter) else set()
    status = verify_eligible_count(DATASET, len(eligible_ids), cfg)

    if status == "SCHEMA_BUILD_COUNT_MISMATCH":
        blocker = REPORT_DIR / f"{DATASET}_SCHEMA_BUILD_BLOCKER.md"
        blocker.write_text(
            f"# XES3G5M Schema Build Blocker\n\n"
            f"**Status:** {status}\n\n"
            f"Expected eligible items: {cfg.frozen_eligible_item_counts[DATASET]}\n"
            f"Actual: {len(eligible_ids)}\n",
            encoding="utf-8",
        )
        exclusion_flow.to_csv(TABLE_DIR / f"{DATASET}_EXCLUSION_FLOW.csv", index=False)
        print(status, file=sys.stderr)
        return 1

    items = build_items(questions, eligible_ids, cfg)
    items = apply_classification(items, DATASET)
    interactions = build_interactions_table(raw_inter, items, cfg)
    splits = build_splits_parquet(DATASET, raw_inter["student_id_raw"].unique(), cfg)
    interactions = assign_splits_to_interactions(interactions, splits)

    official_edges = build_official_concept_edges(questions, eligible_ids, cfg)
    train_edges = build_response_cooccurrence_edges(interactions, DATASET, cfg)
    concept_edges = pd.concat([official_edges, train_edges], ignore_index=True)
    if len(concept_edges):
        concept_edges = concept_edges.sort_values(
            ["edge_source", "source_concept_id_hash", "target_concept_id_hash"],
            kind="mergesort",
        ).reset_index(drop=True)

    llm_items = make_llm_prompt_items(items[items["eligible_for_llm_scoring"]])

    paths = {
        "interactions": OUT_DIR / "interactions.parquet",
        "items": OUT_DIR / "items.parquet",
        "splits": OUT_DIR / "splits.parquet",
        "concept_edges": OUT_DIR / "concept_edges.parquet",
        "llm_prompt_items": OUT_DIR / "llm_prompt_items.parquet",
    }
    manifests = {}
    for name, path in paths.items():
        df = {
            "interactions": interactions,
            "items": items,
            "splits": splits,
            "concept_edges": concept_edges,
            "llm_prompt_items": llm_items,
        }[name]
        write_parquet(df, path)
        rec = file_record(path, source_hashes)
        rec["row_count"] = len(df)
        rec["columns"] = list(df.columns)
        rec["code_commit"] = git_commit()
        rec["config_hash"] = cfg.config_hash()
        rec["protocol_hash"] = protocol_hash()
        manifests[name] = rec

    items.head(5).to_csv(TABLE_DIR / f"{DATASET}_SCHEMA_SAMPLE.csv", index=False)
    split_counts = interactions.groupby("split_assignment").agg(
        interactions=("interaction_id_hash", "count"),
        students=("student_id_hash", "nunique"),
        items=("item_id_hash", "nunique"),
    ).reset_index()
    split_counts.to_csv(TABLE_DIR / f"{DATASET}_SPLIT_COUNTS.csv", index=False)
    exclusion_flow.to_csv(TABLE_DIR / f"{DATASET}_EXCLUSION_FLOW.csv", index=False)

    build_meta = {
        "dataset": DATASET,
        "status": status,
        "eligible_items": len(items),
        "students": splits["student_id_hash"].nunique(),
        "interactions": len(interactions),
        "concept_edges_official": len(official_edges),
        "concept_edges_train_cooc": len(train_edges),
        "outputs": manifests,
        "sequence_length_summary": sequence_length_summary(interactions).to_dict(orient="records")[0],
    }
    meta_path = OUT_DIR / "build_metadata.json"
    meta_path.write_text(json.dumps(build_meta, indent=2), encoding="utf-8")

    print(f"{DATASET}: {status}")
    print(f"  items={len(items)} interactions={len(interactions)} students={splits.shape[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
