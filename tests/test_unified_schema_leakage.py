"""Leakage and schema guards for unified XES3G5M and Junyi outputs."""

from __future__ import annotations

import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "data_prep"))

from unified_schema_common import (  # noqa: E402
    LLM_PROMPT_ALLOWLIST,
    LLM_PROMPT_DENYLIST,
    PROCESSED_ROOT,
    UnifiedSchemaConfig,
    sha256_file,
)

DATASETS = ("xes3g5m", "junyi")
FROZEN = {"xes3g5m": 7618, "junyi": 666}


def _require_built(dataset: str) -> Path:
    d = PROCESSED_ROOT / dataset
    if not (d / "interactions.parquet").exists():
        pytest.skip(f"{dataset} unified schema not built")
    return d


def _read_parquet(dataset: str, name: str) -> pd.DataFrame:
    return pd.read_parquet(_require_built(dataset) / f"{name}.parquet")


def _parquet_columns(dataset: str, name: str) -> set[str]:
    pf = pq.ParquetFile(_require_built(dataset) / f"{name}.parquet")
    return set(pf.schema_arrow.names)


@pytest.mark.parametrize("dataset", DATASETS)
def test_no_raw_student_ids_in_outputs(dataset):
    banned = {"user_id", "uid", "student_id", "exercise", "question_id", "item_id"}
    d = _require_built(dataset)
    for name in ("items", "splits", "concept_edges", "llm_prompt_items"):
        assert banned.isdisjoint(_parquet_columns(dataset, name)), f"{dataset}/{name}"
    assert banned.isdisjoint(_parquet_columns(dataset, "interactions"))


@pytest.mark.parametrize("dataset", DATASETS)
def test_no_student_overlap_across_splits(dataset):
    splits = _read_parquet(dataset, "splits")
    train = set(splits.loc[splits["split_assignment"] == "train", "student_id_hash"])
    val = set(splits.loc[splits["split_assignment"] == "val", "student_id_hash"])
    test = set(splits.loc[splits["split_assignment"] == "test", "student_id_hash"])
    assert not (train & val or train & test or val & test)


@pytest.mark.parametrize("dataset", DATASETS)
def test_interaction_splits_match_splits_table(dataset):
    d = _require_built(dataset)
    splits = pd.read_parquet(d / "splits.parquet")
    mapping = dict(zip(splits["student_id_hash"], splits["split_assignment"]))
    pf = pq.ParquetFile(d / "interactions.parquet")
    for batch in pf.iter_batches(
        batch_size=500_000, columns=["student_id_hash", "split_assignment"]
    ):
        df = batch.to_pandas()
        expected = df["student_id_hash"].map(mapping)
        assert (expected == df["split_assignment"]).all()


@pytest.mark.parametrize("dataset", DATASETS)
def test_no_test_response_in_response_derived_graphs(dataset):
    edges = _read_parquet(dataset, "concept_edges")
    train_e = edges[edges["edge_source"] == "training_response_cooccurrence"]
    if len(train_e):
        assert (train_e["permitted_split"] == "train").all()


@pytest.mark.parametrize("dataset", DATASETS)
def test_no_correct_answer_in_llm_prompt_items(dataset):
    llm = _read_parquet(dataset, "llm_prompt_items")
    assert "correct_answer_separate" not in llm.columns
    assert "answer_options" not in llm.columns


@pytest.mark.parametrize("dataset", DATASETS)
def test_no_correctness_in_llm_prompt_items(dataset):
    llm = _read_parquet(dataset, "llm_prompt_items")
    assert "correct" not in llm.columns
    assert "correctness" not in llm.columns


@pytest.mark.parametrize("dataset", DATASETS)
def test_no_response_derived_fields_in_llm_prompt_items(dataset):
    llm = _read_parquet(dataset, "llm_prompt_items")
    forbidden = {
        "response_count", "exposure_count", "empirical_difficulty",
        "error_rate", "hint_used", "answer_viewed",
    }
    assert forbidden.isdisjoint(llm.columns)


@pytest.mark.parametrize("dataset", DATASETS)
def test_eligible_item_counts(dataset):
    items = _read_parquet(dataset, "items")
    kt_eligible = items[items["eligible_for_kt"] == True]  # noqa: E712
    assert len(kt_eligible) == FROZEN[dataset]
    if "eligible_for_llm_scoring" in items.columns:
        llm_eligible = items[items["eligible_for_llm_scoring"] == True]  # noqa: E712
        assert len(llm_eligible) >= 100
        assert len(llm_eligible) < len(kt_eligible)


@pytest.mark.parametrize("dataset", DATASETS)
def test_every_interaction_joins_eligible_item(dataset):
    d = _require_built(dataset)
    items = pd.read_parquet(d / "items.parquet")
    eligible = set(items.loc[items["eligible_for_kt"], "item_id_hash"])
    pf = pq.ParquetFile(d / "interactions.parquet")
    for batch in pf.iter_batches(batch_size=500_000, columns=["item_id_hash"]):
        assert batch.to_pandas()["item_id_hash"].isin(eligible).all()


@pytest.mark.parametrize("dataset", DATASETS)
def test_every_item_has_source_content_hash(dataset):
    items = _read_parquet(dataset, "items")
    assert items["source_content_hash"].notna().all()
    assert (items["source_content_hash"].astype(str).str.len() == 64).all()


@pytest.mark.parametrize("dataset", DATASETS)
def test_sequence_indices_strictly_increasing_within_student(dataset):
    inter_path = _require_built(dataset) / "interactions.parquet"
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
        assert st["min"] == 0
        assert st["max"] + 1 == st["cnt"]


@pytest.mark.parametrize("dataset", DATASETS)
def test_concept_edge_split_provenance(dataset):
    edges = _read_parquet(dataset, "concept_edges")
    train_e = edges[edges["edge_source"] == "training_response_cooccurrence"]
    if len(train_e):
        assert (train_e["permitted_split"] == "train").all()
    official = edges[edges["edge_source"] == "official_metadata"]
    if len(official):
        assert (official["permitted_split"] == "all").all()


def test_junyi_slug_html_one_to_one():
    items = _read_parquet("junyi", "items")
    if "slug_to_html_status" in items.columns:
        assert (items["slug_to_html_status"] == "one_to_one").all()


def test_junyi_no_excluded_non_math_items():
    items = _read_parquet("junyi", "items")
    domains = items["mathematical_domain"].astype(str).str.lower()
    banned = {"biology", "logics", "history", "chemistry", "physics"}
    assert not domains.isin(banned).any()


def test_llm_prompt_allowlist_only():
    for dataset in DATASETS:
        llm = _read_parquet(dataset, "llm_prompt_items")
        assert set(llm.columns) == set(LLM_PROMPT_ALLOWLIST)
        assert set(llm.columns).isdisjoint(LLM_PROMPT_DENYLIST)


def test_repeated_build_identical_hashes():
    """Rebuild XES3G5M and compare file hashes."""
    d = PROCESSED_ROOT / "xes3g5m"
    if not (d / "interactions.parquet").exists():
        pytest.skip("xes3g5m not built")
    files = (
        "interactions.parquet", "items.parquet", "splits.parquet",
        "concept_edges.parquet", "llm_prompt_items.parquet",
    )
    before = {f: sha256_file(d / f) for f in files}
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "data" / "build_unified_xes3g5m.py")],
        check=True,
        cwd=ROOT,
    )
    after = {f: sha256_file(d / f) for f in files}
    assert before == after


def test_unified_manifest_exists_after_validation():
    path = ROOT / "data_manifests" / "_unified_schema_manifest.json"
    if not path.exists():
        pytest.skip("run validate_unified_schema.py first")
    data = json.loads(path.read_text())
    assert data["overall_status"] == "SCHEMA_BUILD_PASS"


def test_split_seed_frozen():
    cfg = UnifiedSchemaConfig.load()
    assert cfg.split_seed == 2024
