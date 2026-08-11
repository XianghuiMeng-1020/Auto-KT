"""Guards for Phase E full LLM scoring."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "llm_scoring"))

from full_llm_common import (  # noqa: E402
    CACHE_DIR,
    DATASETS,
    FAILURE_CLASSES,
    FullScoringConfig,
    PARSED_DIR,
    amendment_009_hash,
    build_request_plan,
    classify_api_error,
    full_cache_key,
    is_retryable,
    load_cache_index,
    prompt_hash_for_item,
)
from llm_pilot_common import PROMPT_DENYLIST, PROCESSED_ROOT, render_messages  # noqa: E402

FULL_SCRIPTS = [
    ROOT / "src" / "llm_scoring" / "run_full_scoring.py",
    ROOT / "src" / "llm_scoring" / "validate_full_scoring.py",
    ROOT / "src" / "llm_scoring" / "build_llm_feature_tables.py",
    ROOT / "src" / "llm_scoring" / "full_llm_common.py",
]


@pytest.fixture(scope="module")
def cfg() -> FullScoringConfig:
    return FullScoringConfig.load()


def _require_processed(*datasets: str) -> None:
    missing = [ds for ds in datasets if not (PROCESSED_ROOT / ds / "items.parquet").exists()]
    if missing:
        pytest.skip(
            f"processed items not built for {missing}; run src/data_prep/build_unified_*.py "
            "against locally obtained raw data first (see data/README.md)"
        )


def test_expected_item_model_pair_count(cfg: FullScoringConfig):
    _require_processed(*DATASETS)
    plan = build_request_plan(cfg)
    assert len(plan) == 11106
    assert sum(1 for r in plan if r["dataset"] == "xes3g5m") == 5363 * 2
    assert sum(1 for r in plan if r["dataset"] == "junyi") == 190 * 2


def test_no_excluded_items_in_plan(cfg: FullScoringConfig):
    _require_processed(*DATASETS)
    for ds in DATASETS:
        items = pd.read_parquet(PROCESSED_ROOT / ds / "items.parquet")
        excluded = set(items.loc[~items["eligible_for_llm_scoring"], "item_id_hash"])
        plan_ids = {r["item_id_hash"] for r in build_request_plan(cfg) if r["dataset"] == ds}
        assert plan_ids.isdisjoint(excluded)


def test_cache_key_deterministic(cfg: FullScoringConfig):
    kwargs = dict(
        cfg=cfg,
        model="gpt-4o-mini",
        dataset="xes3g5m",
        item_id_hash="abc",
        source_content_hash="def",
        prompt_hash="ghi",
    )
    assert full_cache_key(**kwargs) == full_cache_key(**kwargs)


def test_retry_limits_in_config(cfg: FullScoringConfig):
    assert cfg.max_retries_transient <= 2
    assert cfg.max_retries_format <= 1


def test_no_outcome_tables_in_full_scoring_scripts():
    banned = ("interactions.parquet", "empirical_difficulty", "rasch", "error_rate", "correctness")
    for path in FULL_SCRIPTS:
        text = path.read_text(encoding="utf-8")
        for term in banned:
            assert term not in text, f"{path.name} references {term}"


def test_no_answer_in_prompt_payload(cfg: FullScoringConfig):
    _require_processed(*DATASETS)
    plan = build_request_plan(cfg)[:5]
    for req in plan:
        blob = json.dumps(render_messages(req["stem_text"]))
        for term in ("correct answer", "error rate", "response count"):
            assert term not in blob.lower()


def test_failure_classification():
    assert classify_api_error("APIConnectionError('Connection error.')") == "transient_connection"
    assert is_retryable("rate_limit")


def test_model_specific_token_param_in_runner():
    text = (ROOT / "src" / "llm_scoring" / "full_llm_common.py").read_text(encoding="utf-8")
    assert "max_completion_tokens" in text
    assert "max_tokens" in text


def test_one_primary_score_per_item_model(cfg: FullScoringConfig):
    path = PARSED_DIR / "all_llm_item_features.parquet"
    if not path.exists():
        pytest.skip("feature tables not built")
    df = pd.read_parquet(path)
    dup = df.duplicated(subset=["dataset", "item_id_hash", "model_identifier"]).sum()
    assert dup == 0


def test_cross_model_universe_identity(cfg: FullScoringConfig):
    path = PARSED_DIR / "all_llm_item_features.parquet"
    if not path.exists():
        pytest.skip("feature tables not built")
    df = pd.read_parquet(path)
    for ds in DATASETS:
        pivot = df[df["dataset"] == ds].pivot_table(
            index="item_id_hash", columns="model_identifier", values="scalar_difficulty", aggfunc="count"
        )
        assert pivot.notna().all(axis=None) or len(pivot) == cfg.scoreable_counts[ds]


def test_scalar_in_unit_interval(cfg: FullScoringConfig):
    path = PARSED_DIR / "all_llm_item_features.parquet"
    if not path.exists():
        pytest.skip("feature tables not built")
    df = pd.read_parquet(path)
    vals = pd.to_numeric(df["scalar_difficulty"], errors="coerce")
    assert ((vals >= 0) & (vals <= 1)).all()


def test_no_api_key_in_cache():
    cache_path = CACHE_DIR / "cache_index.json"
    if not cache_path.exists():
        pytest.skip("cache not built")
    text = cache_path.read_text(encoding="utf-8")
    assert "sk-" not in text


def test_paid_call_reconciliation_accounting():
    summary_path = ROOT / "artifacts" / "reports" / "FULL_LLM_VALIDATION_SUMMARY.json"
    if not summary_path.exists():
        pytest.skip("validation summary not built")
    recon = json.loads(summary_path.read_text(encoding="utf-8")).get("reconciliation")
    if not recon:
        pytest.skip("reconciliation not run")
    assert recon["expected_primary_scores"] == 11106
    assert recon["observed_primary_scores"] == 11106
    assert recon["missing_primary_scores"] == 0
    assert recon["duplicate_primary_scores"] == 0
    assert recon["pilot_cache_reuse"] == 208
    assert recon["unique_new_primary_item_model_calls"] == 10898
    assert recon["additional_paid_retry_calls"] == 1
    assert recon["total_paid_api_calls"] == 10899
    assert recon["excess_call_reason"] == "format_retry"
    assert recon["reconciliation_pass"] is True


def test_gate_pass_implies_complete():
    manifest_path = ROOT / "data_manifests" / "_manifest.json"
    if not manifest_path.exists():
        pytest.skip("no manifest")
    m = json.loads(manifest_path.read_text(encoding="utf-8"))
    status = m.get("gate_status", {}).get("full_llm_scoring_status")
    complete = m.get("gate_status", {}).get("full_llm_scoring_complete")
    if status == "FULL_LLM_SCORING_PASS":
        assert complete is True


def test_full_scoring_manifest_complete():
    path = ROOT / "data_manifests" / "full_llm_scoring_manifest.json"
    if not path.exists():
        pytest.skip("manifest not frozen")
    m = json.loads(path.read_text(encoding="utf-8"))
    for key in ("prompt_hash", "amendment_009_hash", "amendment_010_hash", "output_table_hashes"):
        assert key in m
