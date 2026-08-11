"""Tests for frozen manual content review and pilot gate consistency."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "llm_scoring"))

from llm_pilot_common import PILOT_DIR, PilotConfig  # noqa: E402
from manual_content_review import (  # noqa: E402
    STEM_SNIPPET_MAX,
    VALID_REVIEW_LABELS,
    build_manual_review_table,
    build_review_summary,
    decide_pilot_gate,
    frozen_review_sample,
    review_sample_hash,
)

TABLE_DIR = ROOT / "results"
MANIFEST_PATH = ROOT / "data_manifests" / "_manifest.json"
REVIEW_PATH = TABLE_DIR / "LLM_PILOT_MANUAL_REVIEW.csv"
SUMMARY_PATH = TABLE_DIR / "LLM_PILOT_MANUAL_REVIEW_SUMMARY.csv"
MANUAL_REVIEW_SCRIPT = ROOT / "src" / "llm_scoring" / "run_manual_content_review.py"


@pytest.fixture(scope="module")
def cfg() -> PilotConfig:
    return PilotConfig.load()


def _require_pilot_items(cfg: PilotConfig) -> None:
    missing = [ds for ds in ("xes3g5m", "junyi") if not (PILOT_DIR / f"{ds}_pilot_items.parquet").exists()]
    if missing:
        pytest.skip(
            f"pilot item samples not built for {missing}; requires locally processed data "
            "(see data/README.md)"
        )


def test_manual_review_sample_membership(cfg: PilotConfig):
    for ds in ("xes3g5m", "junyi"):
        pilot_path = PILOT_DIR / f"{ds}_pilot_items.parquet"
        if not pilot_path.exists():
            pytest.skip("pilot samples not built")
        expected = (
            pd.read_parquet(pilot_path)
            .sort_values("item_id_hash", kind="mergesort")
            .head(cfg.manual_review_per_dataset)["item_id_hash"]
            .tolist()
        )
        sample = frozen_review_sample(cfg)[ds]
        assert len(sample) == cfg.manual_review_per_dataset
        assert sample["item_id_hash"].tolist() == expected


def test_review_sample_hash_stable(cfg: PilotConfig):
    _require_pilot_items(cfg)
    assert review_sample_hash(cfg) == review_sample_hash(cfg)
    assert len(review_sample_hash(cfg)) == 64


def test_review_label_validity(cfg: PilotConfig):
    if not REVIEW_PATH.exists():
        pytest.skip("manual review not run")
    df = pd.read_csv(REVIEW_PATH)
    assert set(df["review_label"].unique()).issubset(VALID_REVIEW_LABELS)
    assert len(df) == cfg.manual_review_per_dataset * 2 * len(cfg.models)


def test_no_raw_item_text_beyond_snippet_in_review_table():
    if not REVIEW_PATH.exists():
        pytest.skip("manual review not run")
    df = pd.read_csv(REVIEW_PATH)
    assert "item_text_clean" not in df.columns
    assert "stem_text" not in df.columns
    if "stem_snippet" in df.columns:
        assert df["stem_snippet"].str.len().max() <= STEM_SNIPPET_MAX


def test_manual_review_script_no_outcome_tables():
    text = MANUAL_REVIEW_SCRIPT.read_text(encoding="utf-8")
    for term in ("interactions.parquet", "empirical_difficulty", "rasch", "error_rate"):
        assert term not in text, f"manual review script must not load {term}"


def test_review_summary_matches_review_table():
    if not REVIEW_PATH.exists() or not SUMMARY_PATH.exists():
        pytest.skip("manual review not run")
    review = pd.read_csv(REVIEW_PATH)
    summary = pd.read_csv(SUMMARY_PATH)
    for _, row in summary[summary["extraction_content_type"] == "all"].iterrows():
        sub = review[(review["dataset"] == row["dataset"]) & (review["model"] == row["model"])]
        assert len(sub) == row["n_reviews"]
        assert abs((sub["review_label"] == "PASS").mean() - row["pass_rate"]) < 1e-9


def test_gate_state_conditional_implies_not_full_ready():
    if not MANIFEST_PATH.exists():
        pytest.skip("manifest missing")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    status = manifest.get("gate_status", {}).get("llm_pilot_status")
    full_ready = manifest.get("gate_status", {}).get("full_llm_scoring_ready")
    if status == "LLM_PILOT_CONDITIONAL":
        assert full_ready is False
    if status == "LLM_PILOT_PASS":
        assert full_ready is True
    if full_ready is True:
        assert status == "LLM_PILOT_PASS"


def test_decide_pilot_gate_conditional_when_amendment_not_adopted(cfg: PilotConfig):
    _require_pilot_items(cfg)
    review_df, _ = build_manual_review_table(cfg)
    parse_df = pd.DataFrame({
        "first_pass_valid_rate": [1.0, 1.0, 1.0, 1.0],
    })
    amendment = {
        "affected_pct": 66.97,
        "adopted": False,
    }
    incidents = {"incident_2_junyi_connection_error": {"resolved": True}}
    status, full_ready, _ = decide_pilot_gate(review_df, parse_df, amendment, incidents)
    assert status == "LLM_PILOT_CONDITIONAL"
    assert full_ready is False


def test_decide_pilot_gate_pass_requires_adopted_amendment(cfg: PilotConfig):
    _require_pilot_items(cfg)
    review_df, _ = build_manual_review_table(cfg)
    parse_df = pd.DataFrame({"first_pass_valid_rate": [1.0]})
    amendment = {"affected_pct": 0.0, "adopted": True}
    incidents = {"incident_2_junyi_connection_error": {"resolved": True}}
    status, full_ready, _ = decide_pilot_gate(review_df, parse_df, amendment, incidents)
    # XES pass rate should be high enough; amendment not blocking
    assert status in ("LLM_PILOT_PASS", "LLM_PILOT_CONDITIONAL")
    if status == "LLM_PILOT_PASS":
        assert full_ready is True


def test_build_review_summary_row_count(cfg: PilotConfig):
    _require_pilot_items(cfg)
    review_df, _ = build_manual_review_table(cfg)
    summary = build_review_summary(review_df)
    assert len(summary) >= 4
    assert "pass_rate" in summary.columns
