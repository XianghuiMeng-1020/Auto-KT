"""Tests for Amendment 009 content sufficiency and shared confirmatory universe."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "data_prep"))
sys.path.insert(0, str(ROOT / "src" / "llm_scoring"))

from content_sufficiency import (  # noqa: E402
    CONTENT_SUFFICIENCY_RULE_VERSION,
    PASS_REASON,
    PRIMARY_EXCLUSION_REASONS,
    amendment_009_hash,
    apply_classification,
    classification_audit_hash,
    classify_item,
)
from unified_schema_common import LLM_PROMPT_DENYLIST, PROCESSED_ROOT, make_llm_prompt_items  # noqa: E402

TABLE_DIR = ROOT / "results"
MANIFEST_PATH = ROOT / "data_manifests" / "_manifest.json"


@pytest.fixture(scope="module")
def xes_items():
    path = PROCESSED_ROOT / "xes3g5m" / "items.parquet"
    if not path.exists():
        pytest.skip("xes items not built")
    return pd.read_parquet(path)


@pytest.fixture(scope="module")
def junyi_items():
    path = PROCESSED_ROOT / "junyi" / "items.parquet"
    if not path.exists():
        pytest.skip("junyi items not built")
    return pd.read_parquet(path)


def test_exclusion_vocabulary():
    assert PASS_REASON in PRIMARY_EXCLUSION_REASONS
    assert "EXCLUDE_REQUIRED_IMAGE" in PRIMARY_EXCLUSION_REASONS
    assert len(PRIMARY_EXCLUSION_REASONS) == 13


def test_classify_item_returns_controlled_reason(xes_items):
    row = xes_items.iloc[0]
    out = classify_item(row, "xes3g5m")
    assert out["llm_exclusion_primary_reason"] in PRIMARY_EXCLUSION_REASONS
    assert out["content_sufficiency_rule_version"] == CONTENT_SUFFICIENCY_RULE_VERSION


def test_classifier_does_not_read_llm_outputs():
    text = Path(ROOT / "scripts" / "data" / "content_sufficiency.py").read_text(encoding="utf-8")
    for term in ("scalar_difficulty", "raw_response", "cache_index", "parse_valid"):
        assert term not in text


def test_classifier_does_not_read_outcomes():
    text = Path(ROOT / "scripts" / "data" / "content_sufficiency.py").read_text(encoding="utf-8")
    for term in ("interactions.parquet", "correctness", "error_rate", "empirical_difficulty"):
        assert term not in text


def test_no_response_counts_in_inclusion_decision():
    text = Path(ROOT / "scripts" / "data" / "content_sufficiency.py").read_text(encoding="utf-8")
    assert "response_count" not in text
    assert "train_response_count" not in text


def test_prompt_exports_only_scoreable(xes_items, junyi_items):
    for dataset, items in (("xes3g5m", xes_items), ("junyi", junyi_items)):
        if "eligible_for_llm_scoring" not in items.columns:
            pytest.skip("content sufficiency not applied")
        path = PROCESSED_ROOT / dataset / "llm_prompt_items.parquet"
        if not path.exists():
            pytest.skip("llm export missing")
        prompt = pd.read_parquet(path)
        scoreable = set(items.loc[items["eligible_for_llm_scoring"], "item_id_hash"])
        assert set(prompt["item_id_hash"]).issubset(scoreable)
        assert len(prompt) == len(scoreable)


def test_kt_tables_retain_non_scoreable_items(xes_items, junyi_items):
    for items in (xes_items, junyi_items):
        if "eligible_for_llm_scoring" not in items.columns:
            pytest.skip("content sufficiency not applied")
        assert items["eligible_for_kt"].all()
        assert (~items["eligible_for_llm_scoring"]).any() or len(items) < 200


def test_shared_confirmatory_matches_llm_scoreable(xes_items):
    if "eligible_for_shared_confirmatory" not in xes_items.columns:
        pytest.skip("not applied")
    assert (
        xes_items["eligible_for_shared_confirmatory"] == xes_items["eligible_for_llm_scoring"]
    ).all()


def test_invented_content_pilot_items_excluded():
    review_path = TABLE_DIR / "LLM_PILOT_MANUAL_REVIEW.csv"
    score_path = TABLE_DIR / "JUNYI_LLM_SCOREABILITY.csv"
    if not review_path.exists() or not score_path.exists():
        pytest.skip("audit outputs missing")
    review = pd.read_csv(review_path)
    score = pd.read_csv(score_path).set_index("item_id_hash")
    invented = review[review["review_label"] == "INVENTED_CONTENT"]["item_id_hash"].unique()
    for iid in invented:
        assert not bool(score.loc[iid, "eligible_for_llm_scoring"])


def test_prompt_denylist_still_respected(xes_items):
    scoreable = xes_items[xes_items["eligible_for_llm_scoring"]]
    export = make_llm_prompt_items(scoreable)
    assert LLM_PROMPT_DENYLIST.isdisjoint(export.columns)


def test_repeated_audit_identical_hash(xes_items):
    a = classification_audit_hash(apply_classification(xes_items, "xes3g5m"))
    b = classification_audit_hash(apply_classification(xes_items, "xes3g5m"))
    assert a == b


def test_amendment_009_hash_in_manifest():
    if not MANIFEST_PATH.exists():
        pytest.skip("manifest missing")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    expected = amendment_009_hash()
    if not expected:
        pytest.skip("amendment not committed")
    assert manifest.get("amendment_009_hash") == expected


def test_gate_state_pass_implies_full_ready():
    if not MANIFEST_PATH.exists():
        pytest.skip("manifest missing")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    status = manifest.get("gate_status", {}).get("content_eligibility_status")
    pilot = manifest.get("gate_status", {}).get("llm_pilot_status")
    full = manifest.get("gate_status", {}).get("full_llm_scoring_ready")
    if status == "CONTENT_ELIGIBILITY_PASS":
        assert pilot == "LLM_PILOT_PASS"
        assert full is True
    if pilot == "LLM_PILOT_CONDITIONAL":
        assert full is False


def test_llm_conditions_share_item_universe(xes_items, junyi_items):
    """Scalar and confirmatory universes must be identical per dataset."""
    for items in (xes_items, junyi_items):
        if "eligible_for_shared_confirmatory" not in items.columns:
            pytest.skip("not applied")
        assert (
            items["eligible_for_shared_confirmatory"] == items["eligible_for_llm_scoring"]
        ).all()
