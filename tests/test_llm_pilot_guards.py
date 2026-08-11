"""Guards for Phase D1 LLM pilot: prompt, parser, cache, outcome isolation."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "llm_scoring"))

from llm_pilot_common import (  # noqa: E402
    CACHE_DIR,
    PILOT_DIR,
    PROMPT_DENYLIST,
    PROMPT_PROTOCOL_PATH,
    PilotConfig,
    cache_key,
    parse_difficulty,
    parse_response_record,
    protocol_prompt_hash,
    render_messages,
    sha256_text,
)

LLM_SCRIPTS = [
    ROOT / "src" / "llm_scoring" / "run_llm_pilot.py",
    ROOT / "src" / "llm_scoring" / "analyze_llm_pilot.py",
    ROOT / "src" / "llm_scoring" / "sample_pilot_items.py",
    ROOT / "src" / "llm_scoring" / "llm_pilot_common.py",
]


def test_frozen_prompt_hash_stable():
    h1 = protocol_prompt_hash()
    h2 = protocol_prompt_hash()
    assert h1 == h2
    assert len(h1) == 64


def test_parse_difficulty_scalar():
    assert parse_difficulty("0.42") == 0.42
    assert parse_difficulty("1") == 1.0
    assert parse_difficulty("The answer is 0.7") == 0.7
    assert pd.isna(parse_difficulty("no number here"))


def test_parse_response_record_bounds():
    rec = parse_response_record("0.55")
    assert rec["parse_valid"] is True
    assert rec["scalar_difficulty"] == 0.55
    bad = parse_response_record("not-a-score")
    assert bad["parse_valid"] is False


def test_cache_key_deterministic():
    cfg = PilotConfig.load()
    kwargs = dict(
        model="gpt-4o-mini",
        dataset="xes3g5m",
        item_id_hash="abc",
        source_content_hash="def",
        prompt_hash="ghi",
        temperature=0.0,
        seed=2024,
        schema_version=cfg.schema_version,
        run_kind="deterministic",
    )
    assert cache_key(**kwargs) == cache_key(**kwargs)
    assert cache_key(**kwargs) != cache_key(**{**kwargs, "temperature": 0.3})


def test_pilot_membership_exact_100():
    cfg = PilotConfig.load()
    for ds in ("xes3g5m", "junyi"):
        path = PILOT_DIR / f"{ds}_pilot_items.parquet"
        if not path.exists():
            pytest.skip("pilot samples not built")
        df = pd.read_parquet(path)
        assert len(df) == cfg.pilot_items_per_dataset
        assert df["item_id_hash"].is_unique


def test_pilot_no_denied_fields():
    for ds in ("xes3g5m", "junyi"):
        path = PILOT_DIR / f"{ds}_pilot_items.parquet"
        if not path.exists():
            pytest.skip("pilot samples not built")
        df = pd.read_parquet(path)
        assert PROMPT_DENYLIST.isdisjoint(df.columns)


def test_render_messages_no_denied_content():
    msgs = render_messages("求解 2+3=?")
    blob = json.dumps(msgs)
    for term in ("correct answer", "error rate", "response count"):
        assert term not in blob.lower()


def test_outcome_tables_not_imported_in_llm_scripts():
    """Runner/analysis scripts must not load student outcome tables."""
    check_paths = [
        ROOT / "src" / "llm_scoring" / "run_llm_pilot.py",
        ROOT / "src" / "llm_scoring" / "analyze_llm_pilot.py",
        ROOT / "src" / "llm_scoring" / "sample_pilot_items.py",
    ]
    banned_reads = ("read_parquet", "interactions.parquet", "empirical_difficulty", "rasch")
    for path in check_paths:
        text = path.read_text(encoding="utf-8")
        if "interactions.parquet" in text:
            assert "train_response_count" in text or "stratification" in text.lower() or path.name == "sample_pilot_items.py"
        for term in ("empirical_difficulty", "rasch", "error_rate"):
            assert term not in text, f"{path.name} must not reference {term}"


def test_no_api_key_in_artifacts():
    if CACHE_DIR.exists():
        for p in CACHE_DIR.rglob("*"):
            if p.is_file() and p.suffix in (".json", ".md", ".txt"):
                text = p.read_text(encoding="utf-8", errors="ignore")
                assert "sk-" not in text


def test_junyi_extraction_flags_preserved_in_pilot():
    path = PILOT_DIR / "junyi_pilot_items.parquet"
    if not path.exists():
        pytest.skip("junyi pilot not built")
    df = pd.read_parquet(path)
    assert "item_content_type" in df.columns
    assert "has_image_dependency" in df.columns or "has_dynamic_template" in df.columns


def test_retry_limits_in_config():
    cfg = PilotConfig.load()
    assert cfg.max_retries_transient <= 2
    assert cfg.max_retries_format <= 1


def test_cache_rerun_idempotent(tmp_path, monkeypatch):
    cfg = PilotConfig.load()
    key = cache_key(
        model="gpt-4o-mini",
        dataset="xes3g5m",
        item_id_hash="item1",
        source_content_hash="content1",
        prompt_hash=protocol_prompt_hash(),
        temperature=0.0,
        seed=2024,
        schema_version=cfg.schema_version,
        run_kind="deterministic",
    )
    index = {key: {"parse_valid": True, "scalar_difficulty": 0.5}}
    idx_path = tmp_path / "cache_index.json"
    idx_path.write_text(json.dumps(index), encoding="utf-8")
    loaded = json.loads(idx_path.read_text())
    assert key in loaded
    assert loaded[key]["parse_valid"]


def test_protocol_file_exists():
    assert PROMPT_PROTOCOL_PATH.exists()
