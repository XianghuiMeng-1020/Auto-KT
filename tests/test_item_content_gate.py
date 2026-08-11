"""Tests for item content admission gates."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "data_prep"))

from leakage_design import item_content_gate  # noqa: E402


def test_eedi_content_gate_fails():
    # Documented Eedi audit: 0% machine-readable text for LLM pipeline
    assert not item_content_gate(text_coverage=0.0, join_coverage=1.0)


def test_dbe_mirror_content_gate_passes_when_present():
    data = ROOT / "data_raw" / "dbe_kt22" / "hf_mirror"
    if not (data / "Questions.csv").exists():
        return
    qs = pd.read_csv(data / "Questions.csv")
    trans = pd.read_csv(data / "Transaction.csv", usecols=["question_id"])
    interacted = set(trans["question_id"].unique())
    qsub = qs[qs["id"].isin(interacted)]
    text_cov = (
        qsub["question_text"].notna() & (qsub["question_text"].astype(str).str.strip() != "")
    ).mean()
    join_cov = len(set(qs["id"]) & interacted) / len(interacted)
    assert item_content_gate(float(text_cov), float(join_cov))
