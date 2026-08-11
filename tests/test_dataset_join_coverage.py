"""Tests for dataset join coverage gates."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "data_prep"))

from leakage_design import item_content_gate  # noqa: E402


def test_item_content_gate_pass():
    assert item_content_gate(0.96, 0.99)


def test_item_content_gate_fail_text():
    assert not item_content_gate(0.90, 0.99)


def test_item_content_gate_fail_join():
    assert not item_content_gate(0.99, 0.90)


def test_dbe_join_coverage_when_mirror_present():
    data = ROOT / "data_raw" / "dbe_kt22" / "hf_mirror"
    if not (data / "Questions.csv").exists():
        return
    qs = pd.read_csv(data / "Questions.csv")
    trans = pd.read_csv(data / "Transaction.csv", usecols=["question_id"])
    interacted = set(trans["question_id"].unique())
    meta = set(qs["id"].unique())
    join_cov = len(interacted & meta) / len(interacted)
    assert join_cov >= 0.95
