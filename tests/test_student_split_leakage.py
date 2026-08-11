"""Tests for student split leakage rules."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "data_prep"))

from leakage_design import (  # noqa: E402
    assert_no_student_overlap,
    filter_interactions_by_students,
    leakage_test_graph_students,
    leakage_test_no_test_in_reference_table,
    split_students,
)


def test_student_split_disjoint():
    split = split_students(list(range(100)), seed=42)
    assert_no_student_overlap(split)
    assert len(split.train) + len(split.val) + len(split.test) == 100


def test_train_only_empirical_difficulty_students():
    df = pd.DataFrame({"student_id": [1, 1, 2, 2, 3, 3], "question_id": [10, 11, 10, 12, 11, 12]})
    split = split_students([1, 2, 3], seed=1)
    train_df = filter_interactions_by_students(df, "student_id", split.train)
    train_students_used = set(train_df["student_id"])
    assert leakage_test_no_test_in_reference_table(split, train_students_used)
    assert not train_students_used & split.test or split.test.isdisjoint(train_students_used)


def test_graph_excludes_test_students():
    split = split_students([1, 2, 3, 4, 5], seed=7)
    graph_students = set(split.train)  # built from train only in valid pipeline
    assert leakage_test_graph_students(graph_students, split)
    bad_graph = graph_students | split.test
    assert not leakage_test_graph_students(bad_graph, split)


def test_dbe_split_feasible_when_data_present():
    data = ROOT / "data_raw" / "dbe_kt22" / "hf_mirror" / "Transaction.csv"
    if not data.exists():
        return
    trans = pd.read_csv(data, usecols=["student_id"])
    split = split_students(trans["student_id"].unique(), seed=2024)
    assert_no_student_overlap(split)
    assert len(split.test) >= 10
