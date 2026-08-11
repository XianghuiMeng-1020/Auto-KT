"""Prespecified leakage-safe split utilities (design only; no outcome computation)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StudentSplit:
    train: set
    val: set
    test: set
    seed: int


def split_students(
    student_ids: pd.Series | list,
    train_frac: float = 0.7,
    val_frac: float = 0.1,
    seed: int = 2024,
) -> StudentSplit:
    ids = sorted(set(student_ids))
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(train_frac * n)
    n_val = int(val_frac * n)
    train = set(ids[:n_train])
    val = set(ids[n_train : n_train + n_val])
    test = set(ids[n_train + n_val :])
    return StudentSplit(train=train, val=val, test=test, seed=seed)


def assert_no_student_overlap(split: StudentSplit) -> None:
    assert not (split.train & split.val)
    assert not (split.train & split.test)
    assert not (split.val & split.test)


def filter_interactions_by_students(df: pd.DataFrame, student_col: str, students: set) -> pd.DataFrame:
    return df[df[student_col].isin(students)].copy()


def leakage_test_no_test_in_reference_table(
    split: StudentSplit,
    reference_student_col_values: set,
) -> bool:
    return reference_student_col_values.isdisjoint(split.test)


def leakage_test_graph_students(graph_student_ids: set, split: StudentSplit) -> bool:
    return graph_student_ids.isdisjoint(split.test)


def item_content_gate(text_coverage: float, join_coverage: float, min_coverage: float = 0.95) -> bool:
    return text_coverage >= min_coverage and join_coverage >= min_coverage
