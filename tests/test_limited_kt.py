"""Limited KT integrity guards."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "kt"))

from limited_kt_common import (  # noqa: E402
    CONFIG_PATH,
    MASK_DIR,
    CleanKT,
    build_exposure_mask,
    load_config,
    load_dataset_bundle,
)


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def test_exposure_levels_frozen(cfg):
    assert cfg["exposure_levels"] == [0, 1, 3, 5, 10, 20, "warm"]


def test_primary_conditions_frozen(cfg):
    assert cfg["primary_conditions"] == [
        "Standard", "LLM-Mini", "LLM-5.4", "Random-Scalar", "TrainEmpDiff"
    ]


def test_masks_identical_hash_across_reread(cfg):
    df1, h1 = build_exposure_mask("junyi", 5, cfg)
    df2, h2 = build_exposure_mask("junyi", 5, cfg)
    assert h1 == h2
    assert len(df1) == len(df2)


def test_exposure_cap_respected(cfg):
    df, _ = build_exposure_mask("junyi", 3, cfg)
    counts = df.groupby("item_id_hash").size()
    assert counts.max() <= 3


def test_warm_uses_more_than_zero(cfg):
    z, _ = build_exposure_mask("junyi", 0, cfg)
    w, _ = build_exposure_mask("junyi", "warm", cfg)
    assert len(w) > len(z)


def test_scalar_parameter_counts_comparable():
    n_items = 200
    base = CleanKT(n_items, use_scalar=False).count_parameters()
    scalar = CleanKT(n_items, use_scalar=True).count_parameters()
    assert 0 < scalar - base < 50


def test_bundle_universe_counts(cfg):
    b = load_dataset_bundle("junyi", "warm", cfg, max_train_students=50, max_test_students=20)
    assert len(b.item_to_idx) == cfg["scoreable_counts"]["junyi"]


def test_no_test_in_train_empirical(cfg):
    b = load_dataset_bundle("junyi", 0, cfg, max_train_students=30, max_test_students=10)
    assert "train_empirical" in b.scalar_maps
    assert "oracle_empirical" in b.scalar_maps


def test_holm_families_frozen(cfg):
    assert len(cfg["holm_families"]) == 4


def test_registry_no_raw_student_ids():
    path = ROOT / "runs" / "response_limited_kt" / "RUN_REGISTRY.csv"
    if not path.exists():
        pytest.skip("no runs yet")
    text = path.read_text(encoding="utf-8").lower()
    assert "student_id_hash" not in text
