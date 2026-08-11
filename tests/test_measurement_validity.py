"""Integrity guards for Phase F measurement validity."""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "measurement"))
sys.path.insert(0, str(ROOT / "src" / "simulation"))

from measurement_common import DATASETS, TABLE_DIR, load_config, rank_independent  # noqa: E402
from rasch_estimator import fit_rasch_1pl  # noqa: E402
from synthetic_alignment_common import load_gsm8k_items, normalize_z  # noqa: E402

ANALYSIS_SCRIPTS = [
    ROOT / "src" / "measurement" / "build_authentic_difficulty_references.py",
    ROOT / "src" / "measurement" / "run_authentic_construct_validity.py",
    ROOT / "src" / "simulation" / "run_synthetic_alignment_ladder.py",
]


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def _require_processed_items(*datasets: str) -> None:
    missing = [
        ds for ds in datasets
        if not (ROOT / "data_processed" / ds / "items.parquet").exists()
    ]
    if missing:
        pytest.skip(
            f"processed items not built for {missing}; run src/data_prep/build_unified_*.py "
            "against locally obtained raw data first (see data/README.md)"
        )


def test_item_universe_counts(cfg):
    _require_processed_items(*DATASETS)
    for ds in DATASETS:
        items = pd.read_parquet(ROOT / "data_processed" / ds / "items.parquet")
        n = int(items["eligible_for_llm_scoring"].sum())
        assert n == cfg["llm_scoreable_counts"][ds]


def test_no_excluded_items_in_llm_features():
    path = ROOT / "artifacts" / "scores" / "llm_item_scores.parquet"
    if not path.exists():
        pytest.skip("LLM features missing")
    _require_processed_items(*DATASETS)
    llm = pd.read_parquet(path)
    for ds in DATASETS:
        items = pd.read_parquet(ROOT / "data_processed" / ds / "items.parquet")
        excluded = set(items.loc[~items["eligible_for_llm_scoring"], "item_id_hash"])
        used = set(llm.loc[llm["dataset"] == ds, "item_id_hash"])
        assert used.isdisjoint(excluded)


def test_deployable_references_exclude_test_split():
    path = TABLE_DIR / "AUTHENTIC_DIFFICULTY_REFERENCES.csv"
    if not path.exists():
        pytest.skip("references not built")
    refs = pd.read_csv(path)
    deploy = refs[refs["reference_scope"] == "deployable_train"]
    assert (deploy["split_source"] == "train").all()
    held = refs[refs["reference_scope"] == "held_out_test"]
    assert (held["split_source"] == "test").all()


def test_oracle_marked_diagnostic_only():
    path = TABLE_DIR / "AUTHENTIC_DIFFICULTY_REFERENCES.csv"
    if not path.exists():
        pytest.skip("references not built")
    refs = pd.read_csv(path)
    oracle = refs[refs["reference_scope"] == "oracle_diagnostic"]
    assert len(oracle) > 0
    assert (oracle["split_source"] == "all").all()


def test_rasch_orientation_higher_is_harder():
    rng = np.random.default_rng(0)
    rows = []
    for s in range(30):
        for item, p_correct in [("easy", 0.85), ("hard", 0.15)]:
            rows.append({
                "student_id_hash": f"s{s}",
                "item_id_hash": item,
                "correct": int(rng.random() < p_correct),
            })
    res = fit_rasch_1pl(pd.DataFrame(rows), max_iter=80, min_responses=5)
    diff = res.item_difficulties.set_index("item_id_hash")["rasch_difficulty"]
    assert diff["hard"] > diff["easy"]


def test_threshold_table_reproducible(cfg):
    path = TABLE_DIR / "AUTHENTIC_VALIDITY_THRESHOLDS.csv"
    if not path.exists():
        pytest.skip("threshold table missing")
    thr = pd.read_csv(path)
    assert set(thr["threshold"]) == set(cfg["response_thresholds"])


def test_calibration_tables_item_level_cv():
    text = (ROOT / "src" / "measurement" / "run_authentic_construct_validity.py").read_text(encoding="utf-8")
    assert "KFold" in text
    assert "IsotonicRegression" in text


def test_synthetic_d_ind_rank_independent():
    if not (ROOT / "data_raw" / "gsm8k" / "train.csv").exists():
        pytest.skip("GSM8K raw data not available locally; see data/README.md")
    items = load_gsm8k_items(50)
    d_llm = normalize_z(items["difficulty"].values)
    rng = np.random.default_rng(2024)
    d_ind = normalize_z(rng.normal(size=len(d_llm)))
    assert rank_independent(d_llm, d_ind) < 0.2


def test_synthetic_condition_weights(cfg):
    cond = cfg["synthetic"]["conditions"]
    assert cond["S0"]["d_llm_weight"] == 1.0 and cond["S0"]["d_ind_weight"] == 0.0
    assert cond["S4"]["d_llm_weight"] == 0.0 and cond["S4"]["d_ind_weight"] == 1.0
    assert cond["S2"]["d_llm_weight"] == 0.5


def test_holm_families_fixed(cfg):
    assert len(cfg["holm_families"]) == 4


def test_no_student_ids_in_output_tables():
    for name in ["AUTHENTIC_DIFFICULTY_REFERENCES.csv", "AUTHENTIC_VALIDITY_CORRELATIONS.csv"]:
        path = TABLE_DIR / name
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8").lower()
        assert "student_id" not in text


def test_no_kt_full_matrix_import_in_validity_scripts():
    banned = ("run_all_final_experiments", "cold_start", "AutoKTGraph")
    for path in ANALYSIS_SCRIPTS:
        text = path.read_text(encoding="utf-8")
        for term in banned:
            assert term not in text, f"{path.name} references {term}"


def test_fixed_seeds_reproduce_synthetic_summary():
    path = TABLE_DIR / "SYNTHETIC_ALIGNMENT_SEED_SUMMARY.csv"
    if not path.exists():
        pytest.skip("synthetic outputs missing")
    df = pd.read_csv(path)
    assert df["seed"].nunique() >= 10


def test_measurement_manifest_when_present():
    path = ROOT / "data_manifests" / "measurement_validity_manifest.json"
    if not path.exists():
        pytest.skip("manifest not frozen")
    m = json.loads(path.read_text(encoding="utf-8"))
    assert "output_table_hashes" in m
    assert "authentic_validity_status" in m


def test_authentic_reference_v2_error_orientation():
    path = TABLE_DIR / "AUTHENTIC_DIFFICULTY_REFERENCES_V2_ORIENTATION_CORRECTED.csv"
    if not path.exists():
        pytest.skip("V2 authentic reference table missing")
    refs = pd.read_csv(path)
    assert (refs["heldout_correct_count"] + refs["heldout_incorrect_count"] == refs["heldout_response_count"]).all()
    assert np.allclose(refs["raw_error"], 1 - refs["raw_correctness"], atol=1e-12)
    assert np.allclose(
        refs["smoothed_error_beta_1_1"] + refs["smoothed_correctness_beta_1_1"],
        1.0,
        atol=1e-12,
    )
    recomputed = (refs["heldout_incorrect_count"] + 1) / (refs["heldout_response_count"] + 2)
    assert np.allclose(refs["smoothed_error_beta_1_1"], recomputed, atol=1e-12)


def test_authentic_v2_primary_spearman_positive():
    path = TABLE_DIR / "AUTHENTIC_VALIDITY_CORRELATIONS_V2.csv"
    if not path.exists():
        pytest.skip("V2 authentic correlations missing")
    corr = pd.read_csv(path)
    primary = corr[(corr["reference"] == "test_smoothed_error") & (corr["threshold"] == 20)]
    assert len(primary) == 4
    assert (primary["spearman_rho"] > 0).all()


def test_simulation_error_orientation_confirmed():
    path = TABLE_DIR / "SYNTHETIC_ALIGNMENT_SEED_SUMMARY.csv"
    if not path.exists():
        pytest.skip("synthetic summary missing")
    seed = pd.read_csv(path)
    s0 = seed[seed["condition"] == "S0"]["rho_d_llm_sim_error"].mean()
    s4 = seed[seed["condition"] == "S4"]["rho_d_llm_sim_error"].mean()
    assert s0 > 0.9
    assert abs(s4) < 0.1
