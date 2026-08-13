"""TLT-3D Phase 4A — final evidence freeze and claim-boundary gates."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pandas as pd
import json

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "tlt3d"
REP = ROOT / "reports"
FINAL_EXPERIMENT = "a188e64129b8f012caa7ac44ae5c4bea84fa6472"
PROTOCOL = "a459e34a24240c03ba4dbe4b1d0185e42eaf4377"
FAMILY_A_SHA = "49078d1e7044608c666cb689551ffec0265ecb957070a37d78b7d0c0f265ebf4"


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_family_row_counts():
    assert len(pd.read_csv(ART / "FINAL_FAMILY_A.csv")) == 6
    assert len(pd.read_csv(ART / "FINAL_FAMILY_B.csv")) == 6
    assert len(pd.read_csv(ART / "FINAL_FAMILY_C.csv")) == 12
    assert len(pd.read_csv(ART / "FINAL_FAMILY_D.csv")) == 36


def test_family_d_holm_supported_split_and_distinctive():
    audit = json.loads((ART / "FINAL_FAMILY_D_SUPPORTED_AUDIT.json").read_text())
    assert audit["positive_holm_supported_count"] == 3
    assert audit["negative_holm_supported_count"] == 2
    assert len(audit["LLM_BETTER"]) == 3
    assert len(audit["LLM_WORSE"]) == 2
    assert audit["full_distinctive_comparator_triplets_cleared"] == 0
    assert audit["overall_verdict"] == "NO_DISTINCTIVE_SUPPORT"
    assert all(not x["full_triplet_cleared"] for x in audit["positive_triplet_checks"])
    assert all(not x["distinctive"] for x in audit["positive_triplet_checks"])


def test_dataset_universe_primary_counts():
    u = pd.read_csv(ART / "FINAL_DATASET_UNIVERSES.csv").set_index("dataset")
    assert int(u.loc["xes3g5m", "rq2_first_observed_primary"]) == 3265
    assert int(u.loc["junyi", "rq2_first_observed_primary"]) == 169
    assert int(u.loc["dbe_kt22", "rq2_first_observed_primary"]) == 166
    assert int(u.loc["dbe_kt22", "unseen_target_universe"]) == 166
    assert int(u.loc["dbe_kt22", "llm_scored"]) == 166
    notes = " ".join(u["notes"].astype(str).tolist()).lower()
    assert "eligible" not in notes or "do not call" in notes


def test_chronology_repairs_not_preregistered():
    chrono = pd.read_csv(ART / "FINAL_STATISTICAL_CHRONOLOGY.csv")
    text = chrono.to_csv(index=False).lower()
    assert "POST_RESULT_INFERENTIAL_REPAIR_001" in chrono["amendment_id"].tolist()
    assert "POST_RESULT_OPERATIONALIZATION_REPAIR_002" in chrono["amendment_id"].tolist()
    assert "POST_RESULT_OPERATIONALIZATION_REPAIR_003" in chrono["amendment_id"].tolist()
    assert "preregistered" not in text
    b = chrono[chrono["component"] == "Family B p-value"].iloc[0]
    assert b["specified_before_relevant_result"] in (False, "False", False)
    assert b["amendment_id"] == "POST_RESULT_INFERENTIAL_REPAIR_001"
    assert bool(b["manuscript_disclosure_required"]) is True or str(b["manuscript_disclosure_required"]) == "True"


def test_family_status_fields_not_flattened():
    fa = pd.read_csv(ART / "FINAL_FAMILY_A.csv")
    fb = pd.read_csv(ART / "FINAL_FAMILY_B.csv")
    fc = pd.read_csv(ART / "FINAL_FAMILY_C.csv")
    fd = pd.read_csv(ART / "FINAL_FAMILY_D.csv")
    assert (fa["analysis_status"] == "PRE_RESULT_CONFIRMATORY").all()
    assert (fb["effect_status"] == "PRE_RESULT_FROZEN").all()
    assert (fb["CI_status"] == "PRE_RESULT_FROZEN").all()
    assert (fb["inferential_p_status"] == "POST_RESULT_INFERENTIAL_REPAIR_001").all()
    assert (fc["hypothesis_membership_status"] == "PRE_RESULT_FROZEN").all()
    assert (fc["operationalization_status"] == "POST_RESULT_OPERATIONALIZATION_REPAIR_002").all()
    assert (fc["warm_included"] == False).all()
    assert (fc["backbone"] == "GRU").all()
    assert (fd["hypothesis_membership_status"] == "PRE_RESULT_FROZEN").all()
    assert (fd["operationalization_status"] == "POST_RESULT_OPERATIONALIZATION_REPAIR_003").all()


def test_family_a_matches_immutable_hash():
    assert _sha(ART / "family_A_confirmatory_results.csv") == FAMILY_A_SHA
    final = pd.read_csv(ART / "FINAL_FAMILY_A.csv")
    assert (final["source_sha256"] == FAMILY_A_SHA).all()


def test_six_cell_matrix_complete():
    six = pd.read_csv(ART / "FINAL_SIX_CELL_EVIDENCE_MATRIX.csv")
    assert len(six) == 6
    assert six.isna().sum().sum() == 0
    required = {
        "dataset",
        "LLM",
        "A_rho",
        "A_CI",
        "A_Holm",
        "A_supported",
        "B_delta_R2",
        "B_CI",
        "B_Holm_repaired",
        "B_supported",
        "C_vs_standard",
        "C_vs_standard_Holm",
        "C_vs_random",
        "C_vs_random_Holm",
        "C_distinctive",
        "D_GRU_vs_standard",
        "D_GRU_vs_standard_Holm",
        "D_GRU_vs_random",
        "D_GRU_vs_random_Holm",
        "D_GRU_vs_charlen",
        "D_GRU_vs_charlen_Holm",
        "D_SAKT_vs_standard",
        "D_SAKT_vs_standard_Holm",
        "D_SAKT_vs_random",
        "D_SAKT_vs_random_Holm",
        "D_SAKT_vs_charlen",
        "D_SAKT_vs_charlen_Holm",
        "D_distinctive",
        "final_classification",
    }
    assert required.issubset(set(six.columns))
    assert int(six["A_supported"].sum()) == 5
    assert int(six["B_supported"].sum()) == 1
    assert int(six["C_distinctive"].sum()) == 0
    assert int(six["D_distinctive"].sum()) == 0


def test_experimental_status_and_evidence_pack():
    st = json.loads((ART / "TLT3D_FINAL_EXPERIMENTAL_STATUS.json").read_text())
    assert st["experimental_program"] == "COMPLETE"
    assert st["families_complete"] == ["A", "B", "C", "D"]
    assert st["scientific_blockers"] == []
    assert st["new_experiments_authorized"] is False
    assert st["final_distinctive_value_verdict"] == "NO_DISTINCTIVE_SUPPORT"
    assert st["ready_for_manuscript_rewrite"] is True
    assert all(st["gates"][f"F{i}"] is True for i in range(1, 11))
    pack = json.loads((ART / "FINAL_TLT3D_EVIDENCE_PACK.json").read_text())
    for key in [
        "A_dataset_universes",
        "B_llm_scoring_identities",
        "C_rq1_score_characterization",
        "D_family_A",
        "E_family_B",
        "F_rq3_existing_controlled_simulation",
        "G_family_C",
        "H_family_D",
        "I_mandatory_sensitivities",
        "J_dbe_expert_secondary",
        "K_final_distinctive_value_classification",
        "L_protocol_amendments",
        "M_claim_boundaries",
        "N_exact_artifact_hashes",
    ]:
        assert key in pack
    assert pack["commit_chain"]["final_experimental_result"] == FINAL_EXPERIMENT
    assert pack["commit_chain"]["pre_result_protocol"] == PROTOCOL


def test_claim_freeze_and_structure_reports_exist():
    for name in [
        "TLT3D_FINAL_CLAIM_FREEZE.md",
        "TLT3D_FINAL_RESULTS_STRUCTURE.md",
        "TLT3D_FINAL_DISCUSSION_STRUCTURE.md",
        "TLT3D_FINAL_TABLE_FIGURE_PLAN.md",
        "TLT3D_FINAL_REPRODUCIBILITY_AUDIT.md",
    ]:
        p = REP / name
        assert p.exists() and p.stat().st_size > 100
    claim = (REP / "TLT3D_FINAL_CLAIM_FREEZE.md").read_text()
    assert "Authentic correspondence was therefore insufficient evidence of distinctive deployment value." in claim
    assert "All confirmatory analyses were preregistered." in claim  # as prohibited
    assert "Prohibited claims" in claim


def test_no_evidence_artifacts_dirty_in_working_tree():
    """Phase 4A sealed evidence must remain untouched; later compliance may edit manuscript."""
    out = subprocess.check_output(
        ["git", "status", "--porcelain"],
        cwd=ROOT,
        text=True,
    )
    dirty = [ln for ln in out.splitlines() if ln.strip()]
    forbidden = [
        ln
        for ln in dirty
        if "artifacts/tlt3d/" in ln or "FINAL_FAMILY_" in ln or "tlt3d-experimental" in ln
    ]
    assert not forbidden, forbidden


def test_supported_counts_match_source_confirmatory():
    fa = pd.read_csv(ART / "FINAL_FAMILY_A.csv")
    fb = pd.read_csv(ART / "FINAL_FAMILY_B.csv")
    fc = pd.read_csv(ART / "FINAL_FAMILY_C.csv")
    fd = pd.read_csv(ART / "FINAL_FAMILY_D.csv")
    assert int(fa["holm_supported"].sum()) == 5
    assert int(fb["holm_supported"].sum()) == 1
    assert int(fc["holm_supported"].sum()) == 0
    better = fd[(fd["holm_supported"]) & (fd["effect_direction"] == "LLM_BETTER")]
    worse = fd[(fd["holm_supported"]) & (fd["effect_direction"] == "LLM_WORSE")]
    assert len(better) == 3
    assert len(worse) == 2
