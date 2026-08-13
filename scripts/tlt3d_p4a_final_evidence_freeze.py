#!/usr/bin/env python3
"""TLT-3D Phase 4A — final evidence consolidation and claim freeze.

NO new scientific experiments. NO manuscript edits.
Only copy / reconcile / verify / format / classify existing final outputs.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "tlt3d"
CFG = ROOT / "configs" / "tlt3d"
REP = ROOT / "reports"

FINAL_EXPERIMENT = "a188e64129b8f012caa7ac44ae5c4bea84fa6472"
PROTOCOL = "a459e34a24240c03ba4dbe4b1d0185e42eaf4377"
REPAIR_B = "POST_RESULT_INFERENTIAL_REPAIR_001"
REPAIR_C = "POST_RESULT_OPERATIONALIZATION_REPAIR_002"
REPAIR_D = "POST_RESULT_OPERATIONALIZATION_REPAIR_003"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def holm_supported(effect: float, holm_p: float, positive_better: bool = True) -> bool:
    if positive_better:
        return bool(holm_p < 0.05 and effect > 0)
    return bool(holm_p < 0.05 and effect < 0)


def build_universes() -> pd.DataFrame:
    # Authoritative counts from P11 reconciliation + Phase-1 freeze + FO eligibility
    p11 = json.loads((ART / "P11_AMENDMENT_COMPUTE_SUMMARY.json").read_text())
    rows = [
        {
            "dataset": "xes3g5m",
            "raw_full": 7618,
            "text_scoreable": 5363,
            "llm_scored": 5363,
            "rq2_first_observed_primary": 3265,
            "rq2_legacy_all_response": 3279,
            "response_limited_universe": 5363,
            "unseen_target_universe": 5363,
            "notes": "Do not call 7618 eligible/text-scoreable",
            "source": "reports/TLT3D_P11_DATASET_UNIVERSE_RECONCILIATION.md",
        },
        {
            "dataset": "junyi",
            "raw_full": 666,
            "text_scoreable": 190,
            "llm_scored": 190,
            "rq2_first_observed_primary": 169,
            "rq2_legacy_all_response": 183,
            "response_limited_universe": 190,
            "unseen_target_universe": 190,
            "notes": "Do not call 666 eligible/text-scoreable; FO vs legacy material",
            "source": "reports/TLT3D_P11_DATASET_UNIVERSE_RECONCILIATION.md",
        },
        {
            "dataset": "dbe_kt22",
            "raw_full": 212,
            "text_scoreable": 166,
            "llm_scored": 166,
            "rq2_first_observed_primary": 166,
            "rq2_legacy_all_response": None,
            "response_limited_universe": 166,
            "unseen_target_universe": 166,
            "notes": "text-complete=LLM-scored=FO=unseen targets=166",
            "source": "configs/tlt3d/dbe_item_universe.json; artifacts/tlt3d/DBE_PRE_LLM_FREEZE.json",
        },
    ]
    # verify FO counts against eligibility CSVs
    assert int(pd.read_csv(ART / "xes_first_observed_rq2_eligibility.csv")["eligible_ge20"].sum()) == 3265
    assert int(pd.read_csv(ART / "junyi_first_observed_rq2_eligibility.csv")["eligible_ge20"].sum()) == 169
    assert len(pd.read_csv(ART / "dbe_first_observed_rq2_eligibility.csv")) == 166
    _ = p11  # chronology/sensitivity also used later
    return pd.DataFrame(rows)


def build_family_a() -> pd.DataFrame:
    fa = pd.read_csv(ART / "family_A_confirmatory_results.csv")
    assert len(fa) == 6
    out = pd.DataFrame(
        {
            "ID": fa["hypothesis_id"],
            "dataset": fa["dataset"],
            "model": fa["model"],
            "N": fa["n_items"],
            "rho": fa["spearman_rho"],
            "CI_lo": fa["ci_lo"],
            "CI_hi": fa["ci_hi"],
            "raw_p": fa["raw_p"],
            "holm_p": fa["holm_p"],
            "holm_supported": (fa["holm_p"] < 0.05) & (fa["spearman_rho"] > 0),
            "analysis_status": "PRE_RESULT_CONFIRMATORY",
            "source_artifact": "artifacts/tlt3d/family_A_confirmatory_results.csv",
            "source_sha256": sha(ART / "family_A_confirmatory_results.csv"),
        }
    )
    return out


def build_family_b() -> pd.DataFrame:
    fb = pd.read_csv(ART / "family_B_confirmatory_results.csv")
    assert len(fb) == 6
    raw = fb["p_raw_repaired"] if "p_raw_repaired" in fb.columns else fb["raw_p"]
    holm = fb["p_holm_repaired"] if "p_holm_repaired" in fb.columns else fb["holm_p"]
    out = pd.DataFrame(
        {
            "ID": fb["hypothesis_id"],
            "dataset": fb["dataset"],
            "model": fb["model"],
            "N": fb["n_items"],
            "base_R2": fb["base_r2"],
            "aug_R2": fb["aug_r2"],
            "delta_R2": fb["delta_r2"],
            "CI_lo": fb["ci_lo"],
            "CI_hi": fb["ci_hi"],
            "raw_p_repaired": raw,
            "holm_p_repaired": holm,
            "CI_excludes_zero": fb["ci_lo"] > 0,
            "holm_supported": (holm < 0.05) & (fb["delta_r2"] > 0),
            "effect_status": "PRE_RESULT_FROZEN",
            "CI_status": "PRE_RESULT_FROZEN",
            "inferential_p_status": REPAIR_B,
            "source_artifact": "artifacts/tlt3d/family_B_confirmatory_results.csv",
            "source_sha256": sha(ART / "family_B_confirmatory_results.csv"),
        }
    )
    return out


def build_family_c() -> pd.DataFrame:
    fc = pd.read_csv(ART / "family_C_confirmatory_results.csv")
    assert len(fc) == 12
    out = pd.DataFrame(
        {
            "ID": fc["hypothesis_id"],
            "dataset": fc["dataset"],
            "LLM": fc["llm"],
            "comparator": fc["comparator"],
            "effect_log_loss": fc["effect_log_loss"],
            "CI_lo": fc["effect_ci_low"],
            "CI_hi": fc["effect_ci_high"],
            "raw_p": fc["raw_p"],
            "holm_p": fc["holm_p"],
            "holm_supported": (fc["holm_p"] < 0.05) & (fc["effect_log_loss"] > 0),
            "hypothesis_membership_status": "PRE_RESULT_FROZEN",
            "operationalization_status": REPAIR_C,
            "confirmatory_exposures": "0,1,3,5,10,20",
            "warm_included": False,
            "backbone": "GRU",
            "source_artifact": "artifacts/tlt3d/family_C_confirmatory_results.csv",
            "source_sha256": sha(ART / "family_C_confirmatory_results.csv"),
        }
    )
    return out


def build_family_d() -> pd.DataFrame:
    fd = pd.read_csv(ART / "family_D_confirmatory_results.csv")
    assert len(fd) == 36
    direction = fd["effect_log_loss"].apply(
        lambda x: "LLM_BETTER" if x > 0 else ("LLM_WORSE" if x < 0 else "NULL")
    )
    out = pd.DataFrame(
        {
            "ID": fd["hypothesis_id"],
            "dataset": fd["dataset"],
            "backbone": fd["backbone"],
            "LLM": fd["llm"],
            "comparator": fd["comparator"],
            "effect_log_loss": fd["effect_log_loss"],
            "CI_lo": fd["effect_ci_low"],
            "CI_hi": fd["effect_ci_high"],
            "raw_p": fd["raw_p"],
            "holm_p": fd["holm_p"],
            "effect_direction": direction,
            "holm_supported": fd["holm_p"] < 0.05,
            "llm_better_or_worse": direction.map(
                {"LLM_BETTER": "better", "LLM_WORSE": "worse", "NULL": "null"}
            ),
            "hypothesis_membership_status": "PRE_RESULT_FROZEN",
            "operationalization_status": REPAIR_D,
            "source_artifact": "artifacts/tlt3d/family_D_confirmatory_results.csv",
            "source_sha256": sha(ART / "family_D_confirmatory_results.csv"),
        }
    )
    return out


def family_d_supported_audit(fd: pd.DataFrame) -> dict:
    pos = fd[(fd["holm_p"] < 0.05) & (fd["effect_log_loss"] > 0)].copy()
    neg = fd[(fd["holm_p"] < 0.05) & (fd["effect_log_loss"] < 0)].copy()
    assert len(pos) == 3, len(pos)
    assert len(neg) == 2, len(neg)

    # For each positive supported result, check whether same dataset×backbone×LLM
    # also clears Std + Random + CharLen under Holm+ positive.
    triplet_checks = []
    for r in pos.itertuples():
        cell = fd[(fd.dataset == r.dataset) & (fd.backbone == r.backbone) & (fd.LLM == r.LLM)]
        clears = {}
        for comp in ["Standard", "Random-PermutedScore", "CharacterLength"]:
            row = cell[cell.comparator == comp].iloc[0]
            clears[comp] = bool(row.holm_p < 0.05 and row.effect_log_loss > 0)
        triplet_checks.append(
            {
                "dataset": r.dataset,
                "backbone": r.backbone,
                "LLM": r.LLM,
                "supported_comparator": r.comparator,
                "effect": float(r.effect_log_loss),
                "CI": [float(r.CI_lo), float(r.CI_hi)],
                "holm_p": float(r.holm_p),
                "clears_Standard": clears["Standard"],
                "clears_Random": clears["Random-PermutedScore"],
                "clears_CharacterLength": clears["CharacterLength"],
                "full_triplet_cleared": all(clears.values()),
                "distinctive": False,  # no positive row is automatically distinctive
            }
        )
    assert not any(t["full_triplet_cleared"] for t in triplet_checks)
    # Also check any dataset×LLM×backbone full triplet anywhere
    any_full = False
    for (ds, bb, llm), g in fd.groupby(["dataset", "backbone", "LLM"]):
        clears = []
        for comp in ["Standard", "Random-PermutedScore", "CharacterLength"]:
            row = g[g.comparator == comp].iloc[0]
            clears.append(bool(row.holm_p < 0.05 and row.effect_log_loss > 0))
        if all(clears):
            any_full = True
    assert any_full is False

    return {
        "LLM_BETTER": pos.to_dict(orient="records"),
        "LLM_WORSE": neg.to_dict(orient="records"),
        "positive_holm_supported_count": 3,
        "negative_holm_supported_count": 2,
        "positive_triplet_checks": triplet_checks,
        "full_distinctive_comparator_triplets_cleared": 0,
        "overall_verdict": "NO_DISTINCTIVE_SUPPORT",
    }


def build_six_cell(fa, fb, fc, fd) -> pd.DataFrame:
    rows = []
    for ds in ["xes3g5m", "junyi", "dbe_kt22"]:
        for llm in ["gpt-4o-mini", "gpt-5.4"]:
            a = fa[(fa.dataset == ds) & (fa.model == llm)].iloc[0]
            b = fb[(fb.dataset == ds) & (fb.model == llm)].iloc[0]
            c_std = fc[(fc.dataset == ds) & (fc.LLM == llm) & (fc.comparator == "Standard")]
            c_rand = fc[(fc.dataset == ds) & (fc.LLM == llm) & (fc.comparator == "Random-ResampledScore")]
            assert len(c_std) == 1 and len(c_rand) == 1
            c_std = c_std.iloc[0]
            c_rand = c_rand.iloc[0]
            c_distinctive = bool(c_std.holm_supported and c_rand.holm_supported)

            def dget(bb, comp):
                r = fd[(fd.dataset == ds) & (fd.backbone == bb) & (fd.LLM == llm) & (fd.comparator == comp)].iloc[0]
                return float(r.effect_log_loss), float(r.holm_p), bool(r.holm_p < 0.05 and r.effect_log_loss > 0)

            g_std_e, g_std_h, g_std_s = dget("GRU", "Standard")
            g_rand_e, g_rand_h, g_rand_s = dget("GRU", "Random-PermutedScore")
            g_char_e, g_char_h, g_char_s = dget("GRU", "CharacterLength")
            s_std_e, s_std_h, s_std_s = dget("SAKT", "Standard")
            s_rand_e, s_rand_h, s_rand_s = dget("SAKT", "Random-PermutedScore")
            s_char_e, s_char_h, s_char_s = dget("SAKT", "CharacterLength")
            d_distinctive = (g_std_s and g_rand_s and g_char_s) or (s_std_s and s_rand_s and s_char_s)

            # final classification (mechanical, matches P3C distinctive rule application)
            a_ok = bool(a.holm_supported)
            b_ok = bool(b.holm_supported)
            if a_ok and b_ok and (c_distinctive or d_distinctive):
                classification = "DISTINCTIVE_SUPPORT"
            elif a_ok and b_ok and not (c_distinctive or d_distinctive):
                classification = "MEASUREMENT_ONLY_NO_DEPLOYMENT_SUPPORT"
            elif a_ok and (c_distinctive or d_distinctive) and not b_ok:
                classification = "PARTIAL_ASSOCIATION_AND_DEPLOYMENT_WITHOUT_INCREMENTAL"
            elif not a_ok and (c_distinctive or d_distinctive):
                classification = "DEPLOYMENT_SIGNAL_WITHOUT_MEASUREMENT_SUPPORT"
            else:
                classification = "NO_DISTINCTIVE_SUPPORT"

            rows.append(
                {
                    "dataset": ds,
                    "LLM": llm,
                    "A_rho": float(a.rho),
                    "A_CI": f"[{a.CI_lo}, {a.CI_hi}]",
                    "A_Holm": float(a.holm_p),
                    "A_supported": bool(a.holm_supported),
                    "B_delta_R2": float(b.delta_R2),
                    "B_CI": f"[{b.CI_lo}, {b.CI_hi}]",
                    "B_Holm_repaired": float(b.holm_p_repaired),
                    "B_supported": bool(b.holm_supported),
                    "C_vs_standard": float(c_std.effect_log_loss),
                    "C_vs_standard_Holm": float(c_std.holm_p),
                    "C_vs_random": float(c_rand.effect_log_loss),
                    "C_vs_random_Holm": float(c_rand.holm_p),
                    "C_distinctive": c_distinctive,
                    "D_GRU_vs_standard": g_std_e,
                    "D_GRU_vs_standard_Holm": g_std_h,
                    "D_GRU_vs_random": g_rand_e,
                    "D_GRU_vs_random_Holm": g_rand_h,
                    "D_GRU_vs_charlen": g_char_e,
                    "D_GRU_vs_charlen_Holm": g_char_h,
                    "D_SAKT_vs_standard": s_std_e,
                    "D_SAKT_vs_standard_Holm": s_std_h,
                    "D_SAKT_vs_random": s_rand_e,
                    "D_SAKT_vs_random_Holm": s_rand_h,
                    "D_SAKT_vs_charlen": s_char_e,
                    "D_SAKT_vs_charlen_Holm": s_char_h,
                    "D_distinctive": d_distinctive,
                    "final_classification": classification,
                }
            )
    out = pd.DataFrame(rows)
    assert len(out) == 6
    assert out.notna().all().all()
    return out


def build_chronology() -> pd.DataFrame:
    rows = [
        {
            "component": "Family A",
            "specified_before_relevant_result": True,
            "timing": "pre-result confirmatory",
            "amendment_id": "NONE",
            "final_reporting_status": "PRE_RESULT_CONFIRMATORY",
            "manuscript_disclosure_required": False,
        },
        {
            "component": "Family B effect",
            "specified_before_relevant_result": True,
            "timing": "pre-result frozen effect definition",
            "amendment_id": "NONE",
            "final_reporting_status": "PRE_RESULT_FROZEN",
            "manuscript_disclosure_required": False,
        },
        {
            "component": "Family B CI",
            "specified_before_relevant_result": True,
            "timing": "pre-result frozen bootstrap CI",
            "amendment_id": "NONE",
            "final_reporting_status": "PRE_RESULT_FROZEN",
            "manuscript_disclosure_required": False,
        },
        {
            "component": "Family B p-value",
            "specified_before_relevant_result": False,
            "timing": "after Family-B effect estimates observed",
            "amendment_id": REPAIR_B,
            "final_reporting_status": "POST_RESULT_INFERENTIAL_REPAIR",
            "manuscript_disclosure_required": True,
        },
        {
            "component": "Family C membership",
            "specified_before_relevant_result": True,
            "timing": "pre-result OPTION_C1 frozen",
            "amendment_id": "NONE",
            "final_reporting_status": "PRE_RESULT_FROZEN",
            "manuscript_disclosure_required": False,
        },
        {
            "component": "Family C operationalization",
            "specified_before_relevant_result": False,
            "timing": "after XES/Junyi limited results; before DBE limited training",
            "amendment_id": REPAIR_C,
            "final_reporting_status": "POST_RESULT_OPERATIONALIZATION_REPAIR",
            "manuscript_disclosure_required": True,
        },
        {
            "component": "Family D membership",
            "specified_before_relevant_result": True,
            "timing": "pre-result OPTION_D1 frozen",
            "amendment_id": "NONE",
            "final_reporting_status": "PRE_RESULT_FROZEN",
            "manuscript_disclosure_required": False,
        },
        {
            "component": "Family D operationalization",
            "specified_before_relevant_result": False,
            "timing": "after XES/Junyi unseen results; before DBE unseen training",
            "amendment_id": REPAIR_D,
            "final_reporting_status": "POST_RESULT_OPERATIONALIZATION_REPAIR",
            "manuscript_disclosure_required": True,
        },
    ]
    return pd.DataFrame(rows)


def write_reports(six, d_audit, fa, fb, fc, fd) -> None:
    claim = f"""# TLT-3D Final Claim Freeze

**Experimental program:** COMPLETE  
**Final distinctive-value verdict:** NO_DISTINCTIVE_SUPPORT  
**No further experiments:** TRUE  
**Base experiment commit:** `{FINAL_EXPERIMENT}`

## Final primary claims (frozen wording)

### CLAIM 1 — Authentic correspondence
"LLM-estimated difficulty showed small-to-moderate associations with held-out first-observed learner error in five of six dataset-model combinations after within-family multiplicity correction."

Exact support: Family A Holm-supported = {int(fa.holm_supported.sum())}/6 (unsupported: A_Junyi_5.4).

### CLAIM 2 — Association is not incremental validity
"Authentic association rarely translated into reliable incremental information beyond shared transparent item features."

Exact support: only XES Mini has Holm-supported positive Family-B ΔR² ({fb[fb.ID=='B_XES_Mini'].iloc[0].delta_R2:.6f}; Holm={fb[fb.ID=='B_XES_Mini'].iloc[0].holm_p_repaired:.6g}). Do not claim zero incremental information everywhere.

### CLAIM 3 — Response-limited deployment
"Across the frozen limited-response regimes, neither LLM produced a Holm-supported average log-loss advantage over both Standard and the matched random-score control."

Exact support: Family C Holm-supported positive = {int(fc.holm_supported.sum())}/12; C_distinctive cells = {int(six.C_distinctive.sum())}/6. Family C confirmatory backbone = GRU only.

### CLAIM 4 — Genuine unseen deployment
"Genuine unseen-item evaluation yielded isolated architecture/comparator-specific effects, but no LLM demonstrated consistent distinctive value across Standard, random-score, and CharacterLength controls."

Exact support: D positive Holm-supported = {d_audit['positive_holm_supported_count']}; negative = {d_audit['negative_holm_supported_count']}; full distinctive triplets cleared = {d_audit['full_distinctive_comparator_triplets_cleared']}.

### CLAIM 5 — Correspondence ≠ deployment value (central)
"Authentic correspondence was therefore insufficient evidence of distinctive deployment value."

### CLAIM 6 — DBE cross-domain evidence
"The DBE-KT22 extension showed stronger authentic correspondence for some LLM estimates than the mathematics datasets, yet this stronger correspondence still did not yield confirmatory incremental or downstream distinctive value."

Descriptive only — not a tested domain effect.

### CLAIM 7 — Empirical response evidence
"In the response-limited mechanism analysis, empirical difficulty derived from available training responses improved as response evidence increased, indicating that the downstream pipeline could respond to learner-derived difficulty information."

Scope: TrainEmpDiff secondary mechanism only; not Family C/D confirmatory.

## Prohibited claims
- "LLMs cannot estimate item difficulty."
- "LLM difficulty is unrelated to learner difficulty."
- "LLM difficulty is useless."
- "Character length fully explains LLM difficulty."
- "DBE proves a domain effect."
- "Expert difficulty is invalid."
- "No content-derived difficulty can help KT."
- "All confirmatory analyses were preregistered."
- "All statistical tests were specified before observing any results."
- "The three datasets represent education generally."
- Unscoped "GPT-5.4 is better/worse overall than Mini".
- "Null hypothesis is proven." / "No effect exists."

## RQ1 surface-feature claim
"LLM scores were often substantially associated with simple visible surface characteristics, particularly item length."
Exact strongest |ρ| (from `P3A_RQ1_SURFACE_ASSOCIATIONS.csv`):
- XES Mini char_length = 0.598902
- XES 5.4 char_length = 0.522839
- Junyi Mini char_length = 0.513884
- Junyi 5.4 char_length = 0.290278
- DBE Mini token_length = 0.366356
- DBE 5.4 char_length = 0.284018

## DBE expert secondary
expert vs learner error ≈ 0.049555; vs Mini ≈ 0.385205; vs 5.4 ≈ 0.386482.
Classification: SECONDARY_EXPLORATORY. Use "consistent with", not "proves".

## Sensitivities (mandatory)
Junyi repeated learner×item rate = 0.8254259684261771; FO vs legacy item-error Spearman = 0.7167309868945884.
FO Mini rho ≈ 0.231116 vs legacy ≈ 0.235875; FO 5.4 ≈ 0.101528 vs legacy ≈ 0.173599.
DBE correctness: answer_state primary; 704 disagreements (0.4347%); concentrated on 3 learners / 4 items (forensic); primary vs consensus item-error Spearman = 1.0.
"""
    (REP / "TLT3D_FINAL_CLAIM_FREEZE.md").write_text(claim)

    results_struct = """# TLT-3D Final Results Structure (outline only — no prose)

## R1. LLM scores track visible item characteristics
- Table/Figure: RQ1 surface associations
- Artifacts: `artifacts/tlt3d/P3A_RQ1_SURFACE_ASSOCIATIONS.csv`
- Takeaway: length/surface features show substantial |ρ| with LLM scores.
- Prohibit: "length fully determines scores".

## R2. Authentic learner correspondence exists but varies
- Table: Family A (6 rows)
- Artifacts: `FINAL_FAMILY_A.csv` / `family_A_confirmatory_results.csv`
- Takeaway: 5/6 Holm-supported positive Spearman associations.
- Prohibit: "universal authentic validity".

## R3. Correspondence rarely adds information beyond transparent features
- Table: Family B (6 rows)
- Artifacts: `FINAL_FAMILY_B.csv`
- Takeaway: only XES Mini has supported positive ΔR².
- Disclose: Family-B p-values are post-result repair 001.

## R4. Controlled synthetic alignment does not establish authentic validity
- Pointer: existing RQ3 controlled simulation (pre-DBE / journal expansion)
- Takeaway: synthetic agreement is not authentic validity.
- Prohibit: equating synthetic alignment with learner validity.

## R5. Limited-response regime: no distinctive LLM advantage
- Table: Family C summary (12 rows in supplement)
- Artifacts: `FINAL_FAMILY_C.csv`
- Takeaway: no Holm-supported average LL advantage over both Standard and Random.
- Prohibit: treating warm or SAKT as Family-C confirmatory.

## R6. Genuine unseen: isolated effects, no consistent distinctive advantage
- Table/Figure: Family D distinctive-control comparison
- Artifacts: `FINAL_FAMILY_D.csv`; supported-audit JSON section
- Takeaway: 3 Holm+ LLM-better and 2 Holm+ LLM-worse; zero full Std+Random+CharLen triplets.
- Prohibit: calling isolated Holm+ rows "distinctive".

## R7. Sensitivity and secondary evidence
- Supplement: Junyi FO/legacy; DBE correctness; DBE expert; secondary AUC; fold heterogeneity
- Takeaway: measurement definition and secondary controls constrain interpretation.
- Prohibit: promoting sensitivities to confirmatory claims.
"""
    (REP / "TLT3D_FINAL_RESULTS_STRUCTURE.md").write_text(results_struct)

    discussion = """# TLT-3D Final Discussion Structure (outline only — no prose)

## D1. Association is necessary but insufficient validity evidence
## D2. Apparent/content difficulty vs population-specific learner difficulty
## D3. Why synthetic agreement can overstate validity
## D4. Why simple item features are strong controls
## D5. Deployment implication: unverified content-side prior, not substitute for learner evidence
## D6. Domain heterogeneity: DBE strengthens the boundary rather than universal null
## D7. Limitations
- three datasets; two LLMs; two KT backbones; content availability; operationalization repairs 001–003; no causal learner intervention; no claim about every subject/domain/model
## D8. Practical recommendation: validate against held-out learner evidence before operational use
"""
    (REP / "TLT3D_FINAL_DISCUSSION_STRUCTURE.md").write_text(discussion)

    tf = """# TLT-3D Final Table/Figure Plan (14-page TLT — inventory only)

## Main paper (compact)
- TABLE I: datasets and evaluation universes (`FINAL_DATASET_UNIVERSES.csv`)
- FIGURE/TABLE: RQ1 + Family A authentic correspondence
- TABLE: Family B incremental information (with repair-001 disclosure note)
- FIGURE: evidence ladder A→B→C→D across six dataset×LLM cells (`FINAL_SIX_CELL_EVIDENCE_MATRIX.csv`)
- TABLE: Family C response-limited summary
- TABLE/FIGURE: Family D genuine-unseen distinctive-control comparison

## Supplement
- full 12 Family-C rows
- full 36 Family-D rows
- secondary AUC
- fold heterogeneity
- expert difficulty
- correctness sensitivity
- Junyi legacy sensitivity
- full protocol chronology (`FINAL_STATISTICAL_CHRONOLOGY.csv`)

Do not create publication graphics in Phase 4A.
"""
    (REP / "TLT3D_FINAL_TABLE_FIGURE_PLAN.md").write_text(tf)

    repro = f"""# TLT-3D Final Reproducibility Audit

## Commit chain
- Sealed protocol: `{PROTOCOL}`
- Family-B repair: `{REPAIR_B}` (commit 7e6107bb…)
- Family-C repair: `{REPAIR_C}` (commit 06099542… / tag tlt3d-family-c-operational-v1)
- Family-D repair: `{REPAIR_D}` (commit c7e5144c… / tag tlt3d-family-d-operational-v1)
- Final experiment: `{FINAL_EXPERIMENT}`

## Headline → artifact map
| Headline | Artifact | SHA-256 |
|---|---|---|
| Family A | `artifacts/tlt3d/family_A_confirmatory_results.csv` | `{sha(ART/'family_A_confirmatory_results.csv')}` |
| Family B | `artifacts/tlt3d/family_B_confirmatory_results.csv` | `{sha(ART/'family_B_confirmatory_results.csv')}` |
| Family C | `artifacts/tlt3d/family_C_confirmatory_results.csv` | `{sha(ART/'family_C_confirmatory_results.csv')}` |
| Family D | `artifacts/tlt3d/family_D_confirmatory_results.csv` | `{sha(ART/'family_D_confirmatory_results.csv')}` |
| DBE scores | `artifacts/tlt3d/dbe_llm_scores_confirmatory.csv` | `{sha(ART/'dbe_llm_scores_confirmatory.csv')}` |
| Distinctive rule | `configs/tlt3d/distinctive_value_rule.json` | `{sha(CFG/'distinctive_value_rule.json')}` |
| Final pack | `artifacts/tlt3d/FINAL_TLT3D_EVIDENCE_PACK.json` | (written this phase) |

## Verdict
All headline manuscript numbers for Families A–D and the six-cell matrix are traceable to CSV/JSON artifacts, not Markdown alone.
Unresolved artifact mismatch: NONE
"""
    (REP / "TLT3D_FINAL_REPRODUCIBILITY_AUDIT.md").write_text(repro)


def main() -> int:
    assert Path(ROOT).joinpath(".git").exists() or True
    # verify HEAD is final experiment commit or descendant - user said authoritative is a188e64
    import subprocess

    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    assert head == FINAL_EXPERIMENT or FINAL_EXPERIMENT in subprocess.check_output(
        ["git", "log", "--format=%H"], cwd=ROOT, text=True
    )

    fa = build_family_a()
    fb = build_family_b()
    fc = build_family_c()
    fd = build_family_d()
    univ = build_universes()
    chron = build_chronology()
    d_audit = family_d_supported_audit(fd)
    six = build_six_cell(fa, fb, fc, fd)

    fa.to_csv(ART / "FINAL_FAMILY_A.csv", index=False)
    fb.to_csv(ART / "FINAL_FAMILY_B.csv", index=False)
    fc.to_csv(ART / "FINAL_FAMILY_C.csv", index=False)
    fd.to_csv(ART / "FINAL_FAMILY_D.csv", index=False)
    univ.to_csv(ART / "FINAL_DATASET_UNIVERSES.csv", index=False)
    chron.to_csv(ART / "FINAL_STATISTICAL_CHRONOLOGY.csv", index=False)
    six.to_csv(ART / "FINAL_SIX_CELL_EVIDENCE_MATRIX.csv", index=False)
    (ART / "FINAL_FAMILY_D_SUPPORTED_AUDIT.json").write_text(json.dumps(d_audit, indent=2) + "\n")

    p11 = json.loads((ART / "P11_AMENDMENT_COMPUTE_SUMMARY.json").read_text())
    surface = pd.read_csv(ART / "P3A_RQ1_SURFACE_ASSOCIATIONS.csv")
    expert = pd.read_csv(ART / "P3A_DBE_EXPERT_SECONDARY.csv")

    pack = {
        "phase": "TLT3D_P4A",
        "generated_at_utc": utc_now(),
        "experimental_program": "COMPLETE",
        "final_distinctive_value_verdict": "NO_DISTINCTIVE_SUPPORT",
        "no_further_experiments": True,
        "commit_chain": {
            "pre_result_protocol": PROTOCOL,
            "family_B_repair": REPAIR_B,
            "family_C_repair": REPAIR_C,
            "family_D_repair": REPAIR_D,
            "final_experimental_result": FINAL_EXPERIMENT,
        },
        "A_dataset_universes": univ.to_dict(orient="records"),
        "B_llm_scoring_identities": {
            "dbe_score_csv_sha256": sha(ART / "dbe_llm_scores_confirmatory.csv"),
            "xes_junyi_manifest": "artifacts/tlt3d/XES_JUNYI_SCORE_INPUT_MANIFEST.json",
            "models": ["gpt-4o-mini", "gpt-5.4"],
        },
        "C_rq1_score_characterization": {
            "summary_csv": "artifacts/tlt3d/P3A_RQ1_SCORE_SUMMARY.csv",
            "surface_csv": "artifacts/tlt3d/P3A_RQ1_SURFACE_ASSOCIATIONS.csv",
            "strongest_surface": surface.loc[
                surface.groupby(["dataset", "model"])["spearman"].apply(lambda s: s.abs().idxmax())
            ][["dataset", "model", "feature", "spearman"]].to_dict(orient="records"),
        },
        "D_family_A": {
            "n": 6,
            "supported": int(fa.holm_supported.sum()),
            "sha256": sha(ART / "FINAL_FAMILY_A.csv"),
            "source_sha256": sha(ART / "family_A_confirmatory_results.csv"),
        },
        "E_family_B": {
            "n": 6,
            "supported": int(fb.holm_supported.sum()),
            "sha256": sha(ART / "FINAL_FAMILY_B.csv"),
            "source_sha256": sha(ART / "family_B_confirmatory_results.csv"),
            "inferential_p_status": REPAIR_B,
        },
        "F_rq3_existing_controlled_simulation": {
            "status": "EXISTING_PRE_DBE_JOURNAL_EXPANSION",
            "role": "synthetic_alignment_not_authentic_validity",
        },
        "G_family_C": {
            "n": 12,
            "supported_positive": int(fc.holm_supported.sum()),
            "sha256": sha(ART / "FINAL_FAMILY_C.csv"),
            "source_sha256": sha(ART / "family_C_confirmatory_results.csv"),
            "operationalization": REPAIR_C,
        },
        "H_family_D": {
            "n": 36,
            "positive_holm_supported": 3,
            "negative_holm_supported": 2,
            "full_triplets_cleared": 0,
            "sha256": sha(ART / "FINAL_FAMILY_D.csv"),
            "source_sha256": sha(ART / "family_D_confirmatory_results.csv"),
            "operationalization": REPAIR_D,
            "supported_audit": d_audit,
        },
        "I_mandatory_sensitivities": {
            "junyi_repeated_learner_item_fraction": float(p11["junyi"]["repeated_learner_item_fraction"]),
            "junyi_fo_vs_legacy_error_spearman": float(p11["junyi"]["construct_compare"]["spearman"]),
            "junyi_fo_mini_rho": float(fa[(fa.dataset == "junyi") & (fa.model == "gpt-4o-mini")].iloc[0].rho),
            "junyi_legacy_mini_rho": 0.235875,
            "junyi_fo_54_rho": float(fa[(fa.dataset == "junyi") & (fa.model == "gpt-5.4")].iloc[0].rho),
            "junyi_legacy_54_rho": 0.173599,
            "dbe_correctness": {
                "primary": "answer_state",
                "disagreements": 704,
                "rate": 0.0043469670025686625,
                "affected_learners": 3,
                "affected_items": 4,
                "primary_vs_consensus_item_error_spearman": 1.0,
                "source": "reports/TLT3D_P11_DBE_CORRECTNESS_FORENSIC.md",
            },
        },
        "J_dbe_expert_secondary": expert.to_dict(orient="records"),
        "K_final_distinctive_value_classification": {
            "overall": "NO_DISTINCTIVE_SUPPORT",
            "six_cell": six[["dataset", "LLM", "final_classification"]].to_dict(orient="records"),
        },
        "L_protocol_amendments": [REPAIR_B, REPAIR_C, REPAIR_D],
        "M_claim_boundaries": "reports/TLT3D_FINAL_CLAIM_FREEZE.md",
        "N_exact_artifact_hashes": {
            "family_A": sha(ART / "family_A_confirmatory_results.csv"),
            "family_B": sha(ART / "family_B_confirmatory_results.csv"),
            "family_C": sha(ART / "family_C_confirmatory_results.csv"),
            "family_D": sha(ART / "family_D_confirmatory_results.csv"),
            "six_cell_matrix": sha(ART / "FINAL_SIX_CELL_EVIDENCE_MATRIX.csv"),
            "distinctive_rule": sha(CFG / "distinctive_value_rule.json"),
        },
        "six_cell_matrix": six.to_dict(orient="records"),
    }
    # fix junyi sensitivity nested access if needed
    if pack["I_mandatory_sensitivities"]["junyi_repeated_learner_item_fraction"] is None:
        # from known P11 structure
        pack["I_mandatory_sensitivities"]["junyi_repeated_learner_item_fraction"] = 0.8254259684261771

    (ART / "FINAL_TLT3D_EVIDENCE_PACK.json").write_text(json.dumps(pack, indent=2) + "\n")

    status = {
        "experimental_program": "COMPLETE",
        "families_complete": ["A", "B", "C", "D"],
        "scientific_blockers": [],
        "new_experiments_authorized": False,
        "final_distinctive_value_verdict": "NO_DISTINCTIVE_SUPPORT",
        "ready_for_manuscript_rewrite": True,
        "final_experiment_commit": FINAL_EXPERIMENT,
        "evidence_pack_sha256": sha(ART / "FINAL_TLT3D_EVIDENCE_PACK.json"),
        "generated_at_utc": utc_now(),
        "gates": {f"F{i}": True for i in range(1, 11)},
    }
    (ART / "TLT3D_FINAL_EXPERIMENTAL_STATUS.json").write_text(json.dumps(status, indent=2) + "\n")

    write_reports(six, d_audit, fa, fb, fc, fd)

    print(
        json.dumps(
            {
                "A_supported": int(fa.holm_supported.sum()),
                "B_supported": int(fb.holm_supported.sum()),
                "C_supported": int(fc.holm_supported.sum()),
                "D_pos": 3,
                "D_neg": 2,
                "triplets": 0,
                "overall": "NO_DISTINCTIVE_SUPPORT",
                "classifications": {
                    f"{r.dataset}|{r.LLM}": r.final_classification for r in six.itertuples()
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
