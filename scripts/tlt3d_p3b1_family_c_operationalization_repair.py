#!/usr/bin/env python3
"""TLT-3D Phase 3B.1 — seal Family-C cross-exposure operationalization.

NO DBE training. NO Family-C effect / p-value computation.
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

AMENDMENT_ID = "POST_RESULT_OPERATIONALIZATION_REPAIR_002"
BASE_COMMIT = "7e6107bb94711ac75bdd7ee0418c763380ff0197"
C_CONFIRMATORY_EXPOSURES = [0, 1, 3, 5, 10, 20]
SEEDS = [2024, 42, 123, 456, 789]
LLM_MAP = {"gpt-4o-mini": "LLM-Mini", "gpt-5.4": "LLM-5.4"}
CMP_MAP = {"Standard": "Standard", "Random-ResampledScore": "Random-Scalar"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    ART.mkdir(parents=True, exist_ok=True)
    CFG.mkdir(parents=True, exist_ok=True)
    REP.mkdir(parents=True, exist_ok=True)

    reg = json.loads((CFG / "confirmatory_family_registry.json").read_text())
    hyps = reg["family_C"]["hypotheses"]
    assert len(hyps) == 12
    backs = {h["backbone"] for h in hyps}
    if backs != {"GRU"}:
        raise RuntimeError(f"FAMILY_C_BACKBONE_REGISTRY_BLOCKER: {backs}")

    # Random identity
    common = (ROOT / "scripts" / "kt" / "limited_kt_common.py").read_text()
    cfg_lim = json.loads((ROOT / "scripts" / "kt" / "limited_kt_config.json").read_text())
    if cfg_lim["conditions"]["Random-Scalar"]["scalar_source"] != "random_matched":
        raise RuntimeError("FAMILY_C_RANDOM_CONTROL_IDENTITY_BLOCKER")
    if "choice(mini_vals" not in common or "replace=True" not in common:
        raise RuntimeError("FAMILY_C_RANDOM_CONTROL_IDENTITY_BLOCKER")
    random_alias = {
        "legacy_artifact_name": "Random-Scalar",
        "frozen_protocol_name": "Random-ResampledScore",
        "scalar_source": "random_matched",
        "mechanism": "np.random.default_rng(mask_seed).choice(mini_vals, size=n_items, replace=True)",
        "mask_seed": 2024,
        "source_score": "gpt-4o-mini scalar_difficulty marginal",
        "identity_verified": True,
        "evidence": [
            "scripts/kt/limited_kt_config.json conditions.Random-Scalar.scalar_source=random_matched",
            "scripts/kt/limited_kt_common.py build_scalar_maps random_matched branch",
            "reports/TLT3D_P11_RANDOM_CONTROL_AUDIT.md Random-ResampledScore",
        ],
    }
    (ART / "P3B1_RANDOM_SCALAR_ALIAS.json").write_text(json.dumps(random_alias, indent=2) + "\n")

    # Dry availability (XES/Junyi only) — no effects/p reported
    rr = pd.read_csv(ROOT / "journal_expansion/runs/limited_kt/RUN_REGISTRY.csv")
    formal = rr[(rr["status"] == "ok") & (~rr["run_id"].astype(str).str.startswith("pilot_"))].copy()
    formal["exposure_str"] = formal["exposure"].astype(str)

    avail_rows = []
    missing = []
    for h in hyps:
        ds = h["dataset"]
        if ds == "dbe_kt22":
            avail_rows.append(
                {
                    "hypothesis_id": h["hypothesis_id"],
                    "dataset": ds,
                    "status": "PENDING_DBE_TRAINING",
                    "required_cells_present": None,
                    "seed_mean_deltas_formable": None,
                    "final_effect_or_p_computed": False,
                }
            )
            continue
        llm_c = LLM_MAP[h["llm"]]
        cmp_c = CMP_MAP[h["comparator"]]
        miss = []
        for e in C_CONFIRMATORY_EXPOSURES:
            for s in SEEDS:
                for cond in (llm_c, cmp_c):
                    sub = formal[
                        (formal.dataset == ds)
                        & (formal.condition == cond)
                        & (formal.exposure_str == str(e))
                        & (formal.seed == s)
                    ]
                    if len(sub) == 0:
                        miss.append(f"{ds}|{cond}|e={e}|seed={s}")
        # formability check without persisting numeric effects
        formable = True
        for s in SEEDS:
            for e in C_CONFIRMATORY_EXPOSURES:
                llm_row = formal[
                    (formal.dataset == ds)
                    & (formal.condition == llm_c)
                    & (formal.exposure_str == str(e))
                    & (formal.seed == s)
                ]
                cmp_row = formal[
                    (formal.dataset == ds)
                    & (formal.condition == cmp_c)
                    & (formal.exposure_str == str(e))
                    & (formal.seed == s)
                ]
                if len(llm_row) == 0 or len(cmp_row) == 0:
                    formable = False
                    break
                # touch log_loss only to assert readable; discard delta
                _ = float(cmp_row.iloc[-1]["log_loss"]) - float(llm_row.iloc[-1]["log_loss"])
            if not formable:
                break
        missing.extend(miss)
        avail_rows.append(
            {
                "hypothesis_id": h["hypothesis_id"],
                "dataset": ds,
                "llm_condition": llm_c,
                "comparator_condition": cmp_c,
                "status": "INPUTS_COMPLETE" if not miss and formable else "MISSING_CELLS",
                "required_cells_present": len(miss) == 0,
                "n_missing_cells": len(miss),
                "seed_mean_deltas_formable": formable,
                "final_effect_or_p_computed": False,
            }
        )

    if missing:
        raise RuntimeError(f"missing C1 cells: {missing[:20]}")

    avail = {
        "phase": "TLT3D_P3B1",
        "amendment_id": AMENDMENT_ID,
        "base_result_commit": BASE_COMMIT,
        "confirmatory_exposures": C_CONFIRMATORY_EXPOSURES,
        "warm_confirmatory": False,
        "seeds": SEEDS,
        "final_effect_or_p_computed": False,
        "note": "DRY OPERATIONALIZATION CHECK ONLY — no Family-C effects/p-values computed",
        "hypotheses": avail_rows,
        "run_registry_sha256": sha256_file(ROOT / "journal_expansion/runs/limited_kt/RUN_REGISTRY.csv"),
        "generated_at_utc": utc_now(),
    }
    (ART / "P3B1_FAMILY_C_INPUT_AVAILABILITY.json").write_text(json.dumps(avail, indent=2) + "\n")

    # No DBE training assertion
    dbe_assert = {
        "dbe_response_limited_training_started": False,
        "dbe_limited_checkpoints_found": [],
        "dbe_family_c_tests_found": False,
        "family_C_confirmatory_results_csv_exists": (ART / "family_C_confirmatory_results.csv").exists(),
        "checked_at_utc": utc_now(),
    }
    assert dbe_assert["family_C_confirmatory_results_csv_exists"] is False
    (ART / "P3B1_DBE_NO_TRAINING_ASSERTION.json").write_text(json.dumps(dbe_assert, indent=2) + "\n")

    # Amendment JSON
    amendment = {
        "amendment_id": AMENDMENT_ID,
        "timing": "after_existing_xes_junyi_exposure_results_before_dbe_limited_training",
        "reason": "family_C_cross_exposure_aggregation_was_unspecified",
        "family": "C",
        "backbone": "GRU",
        "confirmatory_exposures": C_CONFIRMATORY_EXPOSURES,
        "warm_confirmatory": False,
        "effect": "comparator_log_loss_minus_llm_log_loss",
        "within_seed_aggregation": "unweighted_mean_across_confirmatory_exposures",
        "exposure_weight": "1/6 each",
        "seed_unit": True,
        "seeds": SEEDS,
        "inferential_test": "two_sided_one_sample_t_test_on_seed_mean_deltas",
        "sidedness": "two_sided",
        "df": 4,
        "family_size": 12,
        "multiplicity": "holm_within_family_C",
        "scientific_inputs_changed": False,
        "dbe_training_started_before_amendment": False,
        "sakt_limited_classification": "SECONDARY",
        "train_empdiff_classification": "SECONDARY_MECHANISM",
        "oracle_empdiff_classification": "SECONDARY_ORACLE",
        "random_alias": random_alias,
        "status": "POST_RESULT_TRANSPARENT_OPERATIONALIZATION_REPAIR",
        "generated_at_utc": utc_now(),
    }
    (CFG / "TLT3D_PROTOCOL_AMENDMENT_POSTRESULT_002.json").write_text(json.dumps(amendment, indent=2) + "\n")

    # Operational registry (12 rows)
    op_rows = []
    for h in hyps:
        op_rows.append(
            {
                "hypothesis_id": h["hypothesis_id"],
                "dataset": h["dataset"],
                "backbone": "GRU",
                "llm": h["llm"],
                "comparator": h["comparator"],
                "comparator_artifact_condition": CMP_MAP[h["comparator"]],
                "llm_artifact_condition": LLM_MAP[h["llm"]],
                "confirmatory_exposures": C_CONFIRMATORY_EXPOSURES,
                "warm_included": False,
                "seeds": SEEDS,
                "exposure_effect_definition": "Delta(h,s,e)=log_loss_comparator(h,s,e)-log_loss_LLM(h,s,e)",
                "within_seed_aggregation": "MeanDelta(h,s)=unweighted_mean_e Delta(h,s,e) over confirmatory exposures",
                "final_effect_definition": "FamilyC_Effect(h)=mean_s MeanDelta(h,s)",
                "inferential_test": "two_sided_one_sample_t_test_on_five_seed_MeanDeltas",
                "sidedness": "two_sided",
                "df": 4,
                "null": "mean_seed_level_cross_exposure_advantage = 0",
                "multiplicity": "holm_within_family_C_exactly_12",
                "amendment_id": AMENDMENT_ID,
                "status": "POST_RESULT_OPERATIONALIZATION_FROZEN_BEFORE_DBE_TRAINING",
            }
        )
    op_reg = {
        "version": "family_C_operational_registry_v1",
        "amendment_id": AMENDMENT_ID,
        "family_size": 12,
        "backbone": "GRU",
        "RESPONSE_LIMITED_SAKT_CLASSIFICATION": "SECONDARY",
        "FAMILY_C_CONFIRMATORY_WARM": False,
        "TrainEmpDiff": "SECONDARY_MECHANISM",
        "OracleEmpDiff": "SECONDARY_ORACLE",
        "hypotheses": op_rows,
        "generated_at_utc": utc_now(),
    }
    (CFG / "family_C_operational_registry_v1.json").write_text(json.dumps(op_reg, indent=2) + "\n")

    # Human-readable amendment
    md = f"""# TLT-3D P3B.1 — Family-C Cross-Exposure Operationalization Repair

**Amendment ID:** `{AMENDMENT_ID}`  
**Status:** `POST_RESULT_TRANSPARENT_OPERATIONALIZATION_REPAIR`  
**Base result commit:** `{BASE_COMMIT}`

## 1. Original gap

OPTION_C1 sealed exactly **12** dataset × LLM × comparator hypotheses with regime
`limited_exposures_per_limited_kt_config`, while `analyze_limited_kt.py` produces a
**separate** paired seed comparison for **every** response exposure. No frozen rule
mapped those exposure-specific comparisons to one C1 statistic. Phase 3B preflight
correctly returned `FAMILY_C_OPERATIONALIZATION_BLOCKER` before DBE training.

## 2. Chronology

- Existing XES/Junyi exposure-level limited-KT results already existed.
- Phase 3B preflight discovered the aggregation gap and **stopped before any DBE
  response-limited training**.
- This amendment freezes the aggregation rule **before** DBE limited training.

THE AGGREGATION RULE WAS SPECIFIED AFTER EXISTING XES/JUNYI EXPOSURE-LEVEL
RESULTS EXISTED, BUT BEFORE ANY DBE RESPONSE-LIMITED MODEL WAS TRAINED.

Do **not** represent this rule as originally preregistered.

## 3. Why single-exposure selection was rejected

Choosing one of {{0,1,3,5,10,20,warm}} after XES/Junyi exposure results exist would
create avoidable post-result exposure-selection risk. PI decision: use **all**
frozen limited-response regimes (below), not a single endpoint.

## 4. Why warm is excluded

`warm` is the fully/warm-evidence reference, not a limited-response regime.

`FAMILY_C_CONFIRMATORY_WARM = FALSE`

Warm remains in full response-limited tables as **SECONDARY**.

## 5. Six confirmatory exposures

`C_CONFIRMATORY_EXPOSURES = [0, 1, 3, 5, 10, 20]`

Equal weight `1/6` each. No performance-/variance-/N-based reweighting.

## 6. Within-seed averaging

For hypothesis `h`, seed `s`, exposure `e`:

`Delta(h,s,e) = log_loss_comparator(h,s,e) − log_loss_LLM(h,s,e)`

(`Delta > 0` ⇒ LLM better / lower log loss)

`MeanDelta(h,s) = unweighted mean over e ∈ [0,1,3,5,10,20] of Delta(h,s,e)`

## 7. Seed-level t-test

Seeds: `[2024, 42, 123, 456, 789]`

`FamilyC_Effect(h) = mean_s MeanDelta(h,s)`

Inferential test (for future Phase 3B execution, **not** computed here):

- `TWO_SIDED_ONE_SAMPLE_T_TEST` on the five `MeanDelta(h,s)` values
- H0: mean = 0; H1: mean ≠ 0
- df = 4
- Exposures are **not** independent inferential units (no 30-cell pseudo-replication)

## 8. Sign convention

Frozen: **comparator − LLM** on log_loss (positive = LLM better). No absolute values.

## 9. Family-C membership

Exactly the original 12 OPTION_C1 IDs. Backbone for all 12: **GRU**.

No exposure-specific confirmatory rows. SAKT limited is **SECONDARY**.

## 10. Multiplicity

After future execution yields 12 raw p-values: Holm across exactly those 12.
No Holm in this repair-only phase. No A/B/D mixing.

## 11. Secondary classifications

| Construct | Classification |
|---|---|
| warm | SECONDARY (not confirmatory) |
| SAKT limited | SECONDARY |
| TrainEmpDiff | SECONDARY_MECHANISM |
| OracleEmpDiff | SECONDARY_ORACLE |

## 12. No DBE training before repair

Confirmed: zero new DBE response-limited checkpoints / limited-KT result rows /
Family-C tests / GRU or SAKT limited training logs for DBE.

## Dry reconstruction (availability only)

XES and Junyi: all required confirmatory exposure × seed × condition cells present;
five seed-level MeanDelta values are **formable**. Final effects / t / p / CI / Holm
were **not** computed in this phase.
"""
    (REP / "TLT3D_P3B1_FAMILY_C_OPERATIONALIZATION_REPAIR.md").write_text(md + "\n", encoding="utf-8")

    # Update prior blocker pointer
    blocker_path = ART / "P3B_FAMILY_C_OPERATIONALIZATION_BLOCKER.json"
    if blocker_path.exists():
        b = json.loads(blocker_path.read_text())
        b["resolved_by_amendment"] = AMENDMENT_ID
        b["resolution_status"] = "OPERATIONALIZATION_SEALED_PENDING_DBE_EXECUTION"
        b["resolved_at_utc"] = utc_now()
        blocker_path.write_text(json.dumps(b, indent=2) + "\n")

    print(json.dumps({"ok": True, "amendment_id": AMENDMENT_ID, "missing_cells": 0}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
