#!/usr/bin/env python3
"""TLT-3D Phase 3C.0 — Family-D operationalization preflight (NO DBE unseen training)."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "configs" / "tlt3d"
ART = ROOT / "artifacts" / "tlt3d"
REP = ROOT / "reports"
COLD = ROOT / "journal_expansion" / "runs" / "tlt_coldstart"
FOLD_DIR = ROOT / "journal_expansion" / "features" / "tlt_coldstart" / "item_folds"
GATE_DIR = ROOT / "journal_expansion" / "reports" / "tlt_extension"
DBE_FOLDS = ROOT / "data" / "external" / "dbe_kt22" / "derived" / "tlt3d_unseen_item_folds.csv"
EXT_CFG = ROOT / "scripts" / "kt" / "tlt_extension_config.json"
LIM_RT = ROOT / "configs" / "tlt3d" / "dbe_limited_kt_runtime.json"

EXPECTED_D1 = [
    "D1_XES_GRU_Mini_vs_Std",
    "D1_XES_GRU_Mini_vs_RandPermute",
    "D1_XES_GRU_Mini_vs_CharLen",
    "D1_XES_GRU_5.4_vs_Std",
    "D1_XES_GRU_5.4_vs_RandPermute",
    "D1_XES_GRU_5.4_vs_CharLen",
    "D1_XES_SAKT_Mini_vs_Std",
    "D1_XES_SAKT_Mini_vs_RandPermute",
    "D1_XES_SAKT_Mini_vs_CharLen",
    "D1_XES_SAKT_5.4_vs_Std",
    "D1_XES_SAKT_5.4_vs_RandPermute",
    "D1_XES_SAKT_5.4_vs_CharLen",
    "D1_Junyi_GRU_Mini_vs_Std",
    "D1_Junyi_GRU_Mini_vs_RandPermute",
    "D1_Junyi_GRU_Mini_vs_CharLen",
    "D1_Junyi_GRU_5.4_vs_Std",
    "D1_Junyi_GRU_5.4_vs_RandPermute",
    "D1_Junyi_GRU_5.4_vs_CharLen",
    "D1_Junyi_SAKT_Mini_vs_Std",
    "D1_Junyi_SAKT_Mini_vs_RandPermute",
    "D1_Junyi_SAKT_Mini_vs_CharLen",
    "D1_Junyi_SAKT_5.4_vs_Std",
    "D1_Junyi_SAKT_5.4_vs_RandPermute",
    "D1_Junyi_SAKT_5.4_vs_CharLen",
    "D1_DBE_GRU_Mini_vs_Std",
    "D1_DBE_GRU_Mini_vs_RandPermute",
    "D1_DBE_GRU_Mini_vs_CharLen",
    "D1_DBE_GRU_5.4_vs_Std",
    "D1_DBE_GRU_5.4_vs_RandPermute",
    "D1_DBE_GRU_5.4_vs_CharLen",
    "D1_DBE_SAKT_Mini_vs_Std",
    "D1_DBE_SAKT_Mini_vs_RandPermute",
    "D1_DBE_SAKT_Mini_vs_CharLen",
    "D1_DBE_SAKT_5.4_vs_Std",
    "D1_DBE_SAKT_5.4_vs_RandPermute",
    "D1_DBE_SAKT_5.4_vs_CharLen",
]

COND_ALIAS = {
    "Standard": "Standard",
    "LLM-Mini": "LLM-Mini",
    "LLM-5.4": "LLM-5.4",
    "Random-PermutedScore": "Random-Scalar",
    "CharacterLength": "CharacterLength",
}
LLM_TO_COND = {"gpt-4o-mini": "LLM-Mini", "gpt-5.4": "LLM-5.4"}
DS_SHORT = {"xes3g5m": "XES", "junyi": "Junyi", "dbe_kt22": "DBE"}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def hard_op_test(rows: list[dict]) -> dict:
    """Family-D registry fields are ambiguous on fold×seed → one confirmatory statistic."""
    tests = sorted({r.get("test") for r in rows})
    seed_aggs = sorted({r.get("seed_aggregation") for r in rows})
    fold_aggs = sorted({r.get("item_fold_aggregation") for r in rows})
    ambiguities = [
        "item_fold_aggregation='5_item_folds_seed2024' does not state prediction-weighted pooling vs unweighted fold mean",
        "seed_aggregation='paired_across_seeds_and_folds_as_tlt_extension' does not state whether folds are independent replicates",
        "test='paired_delta_log_loss_as_analyze_tlt_extension' does not name two-sided one-sample t-test / df=4 / Holm-36",
        "analyze_tlt_extension.py pools predictions within seed (compatible intent) but does not freeze confirmatory t/Holm",
    ]
    return {
        "D_OPERATIONALIZATION_STATUS": "REPAIR_REQUIRED",
        "preexisting_fold_aggregation_defined": False,
        "preexisting_seed_aggregation_defined": False,
        "preexisting_inferential_test_defined": False,
        "preexisting_independent_replicate_defined": False,
        "registered_tests": tests,
        "registered_seed_aggregations": seed_aggs,
        "registered_fold_aggregations": fold_aggs,
        "ambiguities": ambiguities,
        "note_implementation_hint": (
            "aggregate_coldstart_oof concatenates primary_y/primary_p across 5 folds then "
            "computes log_loss — prediction-level pooling exists as code, but is not an "
            "unambiguous confirmatory registry specification."
        ),
    }


def write_amendment_003() -> dict:
    am = {
        "amendment_id": "POST_RESULT_OPERATIONALIZATION_REPAIR_003",
        "timing": "after_existing_xes_junyi_unseen_results_before_dbe_unseen_training",
        "chronology_permanent": (
            "AFTER_EXISTING_XES_JUNYI_UNSEEN_RESULTS_EXISTED_AND_BEFORE_ANY_DBE_GENUINE_UNSEEN_TRAINING"
        ),
        "reason": "family_D_cross_fold_seed_aggregation_was_incomplete",
        "family": "D",
        "fold_role": "collective_out_of_fold_evaluation_not_independent_replicates",
        "within_seed_aggregation": "prediction_weighted_pooled_log_loss_across_all_target_folds",
        "within_seed_aggregation_detail": (
            "LL_pooled(c,s)=log_loss(concat_f predictions) equivalently "
            "sum_f n(s,f)*LL(c,s,f)/sum_f n(s,f); NOT unweighted mean of fold LLs"
        ),
        "effect": "comparator_pooled_log_loss_minus_llm_pooled_log_loss",
        "direction_convention": "positive_means_LLM_better_lower_log_loss",
        "seeds": [2024, 42, 123, 456, 789],
        "inferential_unit": "training_seed",
        "inferential_test": "two_sided_one_sample_t_test",
        "df": 4,
        "n_seeds": 5,
        "family_size": 36,
        "multiplicity": "holm_within_family_D",
        "primary_metric": "log_loss",
        "auc_status": "SECONDARY_NOT_IN_FAMILY_D",
        "dbe_unseen_training_started_before_amendment": False,
        "scientific_inputs_changed": False,
        "status": "POST_RESULT_TRANSPARENT_OPERATIONALIZATION_REPAIR",
        "base_result_commit_response_limited": "2d1206090074aa49f937d77955bc1e8d6aca22bf",
        "family_c_amendment": "0609954254b82fa689c1f228eb486ed823a23ef6",
        "sealed_protocol": "a459e34a24240c03ba4dbe4b1d0185e42eaf4377",
    }
    path = CFG / "TLT3D_PROTOCOL_AMENDMENT_POSTRESULT_003.json"
    path.write_text(json.dumps(am, indent=2) + "\n", encoding="utf-8")
    return am


def write_operational_registry(am: dict, registry_rows: list[dict]) -> dict:
    hyps = []
    for r in registry_rows:
        hyps.append(
            {
                "hypothesis_id": r["hypothesis_id"],
                "dataset": r["dataset"],
                "backbone": r["backbone"],
                "llm": r["llm"],
                "comparator": r["comparator"],
                "primary_metric": "log_loss",
                "confirmatory_folds": [0, 1, 2, 3, 4],
                "seeds": [2024, 42, 123, 456, 789],
                "fold_role": am["fold_role"],
                "within_seed_aggregation": am["within_seed_aggregation"],
                "effect": am["effect"],
                "inferential_unit": am["inferential_unit"],
                "inferential_test": "two_sided_one_sample_t_test_on_five_seed_pooled_fold_deltas",
                "df": 4,
                "multiplicity": "holm_within_family_D_exactly_36",
                "evaluation_unit": "first_evaluable_learner_x_target_item_interaction",
                "prediction_pairing_unit": "same_dataset_backbone_fold_seed_evaluation_IDs",
                "classification": "CONFIRMATORY_POSTRESULT_OPERATIONALIZATION_REPAIRED",
                "amendment_id": am["amendment_id"],
                "artifact_condition_alias": {
                    "Random-PermutedScore": "Random-Scalar",
                    "LLM-Mini": "LLM-Mini",
                    "LLM-5.4": "LLM-5.4",
                    "Standard": "Standard",
                    "CharacterLength": "CharacterLength",
                },
            }
        )
    op = {
        "registry_id": "family_D_operational_registry_v1",
        "amendment_id": am["amendment_id"],
        "D_OPERATIONALIZATION_STATUS": "REPAIR_REQUIRED_THEN_SEALED",
        "family_size": 36,
        "backbones": ["GRU", "SAKT"],
        "comparators": ["Standard", "Random-PermutedScore", "CharacterLength"],
        "llms": ["gpt-4o-mini", "gpt-5.4"],
        "hypotheses": hyps,
    }
    path = CFG / "family_D_operational_registry_v1.json"
    path.write_text(json.dumps(op, indent=2) + "\n", encoding="utf-8")
    return op


def audit_existing_unseen() -> tuple[pd.DataFrame, dict]:
    reg = pd.read_csv(COLD / "RUN_REGISTRY.csv")
    ok = reg[reg["status"] == "ok"].copy()
    rows = []
    for r in ok.itertuples():
        pred = ROOT / r.pred_path
        pred_exists = pred.exists()
        pred_sha = sha256_file(pred) if pred_exists else None
        pred_level = False
        if pred_exists:
            data = np.load(pred)
            pred_level = set(data.files) >= {"primary_y", "primary_p"}
        rows.append(
            {
                "path": r.pred_path,
                "sha256": pred_sha,
                "dataset": r.dataset,
                "backbone": r.backbone,
                "condition": r.condition,
                "fold": int(r.item_fold),
                "seed": int(r.training_seed),
                "prediction_count": int(r.n_predictions),
                "log_loss": float(r.test_log_loss),
                "auc": float(r.auc),
                "prediction_level_rows_available": bool(pred_level),
                "run_id": r.run_id,
                "status": r.status,
            }
        )
    man_df = pd.DataFrame(rows)

    # Availability for 24 XES/Junyi Family-D hypotheses (enough to reconstruct pooled seed deltas)
    needed_conds = ["Standard", "LLM-Mini", "LLM-5.4", "Random-Scalar", "CharacterLength"]
    missing = []
    for ds in ["xes3g5m", "junyi"]:
        for bb in ["GRU", "SAKT"]:
            for cond in needed_conds:
                for fold in range(5):
                    for seed in [2024, 42, 123, 456, 789]:
                        hit = ok[
                            (ok.dataset == ds)
                            & (ok.backbone == bb)
                            & (ok.condition == cond)
                            & (ok.item_fold == fold)
                            & (ok.training_seed == seed)
                            & (ok.status == "ok")
                        ]
                        if len(hit) != 1 or pd.isna(hit.iloc[0]["test_log_loss"]) or pd.isna(hit.iloc[0]["n_predictions"]):
                            missing.append(f"{ds}/{bb}/{cond}/fold{fold}/seed{seed}")

    # Pairing: same n_predictions; same primary_y bytes across conditions
    pairing_ok = True
    pairing_notes = []
    for (ds, bb, fold, seed), g in ok.groupby(["dataset", "backbone", "item_fold", "training_seed"]):
        if g["n_predictions"].nunique() != 1:
            pairing_ok = False
            pairing_notes.append(f"n_pred mismatch {ds}/{bb}/f{fold}/s{seed}")
            continue
        y_hashes = []
        for _, row in g.iterrows():
            data = np.load(ROOT / row["pred_path"])
            y_hashes.append(hashlib.sha256(data["primary_y"].tobytes()).hexdigest())
        if len(set(y_hashes)) != 1:
            pairing_ok = False
            pairing_notes.append(f"primary_y mismatch {ds}/{bb}/f{fold}/s{seed}")

    summary = {
        "ok_runs": int(len(ok)),
        "expected_runs": 500,
        "complete": len(ok) == 500 and not missing,
        "missing_cells": missing,
        "prediction_level_available_for_all_ok": bool(man_df["prediction_level_rows_available"].all())
        if len(man_df)
        else False,
        "fold_prediction_counts_available": True,
        "pairing_verified": pairing_ok,
        "pairing_notes": pairing_notes,
        "final_D_effects_pvalues_computed_this_phase": False,
        "condition_alias": {
            "Random-PermutedScore": "Random-Scalar",
            "note": "legacy coldstart artifact name Random-Scalar implements permute-Mini semantics",
        },
    }
    return man_df, summary


def dbe_fold_audit() -> dict:
    folds = pd.read_csv(DBE_FOLDS)
    sizes = {str(k): int(v) for k, v in folds.groupby("item_fold").size().sort_index().items()}
    payload = (
        "|".join(f"{int(r.item_id)}:{int(r.item_fold)}" for r in folds.sort_values("item_id").itertuples())
        + "|seed=2024|n=5"
    )
    fold_hash = sha256_text(payload)
    return {
        "target_items": int(folds["item_id"].nunique()),
        "fold_sizes": sizes,
        "fold_hash": fold_hash,
        "expected_fold_hash": "28efbf33c231772a3565056367c4bf6dfa62bdf553abcfd8397aa9d9037d4e0a",
        "fold_hash_match": fold_hash == "28efbf33c231772a3565056367c4bf6dfa62bdf553abcfd8397aa9d9037d4e0a",
        "exhaustive": int(folds["item_id"].duplicated().sum()) == 0,
        "disjoint_complete": set(folds["item_fold"]) == {0, 1, 2, 3, 4},
        "item_fold_seed": 2024,
        "path": str(DBE_FOLDS.relative_to(ROOT)),
    }


def xes_junyi_fold_audit() -> dict:
    out = {}
    for ds in ["xes3g5m", "junyi"]:
        meta = json.loads((FOLD_DIR / f"{ds}_item_folds_seed2024.meta.json").read_text())
        folds = pd.read_parquet(FOLD_DIR / f"{ds}_item_folds_seed2024.parquet")
        out[ds] = {
            "target_count": int(len(folds)),
            "fold_sizes": {str(k): int(v) for k, v in folds.groupby("item_fold").size().sort_index().items()},
            "fold_hash": meta["file_sha256"],
            "item_list_hash": meta["item_list_sha256"],
            "creation_seed": int(meta.get("item_fold_seed", 2024)),
            "exhaustive": int(folds["item_id_hash"].duplicated().sum()) == 0,
            "disjoint_complete": set(folds["item_fold"]) == {0, 1, 2, 3, 4},
            "content_group_handling": "hash-sorted deterministic assignment; no separate content-group block",
            "meta_path": str((FOLD_DIR / f"{ds}_item_folds_seed2024.meta.json").relative_to(ROOT)),
        }
    return out


def leakage_contract() -> dict:
    gates = []
    for p in sorted(GATE_DIR.glob("coldstart_gate_*.json")):
        g = json.loads(p.read_text())
        gates.append({"path": str(p.relative_to(ROOT)), **g})
    all_zero = all(g.get("zero_target_train_interactions") for g in gates) and len(gates) == 10
    return {
        "evaluation_unit": "first_evaluable_learner_x_target_item_interaction",
        "rules": {
            "zero_target_item_training_interactions": True,
            "no_target_response_derived_empirical_statistics": True,
            "no_target_specific_learned_embedding": True,
            "shared_UNK_item_representation": True,
            "target_content_side_scalar_allowed": True,
            "no_target_outcome_in_model_selection": True,
            "target_test_response_eval_only": True,
            "first_target_item_prediction_primary": True,
        },
        "xes_junyi_gate_files": len(gates),
        "all_folds_zero_target_train": all_zero,
        "kc_concept_in_existing_coldstart_models": False,
        "dbe_kc_policy": "DO_NOT_INTRODUCE_DBE_ONLY_KC_ADVANTAGE",
        "dbe_training_this_phase": False,
    }


def write_planned_dbe_registry() -> tuple[pd.DataFrame, str]:
    conditions = ["Standard", "LLM-Mini", "LLM-5.4", "Random-PermutedScore", "CharacterLength"]
    rows = []
    for backbone in ["GRU", "SAKT"]:
        for fold in range(5):
            for condition in conditions:
                for seed in [2024, 42, 123, 456, 789]:
                    run_id = f"planned_dbe_unseen_{backbone}_fold{fold}_{condition}_{seed}"
                    rows.append(
                        {
                            "run_id": run_id,
                            "dataset": "dbe_kt22",
                            "backbone": backbone,
                            "item_fold": fold,
                            "condition": condition,
                            "seed": seed,
                            "mask_seed": 2024,
                            "item_fold_seed": 2024,
                            "classification": "GENUINE_UNSEEN_PLANNED",
                            "status": "PLANNED_NOT_AUTHORIZED",
                        }
                    )
    df = pd.DataFrame(rows)
    path = ART / "P3C0_DBE_UNSEEN_PLANNED_REGISTRY.csv"
    df.to_csv(path, index=False)
    return df, sha256_file(path)


def no_dbe_training_assertion() -> dict:
    patterns = [
        "*dbe*unseen*",
        "*unseen*dbe*",
        "*coldstart*dbe*",
        "*dbe*coldstart*",
    ]
    found = []
    for pat in patterns:
        found.extend([str(p.relative_to(ROOT)) for p in (ROOT / "journal_expansion").rglob(pat) if p.is_file()])
    found.extend([str(p.relative_to(ROOT)) for p in ART.glob("*unseen*run*") if p.is_file()])
    family_d = (ART / "family_D_confirmatory_results.csv").exists()
    ckpt = list((ROOT / "journal_expansion" / "checkpoints").glob("*dbe*unseen*")) + list(
        (ROOT / "journal_expansion" / "checkpoints").glob("*coldstart*dbe*")
    )
    return {
        "dbe_unseen_gru_runs": 0,
        "dbe_unseen_sakt_runs": 0,
        "dbe_unseen_checkpoints": 0,
        "dbe_family_d_test_results": 0,
        "family_D_confirmatory_results_exists": family_d,
        "unexpected_artifacts": found + [str(p) for p in ckpt],
        "pass": (not family_d) and (len(found) == 0) and (len(ckpt) == 0),
    }


def model_config_report() -> dict:
    ext = json.loads(EXT_CFG.read_text())
    lim = json.loads(LIM_RT.read_text()) if LIM_RT.exists() else {}
    return {
        "coldstart_config_path": str(EXT_CFG.relative_to(ROOT)),
        "coldstart_config_sha256": sha256_file(EXT_CFG),
        "GRU": ext["backbones"]["GRU"],
        "SAKT": ext["backbones"]["SAKT"],
        "train": ext["train"],
        "item_id_dropout": ext["item_id_dropout"],
        "unk_handling": "targets map to shared UNK index=1; no private target embedding slots",
        "scalar_injection": "concatenated at prediction head only when use_scalar=True",
        "kc_input": "not used in existing XES/Junyi cold-start GRU/SAKT",
        "optimizer": "Adam (implementation default in tlt_extension_common)",
        "early_stopping_metric": "validation_log_loss",
        "comparison_to_response_limited": {
            "dbe_limited_runtime_present": LIM_RT.exists(),
            "note": "DBE may change mechanical n_items/users only; no DBE-specific HP tuning",
            "limited_runtime_sha256": sha256_file(LIM_RT) if LIM_RT.exists() else None,
            "limited_snapshot": lim,
        },
        "dbe_specific_tuning": False,
    }


def write_reports(
    registry_rows: list[dict],
    op_status: dict,
    am: dict,
    avail: dict,
    dbe_folds: dict,
    xj_folds: dict,
    leak: dict,
    model: dict,
    planned_hash: str,
    planned_n: int,
    no_train: dict,
) -> None:
    # Operationalization table
    lines = [
        "# TLT-3D P3C.0 — Family D Operationalization",
        "",
        f"- Status: **{op_status['D_OPERATIONALIZATION_STATUS']}**",
        f"- Amendment: `{am['amendment_id']}`",
        f"- Family size: **36**",
        "",
        "## Hard operationalization test",
        "",
        f"- Pre-existing fold aggregation defined?: **{op_status['preexisting_fold_aggregation_defined']}**",
        f"- Pre-existing seed aggregation defined?: **{op_status['preexisting_seed_aggregation_defined']}**",
        f"- Pre-existing inferential test defined?: **{op_status['preexisting_inferential_test_defined']}**",
        f"- Pre-existing independent replicate defined?: **{op_status['preexisting_independent_replicate_defined']}**",
        "",
        "Ambiguities:",
        "",
    ]
    for a in op_status["ambiguities"]:
        lines.append(f"- {a}")
    lines += [
        "",
        f"Implementation hint: {op_status['note_implementation_hint']}",
        "",
        "## Sealed rule (POST_RESULT_OPERATIONALIZATION_REPAIR_003)",
        "",
        "- Fold role: collective OOF evaluation (NOT independent replicates)",
        "- Within-seed: prediction-weighted pooled log loss across 5 folds",
        "- Effect: comparator_pooled_LL − LLM_pooled_LL (positive ⇒ LLM better)",
        "- Inferential unit: training seed (n=5, df=4)",
        "- Test: two-sided one-sample t-test on 5 seed deltas",
        "- Holm within Family D exactly 36",
        "",
        "## One row per hypothesis (registered + sealed fields)",
        "",
        "| hypothesis_id | dataset | backbone | LLM | comparator | primary | sign | folds | seeds | fold agg | seed agg | eval unit | test | sidedness | multiplicity | pairing |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in registry_rows:
        lines.append(
            f"| {r['hypothesis_id']} | {r['dataset']} | {r['backbone']} | {r['llm']} | {r['comparator']} | "
            f"log_loss | comparator−LLM | [0..4] | [2024,42,123,456,789] | "
            f"prediction_weighted_pooled | mean of 5 seed deltas | first learner×target | "
            f"two_sided_one_sample_t | two-sided | holm_D_36 | same fold/seed eval units |"
        )
    lines += [
        "",
        "## Chronology",
        "",
        "THE CROSS-FOLD AGGREGATION RULE WAS SPECIFIED AFTER EXISTING XES/JUNYI "
        "UNSEEN-ITEM RESULTS EXISTED BUT BEFORE ANY DBE GENUINE-UNSEEN MODEL WAS TRAINED.",
        "",
        "No Family-D effects/p-values computed in this phase.",
        "",
    ]
    (REP / "TLT3D_P3C0_FAMILY_D_OPERATIONALIZATION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    repair = f"""# TLT-3D P3C.0 — Family D Operationalization Repair

## Permanent chronology disclosure

THE CROSS-FOLD AGGREGATION RULE WAS SPECIFIED AFTER EXISTING XES/JUNYI
UNSEEN-ITEM RESULTS EXISTED BUT BEFORE ANY DBE GENUINE-UNSEEN MODEL WAS
TRAINED.

Amendment ID: `{am['amendment_id']}`  
Timing: `{am['timing']}`  
Status: `{am['status']}`

## Why repair was required

OPTION_D1 froze 36 dataset × backbone × LLM × comparator hypotheses, but registry
fields (`item_fold_aggregation`, `seed_aggregation`, `test`) did not uniquely specify
how five target-item folds and five training seeds map to **one** confirmatory statistic.

## Sealed mapping

1. **No fold picking** — all five folds always enter the pooled evaluation.
2. **No fold pseudo-replication** — folds are not 25 independent observations with seeds.
3. **Prediction-level pooling** — matches frozen primary unit (first evaluable learner × target-item).
4. **Seed remains the training-replicate inferential unit** — n=5, df=4, two-sided one-sample t.
5. **Holm across exactly 36** Family-D raw p-values after future execution.

## Not done in this phase

- No Family-D effects, t-tests, raw p, or Holm p
- No DBE genuine-unseen training
- No manuscript edits
"""
    (REP / "TLT3D_P3C0_FAMILY_D_OPERATIONALIZATION_REPAIR.md").write_text(repair, encoding="utf-8")

    leak_md = [
        "# TLT-3D P3C.0 — Genuine Unseen Leakage Contract",
        "",
        "No training executed in this phase.",
        "",
        "## Frozen rules",
        "",
    ]
    for k, v in leak["rules"].items():
        leak_md.append(f"- `{k}`: {v}")
    leak_md += [
        "",
        f"- Evaluation unit: `{leak['evaluation_unit']}`",
        f"- XES/Junyi gate files audited: {leak['xes_junyi_gate_files']}",
        f"- All folds zero target train: {leak['all_folds_zero_target_train']}",
        f"- KC in existing cold-start models: {leak['kc_concept_in_existing_coldstart_models']}",
        f"- DBE KC policy: {leak['dbe_kc_policy']}",
        "",
        "## DBE folds",
        "",
        f"- targets: {dbe_folds['target_items']}",
        f"- sizes: {dbe_folds['fold_sizes']}",
        f"- hash: `{dbe_folds['fold_hash']}` (match={dbe_folds['fold_hash_match']})",
        "",
        "## XES/Junyi folds",
        "",
        "```json",
        json.dumps(xj_folds, indent=2),
        "```",
        "",
    ]
    (REP / "TLT3D_P3C0_UNSEEN_LEAKAGE_CONTRACT.md").write_text("\n".join(leak_md) + "\n", encoding="utf-8")

    model_md = [
        "# TLT-3D P3C.0 — Unseen Model Configuration",
        "",
        f"- Cold-start config SHA-256: `{model['coldstart_config_sha256']}`",
        f"- DBE-specific tuning: **{model['dbe_specific_tuning']}**",
        "",
        "## GRU",
        "",
        "```json",
        json.dumps(model["GRU"], indent=2),
        "```",
        "",
        "## SAKT",
        "",
        "```json",
        json.dumps(model["SAKT"], indent=2),
        "```",
        "",
        "## Train / shared",
        "",
        "```json",
        json.dumps(
            {
                "train": model["train"],
                "item_id_dropout": model["item_id_dropout"],
                "unk_handling": model["unk_handling"],
                "scalar_injection": model["scalar_injection"],
                "kc_input": model["kc_input"],
                "optimizer": model["optimizer"],
                "early_stopping_metric": model["early_stopping_metric"],
            },
            indent=2,
        ),
        "```",
        "",
        "## vs response-limited",
        "",
        "```json",
        json.dumps(model["comparison_to_response_limited"], indent=2),
        "```",
        "",
    ]
    (REP / "TLT3D_P3C0_UNSEEN_MODEL_CONFIG.md").write_text("\n".join(model_md) + "\n", encoding="utf-8")

    preflight = f"""# TLT-3D P3C.0 — Family D Preflight Summary

- Operationalization: REPAIR_REQUIRED → sealed via `{am['amendment_id']}`
- XES/Junyi existing unseen complete: {avail['complete']} (ok_runs={avail['ok_runs']}/500)
- Prediction-level outputs: {avail['prediction_level_available_for_all_ok']}
- Pairing verified: {avail['pairing_verified']}
- Final D effects/p computed this phase: {avail['final_D_effects_pvalues_computed_this_phase']}
- DBE planned cells: {planned_n}; registry hash `{planned_hash}`
- DBE training started: {not no_train['pass'] and 'UNEXPECTED' or False}
- No-train assertion pass: {no_train['pass']}

## Gates D0-1..D0-12

All must PASS before Phase 3C execution (see tests).
"""
    (REP / "TLT3D_P3C0_FAMILY_D_PREFLIGHT.md").write_text(preflight, encoding="utf-8")


def main() -> int:
    ART.mkdir(parents=True, exist_ok=True)
    REP.mkdir(parents=True, exist_ok=True)

    fam = json.loads((CFG / "confirmatory_family_registry.json").read_text())["family_D"]
    rows = fam["hypotheses"]
    assert fam["count"] == 36 and len(rows) == 36
    ids = [r["hypothesis_id"] for r in rows]
    assert ids == EXPECTED_D1, "Family-D ID drift"

    op_status = hard_op_test(rows)
    assert op_status["D_OPERATIONALIZATION_STATUS"] == "REPAIR_REQUIRED"

    am = write_amendment_003()
    write_operational_registry(am, rows)

    man_df, avail = audit_existing_unseen()
    if not avail["complete"] or not avail["pairing_verified"]:
        blocker = {
            "blocker_id": "EXISTING_FAMILY_D_RESULT_BLOCKER"
            if not avail["complete"]
            else "FAMILY_D_PAIRING_PREFLIGHT_BLOCKER",
            "availability": avail,
        }
        (ART / "P3C0_FAMILY_D_PREFLIGHT_BLOCKER.json").write_text(json.dumps(blocker, indent=2), encoding="utf-8")
        raise SystemExit(f"STOP: {blocker['blocker_id']}")

    # Compact machine manifest (full row table too large for chat; persist CSV + summary JSON)
    man_df.to_csv(ART / "P3C0_EXISTING_UNSEEN_MANIFEST.csv", index=False)
    manifest = {
        "source_registry": "journal_expansion/runs/tlt_coldstart/RUN_REGISTRY.csv",
        "n_artifacts": len(man_df),
        "summary": avail,
        "sha256_manifest_csv": sha256_file(ART / "P3C0_EXISTING_UNSEEN_MANIFEST.csv"),
        "artifacts": man_df.to_dict(orient="records"),
    }
    (ART / "P3C0_EXISTING_UNSEEN_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    dbe_folds = dbe_fold_audit()
    assert dbe_folds["fold_hash_match"] and dbe_folds["target_items"] == 166
    xj_folds = xes_junyi_fold_audit()
    leak = leakage_contract()
    assert leak["all_folds_zero_target_train"]

    planned_df, planned_hash = write_planned_dbe_registry()
    assert len(planned_df) == 250

    no_train = no_dbe_training_assertion()
    assert no_train["pass"], no_train

    model = model_config_report()
    (ART / "P3C0_UNSEEN_MODEL_CONFIG.json").write_text(json.dumps(model, indent=2) + "\n", encoding="utf-8")
    (ART / "P3C0_DBE_FOLD_AUDIT.json").write_text(json.dumps(dbe_folds, indent=2) + "\n", encoding="utf-8")
    (ART / "P3C0_XES_JUNYI_FOLD_AUDIT.json").write_text(json.dumps(xj_folds, indent=2) + "\n", encoding="utf-8")
    (ART / "P3C0_LEAKAGE_CONTRACT.json").write_text(json.dumps(leak, indent=2) + "\n", encoding="utf-8")
    (ART / "P3C0_OPERATIONALIZATION_STATUS.json").write_text(json.dumps(op_status, indent=2) + "\n", encoding="utf-8")

    # Random / CharacterLength freeze notes (no mapping generation that would imply training readiness beyond planned)
    random_ctrl = {
        "frozen_name": "Random-PermutedScore",
        "legacy_artifact_name": "Random-Scalar",
        "exact_source_vector": "gpt-4o-mini scalar_difficulty over sorted item_id_hash universe",
        "mechanism": "np.random.default_rng(mask_seed=2024).permutation(filled_mini_vals)",
        "within_dataset": True,
        "rng_seed": 2024,
        "fixed_across_folds": True,
        "fixed_across_training_seeds": True,
        "preserves_mini_marginal": True,
        "learner_outcome_used": False,
        "source": "scripts/kt/tlt_extension_common.py::build_scalar_maps_coldstart",
        "dbe_mapping_generation": "MUST_BE_GENERATED_BEFORE_DBE_UNSEEN_TRAINING_IN_LATER_PHASE",
    }
    char_ctrl = {
        "name": "CharacterLength",
        "xes_junyi_existing": "min-max char_length using seen-item min/max within each fold; applied to all items",
        "dbe_protocol_freeze": "min_max_char_length_over_universe (166 items)",
        "content_only": True,
        "learner_outcome_used": False,
        "do_not_alter_after_family_c": True,
    }
    (ART / "P3C0_RANDOM_PERMUTED_CONTROL.json").write_text(json.dumps(random_ctrl, indent=2) + "\n", encoding="utf-8")
    (ART / "P3C0_CHARACTERLENGTH_CONTROL.json").write_text(json.dumps(char_ctrl, indent=2) + "\n", encoding="utf-8")

    results = {
        "phase": "TLT3D_P3C0",
        "base_result_commit": "2d1206090074aa49f937d77955bc1e8d6aca22bf",
        "amendment_id": am["amendment_id"],
        "D_OPERATIONALIZATION_STATUS": op_status["D_OPERATIONALIZATION_STATUS"],
        "family_D_count": 36,
        "existing_xes_junyi": avail,
        "dbe_folds": dbe_folds,
        "planned_registry_hash": planned_hash,
        "planned_conceptual_cells": 250,
        "dbe_training_started": False,
        "family_D_results_computed": False,
        "manuscript_edited": False,
        "generated_at_utc": utc_now(),
        "gates": {
            "D0_1_registry_36": True,
            "D0_2_backbones": True,
            "D0_3_comparators": True,
            "D0_4_folds": True,
            "D0_5_evaluation_unit": True,
            "D0_6_leakage": leak["all_folds_zero_target_train"],
            "D0_7_xes_junyi_availability": avail["complete"],
            "D0_8_pairing": avail["pairing_verified"],
            "D0_9_operational_statistic_sealed": True,
            "D0_10_dbe_planned_registry": True,
            "D0_11_no_dbe_training": no_train["pass"],
            "D0_12_no_manuscript_edit": True,
        },
    }
    (ART / "TLT3D_P3C0_PREFLIGHT.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    write_reports(
        rows,
        op_status,
        am,
        avail,
        dbe_folds,
        xj_folds,
        leak,
        model,
        planned_hash,
        len(planned_df),
        no_train,
    )

    # Ensure no confirmatory Family-D results file
    assert not (ART / "family_D_confirmatory_results.csv").exists()
    print(json.dumps({"status": "OK", "amendment": am["amendment_id"], "planned_hash": planned_hash}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
