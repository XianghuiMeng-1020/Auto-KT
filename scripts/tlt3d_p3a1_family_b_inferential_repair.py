#!/usr/bin/env python3
"""TLT-3D Phase 3A.1 — Family-B post-result inferential repair.

AMENDMENT_ID = POST_RESULT_INFERENTIAL_REPAIR_001
TIMING = AFTER_FAMILY_B_EFFECT_ESTIMATES_OBSERVED

Reproduces frozen Phase-3A OOF paired predictions, verifies ΔR², then applies
two-sided paired prediction-label randomization (B=100000). Does not retrain.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from tlt3d_p3a_measurement_validity import (  # noqa: E402
    SHARED_CORE,
    build_analysis_frame,
    holm,
    load_dbe_scores,
    load_primary_fo_errors,
    load_shared_core_features,
    load_xes_junyi_scores,
    oof_metrics,
)

ART = ROOT / "artifacts" / "tlt3d"
REP = ROOT / "reports"
CFG = ROOT / "configs" / "tlt3d"

AMENDMENT_ID = "POST_RESULT_INFERENTIAL_REPAIR_001"
N_PERM = 100_000
MASTER_SEED = 2024
DELTA_TOL = 1e-12

# Snapshot hashes of immutable artifacts before repair writes
IMMUTABLE_HASHES = {
    "family_A_confirmatory_results.csv": "49078d1e7044608c666cb689551ffec0265ecb957070a37d78b7d0c0f265ebf4",
    "P3A_JUNYI_LEGACY_SENSITIVITY.csv": "03955d29fb57a2c4cebce05f6ed213ab312ee9f69ed9f260b03a59610e1e7295",
    "P3A_XES_LEGACY_SENSITIVITY.csv": "01b4642f22bfffbd145c4d667c71d36eed21f56fcbb8fa48b84112f36d8c0d3d",
    "P3A_DBE_CONSENSUS_SENSITIVITY.csv": "e9161288521ad1a3aba8de29f90480d983f0a7351913532f5a0b39100b2142f4",
    "P3A_DBE_EXPERT_SECONDARY.csv": "4c12ad60939fab0e515a3c793b848c38a6dadbf56d36963523e27830aec494c1",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rng_for_hypothesis(hypothesis_id: str) -> np.random.Generator:
    """Deterministic independent stream per hypothesis.

    Derivation:
      digest = SHA256(f"{AMENDMENT_ID}|{hypothesis_id}|{MASTER_SEED}")
      seed_int = int.from_bytes(digest[:8], "big")
      return np.random.default_rng(seed_int)
    """
    payload = f"{AMENDMENT_ID}|{hypothesis_id}|{MASTER_SEED}".encode()
    digest = hashlib.sha256(payload).digest()
    seed_int = int.from_bytes(digest[:8], "big")
    return np.random.default_rng(seed_int)


def r2_from_preds(y: np.ndarray, pred: np.ndarray) -> float:
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def paired_prediction_label_randomization(
    y: np.ndarray,
    base_pred: np.ndarray,
    aug_pred: np.ndarray,
    *,
    hypothesis_id: str,
    n_perm: int = N_PERM,
) -> dict:
    """Two-sided paired prediction-label randomization; no model refit.

    For each item i independently, with probability 1/2 swap base/aug labels.
    T = R2(y, aug') - R2(y, base').
    p = (1 + #{|T_b| >= |T_obs|}) / (B + 1)
    """
    y = np.asarray(y, dtype=float)
    base_pred = np.asarray(base_pred, dtype=float)
    aug_pred = np.asarray(aug_pred, dtype=float)
    assert y.shape == base_pred.shape == aug_pred.shape

    t_obs = r2_from_preds(y, aug_pred) - r2_from_preds(y, base_pred)

    # Per-item contribution to (SS_base - SS_aug); swaps flip the sign.
    d = (y - base_pred) ** 2 - (y - aug_pred) ** 2
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    if ss_tot <= 0:
        raise RuntimeError("ss_tot <= 0")
    # Consistency: T_obs == d.sum()/ss_tot
    t_from_d = float(d.sum() / ss_tot)
    if abs(t_from_d - t_obs) > 1e-12:
        raise RuntimeError(f"T inconsistency: {t_obs} vs {t_from_d}")

    rng = rng_for_hypothesis(hypothesis_id)
    n = len(y)
    count = 0
    abs_obs = abs(t_obs)
    chunk = 5000
    for start in range(0, n_perm, chunk):
        m = min(chunk, n_perm - start)
        # Bernoulli(0.5) via integers {0,1}; 1 => swap
        s = rng.integers(0, 2, size=(m, n), dtype=np.int8)
        signs = 1 - 2 * s  # +1 keep, -1 swap
        tb = (signs.astype(np.float64) @ d) / ss_tot
        count += int(np.sum(np.abs(tb) >= abs_obs))

    p_raw = (1 + count) / (n_perm + 1)
    mc_se = float(np.sqrt(p_raw * (1.0 - p_raw) / n_perm))
    return {
        "t_obs": float(t_obs),
        "p_raw_repaired": float(p_raw),
        "p_raw_mc_se": mc_se,
        "n_extreme": int(count),
        "n_perm": int(n_perm),
        "rng_payload": f"{AMENDMENT_ID}|{hypothesis_id}|{MASTER_SEED}",
        "rng_seed_int": int.from_bytes(
            hashlib.sha256(f"{AMENDMENT_ID}|{hypothesis_id}|{MASTER_SEED}".encode()).digest()[:8],
            "big",
        ),
    }


def reconstruct_oof_pairs(primary_df: pd.DataFrame, fam_b: pd.DataFrame) -> pd.DataFrame:
    """Rebuild base/aug OOF predictions with frozen Phase-3A code; gate ΔR²."""
    rows = []
    oof_store = []
    for _, h in fam_b.iterrows():
        hid = h["hypothesis_id"]
        sub = primary_df[(primary_df.dataset == h["dataset"]) & (primary_df.model == h["model"])].reset_index(drop=True)
        y = sub["learner_error"].values.astype(float)
        base = oof_metrics(sub, SHARED_CORE, y)
        aug = oof_metrics(sub, SHARED_CORE + ["score"], y)
        delta = float(aug["oof_r2"] - base["oof_r2"])
        expected = float(h["delta_r2"])
        if abs(delta - expected) > DELTA_TOL:
            raise RuntimeError(
                f"FAMILY_B_OOF_REPRODUCTION_BLOCKER: {hid} "
                f"recomputed={delta!r} phase3a={expected!r} diff={delta - expected!r}"
            )
        # also gate base/aug R2
        if abs(base["oof_r2"] - float(h["base_r2"])) > DELTA_TOL:
            raise RuntimeError(f"FAMILY_B_OOF_REPRODUCTION_BLOCKER base_r2 {hid}")
        if abs(aug["oof_r2"] - float(h["aug_r2"])) > DELTA_TOL:
            raise RuntimeError(f"FAMILY_B_OOF_REPRODUCTION_BLOCKER aug_r2 {hid}")

        for i in range(len(sub)):
            oof_store.append(
                {
                    "hypothesis_id": hid,
                    "dataset": h["dataset"],
                    "model": h["model"],
                    "item_id": str(sub.iloc[i]["item_id"]),
                    "y": float(y[i]),
                    "base_oof_prediction": float(base["preds"][i]),
                    "augmented_oof_prediction": float(aug["preds"][i]),
                }
            )
        rows.append(
            {
                "hypothesis_id": hid,
                "delta_r2_recomputed": delta,
                "delta_r2_phase3a": expected,
                "abs_diff": abs(delta - expected),
                "n_items": len(sub),
            }
        )
    oof_df = pd.DataFrame(oof_store)
    oof_df.to_csv(ART / "P3A1_family_B_oof_prediction_pairs.csv", index=False)
    return oof_df, pd.DataFrame(rows)


def main() -> int:
    ART.mkdir(parents=True, exist_ok=True)
    REP.mkdir(parents=True, exist_ok=True)
    CFG.mkdir(parents=True, exist_ok=True)

    # Immutability pre-check
    for name, expected in IMMUTABLE_HASHES.items():
        got = sha256_file(ART / name)
        if got != expected:
            raise RuntimeError(f"immutable artifact drift before repair: {name}")

    fam_b = pd.read_csv(ART / "family_B_confirmatory_results.csv")
    fam_a = pd.read_csv(ART / "family_A_confirmatory_results.csv")
    assert len(fam_b) == 6 and len(fam_a) == 6

    # Snapshot original effect columns
    effect_cols = [
        "delta_r2",
        "ci_lo",
        "ci_hi",
        "base_r2",
        "aug_r2",
        "delta_rmse",
        "delta_mae",
        "base_rmse",
        "aug_rmse",
        "base_mae",
        "aug_mae",
    ]
    original_effects = fam_b[["hypothesis_id"] + effect_cols].copy()

    print("Reconstructing analysis frame + OOF pairs...")
    llm, _ = load_xes_junyi_scores()
    fo = load_primary_fo_errors()
    features = load_shared_core_features()
    dbe_scores = load_dbe_scores()
    primary = build_analysis_frame(fo, llm, dbe_scores, features, eligible_only=True)

    try:
        oof_df, gate = reconstruct_oof_pairs(primary, fam_b)
    except RuntimeError as e:
        if "FAMILY_B_OOF_REPRODUCTION_BLOCKER" in str(e):
            print(str(e), file=sys.stderr)
            return 2
        raise
    gate.to_csv(ART / "P3A1_OOF_REPRODUCTION_GATE.csv", index=False)
    print(gate.to_string(index=False))

    print(f"Running paired prediction-label randomization (B={N_PERM})...")
    repair_rows = []
    for _, h in fam_b.iterrows():
        hid = h["hypothesis_id"]
        sub = oof_df[oof_df.hypothesis_id == hid]
        y = sub["y"].values
        base_p = sub["base_oof_prediction"].values
        aug_p = sub["augmented_oof_prediction"].values
        # Gate T_obs == stored delta_r2
        t_obs = r2_from_preds(y, aug_p) - r2_from_preds(y, base_p)
        if abs(t_obs - float(h["delta_r2"])) > DELTA_TOL:
            raise RuntimeError(f"FAMILY_B_OOF_REPRODUCTION_BLOCKER T_obs {hid}")
        res = paired_prediction_label_randomization(y, base_p, aug_p, hypothesis_id=hid)
        print(
            f"  {hid}: ΔR²={res['t_obs']:.6g} p_raw={res['p_raw_repaired']:.6g} "
            f"MC_SE={res['p_raw_mc_se']:.3g} n_extreme={res['n_extreme']}"
        )
        repair_rows.append({"hypothesis_id": hid, **res})

    repair = pd.DataFrame(repair_rows)
    repair["p_holm_repaired"] = holm(repair["p_raw_repaired"].tolist())
    repair.to_csv(ART / "P3A1_FAMILY_B_REPAIRED_PVALUES.csv", index=False)

    # Update family_B CSV — preserve effect estimates exactly
    out = fam_b.copy()
    # Clear obsolete blocker fields; keep chronology in note
    out["inferential_test"] = "PAIRED_PREDICTION_LABEL_RANDOMIZATION_TEST"
    out["confirmatory_p_available"] = True
    out["blocker"] = ""
    out["note"] = (
        "EFFECT_ESTIMATE_STATUS=FROZEN_PRE_RESULT; "
        "INFERENTIAL_PVALUE_STATUS=POST_RESULT_REPAIR; "
        f"amendment={AMENDMENT_ID}"
    )
    # Map repaired p onto rows by hypothesis_id
    rmap = repair.set_index("hypothesis_id")
    out["p_raw_repaired"] = out["hypothesis_id"].map(rmap["p_raw_repaired"])
    out["p_raw_mc_se"] = out["hypothesis_id"].map(rmap["p_raw_mc_se"])
    out["p_holm_repaired"] = out["hypothesis_id"].map(rmap["p_holm_repaired"])
    out["inferential_status"] = "POST_RESULT_TRANSPARENT_REPAIR"
    # Also populate raw_p / holm_p as the repaired values for gate convenience,
    # while preserving classification via inferential_status
    out["raw_p"] = out["p_raw_repaired"]
    out["holm_p"] = out["p_holm_repaired"]

    # Effect immutability check
    for c in effect_cols:
        if not np.allclose(out[c].astype(float), original_effects[c].astype(float), rtol=0, atol=0, equal_nan=True):
            # exact string/float compare
            if not (out[c].astype(float).values == original_effects[c].astype(float).values).all():
                raise RuntimeError(f"R6 FAIL: effect column changed: {c}")

    # Restore exact pre-repair effect floats (avoid pandas CSV float truncation)
    for _, er in original_effects.iterrows():
        m = out["hypothesis_id"] == er["hypothesis_id"]
        for c in effect_cols:
            out.loc[m, c] = er[c]
    out.to_csv(ART / "family_B_confirmatory_results.csv", index=False, float_format="%.17g")

    # Amendment JSON
    amendment = {
        "amendment_id": AMENDMENT_ID,
        "timing": "after_family_B_effect_estimates_observed",
        "reason": "missing_family_B_pvalue_definition",
        "scientific_inputs_changed": False,
        "effect_estimates_changed": False,
        "test": "paired_prediction_label_randomization",
        "test_full_name": "PAIRED_PREDICTION_LABEL_RANDOMIZATION_TEST",
        "sidedness": "two_sided",
        "permutations": N_PERM,
        "master_seed": MASTER_SEED,
        "rng_derivation": (
            'digest=SHA256(f"{amendment_id}|{hypothesis_id}|{master_seed}"); '
            "seed_int=int.from_bytes(digest[:8],'big'); "
            "np.random.default_rng(seed_int)"
        ),
        "statistic": "delta_oof_r2",
        "family_size": 6,
        "multiplicity": "holm_within_family_B",
        "status": "POST_RESULT_TRANSPARENT_REPAIR",
        "p_formula": "p_raw = (1 + #{|T_b| >= |T_obs|}) / (B + 1)",
        "classification": {
            "delta_r2": "CONFIRMATORY_EFFECT_ESTIMATE",
            "bootstrap_ci": "CONFIRMATORY_PRE_SPECIFIED_UNCERTAINTY",
            "repaired_p": "POST_RESULT_INFERENTIAL_COMPLETION",
            "holm_repaired_p": "POST_RESULT_INFERENTIAL_COMPLETION",
        },
        "generated_at_utc": utc_now(),
    }
    (CFG / "TLT3D_PROTOCOL_AMENDMENT_POSTRESULT_001.json").write_text(
        json.dumps(amendment, indent=2) + "\n", encoding="utf-8"
    )

    # Update confirmatory JSON
    conf_path = ART / "TLT3D_P3A_CONFIRMATORY_RESULTS.json"
    conf = json.loads(conf_path.read_text())
    conf["family_B_test_blocker"] = False
    conf["FAMILY_B_TEST_SPECIFICATION_BLOCKER"] = False
    conf["family_B_inferential_repair"] = {
        "amendment_id": AMENDMENT_ID,
        "timing": "AFTER_FAMILY_B_EFFECT_ESTIMATES_OBSERVED",
        "status": "POST_RESULT_TRANSPARENT_REPAIR",
        "test": "PAIRED_PREDICTION_LABEL_RANDOMIZATION_TEST",
        "sidedness": "two_sided",
        "permutations": N_PERM,
        "master_seed": MASTER_SEED,
        "effect_estimates_changed": False,
        "bootstrap_cis_changed": False,
        "results": repair.to_dict(orient="records"),
    }
    # Refresh family_B records from CSV (effects unchanged; p filled)
    conf["family_B"] = out.to_dict(orient="records")
    # Explicit chronology note
    conf["chronology_note"] = (
        "Family-B ΔR² and bootstrap CIs were produced in Phase 3A under a sealed "
        "registry reference that defined no confirmatory p-value. After those "
        "estimates were observed, POST_RESULT_INFERENTIAL_REPAIR_001 specified "
        "a uniform two-sided paired prediction-label randomization test."
    )
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    conf["analysis_commit_pre_repair"] = commit
    conf["p3a1_generated_at_utc"] = utc_now()
    conf_path.write_text(json.dumps(conf, indent=2, default=str) + "\n", encoding="utf-8")

    # Update synthesis table Holm column
    syn = pd.read_csv(ART / "P3A_THREE_DATASET_SYNTHESIS.csv")
    bmap = out.set_index(["dataset", "model"])
    syn["family_B_holm_p"] = [
        bmap.loc[(r.dataset, r.model), "p_holm_repaired"] for r in syn.itertuples()
    ]
    syn.to_csv(ART / "P3A_THREE_DATASET_SYNTHESIS.csv", index=False)

    write_repair_report(out, repair)
    update_measurement_report(out)
    update_p3a_tests_for_repair()

    # Post immutability of Family A / sensitivities
    for name, expected in IMMUTABLE_HASHES.items():
        got = sha256_file(ART / name)
        if got != expected:
            raise RuntimeError(f"R7/R8 FAIL: immutable artifact changed: {name}")

    print(json.dumps({"ok": True, "amendment_id": AMENDMENT_ID, "n_perm": N_PERM}, indent=2))
    return 0


def write_repair_report(fam_b: pd.DataFrame, repair: pd.DataFrame) -> None:
    lines = [
        "# TLT-3D P3A.1 — Family-B Inferential Repair",
        "",
        f"**Amendment ID:** `{AMENDMENT_ID}`  ",
        "**Timing:** `AFTER_FAMILY_B_EFFECT_ESTIMATES_OBSERVED`  ",
        "**Status:** `POST_RESULT_TRANSPARENT_REPAIR`",
        "",
        "## 1. Why repair was necessary",
        "",
        "The sealed confirmatory registry (`configs/tlt3d/confirmatory_family_registry.json`) ",
        "set Family-B `test` to `as_in_repair_authentic_orientation_v2_incremental_block`. ",
        "That authoritative incremental block produces OOF ΔR² / RMSE / MAE under frozen ",
        "Ridge(alpha=1.0)+StandardScaler+KFold(5,shuffle=True,random_state=2024), but ",
        "**defines no confirmatory p-value**. Phase 3A correctly returned ",
        "`FAMILY_B_TEST_SPECIFICATION_BLOCKER` rather than inventing a substitute test.",
        "",
        "## 2. Chronology",
        "",
        "1. Phase 3A computed and reported all six Family-B effect estimates and bootstrap CIs.",
        "2. No Family-B confirmatory p-values existed (blocker).",
        "3. **After** those estimates/intervals were observed, the PI selected a single fixed ",
        "   two-sided paired prediction-label randomization procedure.",
        "4. This amendment records that post-result timing permanently; repaired p-values are ",
        "   **not** labeled as preregistered.",
        "",
        "## 3. Test definition",
        "",
        "For each hypothesis, item-level triples `(y_i, base_oof_i, aug_oof_i)` are fixed.",
        "",
        "- Observed statistic: `T_obs = R²(y, aug) − R²(y, base)` (= Phase-3A ΔR²).",
        "- Null: for each item independently, swap base/aug labels with probability 1/2.",
        "- For permutation `b`: `T_b = R²(y, aug') − R²(y, base')` after swaps.",
        "- Two-sided Monte-Carlo p: `p_raw = (1 + #{|T_b| ≥ |T_obs|}) / (B + 1)`, `B = 100000`.",
        "- RNG: `SHA256(f\"POST_RESULT_INFERENTIAL_REPAIR_001|{hypothesis_id}|2024\")` → first 8 bytes → `np.random.default_rng(seed_int)`.",
        "",
        "## 4. Why no retraining occurs",
        "",
        "Inference compares already-frozen paired OOF predictions. Permutations only swap ",
        "prediction labels within item; `y` is never permuted; Ridge/StandardScaler are never fitted.",
        "",
        "## 5. Multiplicity",
        "",
        "Holm correction across exactly the six Family-B hypotheses. Family A / sensitivities / ",
        "expert-difficulty p-values are excluded.",
        "",
        "## 6. Result table",
        "",
        "| ID | Delta R2 | bootstrap CI | repaired raw p | MC SE | repaired Holm p |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for _, r in fam_b.iterrows():
        lines.append(
            f"| {r.hypothesis_id} | {r.delta_r2:.6g} | [{r.ci_lo:.6g}, {r.ci_hi:.6g}] | "
            f"{r.p_raw_repaired:.6g} | {r.p_raw_mc_se:.3g} | {r.p_holm_repaired:.6g} |"
        )
    lines += [
        "",
        "## 7. Integrity note",
        "",
        "NO EFFECT ESTIMATE, MODEL, FEATURE, FOLD, OR HYPOTHESIS MEMBERSHIP ",
        "WAS CHANGED DURING THE REPAIR.",
        "",
        "### Classification",
        "",
        "- ΔR²: `CONFIRMATORY_EFFECT_ESTIMATE`",
        "- bootstrap CI: `CONFIRMATORY_PRE_SPECIFIED_UNCERTAINTY`",
        "- repaired p / Holm p: `POST_RESULT_INFERENTIAL_COMPLETION`",
        "",
    ]
    (REP / "TLT3D_P3A1_FAMILY_B_INFERENTIAL_REPAIR.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def update_measurement_report(fam_b: pd.DataFrame) -> None:
    path = REP / "TLT3D_P3A_MEASUREMENT_VALIDITY_RESULTS.md"
    text = path.read_text(encoding="utf-8")
    note = (
        "\n> **Family-B inferential note (P3A.1):** Family-B effect definitions, OOF models, "
        "folds, and bootstrap intervals were fixed before DBE outcomes were analyzed. The sealed "
        "protocol, however, did not operationally define a p-value procedure for this family. "
        "After the six effect estimates and intervals were observed, a uniform two-sided paired "
        "prediction-label randomization procedure was specified to complete the inferential "
        "analysis. These repaired p-values are therefore reported transparently as post-result "
        "inferential completion rather than as pre-specified inferential tests.\n"
    )
    marker = "## 7. Family B — Incremental Information"
    if marker in text and "Family-B inferential note (P3A.1)" not in text:
        text = text.replace(marker, marker + note)

    # Replace Family B table section
    start = text.find("| ID | Dataset | Model | N | Base R² | Aug R² | ΔR² |")
    if start >= 0:
        # find end of table (next ## or end)
        end = text.find("\n## ", start)
        if end < 0:
            end = len(text)
        # rebuild table including header line and separator
        # walk back to header
        hdr = text.rfind("| ID | Dataset | Model | N | Base R²", 0, start + 1)
        if hdr >= 0:
            start = hdr
        rows = [
            "| ID | Dataset | Model | N | Base R² | Aug R² | ΔR² | 95% CI | raw p (repaired) | Holm p (repaired) | ΔRMSE | ΔMAE |",
            "|---|---|---|---:|---:|---:|---:|---|---|---|---:|---:|",
        ]
        for _, r in fam_b.iterrows():
            rows.append(
                f"| {r.hypothesis_id} | {r.dataset} | {r.model} | {int(r.n_items)} | "
                f"{r.base_r2:.4f} | {r.aug_r2:.4f} | {r.delta_r2:.4f} | "
                f"[{r.ci_lo:.4f}, {r.ci_hi:.4f}] | {r.p_raw_repaired:.4g} | {r.p_holm_repaired:.4g} | "
                f"{r.delta_rmse:.4f} | {r.delta_mae:.4f} |"
            )
        rows.append("")
        rows.append(
            "Statuses: ΔR²=`CONFIRMATORY_EFFECT_ESTIMATE`; CI=`CONFIRMATORY_PRE_SPECIFIED_UNCERTAINTY`; "
            "p/Holm=`POST_RESULT_INFERENTIAL_COMPLETION` "
            f"(`{AMENDMENT_ID}`)."
        )
        # remove old blocker paragraph if present immediately before table
        pre = text[:start]
        # strip prior blocker line block if still present
        blocker = "**FAMILY_B_TEST_SPECIFICATION_BLOCKER:**"
        if blocker in pre:
            bi = pre.rfind(blocker)
            # remove from blocker to start
            pre = pre[:bi].rstrip() + "\n\n"
        text = pre + "\n".join(rows) + "\n" + text[end:].lstrip("\n")

    # Update findings bullet
    text = text.replace(
        "- Family B: six ΔR² estimates computed under frozen Ridge/OOF; confirmatory p/Holm blocked pending PI test specification.",
        "- Family B: six ΔR² estimates frozen; confirmatory p/Holm completed via "
        f"`{AMENDMENT_ID}` (post-result transparent repair; two-sided paired prediction-label randomization, B=100000).",
    )
    text = text.replace(
        "PROTOCOL_DEVIATIONS = NONE (except Family B confirmatory p-value specification absent in sealed implementation reference)",
        "PROTOCOL_DEVIATIONS = POST_RESULT_INFERENTIAL_REPAIR_001 "
        "(Family-B p-value procedure specified after effect estimates observed; effects unchanged)",
    )
    path.write_text(text, encoding="utf-8")


def update_p3a_tests_for_repair() -> None:
    """No-op placeholder; dedicated p3a1 tests cover repair. Keep p3a tests compatible."""
    # Patch existing p3a test that expected blocker True
    p = ROOT / "tests" / "test_tlt3d_p3a_measurement_validity.py"
    text = p.read_text(encoding="utf-8")
    old = '''def test_family_b_test_blocker_no_confirmatory_p(fam_b):
    conf = json.loads((ART / "TLT3D_P3A_CONFIRMATORY_RESULTS.json").read_text())
    assert conf["FAMILY_B_TEST_SPECIFICATION_BLOCKER"] is True
    assert fam_b["raw_p"].isna().all()
    assert fam_b["holm_p"].isna().all()
    assert fam_b["confirmatory_p_available"].eq(False).all()
'''
    new = '''def test_family_b_test_blocker_no_confirmatory_p(fam_b):
    """Superseded by P3A.1 repair: confirmatory p-values now post-result repaired."""
    conf = json.loads((ART / "TLT3D_P3A_CONFIRMATORY_RESULTS.json").read_text())
    assert conf.get("FAMILY_B_TEST_SPECIFICATION_BLOCKER") is False
    assert fam_b["p_raw_repaired"].notna().all()
    assert fam_b["p_holm_repaired"].notna().all()
    assert fam_b["inferential_status"].eq("POST_RESULT_TRANSPARENT_REPAIR").all()
'''
    if old in text:
        p.write_text(text.replace(old, new), encoding="utf-8")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        raise SystemExit(2)
