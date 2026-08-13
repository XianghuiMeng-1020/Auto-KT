#!/usr/bin/env python3
"""TLT-3D Phase 3A — RQ1 + confirmatory Families A/B + frozen sensitivities.

NO RQ4 / Family C/D / manuscript edits.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

ART = ROOT / "artifacts" / "tlt3d"
REP = ROOT / "reports"
CFG = ROOT / "configs" / "tlt3d"
TABLE = ROOT / "tables"
LLM_PATH = ROOT / "journal_expansion/features/full_llm/parsed/all_llm_item_features.parquet"
DBE_SCORES = ART / "dbe_llm_scores_confirmatory.csv"
DBE_ITEMS = ROOT / "data/external/dbe_kt22/derived/tlt3d_canonical_items.parquet"
DBE_SURFACE = ROOT / "data/external/dbe_kt22/derived/tlt3d_surface_features.csv"
DBE_TX = ROOT / "data/external/dbe_kt22/raw/official_extracted/csv/Transaction.csv"
DBE_CHOICES = ROOT / "data/external/dbe_kt22/raw/official_extracted/csv/Question_Choices.csv"
DBE_SPLITS = ROOT / "data/external/dbe_kt22/derived/tlt3d_learner_splits.csv"
LEGACY_REF = TABLE / "AUTHENTIC_DIFFICULTY_REFERENCES_V2_ORIENTATION_CORRECTED.csv"

SHARED_CORE = [
    "char_length",
    "token_length",
    "sentence_count",
    "number_count",
    "math_symbol_count",
    "equation_count",
    "answer_option_count",
]
MODELS = ["gpt-4o-mini", "gpt-5.4"]
BOOT_N = 500
BOOT_SEED = 2024
CV_FOLDS = 5
CV_SEED = 2024
RIDGE_ALPHA = 1.0
PROTOCOL_COMMIT = "a459e34a24240c03ba4dbe4b1d0185e42eaf4377"
SCORING_COMMIT = "e726237f003e105d036a7b8e439385a152300e42"

EXPECTED_HASHES = {
    "dbe_score_csv": "484dd79a140372fcbda66275aa472505de47f0852cd37992692ac87c97e22f64",
    "dbe_raw_mini": "8929921d015cf3e253f40ad9e10ea0360aa84b4ac2620d5e39f22c2823c4ec5a",
    "dbe_raw_54": "a32d7cba6d860c29a2afe0e39bc188e033b1049fad38fca758701f5150ef11ec",
    "item_universe": "d62fb95604bac94e65adda498dc21175e66ce75198836996f5f53695c96e38a6",
    "prompt": "d99a9645219033e713bb78fd31dc3d74826bf31adf1a0e5977f30fcbda911c35",
    "split": "65d13de13e7c3a8b366c63628cab3e7ef3f2a67b8eae7cde1aab4e1a53d627dc",
}
EXPECTED_N = {"xes3g5m": 3265, "junyi": 169, "dbe_kt22": 166}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def holm(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = np.argsort(pvals)
    out = [np.nan] * m
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * float(pvals[idx])
        running = max(running, val)
        out[idx] = min(1.0, running)
    return out


def bootstrap_ci(x: np.ndarray, y: np.ndarray, fn: Callable, n_boot=BOOT_N, seed=BOOT_SEED):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    point = float(fn(x, y))
    vals = []
    n = len(x)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            v = float(fn(x[idx], y[idx]))
            if np.isfinite(v):
                vals.append(v)
        except Exception:
            pass
    lo, hi = (np.quantile(vals, [0.025, 0.975]) if vals else (np.nan, np.nan))
    return point, float(lo), float(hi)


def assert_input_integrity() -> dict[str, Any]:
    checks = {
        "dbe_score_csv": ART / "dbe_llm_scores_confirmatory.csv",
        "dbe_raw_mini": ART / "dbe_llm_raw_gpt4omini.jsonl",
        "dbe_raw_54": ART / "dbe_llm_raw_gpt54.jsonl",
    }
    for k, p in checks.items():
        got = sha256_file(p)
        if got != EXPECTED_HASHES[k]:
            raise RuntimeError(f"PHASE3A_BLOCKED_INPUT_DRIFT: {k} {got}")
    universe = json.loads((CFG / "dbe_item_universe.json").read_text())
    freeze = json.loads((ART / "DBE_PRE_LLM_FREEZE.json").read_text())
    p11 = json.loads((ART / "P11_AMENDMENT_COMPUTE_SUMMARY.json").read_text())
    if universe["item_universe_hash"] != EXPECTED_HASHES["item_universe"]:
        raise RuntimeError("PHASE3A_BLOCKED_INPUT_DRIFT: item_universe")
    if freeze["learner_split_hash"] != EXPECTED_HASHES["split"]:
        raise RuntimeError("PHASE3A_BLOCKED_INPUT_DRIFT: split")
    if p11["prompt_twin"]["dbe_prompt_hash"] != EXPECTED_HASHES["prompt"]:
        raise RuntimeError("PHASE3A_BLOCKED_INPUT_DRIFT: prompt")
    return {"universe": universe, "freeze": freeze}


def sentence_count(text: str) -> int:
    parts = re.split(r"[.!?]+", str(text))
    return max(1, len([p for p in parts if p.strip()]))


def load_xes_junyi_scores() -> tuple[pd.DataFrame, dict]:
    llm = pd.read_parquet(LLM_PATH)
    llm = llm[llm["parse_status"] == "valid"].copy()
    manifest = {"generated_at_utc": utc_now(), "source": str(LLM_PATH.relative_to(ROOT)), "models": {}}
    for ds in ["xes3g5m", "junyi"]:
        for model in MODELS:
            sub = llm[(llm["dataset"] == ds) & (llm["model_identifier"] == model)]
            # hash of score vector
            payload = "|".join(
                f"{r.item_id_hash}:{r.scalar_difficulty}"
                for r in sub.sort_values("item_id_hash").itertuples()
            )
            manifest["models"][f"{ds}|{model}"] = {
                "path": str(LLM_PATH.relative_to(ROOT)),
                "row_count": int(len(sub)),
                "item_count": int(sub["item_id_hash"].nunique()),
                "model_identifier": model,
                "score_scale": "[0,1] larger=harder",
                "sha256_score_vector": hashlib.sha256(payload.encode()).hexdigest(),
            }
    # expected scored counts
    assert manifest["models"]["xes3g5m|gpt-4o-mini"]["item_count"] == 5363
    assert manifest["models"]["junyi|gpt-4o-mini"]["item_count"] == 190
    (ART / "XES_JUNYI_SCORE_INPUT_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return llm, manifest


def load_primary_fo_errors() -> pd.DataFrame:
    rows = []
    for ds, fname, idcol in [
        ("xes3g5m", "xes_first_observed_rq2_eligibility.csv", "item_id_hash"),
        ("junyi", "junyi_first_observed_rq2_eligibility.csv", "item_id_hash"),
        ("dbe_kt22", "dbe_first_observed_rq2_eligibility.csv", "item_id"),
    ]:
        df = pd.read_csv(ART / fname)
        for _, r in df.iterrows():
            rows.append(
                {
                    "dataset": ds,
                    "item_id": str(r[idcol]),
                    "heldout_unique_learners": int(r["n_learners"]),
                    "heldout_first_observed_responses": int(r["n_first_observed"]),
                    "learner_error_raw": float(r["raw_error"]),
                    "learner_error": float(r["smoothed_error_beta_1_1"]),
                    "eligibility_threshold": 20,
                    "eligible_primary": bool(r["eligible_ge20"]),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(ART / "primary_first_observed_item_error.csv", index=False)
    for ds, n in EXPECTED_N.items():
        got = int(out[(out.dataset == ds) & (out.eligible_primary)].shape[0])
        if got != n:
            raise RuntimeError(f"primary N mismatch {ds}: {got} != {n}")
    return out


def load_shared_core_features() -> pd.DataFrame:
    # XES/Junyi from surface table
    surf = pd.read_csv(TABLE / "AUTHENTIC_ITEM_SURFACE_FEATURES.csv")
    rows = []
    for ds in ["xes3g5m", "junyi"]:
        sub = surf[surf["dataset"] == ds].copy()
        for _, r in sub.iterrows():
            rows.append(
                {
                    "dataset": ds,
                    "item_id": str(r["item_id_hash"]),
                    "char_length": float(r["char_length"]),
                    "token_length": float(r["token_length"]),
                    "sentence_count": float(r["sentence_count"]),
                    "number_count": float(r["number_count"]),
                    "math_symbol_count": float(r["math_symbol_count"]),
                    "equation_count": float(r["equation_count"]),
                    "answer_option_count": float(r["answer_option_count"]),
                }
            )
    # DBE
    items = pd.read_parquet(DBE_ITEMS, columns=["item_id", "scoring_text", "char_length", "token_length", "n_choices"])
    surf_d = pd.read_csv(DBE_SURFACE)
    m = items.merge(surf_d, on="item_id", suffixes=("", "_s"))
    for _, r in m.iterrows():
        rows.append(
            {
                "dataset": "dbe_kt22",
                "item_id": str(int(r["item_id"])),
                "char_length": float(r["char_length"]),
                "token_length": float(r["token_length"]),
                "sentence_count": float(sentence_count(r["scoring_text"])),
                "number_count": float(r["number_count"]),
                "math_symbol_count": float(r["math_symbol_count"]),
                "equation_count": float(r["equation_count"]),
                "answer_option_count": float(r["n_choices"]),
            }
        )
    return pd.DataFrame(rows)


def load_dbe_scores() -> pd.DataFrame:
    df = pd.read_csv(DBE_SCORES)
    long = []
    for _, r in df.iterrows():
        long.append({"dataset": "dbe_kt22", "item_id": str(int(r["item_id"])), "model": "gpt-4o-mini", "score": float(r["gpt4omini_score"])})
        long.append({"dataset": "dbe_kt22", "item_id": str(int(r["item_id"])), "model": "gpt-5.4", "score": float(r["gpt54_score"])})
    return pd.DataFrame(long)


def build_analysis_frame(fo: pd.DataFrame, llm: pd.DataFrame, dbe_scores: pd.DataFrame, features: pd.DataFrame, *, eligible_only=True) -> pd.DataFrame:
    primary = fo[fo["eligible_primary"]].copy() if eligible_only else fo.copy()
    # XES/Junyi scores
    xj = llm[llm["dataset"].isin(["xes3g5m", "junyi"])][
        ["dataset", "item_id_hash", "model_identifier", "scalar_difficulty"]
    ].rename(columns={"item_id_hash": "item_id", "model_identifier": "model", "scalar_difficulty": "score"})
    xj["item_id"] = xj["item_id"].astype(str)
    all_scores = pd.concat([xj, dbe_scores], ignore_index=True)
    df = primary.merge(all_scores, on=["dataset", "item_id"], how="inner")
    df = df.merge(features, on=["dataset", "item_id"], how="left")
    # integrity
    for ds, n in EXPECTED_N.items():
        for model in MODELS:
            sub = df[(df.dataset == ds) & (df.model == model)]
            if eligible_only and len(sub) != n:
                raise RuntimeError(f"join N mismatch {ds} {model}: {len(sub)} != {n}")
            if sub["item_id"].duplicated().any():
                raise RuntimeError(f"duplicate scores {ds} {model}")
    return df


def _impute_median_train(df_tr: pd.DataFrame, df_te: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    Xtr = df_tr[cols].astype(float).copy()
    Xte = df_te[cols].astype(float).copy()
    for c in cols:
        med = Xtr[c].median()
        fill = float(med) if np.isfinite(med) else 0.0
        Xtr[c] = Xtr[c].fillna(fill)
        Xte[c] = Xte[c].fillna(fill)
    return Xtr, Xte


def oof_metrics(df: pd.DataFrame, feature_cols: list[str], y: np.ndarray) -> dict[str, float]:
    """Ridge + StandardScaler OOF; scaler fit on train fold only (frozen CV settings)."""
    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=CV_SEED)
    preds = np.zeros(len(df))
    for tr, te in kf.split(df):
        Xtr_df, Xte_df = _impute_median_train(df.iloc[tr], df.iloc[te], feature_cols)
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(Xtr_df.values)
        Xte = scaler.transform(Xte_df.values)
        reg = Ridge(alpha=RIDGE_ALPHA).fit(Xtr, y[tr])
        preds[te] = reg.predict(Xte)
    ss_res = np.sum((y - preds) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rmse = float(np.sqrt(np.mean((y - preds) ** 2)))
    mae = float(np.mean(np.abs(y - preds)))
    return {"oof_r2": float(r2), "oof_rmse": rmse, "oof_mae": mae, "preds": preds}


def delta_r2_point(df: pd.DataFrame) -> dict[str, float]:
    y = df["learner_error"].values.astype(float)
    base = oof_metrics(df, SHARED_CORE, y)
    aug = oof_metrics(df, SHARED_CORE + ["score"], y)
    return {
        "base_r2": base["oof_r2"],
        "aug_r2": aug["oof_r2"],
        "delta_r2": aug["oof_r2"] - base["oof_r2"],
        "base_rmse": base["oof_rmse"],
        "aug_rmse": aug["oof_rmse"],
        "delta_rmse": aug["oof_rmse"] - base["oof_rmse"],
        "base_mae": base["oof_mae"],
        "aug_mae": aug["oof_mae"],
        "delta_mae": aug["oof_mae"] - base["oof_mae"],
    }


def bootstrap_delta_r2_ci(df: pd.DataFrame) -> tuple[float, float, float]:
    """Descriptive uncertainty for ΔR² via item bootstrap (not a registered confirmatory p-test)."""
    rng = np.random.default_rng(BOOT_SEED)
    point = delta_r2_point(df)["delta_r2"]
    vals = []
    n = len(df)
    for _ in range(BOOT_N):
        idx = rng.integers(0, n, n)
        sub = df.iloc[idx].reset_index(drop=True)
        try:
            vals.append(delta_r2_point(sub)["delta_r2"])
        except Exception:
            pass
    lo, hi = (np.quantile(vals, [0.025, 0.975]) if vals else (np.nan, np.nan))
    return float(point), float(lo), float(hi)


def family_a(df: pd.DataFrame, registry: dict) -> pd.DataFrame:
    rows = []
    for h in registry["family_A"]["hypotheses"]:
        sub = df[(df.dataset == h["dataset"]) & (df.model == h["llm"])].copy()
        x = sub["score"].values
        y = sub["learner_error"].values
        sp = stats.spearmanr(x, y)
        rho, lo, hi = bootstrap_ci(x, y, lambda a, b: stats.spearmanr(a, b).correlation)
        pr = stats.pearsonr(x, y)
        kt = stats.kendalltau(x, y)
        rows.append(
            {
                "hypothesis_id": h["hypothesis_id"],
                "family": "A",
                "dataset": h["dataset"],
                "model": h["llm"],
                "n_items": len(sub),
                "spearman_rho": float(sp.correlation),
                "ci_lo": lo,
                "ci_hi": hi,
                "raw_p": float(sp.pvalue),
                "pearson_r": float(pr.statistic),
                "pearson_p": float(pr.pvalue),
                "kendall_tau": float(kt.correlation),
                "kendall_p": float(kt.pvalue),
                "confirmatory": True,
                "analysis_code_commit": SCORING_COMMIT,  # parent; updated below
            }
        )
    out = pd.DataFrame(rows)
    out["holm_p"] = holm(out["raw_p"].tolist())
    assert len(out) == 6
    return out


def family_b(df: pd.DataFrame, registry: dict) -> tuple[pd.DataFrame, bool]:
    """Point estimates + descriptive bootstrap CI. Confirmatory p-values BLOCKED (absent in V2)."""
    rows = []
    for h in registry["family_B"]["hypotheses"]:
        sub = df[(df.dataset == h["dataset"]) & (df.model == h["llm"])].reset_index(drop=True)
        met = delta_r2_point(sub)
        d, lo, hi = bootstrap_delta_r2_ci(sub)
        rows.append(
            {
                "hypothesis_id": h["hypothesis_id"],
                "family": "B",
                "dataset": h["dataset"],
                "model": h["llm"],
                "n_items": len(sub),
                "base_r2": met["base_r2"],
                "aug_r2": met["aug_r2"],
                "delta_r2": met["delta_r2"],
                "ci_lo": lo,
                "ci_hi": hi,
                "raw_p": np.nan,
                "holm_p": np.nan,
                "delta_rmse": met["delta_rmse"],
                "delta_mae": met["delta_mae"],
                "base_rmse": met["base_rmse"],
                "aug_rmse": met["aug_rmse"],
                "base_mae": met["base_mae"],
                "aug_mae": met["aug_mae"],
                "inferential_test": h["test"],
                "confirmatory_p_available": False,
                "blocker": "FAMILY_B_TEST_SPECIFICATION_BLOCKER",
                "note": "V2 incremental block defines ΔR²/RMSE/MAE but no confirmatory p-value procedure; no substitute invented",
                "confirmatory": True,
            }
        )
    out = pd.DataFrame(rows)
    assert len(out) == 6
    return out, True  # blocker True


def rq1_summaries(df_all_scored: pd.DataFrame, df_primary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # score characterization on scored/text-scoreable universes
    score_rows = []
    for ds in ["xes3g5m", "junyi", "dbe_kt22"]:
        for model in MODELS:
            # all scored items for ds/model from primary join is only eligible; use broader
            pass
    # rebuild from score sources
    llm, _ = load_xes_junyi_scores()
    dbe = load_dbe_scores()
    xj = llm[llm.dataset.isin(["xes3g5m", "junyi"])][
        ["dataset", "item_id_hash", "model_identifier", "scalar_difficulty"]
    ].rename(columns={"item_id_hash": "item_id", "model_identifier": "model", "scalar_difficulty": "score"})
    xj["item_id"] = xj["item_id"].astype(str)
    all_s = pd.concat([xj, dbe], ignore_index=True)
    for (ds, model), g in all_s.groupby(["dataset", "model"]):
        s = g["score"].astype(float)
        score_rows.append(
            {
                "dataset": ds,
                "model": model,
                "n": int(len(s)),
                "mean": float(s.mean()),
                "sd": float(s.std(ddof=1)),
                "median": float(s.median()),
                "iqr": float(s.quantile(0.75) - s.quantile(0.25)),
                "min": float(s.min()),
                "max": float(s.max()),
                "n_unique": int(s.nunique()),
                "frac_exact_0": float((s == 0).mean()),
                "frac_exact_1": float((s == 1).mean()),
                "label": "SECONDARY",
            }
        )
    score_sum = pd.DataFrame(score_rows)

    # cross-model
    cross_rows = []
    for ds in ["xes3g5m", "junyi", "dbe_kt22"]:
        a = all_s[(all_s.dataset == ds) & (all_s.model == "gpt-4o-mini")][["item_id", "score"]].rename(columns={"score": "mini"})
        b = all_s[(all_s.dataset == ds) & (all_s.model == "gpt-5.4")][["item_id", "score"]].rename(columns={"score": "g54"})
        m = a.merge(b, on="item_id")
        sp = stats.spearmanr(m["mini"], m["g54"])
        pr = stats.pearsonr(m["mini"], m["g54"])
        kt = stats.kendalltau(m["mini"], m["g54"])
        rho, lo, hi = bootstrap_ci(m["mini"].values, m["g54"].values, lambda x, y: stats.spearmanr(x, y).correlation)
        cross_rows.append(
            {
                "dataset": ds,
                "n": len(m),
                "spearman": float(sp.correlation),
                "spearman_ci_lo": lo,
                "spearman_ci_hi": hi,
                "spearman_p": float(sp.pvalue),
                "pearson": float(pr.statistic),
                "pearson_p": float(pr.pvalue),
                "kendall": float(kt.correlation),
                "kendall_p": float(kt.pvalue),
                "label": "SECONDARY",
            }
        )
    cross = pd.DataFrame(cross_rows)

    # surface associations on primary eligible
    surf_rows = []
    for ds in ["xes3g5m", "junyi", "dbe_kt22"]:
        for model in MODELS:
            sub = df_primary[(df_primary.dataset == ds) & (df_primary.model == model)]
            for feat in SHARED_CORE:
                sp = stats.spearmanr(sub["score"], sub[feat])
                surf_rows.append(
                    {
                        "dataset": ds,
                        "model": model,
                        "feature": feat,
                        "n": len(sub),
                        "spearman": float(sp.correlation) if np.isfinite(sp.correlation) else np.nan,
                        "raw_p": float(sp.pvalue) if np.isfinite(sp.pvalue) else np.nan,
                        "label": "SECONDARY",
                    }
                )
    surf = pd.DataFrame(surf_rows)
    return score_sum, cross, surf


def legacy_sensitivity(llm: pd.DataFrame, features: pd.DataFrame, ds: str) -> pd.DataFrame:
    leg = pd.read_csv(LEGACY_REF)
    leg = leg[(leg["dataset"] == ds) & (leg["reference_scope"] == "held_out_test") & (leg["heldout_response_count"] >= 20)].copy()
    leg["item_id"] = leg["item_id_hash"].astype(str)
    leg["learner_error"] = leg["smoothed_error_beta_1_1"].astype(float)
    feat = features[features.dataset == ds]
    rows = []
    for model in MODELS:
        sc = llm[(llm.dataset == ds) & (llm.model_identifier == model)][
            ["item_id_hash", "scalar_difficulty"]
        ].rename(columns={"item_id_hash": "item_id", "scalar_difficulty": "score"})
        sc["item_id"] = sc["item_id"].astype(str)
        df = leg.merge(sc, on="item_id").merge(feat, on=["dataset", "item_id"])
        sp = stats.spearmanr(df["score"], df["learner_error"])
        rho, lo, hi = bootstrap_ci(df["score"].values, df["learner_error"].values, lambda a, b: stats.spearmanr(a, b).correlation)
        pr = stats.pearsonr(df["score"], df["learner_error"])
        kt = stats.kendalltau(df["score"], df["learner_error"])
        met = delta_r2_point(df.reset_index(drop=True))
        d, dlo, dhi = bootstrap_delta_r2_ci(df.reset_index(drop=True))
        rows.append(
            {
                "dataset": ds,
                "model": model,
                "n_items": len(df),
                "spearman_rho": float(sp.correlation),
                "ci_lo": lo,
                "ci_hi": hi,
                "raw_p": float(sp.pvalue),
                "pearson_r": float(pr.statistic),
                "kendall_tau": float(kt.correlation),
                "base_r2": met["base_r2"],
                "aug_r2": met["aug_r2"],
                "delta_r2": met["delta_r2"],
                "delta_r2_ci_lo": dlo,
                "delta_r2_ci_hi": dhi,
                "delta_rmse": met["delta_rmse"],
                "delta_mae": met["delta_mae"],
                "label": "SENSITIVITY",
                "construct": "LEGACY_ALL_RESPONSE",
            }
        )
    return pd.DataFrame(rows)


def dbe_consensus_errors() -> pd.DataFrame:
    """Rebuild consensus-only FIRST_OBSERVED test errors (deterministic Phase-1.1 rule)."""
    from tlt3d_phase11_amendment import first_observed_learner_item, item_error_from_first

    tx = pd.read_csv(DBE_TX)
    choices = pd.read_csv(DBE_CHOICES)
    universe = json.loads((CFG / "dbe_item_universe.json").read_text())
    included = set(int(i) for i in universe["included_item_ids"])
    splits = pd.read_csv(DBE_SPLITS)
    split_map = dict(zip(splits["student_id_raw"].astype(str), splits["split_assignment"]))

    def map_correct(x):
        if x is True or str(x).lower() == "true":
            return True
        if x is False or str(x).lower() == "false":
            return False
        return None

    tx = tx.copy()
    tx["_ans"] = tx["answer_state"].map(map_correct)
    ch_lookup = dict(zip(choices["id"].astype(int), choices["is_correct"].map(map_correct)))
    tx["_from_choice"] = tx["answer_choice_id"].map(
        lambda cid: ch_lookup.get(int(cid)) if pd.notna(cid) and int(cid) in ch_lookup else None
    )
    agree = tx["_ans"].notna() & tx["_from_choice"].notna() & (tx["_ans"] == tx["_from_choice"])
    no_recon = tx["_ans"].notna() & tx["_from_choice"].isna()
    keep = agree | no_recon
    cons = tx[keep & tx["question_id"].astype(int).isin(included) & tx["_ans"].notna()].copy()
    cons["learner_id"] = cons["student_id"].astype(str)
    cons["item_id"] = cons["question_id"].astype(int)
    cons["is_correct"] = cons["_ans"].astype(bool)
    cons["split_assignment"] = cons["learner_id"].map(split_map)
    cons = cons[cons["split_assignment"].notna()].copy()
    cons["ts"] = pd.to_datetime(cons["start_time"], utc=True, errors="coerce")
    cons["_ord"] = cons["id"]
    first = first_observed_learner_item(
        cons,
        learner_col="learner_id",
        item_col="item_id",
        correct_col="is_correct",
        order_cols=["ts", "_ord"],
        split_col="split_assignment",
        split_value="test",
    )
    err = item_error_from_first(first, "item_id", "is_correct")
    err["item_id"] = err["item_id"].astype(str)
    err["dataset"] = "dbe_kt22"
    err["learner_error"] = err["smoothed_error_beta_1_1"]
    err["eligible_primary"] = err["n_first_observed"] >= 20
    return err


def dbe_consensus_sensitivity(features: pd.DataFrame, dbe_scores: pd.DataFrame) -> pd.DataFrame:
    err = dbe_consensus_errors()
    elig = err[err["eligible_primary"]].copy()
    feat = features[features.dataset == "dbe_kt22"]
    rows = []
    for model in MODELS:
        sc = dbe_scores[dbe_scores.model == model][["item_id", "score"]]
        df = elig.merge(sc, on="item_id").merge(feat, on=["dataset", "item_id"])
        sp = stats.spearmanr(df["score"], df["learner_error"])
        rho, lo, hi = bootstrap_ci(df["score"].values, df["learner_error"].values, lambda a, b: stats.spearmanr(a, b).correlation)
        met = delta_r2_point(df.reset_index(drop=True))
        d, dlo, dhi = bootstrap_delta_r2_ci(df.reset_index(drop=True))
        rows.append(
            {
                "dataset": "dbe_kt22",
                "model": model,
                "n_items": len(df),
                "spearman_rho": float(sp.correlation),
                "ci_lo": lo,
                "ci_hi": hi,
                "raw_p": float(sp.pvalue),
                "base_r2": met["base_r2"],
                "aug_r2": met["aug_r2"],
                "delta_r2": met["delta_r2"],
                "delta_r2_ci_lo": dlo,
                "delta_r2_ci_hi": dhi,
                "delta_rmse": met["delta_rmse"],
                "delta_mae": met["delta_mae"],
                "label": "SENSITIVITY",
                "construct": "DBE_CONSENSUS_ONLY",
            }
        )
    return pd.DataFrame(rows)


def dbe_expert_secondary(fo: pd.DataFrame, dbe_scores: pd.DataFrame) -> pd.DataFrame:
    items = pd.read_parquet(DBE_ITEMS, columns=["item_id", "expert_difficulty_secondary_only"])
    items["item_id"] = items["item_id"].astype(str)
    items = items.rename(columns={"expert_difficulty_secondary_only": "expert"})
    primary = fo[(fo.dataset == "dbe_kt22") & (fo.eligible_primary)][["item_id", "learner_error"]]
    m = primary.merge(items, on="item_id")
    rows = []
    # expert vs learner error
    sp = stats.spearmanr(m["expert"], m["learner_error"])
    kt = stats.kendalltau(m["expert"], m["learner_error"])
    rows.append({"contrast": "expert_vs_FO_learner_error", "n": len(m), "spearman": float(sp.correlation), "spearman_p": float(sp.pvalue), "kendall": float(kt.correlation), "kendall_p": float(kt.pvalue), "label": "SECONDARY_EXPLORATORY"})
    for model in MODELS:
        sc = dbe_scores[dbe_scores.model == model][["item_id", "score"]]
        mm = m.merge(sc, on="item_id")
        sp = stats.spearmanr(mm["expert"], mm["score"])
        kt = stats.kendalltau(mm["expert"], mm["score"])
        rows.append({"contrast": f"expert_vs_{model}", "n": len(mm), "spearman": float(sp.correlation), "spearman_p": float(sp.pvalue), "kendall": float(kt.correlation), "kendall_p": float(kt.pvalue), "label": "SECONDARY_EXPLORATORY"})
    return pd.DataFrame(rows)


def write_feature_availability(features: pd.DataFrame) -> None:
    lines = ["# TLT-3D P3A — Feature Availability", ""]
    lines.append("| Dataset | Feature | N | structural_zero | missing | transform | source |")
    lines.append("|---|---|---:|---:|---:|---|---|")
    sources = {
        "xes3g5m": "tables/AUTHENTIC_ITEM_SURFACE_FEATURES.csv",
        "junyi": "tables/AUTHENTIC_ITEM_SURFACE_FEATURES.csv",
        "dbe_kt22": "canonical scoring_text + tlt3d_surface_features (sentence_count computed; answer_option_count=n_choices)",
    }
    for ds in ["xes3g5m", "junyi", "dbe_kt22"]:
        sub = features[features.dataset == ds]
        for feat in SHARED_CORE:
            v = sub[feat]
            lines.append(
                f"| {ds} | {feat} | {len(v)} | {int((v==0).sum())} | {int(v.isna().sum())} | as-is numeric | {sources[ds]} |"
            )
    lines.append("\n`concept_count` excluded from primary shared-core (SECONDARY_METADATA_FEATURE).\n")
    (REP / "TLT3D_P3A_FEATURE_AVAILABILITY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ART.mkdir(parents=True, exist_ok=True)
    REP.mkdir(parents=True, exist_ok=True)
    assert_input_integrity()
    registry = json.loads((CFG / "confirmatory_family_registry.json").read_text())
    llm, score_manifest = load_xes_junyi_scores()
    fo = load_primary_fo_errors()
    features = load_shared_core_features()
    write_feature_availability(features)
    dbe_scores = load_dbe_scores()
    primary = build_analysis_frame(fo, llm, dbe_scores, features, eligible_only=True)

    # mechanical sanity
    assert primary["learner_error"].between(0, 1).all()
    assert primary["score"].between(0, 1).all()

    score_sum, cross, surf = rq1_summaries(primary, primary)
    score_sum.to_csv(ART / "P3A_RQ1_SCORE_SUMMARY.csv", index=False)
    cross.to_csv(ART / "P3A_CROSS_MODEL_AGREEMENT.csv", index=False)
    surf.to_csv(ART / "P3A_RQ1_SURFACE_ASSOCIATIONS.csv", index=False)

    fam_a = family_a(primary, registry)
    fam_a.to_csv(ART / "family_A_confirmatory_results.csv", index=False)

    fam_b, b_blocker = family_b(primary, registry)
    fam_b.to_csv(ART / "family_B_confirmatory_results.csv", index=False)

    jun_leg = legacy_sensitivity(llm, features, "junyi")
    xes_leg = legacy_sensitivity(llm, features, "xes3g5m")
    jun_leg.to_csv(ART / "P3A_JUNYI_LEGACY_SENSITIVITY.csv", index=False)
    xes_leg.to_csv(ART / "P3A_XES_LEGACY_SENSITIVITY.csv", index=False)

    cons = dbe_consensus_sensitivity(features, dbe_scores)
    cons.to_csv(ART / "P3A_DBE_CONSENSUS_SENSITIVITY.csv", index=False)

    expert = dbe_expert_secondary(fo, dbe_scores)
    expert.to_csv(ART / "P3A_DBE_EXPERT_SECONDARY.csv", index=False)

    # synthesis
    syn_rows = []
    for _, a in fam_a.iterrows():
        b = fam_b[(fam_b.dataset == a.dataset) & (fam_b.model == a.model)].iloc[0]
        syn_rows.append(
            {
                "dataset": a.dataset,
                "model": a.model,
                "family_A_rho": a.spearman_rho,
                "family_A_ci": f"[{a.ci_lo:.4f}, {a.ci_hi:.4f}]",
                "family_A_holm_p": a.holm_p,
                "family_B_delta_r2": b.delta_r2,
                "family_B_ci": f"[{b.ci_lo:.4f}, {b.ci_hi:.4f}]",
                "family_B_holm_p": b.holm_p,
            }
        )
    syn = pd.DataFrame(syn_rows)
    syn.to_csv(ART / "P3A_THREE_DATASET_SYNTHESIS.csv", index=False)

    # confirmatory JSON
    import subprocess

    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    conf = {
        "phase": "TLT3D_P3A",
        "protocol_commit": PROTOCOL_COMMIT,
        "scoring_commit": SCORING_COMMIT,
        "analysis_commit": commit,
        "family_A": fam_a.to_dict(orient="records"),
        "family_B": fam_b.to_dict(orient="records"),
        "family_A_count": 6,
        "family_B_count": 6,
        "family_B_test_blocker": b_blocker,
        "FAMILY_B_TEST_SPECIFICATION_BLOCKER": True,
        "source_hashes": EXPECTED_HASHES,
        "generated_at_utc": utc_now(),
    }
    (ART / "TLT3D_P3A_CONFIRMATORY_RESULTS.json").write_text(json.dumps(conf, indent=2, default=str) + "\n")

    # reports written by companion function below via calling write_reports
    write_reports(fam_a, fam_b, score_sum, cross, surf, jun_leg, xes_leg, cons, expert, syn, b_blocker)
    print(json.dumps({"ok": True, "FAMILY_B_TEST_SPECIFICATION_BLOCKER": True, "family_A_rows": 6, "family_B_rows": 6}, indent=2))
    return 0


def _df_md(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except Exception:
        return "```\n" + df.to_csv(index=False) + "```"


def write_reports(fam_a, fam_b, score_sum, cross, surf, jun_leg, xes_leg, cons, expert, syn, b_blocker):
    def fmt_a(r):
        return f"| {r.hypothesis_id} | {r.dataset} | {r.model} | {r.n_items} | {r.spearman_rho:.4f} | [{r.ci_lo:.4f}, {r.ci_hi:.4f}] | {r.raw_p:.4g} | {r.holm_p:.4g} |"

    def fmt_b(r):
        rp = "NA" if pd.isna(r.raw_p) else f"{r.raw_p:.4g}"
        hp = "NA" if pd.isna(r.holm_p) else f"{r.holm_p:.4g}"
        return f"| {r.hypothesis_id} | {r.dataset} | {r.model} | {r.n_items} | {r.base_r2:.4f} | {r.aug_r2:.4f} | {r.delta_r2:.4f} | [{r.ci_lo:.4f}, {r.ci_hi:.4f}] | {rp} | {hp} | {r.delta_rmse:.4f} | {r.delta_mae:.4f} |"

    # strongest surface
    strong = []
    for ds in ["xes3g5m", "junyi", "dbe_kt22"]:
        for model in MODELS:
            sub = surf[(surf.dataset == ds) & (surf.model == model)].dropna(subset=["spearman"])
            if len(sub):
                i = sub["spearman"].abs().idxmax()
                strong.append((ds, model, sub.loc[i, "feature"], sub.loc[i, "spearman"]))

    md = []
    md.append("# TLT-3D P3A — Measurement Validity Results")
    md.append("")
    md.append("## 1. Execution / Protocol Identity")
    md.append(f"- Scoring commit: `{SCORING_COMMIT}`")
    md.append(f"- Protocol commit: `{PROTOCOL_COMMIT}`")
    md.append(f"- Tag: `tlt3d-pre-dbe-scoring-v1.2`")
    md.append("")
    md.append("## 2. Dataset and Join Integrity")
    md.append("| Dataset | Primary FO eligible | Join complete |")
    md.append("|---|---:|---|")
    md.append("| XES3G5M | 3265 | PASS |")
    md.append("| Junyi | 169 | PASS |")
    md.append("| DBE-KT22 | 166 | PASS |")
    md.append("")
    md.append("## 3. RQ1 Score Distributions")
    md.append(_df_md(score_sum))
    md.append("")
    md.append("## 4. RQ1 Cross-Model Agreement")
    md.append(_df_md(cross))
    md.append("")
    md.append("## 5. RQ1 Visible-Feature Associations")
    md.append("See `artifacts/tlt3d/P3A_RQ1_SURFACE_ASSOCIATIONS.csv` (SECONDARY).")
    md.append("")
    md.append("Strongest |ρ| per dataset×model:")
    for ds, model, feat, rho in strong:
        md.append(f"- {ds} / {model}: {feat} = {rho:.4f}")
    md.append("")
    md.append("## 6. Family A — Authentic Learner Correspondence")
    md.append("| ID | Dataset | Model | N | Spearman rho | 95% CI | raw p | Holm p |")
    md.append("|---|---|---|---:|---:|---|---:|---:|")
    for _, r in fam_a.iterrows():
        md.append(fmt_a(r))
    md.append("")
    md.append("## 7. Family B — Incremental Information")
    md.append("")
    md.append("**FAMILY_B_TEST_SPECIFICATION_BLOCKER:** authoritative V2 incremental block defines ΔR²/RMSE/MAE but **no confirmatory p-value**. No substitute test invented. Descriptive bootstrap CIs for ΔR² are reported for uncertainty only; `raw_p`/`holm_p` = NA.")
    md.append("")
    md.append("| ID | Dataset | Model | N | Base R² | Aug R² | ΔR² | 95% CI | raw p | Holm p | ΔRMSE | ΔMAE |")
    md.append("|---|---|---|---:|---:|---:|---:|---|---|---|---:|---:|")
    for _, r in fam_b.iterrows():
        md.append(fmt_b(r))
    md.append("")
    md.append("## 8. Junyi Legacy-All-Response Sensitivity")
    md.append(_df_md(jun_leg))
    md.append("")
    md.append("## 9. XES Legacy-All-Response Sensitivity")
    md.append(_df_md(xes_leg))
    md.append("")
    md.append("## 10. DBE Consensus-Only Sensitivity")
    md.append(_df_md(cons))
    md.append("")
    md.append("## 11. DBE Expert-Difficulty Secondary Analysis")
    md.append(_df_md(expert))
    md.append("")
    md.append("## 12. Cross-Dataset Descriptive Synthesis")
    md.append(_df_md(syn))
    md.append("")
    md.append("## 13. Protocol Deviations")
    md.append("PROTOCOL_DEVIATIONS = NONE (except Family B confirmatory p-value specification absent in sealed implementation reference)")
    md.append("")
    md.append("## 14. Scientific Findings — FACTUAL ONLY")
    md.append("- Family A: six Spearman estimates computed; Holm applied within Family A only.")
    md.append("- Family B: six ΔR² estimates computed under frozen Ridge/OOF; confirmatory p/Holm blocked pending PI test specification.")
    md.append("- No pooled cross-dataset Spearman.")
    md.append("- No Family C/D / KT executed.")
    (REP / "TLT3D_P3A_MEASUREMENT_VALIDITY_RESULTS.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    # surprises
    sur = ["# TLT-3D P3A — Surprises and Tensions", "", "Objective tensions only; no rhetorical resolution.", ""]
    # DBE vs others signs
    for _, r in fam_a.iterrows():
        sur.append(f"- {r.hypothesis_id}: rho={r.spearman_rho:.4f}, Holm p={r.holm_p:.4g}")
    for _, r in fam_b.iterrows():
        sur.append(f"- {r.hypothesis_id}: ΔR²={r.delta_r2:.4f} (sign preserved)")
        if r.delta_r2 < 0:
            sur.append(f"  - NEGATIVE ΔR² observed for {r.hypothesis_id}")
    # Mini vs 5.4
    for ds in ["xes3g5m", "junyi", "dbe_kt22"]:
        a = fam_a[(fam_a.dataset == ds) & (fam_a.model == "gpt-4o-mini")].iloc[0]
        b = fam_a[(fam_a.dataset == ds) & (fam_a.model == "gpt-5.4")].iloc[0]
        if np.sign(a.spearman_rho) != np.sign(b.spearman_rho):
            sur.append(f"- Sign disagreement Mini vs 5.4 on Family A for {ds}")
        if abs(a.spearman_rho - b.spearman_rho) > 0.1:
            sur.append(f"- Material |Δrho| Mini vs 5.4 on {ds}: {abs(a.spearman_rho-b.spearman_rho):.3f}")
    # primary vs junyi legacy
    for model in MODELS:
        prim = fam_a[(fam_a.dataset == "junyi") & (fam_a.model == model)].iloc[0]
        leg = jun_leg[jun_leg.model == model].iloc[0]
        if np.sign(prim.spearman_rho) != np.sign(leg.spearman_rho):
            sur.append(f"- Junyi {model}: primary vs legacy Spearman SIGN CHANGE ({prim.spearman_rho:.3f} vs {leg.spearman_rho:.3f})")
        elif abs(prim.spearman_rho - leg.spearman_rho) > 0.05:
            sur.append(f"- Junyi {model}: primary vs legacy |Δrho|={abs(prim.spearman_rho-leg.spearman_rho):.3f}")
    sur.append("- Family B confirmatory inferential test unspecified in V2 implementation → Holm for B not applied.")
    (REP / "TLT3D_P3A_SURPRISES_AND_TENSIONS.md").write_text("\n".join(sur) + "\n", encoding="utf-8")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        raise SystemExit(2)
