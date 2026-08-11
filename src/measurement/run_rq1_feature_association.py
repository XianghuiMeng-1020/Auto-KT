#!/usr/bin/env python3
"""RQ1 visible-feature association analysis.

Outcome variable: the LLM-estimated difficulty score (scalar_difficulty).
Predictors: text-visible item characteristics only. No learner outcome
(held-out error, correctness) enters this model. Item usage/exposure counts
are excluded because they are not observable from item content.

This complements the RQ2 incremental-validity model, which predicts held-out
learner error. Here the score itself is the dependent variable, so the analysis
characterizes which visible cues are associated with the generated ranking; it
is not a complete decomposition of how the LLM reasoned.

Population: the primary learner-error analysis set (held-out test scope,
>= 20 held-out responses per item), matching the frozen score--length vs
score--error paired comparison. XES3G5M: 3,279 items; Junyi: 183 items.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import scipy
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "measurement"))

from measurement_common import (  # noqa: E402
    DATASETS,
    LLM_FEATURES,
    REPORT_DIR,
    TABLE_DIR,
    bootstrap_ci,
    git_commit,
    holm_correction,
    load_config,
    load_llm_features,
    sha256_file,
    utc_now,
)

PRIMARY_SCOPE = "held_out_test"
MODELS = ["gpt-4o-mini", "gpt-5.4"]

# Text-visible candidate features (the item-content cues an LLM could read).
# Usage/exposure counts and learner outcomes are deliberately excluded.
CAND_NUM = [
    "char_length", "token_length", "math_symbol_count",
    "equation_count", "answer_option_count", "concept_count",
]
CAND_CAT = ["has_image_dependency", "item_format", "mathematical_domain", "educational_level"]

REFS_CSV = TABLE_DIR / "AUTHENTIC_DIFFICULTY_REFERENCES.csv"
SURFACE_CSV = TABLE_DIR / "AUTHENTIC_ITEM_SURFACE_FEATURES.csv"


def build_frame(refs, surface, llm, dataset, model, threshold):
    held = refs[(refs["dataset"] == dataset) & (refs["reference_scope"] == PRIMARY_SCOPE)]
    held = held[held["n_responses"] >= threshold][["item_id_hash"]]
    llm_sub = llm[(llm["dataset"] == dataset) & (llm["model_identifier"] == model)][
        ["item_id_hash", "scalar_difficulty"]
    ]
    surf = surface[surface["dataset"] == dataset]
    df = held.merge(llm_sub, on="item_id_hash", how="inner").merge(
        surf, on="item_id_hash", how="left"
    )
    return df


def active_features(df):
    """Drop within-dataset zero-variance features before fitting."""
    num = [c for c in CAND_NUM if c in df.columns and df[c].nunique(dropna=True) > 1]
    cat = [c for c in CAND_CAT if c in df.columns and df[c].astype(str).nunique(dropna=True) > 1]
    return num, cat


def bivariate(df, dataset, model, num, cfg):
    boot = cfg["bootstrap"]
    rows = []
    y = df["scalar_difficulty"].values
    for feat in num:
        x = pd.to_numeric(df[feat], errors="coerce").values
        rho, lo, hi = bootstrap_ci(
            x, y, lambda a, b: stats.spearmanr(a, b).correlation,
            n_boot=boot["n"], seed=boot["seed"],
        )
        mask = np.isfinite(x) & np.isfinite(y)
        p = float(stats.spearmanr(x[mask], y[mask]).pvalue)
        rows.append({
            "dataset": dataset, "model": model, "feature": feat,
            "n_items": int(mask.sum()), "spearman_rho": rho,
            "ci_lo": lo, "ci_hi": hi, "p_value": p,
        })
    return rows


def make_pipeline(num, cat, alpha_grid):
    transformers = []
    if num:
        transformers.append(("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), num))
    if cat:
        transformers.append(("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore")),
        ]), cat))
    pre = ColumnTransformer(transformers, remainder="drop")
    inner = KFold(n_splits=3, shuffle=True, random_state=cfg_cv_seed)
    grid = GridSearchCV(
        Ridge(), {"alpha": alpha_grid}, cv=inner,
        scoring="neg_mean_squared_error",
    )
    return Pipeline([("pre", pre), ("ridge", grid)])


def feature_names(pipe, num, cat):
    names = list(num)
    if cat:
        ohe = pipe.named_steps["pre"].named_transformers_["cat"].named_steps["ohe"]
        names += list(ohe.get_feature_names_out(cat))
    return names


def multivariable(df, dataset, model, num, cat, cfg):
    alpha_grid = cfg["ridge_alpha_grid"]
    X = df[num + cat]
    y = df["scalar_difficulty"].values
    kf = KFold(n_splits=cfg["cv_folds"], shuffle=True, random_state=cfg["cv_seed"])

    oof = np.zeros(len(df))
    std_coefs = {}      # standardized numeric coefficients across folds
    perm_imp = {}       # grouped permutation importance across folds
    chosen_alphas = []

    for tr_idx, te_idx in kf.split(X):
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]
        pipe = make_pipeline(num, cat, alpha_grid)
        pipe.fit(X_tr, y_tr)
        oof[te_idx] = pipe.predict(X_te)
        chosen_alphas.append(float(pipe.named_steps["ridge"].best_params_["alpha"]))

        best = pipe.named_steps["ridge"].best_estimator_
        names = feature_names(pipe, num, cat)
        for n, c in zip(names, best.coef_):
            if n in num:  # standardized numeric coefficient
                std_coefs.setdefault(n, []).append(float(c))

        # grouped permutation importance on the held-out fold (drop in R2 -> we use increase in MSE)
        groups = {f: [f] for f in num}
        for c in cat:
            groups[c] = [c]
        r = permutation_importance(
            pipe, X_te, y_te, n_repeats=20,
            random_state=cfg["cv_seed"], scoring="r2",
        )
        for gi, g in enumerate(num + cat):
            perm_imp.setdefault(g, []).append(float(r.importances_mean[gi]))

    ss_res = float(np.sum((y - oof) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rmse = float(np.sqrt(np.mean((y - oof) ** 2)))
    mae = float(np.mean(np.abs(y - oof)))

    summary = {
        "dataset": dataset, "model": model, "n_items": len(df),
        "n_numeric_features": len(num), "n_categorical_features": len(cat),
        "oof_r2": float(r2), "oof_rmse": rmse, "oof_mae": mae,
        "chosen_alphas": chosen_alphas,
    }
    coef_rows = []
    for f in num:
        arr = np.array(std_coefs.get(f, [np.nan]))
        pi = np.array(perm_imp.get(f, [np.nan]))
        coef_rows.append({
            "dataset": dataset, "model": model, "feature": f, "feature_type": "numeric",
            "std_coef_mean": float(np.mean(arr)), "std_coef_sd": float(np.std(arr)),
            "perm_importance_mean": float(np.mean(pi)), "perm_importance_sd": float(np.std(pi)),
        })
    for f in cat:
        pi = np.array(perm_imp.get(f, [np.nan]))
        coef_rows.append({
            "dataset": dataset, "model": model, "feature": f, "feature_type": "categorical",
            "std_coef_mean": float("nan"), "std_coef_sd": float("nan"),
            "perm_importance_mean": float(np.mean(pi)), "perm_importance_sd": float(np.std(pi)),
        })
    return summary, coef_rows


cfg_cv_seed = 2024  # module-level default used inside make_pipeline inner CV


def main() -> int:
    cfg = load_config()
    global cfg_cv_seed
    cfg_cv_seed = cfg["cv_seed"]
    cfg.setdefault("ridge_alpha_grid", [0.01, 0.1, 1, 10, 100])

    refs = pd.read_csv(REFS_CSV)
    surface = pd.read_csv(SURFACE_CSV)
    llm = load_llm_features()
    threshold = cfg["primary_response_threshold"]

    biv_rows, multi_rows, coef_rows = [], [], []
    features_used = {}
    for dataset in DATASETS:
        for model in MODELS:
            df = build_frame(refs, surface, llm, dataset, model, threshold)
            assert df["item_id_hash"].is_unique, f"non-unique items {dataset}/{model}"
            assert df["scalar_difficulty"].notna().all(), "missing LLM score"
            num, cat = active_features(df)
            features_used[f"{dataset}/{model}"] = {"numeric": num, "categorical": cat}
            biv_rows.extend(bivariate(df, dataset, model, num, cfg))
            summ, crows = multivariable(df, dataset, model, num, cat, cfg)
            multi_rows.append(summ)
            coef_rows.extend(crows)

    biv_df = pd.DataFrame(biv_rows)
    biv_df["p_value_holm"] = holm_correction(biv_df["p_value"].tolist())
    multi_df = pd.DataFrame(multi_rows)
    coef_df = pd.DataFrame(coef_rows)

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    biv_df.to_csv(TABLE_DIR / "RQ1_FEATURE_BIVARIATE.csv", index=False)
    multi_df.to_csv(TABLE_DIR / "RQ1_FEATURE_MULTIVARIABLE.csv", index=False)
    coef_df.to_csv(TABLE_DIR / "RQ1_FEATURE_COEFFICIENTS.csv", index=False)

    receipt = {
        "analysis": "RQ1 visible-feature association (LLM score as outcome)",
        "generated_utc": utc_now(),
        "git_commit": git_commit(),
        "population": "held_out_test scope, n_responses >= %d" % threshold,
        "outcome": "scalar_difficulty (LLM-estimated difficulty)",
        "learner_outcome_in_model": False,
        "candidate_numeric_features": CAND_NUM,
        "candidate_categorical_features": CAND_CAT,
        "excluded_by_design": ["log_train_exposure", "train_responses", "test_responses",
                               "log_test_exposure", "held-out error (outcome)"],
        "zero_variance_dropped_within_dataset": True,
        "features_used": features_used,
        "cv_folds": cfg["cv_folds"],
        "cv_seed": cfg["cv_seed"],
        "inner_cv_folds": 3,
        "ridge_alpha_grid": cfg["ridge_alpha_grid"],
        "preprocessing": "median-impute+standardize (numeric); most-frequent-impute+one-hot (categorical); fit within training folds only",
        "bootstrap": cfg["bootstrap"],
        "permutation_importance_repeats": 20,
        "software": {
            "python": sys.version.split()[0],
            "numpy": np.__version__, "pandas": pd.__version__,
            "scipy": scipy.__version__, "sklearn": sklearn.__version__,
        },
        "inputs": {
            "references_csv": {"path": str(REFS_CSV.relative_to(ROOT)), "sha256": sha256_file(REFS_CSV)},
            "surface_csv": {"path": str(SURFACE_CSV.relative_to(ROOT)), "sha256": sha256_file(SURFACE_CSV)},
            "llm_features": {"path": str(LLM_FEATURES.relative_to(ROOT)), "sha256": sha256_file(LLM_FEATURES)},
        },
        "outputs": ["tables/RQ1_FEATURE_BIVARIATE.csv",
                    "tables/RQ1_FEATURE_MULTIVARIABLE.csv",
                    "tables/RQ1_FEATURE_COEFFICIENTS.csv"],
    }
    (REPORT_DIR / "RQ1_FEATURE_ASSOCIATION_RECEIPT.json").write_text(
        json.dumps(receipt, indent=2), encoding="utf-8"
    )

    pd.set_option("display.width", 160)
    print("=== RQ1 multivariable (LLM score ~ visible features) ===")
    print(multi_df[["dataset", "model", "n_items", "oof_r2", "oof_rmse", "oof_mae"]].to_string(index=False))
    print("\n=== RQ1 bivariate Spearman (score vs each visible feature) ===")
    print(biv_df[["dataset", "model", "feature", "spearman_rho", "ci_lo", "ci_hi", "p_value_holm"]].to_string(index=False))
    print("\n=== RQ1 feature contributions ===")
    print(coef_df.to_string(index=False))
    print("\nFeatures used:", json.dumps(features_used))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
