#!/usr/bin/env python3
"""Fold-internal incremental-validity robustness analysis."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
TABLE = ROOT / "results"
LLM_FEATURES = ROOT / "artifacts" / "scores" / "llm_item_scores.parquet"

DATASETS = ["xes3g5m", "junyi"]
MODELS = ["gpt-4o-mini", "gpt-5.4"]
PRIMARY_THRESHOLD = 20
CV_FOLDS = 5
CV_SEED = 2024
ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0]
SURFACE_NUM = [
    "char_length",
    "token_length",
    "math_symbol_count",
    "equation_count",
    "answer_option_count",
    "concept_count",
    "log_train_exposure",
]
SURFACE_CAT = ["has_image_dependency", "item_format", "mathematical_domain", "educational_level"]


def encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_model(num_cols: list[str], cat_cols: list[str]) -> Pipeline:
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", encoder()),
    ])
    pre = ColumnTransformer([
        ("num", numeric, num_cols),
        ("cat", categorical, cat_cols),
    ], remainder="drop")
    ridge = RidgeCV(alphas=ALPHAS, cv=3)
    return Pipeline([("preprocess", pre), ("ridge", ridge)])


def analysis_frame(dataset: str) -> pd.DataFrame:
    ref = pd.read_csv(TABLE / "AUTHENTIC_DIFFICULTY_REFERENCES_V2_ORIENTATION_CORRECTED.csv")
    surface = pd.read_csv(TABLE / "AUTHENTIC_ITEM_SURFACE_FEATURES.csv")
    llm = pd.read_parquet(LLM_FEATURES, columns=["dataset", "item_id_hash", "model_identifier", "scalar_difficulty"])
    held = ref[
        (ref["dataset"] == dataset)
        & (ref["reference_scope"] == "held_out_test")
        & (ref["heldout_response_count"] >= PRIMARY_THRESHOLD)
    ][["dataset", "item_id_hash", "smoothed_error_beta_1_1"]]
    df = held.merge(surface[surface["dataset"] == dataset], on=["dataset", "item_id_hash"], how="left")
    for model, col in [("gpt-4o-mini", "gpt4o_scalar"), ("gpt-5.4", "gpt54_scalar")]:
        sub = llm[(llm["dataset"] == dataset) & (llm["model_identifier"] == model)][["item_id_hash", "scalar_difficulty"]]
        df = df.merge(sub.rename(columns={"scalar_difficulty": col}), on="item_id_hash", how="inner")
    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def oof_for_spec(df: pd.DataFrame, num_cols: list[str], cat_cols: list[str], y: np.ndarray) -> tuple[np.ndarray, list[float]]:
    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=CV_SEED)
    preds = np.zeros(len(df))
    alphas = []
    for train_idx, test_idx in kf.split(df):
        model = make_model(num_cols, cat_cols)
        model.fit(df.iloc[train_idx], y[train_idx])
        preds[test_idx] = model.predict(df.iloc[test_idx])
        alphas.append(float(model.named_steps["ridge"].alpha_))
    return preds, alphas


def main() -> int:
    rows = []
    specs = {
        "Surface features only": [],
        "Surface + GPT-4o-mini score": ["gpt4o_scalar"],
        "Surface + GPT-5.4 score": ["gpt54_scalar"],
        "Surface + both LLM scores": ["gpt4o_scalar", "gpt54_scalar"],
    }
    for ds in DATASETS:
        df = analysis_frame(ds)
        surface_num = [c for c in SURFACE_NUM if c in df.columns]
        surface_cat = [c for c in SURFACE_CAT if c in df.columns and df[c].nunique(dropna=True) > 1]
        y = df["smoothed_error_beta_1_1"].to_numpy()
        baseline_r2 = None
        for name, extras in specs.items():
            num_cols = surface_num + extras
            preds, alphas = oof_for_spec(df, num_cols, surface_cat, y)
            ss_res = float(np.sum((y - preds) ** 2))
            ss_tot = float(np.sum((y - y.mean()) ** 2))
            r2 = 1 - ss_res / ss_tot
            if baseline_r2 is None:
                baseline_r2 = r2
            rows.append({
                "dataset": ds,
                "model_spec": name,
                "n_items": int(len(df)),
                "oof_r2": float(r2),
                "delta_r2_vs_surface": float(r2 - baseline_r2),
                "oof_rmse": float(np.sqrt(mean_squared_error(y, preds))),
                "oof_mae": float(mean_absolute_error(y, preds)),
                "selected_alpha_median": float(np.median(alphas)),
                "selected_alpha_values": ";".join(str(a) for a in alphas),
                "preprocessing": "fold-internal imputation, scaling, one-hot encoding; RidgeCV alpha selection on training fold",
            })
    out = pd.DataFrame(rows)
    out.to_csv(TABLE / "INCREMENTAL_ROBUSTNESS.csv", index=False)
    manifest = {
        "status": "INCREMENTAL_ROBUSTNESS_READY",
        "rows": len(out),
        "alphas": ALPHAS,
        "cv_folds": CV_FOLDS,
        "cv_seed": CV_SEED,
        "output": "tables/INCREMENTAL_ROBUSTNESS.csv",
    }
    (ROOT / "data_manifests" / "incremental_robustness_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
