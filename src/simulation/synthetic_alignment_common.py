"""Shared helpers for synthetic alignment ladder."""

from __future__ import annotations

import hashlib
import random
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]


def load_gsm8k_items(n_items: int) -> pd.DataFrame:
    """Load local GSM8K items and join the frozen legacy difficulty scores.

    Raw GSM8K item text is not redistributed with this repository. This
    function expects a local copy of the GSM8K training split (see
    ``data/README.md``) at ``data_raw/gsm8k/train.csv`` with a ``question``
    column, in the same row order as the original dataset. The shipped
    artifact under ``artifacts/scores/gsm8k_legacy_difficulty_scores.csv``
    provides only a content hash and a difficulty score per item; the hash
    is used to verify the local copy matches the one used to generate the
    frozen scores before joining.
    """
    raw_path = ROOT / "data_raw" / "gsm8k" / "train.csv"
    if not raw_path.exists():
        raise FileNotFoundError(
            "GSM8K raw data not found at data_raw/gsm8k/train.csv. "
            "See data/README.md for acquisition instructions."
        )
    raw = pd.read_csv(raw_path).head(n_items).reset_index(drop=True)
    if "question" in raw.columns and "instruction" not in raw.columns:
        raw = raw.rename(columns={"question": "instruction"})

    scores_path = ROOT / "artifacts" / "scores" / "gsm8k_legacy_difficulty_scores.csv"
    scores = pd.read_csv(scores_path).head(n_items).reset_index(drop=True)

    computed_hash = raw["instruction"].astype(str).apply(
        lambda s: hashlib.sha256(s.encode()).hexdigest()
    )
    mismatch = int((computed_hash != scores["instruction_hash"]).sum())
    if mismatch:
        raise ValueError(
            f"{mismatch} GSM8K items do not match the frozen score file by content "
            "hash. Check that the local dataset order/version matches the "
            "original download used to generate the frozen scores."
        )

    df = raw.copy()
    df["item_id"] = scores["item_id"].values
    df["difficulty"] = scores["difficulty"].fillna(0.5).values
    return df


def normalize_z(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return (x - x.mean()) / (x.std() + 1e-8)


def rank_agreement(a: np.ndarray, b: np.ndarray) -> float:
    return float(stats.spearmanr(a, b).correlation)


def build_surface_confounded_difficulty(df: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    text = df["instruction"].astype(str)
    char_len = text.str.len().values.astype(float)
    sym = text.str.count(r"[+\-*/=]").values.astype(float)
    eq = text.str.count(r"=").values.astype(float)
    noise = rng.normal(size=len(df))
    combo = 0.4 * normalize_z(char_len) + 0.3 * normalize_z(sym) + 0.2 * normalize_z(eq) + 0.1 * normalize_z(noise)
    return normalize_z(combo)


def generate_responses(items: pd.DataFrame, d_gen: np.ndarray, seed: int, syn_cfg: dict) -> pd.DataFrame:
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    abilities = np.clip(np_rng.normal(0.5, 0.25, syn_cfg["num_students"]), 0, 1)
    scale = syn_cfg.get("irt_scale", 10.0)
    # map z-scored d_gen to [0,1] difficulty scale for sigmoid
    beta = (d_gen - d_gen.min()) / (d_gen.max() - d_gen.min() + 1e-8)
    logs = []
    item_ids = items["item_id"].values
    for uid, theta in enumerate(abilities):
        k = rng.randint(syn_cfg["responses_per_student_min"], syn_cfg["responses_per_student_max"])
        chosen = rng.sample(range(len(items)), k)
        for idx in chosen:
            diff = beta[idx]
            prob = 1 / (1 + np.exp(-scale * (theta - diff)))
            logs.append({
                "user_id": uid,
                "item_id": item_ids[idx],
                "correct": 1 if rng.random() < prob else 0,
            })
    return pd.DataFrame(logs)


def review_budget_metrics(d_llm: np.ndarray, error_rates: np.ndarray, k: int = 20) -> dict:
    n = len(d_llm)
    k = min(k, n // 5)
    if k < 1:
        return {"review_precision": float("nan"), "review_recall": float("nan"), "review_ndcg": float("nan")}
    hardest_true = set(np.argsort(error_rates)[-k:])
    hardest_pred = set(np.argsort(d_llm)[-k:])
    prec = len(hardest_true & hardest_pred) / k
    rec = len(hardest_true & hardest_pred) / k
    # simple NDCG proxy
    rel_true = np.argsort(np.argsort(error_rates))
    rel_pred_order = np.argsort(d_llm)[::-1][:k]
    dcg = sum((2 ** (error_rates[i] > np.quantile(error_rates, 0.8)) - 1) / np.log2(r + 2) for r, i in enumerate(rel_pred_order))
    idcg = sum(1 / np.log2(r + 2) for r in range(k))
    ndcg = dcg / idcg if idcg > 0 else 0
    return {"review_precision": float(prec), "review_recall": float(rec), "review_ndcg": float(ndcg)}
