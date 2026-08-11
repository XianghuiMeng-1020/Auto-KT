"""Shared utilities for Phase F measurement validity."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "measurement_validity_config.json"
PROCESSED_ROOT = ROOT / "data_processed"
LLM_FEATURES = ROOT / "artifacts" / "scores" / "llm_item_scores.parquet"
TABLE_DIR = ROOT / "results"
REPORT_DIR = ROOT / "reports" / "measurement"
FIGURE_DIR = ROOT / "figures" / "measurement"
DATASETS = ("xes3g5m", "junyi")
MATH_SYMBOL_RE = re.compile(r"[+\-*/=<>≤≥^√∑∫πθαβγΔ×÷]|\\frac|\\sqrt")
NUMBER_RE = re.compile(r"\d+\.?\d*")
EQUATION_RE = re.compile(r"(=|\\frac|\\sqrt|\$[^$]+\$)")


def load_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_scoreable_items(dataset: str) -> pd.DataFrame:
    items = pd.read_parquet(PROCESSED_ROOT / dataset / "items.parquet")
    return items[items["eligible_for_llm_scoring"]].copy()


def load_interactions(dataset: str, columns: list[str] | None = None) -> pd.DataFrame:
    path = PROCESSED_ROOT / dataset / "interactions.parquet"
    if columns:
        return pd.read_parquet(path, columns=columns)
    return pd.read_parquet(path)


def load_llm_features() -> pd.DataFrame:
    return pd.read_parquet(LLM_FEATURES)


def beta_smooth(successes: pd.Series, n: pd.Series, alpha: float, beta: float) -> pd.Series:
    return (successes + alpha) / (n + alpha + beta)


def empirical_bayes_prior(successes: pd.Series, n: pd.Series) -> tuple[float, float]:
    p = (successes / n).clip(1e-6, 1 - 1e-6)
    var = p.var()
    mean = p.mean()
    if var <= 0 or not np.isfinite(var):
        return 1.0, 1.0
    common = mean * (1 - mean) / var - 1
    if common <= 0:
        return 1.0, 1.0
    alpha = mean * common
    beta = (1 - mean) * common
    return float(max(alpha, 0.1)), float(max(beta, 0.1))


def compute_error_rates(
    interactions: pd.DataFrame,
    split: str,
    first_attempt_only: bool = False,
) -> pd.DataFrame:
    sub = interactions[interactions["split_assignment"] == split].copy()
    if first_attempt_only and "first_attempt" in sub.columns:
        sub = sub[sub["first_attempt"].fillna(True)]
    sub["incorrect"] = 1 - sub["correct"].astype(int)
    agg = sub.groupby("item_id_hash", as_index=False).agg(
        n_responses=("correct", "size"),
        n_correct=("correct", "sum"),
        n_students=("student_id_hash", "nunique"),
    )
    agg["n_incorrect"] = agg["n_responses"] - agg["n_correct"]
    agg["raw_error_rate"] = agg["n_incorrect"] / agg["n_responses"]
    return agg


def bootstrap_ci(
    x: np.ndarray,
    y: np.ndarray,
    stat_fn,
    n_boot: int = 500,
    seed: int = 2024,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3:
        return float("nan"), float("nan"), float("nan")
    point = stat_fn(x, y)
    boots = []
    n = len(x)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        boots.append(stat_fn(x[idx], y[idx]))
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(point), float(lo), float(hi)


def holm_correction(p_values: list[float]) -> list[float]:
    m = len(p_values)
    if m == 0:
        return []
    order = np.argsort(p_values)
    adjusted = [1.0] * m
    prev = 0.0
    for rank, idx in enumerate(order):
        adj = min(1.0, p_values[idx] * (m - rank))
        adj = max(adj, prev)
        adjusted[idx] = adj
        prev = adj
    return adjusted


def rank_independent(a: np.ndarray, b: np.ndarray, seed: int = 2024) -> float:
    if len(a) < 3:
        return float("nan")
    rho = stats.spearmanr(a, b).correlation
    return float(abs(rho)) if np.isfinite(rho) else 1.0


def extract_surface_features(items: pd.DataFrame, exposure: pd.DataFrame | None = None) -> pd.DataFrame:
    rows = []
    for _, row in items.iterrows():
        text = str(row.get("item_text_clean") or "")
        tokens = text.split()
        concepts = row.get("concept_ids")
        if isinstance(concepts, (list, tuple)):
            concept_count = len(concepts)
        elif concepts is None or (isinstance(concepts, float) and np.isnan(concepts)):
            concept_count = 0
        else:
            concept_count = len(str(concepts).split(",")) if str(concepts) else 0
        opts = row.get("answer_options")
        if isinstance(opts, (list, tuple)):
            option_count = len(opts)
        elif opts is None or (isinstance(opts, float) and np.isnan(opts)):
            option_count = 0
        else:
            option_count = len(str(opts).split("|")) if str(opts) else 0
        rows.append({
            "dataset": row["dataset"],
            "item_id_hash": row["item_id_hash"],
            "char_length": len(text),
            "token_length": len(tokens),
            "sentence_count": max(1, len(re.split(r"[.!?。！？]", text))),
            "number_count": len(NUMBER_RE.findall(text)),
            "math_symbol_count": len(MATH_SYMBOL_RE.findall(text)),
            "equation_count": len(EQUATION_RE.findall(text)),
            "answer_option_count": option_count,
            "concept_count": concept_count,
            "has_image_dependency": bool(row.get("has_image_dependency")),
            "item_format": row.get("item_format"),
            "mathematical_domain": row.get("mathematical_domain"),
            "educational_level": row.get("educational_level"),
            "language": row.get("language"),
            "item_content_type": row.get("item_content_type"),
        })
    df = pd.DataFrame(rows)
    if exposure is not None:
        df = df.merge(exposure, on=["dataset", "item_id_hash"], how="left")
    return df


def compute_exposure(interactions: pd.DataFrame, dataset: str) -> pd.DataFrame:
    rows = []
    for split in ("train", "test"):
        sub = interactions[interactions["split_assignment"] == split]
        agg = sub.groupby("item_id_hash").size().reset_index(name=f"{split}_responses")
        agg["dataset"] = dataset
        rows.append(agg)
    out = rows[0].merge(rows[1], on=["dataset", "item_id_hash"], how="outer").fillna(0)
    out["log_train_exposure"] = np.log1p(out["train_responses"])
    out["log_test_exposure"] = np.log1p(out["test_responses"])
    return out


def meta_analyze_spearman(effects: list[float], ses: list[float]) -> dict[str, float]:
    effects = np.asarray(effects, dtype=float)
    ses = np.asarray(ses, dtype=float)
    mask = np.isfinite(effects) & np.isfinite(ses) & (ses > 0)
    if mask.sum() < 2:
        return {"pooled_r": float(np.nanmean(effects)), "Q": float("nan"), "I2": float("nan"), "tau2": float("nan")}
    w = 1 / ses[mask] ** 2
    pooled = float(np.sum(w * effects[mask]) / np.sum(w))
    q = float(np.sum(w * (effects[mask] - pooled) ** 2))
    df = int(mask.sum() - 1)
    c = float(np.sum(w) - np.sum(w**2) / np.sum(w))
    tau2 = max(0.0, (q - df) / c) if c > 0 else 0.0
    i2 = max(0.0, (q - df) / q * 100) if q > 0 else 0.0
    return {"pooled_r": pooled, "Q": q, "I2": i2, "tau2": tau2, "df": df}
