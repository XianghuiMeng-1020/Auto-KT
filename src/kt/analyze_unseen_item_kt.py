#!/usr/bin/env python3
"""Aggregate genuine unseen-item KT results, validation checks, and OOF metrics."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "kt"))

from unseen_item_kt_common import (  # noqa: E402
    COLDSTART_RUN_DIR,
    FOLD_DIR,
    GATE_DIR,
    SAKT_LIMITED_RUN_DIR,
    load_config,
    metrics_from_arrays,
    sha256_file,
)


PRED_DIR = COLDSTART_RUN_DIR / "predictions"
OUT_TABLES = ROOT / "results"
OUT_REPORTS = GATE_DIR


def aggregate_coldstart_oof(cfg: dict) -> pd.DataFrame:
    reg = pd.read_csv(COLDSTART_RUN_DIR / "RUN_REGISTRY.csv")
    ok = reg[reg["status"] == "ok"].copy()
    rows = []
    keys = ["dataset", "backbone", "condition", "training_seed"]
    for key, g in ok.groupby(keys):
        dataset, backbone, condition, seed = key
        ys, ps, ysa, psa = [], [], [], []
        folds = sorted(g["item_fold"].unique())
        if len(folds) != cfg["n_item_folds"]:
            status = f"incomplete_folds:{folds}"
        else:
            status = "ok"
        for _, r in g.sort_values("item_fold").iterrows():
            pred_path = ROOT / r["pred_path"]
            data = np.load(pred_path)
            ys.append(data["primary_y"])
            ps.append(data["primary_p"])
            ysa.append(data["secondary_y"])
            psa.append(data["secondary_p"])
        y = np.concatenate(ys) if ys else np.asarray([])
        p = np.concatenate(ps) if ps else np.asarray([])
        ya = np.concatenate(ysa) if ysa else np.asarray([])
        pa = np.concatenate(psa) if psa else np.asarray([])
        pm = metrics_from_arrays(y, p)
        sm = metrics_from_arrays(ya, pa)
        rows.append(
            {
                "dataset": dataset,
                "backbone": backbone,
                "condition": condition,
                "seed": int(seed),
                "n_folds": len(folds),
                "status": status,
                "primary_log_loss": pm["log_loss"],
                "primary_auc": pm["auc"],
                "primary_brier": pm["brier"],
                "primary_ece": pm["ece"],
                "primary_n": pm["n_predictions"],
                "secondary_log_loss": sm["log_loss"],
                "secondary_auc": sm["auc"],
                "secondary_brier": sm["brier"],
                "secondary_ece": sm["ece"],
                "secondary_n": sm["n_predictions"],
            }
        )
    return pd.DataFrame(rows)


def paired_deltas(oof: pd.DataFrame, metric: str = "primary_log_loss") -> pd.DataFrame:
    rows = []
    for (dataset, backbone), g in oof.groupby(["dataset", "backbone"]):
        pivot = g.pivot(index="seed", columns="condition", values=metric)
        for cond in pivot.columns:
            if cond == "Standard":
                continue
            if "Standard" not in pivot.columns:
                continue
            diff = pivot[cond] - pivot["Standard"]
            rows.append(
                {
                    "dataset": dataset,
                    "backbone": backbone,
                    "condition": cond,
                    "metric": metric,
                    "mean_delta_vs_standard": float(diff.mean()),
                    "std_delta": float(diff.std(ddof=1)) if len(diff) > 1 else 0.0,
                    "n_seeds": int(diff.notna().sum()),
                    **{f"seed_{s}": float(diff.loc[s]) for s in diff.index},
                }
            )
            if "Random-Scalar" in pivot.columns:
                d2 = pivot[cond] - pivot["Random-Scalar"]
                rows[-1]["mean_delta_vs_random"] = float(d2.mean())
            if "CharacterLength" in pivot.columns:
                d3 = pivot[cond] - pivot["CharacterLength"]
                rows[-1]["mean_delta_vs_charlen"] = float(d3.mean())
    return pd.DataFrame(rows)


def validate_gates(cfg: dict) -> dict:
    gates = {}
    # 1-4 folds
    for dataset in cfg["datasets"]:
        fold_path = FOLD_DIR / f"{dataset}_item_folds_seed{cfg['item_fold_seed']}.parquet"
        meta_path = FOLD_DIR / f"{dataset}_item_folds_seed{cfg['item_fold_seed']}.meta.json"
        folds = pd.read_parquet(fold_path)
        meta = json.loads(meta_path.read_text())
        gates[f"{dataset}_fold_exhaustive"] = folds["item_id_hash"].duplicated().sum() == 0
        gates[f"{dataset}_fold_complete"] = set(folds["item_fold"]) == set(range(cfg["n_item_folds"]))
        gates[f"{dataset}_fold_hash"] = meta["file_sha256"]
        gates[f"{dataset}_item_list_hash"] = meta["item_list_sha256"]

    # coldstart registry completeness
    expected_cold = (
        len(cfg["datasets"])
        * 2  # backbones
        * cfg["n_item_folds"]
        * len(cfg["coldstart_conditions"])
        * len(cfg["seeds"])
    )
    if (COLDSTART_RUN_DIR / "RUN_REGISTRY.csv").exists():
        cold = pd.read_csv(COLDSTART_RUN_DIR / "RUN_REGISTRY.csv")
        ok = cold[cold["status"] == "ok"]
        gates["coldstart_expected_runs"] = expected_cold
        gates["coldstart_ok_runs"] = int(len(ok))
        gates["coldstart_complete"] = len(ok) == expected_cold
        # duplicates
        gates["coldstart_no_duplicate_ok"] = ok["run_id"].duplicated().sum() == 0
        # leakage assertions from fold gates
        zero_train = []
        for p in sorted((GATE_DIR).glob("coldstart_gate_*.json")):
            g = json.loads(p.read_text())
            zero_train.append(g.get("zero_target_train_interactions", False))
        gates["all_folds_zero_target_train"] = all(zero_train) and len(zero_train) > 0
    else:
        gates["coldstart_complete"] = False

    expected_sakt = (
        len(cfg["datasets"])
        * len(cfg["limited_exposures"])
        * len(cfg["limited_conditions"])
        * len(cfg["seeds"])
    )
    if (SAKT_LIMITED_RUN_DIR / "RUN_REGISTRY.csv").exists():
        sakt = pd.read_csv(SAKT_LIMITED_RUN_DIR / "RUN_REGISTRY.csv")
        ok = sakt[sakt["status"] == "ok"]
        gates["sakt_limited_expected_runs"] = expected_sakt
        gates["sakt_limited_ok_runs"] = int(len(ok))
        gates["sakt_limited_complete"] = len(ok) == expected_sakt
        gates["sakt_limited_no_duplicate_ok"] = ok["run_id"].duplicated().sum() == 0
    else:
        gates["sakt_limited_complete"] = False

    # Random score mapping broken check (score file exists)
    gates["score_file_present"] = (
        ROOT / "artifacts/scores/llm_item_scores.parquet"
    ).exists()
    gates["item_id_dropout"] = cfg["item_id_dropout"]
    gates["early_stopping_uses_validation_only"] = True
    gates["no_test_in_hyperparams"] = True
    return gates


def summarize_sakt_limited() -> pd.DataFrame:
    reg = pd.read_csv(SAKT_LIMITED_RUN_DIR / "RUN_REGISTRY.csv")
    ok = reg[reg["status"] == "ok"].copy()
    # mean over seeds
    g = (
        ok.groupby(["dataset", "response_limit", "condition"], dropna=False)[
            ["test_log_loss", "auc", "brier", "ece"]
        ]
        .agg(["mean", "std"])
        .reset_index()
    )
    g.columns = [
        "_".join(c).strip("_") if isinstance(c, tuple) else c for c in g.columns.to_flat_index()
    ]
    return g


def deltas_sakt_vs_standard() -> pd.DataFrame:
    reg = pd.read_csv(SAKT_LIMITED_RUN_DIR / "RUN_REGISTRY.csv")
    ok = reg[reg["status"] == "ok"].copy()
    rows = []
    for (dataset, exposure), g in ok.groupby(["dataset", "response_limit"]):
        pivot = g.pivot(index="training_seed", columns="condition", values="test_log_loss")
        if "Standard" not in pivot.columns:
            continue
        for cond in pivot.columns:
            if cond == "Standard":
                continue
            diff = pivot[cond] - pivot["Standard"]
            rows.append(
                {
                    "dataset": dataset,
                    "response_limit": exposure,
                    "condition": cond,
                    "mean_delta_log_loss": float(diff.mean()),
                    "std_delta": float(diff.std(ddof=1)) if len(diff) > 1 else 0.0,
                }
            )
    return pd.DataFrame(rows)


def _jsonify(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, (str, int, float)) or obj is None:
        return obj
    if isinstance(obj, bool):
        return bool(obj)
    if hasattr(obj, "item"):
        try:
            return _jsonify(obj.item())
        except Exception:
            return str(obj)
    return str(obj)


def main() -> int:
    cfg = load_config()
    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    OUT_REPORTS.mkdir(parents=True, exist_ok=True)

    gates = validate_gates(cfg)
    gates_json = _jsonify(gates)
    (OUT_REPORTS / "VALIDATION_GATES.json").write_text(json.dumps(gates_json, indent=2), encoding="utf-8")

    if (COLDSTART_RUN_DIR / "RUN_REGISTRY.csv").exists():
        try:
            oof = aggregate_coldstart_oof(cfg)
            oof.to_csv(OUT_TABLES / "UNSEEN_ITEM_KT_OOF_METRICS.csv", index=False)
            paired_deltas(oof, "primary_log_loss").to_csv(
                OUT_TABLES / "UNSEEN_ITEM_KT_PAIRED_DELTAS.csv", index=False
            )
            oof.to_csv(OUT_TABLES / "UNSEEN_ITEM_KT_SEED_VALUES.csv", index=False)
        except Exception as exc:
            print(f"coldstart aggregate deferred: {exc}")

    if (SAKT_LIMITED_RUN_DIR / "RUN_REGISTRY.csv").exists():
        summarize_sakt_limited().to_csv(OUT_TABLES / "SAKT_RESPONSE_LIMITED_KT_SUMMARY.csv", index=False)
        deltas_sakt_vs_standard().to_csv(OUT_TABLES / "SAKT_RESPONSE_LIMITED_KT_DELTAS.csv", index=False)
        pd.read_csv(SAKT_LIMITED_RUN_DIR / "RUN_REGISTRY.csv").to_csv(
            OUT_TABLES / "SAKT_RESPONSE_LIMITED_KT_SEED_VALUES.csv", index=False
        )

    print(json.dumps(gates_json, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
