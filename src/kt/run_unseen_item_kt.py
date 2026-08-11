#!/usr/bin/env python3
"""Genuine unseen-item cold-start KT (GRU + SAKT)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "kt"))

from unseen_item_kt_common import (  # noqa: E402
    COLDSTART_RUN_DIR,
    CONFIG_PATH,
    GATE_DIR,
    append_registry,
    build_item_folds,
    git_commit,
    load_coldstart_bundle,
    load_config,
    metrics_from_arrays,
    run_coldstart_cell,
    sha256_file,
    utc_now,
)


PRED_DIR = COLDSTART_RUN_DIR / "predictions"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["xes3g5m", "junyi"], default=None)
    parser.add_argument("--backbone", choices=["GRU", "SAKT"], default=None)
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--condition", default=None)
    args = parser.parse_args()

    cfg = load_config()
    datasets = [args.dataset] if args.dataset else cfg["datasets"]
    backbones = [args.backbone] if args.backbone else ["GRU", "SAKT"]
    folds = [args.fold] if args.fold is not None else list(range(cfg["n_item_folds"]))
    seeds = [args.seed] if args.seed is not None else cfg["seeds"]
    conditions = [args.condition] if args.condition else cfg["coldstart_conditions"]

    reg_path = COLDSTART_RUN_DIR / "RUN_REGISTRY.csv"
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    GATE_DIR.mkdir(parents=True, exist_ok=True)

    # Materialize folds once
    for dataset in datasets:
        build_item_folds(dataset, cfg)

    for dataset in datasets:
        for fold in folds:
            bundle = load_coldstart_bundle(dataset, fold, cfg)
            gate_path = GATE_DIR / f"coldstart_gate_{dataset}_fold{fold}.json"
            gate_path.write_text(json.dumps(bundle.gate_assertions, indent=2), encoding="utf-8")
            if not bundle.gate_assertions["zero_target_train_interactions"]:
                raise RuntimeError(f"Leakage: target train interactions remain for {dataset} fold {fold}")

            for backbone in backbones:
                for condition in conditions:
                    for seed in seeds:
                        run_id = f"coldstart_{dataset}_{backbone}_fold{fold}_{condition}_{seed}"
                        if reg_path.exists():
                            prior = pd.read_csv(reg_path)
                            if ((prior["run_id"] == run_id) & (prior["status"] == "ok")).any():
                                print(f"skip {run_id}", flush=True)
                                continue
                        t0 = time.time()
                        try:
                            out = run_coldstart_cell(bundle, backbone, condition, seed, cfg)
                            # Persist fold-level predictions for OOF aggregation
                            pred_path = PRED_DIR / f"{run_id}.npz"
                            np.savez_compressed(
                                pred_path,
                                primary_y=out["primary_y"],
                                primary_p=out["primary_p"],
                                secondary_y=out["secondary_y"],
                                secondary_p=out["secondary_p"],
                            )
                            pm = out["primary_metrics"]
                            sm = out["secondary_metrics"]
                            row = {
                                "run_id": run_id,
                                "dataset": dataset,
                                "backbone": backbone,
                                "experiment_type": "unseen_item_coldstart",
                                "item_fold": fold,
                                "response_limit": "NA",
                                "condition": condition,
                                "training_seed": seed,
                                "item_fold_seed": cfg["item_fold_seed"],
                                "mask_dropout_seed": cfg["item_fold_seed"],
                                "data_split_hash": bundle.split_hash,
                                "score_file_hash": sha256_file(
                                    ROOT / "artifacts/scores/llm_item_scores.parquet"
                                ),
                                "target_item_list_hash": bundle.target_list_hash,
                                "code_sha": git_commit(),
                                "config_hash": sha256_file(CONFIG_PATH),
                                "best_epoch": out["best_epoch"],
                                "validation_log_loss": out["best_val_log_loss"],
                                "test_log_loss": pm["log_loss"],
                                "auc": pm["auc"],
                                "brier": pm["brier"],
                                "ece": pm["ece"],
                                "n_predictions": pm["n_predictions"],
                                "secondary_log_loss": sm["log_loss"],
                                "secondary_auc": sm["auc"],
                                "secondary_brier": sm["brier"],
                                "secondary_ece": sm["ece"],
                                "secondary_n_predictions": sm["n_predictions"],
                                "n_parameters": out["n_parameters"],
                                "pred_path": str(pred_path.relative_to(ROOT)),
                                "start_time_utc": utc_now(),
                                "wall_time_s": time.time() - t0,
                                "status": "ok",
                                "gate_assertions": out["gate_assertions"],
                            }
                        except Exception as exc:
                            row = {
                                "run_id": run_id,
                                "dataset": dataset,
                                "backbone": backbone,
                                "experiment_type": "unseen_item_coldstart",
                                "item_fold": fold,
                                "condition": condition,
                                "training_seed": seed,
                                "status": f"error:{exc}",
                                "wall_time_s": time.time() - t0,
                                "start_time_utc": utc_now(),
                            }
                        append_registry(reg_path, row)
                        print(row.get("status"), run_id, row.get("test_log_loss"), flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
