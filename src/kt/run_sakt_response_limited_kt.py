#!/usr/bin/env python3
"""Response-limited knowledge tracing with a SAKT backbone."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "kt"))

from unseen_item_kt_common import (  # noqa: E402
    CONFIG_PATH,
    SAKT_LIMITED_RUN_DIR,
    append_registry,
    git_commit,
    load_config,
    load_limited_bundle,
    run_limited_cell,
    sha256_file,
    utc_now,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["xes3g5m", "junyi"], default=None)
    parser.add_argument("--exposure", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--condition", default=None)
    args = parser.parse_args()

    cfg = load_config()
    datasets = [args.dataset] if args.dataset else cfg["datasets"]
    exposures = [args.exposure] if args.exposure is not None else cfg["limited_exposures"]
    # normalize exposure types
    norm_exp = []
    for e in exposures:
        if str(e) == "warm":
            norm_exp.append("warm")
        else:
            norm_exp.append(int(e))
    seeds = [args.seed] if args.seed is not None else cfg["seeds"]
    conditions = [args.condition] if args.condition else cfg["limited_conditions"]

    reg_path = SAKT_LIMITED_RUN_DIR / "RUN_REGISTRY.csv"
    backbone = "SAKT"

    for dataset in datasets:
        for exposure in norm_exp:
            bundle = load_limited_bundle(dataset, exposure, cfg)
            for condition in conditions:
                for seed in seeds:
                    run_id = f"sakt_limited_{dataset}_{exposure}_{condition}_{seed}"
                    if reg_path.exists():
                        prior = pd.read_csv(reg_path)
                        if ((prior["run_id"] == run_id) & (prior["status"] == "ok")).any():
                            print(f"skip {run_id}", flush=True)
                            continue
                    t0 = time.time()
                    try:
                        out = run_limited_cell(bundle, backbone, condition, seed, cfg)
                        row = {
                            "run_id": run_id,
                            "dataset": dataset,
                            "backbone": backbone,
                            "experiment_type": "response_limited",
                            "item_fold": "NA",
                            "response_limit": exposure,
                            "condition": condition,
                            "training_seed": seed,
                            "item_fold_seed": "NA",
                            "mask_dropout_seed": cfg["train"]["mask_seed"],
                            "data_split_hash": out["split_hash"],
                            "mask_hash": out["mask_hash"],
                            "score_file_hash": sha256_file(
                                ROOT / "artifacts/scores/llm_item_scores.parquet"
                            ),
                            "target_item_list_hash": "NA",
                            "code_sha": git_commit(),
                            "config_hash": sha256_file(CONFIG_PATH),
                            "best_epoch": out["best_epoch"],
                            "validation_log_loss": out["best_val_log_loss"],
                            "test_log_loss": out["log_loss"],
                            "auc": out["auc"],
                            "brier": out["brier"],
                            "ece": out["ece"],
                            "n_predictions": out["n_predictions"],
                            "n_parameters": out["n_parameters"],
                            "start_time_utc": utc_now(),
                            "wall_time_s": time.time() - t0,
                            "status": "ok",
                        }
                    except Exception as exc:
                        row = {
                            "run_id": run_id,
                            "dataset": dataset,
                            "backbone": backbone,
                            "experiment_type": "response_limited",
                            "response_limit": exposure,
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
