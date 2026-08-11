#!/usr/bin/env python3
"""Full limited KT utility evaluation."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "kt"))

from limited_kt_common import (  # noqa: E402
    CONFIG_PATH,
    RUN_DIR,
    _load_interactions_cached,
    build_exposure_mask,
    git_commit,
    load_config,
    load_dataset_bundle,
    sha256_file,
    train_and_evaluate,
    utc_now,
)


def _append_registry(row: dict) -> None:
    reg_path = RUN_DIR / "RUN_REGISTRY.csv"
    new_df = pd.DataFrame([row])
    if reg_path.exists():
        old = pd.read_csv(reg_path)
        old = old[old["run_id"] != row["run_id"]]
        new_df = pd.concat([old, new_df], ignore_index=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    new_df.to_csv(reg_path, index=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-oracle", action="store_true")
    parser.add_argument("--dataset", choices=["xes3g5m", "junyi"], default=None)
    args = parser.parse_args()

    cfg = load_config()
    conditions = list(cfg["primary_conditions"])
    if args.include_oracle:
        conditions.append("OracleEmpDiff")

    datasets = [args.dataset] if args.dataset else cfg["datasets"]
    rows = []
    for dataset in datasets:
        _load_interactions_cached(dataset)  # preload once
        for exposure in cfg["exposure_levels"]:
            mask_df, mask_hash = build_exposure_mask(dataset, exposure, cfg)
            bundle = load_dataset_bundle(dataset, exposure, cfg)
            for condition in conditions:
                for seed in cfg["seeds"]:
                    run_id = f"{dataset}_{exposure}_{condition}_{seed}"
                    reg_path = RUN_DIR / "RUN_REGISTRY.csv"
                    if reg_path.exists():
                        prior = pd.read_csv(reg_path)
                        if ((prior["run_id"] == run_id) & (prior["status"] == "ok")).any():
                            print(f"skip {run_id}", flush=True)
                            continue
                    t0 = time.time()
                    try:
                        metrics = train_and_evaluate(bundle, condition, seed, cfg, run_id=run_id)
                        row = {
                            "run_id": run_id,
                            "dataset": dataset,
                            "exposure": exposure,
                            "condition": condition,
                            "seed": seed,
                            "mask_hash": mask_hash,
                            "split_hash": bundle.split_hash,
                            "universe_hash": bundle.universe_hash,
                            "config_hash": sha256_file(CONFIG_PATH),
                            "code_commit": git_commit(),
                            "start_time_utc": utc_now(),
                            "wall_time_s": time.time() - t0,
                            "status": "ok",
                            **metrics,
                        }
                    except Exception as exc:
                        row = {
                            "run_id": run_id,
                            "dataset": dataset,
                            "exposure": exposure,
                            "condition": condition,
                            "seed": seed,
                            "status": f"error:{exc}",
                            "wall_time_s": time.time() - t0,
                        }
                    _append_registry(row)
                    rows.append(row)
                    print(row.get("status"), run_id, row.get("auc"), flush=True)

    print(f"Recorded {len(rows)} runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
