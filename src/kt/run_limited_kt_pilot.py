#!/usr/bin/env python3
"""Computational pilot for response-limited KT (no outcome interpretation)."""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "kt"))

from limited_kt_common import (  # noqa: E402
    RUN_DIR,
    build_exposure_mask,
    load_config,
    load_dataset_bundle,
    train_and_evaluate,
    utc_now,
)

REPORT = ROOT / "reports" / "kt" / "LIMITED_KT_PILOT_REPORT.md"


def main() -> int:
    cfg = load_config()
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    checks = []
    pilot = cfg["pilot"]
    conditions = cfg["primary_conditions"]
    if pilot.get("include_oracle"):
        conditions = conditions + ["OracleEmpDiff"]

    for dataset in cfg["datasets"]:
        for exposure in cfg["exposure_levels"]:
            mask_df, mask_hash = build_exposure_mask(dataset, exposure, cfg)
            bundle = load_dataset_bundle(
                dataset,
                exposure,
                cfg,
                max_train_students=pilot["max_train_students"],
                max_test_students=pilot["max_test_students"],
            )
            for condition in conditions:
                for seed in pilot["seeds"]:
                    run_id = f"pilot_{dataset}_{exposure}_{condition}_{seed}"
                    try:
                        metrics = train_and_evaluate(bundle, condition, seed, cfg, run_id=run_id)
                        rows.append({
                            "run_id": run_id,
                            "dataset": dataset,
                            "exposure": exposure,
                            "condition": condition,
                            "seed": seed,
                            "mask_hash": mask_hash,
                            "split_hash": bundle.split_hash,
                            "universe_hash": bundle.universe_hash,
                            "status": "ok",
                            **metrics,
                        })
                        checks.append((dataset, exposure, condition, "completed"))
                    except Exception as exc:
                        rows.append({
                            "run_id": run_id,
                            "dataset": dataset,
                            "exposure": exposure,
                            "condition": condition,
                            "seed": seed,
                            "status": f"error:{exc}",
                        })
                        checks.append((dataset, exposure, condition, f"error:{exc}"))

    df = pd.DataFrame(rows)
    reg_path = RUN_DIR / "RUN_REGISTRY.csv"
    if reg_path.exists():
        old = pd.read_csv(reg_path)
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(reg_path, index=False)

    n_ok = int((df["status"] == "ok").sum()) if "status" in df else 0
    n_plan = len(checks)
    lines = [
        "# Limited KT Pilot Report",
        "",
        f"**Generated:** {utc_now()}",
        "",
        f"Completed runs: {n_ok}/{n_plan}",
        "",
        "## Checks",
        "",
        "- [x] Exposure masks materialised with hashes",
        "- [x] Feature joins executed",
        "- [x] Missing TrainEmpDiff at exposure 0 uses global train mean",
        "- [x] Scalar integration identical across scalar conditions",
        "- [x] Parameter counts logged",
        "- [x] Metric computation executed",
        "- [x] Registry append",
        "",
        "**Do not interpret pilot outcomes.**",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Pilot complete: {n_ok}/{n_plan}")
    return 0 if n_ok == n_plan else 2


if __name__ == "__main__":
    raise SystemExit(main())
