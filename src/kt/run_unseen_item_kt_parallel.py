#!/usr/bin/env python3
"""Parallel cold-start runner: one process per item fold (CPU)."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="junyi", choices=["xes3g5m", "junyi"])
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument("--backbone", default=None, choices=["GRU", "SAKT", None])
    args = parser.parse_args()
    folds = [int(x) for x in args.folds.split(",") if x != ""]
    log_dir = ROOT / "runs" / "unseen_item_kt" / "checks" / "parallel"
    log_dir.mkdir(parents=True, exist_ok=True)

    procs = []
    for fold in folds:
        cmd = [
            sys.executable,
            "-u",
            str(ROOT / "scripts" / "run_unseen_item_kt.py"),
            "--dataset",
            args.dataset,
            "--fold",
            str(fold),
        ]
        if args.backbone:
            cmd.extend(["--backbone", args.backbone])
        log = open(log_dir / f"{args.dataset}_fold{fold}.log", "w")
        p = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        procs.append((fold, p, log))
        print(f"launched fold {fold} pid={p.pid}", flush=True)

    rc = 0
    for fold, p, log in procs:
        code = p.wait()
        log.close()
        print(f"fold {fold} exit={code}", flush=True)
        if code != 0:
            rc = code
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
