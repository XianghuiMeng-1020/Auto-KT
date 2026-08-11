#!/usr/bin/env python3
"""Build per-dataset/model feature parquets and unified feature table."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "llm_scoring"))

from full_llm_common import (  # noqa: E402
    DATASETS,
    PARSED_DIR,
    FullScoringConfig,
    load_cache_index,
    response_to_feature_row,
    sha256_file,
)

MODEL_FILE = {
    "gpt-4o-mini": "gpt4o_mini",
    "gpt-5.4": "gpt5_4",
}


def main() -> int:
    cfg = FullScoringConfig.load()
    PARSED_DIR.mkdir(parents=True, exist_ok=True)
    cache = load_cache_index()
    rows = [response_to_feature_row(v) for v in cache.values() if v.get("parse_status") == "valid"]
    df = pd.DataFrame(rows)
    if df.empty:
        print("No valid records", file=sys.stderr)
        return 1

    # one primary score per item-model
    df = df.sort_values("request_timestamp_utc").drop_duplicates(
        subset=["dataset", "item_id_hash", "model_identifier"], keep="first"
    )

    hashes = {}
    for ds in DATASETS:
        for model in cfg.models:
            sub = df[(df["dataset"] == ds) & (df["model_identifier"] == model)]
            fname = f"{ds}_{MODEL_FILE[model]}.parquet"
            path = PARSED_DIR / fname
            sub.to_parquet(path, index=False)
            hashes[fname] = sha256_file(path)

    all_path = PARSED_DIR / "all_llm_item_features.parquet"
    df.to_parquet(all_path, index=False)
    hashes["all_llm_item_features.parquet"] = sha256_file(all_path)

    meta = {"table_hashes": hashes, "row_counts": {k: int(pd.read_parquet(PARSED_DIR / k).shape[0]) for k in hashes}}
    (PARSED_DIR / "feature_build_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Built {len(df)} feature rows across {len(hashes)} tables")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
