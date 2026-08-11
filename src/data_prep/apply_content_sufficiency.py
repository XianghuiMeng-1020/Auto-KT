#!/usr/bin/env python3
"""Apply Amendment 009 scoreability to items and regenerate prompt-safe exports."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "data_prep"))

from content_sufficiency import (  # noqa: E402
    CONTENT_SUFFICIENCY_RULE_VERSION,
    amendment_009_hash,
    apply_classification,
    classification_audit_hash,
)
from unified_schema_common import (  # noqa: E402
    PROCESSED_ROOT,
    make_llm_prompt_items,
    sha256_file,
    write_parquet,
)

DATASETS = ("xes3g5m", "junyi")


def main() -> int:
    manifest_records = {}
    for dataset in DATASETS:
        items_path = PROCESSED_ROOT / dataset / "items.parquet"
        items = pd.read_parquet(items_path)
        classified = apply_classification(items, dataset)
        write_parquet(classified, items_path)

        scoreable = classified[classified["eligible_for_llm_scoring"]].copy()
        llm_export = make_llm_prompt_items(scoreable)
        llm_path = PROCESSED_ROOT / dataset / "llm_prompt_items.parquet"
        write_parquet(llm_export, llm_path)

        manifest_records[dataset] = {
            "kt_universe_items": len(classified),
            "llm_scoreable_items": int(classified["eligible_for_llm_scoring"].sum()),
            "items_sha256": sha256_file(items_path),
            "llm_prompt_items_sha256": sha256_file(llm_path),
            "classification_hash": classification_audit_hash(classified),
        }
        print(
            f"{dataset}: KT={len(classified)} LLM-scoreable={manifest_records[dataset]['llm_scoreable_items']}"
        )

    meta = {
        "content_sufficiency_rule_version": CONTENT_SUFFICIENCY_RULE_VERSION,
        "amendment_009_hash": amendment_009_hash(),
        "datasets": manifest_records,
    }
    out = ROOT / "data_manifests" / "content_sufficiency_manifest.json"
    out.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
