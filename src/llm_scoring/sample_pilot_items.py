#!/usr/bin/env python3
"""Deterministic stratified pilot sample selection (offline strata only)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "llm_scoring"))

from llm_pilot_common import (  # noqa: E402
    PILOT_DIR,
    PROCESSED_ROOT,
    PilotConfig,
    ensure_dirs,
    git_commit,
    protocol_file_hash,
    sha256_file,
    sha256_text,
    utc_now,
)

DATASETS = ("xes3g5m", "junyi")


def train_response_counts(dataset: str) -> pd.Series:
    """Train-split response counts for stratification only (never sent to LLM)."""
    inter_path = PROCESSED_ROOT / dataset / "interactions.parquet"
    pf = pq.ParquetFile(inter_path)
    counts: dict[str, int] = {}
    for batch in pf.iter_batches(
        batch_size=500_000, columns=["item_id_hash", "split_assignment"]
    ):
        df = batch.to_pandas()
        df = df[df["split_assignment"] == "train"]
        for iid, n in df.groupby("item_id_hash").size().items():
            counts[iid] = counts.get(iid, 0) + int(n)
    return pd.Series(counts, name="train_response_count")


def assign_strata(df: pd.DataFrame, cfg: PilotConfig) -> pd.DataFrame:
    out = df.copy()
    out["text_len"] = out["item_text_clean"].astype(str).str.len()
    out["text_len_quartile"] = pd.qcut(
        out["text_len"].rank(method="first"), 4, labels=["q1", "q2", "q3", "q4"]
    )
    rc = out["train_response_count"].fillna(0)
    out["response_count_quartile"] = pd.qcut(
        rc.rank(method="first"), 4, labels=["rq1", "rq2", "rq3", "rq4"]
    )
    out["domain_bucket"] = out["mathematical_domain"].astype(str).str[:32]
    out["format_bucket"] = out["item_format"].astype(str)
    out["extraction_method"] = out.get("item_content_type", pd.Series(["unknown"] * len(out)))
    for col in ("has_image_dependency", "has_dynamic_template", "graphie_only_no_question_text"):
        if col not in out.columns:
            out[col] = False
    out["stratum"] = (
        out["response_count_quartile"].astype(str)
        + "|"
        + out["text_len_quartile"].astype(str)
        + "|"
        + out["format_bucket"].astype(str)
        + "|"
        + out["extraction_method"].astype(str)
        + "|img="
        + out["has_image_dependency"].astype(str)
        + "|dyn="
        + out["has_dynamic_template"].astype(str)
    )
    return out


def deterministic_pick(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    tmp = df.copy()
    tmp["_rank_key"] = tmp["item_id_hash"].map(
        lambda h: sha256_text(f"{seed}|{h}")
    )
    tmp = tmp.sort_values("_rank_key")
    strata = tmp["stratum"].unique()
    per = max(1, n // len(strata))
    picked = []
    for s in sorted(strata):
        grp = tmp[tmp["stratum"] == s]
        picked.append(grp.head(per))
    result = pd.concat(picked, ignore_index=True)
    if len(result) < n:
        remaining = tmp[~tmp["item_id_hash"].isin(result["item_id_hash"])]
        extra = remaining.head(n - len(result))
        result = pd.concat([result, extra], ignore_index=True)
    return result.head(n).drop(columns=["_rank_key"])


def build_dataset_sample(dataset: str, cfg: PilotConfig) -> pd.DataFrame:
    prompt = pd.read_parquet(PROCESSED_ROOT / dataset / "llm_prompt_items.parquet")
    items = pd.read_parquet(PROCESSED_ROOT / dataset / "items.parquet")
    meta_cols = [
        "item_id_hash",
        "has_image_dependency",
        "has_dynamic_template",
        "graphie_only_no_question_text",
        "html_source_file_hash",
        "slug_to_html_status",
    ]
    meta_cols = [c for c in meta_cols if c in items.columns]
    merged = prompt.merge(items[meta_cols], on="item_id_hash", how="left")
    merged["train_response_count"] = merged["item_id_hash"].map(train_response_counts(dataset))
    merged = assign_strata(merged, cfg)
    sample = deterministic_pick(merged, cfg.pilot_items_per_dataset, cfg.pilot_seed)
    sample["dataset"] = dataset
    sample["selection_seed"] = cfg.pilot_seed
    sample["selection_algorithm_version"] = cfg.selection_algorithm_version
    return sample


def write_manifest(samples: dict[str, pd.DataFrame], cfg: PilotConfig) -> None:
    manifest = {
        "generated_at_utc": utc_now(),
        "code_commit": git_commit(),
        "protocol_file_hash": protocol_file_hash(),
        "selection_seed": cfg.pilot_seed,
        "selection_algorithm_version": cfg.selection_algorithm_version,
        "datasets": {},
    }
    for ds, df in samples.items():
        path = PILOT_DIR / f"{ds}_pilot_items.parquet"
        manifest["datasets"][ds] = {
            "path": str(path.relative_to(ROOT)),
            "row_count": len(df),
            "sha256": sha256_file(path),
            "item_hashes": df["item_id_hash"].tolist(),
            "source_content_hashes": df["source_content_hash"].tolist(),
            "strata": df["stratum"].value_counts().to_dict(),
        }
    out = PILOT_DIR / "pilot_sample_manifest.json"
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> int:
    cfg = PilotConfig.load()
    ensure_dirs()
    samples = {}
    for ds in DATASETS:
        sample = build_dataset_sample(ds, cfg)
        path = PILOT_DIR / f"{ds}_pilot_items.parquet"
        sample.to_parquet(path, index=False)
        samples[ds] = sample
        print(f"{ds}: selected {len(sample)} pilot items")
    write_manifest(samples, cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
