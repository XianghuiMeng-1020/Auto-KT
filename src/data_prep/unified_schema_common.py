"""Shared utilities for unified schema construction (XES3G5M & Junyi)."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "unified_schema_config.json"
PROCESSED_ROOT = ROOT / "data_processed"

LLM_PROMPT_ALLOWLIST = frozenset({
    "dataset",
    "item_id_hash",
    "item_text_clean",
    "item_content_type",
    "language",
    "mathematical_domain",
    "educational_level",
    "item_format",
    "source_content_hash",
})

LLM_PROMPT_DENYLIST = frozenset({
    "correct_answer_separate",
    "correct",
    "correctness",
    "answer",
    "answer_options",
    "empirical_difficulty",
    "response_count",
    "exposure_count",
    "error_rate",
    "hint_used",
    "answer_viewed",
    "student_id",
    "student_id_hash",
    "split_assignment",
})


@dataclass(frozen=True)
class UnifiedSchemaConfig:
    split_seed: int
    train_frac: float
    val_frac: float
    test_frac: float
    min_student_interactions: int
    split_algorithm_version: str
    hash_salt: str
    frozen_eligible_item_counts: dict[str, int]
    tie_break_rule: str
    response_cooccurrence_min_count: int

    @classmethod
    def load(cls, path: Path | None = None) -> UnifiedSchemaConfig:
        p = path or CONFIG_PATH
        raw = json.loads(p.read_text(encoding="utf-8"))
        return cls(**raw)

    def config_hash(self) -> str:
        return sha256_text(json.dumps(json.loads(CONFIG_PATH.read_text()), sort_keys=True))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_id(dataset: str, kind: str, raw: str | int, salt: str) -> str:
    payload = f"{salt}|{dataset}|{kind}|{raw}"
    return sha256_text(payload)


def git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def protocol_hash() -> str:
    paths = sorted(ROOT.glob("protocol/*.md")) + sorted(
        ROOT.glob("protocol/amendments/AMENDMENT_008*.md")
    )
    h = hashlib.sha256()
    for p in paths:
        if p.is_file():
            h.update(p.name.encode())
            h.update(p.read_bytes())
    return h.hexdigest()


def split_students_deterministic(
    student_ids: list | set,
    cfg: UnifiedSchemaConfig,
) -> pd.DataFrame:
    ids = sorted(set(str(s) for s in student_ids))
    rng = np.random.default_rng(cfg.split_seed)
    order = np.array(ids)
    rng.shuffle(order)
    n = len(order)
    n_train = int(cfg.train_frac * n)
    n_val = int(cfg.val_frac * n)
    splits = (
        ["train"] * n_train
        + ["val"] * n_val
        + ["test"] * (n - n_train - n_val)
    )
    return pd.DataFrame({
        "student_id_raw": order,
        "split_assignment": splits,
    })


def build_splits_parquet(
    dataset: str,
    student_ids: list | set,
    cfg: UnifiedSchemaConfig,
) -> pd.DataFrame:
    raw = split_students_deterministic(student_ids, cfg)
    return pd.DataFrame({
        "dataset": dataset,
        "student_id_hash": [
            hash_id(dataset, "student", s, cfg.hash_salt) for s in raw["student_id_raw"]
        ],
        "split_assignment": raw["split_assignment"],
        "split_seed": cfg.split_seed,
        "split_algorithm_version": cfg.split_algorithm_version,
    })


def assign_splits_to_interactions(
    interactions: pd.DataFrame,
    splits: pd.DataFrame,
    student_raw_col: str = "student_id_raw",
) -> pd.DataFrame:
    mapping = dict(zip(splits["student_id_hash"], splits["split_assignment"]))
    out = interactions.copy()
    out["split_assignment"] = out["student_id_hash"].map(mapping)
    if out["split_assignment"].isna().any():
        raise ValueError("Some interactions lack split assignment")
    return out


def make_llm_prompt_items(items: pd.DataFrame) -> pd.DataFrame:
    export_cols = sorted(c for c in LLM_PROMPT_ALLOWLIST if c in items.columns)
    missing = LLM_PROMPT_ALLOWLIST - set(export_cols)
    if missing:
        raise ValueError(f"LLM prompt export missing allowlist columns: {missing}")
    out = items[export_cols].copy()
    leaked = set(out.columns) & LLM_PROMPT_DENYLIST
    if leaked:
        raise ValueError(f"Denied fields in LLM export: {leaked}")
    return out


def build_response_cooccurrence_edges(
    interactions: pd.DataFrame | Path,
    dataset: str,
    cfg: UnifiedSchemaConfig,
    train_only: bool = True,
) -> pd.DataFrame:
    pair_counts: dict[tuple[str, str], int] = {}

    if isinstance(interactions, Path):
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(interactions)
        cols = ["student_id_hash", "primary_concept_id", "sequence_index", "split_assignment"]
        for batch in pf.iter_batches(batch_size=500_000, columns=cols):
            df = batch.to_pandas()
            if train_only:
                df = df[df["split_assignment"] == "train"]
            df = df.sort_values(["student_id_hash", "sequence_index"])
            for _, grp in df.groupby("student_id_hash"):
                concepts = [
                    c for c in grp["primary_concept_id"].tolist()
                    if pd.notna(c) and str(c).strip()
                ]
                for i in range(len(concepts) - 1):
                    a, b = concepts[i], concepts[i + 1]
                    if a != b:
                        pair_counts[(a, b)] = pair_counts.get((a, b), 0) + 1
    else:
        df = interactions.copy()
        if train_only:
            df = df[df["split_assignment"] == "train"]
        df = df.sort_values(["student_id_hash", "sequence_index"])
        for _, grp in df.groupby("student_id_hash"):
            concepts = [
                c for c in grp["primary_concept_id"].tolist()
                if pd.notna(c) and str(c).strip()
            ]
            for i in range(len(concepts) - 1):
                a, b = concepts[i], concepts[i + 1]
                if a != b:
                    pair_counts[(a, b)] = pair_counts.get((a, b), 0) + 1

    rows = []
    for (src, tgt), cnt in sorted(pair_counts.items()):
        if cnt < cfg.response_cooccurrence_min_count:
            continue
        rows.append({
            "dataset": dataset,
            "source_concept_id_hash": hash_id(dataset, "concept", src, cfg.hash_salt),
            "target_concept_id_hash": hash_id(dataset, "concept", tgt, cfg.hash_salt),
            "edge_type": "sequential_cooccurrence",
            "edge_weight": float(cnt),
            "edge_source": "training_response_cooccurrence",
            "permitted_split": "train",
            "source_hash": sha256_text(f"{src}|{tgt}|{cnt}|{dataset}"),
        })
    return pd.DataFrame(rows)


def build_hierarchy_edges_from_path(
    path_str: str,
    dataset: str,
    cfg: UnifiedSchemaConfig,
    separator: str = "----",
) -> list[dict]:
    parts = [p.strip() for p in path_str.split(separator) if p.strip()]
    edges = []
    for i in range(len(parts) - 1):
        src, tgt = parts[i], parts[i + 1]
        edges.append({
            "dataset": dataset,
            "source_concept_id_hash": hash_id(dataset, "concept", src, cfg.hash_salt),
            "target_concept_id_hash": hash_id(dataset, "concept", tgt, cfg.hash_salt),
            "edge_type": "prerequisite_hierarchy",
            "edge_weight": 1.0,
            "edge_source": "official_metadata",
            "permitted_split": "all",
            "source_hash": sha256_text(f"meta|{src}|{tgt}|{dataset}"),
        })
    return edges


def verify_eligible_count(dataset: str, actual: int, cfg: UnifiedSchemaConfig) -> str:
    expected = cfg.frozen_eligible_item_counts[dataset]
    if actual == expected:
        return "SCHEMA_BUILD_PASS"
    return "SCHEMA_BUILD_COUNT_MISMATCH"


def write_parquet(df: pd.DataFrame, path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return file_record(path)


def file_record(path: Path, source_hashes: dict | None = None) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.relative_to(ROOT)),
        "row_count": int(pd.read_parquet(path).shape[0]) if path.suffix == ".parquet" else None,
        "columns": list(pd.read_parquet(path).columns) if path.suffix == ".parquet" else [],
        "size_bytes": stat.st_size,
        "sha256": sha256_file(path),
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_file_hashes": source_hashes or {},
    }


def sequence_length_summary(interactions: pd.DataFrame) -> pd.DataFrame:
    lens = interactions.groupby("student_id_hash").size()
    return pd.DataFrame([{
        "min": int(lens.min()),
        "median": float(lens.median()),
        "mean": float(lens.mean()),
        "max": int(lens.max()),
        "n_students": int(len(lens)),
    }])
