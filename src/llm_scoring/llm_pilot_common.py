"""Shared utilities for Phase D1 LLM pilot (frozen prompt, cache, parse)."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "llm_pilot_config.json"
PROMPT_PROTOCOL_PATH = ROOT / "protocol" / "LLM_DIFFICULTY_PROMPT_FREEZE.md"
JOURNAL_ROOT = ROOT / "artifacts"
PILOT_DIR = JOURNAL_ROOT / "features" / "pilot"
LLM_PILOT_ROOT = JOURNAL_ROOT / "features" / "llm_pilot"
RAW_DIR = LLM_PILOT_ROOT / "raw"
PARSED_DIR = LLM_PILOT_ROOT / "parsed"
CACHE_DIR = LLM_PILOT_ROOT / "cache"
MANIFEST_DIR = LLM_PILOT_ROOT / "manifests"
PROCESSED_ROOT = ROOT / "data_processed"

FROZEN_SYSTEM_MESSAGE = (
    "You are a mathematics education expert. Estimate the difficulty of the "
    "following problem for a typical student at the appropriate grade level. "
    "Output a single number between 0.0 (very easy) and 1.0 (very hard). "
    "Output only the number, with no explanation."
)

FROZEN_USER_TEMPLATE = "Problem:\n{stem_text}"

# Outcome tables that pilot code must never import or join.
OUTCOME_TABLE_DENYLIST = frozenset({
    "interactions.parquet",
    "splits.parquet",
    "correctness",
    "error_rate",
    "empirical_difficulty",
    "rasch",
    "kt_results",
})

PROMPT_ALLOWLIST = frozenset({
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

PROMPT_DENYLIST = frozenset({
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
    "rasch_difficulty",
})


@dataclass(frozen=True)
class PilotConfig:
    pilot_seed: int
    selection_algorithm_version: str
    pilot_items_per_dataset: int
    preflight_items_per_dataset: int
    manual_review_per_dataset: int
    models: list[str]
    temperature_deterministic: float
    temperature_stability: float
    stability_replicates: int
    max_tokens: int
    timeout_seconds: int
    max_retries_transient: int
    max_retries_format: int
    parse_success_threshold: float
    pilot_budget_usd: float
    full_scoring_budget_usd: float
    schema_version: str
    frozen_eligible_counts: dict[str, int]
    cost_per_1k_input_tokens_usd: float
    cost_per_1k_output_tokens_usd: float
    avg_input_tokens_per_request: int
    avg_output_tokens_per_request: int

    @classmethod
    def load(cls, path: Path | None = None) -> PilotConfig:
        raw = json.loads((path or CONFIG_PATH).read_text(encoding="utf-8"))
        return cls(
            pilot_seed=raw["pilot_seed"],
            selection_algorithm_version=raw["selection_algorithm_version"],
            pilot_items_per_dataset=raw["pilot_items_per_dataset"],
            preflight_items_per_dataset=raw["preflight_items_per_dataset"],
            manual_review_per_dataset=raw["manual_review_per_dataset"],
            models=list(raw["models"]),
            temperature_deterministic=raw["temperature_deterministic"],
            temperature_stability=raw["temperature_stability"],
            stability_replicates=raw["stability_replicates"],
            max_tokens=raw["max_tokens"],
            timeout_seconds=raw["timeout_seconds"],
            max_retries_transient=raw["max_retries_transient"],
            max_retries_format=raw["max_retries_format"],
            parse_success_threshold=raw["parse_success_threshold"],
            pilot_budget_usd=raw["pilot_budget_usd"],
            full_scoring_budget_usd=raw["full_scoring_budget_usd"],
            schema_version=raw["schema_version"],
            frozen_eligible_counts=dict(raw["frozen_eligible_counts"]),
            cost_per_1k_input_tokens_usd=raw["cost_per_1k_input_tokens_usd"],
            cost_per_1k_output_tokens_usd=raw["cost_per_1k_output_tokens_usd"],
            avg_input_tokens_per_request=raw["avg_input_tokens_per_request"],
            avg_output_tokens_per_request=raw["avg_output_tokens_per_request"],
        )


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def protocol_prompt_hash() -> str:
    payload = FROZEN_SYSTEM_MESSAGE + "\n---\n" + FROZEN_USER_TEMPLATE
    return sha256_text(payload)


def protocol_file_hash() -> str:
    return sha256_file(PROMPT_PROTOCOL_PATH)


def git_branch() -> str:
    try:
        return subprocess.check_output(
            ["git", "branch", "--show-current"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def git_clean() -> bool:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        )
        return out.strip() == ""
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def render_messages(stem_text: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": FROZEN_SYSTEM_MESSAGE},
        {"role": "user", "content": FROZEN_USER_TEMPLATE.format(stem_text=stem_text)},
    ]


def prompt_hash_for_item(stem_text: str) -> str:
    msgs = render_messages(stem_text)
    payload = json.dumps(msgs, sort_keys=True, ensure_ascii=False)
    return sha256_text(payload)


def parse_difficulty(raw: str) -> float:
    """Frozen scalar parser from protocol/LLM_DIFFICULTY_PROMPT_FREEZE.md."""
    m = re.search(r"0?\.\d+|1\.0*|0|1", raw.strip())
    if m:
        v = float(m.group())
        return max(0.0, min(1.0, v))
    return float("nan")


def parse_response_record(raw: str) -> dict[str, Any]:
    """Map frozen scalar output to pilot schema (multi-dim fields N/A per protocol)."""
    scalar = parse_difficulty(raw)
    valid = pd.notna(scalar)
    return {
        "scalar_difficulty": scalar if valid else None,
        "conceptual_demand": None,
        "procedural_steps": None,
        "reading_demand": None,
        "prerequisite_depth": None,
        "distractor_complexity": None,
        "representational_complexity": None,
        "confidence": None,
        "short_rationale": None,
        "parse_valid": valid,
        "parse_mode": "frozen_scalar_v1",
    }


def cache_key(
    *,
    model: str,
    dataset: str,
    item_id_hash: str,
    source_content_hash: str,
    prompt_hash: str,
    temperature: float,
    seed: int | None,
    schema_version: str,
    run_kind: str,
) -> str:
    parts = [
        schema_version,
        model,
        dataset,
        item_id_hash,
        source_content_hash,
        prompt_hash,
        f"t{temperature}",
        f"seed{seed if seed is not None else 'none'}",
        run_kind,
    ]
    return sha256_text("|".join(parts))


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def estimate_request_cost(cfg: PilotConfig, input_chars: int, output_tokens: int = 8) -> float:
    in_tok = max(1, input_chars // 4)
    return (
        in_tok / 1000 * cfg.cost_per_1k_input_tokens_usd
        + output_tokens / 1000 * cfg.cost_per_1k_output_tokens_usd
    )


def audit_payload_fields(row: pd.Series) -> tuple[bool, list[str]]:
    errors = []
    for col in PROMPT_DENYLIST:
        if col in row.index and pd.notna(row.get(col)):
            errors.append(f"denied field present: {col}")
    stem = str(row.get("item_text_clean", ""))
    banned_substrings = ["correct answer", "error rate", "response count", "student got"]
    low = stem.lower()
    for s in banned_substrings:
        if s in low:
            errors.append(f"suspicious stem substring: {s}")
    return len(errors) == 0, errors


def ensure_dirs() -> None:
    for d in (PILOT_DIR, RAW_DIR, PARSED_DIR, CACHE_DIR, MANIFEST_DIR):
        d.mkdir(parents=True, exist_ok=True)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_pilot_items(dataset: str) -> pd.DataFrame:
    path = PILOT_DIR / f"{dataset}_pilot_items.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Pilot sample missing: {path}")
    return pd.read_parquet(path)


def load_llm_prompt_universe(dataset: str) -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_ROOT / dataset / "llm_prompt_items.parquet")


def openai_credentials() -> tuple[str, str]:
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").strip()
    return key, base
