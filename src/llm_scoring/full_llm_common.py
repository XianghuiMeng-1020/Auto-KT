"""Shared utilities for Phase E full LLM scoring."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from llm_pilot_common import (  # noqa: E402
    FROZEN_SYSTEM_MESSAGE,
    FROZEN_USER_TEMPLATE,
    OUTCOME_TABLE_DENYLIST,
    PROMPT_DENYLIST,
    PROCESSED_ROOT,
    audit_payload_fields,
    cache_key as pilot_cache_key,
    estimate_tokens,
    git_branch,
    git_clean,
    git_commit,
    openai_credentials,
    parse_difficulty,
    parse_response_record,
    prompt_hash_for_item,
    protocol_prompt_hash,
    render_messages,
    sha256_file,
    sha256_text,
    utc_now,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "full_llm_scoring_config.json"
JOURNAL_ROOT = ROOT / "artifacts"
FULL_LLM_ROOT = JOURNAL_ROOT / "features" / "full_llm"
RAW_DIR = FULL_LLM_ROOT / "raw"
CACHE_DIR = FULL_LLM_ROOT / "cache"
RATIONALES_DIR = FULL_LLM_ROOT / "rationales"
PARSED_DIR = FULL_LLM_ROOT / "parsed"
MANIFEST_DIR = FULL_LLM_ROOT / "manifests"
PILOT_CACHE_DIR = JOURNAL_ROOT / "features" / "llm_pilot" / "cache"
AMENDMENT_009_PATH = ROOT / "protocol" / "amendments" / "AMENDMENT_009_CONTENT_SUFFICIENCY_AND_ITEM_UNIVERSES.md"
AMENDMENT_010_PATH = ROOT / "protocol" / "amendments" / "AMENDMENT_010_FULL_SCORING_REPLICATION_SCOPE.md"

DATASETS = ("xes3g5m", "junyi")
FAILURE_CLASSES = frozenset({
    "transient_connection",
    "rate_limit",
    "invalid_request",
    "model_unavailable",
    "parse_failure",
    "schema_failure",
    "content_policy",
    "unknown",
})


@dataclass(frozen=True)
class FullScoringConfig:
    schema_version: str
    pilot_schema_version: str
    replication_scope: str
    models: list[str]
    temperature_deterministic: float
    deterministic_seed: int
    max_tokens: int
    timeout_seconds: int
    max_retries_transient: int
    max_retries_format: int
    parse_success_threshold: float
    full_scoring_budget_usd: float
    checkpoint_every_n: int
    scoreable_counts: dict[str, int]
    cost_per_1k_input_tokens_usd: float
    cost_per_1k_output_tokens_usd: float
    content_sufficiency_rule_version: str

    @classmethod
    def load(cls, path: Path | None = None) -> FullScoringConfig:
        raw = json.loads((path or CONFIG_PATH).read_text(encoding="utf-8"))
        return cls(
            schema_version=raw["schema_version"],
            pilot_schema_version=raw["pilot_schema_version"],
            replication_scope=raw["replication_scope"],
            models=list(raw["models"]),
            temperature_deterministic=raw["temperature_deterministic"],
            deterministic_seed=raw["deterministic_seed"],
            max_tokens=raw["max_tokens"],
            timeout_seconds=raw["timeout_seconds"],
            max_retries_transient=raw["max_retries_transient"],
            max_retries_format=raw["max_retries_format"],
            parse_success_threshold=raw["parse_success_threshold"],
            full_scoring_budget_usd=raw["full_scoring_budget_usd"],
            checkpoint_every_n=raw["checkpoint_every_n"],
            scoreable_counts=dict(raw["scoreable_counts"]),
            cost_per_1k_input_tokens_usd=raw["cost_per_1k_input_tokens_usd"],
            cost_per_1k_output_tokens_usd=raw["cost_per_1k_output_tokens_usd"],
            content_sufficiency_rule_version=raw["content_sufficiency_rule_version"],
        )


def amendment_009_hash() -> str:
    return sha256_file(AMENDMENT_009_PATH) if AMENDMENT_009_PATH.exists() else ""


def amendment_010_hash() -> str:
    return sha256_file(AMENDMENT_010_PATH) if AMENDMENT_010_PATH.exists() else ""


def decoding_config_hash(model: str, cfg: FullScoringConfig) -> str:
    payload = {
        "model": model,
        "temperature": cfg.temperature_deterministic,
        "max_tokens": cfg.max_tokens,
        "seed": cfg.deterministic_seed,
        "uses_max_completion_tokens": model.startswith("gpt-5")
        or model.startswith("o1")
        or model.startswith("o3"),
    }
    return sha256_text(json.dumps(payload, sort_keys=True))


def request_payload_hash(messages: list[dict[str, str]], model: str, cfg: FullScoringConfig) -> str:
    payload = {
        "messages": messages,
        "model": model,
        "temperature": cfg.temperature_deterministic,
        "max_tokens": cfg.max_tokens,
        "seed": cfg.deterministic_seed,
        "schema_version": cfg.schema_version,
    }
    return sha256_text(json.dumps(payload, sort_keys=True, ensure_ascii=False))


def full_cache_key(
    *,
    cfg: FullScoringConfig,
    model: str,
    dataset: str,
    item_id_hash: str,
    source_content_hash: str,
    prompt_hash: str,
) -> str:
    return sha256_text("|".join([
        cfg.schema_version,
        model,
        dataset,
        item_id_hash,
        source_content_hash,
        prompt_hash,
        f"t{cfg.temperature_deterministic}",
        f"seed{cfg.deterministic_seed}",
        "deterministic",
        cfg.content_sufficiency_rule_version,
        amendment_009_hash(),
    ]))


def load_scoreable_queue(cfg: FullScoringConfig) -> pd.DataFrame:
    rows = []
    for ds in DATASETS:
        prompt = pd.read_parquet(PROCESSED_ROOT / ds / "llm_prompt_items.parquet")
        items = pd.read_parquet(PROCESSED_ROOT / ds / "items.parquet")
        if "eligible_for_llm_scoring" in items.columns:
            eligible = set(items.loc[items["eligible_for_llm_scoring"], "item_id_hash"])
            prompt = prompt[prompt["item_id_hash"].isin(eligible)]
        expected = cfg.scoreable_counts[ds]
        if len(prompt) != expected:
            raise ValueError(f"{ds}: expected {expected} scoreable items, got {len(prompt)}")
        for _, row in prompt.iterrows():
            rows.append(row.to_dict())
    df = pd.DataFrame(rows)
    return df.sort_values(["dataset", "item_id_hash"], kind="mergesort")


def build_request_plan(cfg: FullScoringConfig) -> list[dict[str, Any]]:
    plan = []
    queue = load_scoreable_queue(cfg)
    queue = queue.sort_values(["dataset", "item_id_hash"], kind="mergesort")
    for _, row in queue.iterrows():
        for model in cfg.models:
            plan.append({
                "dataset": row["dataset"],
                "item_id_hash": row["item_id_hash"],
                "source_content_hash": row["source_content_hash"],
                "stem_text": str(row["item_text_clean"]),
                "model": model,
                "content_sufficiency_rule_version": cfg.content_sufficiency_rule_version,
                "eligible_for_llm_scoring": True,
            })
    return plan


def classify_api_error(error: str) -> str:
    err = error.lower()
    if "connection" in err or "timeout" in err:
        return "transient_connection"
    if "rate" in err and "limit" in err:
        return "rate_limit"
    if "model_not_found" in err or "does not exist" in err:
        return "model_unavailable"
    if "invalid_request" in err or "unsupported_parameter" in err:
        return "invalid_request"
    if "content_policy" in err or "content_filter" in err:
        return "content_policy"
    if "permission" in err or "403" in err:
        return "model_unavailable"
    return "unknown"


def is_retryable(failure_class: str) -> bool:
    return failure_class in {"transient_connection", "rate_limit", "unknown"}


def _json_safe(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if hasattr(obj, "model_dump"):
        return _json_safe(obj.model_dump())
    return str(obj)


def call_openai(
    *,
    api_key: str,
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    cfg: FullScoringConfig,
) -> dict[str, Any]:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("openai package required") from exc

    client = OpenAI(api_key=api_key, base_url=base_url, timeout=cfg.timeout_seconds)
    uses_completion_tokens = model.startswith("gpt-5") or model.startswith("o1") or model.startswith("o3")
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": cfg.temperature_deterministic,
        "seed": cfg.deterministic_seed,
    }
    if uses_completion_tokens:
        kwargs["max_completion_tokens"] = cfg.max_tokens
    else:
        kwargs["max_tokens"] = cfg.max_tokens
    t0 = time.time()
    try:
        resp = client.chat.completions.create(**kwargs)
    except Exception as exc:
        err = repr(exc)
        return {
            "ok": False,
            "error": err,
            "failure_class": classify_api_error(err),
            "latency_s": time.time() - t0,
            "resolved_model": model,
        }
    choice = resp.choices[0]
    usage = getattr(resp, "usage", None)
    usage_dict = _json_safe(usage.model_dump()) if usage and hasattr(usage, "model_dump") else None
    cached_tokens = None
    if usage_dict and isinstance(usage_dict.get("prompt_tokens_details"), dict):
        cached_tokens = usage_dict["prompt_tokens_details"].get("cached_tokens")
    return {
        "ok": True,
        "raw": choice.message.content or "",
        "resolved_model": getattr(resp, "model", model),
        "latency_s": time.time() - t0,
        "input_tokens": getattr(usage, "prompt_tokens", None) if usage else None,
        "output_tokens": getattr(usage, "completion_tokens", None) if usage else None,
        "cached_tokens": cached_tokens,
        "usage": usage_dict,
        "failure_class": None,
    }


def load_cache_index() -> dict[str, dict]:
    path = CACHE_DIR / "cache_index.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def save_cache_index(index: dict[str, dict]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = CACHE_DIR / "cache_index.json.tmp"
    tmp.write_text(json.dumps(index, indent=2), encoding="utf-8")
    tmp.replace(CACHE_DIR / "cache_index.json")


def load_pilot_cache() -> dict[str, dict]:
    path = PILOT_CACHE_DIR / "cache_index.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def pilot_record_reusable(
    record: dict[str, Any],
    req: dict[str, Any],
    cfg: FullScoringConfig,
    manifest_amendment_hash: str,
) -> bool:
    if record.get("run_kind") != "deterministic":
        return False
    if not record.get("parse_valid"):
        return False
    if record.get("dataset") != req["dataset"]:
        return False
    if record.get("item_id_hash") != req["item_id_hash"]:
        return False
    if record.get("source_content_hash") != req["source_content_hash"]:
        return False
    if record.get("model") != req["model"]:
        return False
    if float(record.get("temperature", -1)) != cfg.temperature_deterministic:
        return False
    ph = prompt_hash_for_item(req["stem_text"])
    if record.get("prompt_hash") and record["prompt_hash"] != ph:
        return False
    if manifest_amendment_hash and amendment_009_hash() != manifest_amendment_hash:
        return False
    return True


def import_pilot_record(
    record: dict[str, Any],
    req: dict[str, Any],
    cfg: FullScoringConfig,
    key: str,
) -> dict[str, Any]:
    messages = render_messages(req["stem_text"])
    ph = prompt_hash_for_item(req["stem_text"])
    dec_hash = decoding_config_hash(req["model"], cfg)
    req_hash = request_payload_hash(messages, req["model"], cfg)
    raw = record.get("raw_response", "")
    parsed = parse_response_record(raw)
    return {
        "cache_key": key,
        **req,
        "prompt_hash": ph,
        "request_payload_hash": req_hash,
        "decoding_config_hash": dec_hash,
        "request_timestamp_utc": record.get("request_timestamp_utc", utc_now()),
        "cache_hit": True,
        "pilot_cache_import": True,
        "retry_count": 0,
        "API_status": "ok",
        "raw_response_reference": f"pilot:{record.get('cache_key', '')}",
        "parse_status": "valid" if parsed["parse_valid"] else "invalid",
        "response_hash": sha256_text(raw),
        "resolved_model": record.get("resolved_model"),
        "input_tokens": record.get("input_tokens"),
        "output_tokens": record.get("output_tokens"),
        "cached_tokens": None,
        "estimated_cost": 0.0,
        **parsed,
    }


def estimate_cost(cfg: FullScoringConfig, input_tokens: int, output_tokens: int) -> float:
    return (
        input_tokens / 1000 * cfg.cost_per_1k_input_tokens_usd
        + output_tokens / 1000 * cfg.cost_per_1k_output_tokens_usd
    )


def ensure_dirs() -> None:
    for d in (RAW_DIR, CACHE_DIR, RATIONALES_DIR, PARSED_DIR, MANIFEST_DIR):
        d.mkdir(parents=True, exist_ok=True)


def response_to_feature_row(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset": record["dataset"],
        "item_id_hash": record["item_id_hash"],
        "source_content_hash": record["source_content_hash"],
        "eligible_for_llm_scoring": record.get("eligible_for_llm_scoring", True),
        "content_sufficiency_rule_version": record.get("content_sufficiency_rule_version"),
        "model_identifier": record["model"],
        "prompt_hash": record.get("prompt_hash"),
        "request_payload_hash": record.get("request_payload_hash"),
        "decoding_config_hash": record.get("decoding_config_hash"),
        "request_timestamp_utc": record.get("request_timestamp_utc"),
        "cache_hit": bool(record.get("cache_hit")),
        "pilot_cache_import": bool(record.get("pilot_cache_import")),
        "retry_count": int(record.get("retry_count", 0)),
        "API_status": record.get("API_status"),
        "raw_response_reference": record.get("raw_response_reference"),
        "parse_status": record.get("parse_status"),
        "scalar_difficulty": record.get("scalar_difficulty"),
        "conceptual_demand": record.get("conceptual_demand"),
        "procedural_steps": record.get("procedural_steps"),
        "reading_demand": record.get("reading_demand"),
        "prerequisite_depth": record.get("prerequisite_depth"),
        "distractor_complexity": record.get("distractor_complexity"),
        "representational_complexity": record.get("representational_complexity"),
        "confidence": record.get("confidence"),
        "short_rationale_reference": record.get("short_rationale"),
        "input_tokens": record.get("input_tokens"),
        "output_tokens": record.get("output_tokens"),
        "cached_tokens": record.get("cached_tokens"),
        "estimated_cost": record.get("estimated_cost"),
        "response_hash": record.get("response_hash"),
    }
