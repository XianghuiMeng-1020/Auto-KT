#!/usr/bin/env python3
"""Bounded LLM pilot runner with deterministic cache and frozen prompt."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "llm_scoring"))

from llm_pilot_common import (  # noqa: E402
    CACHE_DIR,
    MANIFEST_DIR,
    PARSED_DIR,
    PILOT_DIR,
    RAW_DIR,
    PilotConfig,
    audit_payload_fields,
    cache_key,
    ensure_dirs,
    estimate_request_cost,
    estimate_tokens,
    git_commit,
    openai_credentials,
    parse_response_record,
    prompt_hash_for_item,
    protocol_prompt_hash,
    render_messages,
    utc_now,
)

DATASETS = ("xes3g5m", "junyi")


def load_cache_index() -> dict[str, dict]:
    path = CACHE_DIR / "cache_index.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def save_cache_index(index: dict[str, dict]) -> None:
    (CACHE_DIR / "cache_index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")


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
    temperature: float,
    max_tokens: int,
    timeout: int,
    seed: int | None,
) -> dict[str, Any]:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("openai package required") from exc

    client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
    # gpt-5.x requires max_completion_tokens; older models use max_tokens
    uses_completion_tokens = model.startswith("gpt-5") or model.startswith("o1") or model.startswith("o3")
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if uses_completion_tokens:
        kwargs["max_completion_tokens"] = max_tokens
    else:
        kwargs["max_tokens"] = max_tokens
    if seed is not None:
        kwargs["seed"] = seed
    t0 = time.time()
    try:
        resp = client.chat.completions.create(**kwargs)
    except Exception as exc:
        return {
            "ok": False,
            "error": repr(exc),
            "latency_s": time.time() - t0,
            "resolved_model": model,
        }
    choice = resp.choices[0]
    usage = getattr(resp, "usage", None)
    usage_dict = _json_safe(usage.model_dump()) if usage and hasattr(usage, "model_dump") else None
    return {
        "ok": True,
        "raw": choice.message.content or "",
        "resolved_model": getattr(resp, "model", model),
        "latency_s": time.time() - t0,
        "input_tokens": getattr(usage, "prompt_tokens", None) if usage else None,
        "output_tokens": getattr(usage, "completion_tokens", None) if usage else None,
        "usage": usage_dict,
    }


def request_plan(cfg: PilotConfig, datasets: list[str], n_per_ds: int) -> list[dict]:
    plan = []
    for ds in datasets:
        pilot = pd.read_parquet(PILOT_DIR / f"{ds}_pilot_items.parquet").head(n_per_ds)
        for _, row in pilot.iterrows():
            for model in cfg.models:
                plan.append({
                    "dataset": ds,
                    "item_id_hash": row["item_id_hash"],
                    "source_content_hash": row["source_content_hash"],
                    "stem_text": str(row["item_text_clean"]),
                    "model": model,
                    "temperature": cfg.temperature_deterministic,
                    "run_kind": "deterministic",
                    "seed": cfg.pilot_seed,
                })
                for rep in range(cfg.stability_replicates):
                    plan.append({
                        "dataset": ds,
                        "item_id_hash": row["item_id_hash"],
                        "source_content_hash": row["source_content_hash"],
                        "stem_text": str(row["item_text_clean"]),
                        "model": model,
                        "temperature": cfg.temperature_stability,
                        "run_kind": f"stability_{rep + 1}",
                        "seed": cfg.pilot_seed + rep + 1,
                    })
    return plan


def is_permanent_api_error(error: str) -> bool:
    err = error.lower()
    return any(
        x in err
        for x in (
            "model_not_found",
            "does not exist",
            "invalid model",
            "you do not have access",
            "permission",
        )
    )


def execute_plan(
    plan: list[dict],
    cfg: PilotConfig,
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    api_key, base_url = openai_credentials()
    cache_index = load_cache_index()
    stats = {
        "expected_requests": len(plan),
        "cache_hits": 0,
        "paid_calls": 0,
        "retries": 0,
        "failures": 0,
        "resolved_models": {},
        "input_tokens": 0,
        "output_tokens": 0,
        "wall_time_s": 0.0,
        "records": [],
    }
    if not api_key and not dry_run:
        stats["api_unavailable"] = True
        return stats

    blocked_models: set[str] = set()
    pending_saves = 0
    t_start = time.time()
    for i, req in enumerate(plan, 1):
        if req["model"] in blocked_models:
            stats["failures"] += 1
            continue
        ph = prompt_hash_for_item(req["stem_text"])
        key = cache_key(
            model=req["model"],
            dataset=req["dataset"],
            item_id_hash=req["item_id_hash"],
            source_content_hash=req["source_content_hash"],
            prompt_hash=ph,
            temperature=req["temperature"],
            seed=req.get("seed"),
            schema_version=cfg.schema_version,
            run_kind=req["run_kind"],
        )
        if key in cache_index and cache_index[key].get("parse_valid"):
            stats["cache_hits"] += 1
            continue

        if dry_run:
            continue

        messages = render_messages(req["stem_text"])
        attempts = 0
        max_attempts = 1 + cfg.max_retries_transient + cfg.max_retries_format
        record = None
        while attempts < max_attempts:
            attempts += 1
            result = call_openai(
                api_key=api_key,
                base_url=base_url,
                model=req["model"],
                messages=messages,
                temperature=req["temperature"],
                max_tokens=cfg.max_tokens,
                timeout=cfg.timeout_seconds,
                seed=req.get("seed"),
            )
            if not result["ok"]:
                stats["retries"] += 1
                if is_permanent_api_error(result.get("error", "")):
                    blocked_models.add(req["model"])
                    stats["model_errors"] = stats.get("model_errors", {})
                    stats["model_errors"][req["model"]] = result["error"]
                    stats["failures"] += 1
                    record = {"cache_key": key, **req, "error": result["error"], "parse_valid": False}
                    break
                if attempts >= max_attempts:
                    stats["failures"] += 1
                    record = {"cache_key": key, **req, "error": result["error"], "parse_valid": False}
                time.sleep(min(2 ** attempts, 8))
                continue

            stats["paid_calls"] += 1
            stats["resolved_models"][req["model"]] = result["resolved_model"]
            stats["input_tokens"] += int(result.get("input_tokens") or estimate_tokens(json.dumps(messages)))
            stats["output_tokens"] += int(result.get("output_tokens") or 8)

            parsed = parse_response_record(result["raw"])
            raw_path = RAW_DIR / f"{key}.json"
            raw_path.write_text(
                json.dumps(
                    {"raw": result["raw"], "meta": _json_safe({k: v for k, v in result.items() if k != "raw"})},
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            parsed_path = PARSED_DIR / f"{key}.json"
            parsed_path.write_text(json.dumps(parsed, indent=2), encoding="utf-8")

            record = {
                "cache_key": key,
                **req,
                "prompt_hash": ph,
                "raw_response": result["raw"],
                "resolved_model": result["resolved_model"],
                "latency_s": result["latency_s"],
                "attempts": attempts,
                **parsed,
            }
            cache_index[key] = record
            pending_saves += 1
            if pending_saves >= 25:
                save_cache_index(cache_index)
                pending_saves = 0

            if not parsed["parse_valid"] and attempts <= cfg.max_retries_format:
                stats["retries"] += 1
                continue
            break

        if record:
            stats["records"].append(record)
        if i % 20 == 0:
            print(
                f"  progress {i}/{len(plan)} paid={stats['paid_calls']} "
                f"cache_hits={stats['cache_hits']} failures={stats['failures']}",
                flush=True,
            )

    if pending_saves:
        save_cache_index(cache_index)
    stats["wall_time_s"] = time.time() - t_start
    save_cache_index(cache_index)
    run_manifest = {
        "generated_at_utc": utc_now(),
        "code_commit": git_commit(),
        "protocol_prompt_hash": protocol_prompt_hash(),
        "stats": {k: v for k, v in stats.items() if k != "records"},
    }
    (MANIFEST_DIR / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
    pd.DataFrame(stats["records"]).to_parquet(MANIFEST_DIR / "request_log.parquet", index=False)
    return stats


def projected_cost(plan: list[dict], cfg: PilotConfig) -> float:
    total = 0.0
    for req in plan:
        msgs = render_messages(req["stem_text"])
        total += estimate_request_cost(cfg, len(json.dumps(msgs)))
    return total


def main() -> int:
    cfg = PilotConfig.load()
    ensure_dirs()

    pilot_paths = [PILOT_DIR / f"{ds}_pilot_items.parquet" for ds in DATASETS]
    if not all(p.exists() for p in pilot_paths):
        print("Pilot samples missing; run sample_pilot_items.py first", file=sys.stderr)
        return 1

    preflight_plan = request_plan(cfg, list(DATASETS), cfg.preflight_items_per_dataset)
    est = projected_cost(preflight_plan, cfg)
    print(f"Preflight projected cost (${cfg.preflight_items_per_dataset} items/ds): ${est:.4f}")
    if est > cfg.pilot_budget_usd:
        print("PILOT_COST_BLOCKED: preflight projection exceeds budget", file=sys.stderr)
        return 2

    api_key, _ = openai_credentials()
    if not api_key:
        print("LLM_PILOT_API_UNAVAILABLE: OPENAI_API_KEY unset")
        return 3

    pre_stats = execute_plan(preflight_plan, cfg)
    print(f"Preflight: paid={pre_stats['paid_calls']} cache_hits={pre_stats['cache_hits']} failures={pre_stats['failures']}")

    full_plan = request_plan(cfg, list(DATASETS), cfg.pilot_items_per_dataset)
    full_est = projected_cost(full_plan, cfg)
    print(f"Full pilot projected cost: ${full_est:.4f}")
    if full_est > cfg.pilot_budget_usd:
        print("PILOT_COST_BLOCKED: full projection exceeds budget", file=sys.stderr)
        return 2

    stats = execute_plan(full_plan, cfg)
    print(
        f"Pilot complete: expected={stats['expected_requests']} "
        f"paid={stats['paid_calls']} cache_hits={stats['cache_hits']} failures={stats['failures']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
