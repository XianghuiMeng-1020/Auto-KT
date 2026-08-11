#!/usr/bin/env python3
"""Resumable full LLM scoring runner (Amendment 010 deterministic scope)."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "llm_scoring"))

from full_llm_common import (  # noqa: E402
    CACHE_DIR,
    MANIFEST_DIR,
    RAW_DIR,
    FullScoringConfig,
    _json_safe,
    amendment_009_hash,
    amendment_010_hash,
    build_request_plan,
    call_openai,
    decoding_config_hash,
    ensure_dirs,
    estimate_cost,
    full_cache_key,
    git_commit,
    import_pilot_record,
    is_retryable,
    load_cache_index,
    load_pilot_cache,
    openai_credentials,
    pilot_cache_key,
    pilot_record_reusable,
    prompt_hash_for_item,
    protocol_prompt_hash,
    render_messages,
    request_payload_hash,
    save_cache_index,
    sha256_text,
    utc_now,
)
from llm_pilot_common import estimate_tokens  # noqa: E402

MANIFEST_PATH = ROOT / "data_manifests" / "_manifest.json"


def update_phase_status(code: str) -> None:
    if not MANIFEST_PATH.exists():
        return
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest["phase_stop_code"] = code
    manifest["generated_at_utc"] = utc_now()
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def execute_full_scoring(cfg: FullScoringConfig, *, dry_run: bool = False) -> dict[str, Any]:
    api_key, base_url = openai_credentials()
    plan = build_request_plan(cfg)
    cache = load_cache_index()
    pilot = load_pilot_cache()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8")) if MANIFEST_PATH.exists() else {}
    amend_hash = manifest.get("amendment_009_hash", amendment_009_hash())

    stats: dict[str, Any] = {
        "expected_requests": len(plan),
        "cache_hits": 0,
        "pilot_imports": 0,
        "paid_calls": 0,
        "retries": 0,
        "failures": 0,
        "failure_classes": {},
        "input_tokens": 0,
        "output_tokens": 0,
        "cached_tokens": 0,
        "estimated_cost_usd": 0.0,
        "wall_time_s": 0.0,
        "connection_incidents": 0,
        "rate_limit_delays_s": 0.0,
    }

    if not api_key and not dry_run:
        stats["api_unavailable"] = True
        return stats

    pending = 0
    t0 = time.time()
    for i, req in enumerate(plan, 1):
        messages = render_messages(req["stem_text"])
        ph = prompt_hash_for_item(req["stem_text"])
        key = full_cache_key(
            cfg=cfg,
            model=req["model"],
            dataset=req["dataset"],
            item_id_hash=req["item_id_hash"],
            source_content_hash=req["source_content_hash"],
            prompt_hash=ph,
        )

        if key in cache and cache[key].get("parse_status") == "valid":
            stats["cache_hits"] += 1
            continue

        if not dry_run:
            pilot_key = pilot_cache_key(
                model=req["model"],
                dataset=req["dataset"],
                item_id_hash=req["item_id_hash"],
                source_content_hash=req["source_content_hash"],
                prompt_hash=ph,
                temperature=cfg.temperature_deterministic,
                seed=cfg.deterministic_seed,
                schema_version=cfg.pilot_schema_version,
                run_kind="deterministic",
            )
            prec = pilot.get(pilot_key)
            if prec and pilot_record_reusable(prec, req, cfg, amend_hash):
                record = import_pilot_record(prec, req, cfg, key)
                cache[key] = record
                stats["pilot_imports"] += 1
                stats["cache_hits"] += 1
                pending += 1
                if pending >= cfg.checkpoint_every_n:
                    save_cache_index(cache)
                    pending = 0
                continue

        if dry_run:
            continue

        dec_hash = decoding_config_hash(req["model"], cfg)
        req_hash = request_payload_hash(messages, req["model"], cfg)
        max_attempts = 1 + cfg.max_retries_transient + cfg.max_retries_format
        attempts = 0
        record = None
        while attempts < max_attempts:
            attempts += 1
            result = call_openai(
                api_key=api_key,
                base_url=base_url,
                model=req["model"],
                messages=messages,
                cfg=cfg,
            )
            if not result["ok"]:
                fc = result.get("failure_class", "unknown")
                stats["failure_classes"][fc] = stats["failure_classes"].get(fc, 0) + 1
                stats["retries"] += 1
                if fc == "transient_connection":
                    stats["connection_incidents"] += 1
                if fc == "rate_limit":
                    delay = min(2 ** attempts, 30)
                    stats["rate_limit_delays_s"] += delay
                    time.sleep(delay)
                elif is_retryable(fc):
                    time.sleep(min(2 ** attempts, 8))
                if attempts >= max_attempts:
                    stats["failures"] += 1
                    record = {
                        "cache_key": key,
                        **req,
                        "prompt_hash": ph,
                        "request_payload_hash": req_hash,
                        "decoding_config_hash": dec_hash,
                        "API_status": "error",
                        "failure_class": fc,
                        "parse_status": "failed",
                        "error": result.get("error"),
                    }
                continue

            from llm_pilot_common import parse_response_record

            parsed = parse_response_record(result["raw"])
            in_tok = int(result.get("input_tokens") or estimate_tokens(json.dumps(messages)))
            out_tok = int(result.get("output_tokens") or 8)
            cost = estimate_cost(cfg, in_tok, out_tok)
            stats["paid_calls"] += 1
            stats["input_tokens"] += in_tok
            stats["output_tokens"] += out_tok
            stats["cached_tokens"] += int(result.get("cached_tokens") or 0)
            stats["estimated_cost_usd"] += cost

            raw_path = RAW_DIR / f"{key}.json"
            raw_path.write_text(
                json.dumps(
                    {"raw": result["raw"], "meta": _json_safe({k: v for k, v in result.items() if k != "raw"})},
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            record = {
                "cache_key": key,
                **req,
                "prompt_hash": ph,
                "request_payload_hash": req_hash,
                "decoding_config_hash": dec_hash,
                "request_timestamp_utc": utc_now(),
                "cache_hit": False,
                "pilot_cache_import": False,
                "retry_count": attempts - 1,
                "API_status": "ok",
                "raw_response_reference": str(raw_path.relative_to(ROOT)),
                "response_hash": sha256_text(result["raw"]),
                "resolved_model": result["resolved_model"],
                "input_tokens": in_tok,
                "output_tokens": out_tok,
                "cached_tokens": result.get("cached_tokens"),
                "estimated_cost": cost,
                "parse_status": "valid" if parsed["parse_valid"] else "invalid",
                "failure_class": None,
                **parsed,
            }
            cache[key] = record
            pending += 1
            if pending >= cfg.checkpoint_every_n:
                save_cache_index(cache)
                pending = 0

            if not parsed["parse_valid"] and attempts <= cfg.max_retries_format:
                stats["retries"] += 1
                stats["failure_classes"]["parse_failure"] = (
                    stats["failure_classes"].get("parse_failure", 0) + 1
                )
                continue
            break

        if i % 50 == 0:
            print(
                f"  progress {i}/{len(plan)} paid={stats['paid_calls']} "
                f"import={stats['pilot_imports']} cache={stats['cache_hits']} "
                f"fail={stats['failures']}",
                flush=True,
            )

    if pending:
        save_cache_index(cache)
    stats["wall_time_s"] = time.time() - t0

    run_manifest = {
        "generated_at_utc": utc_now(),
        "code_commit": git_commit(),
        "protocol_prompt_hash": protocol_prompt_hash(),
        "amendment_009_hash": amendment_009_hash(),
        "amendment_010_hash": amendment_010_hash(),
        "schema_version": cfg.schema_version,
        "stats": stats,
    }
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    (MANIFEST_DIR / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
    return stats


def main() -> int:
    cfg = FullScoringConfig.load()
    ensure_dirs()

    from run_full_scoring_preflight import run_checks, write_report

    passed, fails, meta = run_checks(cfg)
    write_report(passed, meta, cfg)
    if not passed:
        print("Preflight blocked:", fails, file=sys.stderr)
        return 1

    api_key, _ = openai_credentials()
    if not api_key:
        print("FULL_LLM_SCORING_BLOCKED: OPENAI_API_KEY unset", file=sys.stderr)
        return 2

    update_phase_status("FULL_LLM_SCORING_RUNNING")
    stats = execute_full_scoring(cfg)
    print(
        f"Complete: expected={stats['expected_requests']} paid={stats['paid_calls']} "
        f"pilot_import={stats['pilot_imports']} cache_hits={stats['cache_hits']} "
        f"failures={stats['failures']} cost=${stats['estimated_cost_usd']:.4f}"
    )
    return 0 if stats["failures"] == 0 else 3


if __name__ == "__main__":
    raise SystemExit(main())
