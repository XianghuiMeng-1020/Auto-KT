#!/usr/bin/env python3
"""TLT-3D Phase 2 — DBE confirmatory LLM scoring (NO learner outcomes / NO KT).

Resumable, fail-closed scorer under sealed protocol v1.2.
Uses production FullScoringConfig request semantics + DBE domain twin system message.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "llm"))

from full_llm_common import (  # noqa: E402
    FullScoringConfig,
    _json_safe,
    call_openai,
    classify_api_error,
    decoding_config_hash,
    is_retryable,
    request_payload_hash,
)
from llm_pilot_common import (  # noqa: E402
    FROZEN_USER_TEMPLATE,
    estimate_tokens,
    openai_credentials,
    parse_response_record,
    sha256_file,
    sha256_text,
)

PROTOCOL_COMMIT = "a459e34a24240c03ba4dbe4b1d0185e42eaf4377"
PROTOCOL_TAG = "tlt3d-pre-dbe-scoring-v1.2"
PROTOCOL_VERSION = "1.2.0"
EXPECTED_DBE_PROMPT_HASH = "d99a9645219033e713bb78fd31dc3d74826bf31adf1a0e5977f30fcbda911c35"
EXPECTED_MATH_PROMPT_HASH = "f47c02a4cc4a15f46f280908c13ba2ebacfefe36dc7307eb3c18ca1695efa165"
EXPECTED_SPLIT_HASH = "65d13de13e7c3a8b366c63628cab3e7ef3f2a67b8eae7cde1aab4e1a53d627dc"
EXPECTED_FOLD_HASH = "28efbf33c231772a3565056367c4bf6dfa62bdf553abcfd8397aa9d9037d4e0a"
ALLOWED_MODELS = ("gpt-4o-mini", "gpt-5.4")
PARSER_VERSION = "frozen_scalar_v1"

ITEMS_PATH = ROOT / "data/external/dbe_kt22/derived/tlt3d_canonical_items.parquet"
UNIVERSE_PATH = ROOT / "configs/tlt3d/dbe_item_universe.json"
PROMPT_PATH = ROOT / "configs/tlt3d/dbe_prompt_v1.txt"
PROTOCOL_PATH = ROOT / "configs/tlt3d/TLT3D_PROTOCOL_FREEZE_v1_2.json"
FREEZE_PATH = ROOT / "artifacts/tlt3d/DBE_PRE_LLM_FREEZE.json"
SUMMARY_P11 = ROOT / "artifacts/tlt3d/P11_AMENDMENT_COMPUTE_SUMMARY.json"
ART = ROOT / "artifacts/tlt3d"
RAW_MINI = ART / "dbe_llm_raw_gpt4omini.jsonl"
RAW_54 = ART / "dbe_llm_raw_gpt54.jsonl"
SCORE_CSV = ART / "dbe_llm_scores_confirmatory.csv"
MANIFEST_PATH = ART / "DBE_LLM_SCORE_MANIFEST.json"
SUMMARY_PATH = ART / "P2_DBE_LLM_SCORING_SUMMARY.json"
STATE_DIR = ART / "dbe_llm_scoring_state"

FORBIDDEN_ITEM_COLS = {
    "expert_difficulty",
    "expert_difficulty_secondary_only",
    "is_correct",
    "answer_state",
    "correct_choice",
    "learner_error",
    "empirical_difficulty",
    "response_count",
    "acceptance_rate",
}
FORBIDDEN_IMPORT_MODULES = {
    "tlt3d_kt_interactions",
    "learner_error",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_dbe_system_message() -> str:
    s = json.loads(SUMMARY_P11.read_text(encoding="utf-8"))
    return s["prompt_twin"]["dbe_system"]


def dbe_protocol_prompt_hash(system: str) -> str:
    return sha256_text(system + "\n---\n" + FROZEN_USER_TEMPLATE)


def render_dbe_messages(stem_text: str, system: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": FROZEN_USER_TEMPLATE.format(stem_text=stem_text)},
    ]


def assert_protocol_identity() -> dict[str, Any]:
    freeze = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    universe = json.loads(UNIVERSE_PATH.read_text(encoding="utf-8"))
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    system = load_dbe_system_message()
    ph = dbe_protocol_prompt_hash(system)
    math_h = sha256_text(
        json.loads(SUMMARY_P11.read_text(encoding="utf-8"))["prompt_twin"]["math_system"]
        + "\n---\n"
        + FROZEN_USER_TEMPLATE
    )
    errors = []
    if ph != EXPECTED_DBE_PROMPT_HASH:
        errors.append(f"prompt_hash drift: {ph}")
    if math_h != EXPECTED_MATH_PROMPT_HASH:
        errors.append(f"math_prompt_hash drift: {math_h}")
    if freeze["learner_split_hash"] != EXPECTED_SPLIT_HASH:
        errors.append("split_hash drift")
    if freeze["unseen_fold_hash"] != EXPECTED_FOLD_HASH:
        errors.append("fold_hash drift")
    if int(universe["included_count"]) != 166 or len(universe["included_item_ids"]) != 166:
        errors.append("item count drift")
    if not protocol.get("dbe_llm_scoring_authorized_after_commit"):
        errors.append("scoring not authorized in protocol")
    if protocol.get("protocol_status") != "SEALED_PRE_RESULT":
        errors.append(f"unexpected protocol_status={protocol.get('protocol_status')}")
    if errors:
        raise RuntimeError("PHASE2_BLOCKED_PROTOCOL_DRIFT: " + "; ".join(errors))
    return {
        "prompt_hash": ph,
        "math_prompt_hash": math_h,
        "item_universe_hash": universe["item_universe_hash"],
        "learner_split_hash": freeze["learner_split_hash"],
        "unseen_fold_hash": freeze["unseen_fold_hash"],
        "system_message": system,
        "universe": universe,
        "freeze": freeze,
    }


def load_scoring_items(universe: dict[str, Any]) -> pd.DataFrame:
    """Load ONLY item-side scoring fields. Never joins learner tables."""
    if not ITEMS_PATH.exists():
        raise FileNotFoundError(ITEMS_PATH)
    # Explicit column allowlist — excludes expert_difficulty_secondary_only
    cols = ["item_id", "scoring_text", "scoring_text_hash", "dataset"]
    df = pd.read_parquet(ITEMS_PATH, columns=cols)
    for c in FORBIDDEN_ITEM_COLS:
        if c in df.columns:
            raise RuntimeError(f"forbidden column loaded into scorer: {c}")
    included = set(int(i) for i in universe["included_item_ids"])
    excluded = {int(x["item_id"]) for x in universe["excluded"]}
    df["item_id"] = df["item_id"].astype(int)
    if set(df["item_id"]) != included:
        raise RuntimeError("item universe mismatch vs dbe_item_universe.json")
    if set(df["item_id"]) & excluded:
        raise RuntimeError("excluded item present in scoring input")
    # answer-key metadata leakage tokens
    for _, row in df.iterrows():
        low = str(row["scoring_text"]).lower()
        for tok in ("correct answer", "answer key", "is_correct", "expert_difficulty"):
            if tok in low:
                raise RuntimeError(f"forbidden metadata token in item {row['item_id']}: {tok}")
    return df.sort_values("item_id", kind="mergesort").reset_index(drop=True)


def raw_path_for_model(model: str) -> Path:
    if model == "gpt-4o-mini":
        return RAW_MINI
    if model == "gpt-5.4":
        return RAW_54
    raise ValueError(model)


def state_path_for_model(model: str) -> Path:
    safe = model.replace(".", "").replace("-", "")
    return STATE_DIR / f"accepted_{safe}.json"


def load_accepted(model: str) -> dict[str, dict]:
    path = state_path_for_model(model)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def save_accepted(model: str, accepted: dict[str, dict]) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    path = state_path_for_model(model)
    path.write_text(json.dumps(accepted, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def append_raw(model: str, record: dict[str, Any]) -> None:
    path = raw_path_for_model(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def request_params_hash(model: str, cfg: FullScoringConfig) -> str:
    payload = {
        "model": model,
        "temperature": cfg.temperature_deterministic,
        "seed": cfg.deterministic_seed,
        "max_tokens": cfg.max_tokens,
        "timeout_seconds": cfg.timeout_seconds,
        "replication_scope": cfg.replication_scope,
    }
    return sha256_text(json.dumps(payload, sort_keys=True))


def score_model(
    *,
    model: str,
    items: pd.DataFrame,
    system: str,
    prompt_hash: str,
    cfg: FullScoringConfig,
    resume: bool,
    dry_run: bool,
) -> dict[str, Any]:
    if model not in ALLOWED_MODELS:
        raise RuntimeError(f"refuses unknown model ID: {model}")
    api_key, base_url = openai_credentials()
    if not api_key and not dry_run:
        raise RuntimeError("API_CREDENTIAL_BLOCKED")

    accepted = load_accepted(model) if resume else {}
    stats = {
        "model": model,
        "expected": 166,
        "accepted": 0,
        "missing": 0,
        "duplicates": 0,
        "api_requests_attempted": 0,
        "transport_retries": 0,
        "invalid_model_outputs": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_hits": 0,
        "started_utc": utc_now(),
        "completed_utc": None,
    }

    # rebuild raw file from accepted on resume if needed: append-only; skip re-call
    for _, row in items.iterrows():
        item_id = int(row["item_id"])
        item_key = str(item_id)
        text = str(row["scoring_text"])
        text_hash = str(row["scoring_text_hash"])
        if item_key in accepted:
            prev = accepted[item_key]
            if prev.get("item_text_hash") != text_hash:
                raise RuntimeError(
                    f"existing score has different item hash for item {item_id}"
                )
            if prev.get("prompt_hash") != prompt_hash:
                raise RuntimeError("existing score prompt hash mismatch")
            if prev.get("model_id") != model:
                raise RuntimeError("existing score model mismatch")
            score = float(prev["parsed_score"])
            if not (0.0 <= score <= 1.0):
                raise RuntimeError(f"score outside [0,1] for item {item_id}")
            stats["cache_hits"] += 1
            continue

        if dry_run:
            continue

        messages = render_dbe_messages(text, system)
        req_hash = request_payload_hash(messages, model, cfg)
        # request_payload_hash uses math system via messages we pass — OK, messages include DBE system
        dec_hash = decoding_config_hash(model, cfg)
        params_hash = request_params_hash(model, cfg)
        max_attempts = 1 + cfg.max_retries_transient + cfg.max_retries_format
        attempts = 0
        final_record = None
        while attempts < max_attempts:
            attempts += 1
            stats["api_requests_attempted"] += 1
            result = call_openai(
                api_key=api_key,
                base_url=base_url,
                model=model,
                messages=messages,
                cfg=cfg,
            )
            if not result["ok"]:
                fc = result.get("failure_class") or classify_api_error(str(result.get("error")))
                if fc == "model_unavailable":
                    raise RuntimeError("MODEL_UNAVAILABLE_PI_REVIEW_REQUIRED")
                if fc == "invalid_request":
                    # Distinguish mechanical vs semantic — do not silently change semantics
                    err = str(result.get("error", ""))
                    raise RuntimeError(
                        f"API parameter rejection (no silent semantic change): {err[:200]}"
                    )
                stats["transport_retries"] += 1
                if is_retryable(fc) and attempts < max_attempts:
                    time.sleep(min(2 ** attempts, 8))
                    continue
                raise RuntimeError(f"transport failure exhausted for item {item_id}: {fc}")

            parsed = parse_response_record(result["raw"])
            in_tok = int(result.get("input_tokens") or estimate_tokens(json.dumps(messages)))
            out_tok = int(result.get("output_tokens") or 8)
            stats["input_tokens"] += in_tok
            stats["output_tokens"] += out_tok

            if not parsed["parse_valid"]:
                stats["invalid_model_outputs"] += 1
                if attempts <= (1 + cfg.max_retries_transient + cfg.max_retries_format) and (
                    attempts <= cfg.max_retries_format
                    or attempts < max_attempts
                ):
                    # follow production: format retries after content returned
                    if attempts < max_attempts:
                        continue
                raise RuntimeError(
                    f"INVALID_MODEL_OUTPUT item={item_id} model={model} raw_hash="
                    f"{sha256_text(result['raw'])}"
                )

            score = float(parsed["scalar_difficulty"])
            if not (0.0 <= score <= 1.0):
                raise RuntimeError(f"score outside [0,1] item={item_id} score={score}")

            raw_hash = sha256_text(result["raw"])
            final_record = {
                "dataset": "dbe_kt22",
                "item_id": item_id,
                "item_text_hash": text_hash,
                "prompt_hash": prompt_hash,
                "model_id": model,
                "request_parameters_hash": params_hash,
                "request_payload_hash": req_hash,
                "decoding_config_hash": dec_hash,
                "request_timestamp_utc": utc_now(),
                "attempt_count": attempts,
                "raw_model_content": result["raw"],
                "raw_response_content_hash": raw_hash,
                "parsed_score": score,
                "parser_version": PARSER_VERSION,
                "input_tokens": in_tok,
                "output_tokens": out_tok,
                "resolved_model": result.get("resolved_model"),
                "protocol_version": PROTOCOL_VERSION,
                "protocol_tag": PROTOCOL_TAG,
            }
            break

        if final_record is None:
            raise RuntimeError(f"no accepted record for item {item_id}")

        append_raw(model, final_record)
        accepted[item_key] = {
            "item_id": item_id,
            "item_text_hash": text_hash,
            "prompt_hash": prompt_hash,
            "model_id": model,
            "parsed_score": final_record["parsed_score"],
            "raw_response_content_hash": final_record["raw_response_content_hash"],
            "attempt_count": final_record["attempt_count"],
            "request_timestamp_utc": final_record["request_timestamp_utc"],
        }
        save_accepted(model, accepted)

    # completeness
    if not dry_run:
        if len(accepted) != 166:
            missing = sorted(set(items["item_id"].astype(str)) - set(accepted))
            stats["missing"] = len(missing)
            raise RuntimeError(f"incomplete scores for {model}: missing={missing[:10]}")
        scores = [float(v["parsed_score"]) for v in accepted.values()]
        ids = [int(v["item_id"]) for v in accepted.values()]
        if len(ids) != len(set(ids)):
            stats["duplicates"] = len(ids) - len(set(ids))
            raise RuntimeError("duplicate item scores")
        stats["accepted"] = 166
        stats["missing"] = 0
        stats["duplicates"] = 0
        stats["min_score"] = min(scores)
        stats["max_score"] = max(scores)
        stats["unique_scores"] = len(set(scores))
        stats["n_exact_0"] = sum(1 for s in scores if s == 0.0)
        stats["n_exact_1"] = sum(1 for s in scores if s == 1.0)
        stats["n_lt_0"] = sum(1 for s in scores if s < 0.0)
        stats["n_gt_1"] = sum(1 for s in scores if s > 1.0)
    stats["completed_utc"] = utc_now()
    return stats, accepted


def build_canonical_csv(
    *,
    items: pd.DataFrame,
    accepted_mini: dict[str, dict],
    accepted_54: dict[str, dict],
    prompt_hash: str,
) -> pd.DataFrame:
    rows = []
    for _, row in items.iterrows():
        iid = str(int(row["item_id"]))
        m = accepted_mini[iid]
        g = accepted_54[iid]
        assert m["item_text_hash"] == g["item_text_hash"] == row["scoring_text_hash"]
        rows.append(
            {
                "dataset": "dbe_kt22",
                "item_id": int(row["item_id"]),
                "item_text_hash": row["scoring_text_hash"],
                "gpt4omini_score": float(m["parsed_score"]),
                "gpt54_score": float(g["parsed_score"]),
                "gpt4omini_raw_hash": m["raw_response_content_hash"],
                "gpt54_raw_hash": g["raw_response_content_hash"],
                "gpt4omini_attempts": int(m["attempt_count"]),
                "gpt54_attempts": int(g["attempt_count"]),
                "prompt_hash": prompt_hash,
                "protocol_version": PROTOCOL_VERSION,
            }
        )
    df = pd.DataFrame(rows).sort_values("item_id", kind="mergesort")
    # forbidden columns
    for bad in (
        "learner_error",
        "expert_difficulty",
        "answer_state",
        "is_correct",
        "empirical_difficulty",
    ):
        assert bad not in df.columns
    return df


def write_manifest_and_summary(
    *,
    meta: dict[str, Any],
    stats_mini: dict[str, Any],
    stats_54: dict[str, Any],
    score_csv: Path,
    started: str,
) -> None:
    raw_hashes = {
        "gpt-4o-mini": sha256_file(RAW_MINI) if RAW_MINI.exists() else None,
        "gpt-5.4": sha256_file(RAW_54) if RAW_54.exists() else None,
    }
    manifest = {
        "protocol_tag": PROTOCOL_TAG,
        "protocol_commit": PROTOCOL_COMMIT,
        "dataset": "DBE-KT22",
        "items_expected": 166,
        "models": list(ALLOWED_MODELS),
        "expected_scores": 332,
        "accepted_scores": int(stats_mini.get("accepted", 0)) + int(stats_54.get("accepted", 0)),
        "prompt_hash": meta["prompt_hash"],
        "item_universe_hash": meta["item_universe_hash"],
        "score_csv_sha256": sha256_file(score_csv),
        "raw_model_artifact_hashes": raw_hashes,
        "items_parquet_sha256": sha256_file(ITEMS_PATH),
        "prompt_file_sha256": sha256_file(PROMPT_PATH),
        "run_started_utc": started,
        "run_completed_utc": utc_now(),
        "hypothesis_tests_run": False,
        "learner_outcome_join_performed": False,
        "kt_models_trained": False,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    summary = {
        "phase": "TLT3D_P2",
        "protocol_tag": PROTOCOL_TAG,
        "protocol_commit": PROTOCOL_COMMIT,
        "items": 166,
        "models": {
            "gpt-4o-mini": {
                "accepted": stats_mini.get("accepted", 0),
                "missing": stats_mini.get("missing", 0),
                "duplicates": stats_mini.get("duplicates", 0),
                "transport_retries": stats_mini.get("transport_retries", 0),
                "invalid_outputs": stats_mini.get("invalid_model_outputs", 0),
                "api_requests_attempted": stats_mini.get("api_requests_attempted", 0),
                "min_score": stats_mini.get("min_score"),
                "max_score": stats_mini.get("max_score"),
                "unique_scores": stats_mini.get("unique_scores", 0),
                "n_exact_0": stats_mini.get("n_exact_0"),
                "n_exact_1": stats_mini.get("n_exact_1"),
                "input_tokens": stats_mini.get("input_tokens", 0),
                "output_tokens": stats_mini.get("output_tokens", 0),
                "raw_artifact_hash": raw_hashes["gpt-4o-mini"],
            },
            "gpt-5.4": {
                "accepted": stats_54.get("accepted", 0),
                "missing": stats_54.get("missing", 0),
                "duplicates": stats_54.get("duplicates", 0),
                "transport_retries": stats_54.get("transport_retries", 0),
                "invalid_outputs": stats_54.get("invalid_model_outputs", 0),
                "api_requests_attempted": stats_54.get("api_requests_attempted", 0),
                "min_score": stats_54.get("min_score"),
                "max_score": stats_54.get("max_score"),
                "unique_scores": stats_54.get("unique_scores", 0),
                "n_exact_0": stats_54.get("n_exact_0"),
                "n_exact_1": stats_54.get("n_exact_1"),
                "input_tokens": stats_54.get("input_tokens", 0),
                "output_tokens": stats_54.get("output_tokens", 0),
                "raw_artifact_hash": raw_hashes["gpt-5.4"],
            },
        },
        "expected_total_scores": 332,
        "accepted_total_scores": manifest["accepted_scores"],
        "protocol_deviations": [],
        "learner_outcome_join_performed": False,
        "hypothesis_tests_run": False,
        "kt_models_trained": False,
        "score_csv_sha256": manifest["score_csv_sha256"],
        "run_started_utc": started,
        "run_completed_utc": manifest["run_completed_utc"],
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def assert_no_learner_imports_in_source() -> None:
    src = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if "kt_interactions" in mod or "rq2_first_observed" in mod:
                raise RuntimeError("scorer source imports learner interaction modules")
        if isinstance(node, ast.Import):
            for a in node.names:
                if "kt_interactions" in a.name:
                    raise RuntimeError("scorer source imports learner interaction modules")
    # Runtime load path must be canonical items only
    if "ITEMS_PATH" not in src or "tlt3d_canonical_items.parquet" not in src:
        raise RuntimeError("scorer missing canonical items path")
    # Disallow constructing learner artifact paths in this module
    banned_literals = [
        "tlt3d_" + "kt_interactions.parquet",
        "tlt3d_" + "rq2_first_observed.parquet",
        "tlt3d_" + "learner_splits.csv",
        "Trans" + "action.csv",
    ]
    for banned in banned_literals:
        if banned in src:
            raise RuntimeError(f"scorer references learner artifact literal: {banned}")


def main() -> int:
    parser = argparse.ArgumentParser(description="TLT3D Phase2 DBE LLM confirmatory scoring")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--model", choices=list(ALLOWED_MODELS) + ["all"], default=None)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    resume = not args.no_resume

    assert_no_learner_imports_in_source()
    meta = assert_protocol_identity()
    items = load_scoring_items(meta["universe"])
    assert len(items) == 166
    cfg = FullScoringConfig.load()

    if args.validate_only:
        print(
            json.dumps(
                {
                    "ok": True,
                    "validate_only": True,
                    "items": 166,
                    "prompt_hash": meta["prompt_hash"],
                    "item_universe_hash": meta["item_universe_hash"],
                    "models": list(ALLOWED_MODELS),
                },
                indent=2,
            )
        )
        return 0

    if args.model is None:
        print("ERROR: --model required unless --validate-only", file=sys.stderr)
        return 2

    models = list(ALLOWED_MODELS) if args.model == "all" else [args.model]
    started = utc_now()
    ART.mkdir(parents=True, exist_ok=True)

    stats_by_model: dict[str, dict] = {}
    accepted_by_model: dict[str, dict] = {}
    for model in models:
        print(f"Scoring {model} ...", flush=True)
        stats, accepted = score_model(
            model=model,
            items=items,
            system=meta["system_message"],
            prompt_hash=meta["prompt_hash"],
            cfg=cfg,
            resume=resume,
            dry_run=False,
        )
        stats_by_model[model] = stats
        accepted_by_model[model] = accepted
        print(
            f"  done accepted={stats['accepted']} attempts={stats['api_requests_attempted']} "
            f"retries={stats['transport_retries']}",
            flush=True,
        )

    # If both present, write canonical table
    mini = load_accepted("gpt-4o-mini")
    g54 = load_accepted("gpt-5.4")
    if len(mini) == 166 and len(g54) == 166:
        df = build_canonical_csv(
            items=items,
            accepted_mini=mini,
            accepted_54=g54,
            prompt_hash=meta["prompt_hash"],
        )
        df.to_csv(SCORE_CSV, index=False)
        # fill stats for both even if one was skipped this invocation
        if "gpt-4o-mini" not in stats_by_model:
            scores = [float(v["parsed_score"]) for v in mini.values()]
            stats_by_model["gpt-4o-mini"] = {
                "accepted": 166,
                "missing": 0,
                "duplicates": 0,
                "transport_retries": 0,
                "invalid_model_outputs": 0,
                "api_requests_attempted": 0,
                "min_score": min(scores),
                "max_score": max(scores),
                "unique_scores": len(set(scores)),
                "n_exact_0": sum(s == 0.0 for s in scores),
                "n_exact_1": sum(s == 1.0 for s in scores),
                "input_tokens": 0,
                "output_tokens": 0,
            }
        if "gpt-5.4" not in stats_by_model:
            scores = [float(v["parsed_score"]) for v in g54.values()]
            stats_by_model["gpt-5.4"] = {
                "accepted": 166,
                "missing": 0,
                "duplicates": 0,
                "transport_retries": 0,
                "invalid_model_outputs": 0,
                "api_requests_attempted": 0,
                "min_score": min(scores),
                "max_score": max(scores),
                "unique_scores": len(set(scores)),
                "n_exact_0": sum(s == 0.0 for s in scores),
                "n_exact_1": sum(s == 1.0 for s in scores),
                "input_tokens": 0,
                "output_tokens": 0,
            }
        write_manifest_and_summary(
            meta=meta,
            stats_mini=stats_by_model["gpt-4o-mini"],
            stats_54=stats_by_model["gpt-5.4"],
            score_csv=SCORE_CSV,
            started=started,
        )
        print(f"Wrote {SCORE_CSV}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        code = 3
        msg = str(exc)
        if msg.startswith("API_CREDENTIAL_BLOCKED"):
            code = 10
        elif msg.startswith("PHASE2_BLOCKED_PROTOCOL_DRIFT"):
            code = 11
        elif msg.startswith("MODEL_UNAVAILABLE"):
            code = 12
        elif msg.startswith("INVALID_MODEL_OUTPUT"):
            code = 13
        raise SystemExit(code)
