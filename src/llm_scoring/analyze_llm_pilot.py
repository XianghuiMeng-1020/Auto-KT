#!/usr/bin/env python3
"""Analyze LLM pilot results: parse, stability, cost, content audits, gates."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "llm_scoring"))

from llm_pilot_common import (  # noqa: E402
    CACHE_DIR,
    JOURNAL_ROOT,
    MANIFEST_DIR,
    PILOT_DIR,
    PROCESSED_ROOT,
    PilotConfig,
    git_branch,
    git_commit,
    openai_credentials,
    protocol_file_hash,
    protocol_prompt_hash,
    sha256_file,
    utc_now,
)

TABLE_DIR = ROOT / "results"
REPORT_DIR = JOURNAL_ROOT / "reports"
DATASETS = ("xes3g5m", "junyi")


def load_request_log() -> pd.DataFrame:
    index_path = CACHE_DIR / "cache_index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        df = pd.DataFrame(index.values())
        if len(df):
            return df
    path = MANIFEST_DIR / "request_log.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


def expected_request_count(cfg: PilotConfig) -> int:
    calls_per_item = 1 + cfg.stability_replicates
    return cfg.pilot_items_per_dataset * len(DATASETS) * len(cfg.models) * calls_per_item


def junyi_content_flags(row: pd.Series) -> list[str]:
    flags = []
    ctype = str(row.get("item_content_type", ""))
    stem = str(row.get("item_text_clean", ""))
    if ctype == "html_title_fallback":
        flags.append("title_fallback")
        if len(stem.strip()) < 15:
            flags.append("insufficient_stem_text")
    elif ctype == "html_question_div":
        flags.append("question_div")
    if bool(row.get("has_image_dependency", False)) or bool(row.get("graphie_only_no_question_text", False)):
        flags.append("image_or_graphie_dependency")
        if "graphie" in stem.lower() and len(stem.strip()) < 40:
            flags.append("graphie_without_local_text")
    if "{{" in stem or "[[[" in stem:
        flags.append("unresolved_template_variables")
    elif bool(row.get("has_dynamic_template", False)) and len(stem.strip()) < 20:
        flags.append("dynamic_var_template")
    if len(stem.strip()) < 8:
        flags.append("insufficient_stem_text")
    return flags


def xes_content_flags(row: pd.Series) -> list[str]:
    flags = []
    stem = str(row.get("item_text_clean", ""))
    if len(stem.strip()) < 10:
        flags.append("truncated_stem")
    if stem != stem.encode("utf-8", errors="ignore").decode("utf-8"):
        flags.append("encoding_corruption")
    if "\\" in stem and "$" not in stem:
        flags.append("malformed_expression")
    fmt = str(row.get("item_format", ""))
    if fmt == "multiple_choice" and "A." not in stem and "A、" not in stem:
        flags.append("missing_option_context")
    if len(stem) > 1200:
        flags.append("prompt_length_risk")
    return flags


def build_content_limitation_tables(cfg: PilotConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    junyi_rows, xes_rows = [], []
    for ds, builder in (("junyi", junyi_content_flags), ("xes3g5m", xes_content_flags)):
        pilot = pd.read_parquet(PILOT_DIR / f"{ds}_pilot_items.parquet")
        for _, row in pilot.iterrows():
            flags = builder(row)
            if not flags:
                continue
            rec = {
                "dataset": ds,
                "item_id_hash": row["item_id_hash"],
                "source_content_hash": row["source_content_hash"],
                "limitation_categories": "|".join(flags),
                "content_limited": True,
            }
            if ds == "junyi":
                junyi_rows.append(rec)
            else:
                xes_rows.append(rec)
    junyi_df = pd.DataFrame(junyi_rows) if junyi_rows else pd.DataFrame(
        columns=["dataset", "item_id_hash", "source_content_hash", "limitation_categories", "content_limited"]
    )
    xes_df = pd.DataFrame(xes_rows) if xes_rows else pd.DataFrame(
        columns=["dataset", "item_id_hash", "source_content_hash", "limitation_categories", "content_limited"]
    )
    return xes_df, junyi_df


def parse_reliability_table(log: pd.DataFrame, cfg: PilotConfig) -> pd.DataFrame:
    rows = []
    for ds in DATASETS:
        for model in cfg.models:
            sub = log[(log["dataset"] == ds) & (log["model"] == model)] if len(log) else pd.DataFrame()
            expected = cfg.pilot_items_per_dataset * (1 + cfg.stability_replicates)
            completed = len(sub)
            if completed == 0:
                rows.append({
                    "dataset": ds,
                    "model": model,
                    "expected_requests": expected,
                    "completed_requests": 0,
                    "first_pass_valid_rate": np.nan,
                    "valid_after_retry_rate": np.nan,
                    "missing_field_rate": np.nan,
                    "out_of_range_rate": np.nan,
                    "non_finite_rate": np.nan,
                    "api_failure_rate": 1.0,
                    "retry_rate": np.nan,
                })
                continue
            valid = sub["parse_valid"].fillna(False) if "parse_valid" in sub.columns else pd.Series(dtype=bool)
            if len(sub) and "scalar_difficulty" in sub.columns:
                scalar = pd.to_numeric(sub["scalar_difficulty"], errors="coerce")
            else:
                scalar = pd.Series(dtype=float)
            oob = int(((scalar < 0) | (scalar > 1)).sum()) if len(scalar) else 0
            rows.append({
                "dataset": ds,
                "model": model,
                "expected_requests": expected,
                "completed_requests": completed,
                "first_pass_valid_rate": float(valid.mean()),
                "valid_after_retry_rate": float(valid.mean()),
                "missing_field_rate": float((~valid).mean()),
                "out_of_range_rate": float(oob / max(1, len(sub))),
                "non_finite_rate": float(scalar.isna().mean()),
                "api_failure_rate": float((~sub.get("ok", valid)).mean()) if "ok" in sub else 0.0,
                "retry_rate": float((sub.get("attempts", 1) > 1).mean()) if "attempts" in sub else 0.0,
            })
    return pd.DataFrame(rows)


def stability_table(log: pd.DataFrame, cfg: PilotConfig) -> pd.DataFrame:
    rows = []
    if log.empty or "scalar_difficulty" not in log.columns:
        return pd.DataFrame(columns=[
            "dataset", "model", "dimension", "icc", "mean_pairwise_spearman",
            "mad", "median_ad", "max_within_item_range",
            "prop_range_gt_0.10", "prop_range_gt_0.20", "n_items",
        ])
    stab = log[log["run_kind"].astype(str).str.startswith("stability")]
    for (ds, model), grp in stab.groupby(["dataset", "model"]):
        pivot = grp.pivot_table(
            index="item_id_hash", columns="run_kind", values="scalar_difficulty", aggfunc="first"
        )
        if pivot.shape[1] < 2:
            continue
        arr = pivot.to_numpy(dtype=float)
        ranges = np.nanmax(arr, axis=1) - np.nanmin(arr, axis=1)
        icc = np.nan
        if arr.shape[1] >= 2:
            try:
                icc = float(np.corrcoef(arr.T).mean())
            except Exception:
                icc = np.nan
        spears = []
        cols = list(pivot.columns)
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                a, b = pivot[cols[i]], pivot[cols[j]]
                mask = a.notna() & b.notna()
                if mask.sum() > 2:
                    spears.append(stats.spearmanr(a[mask], b[mask]).correlation)
        rows.append({
            "dataset": ds,
            "model": model,
            "dimension": "scalar_difficulty",
            "icc": icc,
            "mean_pairwise_spearman": float(np.nanmean(spears)) if spears else np.nan,
            "mad": float(np.nanmean(np.abs(arr - np.nanmean(arr, axis=1, keepdims=True)))),
            "median_ad": float(np.nanmedian(np.abs(arr - np.nanmedian(arr, axis=1, keepdims=True)))),
            "max_within_item_range": float(np.nanmax(ranges)),
            "prop_range_gt_0.10": float(np.nanmean(ranges > 0.10)),
            "prop_range_gt_0.20": float(np.nanmean(ranges > 0.20)),
            "n_items": int(len(pivot)),
        })
    return pd.DataFrame(rows)


def model_agreement_table(log: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if log.empty:
        return pd.DataFrame(columns=[
            "dataset", "pearson_r", "spearman_r", "mean_signed_diff",
            "mean_abs_diff", "n_items",
        ])
    det = log[log["run_kind"] == "deterministic"]
    for ds, grp in det.groupby("dataset"):
        models = sorted(grp["model"].unique())
        if len(models) < 2:
            continue
        a = grp[grp["model"] == models[0]].set_index("item_id_hash")["scalar_difficulty"]
        b = grp[grp["model"] == models[1]].set_index("item_id_hash")["scalar_difficulty"]
        joined = pd.concat([a, b], axis=1, keys=["m1", "m2"]).dropna()
        if len(joined) < 3:
            continue
        rows.append({
            "dataset": ds,
            "model_a": models[0],
            "model_b": models[1],
            "pearson_r": float(joined["m1"].corr(joined["m2"])),
            "spearman_r": float(stats.spearmanr(joined["m1"], joined["m2"]).correlation),
            "mean_signed_diff": float((joined["m1"] - joined["m2"]).mean()),
            "mean_abs_diff": float((joined["m1"] - joined["m2"]).abs().mean()),
            "n_items": len(joined),
        })
    return pd.DataFrame(rows)


def distribution_table(log: pd.DataFrame, cfg: PilotConfig) -> pd.DataFrame:
    rows = []
    det = log[log["run_kind"] == "deterministic"] if len(log) else pd.DataFrame()
    for ds in DATASETS:
        for model in cfg.models:
            sub = det[(det["dataset"] == ds) & (det["model"] == model)] if len(det) else pd.DataFrame()
            vals = pd.to_numeric(sub.get("scalar_difficulty"), errors="coerce").dropna() if len(sub) else pd.Series(dtype=float)
            if vals.empty:
                rows.append({"dataset": ds, "model": model, "n": 0})
                continue
            rows.append({
                "dataset": ds,
                "model": model,
                "n": len(vals),
                "mean": float(vals.mean()),
                "std": float(vals.std()),
                "median": float(vals.median()),
                "iqr": float(vals.quantile(0.75) - vals.quantile(0.25)),
                "min": float(vals.min()),
                "max": float(vals.max()),
                "floor_concentration": float((vals <= 0.05).mean()),
                "ceiling_concentration": float((vals >= 0.95).mean()),
                "n_unique": int(vals.nunique()),
                "prop_mode_value": float((vals == vals.mode().iloc[0]).mean()) if len(vals) else np.nan,
            })
    return pd.DataFrame(rows)


def manual_review_table(cfg: PilotConfig, junyi_lim: pd.DataFrame, xes_lim: pd.DataFrame) -> pd.DataFrame:
    rows = []
    limited = {
        r["item_id_hash"]: r["limitation_categories"]
        for r in pd.concat([junyi_lim, xes_lim], ignore_index=True).to_dict("records")
    }
    for ds in DATASETS:
        pilot = pd.read_parquet(PILOT_DIR / f"{ds}_pilot_items.parquet")
        pilot = pilot.sort_values("item_id_hash").head(cfg.manual_review_per_dataset)
        for model in cfg.models:
            for _, row in pilot.iterrows():
                iid = row["item_id_hash"]
                if iid in limited:
                    label = "CONTENT_LIMITED"
                elif len(str(row.get("item_text_clean", ""))) < 10:
                    label = "CONTENT_LIMITED"
                else:
                    label = "PENDING_LLM_REVIEW"
                rows.append({
                    "dataset": ds,
                    "model": model,
                    "item_id_hash": iid,
                    "source_content_hash": row["source_content_hash"],
                    "review_label": label,
                    "reviewer": "deterministic_heuristic_v1",
                })
    return pd.DataFrame(rows)


def cost_table(log: pd.DataFrame, cfg: PilotConfig) -> pd.DataFrame:
    rows = []
    run_manifest = {}
    mp = MANIFEST_DIR / "run_manifest.json"
    if mp.exists():
        run_manifest = json.loads(mp.read_text(encoding="utf-8")).get("stats", {})
    for ds in DATASETS:
        for model in cfg.models:
            sub = log[(log["dataset"] == ds) & (log["model"] == model)] if len(log) else pd.DataFrame()
            in_tok = int(sub["input_tokens"].fillna(0).sum()) if "input_tokens" in sub else 0
            out_tok = int(sub["output_tokens"].fillna(0).sum()) if "output_tokens" in sub else 0
            paid = int(run_manifest.get("paid_calls", 0)) // max(1, len(DATASETS) * len(cfg.models))
            cost = (
                in_tok / 1000 * cfg.cost_per_1k_input_tokens_usd
                + out_tok / 1000 * cfg.cost_per_1k_output_tokens_usd
            )
            rows.append({
                "dataset": ds,
                "model": model,
                "input_tokens": in_tok,
                "output_tokens": out_tok,
                "request_count": len(sub),
                "retry_count": int((sub.get("attempts", 1) > 1).sum()) if len(sub) else 0,
                "wall_time_s": run_manifest.get("wall_time_s", 0),
                "estimated_cost_usd": round(cost, 4),
                "cost_per_scored_item": round(cost / max(1, len(sub)), 6),
            })
    return pd.DataFrame(rows)


def projected_full_cost(cfg: PilotConfig) -> str:
    items = cfg.frozen_eligible_counts["xes3g5m"] + cfg.frozen_eligible_counts["junyi"]
    calls = items * (1 + cfg.stability_replicates) * len(cfg.models)
    in_tok = calls * cfg.avg_input_tokens_per_request
    out_tok = calls * cfg.avg_output_tokens_per_request
    cost = (
        in_tok / 1000 * cfg.cost_per_1k_input_tokens_usd
        + out_tok / 1000 * cfg.cost_per_1k_output_tokens_usd
    )
    lines = [
        "# Full LLM Scoring Cost Projection",
        "",
        f"**Generated:** {utc_now()}",
        "",
        "| Parameter | Value |",
        "|---|---:|",
        f"| XES eligible items | {cfg.frozen_eligible_counts['xes3g5m']} |",
        f"| Junyi eligible items | {cfg.frozen_eligible_counts['junyi']} |",
        f"| Models | {', '.join(cfg.models)} |",
        f"| Calls per item per model | {1 + cfg.stability_replicates} |",
        f"| Total projected calls | {calls} |",
        f"| Projected input tokens | {in_tok:,} |",
        f"| Projected output tokens | {out_tok:,} |",
        f"| Projected cost (USD) | ${cost:.2f} |",
        f"| Budget ceiling | ${cfg.full_scoring_budget_usd:.2f} |",
        "",
        "**Not authorised for full run from this projection alone.**",
    ]
    path = REPORT_DIR / "FULL_LLM_SCORING_COST_PROJECTION.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return "LLM_PILOT_COST_BLOCKED" if cost > cfg.full_scoring_budget_usd else "within_budget"


def _junyi_severely_limited(row: pd.Series) -> bool:
    flags = set(junyi_content_flags(row)) - {"question_div"}
    return bool(flags)


def eligibility_amendment(junyi_lim: pd.DataFrame, cfg: PilotConfig) -> dict:
    total = cfg.frozen_eligible_counts["junyi"]
    all_items = pd.read_parquet(PROCESSED_ROOT / "junyi" / "items.parquet")
    affected = sum(1 for _, row in all_items.iterrows() if _junyi_severely_limited(row))
    return {
        "proposed_rule": "Exclude items whose essential stem requires unresolved dynamic variables, unseen graphie/image, or title-only fallback with <15 visible characters for text-only LLM scoring.",
        "affected_item_count": affected,
        "affected_pct": round(100 * affected / total, 2),
        "deterministic_rule": True,
        "kt_eligibility_separate": True,
        "adopted": False,
    }


def decide_status(
    cfg: PilotConfig,
    parse_df: pd.DataFrame,
    log: pd.DataFrame,
    cost_projection_flag: str,
    api_key_present: bool,
    run_manifest: dict | None = None,
) -> tuple[str, bool]:
    run_manifest = run_manifest or {}
    model_errors = (run_manifest.get("stats") or {}).get("model_errors", {})
    if not api_key_present:
        return "LLM_PILOT_API_UNAVAILABLE", False
    if model_errors.get("gpt-5.4"):
        gpt54_status = "LLM_PILOT_API_UNAVAILABLE"
    else:
        gpt54_status = None
    if not len(log):
        return gpt54_status or "LLM_PILOT_API_UNAVAILABLE", False
    if cost_projection_flag == "LLM_PILOT_COST_BLOCKED":
        return "LLM_PILOT_COST_BLOCKED", False
    completed = len(log[log.get("parse_valid", False) == True]) if "parse_valid" in log.columns else len(log)  # noqa: E712
    expected_one_model = cfg.pilot_items_per_dataset * len(("xes3g5m", "junyi")) * (1 + cfg.stability_replicates)
    if completed < expected_one_model:
        if completed > 0:
            return "LLM_PILOT_CONDITIONAL", False
        return gpt54_status or "LLM_PILOT_API_UNAVAILABLE", False
    parse_ok = True
    for _, row in parse_df.iterrows():
        rate = row.get("valid_after_retry_rate")
        if pd.isna(rate) or rate < cfg.parse_success_threshold:
            parse_ok = False
    if not parse_ok:
        return "LLM_PILOT_BLOCKED", False
    if gpt54_status:
        return "LLM_PILOT_CONDITIONAL", False
    # Technical gates passed; content sufficiency requires manual review gate script.
    return "LLM_PILOT_CONDITIONAL", False


def write_phase_report(
    status: str,
    full_ready: bool,
    cfg: PilotConfig,
    parse_df: pd.DataFrame,
    stab_df: pd.DataFrame,
    agreement_df: pd.DataFrame,
    amendment: dict,
    log: pd.DataFrame,
    cost_flag: str,
) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / "PHASE_D1_LLM_PILOT_REPORT.md"
    lines = [
        "# Phase D1 — LLM Pilot Report",
        "",
        f"**Status:** `{status}`",
        f"**Started as:** `LLM_PILOT_RUNNING`",
        f"**Generated:** {utc_now()}",
        f"**Branch:** {git_branch()}",
        f"**Commit:** {git_commit()}",
        f"**Protocol prompt hash:** `{protocol_prompt_hash()}`",
        f"**Protocol file hash:** `{protocol_file_hash()}`",
        f"**full_llm_scoring_ready:** `{full_ready}`",
        "",
        "## Pilot scope",
        "",
        f"- Items per dataset: {cfg.pilot_items_per_dataset}",
        f"- Models: {', '.join(cfg.models)}",
        f"- Expected requests: {expected_request_count(cfg)}",
        f"- Completed records: {len(log)}",
        "",
        "## Gates",
        "",
        "| Gate | Result |",
        "|---|---|",
        f"| A Prompt integrity | frozen hash verified in preflight |",
        f"| B API/cache | {'API unavailable' if log.empty else 'partial/complete'} |",
        f"| C Parse reliability | see LLM_PILOT_PARSE_RELIABILITY.csv |",
        f"| D Stability | descriptive (no frozen ICC threshold) |",
        f"| E Content sufficiency | see content limitation tables |",
        f"| F Cost feasibility | {cost_flag} |",
        "",
        "## Proposed eligibility amendment (not adopted)",
        "",
        json.dumps(amendment, indent=2),
        "",
        "## Notes",
        "",
        "- Frozen protocol uses scalar difficulty output (not multi-field JSON).",
        "- Stability thresholds not frozen → LLM_PILOT_CONDITIONAL when API succeeds.",
        "- No student outcomes inspected in this phase.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    cfg = PilotConfig.load()
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    run_manifest = {}
    mp = MANIFEST_DIR / "run_manifest.json"
    if mp.exists():
        run_manifest = json.loads(mp.read_text(encoding="utf-8"))

    log = load_request_log()
    xes_lim, junyi_lim = build_content_limitation_tables(cfg)
    xes_lim.to_csv(TABLE_DIR / "LLM_PILOT_XES_CONTENT_LIMITATIONS.csv", index=False)
    junyi_lim.to_csv(TABLE_DIR / "LLM_PILOT_JUNYI_CONTENT_LIMITATIONS.csv", index=False)

    parse_df = parse_reliability_table(log, cfg)
    stab_df = stability_table(log, cfg)
    agreement_df = model_agreement_table(log)
    dist_df = distribution_table(log, cfg)
    cost_df = cost_table(log, cfg)
    review_df = manual_review_table(cfg, junyi_lim, xes_lim)
    # Placeholder only; run run_manual_content_review.py for final review labels.

    parse_df.to_csv(TABLE_DIR / "LLM_PILOT_PARSE_RELIABILITY.csv", index=False)
    stab_df.to_csv(TABLE_DIR / "LLM_PILOT_STABILITY.csv", index=False)
    agreement_df.to_csv(TABLE_DIR / "LLM_PILOT_MODEL_AGREEMENT.csv", index=False)
    dist_df.to_csv(TABLE_DIR / "LLM_PILOT_SCORE_DISTRIBUTIONS.csv", index=False)
    cost_df.to_csv(TABLE_DIR / "LLM_PILOT_COSTS.csv", index=False)
    review_df.to_csv(TABLE_DIR / "LLM_PILOT_MANUAL_REVIEW.csv", index=False)

    cost_flag = projected_full_cost(cfg)
    amendment = eligibility_amendment(junyi_lim, cfg)
    api_key, _ = openai_credentials()
    status, full_ready = decide_status(cfg, parse_df, log, cost_flag, bool(api_key), run_manifest)

    write_phase_report(status, full_ready, cfg, parse_df, stab_df, agreement_df, amendment, log, cost_flag)

    manifest_path = ROOT / "data_manifests" / "_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["phase_stop_code"] = (
            "LLM_PILOT_API_UNAVAILABLE_PENDING_KEY"
            if status == "LLM_PILOT_API_UNAVAILABLE"
            else status
        )
        manifest.setdefault("gate_status", {})["full_llm_scoring_ready"] = full_ready
        manifest.setdefault("gate_status", {})["llm_pilot_status"] = status
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
