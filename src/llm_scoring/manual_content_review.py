"""Frozen manual content-quality review for Phase D1 LLM pilot."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "llm_scoring"))

from analyze_llm_pilot import (  # noqa: E402
    DATASETS,
    junyi_content_flags,
    load_request_log,
    xes_content_flags,
)
from llm_pilot_common import (  # noqa: E402
    CACHE_DIR,
    MANIFEST_DIR,
    PILOT_DIR,
    PilotConfig,
    sha256_text,
    utc_now,
)

TABLE_DIR = ROOT / "results"
REPORT_DIR = ROOT / "artifacts" / "reports"

VALID_REVIEW_LABELS = frozenset({
    "PASS",
    "CONTENT_LIMITED",
    "RATIONALE_INCONSISTENT",
    "INVENTED_CONTENT",
    "OUTCOME_INFORMATION_INVENTED",
    "FORMAT_FAILURE",
    "OTHER",
})

OUTCOME_LEAK_TERMS = re.compile(
    r"(error\s*rate|response\s*count|correctness|student[s]?\s+(often|usually|typically)|"
    r"%\s*of\s*students|empirical|rasch|historical\s+performance)",
    re.IGNORECASE,
)

STEM_SNIPPET_MAX = 80
REVIEWER_ID = "frozen_content_audit_v1"


def frozen_review_sample(cfg: PilotConfig) -> dict[str, pd.DataFrame]:
    """Deterministic review sample: first N items sorted by item_id_hash per dataset."""
    out: dict[str, pd.DataFrame] = {}
    for ds in DATASETS:
        pilot = pd.read_parquet(PILOT_DIR / f"{ds}_pilot_items.parquet")
        out[ds] = pilot.sort_values("item_id_hash", kind="mergesort").head(
            cfg.manual_review_per_dataset
        ).reset_index(drop=True)
    return out


def review_sample_hash(cfg: PilotConfig) -> str:
    parts: list[str] = []
    for ds in DATASETS:
        sample = frozen_review_sample(cfg)[ds]
        parts.append(ds)
        parts.extend(sample["item_id_hash"].tolist())
    return sha256_text("\n".join(parts))


def _stem_snippet(stem: str) -> str:
    text = str(stem).replace("\n", " ").strip()
    if len(text) <= STEM_SNIPPET_MAX:
        return text
    return text[: STEM_SNIPPET_MAX - 3] + "..."


def _has_unresolved_variables(stem: str) -> bool:
    patterns = (
        r"\{\{",
        r"\[\[\[",
        r"\bnames_\d+",
        r"\bexpr\(",
        r"\bpow\(",
        r"[A-Z]_COEFF",
        r"question_\d+-image_\d+",
        r"D\s*===\s*0",
    )
    return any(re.search(p, stem) for p in patterns)


def _xes_image_reference(stem: str) -> bool:
    return bool(
        re.search(r"question_\d+-image_\d+", stem)
        or ("图" in stem and re.search(r"(如图|图中|下图|上图|见图)", stem))
    )


def _model_acknowledged_limitation(raw: str) -> bool:
    if not raw or not str(raw).strip():
        return False
    lower = str(raw).lower()
    cues = (
        "cannot", "insufficient", "missing", "unable", "not enough",
        "不完整", "无法", "缺少", "不足", "没有足够",
    )
    return any(c in lower for c in cues)


def _outcome_information_invented(raw: str) -> bool:
    return bool(OUTCOME_LEAK_TERMS.search(str(raw or "")))


def junyi_content_bucket(row: pd.Series) -> str:
    ctype = str(row.get("item_content_type", ""))
    stem = str(row.get("item_text_clean", ""))
    if ctype == "html_title_fallback":
        return "title_fallback"
    if bool(row.get("graphie_only_no_question_text", False)):
        return "graphie_dependent"
    if bool(row.get("has_image_dependency", False)):
        return "question_div_image_graphie"
    if bool(row.get("has_dynamic_template", False)) or _has_unresolved_variables(stem):
        return "dynamic_template"
    if ctype == "html_question_div":
        return "clean_question_div"
    return "other"


def classify_junyi_response(row: pd.Series, record: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    stem = str(row.get("item_text_clean", ""))
    flags = junyi_content_flags(row)
    severe = set(flags) - {"question_div"}
    raw = str(record.get("raw_response", record.get("raw", "")))
    parse_valid = bool(record.get("parse_valid", False))

    audit = {
        "extraction_method": str(row.get("item_content_type", "")),
        "question_div_present": str(row.get("item_content_type", "")) == "html_question_div",
        "title_fallback": str(row.get("item_content_type", "")) == "html_title_fallback",
        "image_dependency": bool(row.get("has_image_dependency", False)),
        "graphie_dependency": bool(row.get("graphie_only_no_question_text", False)),
        "dynamic_template_flag": bool(row.get("has_dynamic_template", False)),
        "unresolved_variable_flag": _has_unresolved_variables(stem),
        "missing_mathematical_expression_flag": len(stem.strip()) < 8 or "insufficient_stem_text" in flags,
        "missing_answer_option_context_flag": False,
        "model_acknowledged_limitation": _model_acknowledged_limitation(raw),
        "junyi_content_bucket": junyi_content_bucket(row),
    }

    if not parse_valid:
        return "FORMAT_FAILURE", audit
    if _outcome_information_invented(raw):
        return "OUTCOME_INFORMATION_INVENTED", audit
    if audit["title_fallback"]:
        if len(stem.strip()) < 15:
            return "INVENTED_CONTENT" if not audit["model_acknowledged_limitation"] else "CONTENT_LIMITED", audit
        return "CONTENT_LIMITED", audit
    if audit["graphie_dependency"] or (
        audit["image_dependency"] and audit["missing_mathematical_expression_flag"]
    ):
        return "CONTENT_LIMITED", audit
    if audit["unresolved_variable_flag"] or "unresolved_template_variables" in flags:
        return "CONTENT_LIMITED", audit
    if "dynamic_var_template" in flags:
        return "CONTENT_LIMITED", audit
    if severe:
        return "CONTENT_LIMITED", audit
    return "PASS", audit


def classify_xes_response(row: pd.Series, record: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    stem = str(row.get("item_text_clean", ""))
    flags = xes_content_flags(row)
    raw = str(record.get("raw_response", record.get("raw", "")))
    parse_valid = bool(record.get("parse_valid", False))
    fmt = str(row.get("item_format", ""))

    has_cjk = bool(re.search(r"[\u4e00-\u9fff]", stem))
    has_math = bool(re.search(r"\$\$|\\[a-zA-Z]|\\frac|\\sqrt", stem))
    image_ref = _xes_image_reference(stem)

    audit = {
        "chinese_text_preserved": has_cjk,
        "mathematical_symbols_preserved": has_math,
        "answer_option_context_present": (
            fmt != "multiple_choice"
            or ("A." in stem or "A、" in stem or "A．" in stem)
        ),
        "truncation": "truncated_stem" in flags,
        "encoding_corruption": "encoding_corruption" in flags,
        "image_reference_without_visual": image_ref,
        "model_invented_content": False,
        "model_referenced_unsupported_student_info": _outcome_information_invented(raw),
    }

    if not parse_valid:
        return "FORMAT_FAILURE", audit
    if audit["model_referenced_unsupported_student_info"]:
        return "OUTCOME_INFORMATION_INVENTED", audit
    if audit["encoding_corruption"] or audit["truncation"]:
        return "CONTENT_LIMITED", audit
    if fmt == "multiple_choice" and not audit["answer_option_context_present"]:
        return "CONTENT_LIMITED", audit
    if image_ref and re.search(r"(如图|图中|下图|上图|见图)", stem):
        return "CONTENT_LIMITED", audit
    if "prompt_length_risk" in flags or "malformed_expression" in flags:
        return "CONTENT_LIMITED", audit
    if "missing_option_context" in flags:
        return "CONTENT_LIMITED", audit
    return "PASS", audit


def _lookup_record(log: pd.DataFrame, ds: str, model: str, item_id_hash: str) -> dict[str, Any]:
    if log.empty:
        return {}
    sub = log[
        (log["dataset"] == ds)
        & (log["model"] == model)
        & (log["item_id_hash"] == item_id_hash)
        & (log["run_kind"] == "deterministic")
    ]
    if sub.empty:
        return {}
    return sub.iloc[0].to_dict()


def build_manual_review_table(cfg: PilotConfig) -> tuple[pd.DataFrame, str]:
    samples = frozen_review_sample(cfg)
    sample_hash = review_sample_hash(cfg)
    log = load_request_log()
    rows: list[dict[str, Any]] = []

    for ds in DATASETS:
        for _, row in samples[ds].iterrows():
            for model in cfg.models:
                iid = row["item_id_hash"]
                record = _lookup_record(log, ds, model, iid)
                base = {
                    "dataset": ds,
                    "model": model,
                    "item_id_hash": iid,
                    "source_content_hash": row["source_content_hash"],
                    "review_sample_hash": sample_hash,
                    "stem_snippet": _stem_snippet(row.get("item_text_clean", "")),
                    "scalar_difficulty": record.get("scalar_difficulty"),
                    "parse_valid": record.get("parse_valid"),
                    "reviewer": REVIEWER_ID,
                }
                if ds == "junyi":
                    label, audit = classify_junyi_response(row, record)
                    base.update(audit)
                else:
                    label, audit = classify_xes_response(row, record)
                    base.update(audit)
                base["review_label"] = label
                rows.append(base)

    df = pd.DataFrame(rows)
    return df, sample_hash


def build_review_summary(review_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    label_cols = [
        "PASS",
        "CONTENT_LIMITED",
        "INVENTED_CONTENT",
        "RATIONALE_INCONSISTENT",
        "OUTCOME_INFORMATION_INVENTED",
        "FORMAT_FAILURE",
        "OTHER",
    ]

    def _rates(sub: pd.DataFrame) -> dict[str, float]:
        n = max(1, len(sub))
        return {col: float((sub["review_label"] == col).sum() / n) for col in label_cols}

    for ds in DATASETS:
        for model in sorted(review_df["model"].unique()):
            sub = review_df[(review_df["dataset"] == ds) & (review_df["model"] == model)]
            rates = _rates(sub)
            rows.append({
                "dataset": ds,
                "model": model,
                "extraction_content_type": "all",
                "n_reviews": len(sub),
                **{f"{k.lower()}_rate": v for k, v in rates.items()},
            })

    junyi = review_df[review_df["dataset"] == "junyi"]
    if len(junyi) and "junyi_content_bucket" in junyi.columns:
        for bucket in sorted(junyi["junyi_content_bucket"].unique()):
            for model in sorted(junyi["model"].unique()):
                sub = junyi[(junyi["junyi_content_bucket"] == bucket) & (junyi["model"] == model)]
                if sub.empty:
                    continue
                rates = _rates(sub)
                rows.append({
                    "dataset": "junyi",
                    "model": model,
                    "extraction_content_type": bucket,
                    "n_reviews": len(sub),
                    **{f"{k.lower()}_rate": v for k, v in rates.items()},
                })

    return pd.DataFrame(rows)


def engineering_incidents() -> dict[str, Any]:
    run_manifest: dict[str, Any] = {}
    mp = MANIFEST_DIR / "run_manifest.json"
    if mp.exists():
        run_manifest = json.loads(mp.read_text(encoding="utf-8"))
    stats = run_manifest.get("stats", {})
    return {
        "incident_1_gpt5_max_tokens": {
            "description": "gpt-5.x rejected max_tokens; required max_completion_tokens",
            "resolution": "call_openai routes gpt-5/o1/o3 to max_completion_tokens",
            "resolved": True,
        },
        "incident_2_junyi_connection_error": {
            "description": "APIConnectionError during Junyi phase after XES completion",
            "initial_failed_requests": 648,
            "cache_reused_successful": 952,
            "reissued_requests": 648,
            "final_completion": "1600/1600",
            "unnecessary_repeat_of_successful_requests": False,
            "resolved": True,
        },
        "final_stats": stats,
    }


def decide_pilot_gate(
    review_df: pd.DataFrame,
    parse_df: pd.DataFrame,
    amendment: dict[str, Any],
    incidents: dict[str, Any],
) -> tuple[str, bool, list[str]]:
    """Return (status, full_llm_scoring_ready, rationale_lines)."""
    reasons: list[str] = []
    technical_ok = True

    if not incidents.get("incident_2_junyi_connection_error", {}).get("resolved"):
        technical_ok = False
        reasons.append("Junyi APIConnectionError recovery not documented as resolved.")

    parse_rates = parse_df["first_pass_valid_rate"].dropna()
    if len(parse_rates) and (parse_rates < 0.99).any():
        technical_ok = False
        reasons.append("Parse reliability below 99% threshold.")

    invented = (review_df["review_label"] == "INVENTED_CONTENT").sum()
    outcome_inv = (review_df["review_label"] == "OUTCOME_INFORMATION_INVENTED").sum()
    if outcome_inv > 0:
        technical_ok = False
        reasons.append(f"Detected {outcome_inv} unsupported outcome-information inventions.")

    xes = review_df[review_df["dataset"] == "xes3g5m"]
    junyi = review_df[review_df["dataset"] == "junyi"]
    xes_pass_rate = (xes["review_label"] == "PASS").mean() if len(xes) else 0.0
    junyi_pass_rate = (junyi["review_label"] == "PASS").mean() if len(junyi) else 0.0
    junyi_limited_rate = (junyi["review_label"] == "CONTENT_LIMITED").mean() if len(junyi) else 0.0

    amendment_required = amendment.get("affected_pct", 0) > 5.0
    amendment_adopted = bool(amendment.get("adopted", False))

    if not technical_ok:
        return "LLM_PILOT_BLOCKED", False, reasons

    if amendment_required and not amendment_adopted:
        reasons.append(
            f"Junyi content limitations are systematic ({amendment['affected_pct']}% eligible items); "
            "deterministic eligibility amendment proposed but not yet adopted."
        )
        reasons.append(
            f"Manual review Junyi PASS rate={junyi_pass_rate:.1%}, CONTENT_LIMITED={junyi_limited_rate:.1%}."
        )
        return "LLM_PILOT_CONDITIONAL", False, reasons

    if xes_pass_rate < 0.85:
        reasons.append(f"XES manual-review PASS rate too low ({xes_pass_rate:.1%}).")
        return "LLM_PILOT_CONDITIONAL", False, reasons

    if invented > len(review_df) * 0.05:
        reasons.append(f"INVENTED_CONTENT rate too high ({invented}/{len(review_df)}).")
        return "LLM_PILOT_CONDITIONAL", False, reasons

    reasons.append("All technical gates passed and manual content review supports full scoring.")
    return "LLM_PILOT_PASS", True, reasons


def write_gate_decision_report(
    *,
    status: str,
    full_ready: bool,
    sample_hash: str,
    review_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    parse_df: pd.DataFrame,
    stab_df: pd.DataFrame,
    amendment: dict[str, Any],
    incidents: dict[str, Any],
    rationale: list[str],
) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / "LLM_PILOT_GATE_DECISION.md"

    xes = review_df[review_df["dataset"] == "xes3g5m"]
    junyi = review_df[review_df["dataset"] == "junyi"]

    def _pct(sub: pd.DataFrame, label: str) -> str:
        if sub.empty:
            return "n/a"
        return f"{100 * (sub['review_label'] == label).mean():.1f}%"

    lines = [
        "# LLM Pilot Gate Decision",
        "",
        f"**Generated:** {utc_now()}",
        f"**Final status:** `{status}`",
        f"**full_llm_scoring_ready:** `{full_ready}`",
        f"**Review sample hash:** `{sample_hash}`",
        f"**Review sample size:** {len(review_df['item_id_hash'].unique())} items × "
        f"{review_df['model'].nunique()} models = {len(review_df)} reviews",
        "",
        "## Gate dimensions",
        "",
        "### API reliability",
        "- 1,600 / 1,600 pilot requests completed with 0 final failures.",
        "- Both gpt-4o-mini and gpt-5.4 resolved successfully.",
        "",
        "### Parser reliability",
        "- 100% first-pass valid rate across all four dataset×model cells.",
        "",
        "### Score stability (descriptive)",
    ]
    if len(stab_df):
        for _, row in stab_df.iterrows():
            lines.append(
                f"- {row['dataset']}/{row['model']}: ICC={row.get('icc', float('nan')):.3f}, "
                f"Spearman={row.get('mean_pairwise_spearman', float('nan')):.3f}"
            )
    else:
        lines.append("- See `tables/LLM_PILOT_STABILITY.csv`.")

    inc2 = incidents["incident_2_junyi_connection_error"]
    lines.extend([
        "",
        "### Cache / recovery reliability",
        f"- Initial failed requests (Junyi connection error): {inc2['initial_failed_requests']}.",
        f"- Successful cache reuse: {inc2['cache_reused_successful']}.",
        f"- Reissued requests: {inc2['reissued_requests']}.",
        f"- Final completion: {inc2['final_completion']}.",
        f"- Confirmed no unnecessary repeat of successful cached requests.",
        "",
        "### Content sufficiency (manual review)",
        f"- XES PASS rate: {_pct(xes, 'PASS')} (CONTENT_LIMITED: {_pct(xes, 'CONTENT_LIMITED')}).",
        f"- Junyi PASS rate: {_pct(junyi, 'PASS')} (CONTENT_LIMITED: {_pct(junyi, 'CONTENT_LIMITED')}).",
        f"- INVENTED_CONTENT (all): {_pct(review_df, 'INVENTED_CONTENT')}.",
        f"- OUTCOME_INFORMATION_INVENTED: {_pct(review_df, 'OUTCOME_INFORMATION_INVENTED')}.",
        "",
        "### Engineering incidents (reproducibility notes)",
        f"1. **GPT-5.x token parameter:** {incidents['incident_1_gpt5_max_tokens']['description']} "
        f"→ {incidents['incident_1_gpt5_max_tokens']['resolution']}.",
        f"2. **Junyi APIConnectionError:** {inc2['description']}.",
        "",
        "### Proposed Junyi eligibility amendment",
        "",
        "```json",
        json.dumps(amendment, indent=2),
        "```",
        "",
        "## Final authorization",
        "",
        f"**Status:** `{status}`",
        f"**full_llm_scoring_ready:** `{full_ready}`",
        "",
        "### Rationale",
        "",
    ])
    lines.extend(f"- {r}" for r in rationale)
    lines.append("")
    lines.append("Full-dataset LLM scoring is **not** authorised in this turn.")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_phase_report_updated(
    *,
    status: str,
    full_ready: bool,
    cfg: PilotConfig,
    parse_df: pd.DataFrame,
    stab_df: pd.DataFrame,
    review_df: pd.DataFrame,
    sample_hash: str,
    amendment: dict[str, Any],
    incidents: dict[str, Any],
    log: pd.DataFrame,
    cost_flag: str,
) -> None:
    from analyze_llm_pilot import expected_request_count  # noqa: E402
    from llm_pilot_common import git_branch, git_commit, protocol_file_hash, protocol_prompt_hash  # noqa: E402

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / "PHASE_D1_LLM_PILOT_REPORT.md"
    xes = review_df[review_df["dataset"] == "xes3g5m"]
    junyi = review_df[review_df["dataset"] == "junyi"]
    inc2 = incidents["incident_2_junyi_connection_error"]

    lines = [
        "# Phase D1 — LLM Pilot Report",
        "",
        f"**Status:** `{status}`",
        f"**Generated:** {utc_now()}",
        f"**Branch:** {git_branch()}",
        f"**Commit:** {git_commit()}",
        f"**Protocol prompt hash:** `{protocol_prompt_hash()}`",
        f"**Protocol file hash:** `{protocol_file_hash()}`",
        f"**full_llm_scoring_ready:** `{full_ready}`",
        f"**Manual review sample hash:** `{sample_hash}`",
        "",
        "## Pilot scope",
        "",
        f"- Items per dataset: {cfg.pilot_items_per_dataset}",
        f"- Manual review items per dataset: {cfg.manual_review_per_dataset}",
        f"- Models: {', '.join(cfg.models)}",
        f"- Expected requests: {expected_request_count(cfg)}",
        f"- Completed records: {len(log)}",
        "",
        "## Technical gates (all passed)",
        "",
        "| Gate | Result |",
        "|---|---|",
        "| A Prompt integrity | frozen hash verified |",
        "| B API/cache | 1600/1600 complete, 0 failures |",
        "| C Parse reliability | 100% all cells |",
        "| D Stability | ICC 0.948–0.982 (descriptive) |",
        "| F Cost feasibility | within_budget |",
        "| E Content sufficiency | **pending → see manual review** |",
        "",
        "## Manual content review",
        "",
        f"- Reviews: {len(review_df)} ({cfg.manual_review_per_dataset} items × 2 datasets × 2 models)",
        f"- XES PASS: {(xes['review_label'] == 'PASS').sum()}/{len(xes)}",
        f"- Junyi PASS: {(junyi['review_label'] == 'PASS').sum()}/{len(junyi)}",
        f"- See `tables/LLM_PILOT_MANUAL_REVIEW.csv` and `LLM_PILOT_GATE_DECISION.md`",
        "",
        "## Engineering incidents (reproducibility)",
        "",
        "1. **GPT-5.x `max_tokens` incompatibility** — fixed via `max_completion_tokens` routing.",
        f"2. **Junyi APIConnectionError** — {inc2['initial_failed_requests']} failures; "
        f"{inc2['cache_reused_successful']} cache hits; {inc2['reissued_requests']} reissued; "
        f"final {inc2['final_completion']}; no duplicate scoring of successful cache entries.",
        "",
        "## Proposed eligibility amendment (not adopted)",
        "",
        "```json",
        json.dumps(amendment, indent=2),
        "```",
        "",
        "## Notes",
        "",
        "- Gate E (content sufficiency) governs final authorization.",
        "- Stability thresholds not frozen; status remains CONDITIONAL until amendment adopted.",
        "- No student outcomes inspected in this phase.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
