"""Cross-dataset static content sufficiency for LLM difficulty scoring (Amendment 009).

Uses only frozen item text and extraction metadata. No LLM outputs, student
outcomes, or response counts in inclusion decisions.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
AMENDMENT_009_PATH = ROOT / "protocol" / "amendments" / "AMENDMENT_009_CONTENT_SUFFICIENCY_AND_ITEM_UNIVERSES.md"

CONTENT_SUFFICIENCY_RULE_VERSION = "content_sufficiency_v1"

PASS_REASON = "PASS_TEXT_COMPLETE"

PRIMARY_EXCLUSION_REASONS = frozenset({
  PASS_REASON,
  "EXCLUDE_MISSING_STEM",
  "EXCLUDE_UNRESOLVED_TEMPLATE",
  "EXCLUDE_REQUIRED_IMAGE",
  "EXCLUDE_REQUIRED_GRAPHIE",
  "EXCLUDE_REQUIRED_DIAGRAM",
  "EXCLUDE_REQUIRED_TABLE",
  "EXCLUDE_MISSING_OPTIONS",
  "EXCLUDE_TITLE_ONLY",
  "EXCLUDE_TRUNCATED_CONTENT",
  "EXCLUDE_ENCODING_CORRUPTION",
  "EXCLUDE_NON_MATHEMATICS",
  "EXCLUDE_OTHER_STATIC_CONTENT_FAILURE",
})

FIGURE_REFERENCE_RE = re.compile(r"(如图|图中|下图|上图|见图)")
IMAGE_PLACEHOLDER_RE = re.compile(r"question_\d+-image_\d+")
UNRESOLVED_TEMPLATE_RE = re.compile(
  r"(\{\{|\[\[\[|\bnames_\d+|\bexpr\(|\bpow\(|[A-Z]_COEFF|===\s*0\s*\?)"
)
MCQ_OPTION_RE = re.compile(r"A[\.、．]")
NON_MATH_DOMAIN = frozenset({"non_mathematics", "unknown", ""})


def amendment_009_hash() -> str:
    if not AMENDMENT_009_PATH.exists():
        return ""
    return hashlib.sha256(AMENDMENT_009_PATH.read_bytes()).hexdigest()


def _stem(row: pd.Series) -> str:
    return str(row.get("item_text_clean", "") or "")


def _encoding_corrupted(stem: str) -> bool:
    return stem != stem.encode("utf-8", errors="ignore").decode("utf-8")


def _secondary_flags(row: pd.Series, dataset: str) -> list[str]:
    flags: list[str] = []
    stem = _stem(row)
    if dataset == "junyi":
        if str(row.get("item_content_type", "")) == "html_question_div":
            flags.append("question_div")
        if bool(row.get("has_dynamic_template", False)):
            flags.append("dynamic_template")
        if bool(row.get("has_image_dependency", False)):
            flags.append("image_dependency_flag")
        if bool(row.get("graphie_only_no_question_text", False)):
            flags.append("graphie_only")
    if dataset == "xes3g5m":
        if IMAGE_PLACEHOLDER_RE.search(stem):
            flags.append("image_placeholder")
        if FIGURE_REFERENCE_RE.search(stem):
            flags.append("figure_reference")
    if len(stem) > 1200:
        flags.append("prompt_length_risk")
    return flags


def classify_xes3g5m_item(row: pd.Series) -> tuple[str, list[str]]:
    """Return (primary_reason, secondary_flags)."""
    stem = _stem(row)
    secondary = _secondary_flags(row, "xes3g5m")
    stripped = stem.strip()

    if len(stripped) < 8:
        return "EXCLUDE_MISSING_STEM", secondary
    if _encoding_corrupted(stem):
        return "EXCLUDE_ENCODING_CORRUPTION", secondary
    if len(stripped) < 15:
        return "EXCLUDE_TRUNCATED_CONTENT", secondary
    if str(row.get("mathematical_domain", "")) in NON_MATH_DOMAIN:
        return "EXCLUDE_NON_MATHEMATICS", secondary

    item_format = str(row.get("item_format", ""))
    if item_format == "multiple_choice" and not MCQ_OPTION_RE.search(stem):
        return "EXCLUDE_MISSING_OPTIONS", secondary

    if IMAGE_PLACEHOLDER_RE.search(stem):
        return "EXCLUDE_REQUIRED_IMAGE", secondary
    if FIGURE_REFERENCE_RE.search(stem):
        return "EXCLUDE_REQUIRED_DIAGRAM", secondary

    if "\\" in stem and "$" not in stem:
        secondary = [*secondary, "malformed_expression"]
    if len(stem) > 1200:
        return "EXCLUDE_OTHER_STATIC_CONTENT_FAILURE", secondary

    return PASS_REASON, secondary


def classify_junyi_item(row: pd.Series) -> tuple[str, list[str]]:
    """Return (primary_reason, secondary_flags)."""
    stem = _stem(row)
    secondary = _secondary_flags(row, "junyi")
    stripped = stem.strip()
    ctype = str(row.get("item_content_type", ""))

    if len(stripped) < 8:
        return "EXCLUDE_MISSING_STEM", secondary
    if _encoding_corrupted(stem):
        return "EXCLUDE_ENCODING_CORRUPTION", secondary
    if ctype == "html_title_fallback":
        return "EXCLUDE_TITLE_ONLY", secondary
    if len(stripped) < 15:
        return "EXCLUDE_TRUNCATED_CONTENT", secondary
    if str(row.get("mathematical_domain", "")) in NON_MATH_DOMAIN:
        return "EXCLUDE_NON_MATHEMATICS", secondary

    if bool(row.get("graphie_only_no_question_text", False)):
        return "EXCLUDE_REQUIRED_GRAPHIE", secondary
    if UNRESOLVED_TEMPLATE_RE.search(stem):
        return "EXCLUDE_UNRESOLVED_TEMPLATE", secondary
    if bool(row.get("has_dynamic_template", False)) and len(stripped) < 20:
        return "EXCLUDE_UNRESOLVED_TEMPLATE", secondary
    if bool(row.get("has_image_dependency", False)):
        return "EXCLUDE_REQUIRED_IMAGE", secondary

    item_format = str(row.get("item_format", ""))
    if item_format == "multiple_choice" and not MCQ_OPTION_RE.search(stem):
        return "EXCLUDE_MISSING_OPTIONS", secondary

    return PASS_REASON, secondary


def classify_item(row: pd.Series, dataset: str) -> dict[str, Any]:
    if dataset == "xes3g5m":
        primary, secondary = classify_xes3g5m_item(row)
    elif dataset == "junyi":
        primary, secondary = classify_junyi_item(row)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    scoreable = primary == PASS_REASON
    return {
        "eligible_for_kt": True,
        "eligible_for_llm_scoring": scoreable,
        "eligible_for_shared_confirmatory": scoreable,
        "eligible_for_llm": scoreable,
        "llm_exclusion_primary_reason": primary,
        "llm_exclusion_secondary_flags": "|".join(secondary) if secondary else "",
        "exclusion_reason": None if scoreable else primary,
        "content_sufficiency_rule_version": CONTENT_SUFFICIENCY_RULE_VERSION,
    }


def apply_classification(items: pd.DataFrame, dataset: str) -> pd.DataFrame:
    out = items.copy()
    results = [classify_item(row, dataset) for _, row in out.iterrows()]
    for col in (
        "eligible_for_kt",
        "eligible_for_llm_scoring",
        "eligible_for_shared_confirmatory",
        "eligible_for_llm",
        "llm_exclusion_primary_reason",
        "llm_exclusion_secondary_flags",
        "exclusion_reason",
        "content_sufficiency_rule_version",
    ):
        out[col] = [r[col] for r in results]
    return out


def classification_audit_hash(items: pd.DataFrame) -> str:
    """Deterministic hash of scoreability decisions (item_id_hash + primary reason)."""
    payload = "\n".join(
        f"{r['item_id_hash']}|{r['llm_exclusion_primary_reason']}"
        for _, r in items.sort_values("item_id_hash", kind="mergesort").iterrows()
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
