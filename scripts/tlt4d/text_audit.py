"""Deterministic rule-based item text completeness audit (no LLM, no OCR)."""

from __future__ import annotations

import hashlib
import re
from html import unescape

IMG_RE = re.compile(
    r"(?i)(<img\b|!\[[^\]]*\]\([^)]+\)|\.(png|jpe?g|gif|svg|webp)\b|figure\s*\d|see\s+(the\s+)?(figure|diagram|image))",
)
DIAGRAM_RE = re.compile(
    r"(?i)\b(diagram|flowchart|er\s*diagram|entity[\s-]*relationship|schema\s+diagram|uml)\b",
)
TEMPLATE_RE = re.compile(
    r"(\{\{[^{}]+\}\}|\$\{[^}]+\}|<%[^%]+%>|\[\[[^\]]+\]\]|__PLACEHOLDER__|TODO_FILL|FIXME)",
)
HTML_TAG_RE = re.compile(r"<[^>]+>")
SCRIPT_STYLE_RE = re.compile(r"(?is)<(script|style)[^>]*>.*?</\1>")
WS_RE = re.compile(r"\s+")
TRUNC_RE = re.compile(r"(\.{3}|…|\bTRUNCATED\b|\b\[cut\]\b)\s*$")


def html_to_plain(text: str) -> str:
    if text is None:
        return ""
    s = str(text)
    s = SCRIPT_STYLE_RE.sub(" ", s)
    s = HTML_TAG_RE.sub(" ", s)
    s = unescape(s)
    s = WS_RE.sub(" ", s).strip()
    return s


def normalize_plain(text: str) -> str:
    if text is None:
        return ""
    return WS_RE.sub(" ", str(text)).strip()


def whitespace_token_count(text: str) -> int:
    t = text.strip()
    return 0 if not t else len(t.split())


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def classify_item_text(
    *,
    stem_raw: str | None,
    stem_html: str | None = None,
    title: str | None = None,
    choices: list[str] | None = None,
    require_choices: bool = False,
) -> dict:
    """Return frozen rule-based text completeness fields."""
    raw_present = bool(
        (stem_raw and str(stem_raw).strip())
        or (stem_html and str(stem_html).strip())
    )
    plain_from_html = html_to_plain(stem_html) if stem_html else ""
    plain_from_raw = normalize_plain(stem_raw) if stem_raw else ""
    # Prefer longer non-empty plain rendering for length metrics.
    normalized = plain_from_raw if len(plain_from_raw) >= len(plain_from_html) else plain_from_html
    title_n = normalize_plain(title)
    choices = [normalize_plain(c) for c in (choices or []) if normalize_plain(c)]
    has_choices = len(choices) > 0

    combined_for_flags = " ".join(
        [str(stem_raw or ""), str(stem_html or ""), normalized, " ".join(choices)]
    )
    has_image = bool(IMG_RE.search(combined_for_flags))
    has_diagram = bool(DIAGRAM_RE.search(combined_for_flags))
    has_template = bool(TEMPLATE_RE.search(combined_for_flags))
    truncated = bool(TRUNC_RE.search(normalized)) if normalized else False
    malformed = False
    if stem_html and ("�" in str(stem_html) or "\x00" in str(stem_html)):
        malformed = True
    if normalized and "�" in normalized:
        malformed = True

    exclusion_reason = ""
    status = "PASS_TEXT_COMPLETE"

    if not raw_present or not normalized:
        status = "EXCLUDE_MISSING_STEM"
        exclusion_reason = "missing_or_blank_stem"
    elif title_n and normalized == title_n and len(normalized) < 40 and not has_choices:
        status = "EXCLUDE_OTHER"
        exclusion_reason = "title_only_record"
    elif truncated:
        status = "EXCLUDE_TRUNCATED_CONTENT"
        exclusion_reason = "terminal_truncation_marker"
    elif has_template:
        status = "EXCLUDE_UNRESOLVED_TEMPLATE"
        exclusion_reason = "unresolved_template_placeholder"
    elif malformed:
        status = "EXCLUDE_MALFORMED_CONTENT"
        exclusion_reason = "replacement_char_or_null_byte"
    elif has_image and (
        len(normalized) < 80
        or bool(re.search(r"(?i)<img\b", str(stem_html or "")))
    ):
        # Short stem with image cue, or explicit <img>, treated as image-dependent.
        status = "EXCLUDE_REQUIRED_IMAGE"
        exclusion_reason = "image_reference_with_insufficient_standalone_text"
    elif has_diagram and len(normalized) < 120:
        status = "EXCLUDE_REQUIRED_DIAGRAM"
        exclusion_reason = "diagram_reference_with_insufficient_standalone_text"
    elif require_choices and not has_choices:
        status = "EXCLUDE_OTHER"
        exclusion_reason = "choices_essential_but_missing"

    return {
        "raw_text_present": bool(raw_present),
        "normalized_text": normalized,
        "normalized_text_from_html": plain_from_html,
        "normalized_text_length_chars": len(normalized),
        "normalized_text_length_tokens_or_whitespace_tokens": whitespace_token_count(normalized),
        "has_choices": has_choices,
        "n_choices": len(choices),
        "has_image_reference": has_image,
        "has_diagram_reference": has_diagram,
        "has_template_reference": has_template,
        "content_hash": content_hash(normalized) if normalized else "",
        "text_complete_status": status,
        "exclusion_reason": exclusion_reason,
        "html_plain_transform": "script_style_strip+tag_strip+unescape+ws_collapse"
        if stem_html
        else "ws_collapse_only",
    }
