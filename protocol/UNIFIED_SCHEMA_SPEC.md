# Unified Schema Specification — Cross-Dataset Interaction & Item Tables

**Status:** FROZEN (Amendment 008)  
**Effective:** 2026-06-30  
**Output directory (planned):** `data/processed/unified/`

---

## 1. Design goal

Produce leakage-safe, analysis-ready tables with identical column semantics for XES3G5M and Junyi Academy so that LLM scoring, Rasch estimation, KT training, and cold-start evaluation share one pipeline.

GSM8K-IRT maps to the same item schema where applicable but retains synthetic provenance flags.

---

## 2. Interaction table (`interactions_{dataset}.parquet`)

| Column | Type | Description |
|---|---|---|
| `dataset` | string | `xes3g5m` \| `junyi` |
| `student_id` | string | Anonymised student identifier |
| `item_id` | string | Canonical item ID (XES: `question_id`; Junyi: `exercise` slug) |
| `correct` | int8 | 1 = correct, 0 = incorrect |
| `timestamp` | int64 | Ordering key (XES: reconstructed from `seq`; Junyi: `time_done`) |
| `seq_idx` | int32 | Within-student sequence index (0-based, after sort) |
| `split` | string | `train` \| `val` \| `test` (from student assignment) |
| `concept_ids` | string (nullable) | Pipe-separated KC/skill IDs |
| `elapsed_ms` | int64 (nullable) | Response duration if available |

### Source mapping

| Unified column | XES3G5M source | Junyi source |
|---|---|---|
| `student_id` | `uid` | `user_id` |
| `item_id` | `questions` | `exercise` |
| `correct` | `cpts` | `correct` |
| `timestamp` | `seq` (ordinal) | `time_done` |
| `concept_ids` | `concepts` | `topic` from exercise table |

---

## 3. Item table (`items_{dataset}.parquet`)

| Column | Type | Description |
|---|---|---|
| `dataset` | string | `xes3g5m` \| `junyi` |
| `item_id` | string | Canonical item ID |
| `stem_text` | string | Machine-readable problem text for LLM |
| `stem_source` | string | `content_json` \| `html_question` \| `html_title_fallback` |
| `answer` | string (nullable) | Correct answer if available |
| `options` | string (nullable) | JSON-encoded options for MCQ |
| `item_type` | string | `fill_in` \| `multiple_choice` \| `unknown` |
| `has_latex` | bool | LaTeX/math notation present |
| `has_image_dep` | bool | Diagram/graphie dependency flagged |
| `eligible_llm` | bool | Passes item eligibility for LLM scoring |
| `concept_ids` | string (nullable) | Knowledge component tags |
| `html_sha256` | string (nullable) | Junyi only: content file hash |
| `content_url` | string (nullable) | Junyi only: download URL |

### Source mapping

| Unified column | XES3G5M source | Junyi source |
|---|---|---|
| `stem_text` | `questions.json` → `content` | HTML extraction |
| `stem_source` | `content_json` | `html_question` or `html_title_fallback` |
| `answer` | `questions.json` → `answer` | — (not in public HTML layer) |
| `options` | `questions.json` → `options` | HTML choices if present |
| `html_sha256` | — | `junyi_exercise_html_manifest.json` |
| `content_url` | — | `raw.githubusercontent.com/.../exercises/{slug}.html` |

---

## 4. Student split table (`splits_{dataset}.parquet`)

| Column | Type | Description |
|---|---|---|
| `dataset` | string | |
| `student_id` | string | |
| `split` | string | `train` \| `val` \| `test` |
| `seed` | int32 | Split seed (2024) |

Produced by `scripts/data/leakage_design.split_students()` with 70/10/20 fractions.

---

## 5. LLM difficulty cache (`llm_difficulty_{dataset}_{model}.json`)

| Field | Type | Description |
|---|---|---|
| `item_id` | string | |
| `model` | string | e.g. `gpt-4o-mini` |
| `temperature` | float | 0.0 for point estimate |
| `difficulty` | float | [0.0, 1.0] after parsing |
| `raw_response` | string | Model output |
| `prompt_hash` | string | SHA-256 of frozen prompt template |
| `stem_hash` | string | SHA-256 of `stem_text` used |

---

## 6. Leakage tests (required before pilot)

Run after schema build:

```python
from scripts.data.leakage_design import assert_no_student_overlap, leakage_test_no_test_in_reference_table

assert_no_student_overlap(split)
leakage_test_no_test_in_reference_table(split, rasch_student_ids)
```

Document results in `reports/LEAKAGE_TEST_RESULTS.md`.

---

## 7. Eligible item counts (frozen reference)

| Dataset | Items in LLM+KT matrix |
|---|---:|
| XES3G5M | 7,618 |
| Junyi Academy | 666 |

Junyi: 679 math items have HTML; 666 pass intelligibility after stem extraction.  
Remaining 13 items with HTML but marginal stems are flagged in reconciliation table.

---

## 8. Build scripts (to implement in pilot phase)

| Script | Purpose |
|---|---|
| `scripts/data/build_unified_xes3g5m.py` | XES → unified schema |
| `scripts/data/build_unified_junyi.py` | Junyi log + HTML → unified schema |
| `scripts/data/validate_unified_schema.py` | Column checks + leakage tests |

**Not implemented in Amendment 008.** Schema is specified; build runs during pilot prep.
