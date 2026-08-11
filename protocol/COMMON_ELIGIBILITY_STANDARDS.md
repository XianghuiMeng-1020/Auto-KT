# Common Eligibility Standards — XES3G5M & Junyi Academy

**Status:** FROZEN (Amendment 008)  
**Effective:** 2026-06-30  
**Applies to:** Authentic mathematics datasets in INCLUDED_MAIN_MATRIX only

---

## 1. Purpose

Define identical eligibility rules for both authentic datasets before unified schema preparation, LLM pilot, and confirmatory experiments.

GSM8K-IRT follows the synthetic alignment protocol separately but uses the **same LLM difficulty prompt** (see `protocol/LLM_DIFFICULTY_PROMPT_FREEZE.md`).

---

## 2. Student eligibility

| Criterion | XES3G5M | Junyi Academy |
|---|---|---|
| Minimum interactions per student | ≥ 10 valid interactions | ≥ 10 interactions |
| Sequence ordering field | `seq` (within student) | `time_done` (Unix ms, ascending) |
| Correctness field | `cpts` (0/1) | `correct` (boolean) |
| Student ID column | `uid` | `user_id` |

Students below the interaction threshold are excluded before splitting.

---

## 3. Item eligibility

### XES3G5M

An item is **eligible** if:

1. Present in `metadata/questions.json` with non-empty `content` field  
2. Appears in at least one **training-split** student interaction  
3. Join coverage: item_id in interactions ↔ questions.json = 100%  

**Eligible count (frozen):** 7,618 items

### Junyi Academy

An item is **eligible** if **all** of:

1. `exercise` slug appears in interaction log  
2. `area` ∉ excluded non-math domains (`biology`, `logics`, `history`, etc.)  
3. Matching HTML file exists in `data_raw/junyi/exercises_html/{slug}.html`  
4. Extracted stem is intelligible (≥10 chars from question div **or** title fallback for graphie exercises)  

**Eligible count (frozen):** 666 items (see `tables/JUNYI_ELIGIBLE_ITEM_SUMMARY.csv`)

Items failing criterion 3–4 may still enter KT with item_id embedding only; they are **excluded from LLM difficulty scoring** and construct-validity analyses requiring text.

---

## 4. Item content requirements (Gate B)

| Requirement | XES3G5M | Junyi |
|---|---|---|
| Machine-readable stem | `questions.json` → `content` (Chinese plain text) | HTML → question div or title fallback |
| Formula representation | Inline text / LaTeX in content | LaTeX in HTML (`\frac`, etc.) |
| Options (if MCQ) | `options` field when `type=1` | Parsed from HTML when present |
| Minimum item coverage (math domain) | 100% | ≥ 95% HTML match (achieved: 95.0%) |
| Minimum response coverage (eligible) | 100% (by construction) | ≥ 94% (achieved: 94.07%) |

### Junyi content extraction tiers (reported separately)

| Tier | Definition | Count |
|---|---|---:|
| A — Question div | Stem from `<div class="question">` | 655 |
| B — Title fallback | Graphie/basic exercises; `<title>` as stem | 28 |
| C — Ineligible | No HTML or unintelligible stem | 49 |

---

## 5. Split policy (leakage-safe)

| Parameter | Value |
|---|---|
| Split unit | **Student** (never item or interaction) |
| Train / Val / Test | 70% / 10% / 20% |
| Random seed | `2024` |
| Implementation | `scripts/data/leakage_design.py` → `split_students()` |

### Leakage prohibitions

- No test-student interactions in Rasch / empirical difficulty reference  
- No test-student interactions in LLM prompt calibration sets  
- Item-level features (LLM scores) computed from **text only**; no student outcome leakage  
- KT validation/tuning uses val split only; test split touched once for confirmatory evaluation  

---

## 6. Empirical difficulty (training-only reference)

| Property | Rule |
|---|---|
| Estimation split | Train students only |
| Estimator | Rasch / 1PL IRT (prespecified) |
| Minimum responses per item | ≥ 30 train-student attempts (item dropped otherwise) |
| Use in analysis | Construct-validity correlation with LLM difficulty |

---

## 7. Cold-start evaluation

Eligible items partitioned by **training-set exposure count**:

| Level | Train exposure |
|---|---|
| L0 | 0 (never seen in train) |
| L1 | 1–5 |
| L2 | 6–20 |
| L3 | 21–100 |
| L4 | 101–500 |
| L5 | > 500 |

Cold-start utility evaluated on test-split interactions for items at each level.

---

## 8. Reconciliation artifacts (Junyi mandatory)

Before LLM pilot:

- `scripts/data/reconcile_junyi_html_coverage.py`  
- `reports/data_audits/JUNYI_HTML_RECONCILIATION.md`  
- `tables/JUNYI_EXERCISE_HTML_RECONCILIATION.csv`  
- `data_manifests/junyi_html_content_layer.json`  

---

## 9. Deviations

Any deviation from this document requires Amendment 009 with dated justification. An earlier XES3G5M pipeline used a 2000-student subsample with an 80/20 split; the current study uses this frozen standard instead.
