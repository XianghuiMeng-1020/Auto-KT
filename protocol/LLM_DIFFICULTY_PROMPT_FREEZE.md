# LLM Difficulty Prompt — Frozen Specification

**Status:** FROZEN (Amendment 008)  
**Effective:** 2026-06-30  
**Applies to:** GSM8K-IRT, XES3G5M, Junyi Academy (all INCLUDED_MAIN_MATRIX datasets)

---

## 1. Rationale

A single frozen prompt ensures cross-dataset comparability of LLM-estimated difficulty and prevents post-hoc prompt tuning on test outcomes.

Any change requires Amendment 009 and re-running all LLM pilots.

---

## 2. Frozen prompt template

### System message

```
You are a mathematics education expert. Estimate the difficulty of the following problem for a typical student at the appropriate grade level. Output a single number between 0.0 (very easy) and 1.0 (very hard). Output only the number, with no explanation.
```

### User message

```
Problem:
{stem_text}
```

`{stem_text}` is the unified `stem_text` from `items_{dataset}.parquet` (see `protocol/UNIFIED_SCHEMA_SPEC.md`).

For Junyi graphie exercises, `stem_text` uses title fallback when question div is empty (documented in reconciliation).

---

## 3. Model and API parameters (pilot)

| Parameter | Pilot value | Confirmatory (if pilot passes) |
|---|---|---|
| Primary model | `gpt-4o-mini` | Same unless pilot approves `gpt-5.4` |
| Temperature (point estimate) | `0.0` | `0.0` |
| Temperature (stochastic replicate) | `0.3` | `0.3` |
| Stochastic replicates | 3 | 3 |
| Max tokens | 16 | 16 |
| Timeout | 30 s | 30 s |
| Max retries | 3 | 3 |

**Total calls per item (confirmatory):** 4 per model (1× T=0 + 3× T=0.3)

---

## 4. Output parsing

```python
def parse_difficulty(raw: str) -> float:
    """Extract first float in [0, 1] from model response."""
    import re
    m = re.search(r"0?\.\d+|1\.0*|0|1", raw.strip())
    if m:
        v = float(m.group())
        return max(0.0, min(1.0, v))
    return float("nan")  # mark for retry; do not impute silently
```

### Post-processing (frozen)

| Step | Rule |
|---|---|
| Clipping | None in confirmatory (raw [0,1] preserved) |
| Pilot clipping | Optional [0.05, 0.95] for stability diagnostics only |
| Aggregation | Mean of 3 stochastic + T=0 point for variance estimation |
| Missing parse | Retry up to 3×; flag item in audit log if still missing |

**Note:** An earlier pipeline used a [0.1, 0.9] clamp; the current study uses unclamped values to avoid ceiling compression.

---

## 5. Caching and reproducibility

| Requirement | Implementation |
|---|---|
| Cache path | `data/{dataset}/cache/llm_difficulty_{model}_v1.json` |
| Resume | Skip items already in cache with matching `prompt_hash` + `stem_hash` |
| Prompt hash | SHA-256 of system + user template (excluding `{stem_text}`) |
| Stem hash | SHA-256 of `stem_text` per item |
| Log | `retrieval_date_utc`, `model`, `prompt_hash` in manifest |

---

## 6. Pilot scope (before confirmatory)

| Dataset | Pilot items | Purpose |
|---|---:|---|
| XES3G5M | 100 (stratified by empirical difficulty quartile) | Parse success, correlation with train Rasch |
| Junyi Academy | 100 (stratified by response count) | Parse success, title-fallback vs question-div quality |
| GSM8K-IRT | 50 | Prompt stability check on English stems |

### Pilot pass criteria

1. ≥ 98% parse success (valid float in [0,1])  
2. Train-split Pearson r(LLM, Rasch) ≥ 0.25 for both authentic datasets  
3. No systematic failure on title-fallback stems  
4. Cost within budget gate (`reports/RESOURCE_ESTIMATE.md`)  

---

## 7. Prohibited practices

- No few-shot examples from test-set items  
- No student outcome information in prompt  
- No dataset-specific prompt variants (same template for XES and Junyi)  
- No prompt revision after observing test-set KT results  

---

## 8. Relationship to other extraction

| Feature | Method | LLM? |
|---|---|---|
| Difficulty scalar | This prompt | Yes |
| Skill/concept tags | Dataset metadata or rules | No |
| Prerequisite graph | Curriculum rules / dataset KC tree | No |

Decoupling reduces circularity between graph construction and difficulty estimation.
