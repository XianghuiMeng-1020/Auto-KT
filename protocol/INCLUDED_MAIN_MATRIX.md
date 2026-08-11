# Included Main Matrix — Frozen Three-Dataset Design

**Status:** FROZEN (Amendment 008)  
**Effective:** 2026-06-30  
**Project stop code:** `TWO_AUTHENTIC_MATH_DATASETS_PROTOCOL_FROZEN_SCHEMA_BUILD_PENDING`

---

## Final roles (no substitutions)

| Dataset | Role | `project_role` | Setting |
|---|---|---|---|
| **GSM8K-IRT** | Synthetic alignment environment | `INCLUDED_MAIN_MATRIX` | Synthetic |
| **XES3G5M** | Authentic mathematics dataset 1 | `INCLUDED_MAIN_MATRIX` | Authentic |
| **Junyi Academy** | Authentic mathematics dataset 2 | `INCLUDED_MAIN_MATRIX` | Authentic |

All other candidate datasets considered are **EXCLUDED** and must not appear in experiments or reported results.

---

## Permanent exclusions

| Dataset | Reason |
|---|---|
| FoundationalASSIST | Owner rejected access; gated |
| DBE-KT22 | Non-mathematics framing |
| Eedi NeurIPS 2020 | No machine-readable stems |
| MoocRadar | Reproducible download failed |
| MathE (UCI) | Keywords only; no temporal ordering |
| EDNet | English TOEIC prep, not mathematics |

---

## Nine-requirement matrix (authentic datasets)

Both XES3G5M and Junyi must support all nine:

1. Machine-readable mathematics item content for LLM scoring  
2. Authentic student–item responses with correctness  
3. Defensible interaction ordering or sequence information  
4. Student-level train/validation/test splitting  
5. Training-only empirical item difficulty  
6. Construct-validity analysis  
7. Item cold-start evaluation  
8. KT training  
9. Item-prioritisation analysis  

---

## Junyi content architecture (two-layer)

```
junyi_ProblemLog_original.csv     →  interactions (25.9M rows, 247K students)
junyi_Exercise_table.csv          →  metadata (slug, topic, area)
junyiexercise HTML (966 files)    →  formal item stems (CC BY-NC-SA 3.0)
         ↓ slug join
eligible items for LLM + KT matrix
```

**Critical:** The USTC/EduData archive alone does **not** satisfy requirement 1. The junyiexercise HTML layer is mandatory and formally versioned.

Provenance:
- `data_raw/junyi/junyi_exercise_html_manifest.json` — per-file URL + SHA-256  
- `data_manifests/junyi_html_content_layer.json` — reconciliation summary  
- `reports/data_audits/JUNYI_HTML_RECONCILIATION.md` — full audit  

---

## Eligible item counts (entering shared matrix)

| Dataset | Eligible items | Source |
|---|---:|---|
| XES3G5M | 7,618 | `metadata/questions.json` + train interactions |
| Junyi Academy | See `tables/JUNYI_ELIGIBLE_ITEM_SUMMARY.csv` | HTML reconciliation |

---

## Paper narrative (frozen)

**GSM8K-IRT** shows how difficulty validity and KT utility are amplified when the response generator and LLM difficulty are artificially aligned.

**XES3G5M** and **Junyi Academy** test construct validity, surface-feature confounding, cold-start utility, and scalar representation ceiling in two real mathematics learning environments.

---

## Next phase gates

| Gate | Requirement |
|---|---|
| Schema freeze | `protocol/UNIFIED_SCHEMA_SPEC.md` |
| Eligibility freeze | `protocol/COMMON_ELIGIBILITY_STANDARDS.md` |
| Prompt freeze | `protocol/LLM_DIFFICULTY_PROMPT_FREEZE.md` |
| LLM pilot | Both authentic datasets; pass before confirmatory |
| E2E pilot | Small-scale; pass before confirmatory |

Do **not** begin full confirmatory experiments until pilots pass.
