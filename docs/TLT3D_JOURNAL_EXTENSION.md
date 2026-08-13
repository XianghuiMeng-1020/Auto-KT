# TLT-3D Journal Extension (IEEE TLT)

This directory set extends the original Auto-KT conference reproducibility package
with the **three authentic datasets** program used in the IEEE TLT manuscript:

- XES3G5M
- Junyi Academy
- DBE-KT22 (external / non-mathematics instructional context)

GSM8K remains a **controlled simulation** resource (not a fourth authentic validation set).

## What is shipped here

- `configs/tlt3d/` — frozen protocol / family registries / DBE runtime configs
- `artifacts/tlt3d/FINAL_*.csv|json` — sealed confirmatory evidence tables
- `artifacts/tlt3d/dbe_llm_scores_confirmatory.csv` — sanitized DBE LLM scores (hashes + scalars; no item text)
- `scripts/tlt3d_*.py`, `scripts/render_tlt_*.py` — preparation, analysis, freeze, and table/figure renderers
- `scripts/tlt4d/`, `scripts/data/unified_schema_common.py` — helpers required by DBE prep

## Not redistributed

- Raw educational datasets (`data_raw/`, `data_processed/`)
- Paid LLM raw response caches (`*.jsonl`)
- Large KT training checkpoints / run directories

## Minimal table regeneration (from frozen FINAL_* artifacts)

```bash
python scripts/render_tlt_final_supplement_tables.py
python scripts/tlt3d_p4d2_postprocess_generated_tables.py
python scripts/render_tlt_study_overview.py
```

Full re-execution of Families A–D requires acquiring the public datasets and
rebuilding derived tables under the project’s local data layout.
