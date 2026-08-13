# Data Acquisition and Preparation

This repository does not redistribute raw learner-response or item-content
data from any source dataset. You must obtain each dataset yourself and
place it under `data_raw/` in your local clone (this path is gitignored).
Preprocessing code then builds a hashed, unified schema under
`data_processed/` (also gitignored).

## Datasets used

### XES3G5M

A large-scale knowledge-tracing benchmark of mathematics items and learner
interactions. Obtain the dataset from its original public release and place
the extracted files under:

```
data_raw/xes3g5m/
```

Then build the unified schema:

```
python src/data_prep/build_unified_xes3g5m.py
```

### Junyi Academy Math Practicing Log

A large-scale mathematics practice log released by Junyi Academy. Download
the interaction log and, if you also want machine-readable item stems, the
`junyiexercise` HTML source:

```
python src/data_prep/download_junyi.py
python src/data_prep/download_junyi_exercise_html.py
```

This populates `data_raw/junyi/`. Then build the unified schema:

```
python src/data_prep/build_unified_junyi.py
```

### GSM8K

A grade-school math word-problem dataset, used only for a legacy simulation
diagnostic (`src/simulation/synthetic_alignment_common.py`). Obtain the
standard GSM8K training split (e.g. via the Hugging Face `gsm8k` dataset,
`main` configuration) and save it as a CSV with a `question` column, in the
original row order, at:

```
data_raw/gsm8k/train.csv
```

Raw item text is not shipped. `artifacts/scores/gsm8k_legacy_difficulty_scores.csv`
provides a per-item SHA-256 content hash and a frozen difficulty score; the
loader verifies your local copy against this hash before joining.

## Item eligibility and content sufficiency

Not every item in each dataset has enough machine-readable text content for
LLM scoring. The eligibility rule and its validation are implemented in:

- `src/data_prep/content_sufficiency.py`
- `src/data_prep/apply_content_sufficiency.py`
- `src/data_prep/validate_content_sufficiency_rules.py`
- `src/data_prep/check_content_sufficiency.py`
- `src/data_prep/check_junyi_eligibility.py` / `check_junyi_eligibility_with_html.py`

Run `src/data_prep/validate_unified_schema.py` after building the unified
schema to check join integrity and leakage guards (see `tests/`).

## Expected local layout after preparation

```
data_raw/
  xes3g5m/...
  junyi/...
  gsm8k/train.csv
data_processed/
  xes3g5m/items.parquet
  xes3g5m/interactions.parquet
  xes3g5m/splits.parquet
  junyi/items.parquet
  junyi/interactions.parquet
  junyi/splits.parquet
```

All item and student identifiers in `data_processed/` are hashed
(`item_id_hash`, `student_id_hash`); no raw item text or learner-identifying
information is written outside your local `data_processed/` directory.

## DBE-KT22 (journal extension)

DBE-KT22 is used as an external authentic dataset in the IEEE TLT extension.
Raw DBE files are **not** redistributed here. Obtain the public release from the
dataset authors / hosting venue cited in the manuscript, then place derived
tables under your local `data/external/dbe_kt22/` layout as expected by
`scripts/tlt3d_prepare_dbe.py`. Sanitized confirmatory LLM scores (no item text)
are already provided in `artifacts/tlt3d/dbe_llm_scores_confirmatory.csv`.
