# 📐 Validating LLM-Estimated Item Difficulty for Knowledge Tracing

Reproducibility code for a study of whether **large-language-model (LLM) estimates of item difficulty** are valid measures of authentic learner difficulty, and whether they provide useful signal for **knowledge tracing (KT)** — including for items with little or no prior response history.

---

## ✨ What's in this study

| # | Component | Description |
|---|------------|-------------|
| 1 | 🧮 **LLM difficulty scoring** | Score mathematics items from two KT datasets (XES3G5M, Junyi Academy) using visible item content only. |
| 2 | 📊 **Authentic-response validation** | Validate LLM scores against authentic held-out learner error rates (Rasch-based and empirical references), with nested-CV incremental-validity and sensitivity analyses. |
| 3 | 🧪 **Controlled signal-decoupling simulation** | Use GSM8K items to separate a generative difficulty signal from surface-feature confounds under a known ground truth. |
| 4 | 🧠 **Knowledge tracing experiments** | Inject the LLM difficulty score as an auxiliary scalar feature into GRU and SAKT backbones, under response-limited training and genuine unseen-item cold-start regimes (5-fold item holdout, shared `UNK_ITEM` representation, item-ID dropout). |

---

## 🗂️ Repository structure

```text
README.md
requirements.txt
.gitignore

src/
├── data_prep/     dataset unification, item eligibility, content sufficiency
├── llm_scoring/   LLM difficulty scoring pipeline (pilot + full scoring)
├── measurement/   Rasch estimation, authentic construct validity, RQ1
├── simulation/    controlled signal-decoupling simulation (GSM8K)
└── kt/            GRU / SAKT knowledge tracing (response-limited + unseen-item)

scripts/
├── run_authentic_analysis.py    authentic learner-response validity pipeline
├── run_simulation.py            controlled signal-decoupling simulation
├── run_response_limited_kt.py   response-limited KT (GRU by default, --backbone SAKT)
├── run_unseen_item_kt.py        genuine unseen-item KT (GRU + SAKT)
└── build_results.py             aggregate raw runs into compact result tables

configs/           configuration for each pipeline stage (*.json)
data/README.md     dataset acquisition and preparation instructions

artifacts/
├── scores/        frozen, sanitized LLM/legacy difficulty scores (hash-keyed)
├── item_folds/    frozen 5-fold genuine unseen-item partitions (hash-keyed)
└── manifests/     frozen provenance manifests (hashes, config, no content)

results/           compact aggregate result tables produced by the analyses
tests/             unit tests, including leakage/eligibility guard tests
```

> 💡 `runs/`, `data_raw/`, `data_processed/`, and `data_manifests/` are created locally when you execute the pipeline. They are gitignored and are **not** part of this repository.

---

## ⚙️ Environment

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Running the LLM scoring pipeline from scratch is optional — frozen scores are already provided (see below). If you do want to rescore, you'll additionally need an OpenAI-compatible API key set as `OPENAI_API_KEY`.

---

## 📥 Data acquisition and preparation

Raw datasets (XES3G5M, Junyi Academy, GSM8K) are **not redistributed** with this repository. See [`data/README.md`](data/README.md) for download and preprocessing instructions. After preparation you should have hashed, unified interaction and item tables under `data_processed/<dataset>/`.

---

## ❄️ Frozen difficulty-score artifacts

To let you reproduce the downstream analyses without repeating paid, model-version-dependent API calls, this repository ships sanitized frozen score artifacts under `artifacts/scores/`:

- **`llm_item_scores.parquet`** — per-item, per-model LLM difficulty scores for XES3G5M and Junyi (`item_id_hash`, `model_identifier`, `scalar_difficulty`, plus sub-dimension ratings and hashes). Contains no item text, learner identifiers, or response records.
- **`gsm8k_legacy_difficulty_scores.csv`** — a content-hash-keyed difficulty score per GSM8K item, used only by the controlled simulation.

To regenerate LLM scores from scratch instead (requires an API key and the prepared item content), see `src/llm_scoring/run_full_scoring.py` and `src/llm_scoring/run_llm_pilot.py`.

---

## 🔬 Authentic-response analysis

Builds held-out difficulty references (Rasch and empirical, including an orientation-corrected variant), validates LLM scores against them, and produces the RQ1 visible-feature association, incremental-validity, and sensitivity tables:

```bash
python scripts/run_authentic_analysis.py
```

## 🧪 Controlled signal-decoupling simulation

Runs the synthetic alignment ladder over GSM8K items with a known generative difficulty signal, decoupled from surface-feature confounds:

```bash
python scripts/run_simulation.py
```

## 📉 Response-limited KT

Trains and evaluates KT models under varying numbers of training responses per item, across LLM-score, random-score, character-length, and training-set empirical-difficulty scalar conditions:

```bash
python scripts/run_response_limited_kt.py --dataset xes3g5m --exposure 5 --condition LLM-Mini --seed 2024
python scripts/run_response_limited_kt.py --backbone SAKT --dataset junyi --exposure 5 --condition LLM-5.4 --seed 2024
```

## 🧊 Genuine unseen-item KT

Trains and evaluates KT models on a 5-fold item holdout, where all training responses to held-out target items are removed and item IDs are mapped to a shared `UNK_ITEM` representation:

```bash
python scripts/run_unseen_item_kt.py --dataset xes3g5m --fold 0
```

Use `src/kt/run_unseen_item_kt_parallel.py` to launch one process per fold.

## 📦 Building result tables

After running the KT experiments above (which append to `runs/response_limited_kt/RUN_REGISTRY.csv`, `runs/sakt_response_limited_kt/RUN_REGISTRY.csv`, and `runs/unseen_item_kt/RUN_REGISTRY.csv`), aggregate the raw runs into the compact tables under `results/`:

```bash
python scripts/build_results.py
```

---

## 📤 Expected outputs

| Prefix | Contents |
|---|---|
| `results/AUTHENTIC_*`, `RASCH_*`, `RQ1_*` | Authentic-response validity results |
| `results/SYNTHETIC_*` | Controlled simulation results |
| `results/LIMITED_KT_*`, `results/*RESPONSE_LIMITED_KT*` | Response-limited KT results |
| `results/UNSEEN_ITEM_KT_*`, `results/SAKT_RESPONSE_LIMITED_KT_*` | Genuine unseen-item and SAKT-backbone KT results |
| `results/FULL_LLM_*`, `results/*_LLM_SCOREABILITY*` | LLM scoring coverage and reliability diagnostics |

---

## 🔁 Reproducibility notes

- ⚠️ All dataset identifiers (`item_id_hash`, `student_id_hash`) are computed deterministically from a fixed hash salt (`src/data_prep/unified_schema_common.py`, `configs/unified_schema_config.json`). **Do not change this value** — otherwise your hashes will no longer match the frozen score and fold artifacts shipped here.
- 🎲 All random seeds, item folds, item-ID dropout policy, response-exposure masks, and evaluation metrics are unchanged from the original study implementation.
- ✅ `tests/` includes unit tests for dataset join coverage, student-split leakage, unified-schema leakage, content-sufficiency rules, and LLM scoring/pilot guards.

---

## 📖 Citation

If you use this code, please cite the associated publication once available. Citation details are intentionally omitted here pending final publication.
