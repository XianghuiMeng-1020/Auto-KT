# Protocol Documents

Frozen methodology specifications referenced by the code in `src/`:

| Document | Purpose |
|---|---|
| `INCLUDED_MAIN_MATRIX.md` | Dataset roles (GSM8K, XES3G5M, Junyi Academy) |
| `COMMON_ELIGIBILITY_STANDARDS.md` | Shared student/item eligibility and split rules |
| `UNIFIED_SCHEMA_SPEC.md` | Cross-dataset interaction and item table schema |
| `LLM_DIFFICULTY_PROMPT_FREEZE.md` | Frozen LLM difficulty-scoring prompt and parameters |

These documents describe the frozen specification implemented by
`src/data_prep/`, `src/llm_scoring/`, and `src/measurement/`. See the
top-level `README.md` for how to run the corresponding pipeline stages.
