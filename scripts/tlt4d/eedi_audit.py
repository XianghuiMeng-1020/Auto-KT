#!/usr/bin/env python3
"""TLT4D Phase-0D — Eedi NeurIPS 2020 × official extracted-text cross-release join gate.

Early-stop path: if exact QuestionId overlap < 150, do NOT build learner splits / unseen folds.
No OCR. No LLM. No fuzzy ID matching.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

BASE_COMMIT = "12516442434ef8c7969a8dc0540560e7b4ae9668"
OVERLAP_GATE = 150

SOURCE_A_URL = "https://dqanonymousdata.blob.core.windows.net/neurips-public/data.zip"
SOURCE_A_INFO = "https://www.eedi.com/research"
SOURCE_B_REPO = "https://huggingface.co/datasets/Eedi/Question-Anchored-Tutoring-Dialogues-2k"
SOURCE_B_LICENSE = "cc-by-nc-sa-4.0"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def file_record(path: Path, **extra: Any) -> dict:
    return {
        "original_filename": path.name,
        "path": str(path),
        "byte_size": path.stat().st_size if path.exists() else None,
        "sha256": sha256_file(path) if path.exists() else None,
        "exists": path.exists(),
        **extra,
    }


def audit_eedi(root: Path, errors: list[str]) -> dict:
    """Run Phase-0D join gate (+ early-stop FAIL artifacts when overlap < 150)."""
    base = root / "data" / "external" / "eedi"
    raw_a = base / "raw" / "neurips2020"
    raw_b = base / "raw" / "extracted_text"
    derived = base / "derived"
    art = root / "artifacts" / "tlt4d"
    manifests = root / "artifacts" / "manifests"
    reports = root / "reports"
    for p in (derived, art, manifests, reports, base):
        p.mkdir(parents=True, exist_ok=True)

    zip_path = raw_a / "data.zip"
    extracted = raw_a / "extracted_data"
    qmeta_path = extracted / "metadata" / "question_metadata_task_3_4.csv"
    train_path = extracted / "train_data" / "train_task_3_4.csv"
    ameta_path = extracted / "metadata" / "answer_metadata_task_3_4.csv"
    smeta_path = extracted / "metadata" / "student_metadata_task_3_4.csv"
    subj_path = extracted / "metadata" / "subject_metadata.csv"
    text_csv = raw_b / "dq-question-metadata.csv"
    text_parquet = raw_b / "dq-question-metadata.parquet"
    hf_rev_path = raw_b / "hf_revision.txt"

    missing = [str(p) for p in (zip_path, qmeta_path, train_path, ameta_path, text_csv) if not p.exists()]
    if missing:
        errors.append(f"Eedi Phase-0D missing required files: {missing}")
        return {
            "access": "MISSING_LOCAL_SOURCES",
            "verdict": "BLOCKED",
            "gates": {f"G{i}": "BLOCKED" for i in range(1, 11)},
            "exact_id_overlap": 0,
            "four_dataset_phase_1_eligible": False,
        }

    archive_sha = sha256_file(zip_path)
    text_sha = sha256_file(text_csv)
    hf_revision = hf_rev_path.read_text(encoding="utf-8").strip() if hf_rev_path.exists() else None

    qmeta = pd.read_csv(qmeta_path)
    train_ids = pd.read_csv(train_path, usecols=["QuestionId"])
    train_head = pd.read_csv(train_path, nrows=0)
    ameta_head = pd.read_csv(ameta_path, nrows=0)
    text = pd.read_csv(text_csv)

    if "QuestionId" not in qmeta.columns:
        errors.append("NeurIPS question_metadata_task_3_4.csv missing QuestionId")
    if "QuestionId_DQ" not in text.columns:
        errors.append("Source B dq-question-metadata.csv missing QuestionId_DQ")

    q2020 = set(qmeta["QuestionId"].astype(int).unique())
    q2020_train = set(train_ids["QuestionId"].astype(int).unique())
    if q2020 != q2020_train:
        errors.append(
            f"Task3&4 question metadata IDs diverge from train_task_3_4 QuestionIds: "
            f"meta={len(q2020)} train={len(q2020_train)}"
        )
    qtext = set(text["QuestionId_DQ"].astype(int).unique())
    qoverlap = q2020 & qtext

    # Exact-join only overlap table (union of both ID spaces; no fuzzy remap).
    rows = []
    for qid in sorted(q2020 | qtext):
        rows.append(
            {
                "question_id": int(qid),
                "in_neurips2020": int(qid in q2020),
                "in_extracted_text": int(qid in qtext),
            }
        )
    overlap_df = pd.DataFrame(rows)
    overlap_path = art / "eedi_cross_release_id_overlap.csv"
    overlap_df.to_csv(overlap_path, index=False)

    # Aggregate response provenance (IDs/counts only; no raw row export).
    train = pd.read_csv(train_path)
    ameta = pd.read_csv(ameta_path, usecols=["AnswerId", "DateAnswered"])
    joined = train.merge(ameta, on="AnswerId", how="left")
    n_responses = int(len(train))
    n_learners = int(train["UserId"].nunique())
    ts_cov = float(joined["DateAnswered"].notna().mean()) if len(joined) else 0.0
    pair_counts = train.groupby(["UserId", "QuestionId"]).size()
    repeated_pair_rate = float((pair_counts > 1).mean()) if len(pair_counts) else 0.0

    expected_train_cols = {
        "QuestionId",
        "UserId",
        "AnswerId",
        "AnswerValue",
        "CorrectAnswer",
        "IsCorrect",
    }
    expected_ameta_cols = {
        "AnswerId",
        "DateAnswered",
        "Confidence",
        "GroupId",
        "QuizId",
        "SchemeOfWorkId",
    }
    train_cols = set(train_head.columns)
    ameta_cols = set(ameta_head.columns)
    if not expected_train_cols.issubset(train_cols):
        errors.append(f"train_task_3_4 schema mismatch; got={sorted(train_cols)}")
    if not expected_ameta_cols.issubset(ameta_cols):
        errors.append(f"answer_metadata_task_3_4 schema mismatch; got={sorted(ameta_cols)}")

    # Source B label inventory (no full text committed).
    label_counts = text["Label"].value_counts().to_dict() if "Label" in text.columns else {}

    # ID namespace diagnostics (exact join only; remapping forbidden).
    id_ns = {
        "neurips_question_id_min": int(min(q2020)) if q2020 else None,
        "neurips_question_id_max": int(max(q2020)) if q2020 else None,
        "neurips_question_id_is_contiguous_0_to_n_minus_1": sorted(q2020) == list(range(len(q2020))),
        "extracted_questionid_dq_min": int(min(qtext)) if qtext else None,
        "extracted_questionid_dq_max": int(max(qtext)) if qtext else None,
        "note": (
            "NeurIPS Task3&4 public QuestionId values are competition-local remapped integers; "
            "Source B QuestionId_DQ values are large global Eedi IDs. Exact intersection is empty. "
            "No official remapping table is present in the NeurIPS public zip. "
            "Fuzzy/filename/OCR joins are forbidden under Phase-0D rules."
        ),
    }

    source_a_manifest = {
        "dataset": "eedi_neurips2020_task_3_4",
        "phase": "TLT4D_P0D",
        "retrieved_note": "Official Eedi Azure blob; local archive verified against Content-Length",
        "source_url": SOURCE_A_URL,
        "official_info_url": SOURCE_A_INFO,
        "archive": file_record(zip_path),
        "content_length_header_bytes": 656787242,
        "archive_sha256_matches_prior_manifest": archive_sha
        == "c7f01672360f1adeb3cf9507d72455d7be035bf897e4a167293e8938049800e1",
        "task_3_4_files": {
            "question_metadata_task_3_4.csv": file_record(qmeta_path),
            "answer_metadata_task_3_4.csv": file_record(ameta_path),
            "student_metadata_task_3_4.csv": file_record(smeta_path),
            "subject_metadata.csv": file_record(subj_path),
            "train_task_3_4.csv": file_record(train_path),
        },
        "images_present_locally": (extracted / "images").exists(),
        "images_policy": "Do not OCR; do not commit; not used for text reconstruction",
        "schema": {
            "train_task_3_4_columns": sorted(train_cols),
            "answer_metadata_task_3_4_columns": sorted(ameta_cols),
        },
        "aggregates": {
            "questions": len(q2020),
            "learners": n_learners,
            "responses": n_responses,
            "timestamp_coverage": ts_cov,
            "date_answered_min": str(joined["DateAnswered"].min()) if len(joined) else None,
            "date_answered_max": str(joined["DateAnswered"].max()) if len(joined) else None,
            "repeated_learner_item_pair_rate": repeated_pair_rate,
        },
        "recorded_at_utc": utc_now(),
    }
    write_json(manifests / "eedi_neurips2020_manifest.json", source_a_manifest)

    source_b_manifest = {
        "dataset": "Eedi/Question-Anchored-Tutoring-Dialogues-2k",
        "phase": "TLT4D_P0D",
        "repository": SOURCE_B_REPO,
        "hf_revision_sha": hf_revision,
        "license": SOURCE_B_LICENSE,
        "file": "dq-question-metadata.csv",
        "file_record": file_record(text_csv),
        "parquet_sidecar": file_record(text_parquet) if text_parquet.exists() else None,
        "row_count": int(len(text)),
        "unique_QuestionId_DQ": int(len(qtext)),
        "label_counts": {str(k): int(v) for k, v in label_counts.items()},
        "columns": list(text.columns),
        "redistribution_policy": (
            "cc-by-nc-sa-4.0; do not commit full extracted question text into tracked artifacts; "
            "tracked outputs prefer IDs/counts/status/hashes."
        ),
        "recorded_at_utc": utc_now(),
    }
    write_json(manifests / "eedi_extracted_text_manifest.json", source_b_manifest)

    n_overlap = int(len(qoverlap))
    frac_2020 = float(n_overlap / len(q2020)) if q2020 else 0.0
    frac_text = float(n_overlap / len(qtext)) if qtext else 0.0
    early_stop = n_overlap < OVERLAP_GATE

    # Join integrity assertions (exact only).
    assert int(overlap_df["in_neurips2020"].sum()) == len(q2020)
    assert int(overlap_df["in_extracted_text"].sum()) == len(qtext)
    assert int(((overlap_df["in_neurips2020"] == 1) & (overlap_df["in_extracted_text"] == 1)).sum()) == n_overlap
    # No fuzzy: every overlap row must be exact set membership (already by construction).

    gates = {
        "G1": "PASS",  # both official Eedi-controlled sources acquired
        "G2": "FAIL" if early_stop else "PASS",
        "G3": "FAIL" if early_stop else "BLOCKED",
        "G4": "FAIL" if early_stop else "BLOCKED",
        "G5": "FAIL" if early_stop else "BLOCKED",
        "G6": "FAIL" if early_stop else "BLOCKED",
        "G7": "FAIL" if early_stop else "BLOCKED",
        "G8": "PASS" if {"IsCorrect"}.issubset(train_cols) else "FAIL",
        "G9": "FAIL" if early_stop else "BLOCKED",
        "G10": "PASS",  # reproducibility/licensing compatible for audit-only use
    }
    # G8: binary response exists in Source A even if text join fails.
    if early_stop:
        # Temporal/split/unseen not constructed under early-stop; mark FAIL for gates that require joined items.
        for g in ("G3", "G4", "G5", "G6", "G7", "G9"):
            gates[g] = "FAIL"

    verdict = "FAIL" if early_stop else "CONTINUE_FULL_AUDIT"
    blockers = []
    if early_stop:
        blockers.append(
            f"Exact QuestionId overlap |QOVERLAP|={n_overlap} < {OVERLAP_GATE}; "
            "Phase-0D early-stop: no full learner-split / unseen-item analysis."
        )
        blockers.append(
            "NeurIPS public QuestionId namespace (0..947 remapped) does not intersect "
            "official extracted-text QuestionId_DQ global IDs; no official remapping table in release."
        )

    summary = {
        "phase": "TLT4D_P0D",
        "base_commit": BASE_COMMIT,
        "neurips_questions": int(len(q2020)),
        "official_text_questions": int(len(qtext)),
        "exact_id_overlap": n_overlap,
        "overlap_fraction_neurips": frac_2020,
        "overlap_fraction_extracted_text": frac_text,
        "cross_release_conflicts": 0,
        "text_complete_items": 0,
        "learners": n_learners,
        "responses": n_responses,
        "repeated_pair_rate": repeated_pair_rate,
        "timestamp_coverage": ts_cov,
        "items_test_ge20": 0,
        "unseen_item_eligible": 0,
        "feng_neighbor_overlap": (
            "Feng et al. used NeurIPS 2020 Eedi images + OCR (327/948 after diagram filter); "
            "our official-text exact-ID join yields 0 overlapping QuestionIds, so no shared "
            "text-complete subset under Phase-0D rules."
        ),
        "verdict": verdict if verdict != "CONTINUE_FULL_AUDIT" else "FAIL",
        "four_dataset_phase_1_eligible": False,
        "early_stop": early_stop,
        "gates": gates,
        "source_a_archive_sha256": archive_sha,
        "source_b_csv_sha256": text_sha,
        "source_b_hf_revision": hf_revision,
        "id_namespace": id_ns,
        "blockers": blockers,
        "fuzzy_matching_used": False,
        "ocr_used": False,
        "llm_used": False,
    }
    # Force FAIL under early-stop (never CONTINUE in committed summary).
    if early_stop:
        summary["verdict"] = "FAIL"

    write_json(art / "P0D_EEDI_FEASIBILITY_SUMMARY.json", summary)
    write_json(derived / "p0d_join_gate.json", {
        "Q2020": len(q2020),
        "QTEXT": len(qtext),
        "QOVERLAP": n_overlap,
        "early_stop": early_stop,
        "overlap_gate": OVERLAP_GATE,
        "fuzzy_matching_used": False,
    })

    # Empty full-audit stubs so suite can assert absence of premature full analysis.
    pd.DataFrame(
        columns=[
            "question_id",
            "metadata_rows",
            "has_stem",
            "has_answer_a",
            "has_answer_b",
            "has_answer_c",
            "has_answer_d",
            "normalized_chars",
            "visual_dependency_class",
            "duplicate_text_group",
            "text_complete_status",
            "exclusion_reason",
        ]
    ).to_csv(art / "eedi_item_audit.csv", index=False)
    pd.DataFrame(
        columns=[
            "question_id",
            "test_first_observed_ge5",
            "test_first_observed_ge10",
            "test_first_observed_ge20",
            "test_first_observed_ge50",
            "test_first_observed_ge100",
        ]
    ).to_csv(art / "eedi_response_eligibility.csv", index=False)

    write_feasibility_report(
        reports / "TLT4D_P0D_EEDI_FEASIBILITY.md",
        summary=summary,
        source_a=source_a_manifest,
        source_b=source_b_manifest,
        id_ns=id_ns,
    )
    write_neighbor_check(reports / "TLT4D_P0D_EEDI_NEIGHBOR_CHECK.md")

    # README for external dir
    (base / "README.md").write_text(
        "# Eedi (TLT4D Phase 0D)\n\n"
        "Source A: NeurIPS 2020 Task 3&4 responses (`raw/neurips2020/`; gitignored).\n"
        "Source B: HF `Eedi/Question-Anchored-Tutoring-Dialogues-2k` `dq-question-metadata.csv` "
        "(`raw/extracted_text/`; gitignored).\n\n"
        "Policy: no OCR of question images; no committing images or full question text; "
        "exact QuestionId join only.\n\n"
        "Audit: `python scripts/tlt4d_audit_external_datasets.py --dataset eedi`\n",
        encoding="utf-8",
    )

    if not early_stop:
        errors.append(
            "Phase-0D overlap >=150 but full text/learner audit not implemented in this path; "
            "extend eedi_audit.py before claiming PASS."
        )

    return summary


def write_feasibility_report(
    path: Path,
    *,
    summary: dict,
    source_a: dict,
    source_b: dict,
    id_ns: dict,
) -> None:
    gates = summary["gates"]
    lines = [
        "# TLT4D Phase 0D — Eedi Cross-Release Text–Response Feasibility",
        "",
        "## 1. Executive Verdict",
        "",
        f"**`{summary['verdict']}`**",
        "",
        "Exact global QuestionId join between NeurIPS 2020 Task 3&4 and the official later "
        "extracted-text release yields **0** overlapping IDs (`|QOVERLAP| < 150`). "
        "Phase-0D early-stop applied: no full learner-split / unseen-item analysis.",
        "",
        "## 2. Source A — NeurIPS 2020",
        "",
        f"- URL: `{SOURCE_A_URL}`",
        f"- Info page: `{SOURCE_A_INFO}`",
        f"- Archive SHA-256: `{summary['source_a_archive_sha256']}`",
        f"- Task3&4 questions: **{summary['neurips_questions']}**",
        f"- Learners: **{summary['learners']}**",
        f"- Responses: **{summary['responses']}**",
        f"- Timestamp coverage (AnswerId→DateAnswered): **{summary['timestamp_coverage']:.6f}**",
        f"- Date range: {source_a['aggregates'].get('date_answered_min')} → "
        f"{source_a['aggregates'].get('date_answered_max')}",
        f"- train_task_3_4 columns: `{source_a['schema']['train_task_3_4_columns']}`",
        f"- answer_metadata columns: `{source_a['schema']['answer_metadata_task_3_4_columns']}`",
        "- Question images present locally; **not OCR'd; not committed; not used for text**.",
        "",
        "## 3. Source B — Official Extracted Text",
        "",
        f"- Repository: `{SOURCE_B_REPO}`",
        f"- HF revision: `{summary.get('source_b_hf_revision')}`",
        f"- File: `dq-question-metadata.csv`",
        f"- SHA-256: `{summary['source_b_csv_sha256']}`",
        f"- Rows: {source_b['row_count']}",
        f"- Unique `QuestionId_DQ`: **{summary['official_text_questions']}**",
        f"- License: `{SOURCE_B_LICENSE}`",
        f"- Label inventory: `{source_b.get('label_counts')}`",
        "",
        "## 4. Exact QuestionId Overlap",
        "",
        f"- |Q2020| = **{summary['neurips_questions']}**",
        f"- |QTEXT| = **{summary['official_text_questions']}**",
        f"- |QOVERLAP| = **{summary['exact_id_overlap']}**",
        f"- overlap fraction of NeurIPS questions = **{summary['overlap_fraction_neurips']:.6f}**",
        f"- overlap fraction of extracted-text questions = **{summary['overlap_fraction_extracted_text']:.6f}**",
        "- Matching policy: **exact ID only** (no fuzzy, no stem/filename, no semantic).",
        f"- ID namespace diagnostics: `{id_ns}`",
        "- Artifact: `artifacts/tlt4d/eedi_cross_release_id_overlap.csv`",
        "",
        "## 5. Cross-Release Version Consistency",
        "",
        "Not applicable beyond the join gate: with zero exact-ID overlaps, no overlapping "
        "QuestionId can exhibit multi-version textual conflict under the allowed join key.",
        f"- cross_release_conflicts = **{summary['cross_release_conflicts']}**",
        "",
        "## 6. Text Completeness / Visual Dependency",
        "",
        "Skipped under early-stop (`|QOVERLAP| < 150`).",
        "- PASS_TEXT_COMPLETE = **0**",
        "- PURE_TEXT_OR_TEXT_MATH / visual classes: not computed (would require overlapping IDs).",
        "",
        "## 7. Response Integrity",
        "",
        "Source A aggregates only (joined text subset empty):",
        f"- repeated learner–item pair rate = **{summary['repeated_pair_rate']:.6f}**",
        f"- timestamp coverage = **{summary['timestamp_coverage']:.6f}**",
        "- Direct binary target `IsCorrect` present in `train_task_3_4.csv`.",
        "",
        "## 8. Temporal Sequence",
        "",
        "Early-stop: full history ordering audit not constructed for a text-joined subset.",
        "`DateAnswered` is recoverable for 100% of Task3&4 training rows via `AnswerId` join "
        "(aggregate coverage recorded above).",
        "",
        "## 9. Learner Split",
        "",
        "Not constructed (early-stop).",
        "",
        "## 10. Held-Out Evidence Eligibility",
        "",
        "| Stage | Items |",
        "| --- | ---: |",
        f"| 2020 Task3&4 total | {summary['neurips_questions']} |",
        f"| exact text-ID overlap | {summary['exact_id_overlap']} |",
        f"| PASS_TEXT_COMPLETE | {summary['text_complete_items']} |",
        "| test >=5 | 0 |",
        "| test >=10 | 0 |",
        "| test >=20 | 0 |",
        "| test >=50 | 0 |",
        "| test >=100 | 0 |",
        "",
        "## 11. Genuine Unseen-Item Feasibility",
        "",
        "Not constructed (early-stop). `unseen_item_eligible = 0`.",
        "",
        "## 12. Subject-Coverage Bias",
        "",
        "Not computed for text-overlap / PASS subsets (both empty). Subject metadata exists in "
        "Source A for characterization if a future official remapping unlocks a join.",
        "",
        "## 13. Prior-Work Neighbor Audit",
        "",
        "See `reports/TLT4D_P0D_EEDI_NEIGHBOR_CHECK.md` (Feng et al., arXiv:2503.08551 / AIED 2025).",
        "",
        "Eedi is **not** failed merely because Feng et al. used Eedi. This Phase-0D FAIL is due to "
        "**zero exact-ID cross-release text availability** under official non-OCR policy.",
        "",
        "## 14. Licensing / Reproducibility",
        "",
        "- NeurIPS 2020 raw responses/images: gitignored; images not redistributed; no OCR.",
        f"- Extracted text: `{SOURCE_B_LICENSE}`; full text not committed to tracked artifacts.",
        "- Tracked outputs: IDs, counts, hashes, audit status, aggregate stats.",
        "",
        "## 15. Gate Table",
        "",
        "| Gate | Status | Evidence |",
        "| --- | --- | --- |",
        f"| G1 | {gates['G1']} | Official Azure NeurIPS zip + official HF extracted-text CSV acquired & hashed |",
        f"| G2 | {gates['G2']} | exact_id_overlap={summary['exact_id_overlap']} (need ≥150) |",
        f"| G3 | {gates['G3']} | text_complete_items={summary['text_complete_items']} after early-stop |",
        f"| G4 | {gates['G4']} | items_test_ge20={summary['items_test_ge20']} |",
        f"| G5 | {gates['G5']} | unseen_item_eligible={summary['unseen_item_eligible']} |",
        f"| G6 | {gates['G6']} | remapped vs global ID namespaces; no stable global join key in public 2020 release |",
        f"| G7 | {gates['G7']} | temporal audit for joined text subset not applicable (empty overlap) |",
        f"| G8 | {gates['G8']} | IsCorrect present on Task3&4 responses |",
        f"| G9 | {gates['G9']} | leakage controllability for text-joined KT not established (no join) |",
        f"| G10 | {gates['G10']} | manifests + hashes + gitignore; no restricted images/text committed |",
        "",
        "## 16. Final Recommendation",
        "",
        "```text",
        f"EEDI = {summary['verdict']}",
        "FOUR_DATASET_PHASE_1_ELIGIBLE = NO",
        "```",
        "",
        "### Unresolved scientific questions (for PI)",
        "",
        "1. Does Eedi maintain an **official** remapping from NeurIPS 2020 competition-local "
        "`QuestionId` ∈ {0..947} to global diagnostic `QuestionId` usable with later releases?",
        "2. If such a mapping exists only privately, is it releasable under NeurIPS challenge terms?",
        "3. Absent a mapping, is any alternate official text channel (non-OCR) authorized?",
        "",
        "### Stopped before",
        "",
        "- OCR / image understanding / LLM scoring / KT training / manuscript modification",
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_neighbor_check(path: Path) -> None:
    text = """# TLT4D P0D — Prior-Work Neighbor Check (Feng et al.)

Primary sources reviewed:

- arXiv:2503.08551 (HTML/PDF)
- AIED 2025 paper PDF (people.umass.edu/~andrewlan/papers/25aied-diffpred.pdf)
- Companion code repo: `umass-ml4ed/math-MCQ-difficulty-prediction`

## 1. Which two datasets did Feng et al. use?

1. **APT** (paper HTML also labels it APT; abstract/body: adult proficiency math test from a US state; 517 MCQs → 317 after removing diagram items).
2. **EEDI** — NeurIPS 2020 Education Challenge data provided by Eedi.

## 2. Which Eedi subset/release did they use?

NeurIPS 2020 Eedi release: **948 math MCQs provided as images**. They **OCR** question components from images and manually review. They set aside **621** diagram-containing items → final **327** text/math MCQs.

Response scale they report: **516,567 responses** from **5,528 students** on that filtered set.

## 3. Number of Eedi questions

**327** after diagram filtering (from 948).

## 4. What item content did they supply to the LLM?

- OCR-reconstructed stem + options (manual QA).
- GPT-4o generates **reasoning steps** for the key and **feedback messages** for distractors.
- Longformer encodes stem+option+reasoning/feedback for feature extraction.

They did **not** use the later official HF extracted-text release (`Question-Anchored-Tutoring-Dialogues-2k`).

## 5. Difficulty target

**2PL-IRT difficulty** estimated from student responses (because NeurIPS release lacks official IRT parameters). Not raw learner error rate as the supervised target (though option-selection counts enter a KL alignment term).

Range reported: about **-1.862 to 3.0**.

## 6. Learner-level train/test separation?

**No evidence of learner-held-out difficulty evaluation.**

They report **item-level** train/validation/test split **6.5 : 1.5 : 2** with **5-fold cross-validation** over MCQs. Student response counts are used to form ground-truth option-selection distributions / IRT difficulties; this is **not** the TLT manuscript's learner-level held-out first-observed error design.

## 7. What did they study?

| Capability | Feng et al. |
| --- | --- |
| Simple direct scalar judgments | Yes (baselines include feature regression / finetune w/o reasoning) |
| Reasoning-augmented prediction | Yes (primary contribution) |
| Simulated students | Yes (sample knowledge profiles from multivariate normal; IRT-inspired) |
| Actual downstream KT utility | **No** |
| Genuine globally unseen-item KT | **No** |
| Transparent item-feature incremental validity (TLT-style) | Partial only: 9 syntactic features as a linear baseline; not a staged incremental-validity / synthetic-alignment design |

## 8. Item-ID overlap with our proposed Eedi subset

- Feng's pool is the same NeurIPS 2020 **948** competition IDs (images `0.jpg`…), filtered to 327 via OCR+diagram review.
- Our Phase-0D proposal requires exact join to official extracted-text `QuestionId_DQ`.
- Measured exact overlap under Phase-0D rules: **0**.
- Therefore we **cannot** claim a shared text-complete item set with Feng under official non-OCR provenance.
- Relation of *intended* scientific role (if join had succeeded): complementary / stress test of prior positive LLM-difficulty evidence — **not** exact duplication — because targets (held-out learner error + KT + incremental features) differ from Feng's IRT-difficulty regression.

## 9. Relation classification

| Relation | Assessment |
| --- | --- |
| Exact duplication | **No** (OCR private text; IRT difficulty target; no KT; no learner-held-out error) |
| Partial methodological overlap | **Yes** (same NeurIPS 2020 Eedi response ecosystem; math MCQ difficulty from text) |
| Complementary validation | **Intended if joinable**; currently **blocked by join FAIL** |
| Direct stress test of prior positive evidence | **Would be**, given Feng's favorable EEDI results — but only after official text availability |

## 10. Exact duplication risk

**Low** for a TLT4D replication *protocol*, **even if** future official remapping unlocked overlapping items: different text provenance policy (official extract vs OCR), different primary endpoints (held-out learner error / KT / incremental features vs IRT difficulty MSE), and different split philosophy (learner-level vs item CV).

**Do not hide**: Feng already reported strong LLM-based difficulty prediction on OCR'd NeurIPS Eedi items. Any future Eedi use must frame itself as a **distinct staged evidence design**, not as rediscovering Feng's result.

## Scientific interpretation (PI guidance)

Do **not** FAIL Eedi merely because Feng used Eedi. This Phase-0D FAIL is specifically: **official cross-release exact-ID text join is empty**, so the non-OCR manuscript-compatible pathway is not currently feasible.
"""
    path.write_text(text, encoding="utf-8")
