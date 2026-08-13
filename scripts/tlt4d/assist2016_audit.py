#!/usr/bin/env python3
"""ASSISTments 2016 Phase-0C feasibility audit (blocked path + re-run when data present)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

SPLIT_LABEL = "FEASIBILITY_ONLY_NOT_SCIENTIFICALLY_FROZEN"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def locate_format1(root: Path) -> list[Path]:
    raw = root / "data" / "external" / "assist2016" / "raw"
    hits = []
    for p in raw.rglob("*"):
        if not p.is_file():
            continue
        name = p.name.lower()
        if p.suffix.lower() in {".csv", ".tsv", ".txt", ".zip", ".gz"} and (
            "format1" in str(p).lower()
            or "target" in name
            or "assist" in name
            or name.endswith(".csv")
        ):
            # ignore empty stubs
            if p.stat().st_size > 0:
                hits.append(p)
    return sorted(hits)


def write_empty_item_audit(art: Path) -> None:
    cols = [
        "item_id",
        "problem_set_id",
        "problem_type",
        "raw_text_present",
        "normalized_text_chars",
        "has_choices",
        "has_image_reference",
        "has_diagram_reference",
        "has_template_reference",
        "duplicate_text_group",
        "text_complete_status",
        "exclusion_reason",
        "requires_prior_problem_context",
    ]
    pd.DataFrame(columns=cols).to_csv(art / "assist2016_item_audit.csv", index=False)
    pd.DataFrame(
        columns=[
            "item_id",
            "train_responses",
            "validation_responses",
            "test_responses",
            "unique_test_learners",
            "first_observed_test_responses",
            "test_error_rate",
        ]
    ).to_csv(art / "assist2016_response_eligibility.csv", index=False)


def audit_assist2016(root: Path, errors: list[str]) -> dict:
    base = root / "data" / "external" / "assist2016"
    derived = base / "derived"
    derived.mkdir(parents=True, exist_ok=True)
    art = root / "artifacts" / "tlt4d"
    art.mkdir(parents=True, exist_ok=True)

    official_page = "https://sites.google.com/site/assistmentskdd2016/dataset"
    drive_ids = {
        "format1_target_on_target_sets": {
            "url": "https://drive.google.com/open?id=0B8VPP_XVUjoaR2UxSlM3TUQxbGc",
            "id": "0B8VPP_XVUjoaR2UxSlM3TUQxbGc",
        },
        "format1_target_outside_target_sets": {
            "url": "https://drive.google.com/open?id=0B8VPP_XVUjoaWlFWYmN5V2s5Zzg",
            "id": "0B8VPP_XVUjoaWlFWYmN5V2s5Zzg",
        },
        "format2_target_on_target_sets": {
            "url": "https://drive.google.com/open?id=0B8VPP_XVUjoaakRGQlZOVFdnVFU",
            "id": "0B8VPP_XVUjoaakRGQlZOVFdnVFU",
        },
    }

    files = locate_format1(root)
    if files:
        # Full audit reserved for when official files are present; fail loudly rather than guess schema.
        errors.append(
            "ASSISTments2016 files present under raw/ but full parser not auto-invoked in blocked-phase stub; "
            f"found={[str(p.relative_to(root)) for p in files]}. Extend assist2016_audit after PI drop-in."
        )
        # Still mark as needing implementation — do not invent schema.
        access = "LOCAL_FILES_PRESENT_BUT_UNPARSED"
        verdict = "BLOCKED"
        gates = {f"G{i}": "BLOCKED" for i in range(1, 11)}
        blockers = [
            "Local files detected but Phase-0C acquisition was blocked at official Drive; "
            "do not treat unparsed local copies as verified until PI confirms official provenance."
        ]
    else:
        access = "ACCESS_BLOCKED"
        verdict = "BLOCKED"
        gates = {f"G{i}": "BLOCKED" for i in range(1, 11)}
        # G1 is the definitive blocker
        gates["G1"] = "BLOCKED"
        blockers = [
            "Official Format 1/2 Google Drive URLs redirect to Google account sign-in; "
            "gdown/public export cannot retrieve file without authentication",
            "No official non-Drive redistribution identified as source of record",
            "Per policy: do not use Kaggle/unverified mirrors; do not bypass auth",
        ]

    attempts = [
        {
            "method": "gdown uc?id=",
            "result": "Cannot retrieve public link / permission",
            "ids": [v["id"] for v in drive_ids.values()],
        },
        {
            "method": "browser navigate drive.google.com/open?id=...",
            "result": "Redirected to accounts.google.com Sign in to continue to Google Drive",
        },
        {
            "method": "urllib uc?export=download",
            "result": "HTML Google sign-in page (not binary CSV)",
        },
        {
            "method": "official_page_probe",
            "url": official_page,
            "result": "Page accessible; describes Format1/Format2 and Drive links; no direct file attachment download",
        },
    ]

    manifest = {
        "dataset": "assist2016",
        "source_of_record": "ASSISTments Data Mining Competition 2016 official dataset page + linked Google Drive",
        "official_page": official_page,
        "official_page_state_utc": utc_now(),
        "official_page_note": (
            "Page states Format 1 is one-row-per-problem-per-student CSV with student features, "
            "problem text, prerequisite/post-requisite skills, school/district demographics; "
            "covers problems before and during A/B experiments. Terms: agree not to deanonymize."
        ),
        "drive_targets": drive_ids,
        "access_status": access,
        "download_utc": utc_now(),
        "attempts": attempts,
        "files": [],
        "terms_of_use_note": "By downloading, users agree not to attempt deanonymization (official page).",
        "license_access_note": "Official competition data; acquisition currently requires authenticated Google Drive access.",
    }
    write_json(root / "artifacts" / "manifests" / "assist2016_source_manifest.json", manifest)
    write_empty_item_audit(art)

    result = {
        "access": access,
        "verdict": verdict,
        "executive_verdict": "ACCESS_BLOCKED",
        "gates": gates,
        "blockers": blockers,
        "raw_rows": 0,
        "raw_items": 0,
        "text_complete_items": 0,
        "learners": 0,
        "unique_learner_item_pairs": 0,
        "repeated_pair_rate": 0.0,
        "temporal_order_available": False,
        "items_test_ge20": 0,
        "items_test_ge50": 0,
        "items_test_ge100": 0,
        "unseen_item_eligible": 0,
        "four_dataset_phase_1_eligible": False,
        "schema_status": "NOT_OBSERVABLE_WITHOUT_FILES",
        "correctness_status": "NOT_OBSERVABLE_WITHOUT_FILES",
        "observational_unit_status": "NOT_OBSERVABLE_WITHOUT_FILES",
        "split_label_if_acquired": SPLIT_LABEL,
        "privacy_note": (
            "Dataset may contain anonymized demographics/school/teacher fields; "
            "these must not be used for prediction if later acquired."
        ),
    }
    write_json(derived / "assist2016_audit_summary.json", result)

    summary = {
        "phase": "TLT4D_P0C",
        "base_commit": "c2bfc914d43ec413126ed8f07435b47bbff9c2ac",
        "access": access,
        "raw_rows": 0,
        "raw_items": 0,
        "text_complete_items": 0,
        "learners": 0,
        "unique_learner_item_pairs": 0,
        "repeated_pair_rate": 0.0,
        "temporal_order_available": False,
        "items_test_ge20": 0,
        "unseen_item_eligible": 0,
        "verdict": verdict,
        "executive_verdict": "ACCESS_BLOCKED",
        "gates": gates,
        "blockers": blockers,
        "four_dataset_phase_1_eligible": False,
        "karl_fallback_performed": True,
        "karl_fallback_reason": "ASSISTments2016 G1 ACCESS_BLOCKED",
    }
    write_json(art / "P0C_ASSIST2016_FEASIBILITY_SUMMARY.json", summary)
    return result
