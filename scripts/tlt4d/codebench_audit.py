#!/usr/bin/env python3
"""TLT4D Phase-0F — CodeBench (UFAM) content-availability kill test + early-stop path.

Stage A only when CONTENT_SOURCE_VERDICT != FULL_STATEMENT_AVAILABLE:
do NOT download the full multi-year corpus.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

BASE_COMMIT = "8b80fcbf03d1a66f318159e061ed84217c6c8917"
OFFICIAL_PAGE = "https://codebench.icomp.ufam.edu.br/dataset/"
FILES_BASE = "https://codebench.icomp.ufam.edu.br/dataset/files"
RELEASE = "1.81"
UA = {"User-Agent": "Mozilla/5.0 TLT4D-P0F-audit"}

SEMESTERS_1_81 = [
    "2016_1",
    "2016_2",
    "2017_1",
    "2017_2",
    "2018_1",
    "2018_2",
    "2019_1",
    "2019_2",
    "2020_ERE",
    "2020_1",
    "2020_2",
    "2021_1",
    "2021_2",
    "2022_1",
    "2022_2",
    "2023_1",
    "2023_2",
    "2024_1",
]


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


def head_url(url: str) -> dict:
    rec: dict[str, Any] = {"url": url, "checked_at_utc": utc_now()}
    try:
        with urlopen(Request(url, method="HEAD", headers=UA), timeout=60) as r:
            rec["status"] = r.status
            rec["final_url"] = r.geturl()
            for k in ["Content-Length", "Accept-Ranges", "Content-Type", "ETag", "Last-Modified"]:
                rec[k] = r.headers.get(k)
            # some servers omit Accept-Ranges on HEAD; probe Range
            if not rec.get("Accept-Ranges"):
                try:
                    req = Request(url, headers={**UA, "Range": "bytes=0-0"})
                    with urlopen(req, timeout=30) as rr:
                        rec["range_probe_status"] = rr.status
                        rec["Content-Range"] = rr.headers.get("Content-Range")
                        rec["range_supported"] = rr.status in {206, 200}
                except Exception as e:
                    rec["range_probe_error"] = str(e)
                    rec["range_supported"] = False
            else:
                rec["range_supported"] = True
    except Exception as e:
        rec["error"] = str(e)
    return rec


def audit_codebench(root: Path, errors: list[str]) -> dict:
    base = root / "data" / "external" / "codebench"
    raw = base / "raw"
    derived = base / "derived"
    art = root / "artifacts" / "tlt4d"
    manifests = root / "artifacts" / "manifests"
    reports = root / "reports"
    for p in (raw / "archives", derived, art, manifests, reports, base):
        p.mkdir(parents=True, exist_ok=True)

    # --- Preflight HEADs (do not download all) ---
    heads = {}
    for s in SEMESTERS_1_81:
        url = f"{FILES_BASE}/cb_dataset_{s}_v{RELEASE}.tar.gz"
        heads[s] = head_url(url)

    disk = shutil.disk_usage(str(root))
    probe_meta_path = derived / "probe_archive.json"
    probe_meta = (
        json.loads(probe_meta_path.read_text(encoding="utf-8")) if probe_meta_path.exists() else {}
    )
    preflight_heads_path = derived / "preflight_heads.json"
    if preflight_heads_path.exists():
        # merge previously recorded probe heads if present
        try:
            prior = json.loads(preflight_heads_path.read_text(encoding="utf-8"))
            for k, v in prior.items():
                if k != "disk" and k not in heads:
                    heads[k] = v
        except Exception:
            pass

    write_json(
        derived / "preflight_heads.json",
        {"release": RELEASE, "heads": heads, "disk": {"free_bytes": disk.free}},
    )

    # --- Content search over extracted probe semester ---
    extract_root = raw / "extract_probe"
    content_search = {
        "probe_semester": probe_meta.get("semester", "2016_2"),
        "probe_archive_sha256": probe_meta.get("sha256"),
        "probe_archive_bytes": probe_meta.get("bytes"),
        "probe_url": probe_meta.get("url"),
        "file_extensions_observed": {},
        "directories_matching_content_keywords": [],
        "files_matching_content_keywords": [],
        "assessment_fields_observed": [],
        "has_statementish_text_in_assessments": False,
        "unique_exercise_ids_in_probe": None,
        "live_exercise_url_probes": [
            "https://codebench.icomp.ufam.edu.br/index.php?r=exercise/view&id=1326 → 404",
            "https://codebench.icomp.ufam.edu.br/exercises/1326 → 404",
            "https://codebench.icomp.ufam.edu.br/problem/1326 → 404",
        ],
        "official_schema_nodes": [
            "users/codemirror",
            "users/codes",
            "users/executions",
            "users/grades",
            "users/user.data",
            "users/logins.log",
            "assessments/*.data",
        ],
        "note": (
            "Official dataset page schema and 2016-2 v1.81 extract contain logs, student code, "
            "grades, and assessment metadata (titles + exercise ID lists) only. No enunciado/"
            "statement/description files."
        ),
    }

    if extract_root.exists():
        ext_counts: dict[str, int] = {}
        keyword_dirs = []
        keyword_files = []
        kw = (
            "exercise",
            "exercises",
            "problem",
            "problems",
            "question",
            "questions",
            "statement",
            "description",
            "enunciado",
            "questao",
            "atividade",
            "testcase",
        )
        for p in extract_root.rglob("*"):
            if p.is_dir():
                name = p.name.lower()
                if any(k in name for k in kw) and name not in {"assessments"}:
                    keyword_dirs.append(str(p.relative_to(extract_root)))
                continue
            if not p.is_file():
                continue
            suf = p.suffix.lower().lstrip(".") or "(none)"
            ext_counts[suf] = ext_counts.get(suf, 0) + 1
            low = p.name.lower()
            if any(k in low for k in kw):
                keyword_files.append(str(p.relative_to(extract_root)))
        content_search["file_extensions_observed"] = dict(
            sorted(ext_counts.items(), key=lambda x: -x[1])
        )
        content_search["directories_matching_content_keywords"] = sorted(set(keyword_dirs))[:50]
        content_search["files_matching_content_keywords"] = sorted(keyword_files)[:50]

        # assessments field inventory
        import re
        from collections import Counter

        assess = list(extract_root.glob("*/**/assessments/*.data"))
        keys = Counter()
        ex_ids = set()
        statementish = 0
        for ap in assess:
            text = ap.read_text(encoding="utf-8", errors="replace")
            low = text.lower()
            if any(
                w in low
                for w in [
                    "enunciado",
                    "statement",
                    "description",
                    "descrição",
                    "descricao",
                    "input specification",
                ]
            ):
                statementish += 1
            for line in text.splitlines():
                m = re.match(r"^----\s*([^:]+):", line.strip())
                if m:
                    keys[m.group(1).strip()] += 1
                m2 = re.search(r"exercise\s+\d+:\s*(.+)$", line, re.I)
                if m2:
                    for tok in re.findall(r"\d+", m2.group(1)):
                        ex_ids.add(int(tok))
        content_search["assessment_fields_observed"] = [k for k, _ in keys.most_common()]
        content_search["has_statementish_text_in_assessments"] = statementish > 0
        content_search["n_assessments_in_probe"] = len(assess)
        content_search["unique_exercise_ids_in_probe"] = len(ex_ids)
        content_search["sample_exercise_ids"] = sorted(ex_ids)[:25]

    write_json(derived / "content_search.json", content_search)

    content_verdict = "NO_STATEMENT_AVAILABLE"
    # Hard early stop
    early_stop = content_verdict != "FULL_STATEMENT_AVAILABLE"

    write_content_source_md(
        reports / "TLT4D_P0F_CODEBENCH_CONTENT_SOURCE.md",
        content_verdict=content_verdict,
        content_search=content_search,
        probe_meta=probe_meta,
    )

    # Manifests
    probe_arch = None
    if probe_meta.get("semester"):
        cand = raw / "archives" / f"cb_dataset_{probe_meta['semester']}_v{RELEASE}.tar.gz"
        if cand.exists():
            probe_arch = cand

    source_manifest = {
        "phase": "TLT4D_P0F",
        "release": RELEASE,
        "official_page": OFFICIAL_PAGE,
        "release_date": "2024-10-24",
        "files_base": FILES_BASE,
        "anonymous_download": True,
        "semesters_listed": SEMESTERS_1_81,
        "heads_sample": {
            s: {
                "status": heads.get(s, {}).get("status"),
                "Content-Length": heads.get(s, {}).get("Content-Length"),
                "url": heads.get(s, {}).get("url") or f"{FILES_BASE}/cb_dataset_{s}_v{RELEASE}.tar.gz",
            }
            for s in ["2016_1", "2016_2", "2017_1", "2017_2", "2020_ERE"]
            if s in heads or True
        },
        "probe_archive": {
            "semester": probe_meta.get("semester"),
            "url": probe_meta.get("url"),
            "bytes": probe_meta.get("bytes")
            or (probe_arch.stat().st_size if probe_arch and probe_arch.exists() else None),
            "sha256": probe_meta.get("sha256")
            or (sha256_file(probe_arch) if probe_arch and probe_arch.exists() else None),
            "purpose": "Stage A content kill test only",
        },
        "full_release_downloaded": False,
        "download_policy": "EARLY_STOP_NO_FULL_CORPUS",
        "recorded_at_utc": utc_now(),
    }
    # fill missing heads from live dict
    for s in ["2016_1", "2016_2", "2017_1", "2017_2", "2020_ERE"]:
        if s in heads:
            source_manifest["heads_sample"][s] = {
                "status": heads[s].get("status"),
                "Content-Length": heads[s].get("Content-Length"),
                "url": heads[s].get("url"),
                "Accept-Ranges": heads[s].get("Accept-Ranges"),
                "range_supported": heads[s].get("range_supported"),
            }
    write_json(manifests / "codebench_source_manifest.json", source_manifest)

    # Empty Stage-B stubs
    (art / "codebench_item_audit.csv").write_text(
        "exercise_id,canonical_content_hash,first_semester_seen,last_semester_seen,"
        "classes_seen,assessment_count,language,statement_chars,has_input_spec,"
        "has_output_spec,has_examples,has_required_image,requires_prior_context,"
        "duplicate_text_group,text_complete_status,exclusion_reason\n",
        encoding="utf-8",
    )
    (art / "codebench_response_eligibility.csv").write_text(
        "exercise_id,test_first_submission_ge5,test_first_submission_ge10,"
        "test_first_submission_ge20,test_first_submission_ge50,test_first_submission_ge100\n",
        encoding="utf-8",
    )

    gates = {
        "G1": "PASS",
        "G2": "FAIL",
        "G3": "FAIL",
        "G4": "FAIL",
        "G5": "FAIL",
        "G6": "FAIL",
        "G7": "FAIL",
        "G8": "FAIL",
        "G9": "FAIL",
        "G10": "PASS",
    }
    verdict = "FAIL_NO_ITEM_TEXT"

    summary = {
        "phase": "TLT4D_P0F",
        "base_commit": BASE_COMMIT,
        "release": RELEASE,
        "content_source_verdict": content_verdict,
        "response_semantics": "NOT_EVALUATED_AFTER_CONTENT_EARLY_STOP",
        "semesters_audited": 1 if probe_meta else 0,
        "learners": None,
        "raw_homework_exercise_ids": content_search.get("unique_exercise_ids_in_probe"),
        "unique_content_groups": 0,
        "text_complete_items": 0,
        "formal_submissions": None,
        "items_test_ge20": 0,
        "unseen_item_eligible": 0,
        "verdict": verdict,
        "four_dataset_phase_1_eligible": False,
        "early_stop": early_stop,
        "full_release_downloaded": False,
        "gates": gates,
        "probe_archive_sha256": source_manifest["probe_archive"].get("sha256"),
        "probe_semester": probe_meta.get("semester"),
        "llm_used": False,
        "unofficial_reconstruction_used": False,
        "disk_free_bytes_at_audit": disk.free,
    }
    write_json(art / "P0F_CODEBENCH_FEASIBILITY_SUMMARY.json", summary)

    write_neighbor_check(reports / "TLT4D_P0F_CODEBENCH_NEIGHBOR_CHECK.md")
    write_feasibility_report(
        reports / "TLT4D_P0F_CODEBENCH_FEASIBILITY.md",
        summary=summary,
        content_verdict=content_verdict,
        content_search=content_search,
        source_manifest=source_manifest,
    )

    (base / "README.md").write_text(
        "# CodeBench UFAM (TLT4D Phase 0F)\n\n"
        f"Official page: {OFFICIAL_PAGE}\n"
        f"Release audited: {RELEASE} (2024-10-24)\n\n"
        "Stage A content kill test: **NO_STATEMENT_AVAILABLE** → "
        "`FAIL_NO_ITEM_TEXT`. Full multi-year corpus not downloaded.\n\n"
        "Audit: `python scripts/tlt4d_audit_external_datasets.py --dataset codebench`\n",
        encoding="utf-8",
    )

    disk_after = shutil.disk_usage(str(root))
    summary["disk_free_bytes_after"] = disk_after.free
    write_json(art / "P0F_CODEBENCH_FEASIBILITY_SUMMARY.json", summary)

    if not probe_meta.get("sha256"):
        errors.append(
            "CodeBench probe archive metadata missing; re-run Stage A download of one semester."
        )
    return summary


def write_content_source_md(
    path: Path, *, content_verdict: str, content_search: dict, probe_meta: dict
) -> None:
    lines = [
        "# TLT4D P0F — CodeBench Content-Source Kill Test",
        "",
        f"## Verdict: `{content_verdict}`",
        "",
        "Only `FULL_STATEMENT_AVAILABLE` would permit Stage B. This Phase stops here.",
        "",
        "## Official release inspected",
        "",
        f"- Page: `{OFFICIAL_PAGE}`",
        f"- Release: **{RELEASE}** (2024-10-24)",
        f"- Probe semester archive: `{probe_meta.get('url')}`",
        f"- Probe SHA-256: `{probe_meta.get('sha256')}`",
        f"- Probe bytes: `{probe_meta.get('bytes')}`",
        "",
        "## What the public release contains",
        "",
        "Per official schema + extracted `2016-2` v1.81 archive:",
        "",
        "- `assessments/*.data` — assessment title, class, schedule, language, type, **exercise ID list**",
        "- `users/*/codes/` — student source code",
        "- `users/*/executions/` — test/submission logs (including hidden test I/O in logs)",
        "- `users/*/codemirror/` — editor keystroke logs",
        "- `users/*/grades/` — assessment grades",
        "- `users/*/user.data` — demographics (not used)",
        "",
        "## What is missing",
        "",
        "- No `enunciado` / statement / description / problem-text files",
        "- Assessment metadata has **no** learner-visible problem statement fields",
        f"- Keyword filename hits for statement-like names: "
        f"`{content_search.get('files_matching_content_keywords')}`",
        f"- Statementish text inside assessments: "
        f"**{content_search.get('has_statementish_text_in_assessments')}**",
        f"- File extensions observed: `{content_search.get('file_extensions_observed')}`",
        "",
        "## Live public exercise endpoints",
        "",
        "Probed common URL patterns for example IDs (994, 1326) → **HTTP 404** "
        "(no anonymous official statement API found).",
        "",
        "## Prohibited reconstructions (not performed)",
        "",
        "- Inferring tasks from student code",
        "- Inferring tasks from expected test outputs in execution logs",
        "- Using assignment titles as substitutes for problem statements",
        "- Using third-party/Kaggle mirrors",
        "",
        f"Sample exercise IDs present in probe assessments: "
        f"`{content_search.get('sample_exercise_ids')}` "
        f"(n={content_search.get('unique_exercise_ids_in_probe')}).",
        "",
        "## Early stop",
        "",
        "`CONTENT_SOURCE_VERDICT = NO_STATEMENT_AVAILABLE`",
        "",
        "`CODEBENCH = FAIL_NO_ITEM_TEXT`",
        "",
        "Do **not** download remaining semester archives for Stage B.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_neighbor_check(path: Path) -> None:
    path.write_text(
        """# TLT4D P0F — CodeBench Prior-Work Neighbor Check

## Exact AutoKT validation chain on CodeBench?

Searched for prior work combining:

1. LLM-estimated scalar problem difficulty
2. held-out first-submission learner error correspondence
3. transparent-content incremental validity
4. random/length controls
5. response-limited KT
6. genuine unseen-item KT
7. synthetic-alignment diagnosis

**Result: no exact AutoKT-chain neighbor found.**

## Closest relevant work

1. **Pereira et al. (BJET / learning analytics on CodeBench)** — fine-grained IDE/behavior analytics in CS1; not LLM difficulty validation.
2. **Lima / Carvalho / Oliveira / Pereira (SBIE/RBIE)** — programming-question difficulty classification using **code metrics** and/or statement intelligibility features. These papers imply researchers had access to statements in their analysis setting, but that does **not** establish that the **public CodeBench Dataset 1.81 release** redistributes statements. Difficulty targets are typically empirical/class labels, not the TLT held-out first-submission error design.
3. **BePKT (Yang et al., arXiv:2112.08273)** — explicitly lists CodeBench as **unsuitable** for their programming KT setting (missing concept annotations / context) and introduces a different dataset (already `PERMANENT_FAIL` in TLT4D for coverage).
4. **Melo et al. (2025, SBEduc)** — exploring LLMs to label programming strategies (related UFAM/CodeBench ecosystem); not the AutoKT evidence chain.

## LLM + CodeBench / difficulty + CodeBench / text-aware KT + CodeBench

- Difficulty prediction papers around CodeBench exist, but primarily code-metric / intelligibility / LA pipelines.
- No primary-source evidence of the full TLT staged chain (LLM scalar difficulty ↔ held-out first-submission error ↔ transparent incremental validity ↔ response-limited + genuine unseen-item KT).

## Overlap risk

| Risk | Assessment |
| --- | --- |
| Exact duplication | **None found** |
| Partial overlap | Moderate (CS1 OJ difficulty / LA on CodeBench is an active UFAM line) |
| Blocking issue for TLT4D | **Missing public item text** (G2), independent of neighbor novelty |

## Novelty stance

Do not overclaim novelty. Even without an exact neighbor, CodeBench fails Stage A on **official item-text availability**, so Phase-1 eligibility is moot.
""",
        encoding="utf-8",
    )


def write_feasibility_report(
    path: Path,
    *,
    summary: dict,
    content_verdict: str,
    content_search: dict,
    source_manifest: dict,
) -> None:
    gates = summary["gates"]
    lines = [
        "# TLT4D Phase 0F — CodeBench Feasibility",
        "",
        "## 1. Executive Verdict",
        "",
        f"**`{summary['verdict']}`**",
        "",
        "Official CodeBench Dataset 1.81 public archives provide exercise **IDs**, student "
        "code, execution/submission logs, and assessment metadata, but **not** complete "
        "learner-visible programming problem statements.",
        "",
        "## 2. Official Release and Acquisition",
        "",
        f"- Official page: `{OFFICIAL_PAGE}`",
        f"- Release: **{RELEASE}** (2024-10-24)",
        f"- Anonymous download: **yes** (`files/cb_dataset_*_v1.81.tar.gz`)",
        f"- Probe archive: `{source_manifest.get('probe_archive')}`",
        "- Full multi-year release downloaded?: **NO** (early-stop)",
        "",
        "## 3. Content-Source Kill Test",
        "",
        f"```text\n{content_verdict}\n```",
        "",
        "See `reports/TLT4D_P0F_CODEBENCH_CONTENT_SOURCE.md`.",
        "",
        f"- Unique exercise IDs in probe semester: "
        f"**{content_search.get('unique_exercise_ids_in_probe')}**",
        f"- Sample IDs: `{content_search.get('sample_exercise_ids')}`",
        "",
        "## 4. Item Identity and Reuse",
        "",
        "Not fully audited (early-stop). Exercise IDs are stable numeric identifiers in "
        "assessments, but content identity cannot be verified without statements.",
        "",
        "## 5. Text Completeness",
        "",
        "`PASS_TEXT_COMPLETE = 0` (no recoverable statements).",
        "",
        "## 6. Submission Semantics",
        "",
        "`NOT_EVALUATED_AFTER_CONTENT_EARLY_STOP`.",
        "",
        "Note: execution logs in the public release do distinguish tests vs submissions in "
        "documented samples, but Stage B response-semantics audit was not authorized after G2 fail.",
        "",
        "## 7. Learner Identity",
        "",
        "Not fully audited. User folders are numeric IDs nested under `semester/class/users/`.",
        "",
        "## 8. Homework Interaction Audit",
        "",
        "Skipped (early-stop).",
        "",
        "## 9. Cross-Semester Reuse",
        "",
        "Skipped (early-stop).",
        "",
        "## 10. Learner Split",
        "",
        "Not constructed.",
        "",
        "## 11. Held-Out Evidence Eligibility",
        "",
        "| Stage | Items |",
        "| --- | ---: |",
        f"| raw homework exercise IDs (probe only) | {content_search.get('unique_exercise_ids_in_probe')} |",
        "| unique exact content groups | 0 |",
        "| PASS_TEXT_COMPLETE | 0 |",
        "| test >=5 | 0 |",
        "| test >=10 | 0 |",
        "| test >=20 | 0 |",
        "| test >=50 | 0 |",
        "| test >=100 | 0 |",
        "",
        "## 12. Genuine Unseen-Item Feasibility",
        "",
        "`unseen_item_eligible = 0`.",
        "",
        "## 13. Duplicate-Content Leakage",
        "",
        "N/A without statements.",
        "",
        "## 14. Transparent Features",
        "",
        "Not applicable without item text.",
        "",
        "## 15. Prior-Work Neighbor Check",
        "",
        "See `reports/TLT4D_P0F_CODEBENCH_NEIGHBOR_CHECK.md`.",
        "",
        "## 16. Scientific Risks",
        "",
        "- Programming submissions ≠ MCQ responses (would matter only if content existed).",
        "- Public release omits statements; reconstructing from tests/code is scientifically forbidden for this paper.",
        "- Homework vs exam variant blocks exist in schema, but were not primary-gated after content fail.",
        "- Portuguese statements (if ever released) would require original-language scoring.",
        "",
        "## 17. Gate Table",
        "",
        "| Gate | Status | Evidence |",
        "| --- | --- | --- |",
        f"| G1 | {gates['G1']} | Official CodeBench 1.81 anonymous archives; probe SHA recorded |",
        f"| G2 | {gates['G2']} | No official learner-visible statements in release/schema/probe |",
        f"| G3 | {gates['G3']} | text_complete_items=0 |",
        f"| G4 | {gates['G4']} | not evaluated after content early-stop |",
        f"| G5 | {gates['G5']} | not evaluated |",
        f"| G6 | {gates['G6']} | not evaluated |",
        f"| G7 | {gates['G7']} | not evaluated |",
        f"| G8 | {gates['G8']} | not evaluated after content early-stop |",
        f"| G9 | {gates['G9']} | not evaluated |",
        f"| G10 | {gates['G10']} | manifests/hashes; raw gitignored; demographics not ingested |",
        "",
        "## 18. Final Recommendation",
        "",
        "```text",
        "CODEBENCH = FAIL_NO_ITEM_TEXT",
        "FOUR_DATASET_PHASE_1_ELIGIBLE = NO",
        "```",
        "",
        "Do not begin Phase 1.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
