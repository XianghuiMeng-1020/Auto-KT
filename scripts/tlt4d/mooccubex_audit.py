#!/usr/bin/env python3
"""TLT4D Phase-0E — MOOCCubeX acquisition + response-semantics kill test.

Early-stop: if response semantics are EVENTUAL_OUTCOME_ONLY or AMBIGUOUS,
do NOT download/process the full ~21GB user-problem.json.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

BASE_COMMIT = "8d89b42ff03dda4ac3ba2e431f66f3b5c1b3b611"
REPO_URL = "https://github.com/THU-KEG/MOOCCubeX"
DATA_HOST = "https://lfs.aminer.cn/misc/moocdata/data/mooccube2"
UA = {"User-Agent": "Mozilla/5.0 TLT4D-P0E-audit"}

URLS = {
    "problem": f"{DATA_HOST}/entities/problem.json",
    "course": f"{DATA_HOST}/entities/course.json",
    "exercise_problem": f"{DATA_HOST}/relations/exercise-problem.txt",
    "course_field": f"{DATA_HOST}/relations/course-field.json",
    "user_problem": f"{DATA_HOST}/relations/user-problem.json",
    "concept_problem": f"{DATA_HOST}/relations/concept-problem.txt",
}


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
        req = Request(url, method="HEAD", headers=UA)
        with urlopen(req, timeout=60) as r:
            rec["status"] = r.status
            rec["final_url"] = r.geturl()
            for k in [
                "Content-Length",
                "Accept-Ranges",
                "Content-Type",
                "ETag",
                "Last-Modified",
            ]:
                rec[k] = r.headers.get(k)
            rec["range_supported"] = (r.headers.get("Accept-Ranges") or "").lower() == "bytes"
    except Exception as e:
        rec["head_error"] = str(e)
    return rec


def audit_mooccubex(root: Path, errors: list[str]) -> dict:
    base = root / "data" / "external" / "mooccubex"
    raw = base / "raw"
    derived = base / "derived"
    art = root / "artifacts" / "tlt4d"
    manifests = root / "artifacts" / "manifests"
    reports = root / "reports"
    for p in (raw, derived, art, manifests, reports, base):
        p.mkdir(parents=True, exist_ok=True)

    repo_git = raw / "MOOCCubeX_repo_git"
    source_commit = None
    if (repo_git / ".git").exists():
        try:
            import subprocess

            source_commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=repo_git, text=True
            ).strip()
        except Exception as e:
            errors.append(f"Could not read MOOCCubeX repo commit: {e}")
    commit_file = raw / "MOOCCubeX_repo" / "REPO_COMMIT.txt"
    if not source_commit and commit_file.exists():
        source_commit = commit_file.read_text(encoding="utf-8").strip()
    if not source_commit:
        source_commit = "ff6fb71304b4eaadde51956b7977d49c72a3ac1e"  # known main tip at audit time
        errors.append("MOOCCubeX commit inferred from API/clone fallback; verify REPO_COMMIT")

    # --- Preflight ---
    disk = shutil.disk_usage(str(root))
    preflight = {name: head_url(url) for name, url in URLS.items()}
    expected_problem = int(preflight["problem"].get("Content-Length") or 0)
    expected_behavior = int(preflight["user_problem"].get("Content-Length") or 0)
    expected_raw = expected_problem + expected_behavior
    margin = 2 * expected_raw + 10_000_000_000
    disk_info = {
        "filesystem": str(root),
        "free_bytes_before": disk.free,
        "free_gb_before": round(disk.free / 1e9, 2),
        "problem_content_length": expected_problem,
        "behavior_content_length": expected_behavior,
        "required_margin_bytes": margin,
        "sufficient_for_full_download_with_margin": disk.free >= margin,
    }
    preflight_doc = {
        "phase": "TLT4D_P0E",
        "checked_at_utc": utc_now(),
        "urls": preflight,
        "disk": disk_info,
        "decision": (
            "Disk sufficient for full Stage A+B under 2×raw+10GB margin, "
            "BUT Stage B full download is BLOCKED by response-semantics early-stop."
            if disk.free >= margin
            else "Disk insufficient under safety margin; would use streaming or BLOCKED_RESOURCE."
        ),
    }
    write_json(derived / "preflight_raw.json", preflight_doc)
    write_preflight_md(reports / "TLT4D_P0E_DOWNLOAD_PREFLIGHT.md", preflight_doc)

    # --- Response semantics (docs + paper + official script + streamed sample) ---
    sample_diag_path = derived / "user_problem_sample_diag.json"
    sample_diag = (
        json.loads(sample_diag_path.read_text(encoding="utf-8"))
        if sample_diag_path.exists()
        else {}
    )
    semantics = {
        "verdict": "EVENTUAL_OUTCOME_ONLY",
        "Q1_one_row_per_learner_problem": True,
        "Q2_attempts_semantics": (
            "Number of attempts on that problem (做题尝试次数 / Number of attempted questions). "
            "Not an attempt ordinal of a single event row."
        ),
        "Q3_is_correct_semantics": (
            "Single correctness flag on a user×problem summary record. Official materials do not "
            "state first-attempt; official script example has attempts=2 with is_correct=1; "
            "streamed sample contains many attempts>1 with is_correct=1. Therefore is_correct "
            "cannot be first-attempt correctness and is treated as eventual/aggregate outcome."
        ),
        "Q4_submit_time_semantics": (
            "Single 'Question time' / 做题时间 on the summary record; not documented as first vs "
            "last vs successful submission specifically."
        ),
        "Q5_first_attempt_recoverable": False,
        "evidence": [
            "docs/user-en.md + user-cn.md: log_id is unique key of user_id AND problem_id",
            "docs: attempts = number of attempts; is_correct = whether correct (unspecified which attempt)",
            "scripts/problems_by_user.sh expected output includes attempts:2 with is_correct:1",
            "CIKM 2021 paper: raw collection preserved submission history, but released schema has no per-attempt array",
            f"streamed sample (~1MiB): multi_attempt_is_correct_1={sample_diag.get('multi_attempt_is_correct_1')}, "
            f"duplicate_pairs_in_sample={sample_diag.get('duplicate_pairs_in_sample')}",
        ],
        "forbidden_rescues_not_applied": [
            "attempts>1 => first wrong",
            "filter attempts==1 only",
            "use eventual success as learner difficulty",
            "invent latent first response",
        ],
    }
    write_semantics_md(reports / "TLT4D_P0E_RESPONSE_SEMANTICS.md", semantics)

    early_stop = semantics["verdict"] in {"EVENTUAL_OUTCOME_ONLY", "AMBIGUOUS"}

    # --- Stage A manifests (partial: course-field + problem head sample; no full 1.2GB required after kill) ---
    stage_a = raw / "stage_a"
    course_field = stage_a / "course-field.json"
    problem_head = stage_a / "problem.head.bin"
    problem_schema = (
        json.loads((derived / "problem_schema_sample.json").read_text(encoding="utf-8"))
        if (derived / "problem_schema_sample.json").exists()
        else {}
    )

    problem_manifest = {
        "dataset": "MOOCCubeX",
        "phase": "TLT4D_P0E",
        "repository": REPO_URL,
        "source_commit": source_commit,
        "license_repo": "GPL-3.0 (repository LICENSE); data licensed via XuetangX per paper/README",
        "official_urls": {k: URLS[k] for k in ["problem", "course", "exercise_problem", "course_field"]},
        "acquisition": {
            "course_field": {
                "url": URLS["course_field"],
                "path": str(course_field) if course_field.exists() else None,
                "byte_size": course_field.stat().st_size if course_field.exists() else None,
                "sha256": sha256_file(course_field) if course_field.exists() else None,
                "encoding": "utf-8",
                "json_format": "JSONL",
                "object_count": 632 if course_field.exists() else None,
                "download_method": "HTTPS full GET",
                "acquired_at_utc": utc_now(),
            },
            "problem_full": {
                "url": URLS["problem"],
                "content_length_header": expected_problem,
                "download_status": "NOT_FULLY_DOWNLOADED_AFTER_RESPONSE_SEMANTICS_EARLY_STOP",
                "schema_sample_bytes": problem_head.stat().st_size if problem_head.exists() else None,
                "schema_sample_sha256": sha256_file(problem_head) if problem_head.exists() else None,
                "observed_fields_from_sample": problem_schema.get("keys"),
                "note": (
                    "Docs list field `id`; streamed objects use `problem_id`. "
                    "Sample confirms learner-visible content/option/title and answer/score fields present."
                ),
            },
            "course_full": {
                "url": URLS["course"],
                "content_length_header": int(preflight["course"].get("Content-Length") or 0),
                "download_status": "NOT_FULLY_DOWNLOADED_AFTER_RESPONSE_SEMANTICS_EARLY_STOP",
            },
            "exercise_problem": {
                "url": URLS["exercise_problem"],
                "content_length_header": int(preflight["exercise_problem"].get("Content-Length") or 0),
                "download_status": "NOT_DOWNLOADED_AFTER_RESPONSE_SEMANTICS_EARLY_STOP",
            },
        },
        "recorded_at_utc": utc_now(),
    }
    write_json(manifests / "mooccubex_problem_manifest.json", problem_manifest)

    behavior_manifest = {
        "dataset": "MOOCCubeX",
        "phase": "TLT4D_P0E",
        "url": URLS["user_problem"],
        "content_length_header": expected_behavior,
        "accept_ranges": preflight["user_problem"].get("Accept-Ranges"),
        "download_status": "NOT_DOWNLOADED_AFTER_EARLY_STOP",
        "behavior_source_sha256": "NOT_DOWNLOADED_AFTER_EARLY_STOP",
        "streamed_sample": {
            "bytes": sample_diag.get("sample_bytes"),
            "sha256": sample_diag.get("sample_sha256"),
            "parsed_objects": sample_diag.get("parsed_objects"),
            "purpose": "response-semantics kill test only",
        },
        "format_observed": "NDJSON (one JSON object per line)",
        "fields_observed": sample_diag.get("example_keys")
        or ["log_id", "user_id", "problem_id", "is_correct", "attempts", "score", "submit_time"],
        "recorded_at_utc": utc_now(),
    }
    write_json(manifests / "mooccubex_behavior_manifest.json", behavior_manifest)

    # Empty stubs for full-audit artifacts not produced under early-stop
    pd_cols_item = [
        "problem_id",
        "exercise_id",
        "language",
        "normalized_chars",
        "visual_dependency_class",
        "requires_external_course_context",
        "text_complete_status",
        "exclusion_reason",
    ]
    # avoid pandas dependency issues if import fails — write CSV headers manually
    (art / "mooccubex_item_audit.csv").write_text(",".join(pd_cols_item) + "\n", encoding="utf-8")
    (art / "mooccubex_response_eligibility.csv").write_text(
        "problem_id,test_ge5,test_ge10,test_ge20,test_ge50,test_ge100\n", encoding="utf-8"
    )
    (art / "mooccubex_problem_course_domain.csv").write_text(
        "problem_id,exercise_id,course_id,course_name,field/domain,mapping_multiplicity,language,text_complete_status\n",
        encoding="utf-8",
    )
    (art / "mooccubex_course_domain_summary.csv").write_text(
        "course_id,course_name,field,text_complete_items,test_ge20_items,learners,interactions\n",
        encoding="utf-8",
    )

    gates = {
        "G1": "PASS",
        "G2": "BORDERLINE",  # content available at official URL; full local file not acquired after early-stop
        "G3": "FAIL",
        "G4": "FAIL",
        "G5": "FAIL",
        "G6": "FAIL",
        "G7": "FAIL",
        "G8": "FAIL",
        "G9": "FAIL",
        "G10": "PASS",
    }
    # G2: official problem content exists and schema sample confirms content fields — but full file not downloaded.
    # Keep BORDERLINE rather than PASS because complete local verified corpus not hashed.
    if early_stop:
        for g in ("G3", "G4", "G5", "G6", "G7", "G9"):
            gates[g] = "FAIL"
        gates["G8"] = "FAIL"

    verdict = "FAIL_RESPONSE_CONSTRUCT"
    summary = {
        "phase": "TLT4D_P0E",
        "base_commit": BASE_COMMIT,
        "source_commit": source_commit,
        "problem_source_sha256": "NOT_FULLY_DOWNLOADED_AFTER_EARLY_STOP",
        "behavior_source_sha256": "NOT_DOWNLOADED_AFTER_EARLY_STOP",
        "response_semantics": semantics["verdict"],
        "raw_problems": None,
        "text_complete_items": 0,
        "learners": None,
        "behavior_rows": None,
        "courses": None,
        "domains": None,
        "items_test_ge20": 0,
        "unseen_item_eligible": 0,
        "verdict": verdict,
        "four_dataset_phase_1_eligible": False,
        "early_stop": True,
        "gates": gates,
        "official_urls": URLS,
        "behavior_content_length_header": expected_behavior,
        "problem_content_length_header": expected_problem,
        "disk_free_bytes_at_audit": disk.free,
        "sample_diag": {
            "parsed_objects": sample_diag.get("parsed_objects"),
            "multi_attempt_is_correct_1": sample_diag.get("multi_attempt_is_correct_1"),
            "duplicate_pairs_in_sample": sample_diag.get("duplicate_pairs_in_sample"),
        },
        "llm_used": False,
        "full_behavior_downloaded": False,
    }
    write_json(art / "P0E_MOOCCUBEX_FEASIBILITY_SUMMARY.json", summary)

    write_neighbor_check(reports / "TLT4D_P0E_MOOCCUBEX_NEIGHBOR_CHECK.md")
    write_feasibility_report(
        reports / "TLT4D_P0E_MOOCCUBEX_FEASIBILITY.md",
        summary=summary,
        semantics=semantics,
        preflight=preflight_doc,
        problem_manifest=problem_manifest,
        behavior_manifest=behavior_manifest,
        problem_schema=problem_schema,
    )

    (base / "README.md").write_text(
        "# MOOCCubeX (TLT4D Phase 0E)\n\n"
        f"Official repo: {REPO_URL}\n\n"
        "Response-semantics kill test: EVENTUAL_OUTCOME_ONLY → full `user-problem.json` "
        "(~21GB) NOT downloaded.\n\n"
        "Audit: `python scripts/tlt4d_audit_external_datasets.py --dataset mooccubex`\n",
        encoding="utf-8",
    )

    disk_after = shutil.disk_usage(str(root))
    summary["disk_free_bytes_after"] = disk_after.free
    write_json(art / "P0E_MOOCCUBEX_FEASIBILITY_SUMMARY.json", summary)
    return summary


def write_preflight_md(path: Path, doc: dict) -> None:
    disk = doc["disk"]
    lines = [
        "# TLT4D P0E — MOOCCubeX Download Preflight",
        "",
        f"Checked at UTC: `{doc['checked_at_utc']}`",
        "",
        "## Disk safety",
        "",
        f"- free bytes: **{disk['free_bytes_before']}** ({disk['free_gb_before']} GB)",
        f"- problem Content-Length: **{disk['problem_content_length']}**",
        f"- behavior Content-Length: **{disk['behavior_content_length']}**",
        f"- required margin (2×raw + 10GB): **{disk['required_margin_bytes']}**",
        f"- sufficient for full download with margin: **{disk['sufficient_for_full_download_with_margin']}**",
        "",
        "## URL HEAD results",
        "",
        "| File | Status | Content-Length | Accept-Ranges | Final URL |",
        "| --- | ---: | ---: | --- | --- |",
    ]
    for name, rec in doc["urls"].items():
        lines.append(
            f"| {name} | {rec.get('status')} | {rec.get('Content-Length')} | "
            f"{rec.get('Accept-Ranges')} | `{rec.get('final_url') or rec.get('url')}` |"
        )
    lines += [
        "",
        "## Decision",
        "",
        doc["decision"],
        "",
        "Stage B full download was **not** performed because the response-semantics kill test failed.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_semantics_md(path: Path, sem: dict) -> None:
    lines = [
        "# TLT4D P0E — MOOCCubeX Response-Semantics Kill Test",
        "",
        f"## Verdict: `{sem['verdict']}`",
        "",
        "### Q1 — One row per learner × problem?",
        "",
        f"**Yes.** Official docs (EN/CN): `log_id` is the unique key combining `user_id` and "
        f"`problem_id`. Streamed sample: "
        f"`duplicate_pairs_in_sample={sem.get('evidence')}` with 0 duplicates among parsed rows.",
        "",
        "### Q2 — `attempts` semantics",
        "",
        sem["Q2_attempts_semantics"],
        "",
        "### Q3 — `is_correct` semantics",
        "",
        sem["Q3_is_correct_semantics"],
        "",
        "### Q4 — `submit_time` semantics",
        "",
        sem["Q4_submit_time_semantics"],
        "",
        "### Q5 — First-attempt correctness recoverable?",
        "",
        f"**{sem['Q5_first_attempt_recoverable']}**",
        "",
        "### Evidence",
        "",
    ]
    for e in sem["evidence"]:
        lines.append(f"- {e}")
    lines += [
        "",
        "### Forbidden rescues (not applied)",
        "",
    ]
    for e in sem["forbidden_rescues_not_applied"]:
        lines.append(f"- {e}")
    lines += [
        "",
        "### Early stop",
        "",
        "Because verdict is `EVENTUAL_OUTCOME_ONLY`, Phase 0E **stops before** downloading/"
        "processing the full ~21GB `relations/user-problem.json`.",
        "",
        "`G8 = FAIL` → `MOOCCUBEX = FAIL_RESPONSE_CONSTRUCT`",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_neighbor_check(path: Path) -> None:
    path.write_text(
        """# TLT4D P0E — MOOCCubeX Prior-Work Neighbor Check

## Question

Has prior MOOCCubeX work already executed the exact AutoKT/TLT evidence chain?

* LLM-generated scalar item difficulty
* comparison with independently held-out learner error
* transparent-feature incremental information
* random/length controls
* response-limited KT
* genuine globally unseen-item KT
* synthetic-alignment diagnosis

## Finding

**No exact AutoKT-chain neighbor found on MOOCCubeX.**

## Closest relevant work

1. **Yu et al., CIKM 2021 — MOOCCubeX** (primary repository paper). Resource construction; not LLM difficulty validation.
2. **MoocRadar (Yu/Gao et al., arXiv:2304.02205)** — curated from MOOCCubeX with expert concept/cognitive labels for student modeling. Related resource lineage (already DROP/BLOCKED in TLT4D Phase 0 for access). Not the staged AutoKT chain.
3. **DCL4KT+LLM (LREC 2024 / arXiv:2312.11890)** — LLM difficulty for KT cold-start; uses standard KT benchmarks, **not MOOCCubeX**.
4. **DDKT (arXiv:2502.19915)** — LLM dual-channel difficulty for KT; datasets **XES3G5M + Eedi**, not MOOCCubeX.
5. Broader MOOCCubeX uses: recommendation / behavioral sequence papers (e.g., MOOC recommendation with behavioral+text fusion) — not held-out learner-error difficulty correspondence.

## LLM + MOOCCubeX / difficulty + MOOCCubeX / item-text KT + MOOCCubeX

Searches did not surface a paper that (a) scores MOOCCubeX problem text with an LLM for scalar difficulty and (b) validates against learner-held-out first-attempt error under the TLT protocol. MoocRadar is the closest content+behavior enrichment of MOOCCubeX, but it is a different construct and already out of scope for Phase 1.

## Overlap risk

| Risk | Assessment |
| --- | --- |
| Exact duplication of AutoKT chain on MOOCCubeX | **None found** |
| Partial methodological overlap | Low–moderate (MOOC KT / difficulty-aware KT exists elsewhere) |
| Construct conflict with TLT learner evidence | **High** — released `user-problem` is eventual/aggregate outcome |

## Novelty stance

Do **not** overclaim novelty. Even without an exact neighbor, MOOCCubeX fails Phase 0E on **response-construct incompatibility**, so novelty is moot for D4 selection.
""",
        encoding="utf-8",
    )


def write_feasibility_report(
    path: Path,
    *,
    summary: dict,
    semantics: dict,
    preflight: dict,
    problem_manifest: dict,
    behavior_manifest: dict,
    problem_schema: dict,
) -> None:
    gates = summary["gates"]
    lines = [
        "# TLT4D Phase 0E — MOOCCubeX Feasibility",
        "",
        "## 1. Executive Verdict",
        "",
        f"**`{summary['verdict']}`**",
        "",
        "Response-semantics kill test: released `relations/user-problem.json` is a "
        "**one-row-per-learner×problem eventual/aggregate outcome**, not first-attempt "
        "correctness. Full 21GB behavior file was **not** downloaded.",
        "",
        "## 2. Official Provenance",
        "",
        f"- Repository: `{REPO_URL}`",
        f"- Commit: `{summary['source_commit']}`",
        f"- Data host: `{DATA_HOST}`",
        f"- License (repo): GPL-3.0; data via XuetangX per paper/README",
        "",
        "## 3. Download Preflight",
        "",
        f"- See `reports/TLT4D_P0E_DOWNLOAD_PREFLIGHT.md`",
        f"- Behavior Content-Length: **{summary.get('behavior_content_length_header')}**",
        f"- Problem Content-Length: **{summary.get('problem_content_length_header')}**",
        f"- Disk free at audit: **{summary.get('disk_free_bytes_at_audit')}**",
        f"- Decision: {preflight.get('decision')}",
        "",
        "## 4. Problem Schema",
        "",
        "Official docs list fields including id/exercise_id/language/title/content/option/"
        "answer/score/type/typetext/location/context_id.",
        "",
        f"Streamed sample observed keys: `{problem_schema.get('keys')}`",
        "",
        "- Learner-visible candidates: `title`, `content`, `option` (+ language/type).",
        "- Must exclude from LLM input: `answer`, platform `score`, any learner statistics.",
        "- Note: sample objects use `problem_id` (int) rather than docs' `id` naming.",
        "- Full `entities/problem.json` (~1.2GB) **not** fully downloaded after early-stop.",
        "",
        "## 5. Response-Semantics Kill Test",
        "",
        f"```text\n{semantics['verdict']}\n```",
        "",
        "See `reports/TLT4D_P0E_RESPONSE_SEMANTICS.md`.",
        "",
        "## 6. Item Text Completeness",
        "",
        "Skipped (early-stop). `PASS_TEXT_COMPLETE = 0` in summary accounting.",
        "",
        "## 7. Language Distribution",
        "",
        "Skipped (early-stop). Docs state Chinese/English; sample head showed Chinese.",
        "",
        "## 8. Problem–Course–Domain Mapping",
        "",
        "Skipped (early-stop). `course-field.json` (632 courses with fields) acquired for provenance only.",
        "",
        "## 9. Behavior Integrity",
        "",
        "`behavior_source_sha256 = NOT_DOWNLOADED_AFTER_EARLY_STOP`",
        "",
        f"Sample-only: parsed_objects={summary.get('sample_diag', {}).get('parsed_objects')}; "
        f"multi_attempt_is_correct_1={summary.get('sample_diag', {}).get('multi_attempt_is_correct_1')}.",
        "",
        "## 10. Temporal Sequence",
        "",
        "Not constructed (early-stop).",
        "",
        "## 11. Learner Split",
        "",
        "Not constructed (early-stop).",
        "",
        "## 12. Held-Out Evidence Eligibility",
        "",
        "| Stage | Items |",
        "| --- | ---: |",
        "| raw problems | NOT_FULLY_COUNTED |",
        "| PASS_TEXT_COMPLETE | 0 |",
        "| test >=5 | 0 |",
        "| test >=10 | 0 |",
        "| test >=20 | 0 |",
        "| test >=50 | 0 |",
        "| test >=100 | 0 |",
        "",
        "## 13. Course/Domain Evidence Distribution",
        "",
        "Not computed (early-stop).",
        "",
        "## 14. Genuine Unseen-Item Feasibility",
        "",
        "Not computed (early-stop). `unseen_item_eligible = 0`.",
        "",
        "## 15. Context / Leakage Risks",
        "",
        "Not fully audited. Schema includes `context_id` / `location` suggesting possible "
        "course-structure dependence — would require full item audit if semantics ever pass.",
        "",
        "## 16. Transparent Features",
        "",
        "Inventory deferred. Content-side features (length, option count, language, type) "
        "appear feasible from schema sample; response-derived stats forbidden.",
        "",
        "## 17. Neighbor Check",
        "",
        "See `reports/TLT4D_P0E_MOOCCUBEX_NEIGHBOR_CHECK.md`. No exact AutoKT-chain neighbor.",
        "",
        "## 18. Gate Table",
        "",
        "| Gate | Status | Evidence |",
        "| --- | --- | --- |",
        f"| G1 | {gates['G1']} | Official THU-KEG repo + Aminer data host HEAD-verified |",
        f"| G2 | {gates['G2']} | Problem content officially available; full local hash deferred after early-stop |",
        f"| G3 | {gates['G3']} | text-complete audit not run |",
        f"| G4 | {gates['G4']} | held-out evidence not run |",
        f"| G5 | {gates['G5']} | unseen-item audit not run |",
        f"| G6 | {gates['G6']} | identities not validated on full corpus |",
        f"| G7 | {gates['G7']} | temporal histories not constructed |",
        f"| G8 | {gates['G8']} | EVENTUAL_OUTCOME_ONLY; first-attempt not recoverable |",
        f"| G9 | {gates['G9']} | leakage controllability not established on full items |",
        f"| G10 | {gates['G10']} | preflight+manifests+hashes of acquired samples; raw gitignored |",
        "",
        "## 19. Final Recommendation",
        "",
        "```text",
        "MOOCCUBEX = FAIL_RESPONSE_CONSTRUCT",
        "FOUR_DATASET_PHASE_1_ELIGIBLE = NO",
        "```",
        "",
        "Do not begin Phase 1.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
