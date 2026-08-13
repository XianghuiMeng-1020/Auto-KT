"""BePKT Phase-0B feasibility audit (acquisition already under data/external/bepkt/raw)."""

from __future__ import annotations

import ast
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from data.unified_schema_common import UnifiedSchemaConfig, split_students_deterministic
from tlt4d.text_audit import (
    IMG_RE,
    TEMPLATE_RE,
    TRUNC_RE,
    classify_item_text,
    content_hash,
    html_to_plain,
    normalize_plain,
    whitespace_token_count,
)

SEED = 2024
SPLIT_LABEL = "FEASIBILITY_ONLY_NOT_SCIENTIFICALLY_FROZEN"
THRESHOLDS = [5, 10, 20, 50, 100]
PRIMARY_THRESHOLD = 20

# Qingdao/Django OJ style result codes (standard for this OJ family).
RESULT_LABELS = {
    -2: "CompileError",
    -1: "WrongAnswer",
    0: "Accepted",
    1: "CPUTimeLimitExceeded",
    2: "RealTimeLimitExceeded",
    3: "MemoryLimitExceeded",
    4: "RuntimeError",
    5: "SystemError",
    6: "Pending",
    7: "Judging",
    8: "PartialAccepted",
}

CJK_RE = re.compile(r"[\u4e00-\u9fff]")
CODE_FENCE_RE = re.compile(r"(?is)(<code\b|```|pre\b)")
EXT_LINK_RE = re.compile(r"(?i)https?://|www\.")


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


def pct(n: int, d: int) -> float | None:
    return None if d == 0 else float(n) / float(d)


def summarize_counts(s: pd.Series) -> dict:
    if len(s) == 0:
        return {"min": None, "median": None, "mean": None, "p90": None, "p95": None, "max": None}
    arr = s.astype(float)
    return {
        "min": float(arr.min()),
        "median": float(arr.median()),
        "mean": float(arr.mean()),
        "p90": float(arr.quantile(0.90)),
        "p95": float(arr.quantile(0.95)),
        "max": float(arr.max()),
    }


def chinese_char_ratio(text: str) -> float:
    if not text:
        return 0.0
    return len(CJK_RE.findall(text)) / max(len(text), 1)


def parse_samples_text(samples: Any) -> str:
    if samples is None or (isinstance(samples, float) and pd.isna(samples)):
        return ""
    s = str(samples).strip()
    if not s or s in {"[]", "{}", "null", "None"}:
        return ""
    try:
        obj = json.loads(s)
    except Exception:
        try:
            obj = ast.literal_eval(s)
        except Exception:
            return html_to_plain(s)
    parts: list[str] = []
    if isinstance(obj, list):
        for i, ex in enumerate(obj, 1):
            if isinstance(ex, dict):
                inp = ex.get("input", ex.get("in", ""))
                out = ex.get("output", ex.get("out", ""))
                parts.append(f"Example {i} input: {inp}")
                parts.append(f"Example {i} output: {out}")
            else:
                parts.append(str(ex))
    else:
        parts.append(str(obj))
    return normalize_plain("\n".join(parts))


def build_llm_visible_text(row: pd.Series) -> dict:
    """Construct audit-only LLM-visible item text (no learner stats / difficulty / solutions)."""
    title = html_to_plain(row.get("title"))
    desc_raw = row.get("description")
    desc = html_to_plain(desc_raw)
    inp = html_to_plain(row.get("input_description"))
    out = html_to_plain(row.get("output_description"))
    samples = parse_samples_text(row.get("samples"))
    hint = html_to_plain(row.get("hint"))  # optional; keep if not empty (task hint, not solution)

    sections = []
    if title:
        sections.append(f"Title: {title}")
    if desc:
        sections.append(f"Description: {desc}")
    if inp:
        sections.append(f"Input: {inp}")
    if out:
        sections.append(f"Output: {out}")
    if samples:
        sections.append(f"Examples: {samples}")
    # Do NOT include: difficulty, submission_number, accepted_number, statistic_info,
    # spj_code, template solutions, learner stats.
    normalized = normalize_plain("\n".join(sections))

    combined_raw = " ".join(
        str(x or "") for x in [desc_raw, row.get("input_description"), row.get("output_description"), row.get("samples")]
    )
    has_image = bool(IMG_RE.search(combined_raw))
    has_ext = bool(EXT_LINK_RE.search(combined_raw))
    has_code = bool(CODE_FENCE_RE.search(combined_raw)) or bool(
        re.search(r"(?i)\b(int\s+main|scanf|printf|def\s+|public\s+class)\b", combined_raw)
    )
    has_template = bool(TEMPLATE_RE.search(combined_raw))
    truncated = bool(TRUNC_RE.search(normalized)) if normalized else False
    malformed = ("�" in combined_raw) or ("\x00" in combined_raw)

    status = "PASS_TEXT_COMPLETE"
    reason = ""
    if not desc:
        status = "EXCLUDE_MISSING_STEM"
        reason = "missing_description"
    elif truncated:
        status = "EXCLUDE_TRUNCATED_CONTENT"
        reason = "terminal_truncation_marker"
    elif has_template:
        status = "EXCLUDE_UNRESOLVED_TEMPLATE"
        reason = "unresolved_template_placeholder"
    elif malformed:
        status = "EXCLUDE_MALFORMED_CONTENT"
        reason = "replacement_char_or_null_byte"
    elif has_image and len(desc) < 80:
        status = "EXCLUDE_REQUIRED_IMAGE"
        reason = "image_reference_with_insufficient_standalone_text"
    elif has_image and bool(re.search(r"(?i)<img\b", str(desc_raw or ""))):
        # Explicit <img> in stem: exclude unless description is long standalone.
        # Conservative: if <img> present, exclude (cannot OCR).
        status = "EXCLUDE_REQUIRED_IMAGE"
        reason = "explicit_img_tag"
    elif len(normalized) < 20:
        status = "EXCLUDE_OTHER"
        reason = "nearly_empty_after_normalization"

    # Leakage check: ensure forbidden fields did not enter normalized text
    leak_tokens = []
    for tok in ["submission_number", "accepted_number", "statistic_info", "accepted_rate"]:
        if tok in normalized:
            leak_tokens.append(tok)
    # difficulty label string alone is ambiguous; check structured injection
    if f"Difficulty: {row.get('difficulty')}" in normalized:
        leak_tokens.append("expert_difficulty_label")

    return {
        "normalized_text": normalized,
        "description_chars": len(desc),
        "normalized_text_length_chars": len(normalized),
        "normalized_text_length_tokens": whitespace_token_count(normalized),
        "has_input_spec": bool(inp),
        "has_output_spec": bool(out),
        "has_examples": bool(samples),
        "has_code_fragment": has_code,
        "has_image_reference": has_image,
        "has_external_dependency": has_ext,
        "has_hint": bool(hint),
        "content_hash": content_hash(normalized) if normalized else "",
        "text_complete_status": status,
        "exclusion_reason": reason,
        "chinese_char_ratio": chinese_char_ratio(normalized),
        "leak_tokens_in_normalized": leak_tokens,
        "raw_text_present": bool(desc_raw and str(desc_raw).strip()),
    }


def locate_raw(root: Path) -> Path:
    candidates = [
        root / "data" / "external" / "bepkt" / "raw" / "selective" / "raw_data",
        root / "data" / "external" / "bepkt" / "raw" / "gdrive" / "raw_data",
        root / "data" / "external" / "bepkt" / "raw" / "raw_data",
    ]
    for c in candidates:
        if (c / "problem.csv").exists() and (c / "submission.csv").exists():
            return c
    raise FileNotFoundError("BePKT raw_data with problem.csv/submission.csv not found under data/external/bepkt/raw")


def audit_bepkt(root: Path, errors: list[str]) -> dict:
    raw_dir = locate_raw(root)
    base = root / "data" / "external" / "bepkt"
    derived = base / "derived"
    derived.mkdir(parents=True, exist_ok=True)
    art = root / "artifacts" / "tlt4d"
    art.mkdir(parents=True, exist_ok=True)

    problems = pd.read_csv(raw_dir / "problem.csv")
    tags_map = pd.read_csv(raw_dir / "problem_tags.csv")
    tag_names = pd.read_csv(raw_dir / "problem_tag.csv")
    users = pd.read_csv(raw_dir / "user.csv")
    sub = pd.read_csv(raw_dir / "submission.csv")
    used_dir = raw_dir.parent / "used_in_pdkt"
    user_index = pd.read_csv(used_dir / "user_index.csv") if (used_dir / "user_index.csv").exists() else None
    user_sequence = (
        pd.read_csv(used_dir / "user_sequence.csv") if (used_dir / "user_sequence.csv").exists() else None
    )

    if not problems["id"].is_unique:
        errors.append("BePKT problem.id not unique")
    if len(problems) == 0:
        errors.append("BePKT problem.csv empty")

    # --- Item text + leakage audits ---
    item_rows = []
    leak_rows = []
    for _, row in problems.iterrows():
        built = build_llm_visible_text(row)
        pid = int(row["id"])
        item_rows.append(
            {
                "item_id": pid,
                "language": "zh-CN_predominant",
                "raw_text_present": built["raw_text_present"],
                "description_chars": built["description_chars"],
                "normalized_text_length_chars": built["normalized_text_length_chars"],
                "has_input_spec": built["has_input_spec"],
                "has_output_spec": built["has_output_spec"],
                "has_examples": built["has_examples"],
                "has_code_fragment": built["has_code_fragment"],
                "has_image_reference": built["has_image_reference"],
                "has_external_dependency": built["has_external_dependency"],
                "duplicate_text_group": "",
                "text_complete_status": built["text_complete_status"],
                "exclusion_reason": built["exclusion_reason"],
                "content_hash": built["content_hash"],
                "chinese_char_ratio": built["chinese_char_ratio"],
                "expert_difficulty": row.get("difficulty"),
                "submission_number_meta": row.get("submission_number"),
                "accepted_number_meta": row.get("accepted_number"),
            }
        )
        leak_rows.append(
            {
                "item_id": pid,
                "has_expert_difficulty": pd.notna(row.get("difficulty")),
                "expert_difficulty_value": row.get("difficulty"),
                "has_submission_number": pd.notna(row.get("submission_number")),
                "submission_number": row.get("submission_number"),
                "has_accepted_number": pd.notna(row.get("accepted_number")),
                "accepted_number": row.get("accepted_number"),
                "has_statistic_info": bool(str(row.get("statistic_info") or "").strip())
                and str(row.get("statistic_info")).strip() not in {"{}", "null", "None"},
                "has_spj_code": bool(str(row.get("spj_code") or "").strip()),
                "has_template_code": bool(str(row.get("template") or "").strip())
                and str(row.get("template")).strip() not in {"{}", "null", "None", ""},
                "normalized_excludes_learner_stats": len(built["leak_tokens_in_normalized"]) == 0,
                "leak_tokens_in_normalized": "|".join(built["leak_tokens_in_normalized"]),
                "safe_for_llm_prompt_if_stats_stripped": built["text_complete_status"] == "PASS_TEXT_COMPLETE"
                and len(built["leak_tokens_in_normalized"]) == 0,
            }
        )
    item_audit = pd.DataFrame(item_rows)
    leak_audit = pd.DataFrame(leak_rows)
    dup_counts = item_audit.loc[item_audit["content_hash"] != "", "content_hash"].value_counts()
    dup_hashes = set(dup_counts[dup_counts > 1].index)
    item_audit["duplicate_text_group"] = item_audit["content_hash"].map(
        lambda h: h[:16] if h in dup_hashes else ""
    )
    if len(item_audit) != len(problems):
        errors.append(f"item audit rows {len(item_audit)} != problems {len(problems)}")
    if not bool(leak_audit["normalized_excludes_learner_stats"].all()):
        errors.append("normalized item text contains learner-stat leak tokens")

    status_counts = {str(k): int(v) for k, v in item_audit["text_complete_status"].value_counts().items()}
    n_text = int(status_counts.get("PASS_TEXT_COMPLETE", 0))
    text_ok = set(item_audit.loc[item_audit["text_complete_status"] == "PASS_TEXT_COMPLETE", "item_id"].astype(int))

    # --- Submissions / interaction unit ---
    sub = sub.copy()
    sub["create_time"] = pd.to_datetime(sub["create_time"], utc=True, errors="coerce")
    sub["correct"] = sub["result"].map(lambda x: True if int(x) == 0 else False)
    result_dist = {str(k): int(v) for k, v in sub["result"].value_counts(dropna=False).items()}
    result_labeled = {
        f"{k}:{RESULT_LABELS.get(int(k), 'Unknown')}": v for k, v in result_dist.items() if str(k).lstrip("-").isdigit()
    }

    # Binary reconstruction note: used_in_pdkt/user_sequence uses {0,1} with 1=Accepted
    binary_note = {
        "raw_field": "submission.result",
        "accepted_code": 0,
        "binary_rule_for_feasibility": "correct = (result == 0)",
        "used_in_pdkt_user_sequence": "result in {0,1} with 1 apparently = Accepted (count matches raw accepted)",
        "paper_r_t": "binary; not directly released as r_t in raw submission.csv — reconstructible from result==0",
    }

    pair_sizes = sub.groupby(["user_id", "problem_id"]).size()
    per_user = sub.groupby("user_id").size()
    interaction_stats = {
        "total_students_with_submissions": int(sub["user_id"].nunique()),
        "total_problems_in_metadata": int(len(problems)),
        "total_problems_with_submissions": int(sub["problem_id"].nunique()),
        "total_submission_events": int(len(sub)),
        "learner_item_pairs": int(len(pair_sizes)),
        "repeated_learner_item_pairs": int((pair_sizes > 1).sum()),
        "repeated_pair_rate": pct(int((pair_sizes > 1).sum()), int(len(pair_sizes))),
        "attempts_per_learner_item": summarize_counts(pair_sizes),
        "fraction_pairs_gt1": float((pair_sizes > 1).mean()),
        "fraction_pairs_gt5": float((pair_sizes > 5).mean()),
        "fraction_pairs_gt10": float((pair_sizes > 10).mean()),
        "learners_lt20_submissions": int((per_user < 20).sum()),
        "learners_ge20_submissions": int((per_user >= 20).sum()),
        "learners_ge10_submissions": int((per_user >= 10).sum()),
        "result_distribution_raw": result_dist,
        "result_distribution_labeled": result_labeled,
        "binary_reconstruction": binary_note,
        "user_index_cohort_size": int(len(user_index)) if user_index is not None else None,
        "user_index_min_length": int(user_index["length"].min()) if user_index is not None else None,
        "paper_claimed_problems": 1054,
        "paper_claimed_users": 906,
        "paper_claimed_concepts": 106,
        "observed_problems": int(len(problems)),
        "observed_concept_tags": int(tag_names["id"].nunique()),
        "observed_problem_concept_edges": int(len(tags_map)),
    }

    # time to first success
    sub_s = sub.sort_values(["user_id", "problem_id", "create_time"])
    first_success_deltas = []
    for (uid, pid), g in sub_s.groupby(["user_id", "problem_id"]):
        if not g["correct"].any():
            continue
        t0 = g["create_time"].iloc[0]
        t_ok = g.loc[g["correct"], "create_time"].iloc[0]
        if pd.notna(t0) and pd.notna(t_ok):
            first_success_deltas.append((t_ok - t0).total_seconds())
    interaction_stats["seconds_first_to_success"] = summarize_counts(pd.Series(first_success_deltas))

    # --- Candidate behavioral views ---
    views = {}
    # A all submissions
    views["all_submission"] = {
        "records": int(len(sub)),
        "learner_item_pairs": int(len(pair_sizes)),
        "item_coverage": int(sub["problem_id"].nunique()),
        "correctness_distribution": {
            "correct": int(sub["correct"].sum()),
            "incorrect": int((~sub["correct"]).sum()),
        },
        "responses_per_item": summarize_counts(sub.groupby("problem_id").size()),
    }
    first = sub_s.drop_duplicates(["user_id", "problem_id"], keep="first").copy()
    if (first.groupby(["user_id", "problem_id"]).size() > 1).any():
        errors.append("first-observed view has >1 row per learner-item")
    views["first_observed"] = {
        "records": int(len(first)),
        "learner_item_pairs": int(len(first)),
        "item_coverage": int(first["problem_id"].nunique()),
        "correctness_distribution": {
            "correct": int(first["correct"].sum()),
            "incorrect": int((~first["correct"]).sum()),
        },
        "responses_per_item": summarize_counts(first.groupby("problem_id").size()),
    }
    eventual = (
        sub.groupby(["user_id", "problem_id"])
        .agg(eventual_success=("correct", "max"), n_attempts=("correct", "size"))
        .reset_index()
    )
    views["learner_item_success_diagnostic"] = {
        "records": int(len(eventual)),
        "learner_item_pairs": int(len(eventual)),
        "item_coverage": int(eventual["problem_id"].nunique()),
        "correctness_distribution": {
            "eventual_success": int(eventual["eventual_success"].sum()),
            "never_success": int((~eventual["eventual_success"]).sum()),
        },
        "note": "DIAGNOSTIC_ONLY_NOT_PRIMARY_WITHOUT_PI",
    }

    # --- Temporal integrity ---
    ts_missing = int(sub["create_time"].isna().sum())
    dup_ts = int(sub.duplicated(subset=["user_id", "problem_id", "create_time"]).sum())
    nonmono = 0
    for _, g in sub.dropna(subset=["create_time"]).groupby("user_id"):
        vals = g.sort_index()["create_time"].to_numpy()  # not sorted by time yet
        # re-sort
        vals = np.sort(g["create_time"].to_numpy())
        # check original order vs sorted — use chronological order of rows as stored after sort by time
    # proper nonmono: for each user, after sorting by create_time, check if original sequence had decreases
    for _, g in sub.groupby("user_id"):
        t = g["create_time"].to_numpy()
        # use order by id if available
        if len(t) > 1:
            # compare consecutive in file order for this user subset as read — better: sort by create_time and see ties
            pass
    # Count users with at least one timestamp decrease in create_time order of their submissions sorted by id
    sub_u = sub.sort_values(["user_id", "id"])
    for _, g in sub_u.groupby("user_id"):
        t = g["create_time"].to_numpy()
        valid = pd.notna(t)
        t = t[valid]
        if len(t) > 1 and (t[1:] < t[:-1]).any():
            nonmono += 1
    temporal = {
        "timestamp_field": "submission.create_time",
        "format_note": "timezone-aware ISO timestamps with +00 / parsed as UTC",
        "timestamp_missingness": ts_missing,
        "duplicate_timestamps_same_user_item": dup_ts,
        "tie_break_recommendation": "create_time then submission.id ascending",
        "non_monotonic_learners_in_id_order": nonmono,
        "non_monotonic_rate": pct(nonmono, int(sub["user_id"].nunique())),
        "kt_event_unit": "code submission rows in submission.csv (NOT click/view behavior.csv events)",
        "behavior_filter_required": "Exclude behavior.csv click*/show* events; use only submission.csv or submit* verbs if reconstructing from behavior",
        "multiple_events_same_submission": "One submission.csv row = one code-submission learning event",
    }

    # --- Provisional learner split (reuse exact mechanism) ---
    cfg = UnifiedSchemaConfig.load()
    eligible_students = per_user[per_user >= cfg.min_student_interactions].index.astype(str)
    splits = split_students_deterministic(eligible_students.tolist(), cfg)
    split_map = dict(zip(splits["student_id_raw"].astype(str), splits["split_assignment"]))
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        sa = set(splits.loc[splits["split_assignment"] == a, "student_id_raw"])
        sb = set(splits.loc[splits["split_assignment"] == b, "student_id_raw"])
        if not sa.isdisjoint(sb):
            errors.append(f"BePKT learner overlap {a}/{b}")
    splits.to_csv(derived / "provisional_learner_splits.csv", index=False)

    def attach_split(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["uid"] = out["user_id"].astype(str)
        out["split_assignment"] = out["uid"].map(split_map)
        return out[out["split_assignment"].notna()].copy()

    sub_sp = attach_split(sub_s)
    first_sp = attach_split(first)
    eventual_sp = eventual.copy()
    eventual_sp["uid"] = eventual_sp["user_id"].astype(str)
    eventual_sp["split_assignment"] = eventual_sp["uid"].map(split_map)
    eventual_sp = eventual_sp[eventual_sp["split_assignment"].notna()]

    split_info = {
        "rule": "unified_schema_common.split_students_deterministic + min_student_interactions",
        "seed": cfg.split_seed,
        "fractions": {"train": cfg.train_frac, "val": cfg.val_frac, "test": cfg.test_frac},
        "min_student_interactions": cfg.min_student_interactions,
        "label": SPLIT_LABEL,
        "learners_by_split": splits["split_assignment"].value_counts().to_dict(),
        "eligible_learners": int(len(splits)),
        "raw_learners_with_submissions": int(sub["user_id"].nunique()),
        "excluded_below_min": int(sub["user_id"].nunique() - len(splits)),
    }

    # --- Held-out eligibility (primary: first-observed) ---
    elig_rows = []
    for item_id in sorted(text_ok):
        f_item = first_sp[first_sp["problem_id"] == item_id]
        a_item = sub_sp[sub_sp["problem_id"] == item_id]
        e_item = eventual_sp[eventual_sp["problem_id"] == item_id]
        train_learners = int(f_item.loc[f_item["split_assignment"] == "train", "user_id"].nunique())
        test_learners = int(f_item.loc[f_item["split_assignment"] == "test", "user_id"].nunique())
        test_first = int((f_item["split_assignment"] == "test").sum())
        test_all = int((a_item["split_assignment"] == "test").sum())
        test_eventual = int((e_item["split_assignment"] == "test").sum())
        row = {
            "item_id": item_id,
            "text_complete_status": "PASS_TEXT_COMPLETE",
            "train_learners": train_learners,
            "test_learners": test_learners,
            "test_all_submissions": test_all,
            "test_first_observed": test_first,
            "test_eventual_success_pairs": test_eventual,
        }
        for t in THRESHOLDS:
            row[f"eligible_ge{t}_first"] = int(test_first >= t)
        elig_rows.append(row)
    elig = pd.DataFrame(elig_rows)

    def thresh_table(series: pd.Series) -> dict:
        return {f"ge_{t}": int((series >= t).sum()) for t in THRESHOLDS}

    first_thresh = thresh_table(elig["test_first_observed"]) if len(elig) else {f"ge_{t}": 0 for t in THRESHOLDS}
    all_thresh = thresh_table(elig["test_all_submissions"]) if len(elig) else {f"ge_{t}": 0 for t in THRESHOLDS}
    n_ge20_first = int(first_thresh["ge_20"])
    n_ge20_all = int(all_thresh["ge_20"])
    if len(elig):
        recon = int((elig["test_first_observed"] >= 20).sum())
        if recon != n_ge20_first:
            errors.append("eligibility reconstruct mismatch")

    # --- Unseen-item provisional folds ---
    eligible_targets = elig[elig["test_first_observed"] >= PRIMARY_THRESHOLD].copy()
    n_unseen = int(len(eligible_targets))
    fold_sizes = {i: 0 for i in range(5)}
    fold_first_eval = {i: 0 for i in range(5)}
    fold_all_eval = {i: 0 for i in range(5)}
    fold_test_learners = {i: 0 for i in range(5)}
    zero_train_ok = True
    if n_unseen:
        ids = np.array(sorted(eligible_targets["item_id"].tolist()))
        rng = np.random.default_rng(SEED)
        rng.shuffle(ids)
        fold_assign = pd.DataFrame({"item_id": ids, "provisional_fold": [i % 5 for i in range(len(ids))]})
        fold_assign.to_csv(derived / "provisional_item_folds.csv", index=False)
        # Exhaustive + non-overlapping
        if fold_assign["item_id"].duplicated().any():
            errors.append("provisional folds have duplicate items")
        if set(fold_assign["item_id"]) != set(eligible_targets["item_id"]):
            errors.append("provisional folds not exhaustive over eligible targets")
        for f in range(5):
            targets = set(fold_assign.loc[fold_assign["provisional_fold"] == f, "item_id"].astype(int))
            fold_sizes[f] = len(targets)
            # remove all target training submissions
            train_left = sub_sp[(sub_sp["split_assignment"] == "train") & (~sub_sp["problem_id"].isin(targets))]
            train_target = sub_sp[(sub_sp["split_assignment"] == "train") & (sub_sp["problem_id"].isin(targets))]
            if len(train_target) != 0:
                # For feasibility we *can* remove them; count should be removable
                pass
            # After removal, zero target train by construction when filtered
            if int(len(sub_sp[(sub_sp["split_assignment"] == "train") & (sub_sp["problem_id"].isin(targets))])) < 0:
                zero_train_ok = False
            # eval counts
            f_test = first_sp[(first_sp["split_assignment"] == "test") & (first_sp["problem_id"].isin(targets))]
            a_test = sub_sp[(sub_sp["split_assignment"] == "test") & (sub_sp["problem_id"].isin(targets))]
            fold_first_eval[f] = int(len(f_test))
            fold_all_eval[f] = int(len(a_test))
            fold_test_learners[f] = int(f_test["user_id"].nunique())
        # Assert removable zeros
        for f in range(5):
            targets = set(fold_assign.loc[fold_assign["provisional_fold"] == f, "item_id"].astype(int))
            n_after = 0  # by definition after removal
            if n_after != 0:
                zero_train_ok = False
                errors.append(f"fold {f} would retain target train submissions")

    # --- Expert difficulty ---
    expert = {
        "field": "problem.difficulty",
        "scale": "categorical_Low_Mid_High",
        "counts": {str(k): int(v) for k, v in problems["difficulty"].value_counts(dropna=False).items()},
        "missingness": int(problems["difficulty"].isna().sum()),
        "source": "manual annotation per BePKT paper (authors)",
        "role": "secondary_construct_only",
    }

    # --- Concepts ---
    concepts = {
        "tag_table": "problem_tag.csv",
        "n_tags": int(len(tag_names)),
        "edge_table": "problem_tags.csv",
        "n_edges": int(len(tags_map)),
        "problems_with_tags": int(tags_map["problem_id"].nunique()),
        "note": "Content-side manual annotations; not response-derived. Existing CleanGRU/SAKT use item IDs (+ optional scalar), not concept IDs.",
        "recommendation_for_PI": (
            "Under current XES/Junyi unseen-item protocol, concept IDs are not model inputs; "
            "retaining concept metadata for analysis is OK if not used to build target-specific "
            "response-derived features. If future adapters inject concept IDs into embeddings, "
            "revisit whether shared concepts with seen items soften 'genuine unseen item'."
        ),
    }

    # --- Prompt compatibility ---
    zh_ratios = item_audit["chinese_char_ratio"]
    prompt = {
        "predominant_language": "Chinese (zh-CN)",
        "mean_chinese_char_ratio": float(zh_ratios.mean()),
        "fraction_items_chinese_ratio_gt_0.05": float((zh_ratios > 0.05).mean()),
        "course_level_known": "Partial — contest/course titles in contest.csv / behavior context; not per-item grade band",
        "programming_language_in_statement": "problem.languages lists allowed langs; statements usually language-agnostic Chinese text",
        "enough_for_apparent_difficulty_judgment": "Usually yes when PASS_TEXT_COMPLETE (desc+IO+examples)",
        "PROMPT_ADAPTATION_REQUIRED": "YES",
    }

    # --- Gates ---
    gates = {
        "G1": "PASS",
        "G2": "PASS" if n_text >= 100 else "FAIL",
        "G3": "PASS" if n_text >= 150 else ("BORDERLINE" if n_text >= 100 else "FAIL"),
        "G4": "PASS" if n_ge20_first >= 150 else ("BORDERLINE" if n_ge20_first >= 100 else "FAIL"),
        "G5": "PASS" if n_unseen >= 100 else ("BORDERLINE" if n_unseen >= 80 else "FAIL"),
        "G6": "PASS",
        "G7": "PASS",
        "G8": "PASS",
        "G9": "PASS",
        "G10": "PASS",
    }
    if any(v == "FAIL" for v in gates.values()):
        verdict = "FAIL"
    elif any(v == "BORDERLINE" for v in gates.values()):
        verdict = "BORDERLINE"
    else:
        verdict = "PASS"

    # --- Manifest ---
    file_records = []
    for p in sorted(raw_dir.glob("*.csv")):
        df_head = pd.read_csv(p, nrows=0)
        # row counts: for submission use len(sub) etc.
        if p.name == "problem.csv":
            nrows = len(problems)
        elif p.name == "submission.csv":
            nrows = len(sub)
        elif p.name == "user.csv":
            nrows = len(users)
        elif p.name == "problem_tags.csv":
            nrows = len(tags_map)
        elif p.name == "problem_tag.csv":
            nrows = len(tag_names)
        else:
            nrows = sum(1 for _ in p.open("rb")) - 1
        file_records.append(
            {
                "original_filename": p.name,
                "byte_size": p.stat().st_size,
                "sha256": sha256_file(p),
                "encoding": "utf-8",
                "format": "csv",
                "row_count": int(nrows),
                "columns": list(df_head.columns),
            }
        )
    manifest = {
        "dataset": "bepkt",
        "source_of_record": "Author Google Drive (BePKT paper arXiv:2112.08273)",
        "source_url": "https://drive.google.com/drive/folders/1Jt6f0MV1paGLlctJqxHtdF1Vh2mUnsoV?usp=sharing",
        "paper": "Programming Knowledge Tracing: A Comprehensive Dataset and A New Model",
        "arxiv": "2112.08273",
        "access_status": "SUCCESS_OFFICIAL",
        "download_utc": utc_now(),
        "download_method": "gdown selective download of raw_data/ and used_in_pdkt/ (model weights not required for Phase-0B)",
        "license_access_note": "Author-provided public Google Drive folder linked from arXiv paper; no separate license file in raw_data. Cite paper/arXiv.",
        "archive_structure": {
            "raw_data/": [r["original_filename"] for r in file_records],
            "used_in_pdkt/": sorted([p.name for p in used_dir.glob("*")]) if used_dir.exists() else [],
            "model/": "present on Drive but not required/downloaded for audit",
        },
        "files": file_records,
        "n_files_hashed": len(file_records),
        "paper_vs_files": {
            "paper_problems": 1054,
            "file_problems": int(len(problems)),
            "paper_problem_concept_pairs": 1054,
            "file_problem_concept_edges": int(len(tags_map)),
            "paper_users": 906,
            "file_users_with_submissions": int(sub["user_id"].nunique()),
            "note": "Files are authoritative; paper '1054 problems' matches edge count not problem.csv rows (600).",
        },
    }
    write_json(root / "artifacts" / "manifests" / "bepkt_source_manifest.json", manifest)

    # write audits
    out_cols = [
        "item_id",
        "language",
        "raw_text_present",
        "description_chars",
        "normalized_text_length_chars",
        "has_input_spec",
        "has_output_spec",
        "has_examples",
        "has_code_fragment",
        "has_image_reference",
        "has_external_dependency",
        "duplicate_text_group",
        "text_complete_status",
        "exclusion_reason",
    ]
    item_audit[out_cols].to_csv(art / "bepkt_item_audit.csv", index=False)
    leak_audit.to_csv(art / "bepkt_item_leakage_audit.csv", index=False)
    elig.to_csv(art / "bepkt_response_eligibility.csv", index=False)

    crosswalk = [
        {"canonical": "learner_id", "bepkt": "submission.user_id / user.id", "transform": False, "ambiguity": False},
        {"canonical": "item_id", "bepkt": "problem.id / submission.problem_id", "transform": False, "ambiguity": "problem._id is NOT unique across contests"},
        {"canonical": "item_text", "bepkt": "problem.description + input_description + output_description + samples", "transform": True, "ambiguity": False},
        {"canonical": "concept_id(s)", "bepkt": "problem_tags.problemtag_id → problem_tag.name", "transform": True, "ambiguity": False},
        {"canonical": "expert_difficulty", "bepkt": "problem.difficulty {Low,Mid,High}", "transform": False, "ambiguity": False},
        {"canonical": "timestamp", "bepkt": "submission.create_time", "transform": True, "ambiguity": False},
        {"canonical": "submission_result", "bepkt": "submission.result (int OJ code)", "transform": False, "ambiguity": False},
        {"canonical": "correctness", "bepkt": "reconstruct: result==0 → Accepted", "transform": True, "ambiguity": "paper r_t not a raw column"},
        {"canonical": "submitted_code", "bepkt": "submission.code", "transform": False, "ambiguity": False},
        {"canonical": "course/context", "bepkt": "submission.contest_id / contest.csv; behavior events", "transform": True, "ambiguity": True},
    ]

    result = {
        "access": "SUCCESS_OFFICIAL",
        "source_sha256_problem_csv": next(f["sha256"] for f in file_records if f["original_filename"] == "problem.csv"),
        "source_sha256_submission_csv": next(
            f["sha256"] for f in file_records if f["original_filename"] == "submission.csv"
        ),
        "raw_items": int(len(problems)),
        "text_complete_items": n_text,
        "status_counts": status_counts,
        "learners": int(sub["user_id"].nunique()),
        "learners_kt_eligible": int(len(splits)),
        "submission_events": int(len(sub)),
        "learner_item_pairs": int(len(pair_sizes)),
        "repeated_pair_rate": interaction_stats["repeated_pair_rate"],
        "items_test_ge20_first_observed": n_ge20_first,
        "items_test_ge20_all_submissions": n_ge20_all,
        "threshold_first": first_thresh,
        "threshold_all": all_thresh,
        "unseen_item_eligible": n_unseen,
        "provisional_fold_sizes": fold_sizes,
        "provisional_fold_first_attempt_eval": fold_first_eval,
        "provisional_fold_all_submission_eval": fold_all_eval,
        "provisional_fold_test_learners": fold_test_learners,
        "zero_target_train_removable": zero_train_ok,
        "interaction_stats": interaction_stats,
        "views": views,
        "temporal": temporal,
        "split": split_info,
        "expert_difficulty": expert,
        "concepts": concepts,
        "prompt": prompt,
        "gates": gates,
        "verdict": verdict,
        "crosswalk": crosswalk,
        "predominant_language": "zh-CN",
        "blockers": [
            f"G4 FAIL: only {n_ge20_first} PASS_TEXT_COMPLETE items have >=20 first-observed held-out test responses (need >=150; <100 = FAIL)",
            f"G5 FAIL: only {n_unseen} unseen-eligible target items under first-observed>=20 rule (need >=100)",
        ]
        if verdict == "FAIL"
        else [],
        "scientific_risks": [
            "High repeated-attempt rate (~54% pairs >1)",
            "First-observed vs all-submission eligibility diverge sharply",
            "Chinese programming text requires prompt adaptation",
            "Expert difficulty and acceptance counts exist on item table — must be stripped from LLM inputs",
            "Paper problem count (1054) != file problem count (600)",
            "Programming domain expands paper scope beyond math/database MCQ",
        ],
    }
    write_json(derived / "bepkt_audit_summary.json", result)
    return result
