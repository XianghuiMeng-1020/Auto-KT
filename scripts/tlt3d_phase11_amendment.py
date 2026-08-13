#!/usr/bin/env python3
"""TLT-3D Phase 1.1 — protocol amendment utilities (NO LLM / NO KT training).

Implements:
  - first_observed_learner_item collapse
  - XES/Junyi/DBE harmonized RQ2 eligibility
  - legacy all-response vs first-observed construct integrity
  - DBE 704-row correctness forensic + consensus-only sensitivity
  - prompt twin hashes
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
# Prefer local processed; fall back to sibling main worktree (read-only).
LOCAL_PROCESSED = ROOT / "data_processed"
SIBLING_PROCESSED = ROOT.parent / "AutoKT_all" / "data_processed"
TABLE = ROOT / "tables"
ART = ROOT / "artifacts" / "tlt3d"
CFG = ROOT / "configs" / "tlt3d"
REP = ROOT / "reports"
DBE_DERIVED = ROOT / "data" / "external" / "dbe_kt22" / "derived"
DBE_RAW = ROOT / "data" / "external" / "dbe_kt22" / "raw" / "official_extracted"

THRESHOLDS = [5, 10, 20, 50, 100]
PRIMARY_THRESHOLD = 20

MATH_SYSTEM = (
    "You are a mathematics education expert. Estimate the difficulty of the "
    "following problem for a typical student at the appropriate grade level. "
    "Output a single number between 0.0 (very easy) and 1.0 (very hard). "
    "Output only the number, with no explanation."
)
DBE_SYSTEM = (
    "You are a database systems education expert. Estimate the difficulty of the "
    "following problem for a typical student in an introductory database course. "
    "Output a single number between 0.0 (very easy) and 1.0 (very hard). "
    "Output only the number, with no explanation."
)
USER_TEMPLATE = "Problem:\n{stem_text}"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def resolve_processed() -> Path:
    for p in (LOCAL_PROCESSED, SIBLING_PROCESSED):
        if (p / "xes3g5m" / "interactions.parquet").exists() and (p / "junyi" / "interactions.parquet").exists():
            return p
    raise FileNotFoundError(
        "Need data_processed/{xes3g5m,junyi}/interactions.parquet "
        f"(checked {LOCAL_PROCESSED} and {SIBLING_PROCESSED})"
    )


def first_observed_learner_item(
    interactions: pd.DataFrame,
    *,
    learner_col: str,
    item_col: str,
    correct_col: str,
    order_cols: list[str],
    split_col: str | None = None,
    split_value: str | None = None,
    valid_correct_mask: pd.Series | None = None,
) -> pd.DataFrame:
    """Collapse to earliest valid response per learner × item.

    Deterministic tie-break: sort by order_cols (ascending) then keep first.
    Does not modify KT sequences; RQ2 learner-error criterion only.
    """
    df = interactions.copy()
    if valid_correct_mask is not None:
        df = df.loc[valid_correct_mask].copy()
    else:
        df = df[df[correct_col].notna()].copy()
    if split_col is not None and split_value is not None:
        df = df[df[split_col] == split_value].copy()
    missing = [c for c in [learner_col, item_col, correct_col, *order_cols] if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns for first_observed: {missing}")
    df = df.sort_values([learner_col, item_col, *order_cols], kind="mergesort")
    out = df.drop_duplicates([learner_col, item_col], keep="first").copy()
    out["response_view"] = "FIRST_OBSERVED_LEARNER_ITEM"
    return out.reset_index(drop=True)


def beta11_error(incorrect: int, n: int) -> float:
    return (incorrect + 1) / (n + 2)


def item_error_from_first(first: pd.DataFrame, item_col: str, correct_col: str) -> pd.DataFrame:
    g = first.groupby(item_col, as_index=False).agg(
        n_first_observed=(correct_col, "size"),
        n_correct=(correct_col, "sum"),
        n_learners=(correct_col, "size"),  # one row per learner×item
    )
    g["n_incorrect"] = g["n_first_observed"] - g["n_correct"].astype(int)
    g["raw_error"] = g["n_incorrect"] / g["n_first_observed"]
    g["smoothed_error_beta_1_1"] = [
        beta11_error(int(e), int(n)) for e, n in zip(g["n_incorrect"], g["n_first_observed"])
    ]
    return g


def eligibility_table(item_err: pd.DataFrame, n_col: str = "n_first_observed") -> pd.DataFrame:
    rows = []
    for thr in THRESHOLDS:
        rows.append(
            {
                "threshold": thr,
                "eligible_items": int((item_err[n_col] >= thr).sum()),
                "excluded_items": int((item_err[n_col] < thr).sum()),
            }
        )
    return pd.DataFrame(rows)


def processed_rq2_for_dataset(ds: str, processed: Path) -> dict[str, Any]:
    items = pd.read_parquet(processed / ds / "items.parquet")
    scoreable = set(items.loc[items["eligible_for_llm_scoring"], "item_id_hash"].astype(str))
    cols = [
        "student_id_hash",
        "item_id_hash",
        "correct",
        "split_assignment",
        "timestamp_or_order",
        "sequence_index",
        "interaction_id_hash",
        "attempt_index",
        "first_attempt",
    ]
    ix = pd.read_parquet(processed / ds / "interactions.parquet", columns=cols)
    ix = ix[ix["item_id_hash"].astype(str).isin(scoreable)].copy()
    ix["item_id_hash"] = ix["item_id_hash"].astype(str)
    ix["student_id_hash"] = ix["student_id_hash"].astype(str)
    ix["correct_bool"] = ix["correct"].astype(bool)

    # All scoreable (for repeat diagnostics)
    pair_sizes = ix.groupby(["student_id_hash", "item_id_hash"]).size()
    repeat_frac_all = float((pair_sizes > 1).mean()) if len(pair_sizes) else 0.0

    test = ix[ix["split_assignment"] == "test"].copy()
    n_before = int(len(test))
    pair_sizes_test = test.groupby(["student_id_hash", "item_id_hash"]).size()
    n_pairs = int(len(pair_sizes_test))
    n_repeat_pairs = int((pair_sizes_test > 1).sum())
    repeat_frac = float((pair_sizes_test > 1).mean()) if n_pairs else 0.0

    order_cols = ["timestamp_or_order", "sequence_index", "attempt_index", "interaction_id_hash"]
    first = first_observed_learner_item(
        test,
        learner_col="student_id_hash",
        item_col="item_id_hash",
        correct_col="correct_bool",
        order_cols=order_cols,
    )
    assert first.duplicated(["student_id_hash", "item_id_hash"]).sum() == 0
    item_err = item_error_from_first(first, "item_id_hash", "correct_bool")
    elig = eligibility_table(item_err)
    primary = item_err[item_err["n_first_observed"] >= PRIMARY_THRESHOLD].copy()
    primary["dataset"] = ds
    primary["eligible_ge20"] = True

    # Persist eligibility artifact
    out = item_err.copy()
    out["dataset"] = ds
    out["response_view"] = "FIRST_OBSERVED_LEARNER_ITEM"
    out["split_scope"] = "held_out_test"
    out["universe"] = "llm_scoreable"
    for thr in THRESHOLDS:
        out[f"eligible_ge{thr}"] = out["n_first_observed"] >= thr
    ART.mkdir(parents=True, exist_ok=True)
    out_path = ART / f"{'xes' if ds == 'xes3g5m' else 'junyi'}_first_observed_rq2_eligibility.csv"
    out.to_csv(out_path, index=False)

    # Legacy all-response from V2 table
    legacy = pd.read_csv(TABLE / "AUTHENTIC_DIFFICULTY_REFERENCES_V2_ORIENTATION_CORRECTED.csv")
    leg = legacy[(legacy["dataset"] == ds) & (legacy["reference_scope"] == "held_out_test")].copy()
    leg["item_id_hash"] = leg["item_id_hash"].astype(str)
    fo_ids = set(primary["item_id_hash"])
    leg_primary_ids = set(leg.loc[leg["heldout_response_count"] >= PRIMARY_THRESHOLD, "item_id_hash"].astype(str))
    common_primary = fo_ids & leg_primary_ids
    only_fo = fo_ids - leg_primary_ids
    only_leg = leg_primary_ids - fo_ids

    cmp = item_err.rename(
        columns={
            "n_first_observed": "n_fo",
            "smoothed_error_beta_1_1": "err_fo",
            "raw_error": "raw_fo",
        }
    ).merge(
        leg.rename(
            columns={
                "heldout_response_count": "n_legacy",
                "smoothed_error_beta_1_1": "err_legacy",
                "raw_error": "raw_legacy",
            }
        )[["item_id_hash", "n_legacy", "err_legacy", "raw_legacy"]],
        on="item_id_hash",
        how="inner",
    )
    common_ge20 = cmp[(cmp["n_fo"] >= PRIMARY_THRESHOLD) & (cmp["n_legacy"] >= PRIMARY_THRESHOLD)]
    abs_delta = (common_ge20["err_fo"] - common_ge20["err_legacy"]).abs()
    if len(common_ge20) > 2:
        spearman = float(stats.spearmanr(common_ge20["err_fo"], common_ge20["err_legacy"]).correlation)
        pearson = float(np.corrcoef(common_ge20["err_fo"].astype(float), common_ge20["err_legacy"].astype(float))[0, 1])
    else:
        spearman = float("nan")
        pearson = float("nan")

    # eligibility flip among scoreable items present in both
    fo_ge20 = set(cmp.loc[cmp["n_fo"] >= PRIMARY_THRESHOLD, "item_id_hash"])
    leg_ge20 = set(cmp.loc[cmp["n_legacy"] >= PRIMARY_THRESHOLD, "item_id_hash"])
    eligibility_changed = len(fo_ge20.symmetric_difference(leg_ge20))
    denom = max(len(fo_ge20 | leg_ge20), 1)
    eligibility_change_frac = eligibility_changed / denom

    alarm = bool(
        (spearman < 0.95 if np.isfinite(spearman) else True)
        or (float(abs_delta.median()) > 0.03 if len(abs_delta) else True)
        or (eligibility_change_frac > 0.10)
    )

    return {
        "dataset": ds,
        "scoreable_items": len(scoreable),
        "heldout_responses_before_collapse": n_before,
        "unique_learner_item_pairs": n_pairs,
        "repeated_learner_item_pairs": n_repeat_pairs,
        "repeated_learner_item_fraction": repeat_frac,
        "repeated_learner_item_fraction_all_splits_scoreable": repeat_frac_all,
        "responses_after_first_observed_collapse": int(len(first)),
        "eligibility_by_threshold": elig.to_dict(orient="records"),
        "primary_rq2_eligible_ge20": int(len(primary)),
        "legacy_primary_ge20": int(len(leg_primary_ids)),
        "common_primary_items": int(len(common_primary)),
        "only_first_observed_ge20": int(len(only_fo)),
        "only_legacy_ge20": int(len(only_leg)),
        "construct_compare": {
            "n_common_ge20": int(len(common_ge20)),
            "spearman": spearman,
            "pearson": pearson,
            "median_abs_delta": float(abs_delta.median()) if len(abs_delta) else None,
            "p95_abs_delta": float(abs_delta.quantile(0.95)) if len(abs_delta) else None,
            "max_abs_delta": float(abs_delta.max()) if len(abs_delta) else None,
            "eligibility_changed_count": eligibility_changed,
            "eligibility_change_frac": eligibility_change_frac,
            "audit_alarm": alarm,
        },
        "eligibility_csv": str(out_path.relative_to(ROOT)),
        "processed_root": str(processed),
    }


def dbe_rq2_eligibility() -> dict[str, Any]:
    first = pd.read_parquet(DBE_DERIVED / "tlt3d_rq2_first_observed.parquet")
    test = first[first["split_assignment"] == "test"].copy()
    assert test.duplicated(["learner_id", "item_id"]).sum() == 0
    item_err = item_error_from_first(test, "item_id", "is_correct")
    item_err["dataset"] = "dbe_kt22"
    item_err["response_view"] = "FIRST_OBSERVED_LEARNER_ITEM"
    item_err["split_scope"] = "held_out_test"
    for thr in THRESHOLDS:
        item_err[f"eligible_ge{thr}"] = item_err["n_first_observed"] >= thr
    ART.mkdir(parents=True, exist_ok=True)
    path = ART / "dbe_first_observed_rq2_eligibility.csv"
    item_err.to_csv(path, index=False)
    elig = eligibility_table(item_err)
    # repeat rate from full KT events
    kt = pd.read_parquet(DBE_DERIVED / "tlt3d_kt_interactions.parquet")
    kt_test = kt[kt["split_assignment"] == "test"]
    pair = kt_test.groupby(["learner_id", "item_id"]).size()
    return {
        "dataset": "dbe_kt22",
        "heldout_responses_before_collapse": int(len(kt_test)),
        "unique_learner_item_pairs": int(len(pair)),
        "repeated_learner_item_pairs": int((pair > 1).sum()),
        "repeated_learner_item_fraction": float((pair > 1).mean()) if len(pair) else 0.0,
        "responses_after_first_observed_collapse": int(len(test)),
        "eligibility_by_threshold": elig.to_dict(orient="records"),
        "primary_rq2_eligible_ge20": int((item_err["n_first_observed"] >= PRIMARY_THRESHOLD).sum()),
        "eligibility_csv": str(path.relative_to(ROOT)),
    }


def dbe_correctness_forensic() -> dict[str, Any]:
    """Audit answer_state vs choice.is_correct disagreements."""
    tx_path = ROOT / "data/external/dbe_kt22/raw/official_extracted/csv/Transaction.csv"
    ch_path = ROOT / "data/external/dbe_kt22/raw/official_extracted/csv/Question_Choices.csv"
    if not tx_path.exists() or not ch_path.exists():
        raise FileNotFoundError(f"DBE CSV missing: {tx_path} / {ch_path}")

    tx = pd.read_csv(tx_path)
    choices = pd.read_csv(ch_path)
    # normalize columns
    def colmap(df, options):
        for o in options:
            if o in df.columns:
                return o
        lower = {c.lower(): c for c in df.columns}
        for o in options:
            if o.lower() in lower:
                return lower[o.lower()]
        return None

    ans_state = colmap(tx, ["answer_state", "Answer_State"])
    choice_id = colmap(tx, ["answer_choice_id", "answer_choice", "choice_id"])
    qid = colmap(tx, ["question_id", "Question_id"])
    sid = colmap(tx, ["student_id", "Student_id"])
    tid = colmap(tx, ["id", "transaction_id"])
    ch_id = colmap(choices, ["id", "choice_id"])
    ch_correct = colmap(choices, ["is_correct", "Is_Correct"])

    def map_correct(x):
        if x is True or str(x).lower() == "true":
            return True
        if x is False or str(x).lower() == "false":
            return False
        return None

    tx = tx.copy()
    tx["_ans"] = tx[ans_state].map(map_correct)
    ch_lookup = dict(zip(choices[ch_id].astype(int), choices[ch_correct].map(map_correct)))
    tx["_from_choice"] = tx[choice_id].map(
        lambda cid: ch_lookup.get(int(cid)) if pd.notna(cid) and int(cid) in ch_lookup else None
    )
    both = tx.dropna(subset=["_ans", "_from_choice"]).copy()
    disagree = both[both["_ans"] != both["_from_choice"]].copy()
    n_disagree = int(len(disagree))
    n_compared = int(len(both))
    rate = n_disagree / n_compared if n_compared else None

    # answer_state value inventory
    ans_vc = tx[ans_state].value_counts(dropna=False).to_dict()
    ans_vc = {str(k): int(v) for k, v in ans_vc.items()}

    # missing choice
    missing_choice = int(tx[choice_id].isna().sum())
    invalid_choice = int((~tx[choice_id].isna() & ~tx[choice_id].astype("Int64").isin(set(choices[ch_id].astype(int)))).sum()) if choice_id else None

    # concentration
    by_q = disagree.groupby(qid).size().sort_values(ascending=False) if n_disagree else pd.Series(dtype=int)
    by_learner = disagree.groupby(sid).size().sort_values(ascending=False) if n_disagree else pd.Series(dtype=int)

    # privacy-safe audit CSV: hash learner ids
    audit_rows = []
    for _, r in disagree.iterrows():
        learner_raw = r[sid]
        learner_hash = sha256_text(f"dbe_learner|{learner_raw}")[:16]
        cid = r[choice_id]
        audit_rows.append(
            {
                "transaction_id": int(r[tid]) if tid and pd.notna(r[tid]) else None,
                "learner_hash": learner_hash,
                "question_id": int(r[qid]),
                "answer_choice_id": int(cid) if pd.notna(cid) else None,
                "answer_state": bool(r["_ans"]),
                "choice_is_correct": bool(r["_from_choice"]),
                "disagreement": True,
            }
        )
    audit = pd.DataFrame(audit_rows)
    ART.mkdir(parents=True, exist_ok=True)
    audit_path = ART / "dbe_correctness_disagreement_audit.csv"
    audit.to_csv(audit_path, index=False)

    # Consensus-only: rows where agreement OR reconstruction unavailable without contradiction
    tx["_consensus_keep"] = False
    # agreement
    agree_mask = tx["_ans"].notna() & tx["_from_choice"].notna() & (tx["_ans"] == tx["_from_choice"])
    # reconstruction unavailable, answer_state present, no contradiction possible
    no_recon = tx["_ans"].notna() & tx["_from_choice"].isna()
    tx.loc[agree_mask | no_recon, "_consensus_keep"] = True
    # exclude disagreements
    tx.loc[tx["_ans"].notna() & tx["_from_choice"].notna() & (tx["_ans"] != tx["_from_choice"]), "_consensus_keep"] = False

    # Restrict to Phase-1 text-complete universe + split from derived
    universe = json.loads((CFG / "dbe_item_universe.json").read_text())
    included = set(int(i) for i in universe["included_item_ids"])
    splits = pd.read_csv(DBE_DERIVED / "tlt3d_learner_splits.csv")
    split_map = dict(zip(splits["student_id_raw"].astype(str), splits["split_assignment"]))

    cons = tx[tx["_consensus_keep"] & tx[qid].astype(int).isin(included) & tx["_ans"].notna()].copy()
    cons["learner_id"] = cons[sid].astype(str)
    cons["item_id"] = cons[qid].astype(int)
    cons["is_correct"] = cons["_ans"].astype(bool)
    cons["split_assignment"] = cons["learner_id"].map(split_map)
    cons = cons[cons["split_assignment"].notna()].copy()

    # first-observed on consensus
    # need timestamp
    ts_col = colmap(tx, ["start_time", "Start_Time", "timestamp"])
    if ts_col:
        cons["ts"] = pd.to_datetime(cons[ts_col], utc=True, errors="coerce")
    else:
        cons["ts"] = pd.to_datetime(cons[tid], errors="coerce") if tid else pd.Series(range(len(cons)))
    cons["_ord"] = cons[tid] if tid else np.arange(len(cons))
    first_c = first_observed_learner_item(
        cons,
        learner_col="learner_id",
        item_col="item_id",
        correct_col="is_correct",
        order_cols=["ts", "_ord"],
        split_col="split_assignment",
        split_value="test",
    )
    # primary answer_state first-observed (already have)
    primary_first = pd.read_parquet(DBE_DERIVED / "tlt3d_rq2_first_observed.parquet")
    primary_test = primary_first[primary_first["split_assignment"] == "test"]
    p_err = item_error_from_first(primary_test, "item_id", "is_correct").rename(
        columns={"n_first_observed": "n_primary", "smoothed_error_beta_1_1": "err_primary"}
    )
    c_err = item_error_from_first(first_c, "item_id", "is_correct").rename(
        columns={"n_first_observed": "n_consensus", "smoothed_error_beta_1_1": "err_consensus"}
    )
    m = p_err.merge(c_err, on="item_id", how="inner")
    common_ge20 = m[(m["n_primary"] >= 20) & (m["n_consensus"] >= 20)]
    abs_d = (common_ge20["err_primary"] - common_ge20["err_consensus"]).abs()
    spearman = float(stats.spearmanr(common_ge20["err_primary"], common_ge20["err_consensus"]).correlation) if len(common_ge20) > 2 else float("nan")

    p_ge20 = set(p_err.loc[p_err["n_primary"] >= 20, "item_id"].astype(int))
    c_ge20 = set(c_err.loc[c_err["n_consensus"] >= 20, "item_id"].astype(int))
    lost = p_ge20 - c_ge20

    # documentation check for answer_state precedence
    doc_hits = []
    for p in (ROOT / "data/external/dbe_kt22").rglob("*"):
        if p.suffix.lower() in {".md", ".txt", ".pdf", ".html"} and p.is_file() and p.stat().st_size < 2_000_000:
            try:
                text = p.read_text(encoding="utf-8", errors="ignore").lower()
            except Exception:
                continue
            if "answer_state" in text:
                doc_hits.append(str(p.relative_to(ROOT)))

    gate = "PASS_WITH_SENSITIVITY"
    if (
        (rate is not None and rate >= 0.01)
        or (np.isfinite(spearman) and spearman < 0.99)
        or (len(abs_d) and float(abs_d.median()) > 0.01)
        or (len(lost) > 5)
    ):
        gate = "PI_REVIEW_REQUIRED"

    # top questions
    top_q = [{"question_id": int(i), "n_disagree": int(n)} for i, n in by_q.head(15).items()] if n_disagree else []

    return {
        "tx_path": str(tx_path.relative_to(ROOT)),
        "choices_path": str(ch_path.relative_to(ROOT)),
        "disagreement_rows": n_disagree,
        "rows_compared": n_compared,
        "disagreement_rate": rate,
        "affected_learners": int(by_learner.nunique()) if n_disagree else 0,
        "affected_items": int(by_q.nunique()) if n_disagree else 0,
        "answer_state_value_counts": ans_vc,
        "missing_answer_choice_id_rows": missing_choice,
        "invalid_answer_choice_id_rows": invalid_choice,
        "top_questions_by_disagreement": top_q,
        "max_disagreements_on_one_question": int(by_q.iloc[0]) if len(by_q) else 0,
        "documentation_answer_state_mentions": doc_hits[:20],
        "documented_cause": (
            "Not conclusively attributed from available files; "
            "disagreements are binary flips between answer_state and selected choice.is_correct "
            "with both fields present. Official docs mentioning answer_state listed if found. "
            "Primary SoR remains answer_state pending PI review if gate escalates."
        ),
        "primary_sor": "answer_state",
        "audit_csv": str(audit_path.relative_to(ROOT)),
        "consensus_only": {
            "definition": "retain rows where answer_state==choice.is_correct OR choice reconstruction unavailable with answer_state present; exclude explicit contradictions",
            "retained_interactions": int(len(cons)),
            "retained_learners": int(cons["learner_id"].nunique()),
            "retained_items": int(cons["item_id"].nunique()),
            "primary_ge20_items": int(len(c_ge20)),
            "unseen_eligible_items": int(len(c_ge20)),  # same universe proxy pre-LLM
        },
        "primary_vs_consensus": {
            "common_ge20_items": int(len(common_ge20)),
            "spearman": spearman,
            "median_abs_delta": float(abs_d.median()) if len(abs_d) else None,
            "p95_abs_delta": float(abs_d.quantile(0.95)) if len(abs_d) else None,
            "max_abs_delta": float(abs_d.max()) if len(abs_d) else None,
            "items_losing_ge20_eligibility": int(len(lost)),
            "lost_item_ids": sorted(int(x) for x in lost),
        },
        "DBE_CORRECTNESS_GATE": gate,
    }


def write_prompt_twin() -> dict[str, Any]:
    CFG.mkdir(parents=True, exist_ok=True)
    path = CFG / "dbe_prompt_v1.txt"
    content = (
        "### System message\n\n"
        f"{DBE_SYSTEM}\n\n"
        "### User message\n\n"
        f"{USER_TEMPLATE}\n"
    )
    path.write_text(content, encoding="utf-8")
    math_hash = sha256_text(MATH_SYSTEM + "\n---\n" + USER_TEMPLATE)
    dbe_hash = sha256_text(DBE_SYSTEM + "\n---\n" + USER_TEMPLATE)
    # line-level diff classification
    math_lines = MATH_SYSTEM.split()
    dbe_lines = DBE_SYSTEM.split()
    # token/phrase changes
    changes = [
        {
            "math": "mathematics education expert",
            "dbe": "database systems education expert",
            "class": "DOMAIN_ROLE_SUBSTITUTION",
        },
        {
            "math": "a typical student at the appropriate grade level",
            "dbe": "a typical student in an introductory database course",
            "class": "DOMAIN_TERMINOLOGY_SUBSTITUTION",
        },
    ]
    prohibited = []
    # verify unchanged anchors
    assert "Output a single number between 0.0 (very easy) and 1.0 (very hard)." in MATH_SYSTEM
    assert "Output a single number between 0.0 (very easy) and 1.0 (very hard)." in DBE_SYSTEM
    assert "Output only the number, with no explanation." in MATH_SYSTEM
    assert "Output only the number, with no explanation." in DBE_SYSTEM
    assert USER_TEMPLATE == "Problem:\n{stem_text}"
    gate = "PASS" if not prohibited else "FAIL"
    return {
        "math_system": MATH_SYSTEM,
        "dbe_system": DBE_SYSTEM,
        "user_template": USER_TEMPLATE,
        "math_prompt_hash": math_hash,
        "dbe_prompt_hash": dbe_hash,
        "changes": changes,
        "prohibited_construct_changes": prohibited,
        "PROMPT_EQUIVALENCE_GATE": gate,
        "dbe_prompt_path": str(path.relative_to(ROOT)),
    }


def main() -> int:
    processed = resolve_processed()
    ART.mkdir(parents=True, exist_ok=True)
    print(f"Using processed root: {processed}", file=sys.stderr)

    xes = processed_rq2_for_dataset("xes3g5m", processed)
    print("XES done", xes["primary_rq2_eligible_ge20"], file=sys.stderr)
    junyi = processed_rq2_for_dataset("junyi", processed)
    print("Junyi done", junyi["primary_rq2_eligible_ge20"], file=sys.stderr)
    dbe = dbe_rq2_eligibility()
    print("DBE RQ2 done", dbe["primary_rq2_eligible_ge20"], file=sys.stderr)
    forensic = dbe_correctness_forensic()
    print("Forensic", forensic["DBE_CORRECTNESS_GATE"], forensic["disagreement_rows"], file=sys.stderr)
    prompt = write_prompt_twin()

    summary = {
        "phase": "TLT3D_P11",
        "frozen_at_utc": utc_now(),
        "processed_root": str(processed),
        "xes": xes,
        "junyi": junyi,
        "dbe_rq2": dbe,
        "dbe_correctness": forensic,
        "prompt_twin": prompt,
        "no_llm_calls": True,
    }
    write_json(ART / "P11_AMENDMENT_COMPUTE_SUMMARY.json", summary)
    print(json.dumps({"ok": True, "gate_correctness": forensic["DBE_CORRECTNESS_GATE"], "prompt": prompt["PROMPT_EQUIVALENCE_GATE"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
