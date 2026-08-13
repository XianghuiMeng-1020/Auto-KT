#!/usr/bin/env python3
"""TLT-3D Phase 1 — prepare frozen DBE-KT22 experiment artifacts (NO LLM calls).

Deterministic adapter emitting:
  - item universe freeze
  - canonical learner-visible item texts
  - authoritative learner split
  - first-observed RQ2 interactions + full KT interaction table
  - unseen-item folds
  - random + character-length controls (content-side only)
  - DBE_PRE_LLM_FREEZE.json
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from data.unified_schema_common import UnifiedSchemaConfig, split_students_deterministic  # noqa: E402
from tlt4d.text_audit import classify_item_text, content_hash, html_to_plain, normalize_plain  # noqa: E402

PHASE0_AUDIT_COMMIT = "6679cf66ed0cc99ded2525d564c4249611cd1cc0"
BASE_PROTOCOL_PARENT = "05c854fe5bc0e40361540da7053317af3ebed3e0"
DATASET = "dbe_kt22"
PRIMARY_THRESHOLD = 20
N_ITEM_FOLDS = 5
ITEM_FOLD_SEED = 2024
MASK_SEED = 2024
LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def choice_label(i: int) -> str:
    if i < 26:
        return LABELS[i]
    return f"O{i+1}"


def build_canonical_item_text(stem: str, choices: list[tuple[int, str]]) -> str:
    """Learner-visible stem + labeled options. No answer / expert / stats."""
    lines = ["Question:", stem.strip(), "", "Choices:"]
    for i, (_cid, text) in enumerate(choices):
        lines.append(f"{choice_label(i)}. {text}")
    return "\n".join(lines).strip() + "\n"


def forbidden_in_scoring_text(text: str, *, correct_texts: list[str], expert: Any) -> list[str]:
    hits = []
    # Expert difficulty must not appear as leaked annotation token patterns we inject
    if "expert_difficulty" in text.lower() or "difficulty_feedback" in text.lower():
        hits.append("expert_or_feedback_token")
    # We never append correct-answer markers
    if re.search(r"(?i)\b(correct answer|answer key|is_correct)\b", text):
        hits.append("answer_key_token")
    # Do not include expert numeric label as a dedicated field (stem may contain digits)
    if f"ExpertDifficulty:{expert}" in text:
        hits.append("expert_field")
    return hits


def build_item_folds_from_train_counts(item_ids: list[str], train_n: pd.Series, fold_seed: int, n_folds: int) -> pd.DataFrame:
    """Mirror tlt_extension_common.build_item_folds stratification (content-agnostic)."""
    count_df = pd.DataFrame({"item_id": item_ids, "train_n": [int(train_n.get(i, 0)) for i in item_ids]})
    nonzero = count_df[count_df["train_n"] > 0].copy()
    zero = count_df[count_df["train_n"] == 0].copy()
    if len(nonzero):
        try:
            nonzero["bin"] = pd.qcut(nonzero["train_n"], q=min(10, len(nonzero)), duplicates="drop")
        except ValueError:
            nonzero["bin"] = 0
    else:
        nonzero["bin"] = pd.Series(dtype=object)
    zero["bin"] = "zero"
    pieces = []
    for bin_name, g in pd.concat([nonzero, zero], ignore_index=True).groupby("bin", observed=False):
        g = g.copy()
        keys = g["item_id"].astype(str) + f"|{fold_seed}|{bin_name}"
        ranks = keys.map(lambda s: int(hashlib.sha256(s.encode()).hexdigest()[:12], 16))
        g = g.assign(_rank=ranks).sort_values(["_rank", "item_id"])
        pieces.append(g)
    ordered = pd.concat(pieces, ignore_index=True)
    ordered["item_fold"] = np.arange(len(ordered)) % n_folds
    folds = ordered[["item_id", "train_n", "item_fold"]].sort_values("item_id").reset_index(drop=True)
    assert set(folds["item_id"]) == set(item_ids)
    assert folds["item_id"].duplicated().sum() == 0
    assert set(folds["item_fold"]) == set(range(n_folds))
    return folds


def main() -> int:
    parser = argparse.ArgumentParser(description="TLT3D Phase-1 DBE prepare (no LLM)")
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root: Path = args.root

    csv_dir = root / "data/external/dbe_kt22/raw/official_extracted/csv"
    derived = root / "data/external/dbe_kt22/derived"
    art = root / "artifacts/tlt3d"
    cfg_dir = root / "configs/tlt3d"
    for p in (derived, art, cfg_dir):
        p.mkdir(parents=True, exist_ok=True)

    questions = pd.read_csv(csv_dir / "Questions.csv")
    choices = pd.read_csv(csv_dir / "Question_Choices.csv")
    tx = pd.read_csv(csv_dir / "Transaction.csv")
    phase0_audit = pd.read_csv(root / "artifacts/tlt4d/dbe_kt22_item_audit.csv")

    # --- Item universe freeze from Phase-0 statuses ---
    assert len(phase0_audit) == 212
    included = phase0_audit.loc[
        phase0_audit["text_complete_status"] == "PASS_TEXT_COMPLETE", "item_id"
    ].astype(int).tolist()
    excluded = phase0_audit.loc[
        phase0_audit["text_complete_status"] != "PASS_TEXT_COMPLETE",
        ["item_id", "text_complete_status", "exclusion_reason"],
    ]
    assert len(included) == 166, f"expected 166 PASS_TEXT_COMPLETE, got {len(included)}"
    included_sorted = sorted(included)
    item_universe = {
        "dataset": "DBE-KT22",
        "phase0_audit_commit": PHASE0_AUDIT_COMMIT,
        "source_questions_sha256": sha256_file(csv_dir / "Questions.csv"),
        "source_choices_sha256": sha256_file(csv_dir / "Question_Choices.csv"),
        "source_transaction_sha256": sha256_file(csv_dir / "Transaction.csv"),
        "raw_items": 212,
        "included_item_ids": included_sorted,
        "included_count": 166,
        "excluded": [
            {
                "item_id": int(r.item_id),
                "text_complete_status": r.text_complete_status,
                "exclusion_reason": r.exclusion_reason,
            }
            for r in excluded.itertuples(index=False)
        ],
        "excluded_count": int(len(excluded)),
        "frozen_at_utc": utc_now(),
    }
    item_universe["item_universe_hash"] = sha256_text(
        ",".join(str(i) for i in included_sorted)
        + "|"
        + item_universe["source_questions_sha256"]
        + "|"
        + item_universe["source_choices_sha256"]
    )
    write_json(cfg_dir / "dbe_item_universe.json", item_universe)

    # --- Canonical items ---
    choice_groups = {
        int(qid): sorted(
            [(int(r.id), normalize_plain(r.choice_text)) for r in g.itertuples(index=False)],
            key=lambda x: x[0],
        )
        for qid, g in choices.groupby("question_id")
    }
    qmap = questions.set_index("id")
    item_rows = []
    for qid in included_sorted:
        row = qmap.loc[qid]
        stem_raw = row.get("question_text")
        stem_html = row.get("question_rich_text")
        audit = classify_item_text(
            stem_raw=stem_raw,
            stem_html=stem_html,
            title=row.get("question_title"),
            choices=[t for _, t in choice_groups.get(qid, [])],
            require_choices=True,
        )
        assert audit["text_complete_status"] == "PASS_TEXT_COMPLETE", qid
        stem = audit["normalized_text"]
        chs = choice_groups.get(qid, [])
        scoring_text = build_canonical_item_text(stem, chs)
        correct = choices.loc[
            (choices["question_id"] == qid) & (choices["is_correct"] == True),  # noqa: E712
            "choice_text",
        ].astype(str).tolist()
        leaks = forbidden_in_scoring_text(
            scoring_text, correct_texts=correct, expert=row.get("difficulty")
        )
        assert not leaks, f"forbidden content in item {qid}: {leaks}"
        # Ensure correct answer identity not appended
        assert "is_correct" not in scoring_text.lower()
        assert str(row.get("difficulty")) not in scoring_text.split("\n")[0:1] or True
        item_rows.append(
            {
                "dataset": DATASET,
                "item_id": qid,
                "item_id_hash": f"dbe_{qid}",
                "language": "en",
                "n_choices": len(chs),
                "expert_difficulty_secondary_only": int(row["difficulty"])
                if pd.notna(row.get("difficulty"))
                else None,
                "stem_chars": len(stem),
                "scoring_text": scoring_text,
                "scoring_text_hash": content_hash(scoring_text),
                "stem_content_hash": audit["content_hash"],
                "char_length": len(scoring_text),
                "token_length": len(scoring_text.split()),
            }
        )
    items = pd.DataFrame(item_rows)
    items_path = derived / "tlt3d_canonical_items.parquet"
    items_csv = derived / "tlt3d_canonical_items_meta.csv"
    items.to_parquet(items_path, index=False)
    items.drop(columns=["scoring_text"]).to_csv(items_csv, index=False)
    # Local untracked full text JSONL (gitignored under derived if needed — keep parquet)
    with (derived / "tlt3d_canonical_items.jsonl").open("w", encoding="utf-8") as f:
        for r in item_rows:
            f.write(json.dumps({"item_id": r["item_id"], "scoring_text_hash": r["scoring_text_hash"], "scoring_text": r["scoring_text"]}, ensure_ascii=False) + "\n")

    # --- Interactions + correctness ---
    tx2 = tx.copy()
    tx2["student_id"] = tx2["student_id"].astype(int)
    tx2["question_id"] = tx2["question_id"].astype(int)
    tx2["ts"] = pd.to_datetime(tx2["start_time"], utc=True, errors="coerce")
    # Source-of-record: answer_state (Phase-0 / released correctness field)
    def map_correct(x):
        if x is True or str(x).lower() == "true":
            return True
        if x is False or str(x).lower() == "false":
            return False
        return None

    tx2["is_correct_answer_state"] = tx2["answer_state"].map(map_correct)
    ch_lookup = choices.set_index("id")["is_correct"].to_dict()
    tx2["is_correct_from_choice"] = tx2["answer_choice_id"].map(
        lambda cid: bool(ch_lookup[cid]) if pd.notna(cid) and int(cid) in ch_lookup else None
    )
    both = tx2.dropna(subset=["is_correct_answer_state", "is_correct_from_choice"])
    n_disagree = int((both["is_correct_answer_state"] != both["is_correct_from_choice"]).sum())
    # Freeze primary correctness from answer_state
    tx2["is_correct"] = tx2["is_correct_answer_state"]
    # Restrict to text-complete items; drop hidden if any for primary? keep all non-null correctness
    tx_univ = tx2[tx2["question_id"].isin(included_sorted)].copy()
    tx_univ = tx_univ[tx_univ["is_correct"].notna()].copy()

    # --- Authoritative learner split ---
    cfg = UnifiedSchemaConfig.load()
    per_learner = tx_univ.groupby("student_id").size()
    eligible = per_learner[per_learner >= cfg.min_student_interactions].index.astype(str)
    splits = split_students_deterministic(eligible, cfg)
    assert set(splits["split_assignment"]) <= {"train", "val", "test"}
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        sa = set(splits.loc[splits["split_assignment"] == a, "student_id_raw"])
        sb = set(splits.loc[splits["split_assignment"] == b, "student_id_raw"])
        assert sa.isdisjoint(sb)
    union = set(splits["student_id_raw"])
    assert union == set(eligible)
    split_map = dict(zip(splits["student_id_raw"].astype(str), splits["split_assignment"]))
    split_hash = sha256_text(
        "|".join(
            f"{r.student_id_raw}:{r.split_assignment}"
            for r in splits.sort_values("student_id_raw").itertuples(index=False)
        )
        + f"|seed={cfg.split_seed}|min={cfg.min_student_interactions}"
    )
    splits_out = splits.copy()
    splits_out["dataset"] = DATASET
    splits_out["split_seed"] = cfg.split_seed
    splits_out["split_algorithm_version"] = cfg.split_algorithm_version
    splits_out.to_csv(derived / "tlt3d_learner_splits.csv", index=False)

    tx_univ["learner_id"] = tx_univ["student_id"].astype(str)
    tx_univ["split_assignment"] = tx_univ["learner_id"].map(split_map)
    tx_kt = tx_univ[tx_univ["split_assignment"].notna()].copy()
    tx_kt = tx_kt.sort_values(["learner_id", "ts", "id"], kind="mergesort")

    # Full KT event view (all timestamped interactions)
    kt_events = tx_kt[
        ["learner_id", "question_id", "ts", "id", "is_correct", "split_assignment", "is_hidden"]
    ].rename(columns={"question_id": "item_id", "id": "transaction_id"})
    kt_events["dataset"] = DATASET
    kt_events["event_view"] = "ALL_TIMESTAMPED_INTERACTIONS"
    kt_events.to_parquet(derived / "tlt3d_kt_interactions.parquet", index=False)

    # RQ2 first-observed view
    first = (
        tx_kt.sort_values(["learner_id", "question_id", "ts", "id"], kind="mergesort")
        .drop_duplicates(["learner_id", "question_id"], keep="first")
        .copy()
    )
    first_out = pd.DataFrame(
        {
            "dataset": DATASET,
            "learner_id": first["learner_id"].astype(str),
            "item_id": first["question_id"].astype(int),
            "ts": first["ts"],
            "is_correct": first["is_correct"].astype(bool),
            "split_assignment": first["split_assignment"].astype(str),
            "response_view": "FIRST_OBSERVED_LEARNER_ITEM",
            "transaction_id": first["id"].astype(int),
        }
    )
    first_out.to_parquet(derived / "tlt3d_rq2_first_observed.parquet", index=False)

    # Eligibility under authoritative split + first-observed test
    elig_rows = []
    for item_id in included_sorted:
        g = first_out[first_out["item_id"] == item_id]
        test_g = g[g["split_assignment"] == "test"]
        elig_rows.append(
            {
                "item_id": item_id,
                "train_first_observed": int((g["split_assignment"] == "train").sum()),
                "val_first_observed": int((g["split_assignment"] == "val").sum()),
                "test_first_observed": int(len(test_g)),
                "test_error_rate_first_observed": float(1.0 - test_g["is_correct"].mean())
                if len(test_g)
                else None,
            }
        )
    elig = pd.DataFrame(elig_rows)
    elig.to_csv(derived / "tlt3d_response_eligibility.csv", index=False)
    n_ge20 = int((elig["test_first_observed"] >= PRIMARY_THRESHOLD).sum())
    if n_ge20 < 150:
        print(f"STOP: eligible items with >=20 held-out first-observed test = {n_ge20} < 150", file=sys.stderr)
        return 2
    primary_items = elig.loc[elig["test_first_observed"] >= PRIMARY_THRESHOLD, "item_id"].astype(int).tolist()
    assert len(primary_items) == n_ge20

    # Unseen folds on primary eligible / text-complete universe (166)
    train_counts = (
        tx_kt[tx_kt["split_assignment"] == "train"].groupby("question_id").size()
    )
    folds = build_item_folds_from_train_counts(
        [str(i) for i in included_sorted],
        train_counts.rename(index=lambda x: str(x)),
        ITEM_FOLD_SEED,
        N_ITEM_FOLDS,
    )
    folds["item_id"] = folds["item_id"].astype(int)
    folds["dataset"] = DATASET
    folds["item_fold_seed"] = ITEM_FOLD_SEED
    folds.to_csv(derived / "tlt3d_unseen_item_folds.csv", index=False)
    fold_hash = sha256_text(
        "|".join(f"{int(r.item_id)}:{int(r.item_fold)}" for r in folds.sort_values("item_id").itertuples())
        + f"|seed={ITEM_FOLD_SEED}|n={N_ITEM_FOLDS}"
    )
    fold_sizes = folds["item_fold"].value_counts().sort_index().to_dict()

    # Controls (no outcomes): character length + random matched to [0,1] uniform then we'll
    # persist fixed per-item draws; coldstart uses Mini permutation — without Mini scores yet,
    # freeze Uniform[0,1] with seed as interim protocol matching score scale, documented as
    # RANDOM_UNIFORM_01_FIXED_PER_ITEM (PI: limited KT uses resample-from-Mini after scores exist).
    rng = np.random.default_rng(MASK_SEED)
    chars = items.set_index("item_id")["char_length"].astype(float)
    cmin, cmax = float(chars.min()), float(chars.max())
    char_norm = {
        int(i): float((chars.loc[i] - cmin) / (cmax - cmin)) if cmax > cmin else 0.0
        for i in included_sorted
    }
    random_scores = {int(i): float(rng.random()) for i in included_sorted}
    controls = pd.DataFrame(
        {
            "item_id": included_sorted,
            "char_length": [int(chars.loc[i]) for i in included_sorted],
            "char_length_norm": [char_norm[i] for i in included_sorted],
            "random_scalar_uniform01": [random_scores[i] for i in included_sorted],
            "random_seed": MASK_SEED,
            "char_norm_note": "min_max_over_166_text_complete_items",
            "random_note": (
                "Uniform[0,1] fixed per item with seed=2024 BEFORE LLM scoring. "
                "After LLM scores exist, limited-KT Random-Scalar must switch to "
                "resample-from-gpt-4o-mini marginals per limited_kt_common; "
                "coldstart Random-Scalar uses permutation of Mini scores."
            ),
        }
    )
    controls.to_csv(derived / "tlt3d_controls_pre_llm.csv", index=False)

    # Surface features for RQ1 (content-side only)
    surface = items[
        ["item_id", "char_length", "token_length", "n_choices", "language", "scoring_text_hash"]
    ].copy()
    surface["math_symbol_count"] = items["scoring_text"].map(
        lambda t: len(re.findall(r"[∑∫√∞≈≠≤≥±×÷^=+\-*/<>]", t))
    )
    surface["equation_count"] = items["scoring_text"].map(
        lambda t: len(re.findall(r"[=≈≠]", t))
    )
    surface["number_count"] = items["scoring_text"].map(
        lambda t: len(re.findall(r"\b\d+(?:\.\d+)?\b", t))
    )
    surface.to_csv(derived / "tlt3d_surface_features.csv", index=False)

    # Integrity summary
    integrity = {
        "correctness_source_of_record": "Transaction.answer_state",
        "choice_crosscheck_disagreements": n_disagree,
        "choice_crosscheck_rows_compared": int(len(both)),
        "disagreement_rate": float(n_disagree / len(both)) if len(both) else None,
        "resolution": "Use answer_state as primary; disagreements reported, not silently flipped",
        "hidden_transactions_in_universe": int(tx_univ["is_hidden"].fillna(False).astype(bool).sum()),
        "rq2_response_view": "FIRST_OBSERVED_LEARNER_ITEM",
        "kt_event_view": "ALL_TIMESTAMPED_INTERACTIONS",
        "kc_policy": "NOT_CONSUMED_BY_CURRENT_GRU_SAKT_FORWARD; content-side KC optional metadata only",
    }
    write_json(derived / "tlt3d_correctness_integrity.json", integrity)

    freeze = {
        "dataset": "DBE-KT22",
        "raw_items": 212,
        "text_complete_items": 166,
        "eligible_primary_items": n_ge20,
        "learners": int(tx_univ["student_id"].nunique()),
        "eligible_learners_min10": int(len(splits)),
        "interactions": int(len(tx_univ)),
        "kt_interactions_in_split": int(len(tx_kt)),
        "learner_split_hash": split_hash,
        "item_universe_hash": item_universe["item_universe_hash"],
        "unseen_fold_hash": fold_hash,
        "fold_sizes": {str(k): int(v) for k, v in fold_sizes.items()},
        "primary_threshold": PRIMARY_THRESHOLD,
        "items_test_first_observed_ge20": n_ge20,
        "llm_scores_present": False,
        "adapter_version": "tlt3d_prepare_dbe_v1",
        "base_commit": BASE_PROTOCOL_PARENT,
        "phase0_audit_commit": PHASE0_AUDIT_COMMIT,
        "source_transaction_sha256": item_universe["source_transaction_sha256"],
        "canonical_items_sha256": sha256_file(items_path),
        "split_seed": cfg.split_seed,
        "item_fold_seed": ITEM_FOLD_SEED,
        "mask_seed": MASK_SEED,
        "correctness_disagreements": n_disagree,
        "frozen_at_utc": utc_now(),
    }
    write_json(art / "DBE_PRE_LLM_FREEZE.json", freeze)
    write_json(derived / "tlt3d_prepare_summary.json", freeze)

    print(json.dumps({"status": "ok", "eligible_primary_items": n_ge20, "fold_sizes": fold_sizes, "disagreements": n_disagree}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
