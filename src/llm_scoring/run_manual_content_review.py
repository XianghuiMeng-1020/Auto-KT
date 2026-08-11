#!/usr/bin/env python3
"""Run frozen manual content review and emit pilot gate decision."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "llm_scoring"))

from analyze_llm_pilot import (  # noqa: E402
    build_content_limitation_tables,
    eligibility_amendment,
    load_request_log,
    parse_reliability_table,
    projected_full_cost,
    stability_table,
)
from llm_pilot_common import PilotConfig, utc_now  # noqa: E402
from manual_content_review import (  # noqa: E402
    TABLE_DIR,
    build_manual_review_table,
    build_review_summary,
    decide_pilot_gate,
    engineering_incidents,
    write_gate_decision_report,
    write_phase_report_updated,
)

MANIFEST_PATH = ROOT / "data_manifests" / "_manifest.json"


def update_manifest(status: str, full_ready: bool, sample_hash: str) -> None:
    if not MANIFEST_PATH.exists():
        return
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest["generated_at_utc"] = utc_now()
    manifest["phase_stop_code"] = status
    gates = manifest.setdefault("gate_status", {})
    gates["llm_pilot_status"] = status
    gates["full_llm_scoring_ready"] = full_ready
    manifest["llm_pilot_manual_review"] = {
        "review_sample_hash": sample_hash,
        "reviewer": "frozen_content_audit_v1",
        "completed_at_utc": utc_now(),
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> int:
    cfg = PilotConfig.load()
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    log = load_request_log()
    parse_df = parse_reliability_table(log, cfg)
    stab_df = stability_table(log, cfg)
    xes_lim, junyi_lim = build_content_limitation_tables(cfg)
    xes_lim.to_csv(TABLE_DIR / "LLM_PILOT_XES_CONTENT_LIMITATIONS.csv", index=False)
    junyi_lim.to_csv(TABLE_DIR / "LLM_PILOT_JUNYI_CONTENT_LIMITATIONS.csv", index=False)

    review_df, sample_hash = build_manual_review_table(cfg)
    summary_df = build_review_summary(review_df)
    review_df.to_csv(TABLE_DIR / "LLM_PILOT_MANUAL_REVIEW.csv", index=False)
    summary_df.to_csv(TABLE_DIR / "LLM_PILOT_MANUAL_REVIEW_SUMMARY.csv", index=False)

    amendment = eligibility_amendment(junyi_lim, cfg)
    incidents = engineering_incidents()
    cost_flag = projected_full_cost(cfg)
    status, full_ready, rationale = decide_pilot_gate(review_df, parse_df, amendment, incidents)

    write_gate_decision_report(
        status=status,
        full_ready=full_ready,
        sample_hash=sample_hash,
        review_df=review_df,
        summary_df=summary_df,
        parse_df=parse_df,
        stab_df=stab_df,
        amendment=amendment,
        incidents=incidents,
        rationale=rationale,
    )
    write_phase_report_updated(
        status=status,
        full_ready=full_ready,
        cfg=cfg,
        parse_df=parse_df,
        stab_df=stab_df,
        review_df=review_df,
        sample_hash=sample_hash,
        amendment=amendment,
        incidents=incidents,
        log=log,
        cost_flag=cost_flag,
    )
    update_manifest(status, full_ready, sample_hash)

    print(status)
    print(f"full_llm_scoring_ready={full_ready}")
    print(f"review_sample_hash={sample_hash}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
