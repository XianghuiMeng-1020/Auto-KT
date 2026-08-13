#!/usr/bin/env python3
"""TLT-3D P3B.2 — Family C computation + result tables (after DBE runs complete)."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
ART = ROOT / "artifacts" / "tlt3d"
REP = ROOT / "reports"
CFG = ROOT / "configs" / "tlt3d"

AMENDMENT_ID = "POST_RESULT_OPERATIONALIZATION_REPAIR_002"
EXPOSURES = [0, 1, 3, 5, 10, 20]
SEEDS = [2024, 42, 123, 456, 789]
LLM_MAP = {"gpt-4o-mini": "LLM-Mini", "gpt-5.4": "LLM-5.4"}
CMP_MAP = {"Standard": "Standard", "Random-ResampledScore": "Random-Scalar"}


def holm(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = np.argsort(pvals)
    out = [np.nan] * m
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * float(pvals[idx])
        running = max(running, val)
        out[idx] = min(1.0, running)
    return out


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_xes_junyi_seed() -> pd.DataFrame:
    rr = pd.read_csv(ROOT / "journal_expansion/runs/limited_kt/RUN_REGISTRY.csv")
    formal = rr[(rr.status == "ok") & (~rr.run_id.astype(str).str.startswith("pilot_"))].copy()
    formal["exposure"] = formal["exposure"].astype(str)
    formal["backbone"] = "GRU"
    formal["dataset"] = formal["dataset"].astype(str)
    return formal.rename(columns={"exposure": "response_exposure"})[
        ["dataset", "backbone", "response_exposure", "condition", "seed", "log_loss", "auc", "n_predictions", "best_epoch", "run_id"]
    ]


def load_dbe_seed() -> pd.DataFrame:
    path = ROOT / "journal_expansion/runs/limited_kt_dbe/RUN_REGISTRY.csv"
    r = pd.read_csv(path)
    r = r[r.status == "ok"].copy()
    r["response_exposure"] = r["response_exposure"].astype(str)
    # some rows may use run_id parsing if columns missing
    cols = ["dataset", "backbone", "response_exposure", "condition", "seed", "log_loss", "auc", "n_predictions", "best_epoch", "run_id"]
    for c in cols:
        if c not in r.columns:
            raise RuntimeError(f"missing column {c} in DBE registry")
    r["classification"] = r.get("classification", "SECONDARY")
    return r[cols + (["classification"] if "classification" in r.columns else [])]


def ci_mean(x: np.ndarray) -> tuple[float, float, float, float]:
    x = np.asarray(x, float)
    n = len(x)
    mean = float(np.mean(x))
    sd = float(np.std(x, ddof=1)) if n > 1 else float("nan")
    se = sd / np.sqrt(n) if n > 1 else float("nan")
    # existing limited_kt uses 1.96*sd/sqrt(n) normal approx for seed diffs; for means of 5 use t critical
    tcrit = float(stats.t.ppf(0.975, n - 1)) if n > 1 else float("nan")
    return mean, sd, mean - tcrit * se, mean + tcrit * se


def main() -> int:
    op = json.loads((CFG / "family_C_operational_registry_v1.json").read_text())
    hyps = op["hypotheses"]
    assert len(hyps) == 12

    xj = load_xes_junyi_seed()
    dbe = load_dbe_seed()
    # unify
    xj["classification"] = np.where(
        xj["response_exposure"].isin([str(e) for e in EXPOSURES])
        & xj["condition"].isin(["Standard", "LLM-Mini", "LLM-5.4", "Random-Scalar"]),
        "CONFIRMATORY_SOURCE",
        "SECONDARY",
    )
    all_seed = pd.concat([xj, dbe], ignore_index=True, sort=False)
    all_seed.to_csv(ART / "P3B2_RESPONSE_LIMITED_SEED_RESULTS.csv", index=False)

    # Aggregated
    agg_rows = []
    for keys, g in all_seed.groupby(["dataset", "backbone", "response_exposure", "condition"]):
        mean, sd, lo, hi = ci_mean(g["log_loss"].values)
        am, asd, alo, ahi = ci_mean(g["auc"].values)
        agg_rows.append(
            {
                "dataset": keys[0],
                "backbone": keys[1],
                "exposure": keys[2],
                "condition": keys[3],
                "n_seeds": len(g),
                "log_loss_mean": mean,
                "log_loss_sd": sd,
                "log_loss_ci_low": lo,
                "log_loss_ci_high": hi,
                "auc_mean": am,
                "auc_sd": asd,
                "auc_ci_low": alo,
                "auc_ci_high": ahi,
                "classification": "SECONDARY" if str(keys[2]) == "warm" or keys[1] == "SAKT" else "MIXED",
            }
        )
    pd.DataFrame(agg_rows).to_csv(ART / "P3B2_RESPONSE_LIMITED_AGGREGATED_RESULTS.csv", index=False)

    # Family C exposure deltas
    exp_rows = []
    for h in hyps:
        ds = h["dataset"]
        llm_c = LLM_MAP[h["llm"]]
        cmp_c = CMP_MAP[h["comparator"]]
        sub = all_seed[(all_seed.dataset == ds) & (all_seed.backbone == "GRU")]
        for s in SEEDS:
            for e in EXPOSURES:
                a = sub[(sub.condition == llm_c) & (sub.seed == s) & (sub.response_exposure == str(e))]
                b = sub[(sub.condition == cmp_c) & (sub.seed == s) & (sub.response_exposure == str(e))]
                if len(a) == 0 or len(b) == 0:
                    raise RuntimeError(f"missing cell {h['hypothesis_id']} seed={s} e={e}")
                # pairing integrity: same n_predictions
                aa = a.iloc[-1]
                bb = b.iloc[-1]
                if int(aa.n_predictions) != int(bb.n_predictions):
                    raise RuntimeError(f"FAMILY_C_PAIRING_INTEGRITY_BLOCKER {h['hypothesis_id']} e={e} seed={s}")
                delta = float(bb.log_loss) - float(aa.log_loss)  # comparator - LLM
                exp_rows.append(
                    {
                        "hypothesis_id": h["hypothesis_id"],
                        "dataset": ds,
                        "llm": h["llm"],
                        "comparator": h["comparator"],
                        "seed": s,
                        "exposure": e,
                        "llm_log_loss": float(aa.log_loss),
                        "comparator_log_loss": float(bb.log_loss),
                        "delta": delta,
                        "n_predictions": int(aa.n_predictions),
                    }
                )
    exp_df = pd.DataFrame(exp_rows)
    assert len(exp_df) == 360, len(exp_df)
    exp_df.to_csv(ART / "P3B2_FAMILY_C_EXPOSURE_DELTAS.csv", index=False)

    seed_rows = []
    for (hid, seed), g in exp_df.groupby(["hypothesis_id", "seed"]):
        assert len(g) == 6 and set(g.exposure) == set(EXPOSURES)
        seed_rows.append(
            {
                "hypothesis_id": hid,
                "dataset": g.dataset.iloc[0],
                "llm": g.llm.iloc[0],
                "comparator": g.comparator.iloc[0],
                "seed": int(seed),
                "mean_delta": float(g.delta.mean()),
                "n_exposures": 6,
            }
        )
    seed_df = pd.DataFrame(seed_rows)
    assert len(seed_df) == 60
    seed_df.to_csv(ART / "P3B2_FAMILY_C_SEED_DELTAS.csv", index=False)

    fam_rows = []
    for h in hyps:
        g = seed_df[seed_df.hypothesis_id == h["hypothesis_id"]].sort_values("seed")
        assert len(g) == 5
        vals = g.mean_delta.values.astype(float)
        effect = float(vals.mean())
        sd = float(vals.std(ddof=1))
        se = sd / np.sqrt(5)
        tcrit = float(stats.t.ppf(0.975, 4))
        t_stat, p_raw = stats.ttest_1samp(vals, 0.0)
        fam_rows.append(
            {
                "hypothesis_id": h["hypothesis_id"],
                "dataset": h["dataset"],
                "backbone": "GRU",
                "llm": h["llm"],
                "comparator": h["comparator"],
                "confirmatory_exposures": json.dumps(EXPOSURES),
                "n_seeds": 5,
                "effect_log_loss": effect,
                "effect_ci_low": effect - tcrit * se,
                "effect_ci_high": effect + tcrit * se,
                "seed_sd": sd,
                "t_statistic": float(t_stat),
                "df": 4,
                "raw_p": float(p_raw),
                "direction_convention": "comparator_log_loss_minus_llm_log_loss",
                "classification": "CONFIRMATORY_POSTRESULT_OPERATIONALIZATION_REPAIRED",
                "amendment_id": AMENDMENT_ID,
            }
        )
    fam = pd.DataFrame(fam_rows)
    # preserve registry order
    order = [h["hypothesis_id"] for h in hyps]
    fam["hypothesis_id"] = pd.Categorical(fam["hypothesis_id"], categories=order, ordered=True)
    fam = fam.sort_values("hypothesis_id").reset_index(drop=True)
    fam["holm_p"] = holm(fam["raw_p"].tolist())
    assert len(fam) == 12
    fam.to_csv(ART / "family_C_confirmatory_results.csv", index=False)

    print(json.dumps({"family_C_rows": 12, "exposure_deltas": 360, "seed_deltas": 60}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
