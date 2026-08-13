#!/usr/bin/env python3
"""Mechanically render TLT-3D final supplement LaTeX table fragments.

Deterministic. Frozen artifacts only -- no new science.
Outputs overwrite manuscript_tlt/generated_supplement/ each run.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "tlt3d"
OUT = ROOT / "manuscript_tlt" / "generated_supplement"

SHARED_CORE_FEATURES = [
    "char_length",
    "token_length",
    "sentence_count",
    "number_count",
    "math_symbol_count",
    "equation_count",
    "answer_option_count",
]

DATASET_DISPLAY = {
    "xes3g5m": "XES3G5M",
    "junyi": "Junyi Academy",
    "dbe_kt22": "DBE-KT22",
    "gsm8k": "GSM8K",
}

LLM_DISPLAY = {
    "gpt-4o-mini": "gpt-4o-mini",
    "gpt-5.4": "gpt-5.4",
}

SIM_CONDITION_LABELS = {
    "S0": "LLM-only",
    "S1": "LLM-dominant",
    "S2": "Balanced",
    "S3": "Independent-dominant",
    "S4": "Independent-only",
    "S5": "Visible-feature",
}

CLASSIFICATION_MAP = {
    "MEASUREMENT_ONLY_NO_DEPLOYMENT_SUPPORT": "Measurement correspondence only",
    "NO_DISTINCTIVE_SUPPORT": "No distinctive deployment support",
}

# PI section 34 authorized reproducibility entries (immutable).
REPRO_MANIFEST_ROWS: list[tuple[str, str]] = [
    ("Evidence freeze commit", "a66ff788d901cc815b3a248501005eb9d523b1a7"),
    ("DBE scoring commit", "e726237f003e105d036a7b8e439385a152300e42"),
    (
        "DBE item universe",
        "d62fb95604bac94e65adda498dc21175e66ce75198836996f5f53695c96e38a6",
    ),
    (
        "DBE score CSV",
        "484dd79a140372fcbda66275aa472505de47f0852cd37992692ac87c97e22f64",
    ),
    (
        "DBE learner split",
        "65d13de13e7c3a8b366c63628cab3e7ef3f2a67b8eae7cde1aab4e1a53d627dc",
    ),
    (
        "DBE unseen folds",
        "28efbf33c231772a3565056367c4bf6dfa62bdf553abcfd8397aa9d9037d4e0a",
    ),
    (
        "DBE Random-Permuted",
        "d859d9aafac5209b34921d3efae74ed5bfaf0c08ab7d2aef3dfc8239d7f937ce",
    ),
    (
        "DBE CharacterLength",
        "280e98d48dfc19b903719873d58f25b839d3cce6d2562461e13f946f109fe9cd",
    ),
    (
        "Family-C operational amendment commit",
        "0609954254b82fa689c1f228eb486ed823a23ef6",
    ),
    (
        "Final Family-C result commit",
        "2d1206090074aa49f937d77955bc1e8d6aca22bf",
    ),
    (
        "Family-D preflight",
        "c7e5144c3516a2b63c1ab4fa4014191fe032159f",
    ),
    (
        "Final Family-D result",
        "a188e64129b8f012caa7ac44ae5c4bea84fa6472",
    ),
]

# Verified universe expert-difficulty counts (PI-authorized hardcoded).
DBE_EXPERT_TEXT_COMPLETE = {1: 80, 2: 71, 3: 15}
DBE_EXPERT_RAW = {1: 95, 2: 84, 3: 33}


def esc(text: Any) -> str:
    """Escape LaTeX specials in plain text cells."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return "--"
    s = str(text)
    # Prefer ASCII double-hyphen over unicode dashes that break LaTeX.
    for bad in ("\u2013", "\u2014", "\u2212", "\u2010", "\u2011"):
        s = s.replace(bad, "--")
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    out = []
    for ch in s:
        out.append(repl.get(ch, ch))
    return "".join(out)


def fmt_int(x: Any) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "--"
    return f"{int(x):,}".replace(",", "{,}")


def fmt_float(x: Any, digits: int = 3) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "--"
    return f"{float(x):.{digits}f}"


def fmt_sci_or_sig(p: Any, thresh: float = 1e-3, sig: int = 3) -> str:
    if p is None or (isinstance(p, float) and pd.isna(p)):
        return "--"
    p = float(p)
    if p < thresh:
        return f"{p:.2e}".replace("e-0", "e-").replace("e+0", "e+")
    # 3 significant digits
    return f"{p:.{sig}g}"


def fmt_ci(lo: Any, hi: Any, digits: int = 3) -> str:
    if any(v is None or (isinstance(v, float) and pd.isna(v)) for v in (lo, hi)):
        return "--"
    return f"[{fmt_float(lo, digits)}, {fmt_float(hi, digits)}]"


def fmt_yes_no(flag: Any) -> str:
    if isinstance(flag, str):
        return "Yes" if flag.strip().lower() in {"true", "1", "yes"} else "No"
    return "Yes" if bool(flag) else "No"


def dataset_name(key: Any) -> str:
    return DATASET_DISPLAY.get(str(key), esc(key))


def llm_name(key: Any) -> str:
    k = str(key)
    return LLM_DISPLAY.get(k, esc(k))


def write_tex(name: str, body: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    # Ensure trailing newline; deterministic LF.
    text = body if body.endswith("\n") else body + "\n"
    path.write_text(text, encoding="utf-8")
    return path


def tabular(
    colspec: str,
    header: Sequence[str],
    rows: Iterable[Sequence[str]],
    *,
    footnote: str | None = None,
    scriptsize: bool = False,
) -> str:
    lines: list[str] = []
    if scriptsize:
        lines.append(r"\scriptsize")
    lines.append(rf"\begin{{tabular}}{{{colspec}}}")
    lines.append(r"\toprule")
    lines.append(" & ".join(header) + r" \\")
    lines.append(r"\midrule")
    for row in rows:
        lines.append(" & ".join(row) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    if footnote:
        lines.append(r"\par\vspace{0.35em}")
        lines.append(r"{\footnotesize " + footnote + "}")
    return "\n".join(lines) + "\n"


def load_csv(rel: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / rel)


def load_json(rel: str) -> Any:
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Table generators
# ---------------------------------------------------------------------------


def gen_dataset_universes() -> None:
    df = load_csv("artifacts/tlt3d/FINAL_DATASET_UNIVERSES.csv")
    role_map = {
        "xes3g5m": "Authentic learner evidence + KT",
        "junyi": "Authentic learner evidence + KT",
        "dbe_kt22": "Authentic learner evidence + KT",
    }
    rows: list[list[str]] = []
    for _, r in df.iterrows():
        ds = str(r["dataset"])
        legacy = r["rq2_legacy_all_response"]
        notes = []
        if pd.notna(legacy):
            notes.append("LEGACY ALL-RESPONSE SENSITIVITY")
        if ds == "dbe_kt22":
            notes.append("text-complete primary universe")
        rows.append(
            [
                esc(dataset_name(ds)),
                fmt_int(r["raw_full"]),
                fmt_int(r["text_scoreable"]),
                fmt_int(r["llm_scored"]),
                fmt_int(r["rq2_first_observed_primary"]),
                fmt_int(legacy) if pd.notna(legacy) else "--",
                fmt_int(r["unseen_target_universe"]),
                esc(role_map.get(ds, "Authentic")),
                esc("; ".join(notes) if notes else "--"),
            ]
        )
    # PI-authorized GSM8K simulation-only row.
    rows.append(
        [
            "GSM8K",
            "--",
            "200",
            "--",
            "--",
            "--",
            "--",
            "Simulation only",
            "Controlled signal-decoupling; not authentic learner evidence",
        ]
    )
    body = tabular(
        "lrrrrrrll",
        [
            "Dataset",
            "Raw/full",
            "Text-scoreable",
            "LLM-scored",
            "Primary FO RQ2",
            "Legacy all-response",
            "Unseen targets",
            "Role",
            "Notes",
        ],
        rows,
        scriptsize=True,
        footnote=(
            "Legacy all-response Ns (XES 3279; Junyi 183) are "
            "LEGACY ALL-RESPONSE SENSITIVITY only; never primary."
        ),
    )
    write_tex("dataset_universes.tex", body)


def gen_family_a() -> None:
    df = load_csv("artifacts/tlt3d/FINAL_FAMILY_A.csv")
    assert len(df) == 6, f"Family A expected 6 rows, got {len(df)}"
    rows = []
    for _, r in df.iterrows():
        rows.append(
            [
                esc(r["ID"]),
                esc(dataset_name(r["dataset"])),
                esc(llm_name(r["model"])),
                fmt_int(r["N"]),
                fmt_float(r["rho"], 3),
                fmt_ci(r["CI_lo"], r["CI_hi"], 3),
                fmt_sci_or_sig(r["raw_p"]),
                fmt_sci_or_sig(r["holm_p"]),
                fmt_yes_no(r["holm_supported"]),
            ]
        )
    body = tabular(
        "lllrrrrrc",
        [
            "ID",
            "Dataset",
            "LLM",
            "N",
            r"Spearman $\rho$",
            "95\\% CI",
            "raw $p$",
            "Holm $p$",
            "Holm-supported",
        ],
        rows,
        scriptsize=True,
        footnote=r"analysis\_status = PRE\_RESULT\_CONFIRMATORY.",
    )
    write_tex("family_a.tex", body)


def gen_family_a_secondary() -> None:
    df = load_csv("artifacts/tlt3d/family_A_confirmatory_results.csv")
    rows = []
    for _, r in df.iterrows():
        rows.append(
            [
                esc(r["hypothesis_id"]),
                esc(dataset_name(r["dataset"])),
                esc(llm_name(r["model"])),
                fmt_int(r["n_items"]),
                fmt_float(r["pearson_r"], 3),
                fmt_sci_or_sig(r["pearson_p"]),
                fmt_float(r["kendall_tau"], 3),
                fmt_sci_or_sig(r["kendall_p"]),
            ]
        )
    body = tabular(
        "lllrrrrr",
        [
            "ID",
            "Dataset",
            "LLM",
            "N",
            "Pearson $r$",
            "Pearson $p$",
            r"Kendall $\tau$",
            "Kendall $p$",
        ],
        rows,
        scriptsize=True,
        footnote="Secondary supporting metrics only; primary Family-A inference uses Spearman.",
    )
    write_tex("family_a_secondary.tex", body)


def gen_family_b() -> None:
    df = load_csv("artifacts/tlt3d/FINAL_FAMILY_B.csv")
    assert len(df) == 6, f"Family B expected 6 rows, got {len(df)}"
    rows = []
    for _, r in df.iterrows():
        rows.append(
            [
                esc(r["ID"]),
                esc(dataset_name(r["dataset"])),
                esc(llm_name(r["model"])),
                fmt_int(r["N"]),
                fmt_float(r["base_R2"], 3),
                fmt_float(r["aug_R2"], 3),
                fmt_float(r["delta_R2"], 3),
                fmt_ci(r["CI_lo"], r["CI_hi"], 3),
                fmt_sci_or_sig(r["raw_p_repaired"]),
                fmt_sci_or_sig(r["holm_p_repaired"]),
                fmt_yes_no(r["holm_supported"]),
            ]
        )
    body = tabular(
        "lllrrrrrrrc",
        [
            "ID",
            "Dataset",
            "LLM",
            "N",
            r"Base $R^2$",
            r"Aug $R^2$",
            r"$\Delta R^2$",
            "95\\% CI",
            "repaired raw $p$",
            "repaired Holm $p$",
            "Supported",
        ],
        rows,
        scriptsize=True,
        footnote=(
            r"effect/CI status = PRE\_RESULT\_FROZEN; "
            r"$p$ status = POST\_RESULT\_INFERENTIAL\_REPAIR\_001."
        ),
    )
    write_tex("family_b.tex", body)


def gen_family_c() -> None:
    final = load_csv("artifacts/tlt3d/FINAL_FAMILY_C.csv")
    conf = load_csv("artifacts/tlt3d/family_C_confirmatory_results.csv")[
        ["hypothesis_id", "t_statistic"]
    ]
    df = final.merge(conf, left_on="ID", right_on="hypothesis_id", how="left")
    assert len(df) == 12, f"Family C expected 12 rows, got {len(df)}"
    assert (df["holm_p"] == 1.0).all()
    rows = []
    for _, r in df.iterrows():
        rows.append(
            [
                esc(r["ID"]),
                esc(dataset_name(r["dataset"])),
                esc(llm_name(r["LLM"])),
                esc(r["comparator"]),
                esc(str(r["confirmatory_exposures"]).replace(",", ", ")),
                fmt_float(r["effect_log_loss"], 6),
                fmt_ci(r["CI_lo"], r["CI_hi"], 6),
                fmt_float(r["t_statistic"], 3),
                fmt_sci_or_sig(r["raw_p"]),
                fmt_float(r["holm_p"], 1),
                fmt_yes_no(r["holm_supported"]),
            ]
        )
    body = tabular(
        "lllllrrrrrc",
        [
            "ID",
            "Dataset",
            "LLM",
            "Comparator",
            "Exposures",
            "Effect",
            "95\\% CI",
            "$t(4)$",
            "raw $p$",
            "Holm $p$",
            "Supported",
        ],
        rows,
        scriptsize=True,
        footnote=(
            "Positive effect = comparator log loss $-$ LLM log loss "
            "(positive favors LLM). All Holm $p=1.0$. "
            r"Membership PRE\_RESULT\_FROZEN; aggregation "
            r"POST\_RESULT\_OPERATIONALIZATION\_REPAIR\_002."
        ),
    )
    write_tex("family_c.tex", body)


def _family_d_merged() -> pd.DataFrame:
    final = load_csv("artifacts/tlt3d/FINAL_FAMILY_D.csv")
    conf = load_csv("artifacts/tlt3d/family_D_confirmatory_results.csv")[
        ["hypothesis_id", "t_statistic"]
    ]
    df = final.merge(conf, left_on="ID", right_on="hypothesis_id", how="left")
    assert len(df) == 36, f"Family D expected 36 rows, got {len(df)}"
    return df


def _family_d_rows(df: pd.DataFrame) -> list[list[str]]:
    rows = []
    for _, r in df.iterrows():
        rows.append(
            [
                esc(r["ID"]),
                esc(r["backbone"]),
                esc(llm_name(r["LLM"])),
                esc(r["comparator"]),
                fmt_float(r["effect_log_loss"], 6),
                fmt_ci(r["CI_lo"], r["CI_hi"], 6),
                fmt_float(r["t_statistic"], 3),
                fmt_sci_or_sig(r["raw_p"]),
                fmt_sci_or_sig(r["holm_p"]),
                esc(str(r["effect_direction"]).replace("_", " ")),
                fmt_yes_no(r["holm_supported"]),
            ]
        )
    return rows


def gen_family_d() -> None:
    df = _family_d_merged()
    header = [
        "ID",
        "Backbone",
        "LLM",
        "Comparator",
        "Effect",
        "95\\% CI",
        "$t(4)$",
        "raw $p$",
        "Holm $p$",
        "Direction",
        "Supported",
    ]
    footnote = (
        "Effect = comparator pooled log loss $-$ LLM pooled log loss "
        "(positive = LLM better). "
        r"Membership PRE\_RESULT\_FROZEN; aggregation "
        r"POST\_RESULT\_OPERATIONALIZATION\_REPAIR\_003."
    )
    parts: list[str] = []
    for key, out_name in [
        ("xes3g5m", "family_d_xes.tex"),
        ("junyi", "family_d_junyi.tex"),
        ("dbe_kt22", "family_d_dbe.tex"),
    ]:
        sub = df[df["dataset"] == key].copy()
        assert len(sub) == 12, f"Family D {key} expected 12 rows, got {len(sub)}"
        body = tabular(
            "llllrrrrrlc",
            header,
            _family_d_rows(sub),
            scriptsize=True,
            footnote=f"{dataset_name(key)}. {footnote}",
        )
        write_tex(out_name, body)
        parts.append(rf"\input{{generated_supplement/{out_name}}}")
    # Wrapper inputs the three split tabulars.
    wrap = (
        "% Auto-generated Family D split by dataset\n"
        + "\n".join(parts)
        + "\n"
    )
    write_tex("family_d.tex", wrap)


def gen_family_d_supported() -> None:
    audit = load_json("artifacts/tlt3d/FINAL_FAMILY_D_SUPPORTED_AUDIT.json")
    final = load_csv("artifacts/tlt3d/FINAL_FAMILY_D.csv").set_index("ID")
    ordered_ids: list[str] = []
    for group in ("LLM_BETTER", "LLM_WORSE"):
        for item in audit[group]:
            ordered_ids.append(item["ID"])
    assert len(ordered_ids) == 5, f"Expected 5 supported rows, got {len(ordered_ids)}"
    assert audit.get("full_distinctive_comparator_triplets_cleared", None) == 0

    # Build manually because of multicolumn group headers.
    lines = [
        r"\scriptsize",
        r"\begin{tabular}{lllllrrr}",
        r"\toprule",
        r"ID & Dataset & Backbone & LLM & Comparator & Effect & 95\% CI & Holm $p$ \\",
        r"\midrule",
    ]
    # POSITIVE header
    lines.append(r"\multicolumn{8}{l}{\textbf{POSITIVE LLM EFFECTS}} \\")
    for item in audit["LLM_BETTER"]:
        r = final.loc[item["ID"]]
        lines.append(
            " & ".join(
                [
                    esc(item["ID"]),
                    esc(dataset_name(r["dataset"])),
                    esc(r["backbone"]),
                    esc(llm_name(r["LLM"])),
                    esc(r["comparator"]),
                    fmt_float(r["effect_log_loss"], 6),
                    fmt_ci(r["CI_lo"], r["CI_hi"], 6),
                    fmt_sci_or_sig(r["holm_p"]),
                ]
            )
            + r" \\"
        )
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{8}{l}{\textbf{NEGATIVE LLM EFFECTS}} \\")
    for item in audit["LLM_WORSE"]:
        r = final.loc[item["ID"]]
        lines.append(
            " & ".join(
                [
                    esc(item["ID"]),
                    esc(dataset_name(r["dataset"])),
                    esc(r["backbone"]),
                    esc(llm_name(r["LLM"])),
                    esc(r["comparator"]),
                    fmt_float(r["effect_log_loss"], 6),
                    fmt_ci(r["CI_lo"], r["CI_hi"], 6),
                    fmt_sci_or_sig(r["holm_p"]),
                ]
            )
            + r" \\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\par\vspace{0.35em}")
    lines.append(
        r"{\footnotesize FULL DISTINCTIVE TRIPLETS CLEARED = 0. "
        r"A Holm-supported individual comparison is not distinctive deployment support.}"
    )
    write_tex("family_d_supported.tex", "\n".join(lines) + "\n")


def gen_statistical_chronology() -> None:
    df = load_csv("artifacts/tlt3d/FINAL_STATISTICAL_CHRONOLOGY.csv")
    # Reader-facing component names for operationalization -> aggregation wording.
    component_map = {
        "Family C operationalization": "Family C aggregation",
        "Family D operationalization": "Family D aggregation",
    }
    rows = []
    for _, r in df.iterrows():
        comp = component_map.get(str(r["component"]), str(r["component"]))
        before = fmt_yes_no(r["specified_before_relevant_result"])
        amend = str(r["amendment_id"])
        if amend == "NONE":
            amend = "--"
        rows.append(
            [
                esc(comp),
                before,
                esc(r["timing"]),
                esc(amend),
                esc(r["final_reporting_status"]),
            ]
        )
    body = tabular(
        "lp{2.2cm}p{4.2cm}p{3.6cm}p{3.4cm}",
        [
            "Component",
            "Specified before results?",
            "Timing",
            "Amendment",
            "Final status",
        ],
        rows,
        scriptsize=True,
    )
    write_tex("statistical_chronology.tex", body)


def gen_six_cell_matrix() -> None:
    df = load_csv("artifacts/tlt3d/FINAL_SIX_CELL_EVIDENCE_MATRIX.csv")
    rows = []
    for _, r in df.iterrows():
        klass = CLASSIFICATION_MAP.get(
            str(r["final_classification"]), str(r["final_classification"])
        )
        rows.append(
            [
                esc(dataset_name(r["dataset"])),
                esc(llm_name(r["LLM"])),
                fmt_yes_no(r["A_supported"]),
                fmt_yes_no(r["B_supported"]),
                fmt_yes_no(r["C_distinctive"]),
                fmt_yes_no(r["D_distinctive"]),
                esc(klass),
            ]
        )
    body = tabular(
        "llccccp{3.6cm}",
        [
            "Dataset",
            "LLM",
            "A supported",
            "B supported",
            "C distinctive",
            "D distinctive",
            "Final classification",
        ],
        rows,
        scriptsize=True,
    )
    write_tex("six_cell_matrix.tex", body)


def gen_rq1_score_summary() -> None:
    df = load_csv("artifacts/tlt3d/P3A_RQ1_SCORE_SUMMARY.csv")
    # Stable order: XES, Junyi, DBE; Mini then 5.4
    order_ds = ["xes3g5m", "junyi", "dbe_kt22"]
    order_m = ["gpt-4o-mini", "gpt-5.4"]
    df["dataset"] = pd.Categorical(df["dataset"], order_ds, ordered=True)
    df["model"] = pd.Categorical(df["model"], order_m, ordered=True)
    df = df.sort_values(["dataset", "model"])
    rows = []
    for _, r in df.iterrows():
        rows.append(
            [
                esc(dataset_name(r["dataset"])),
                esc(llm_name(r["model"])),
                fmt_int(r["n"]),
                fmt_float(r["mean"], 3),
                fmt_float(r["sd"], 3),
                fmt_float(r["median"], 3),
                fmt_float(r["iqr"], 3),
                fmt_float(r["min"], 3),
                fmt_float(r["max"], 3),
            ]
        )
    body = tabular(
        "llrrrrrrr",
        [
            "Dataset",
            "LLM",
            "N",
            "Mean",
            "SD",
            "Median",
            "IQR",
            "Min",
            "Max",
        ],
        rows,
        scriptsize=True,
        footnote="Score-summary universe = LLM-scored items (cross-model agreement Ns).",
    )
    write_tex("rq1_score_summary.tex", body)


def gen_rq1_cross_model() -> None:
    df = load_csv("artifacts/tlt3d/P3A_CROSS_MODEL_AGREEMENT.csv")
    order_ds = ["xes3g5m", "junyi", "dbe_kt22"]
    df["dataset"] = pd.Categorical(df["dataset"], order_ds, ordered=True)
    df = df.sort_values("dataset")
    rows = []
    for _, r in df.iterrows():
        rows.append(
            [
                esc(dataset_name(r["dataset"])),
                fmt_int(r["n"]),
                fmt_float(r["spearman"], 3),
                fmt_ci(r["spearman_ci_lo"], r["spearman_ci_hi"], 3),
                fmt_sci_or_sig(r["spearman_p"]),
                fmt_float(r["pearson"], 3),
                fmt_float(r["kendall"], 3),
            ]
        )
    body = tabular(
        "lrrrrrr",
        [
            "Dataset",
            "N",
            r"Spearman $\rho$",
            "95\\% CI",
            "Spearman $p$",
            "Pearson $r$",
            r"Kendall $\tau$",
        ],
        rows,
        footnote="Cross-model agreement universe = full LLM-scored item set.",
    )
    write_tex("rq1_cross_model.tex", body)


def gen_rq1_surface() -> None:
    df = load_csv("artifacts/tlt3d/P3A_RQ1_SURFACE_ASSOCIATIONS.csv")
    df = df[df["feature"].isin(SHARED_CORE_FEATURES)].copy()
    assert set(df["feature"].unique()) <= set(SHARED_CORE_FEATURES)
    order_ds = ["xes3g5m", "junyi", "dbe_kt22"]
    order_m = ["gpt-4o-mini", "gpt-5.4"]
    df["dataset"] = pd.Categorical(df["dataset"], order_ds, ordered=True)
    df["model"] = pd.Categorical(df["model"], order_m, ordered=True)
    df["feature"] = pd.Categorical(df["feature"], SHARED_CORE_FEATURES, ordered=True)
    df = df.sort_values(["dataset", "model", "feature"])
    rows = []
    for _, r in df.iterrows():
        rows.append(
            [
                esc(dataset_name(r["dataset"])),
                esc(llm_name(r["model"])),
                esc(r["feature"]),
                fmt_int(r["n"]),  # surface Ns
                fmt_float(r["spearman"], 3) if pd.notna(r["spearman"]) else "--",
                fmt_sci_or_sig(r["raw_p"]) if pd.notna(r["raw_p"]) else "--",
            ]
        )
    body = tabular(
        "lllrrr",
        [
            "Dataset",
            "LLM",
            "Feature",
            "N (surface)",
            r"Spearman $\rho$",
            "raw $p$",
        ],
        rows,
        scriptsize=True,
        footnote=(
            "Shared-core features only. N = surface-association universe "
            "(FO primary: XES 3265 / Junyi 169 / DBE 166)."
        ),
    )
    write_tex("rq1_surface.tex", body)


def gen_junyi_sensitivity() -> None:
    pack = load_json("artifacts/tlt3d/FINAL_TLT3D_EVIDENCE_PACK.json")
    sens = pack["I_mandatory_sensitivities"]
    legacy = load_csv("artifacts/tlt3d/P3A_JUNYI_LEGACY_SENSITIVITY.csv")
    primary = load_csv("artifacts/tlt3d/FINAL_FAMILY_A.csv")
    primary = primary[primary["dataset"] == "junyi"].set_index("model")
    fam_b = load_csv("artifacts/tlt3d/FINAL_FAMILY_B.csv")
    fam_b = fam_b[fam_b["dataset"] == "junyi"].set_index("model")

    lines = [
        r"\scriptsize",
        r"\begin{tabular}{lp{8.5cm}}",
        r"\toprule",
        r"Quantity & Value \\",
        r"\midrule",
        rf"Repeated learner--item fraction & {fmt_float(sens['junyi_repeated_learner_item_fraction'], 6)} \\",
        rf"FO vs legacy item-error Spearman & {fmt_float(sens['junyi_fo_vs_legacy_error_spearman'], 6)} \\",
        r"Primary FO $N$ & 169 \\",
        r"Legacy all-response $N$ & 183 \\",
        r"\midrule",
        r"\multicolumn{2}{l}{\textbf{Family A Spearman $\rho$ (PRIMARY FO vs LEGACY)}} \\",
    ]
    for model, label in [("gpt-4o-mini", "Mini"), ("gpt-5.4", "5.4")]:
        fo = float(primary.loc[model, "rho"])
        leg = float(legacy.loc[legacy["model"] == model, "spearman_rho"].iloc[0])
        lines.append(
            rf"{label}: FO / legacy $\rho$ & {fmt_float(fo, 3)} / {fmt_float(leg, 3)} \\"
        )
    lines.append(r"\midrule")
    lines.append(
        r"\multicolumn{2}{l}{\textbf{Family B $\Delta R^2$ (PRIMARY FO vs LEGACY)}} \\"
    )
    for model, label in [("gpt-4o-mini", "Mini"), ("gpt-5.4", "5.4")]:
        fo_d = float(fam_b.loc[model, "delta_R2"])
        leg_d = float(legacy.loc[legacy["model"] == model, "delta_r2"].iloc[0])
        lines.append(
            rf"{label}: FO / legacy $\Delta R^2$ & {fmt_float(fo_d, 3)} / {fmt_float(leg_d, 3)} \\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\par\vspace{0.35em}",
        r"{\footnotesize FIRST\_OBSERVED = PRIMARY; ALL\_RESPONSE = LEGACY\_SENSITIVITY.}",
        "",
    ]
    write_tex("junyi_sensitivity.tex", "\n".join(lines))


def gen_xes_legacy_sensitivity() -> None:
    df = load_csv("artifacts/tlt3d/P3A_XES_LEGACY_SENSITIVITY.csv")
    primary = load_csv("artifacts/tlt3d/FINAL_FAMILY_A.csv")
    primary = primary[primary["dataset"] == "xes3g5m"].set_index("model")
    fam_b = load_csv("artifacts/tlt3d/FINAL_FAMILY_B.csv")
    fam_b = fam_b[fam_b["dataset"] == "xes3g5m"].set_index("model")
    rows = []
    for _, r in df.iterrows():
        model = r["model"]
        rows.append(
            [
                esc(llm_name(model)),
                "PRIMARY FO",
                "3265",
                fmt_float(primary.loc[model, "rho"], 3),
                fmt_ci(primary.loc[model, "CI_lo"], primary.loc[model, "CI_hi"], 3),
                fmt_float(fam_b.loc[model, "delta_R2"], 3),
            ]
        )
        rows.append(
            [
                esc(llm_name(model)),
                "LEGACY",
                fmt_int(r["n_items"]),
                fmt_float(r["spearman_rho"], 3),
                fmt_ci(r["ci_lo"], r["ci_hi"], 3),
                fmt_float(r["delta_r2"], 3),
            ]
        )
    body = tabular(
        "llrrrr",
        [
            "LLM",
            "Construct",
            "N",
            r"Spearman $\rho$",
            "95\\% CI",
            r"$\Delta R^2$",
        ],
        rows,
        footnote=(
            "PRIMARY FO $N=3265$; LEGACY all-response $N=3279$ "
            "(LEGACY ALL-RESPONSE SENSITIVITY only)."
        ),
    )
    write_tex("xes_legacy_sensitivity.tex", body)


def gen_dbe_correctness() -> None:
    pack = load_json("artifacts/tlt3d/FINAL_TLT3D_EVIDENCE_PACK.json")
    d = pack["I_mandatory_sensitivities"]["dbe_correctness"]
    rows = [
        ["Primary response label", esc(d["primary"])],
        ["Disagreements", fmt_int(d["disagreements"])],
        ["Disagreement rate", f"{100.0 * float(d['rate']):.4f}\\%"],
        ["Affected learners", fmt_int(d["affected_learners"])],
        ["Affected items", fmt_int(d["affected_items"])],
        [
            "Primary vs consensus item-error Spearman",
            fmt_float(d["primary_vs_consensus_item_error_spearman"], 3),
        ],
        ["Items losing $\\geq 20$ threshold", "0"],
        ["Family A/B qualitative conclusions", "unchanged"],
    ]
    body = tabular(
        "ll",
        ["Quantity", "Value"],
        rows,
        footnote="DBE correctness sensitivity; primary label = answer\\_state.",
    )
    write_tex("dbe_correctness.tex", body)


def gen_dbe_expert() -> None:
    df = load_csv("artifacts/tlt3d/P3A_DBE_EXPERT_SECONDARY.csv")
    lines = [
        r"\scriptsize",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Expert level & Text-complete count & Raw source count \\",
        r"\midrule",
    ]
    for level in (1, 2, 3):
        lines.append(
            f"{level} & {DBE_EXPERT_TEXT_COMPLETE[level]} & {DBE_EXPERT_RAW[level]} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\par\vspace{0.6em}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Contrast & $N$ & Spearman $\rho$ & Kendall $\tau$ \\",
        r"\midrule",
    ]
    contrast_labels = {
        "expert_vs_FO_learner_error": "Expert vs FO learner error",
        "expert_vs_gpt-4o-mini": "Expert vs gpt-4o-mini",
        "expert_vs_gpt-5.4": "Expert vs gpt-5.4",
    }
    for _, r in df.iterrows():
        lines.append(
            " & ".join(
                [
                    esc(contrast_labels.get(str(r["contrast"]), r["contrast"])),
                    fmt_int(r["n"]),
                    fmt_float(r["spearman"], 3),
                    fmt_float(r["kendall"], 3),
                ]
            )
            + r" \\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\par\vspace{0.35em}",
        r"{\footnotesize SECONDARY EXPLORATORY ANALYSIS. "
        r"Text-complete expert levels hardcoded from verified universe "
        r"(1:80, 2:71, 3:15); raw source (1:95, 2:84, 3:33).}",
        "",
    ]
    write_tex("dbe_expert.tex", "\n".join(lines))


def gen_simulation_ladder() -> None:
    df = load_csv("tables/SYNTHETIC_ALIGNMENT_SEED_SUMMARY.csv")
    g = (
        df.groupby("condition", sort=True)["rho_d_llm_sim_error"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    rows = []
    for _, r in g.iterrows():
        cond = str(r["condition"])
        label = SIM_CONDITION_LABELS.get(cond, cond)
        rows.append(
            [
                esc(cond),
                esc(label),
                fmt_int(r["count"]),
                f"{fmt_float(r['mean'], 3)}$\\pm${fmt_float(r['std'], 3)}",
            ]
        )
    body = tabular(
        "llrl",
        [
            "Condition",
            "Label",
            "Seeds",
            r"mean$\pm$sd $\rho(d_{\mathrm{LLM}},\mathrm{sim\_error})$",
        ],
        rows,
        footnote=(
            "GSM8K simulation-only (200 problems; 10 paired seeds). "
            "Values from tables/SYNTHETIC\\_ALIGNMENT\\_SEED\\_SUMMARY.csv."
        ),
    )
    write_tex("simulation_ladder.tex", body)


def gen_reproducibility_manifest() -> None:
    # Cross-check against P3C_DBE_INPUT_CONTROL_MANIFEST.json (authorized source).
    man = load_json("artifacts/tlt3d/P3C_DBE_INPUT_CONTROL_MANIFEST.json")
    expected = {
        "DBE item universe": man["item_universe_hash"],
        "DBE score CSV": man["llm_score_csv_hash"],
        "DBE learner split": man["learner_split_hash"],
        "DBE unseen folds": man["unseen_fold_hash"],
        "DBE Random-Permuted": man["random_permuted_hash"],
        "DBE CharacterLength": man["character_length_hash"],
        "Family-D preflight": man["model_manifest"]["code_commit"],
    }
    for label, val in REPRO_MANIFEST_ROWS:
        if label in expected and expected[label] != val:
            raise SystemExit(
                f"Repro hash mismatch for {label}: PI list {val} vs manifest {expected[label]}"
            )
    rows = [[esc(a), r"\texttt{" + esc(b) + "}"] for a, b in REPRO_MANIFEST_ROWS]
    body = tabular(
        "lp{9.2cm}",
        ["Artifact", "Hash / commit"],
        rows,
        scriptsize=True,
        footnote=(
            "Authorized PI section-34 hashes; DBE content hashes cross-checked against "
            r"P3C\_DBE\_INPUT\_CONTROL\_MANIFEST.json."
        ),
    )
    write_tex("reproducibility_manifest.tex", body)


def gen_family_b_repair001_mcse() -> None:
    path = ART / "P3A1_FAMILY_B_REPAIRED_PVALUES.csv"
    if not path.exists():
        write_tex(
            "family_b_repair001_mcse.tex",
            "% P3A1_FAMILY_B_REPAIRED_PVALUES.csv not present; table omitted.\n",
        )
        return
    df = pd.read_csv(path)
    rows = []
    for _, r in df.iterrows():
        rows.append(
            [
                esc(r["hypothesis_id"]),
                fmt_sci_or_sig(r["p_raw_repaired"]),
                fmt_float(r["p_raw_mc_se"], 6),
                fmt_sci_or_sig(r["p_holm_repaired"]),
            ]
        )
    body = tabular(
        "lrrr",
        [
            "ID",
            r"$p_{\mathrm{raw}}$",
            "MC SE",
            "Holm $p$",
        ],
        rows,
        footnote=(
            r"Monte-Carlo SE for POST\_RESULT\_INFERENTIAL\_REPAIR\_001 "
            r"($B=100000$ paired prediction-label randomization)."
        ),
    )
    write_tex("family_b_repair001_mcse.tex", body)


def write_readme(generated: list[str]) -> None:
    lines = [
        "# Generated supplement table fragments",
        "",
        "Deterministically produced by `scripts/render_tlt_final_supplement_tables.py`.",
        "Overwrite on each run. Frozen artifacts only -- no new science.",
        "",
        "## Outputs and sources",
        "",
        "| File | Source |",
        "|---|---|",
        "| `dataset_universes.tex` | `artifacts/tlt3d/FINAL_DATASET_UNIVERSES.csv` + PI-authorized GSM8K row |",
        "| `family_a.tex` | `artifacts/tlt3d/FINAL_FAMILY_A.csv` |",
        "| `family_a_secondary.tex` | `artifacts/tlt3d/family_A_confirmatory_results.csv` |",
        "| `family_b.tex` | `artifacts/tlt3d/FINAL_FAMILY_B.csv` |",
        "| `family_b_repair001_mcse.tex` | `artifacts/tlt3d/P3A1_FAMILY_B_REPAIRED_PVALUES.csv` |",
        "| `family_c.tex` | `FINAL_FAMILY_C.csv` + `family_C_confirmatory_results.csv` |",
        "| `family_d_xes.tex` / `family_d_junyi.tex` / `family_d_dbe.tex` | `FINAL_FAMILY_D.csv` + `family_D_confirmatory_results.csv` |",
        "| `family_d.tex` | `\\input` wrapper of the three Family-D splits |",
        "| `family_d_supported.tex` | `FINAL_FAMILY_D_SUPPORTED_AUDIT.json` + `FINAL_FAMILY_D.csv` |",
        "| `statistical_chronology.tex` | `FINAL_STATISTICAL_CHRONOLOGY.csv` |",
        "| `six_cell_matrix.tex` | `FINAL_SIX_CELL_EVIDENCE_MATRIX.csv` |",
        "| `rq1_score_summary.tex` | `P3A_RQ1_SCORE_SUMMARY.csv` |",
        "| `rq1_cross_model.tex` | `P3A_CROSS_MODEL_AGREEMENT.csv` |",
        "| `rq1_surface.tex` | `P3A_RQ1_SURFACE_ASSOCIATIONS.csv` (shared-core 7) |",
        "| `junyi_sensitivity.tex` | evidence pack + `P3A_JUNYI_LEGACY_SENSITIVITY.csv` |",
        "| `xes_legacy_sensitivity.tex` | `P3A_XES_LEGACY_SENSITIVITY.csv` + primary FO rows |",
        "| `dbe_correctness.tex` | evidence pack `I_mandatory_sensitivities.dbe_correctness` |",
        "| `dbe_expert.tex` | evidence pack / `P3A_DBE_EXPERT_SECONDARY.csv` + hardcoded expert counts |",
        "| `simulation_ladder.tex` | `tables/SYNTHETIC_ALIGNMENT_SEED_SUMMARY.csv` |",
        "| `reproducibility_manifest.tex` | PI section 34 + `P3C_DBE_INPUT_CONTROL_MANIFEST.json` |",
        "",
        "## Generated this run",
        "",
    ]
    for name in generated:
        lines.append(f"- `{name}`")
    lines.append("")
    write_tex("GENERATED_README.md", "\n".join(lines))


def count_summary() -> dict[str, int]:
    a = load_csv("artifacts/tlt3d/FINAL_FAMILY_A.csv")
    b = load_csv("artifacts/tlt3d/FINAL_FAMILY_B.csv")
    c = load_csv("artifacts/tlt3d/FINAL_FAMILY_C.csv")
    d = load_csv("artifacts/tlt3d/FINAL_FAMILY_D.csv")
    audit = load_json("artifacts/tlt3d/FINAL_FAMILY_D_SUPPORTED_AUDIT.json")
    return {
        "A_rows": len(a),
        "A_supported": int(a["holm_supported"].astype(bool).sum()),
        "B_rows": len(b),
        "B_supported": int(b["holm_supported"].astype(bool).sum()),
        "C_rows": len(c),
        "C_supported": int(c["holm_supported"].astype(bool).sum()),
        "D_rows": len(d),
        "D_pos": int(audit["positive_holm_supported_count"]),
        "D_neg": int(audit["negative_holm_supported_count"]),
        "triplets": int(audit["full_distinctive_comparator_triplets_cleared"]),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    generators = [
        gen_dataset_universes,
        gen_family_a,
        gen_family_a_secondary,
        gen_family_b,
        gen_family_c,
        gen_family_d,
        gen_family_d_supported,
        gen_statistical_chronology,
        gen_six_cell_matrix,
        gen_rq1_score_summary,
        gen_rq1_cross_model,
        gen_rq1_surface,
        gen_junyi_sensitivity,
        gen_xes_legacy_sensitivity,
        gen_dbe_correctness,
        gen_dbe_expert,
        gen_simulation_ladder,
        gen_reproducibility_manifest,
        gen_family_b_repair001_mcse,
    ]
    for fn in generators:
        fn()

    generated = sorted(p.name for p in OUT.iterdir() if p.is_file())
    write_readme(generated)

    counts = count_summary()
    print(
        "COUNTS: "
        f"A{counts['A_rows']}/{counts['A_supported']}, "
        f"B{counts['B_rows']}/{counts['B_supported']}, "
        f"C{counts['C_rows']}/{counts['C_supported']}, "
        f"D{counts['D_rows']}/{counts['D_pos']}/{counts['D_neg']}, "
        f"triplets{counts['triplets']}"
    )
    print(f"Wrote {len(generated)} files under {OUT}")


if __name__ == "__main__":
    main()

# After generation, run: python3 scripts/tlt3d_p4d2_postprocess_generated_tables.py
