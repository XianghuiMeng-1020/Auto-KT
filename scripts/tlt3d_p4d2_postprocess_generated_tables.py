#!/usr/bin/env python3
"""Post-process generated supplement tables for LaTeX safety + secondary summaries."""
from __future__ import annotations
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "manuscript_tlt" / "generated_supplement"


def esc(s: str) -> str:
    return str(s).replace("_", "\\_").replace("&", "\\&")


def rewrite_chronology() -> None:
    df = pd.read_csv(ROOT / "artifacts/tlt3d/FINAL_STATISTICAL_CHRONOLOGY.csv")

    def short_amend(a: str) -> str:
        a = str(a)
        if "001" in a:
            return "REPAIR\\_001"
        if "002" in a:
            return "REPAIR\\_002"
        if "003" in a:
            return "REPAIR\\_003"
        return "NONE"

    def short_status(s: str) -> str:
        s = str(s)
        if "CONFIRMATORY" in s:
            return "PRE\\_CONFIRMATORY"
        if "INFERENTIAL" in s:
            return "REPAIR\\_001"
        if "OPERATIONALIZATION" in s:
            return "POST\\_OP\\_REPAIR"
        if "FROZEN" in s:
            return "PRE\\_FROZEN"
        return esc(s)

    lines = [
        "\\scriptsize",
        "\\begin{tabular}{lp{3.2cm}p{4.0cm}ll}",
        "\\toprule",
        "Component & Specified before results? & Timing & Amendment & Final status \\\\",
        "\\midrule",
    ]
    for _, r in df.iterrows():
        before = "Yes" if r.specified_before_relevant_result else "No"
        lines.append(
            f"{esc(r.component)} & {before} & {esc(r.timing)} & "
            f"{short_amend(r.amendment_id)} & {short_status(r.final_reporting_status)} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    (OUT / "statistical_chronology.tex").write_text("\n".join(lines) + "\n")


def rewrite_repro() -> None:
    rows = [
        ("Evidence freeze commit", "a66ff788d901cc815b3a248501005eb9d523b1a7"),
        ("DBE scoring commit", "e726237f003e105d036a7b8e439385a152300e42"),
        ("DBE item universe", "d62fb95604bac94e65adda498dc21175e66ce75198836996f5f53695c96e38a6"),
        ("DBE score CSV", "484dd79a140372fcbda66275aa472505de47f0852cd37992692ac87c97e22f64"),
        ("DBE learner split", "65d13de13e7c3a8b366c63628cab3e7ef3f2a67b8eae7cde1aab4e1a53d627dc"),
        ("DBE unseen folds", "28efbf33c231772a3565056367c4bf6dfa62bdf553abcfd8397aa9d9037d4e0a"),
        ("DBE Random-Permuted", "d859d9aafac5209b34921d3efae74ed5bfaf0c08ab7d2aef3dfc8239d7f937ce"),
        ("DBE CharacterLength", "280e98d48dfc19b903719873d58f25b839d3cce6d2562461e13f946f109fe9cd"),
        ("Family-C operational amendment", "0609954254b82fa689c1f228eb486ed823a23ef6"),
        ("Final Family-C result", "2d1206090074aa49f937d77955bc1e8d6aca22bf"),
        ("Family-D preflight", "c7e5144c3516a2b63c1ab4fa4014191fe032159f"),
        ("Final Family-D result", "a188e64129b8f012caa7ac44ae5c4bea84fa6472"),
    ]
    lines = ["\\begin{itemize}"]
    for name, h in rows:
        parts = "\\allowbreak ".join(h[i : i + 16] for i in range(0, len(h), 16))
        lines.append(f"\\item \\textbf{{{name}}}: \\texttt{{{parts}}}")
    lines.append("\\end{itemize}")
    (OUT / "reproducibility_manifest.tex").write_text("\n".join(lines) + "\n")


def secondary_summaries() -> None:
    df = pd.read_csv(ROOT / "artifacts/tlt3d/P3B2_RESPONSE_LIMITED_AGGREGATED_RESULTS.csv")
    sub = df[(df.backbone == "GRU") & (df.exposure.astype(str).isin(list("013520") + ["10", "20", "warm"]))].copy()
    # limited means
    rows = []
    for ds in sorted(sub.dataset.unique()):
        for cond in ["TrainEmpDiff", "LLM-Mini", "LLM-5.4", "Standard", "Random-Scalar"]:
            g = sub[(sub.dataset == ds) & (sub.condition == cond) & (sub.exposure.astype(str) != "warm")]
            if g.empty:
                continue
            rows.append((ds, cond, g.log_loss_mean.mean(), g.auc_mean.mean()))
    lines = [
        "\\scriptsize",
        "\\begin{tabular}{llrr}",
        "\\toprule",
        "Dataset & Condition & Mean log loss (k$\\in\\{0,1,3,5,10,20\\}$) & Mean AUC \\\\",
        "\\midrule",
    ]
    for ds, cond, ll, auc in rows:
        lines.append(f"{esc(ds)} & {esc(cond)} & {ll:.6f} & {auc:.3f} \\\\")
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\par\\vspace{0.35em}",
        "{\\footnotesize SECONDARY. Random-Scalar naming retained from secondary archive tables; Family~C confirmatory comparator is Random-ResampledScore.}",
    ]
    (OUT / "secondary_limited_gru_means.tex").write_text("\n".join(lines) + "\n")

    warm = sub[sub.exposure.astype(str) == "warm"]
    lines = [
        "\\scriptsize",
        "\\begin{tabular}{llrr}",
        "\\toprule",
        "Dataset & Condition & Warm log loss & Warm AUC \\\\",
        "\\midrule",
    ]
    for _, r in warm.sort_values(["dataset", "condition"]).iterrows():
        lines.append(f"{esc(r.dataset)} & {esc(r.condition)} & {r.log_loss_mean:.6f} & {r.auc_mean:.3f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\par\\vspace{0.35em}", "{\\footnotesize SECONDARY warm reference.}"]
    (OUT / "secondary_warm.tex").write_text("\n".join(lines) + "\n")

    un = pd.read_csv(ROOT / "artifacts/tlt3d/P3C_UNSEEN_AGGREGATED_RESULTS.csv")
    lines = [
        "\\scriptsize",
        "\\begin{tabular}{lllrr}",
        "\\toprule",
        "Dataset & Backbone & Condition & Log loss mean & AUC mean \\\\",
        "\\midrule",
    ]
    for _, r in un.sort_values(["dataset", "backbone", "condition"]).iterrows():
        lines.append(
            f"{esc(r.dataset)} & {esc(r.backbone)} & {esc(r.condition)} & {r.log_loss_mean:.6f} & {r.auc_mean:.3f} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", "\\par\\vspace{0.35em}", "{\\footnotesize SECONDARY unseen aggregated means; AUC secondary.}"]
    (OUT / "secondary_unseen_auc.tex").write_text("\n".join(lines) + "\n")

    sakt = pd.read_csv(ROOT / "tables/TLT_SAKT_LIMITED_SUMMARY.csv")
    keep = sakt[
        sakt.condition.isin(["LLM-Mini", "LLM-5.4", "Standard", "Random-Scalar", "TrainEmpDiff", "CharacterLength"])
    ]
    lines = [
        "\\scriptsize",
        "\\begin{tabular}{llrrr}",
        "\\toprule",
        "Dataset & Limit & Condition & Log loss & AUC \\\\",
        "\\midrule",
    ]
    for _, r in keep.sort_values(["dataset", "response_limit", "condition"]).iterrows():
        lines.append(
            f"{esc(r.dataset)} & {esc(r.response_limit)} & {esc(r.condition)} & {r.test_log_loss_mean:.4f} & {r.auc_mean:.3f} \\\\"
        )
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\par\\vspace{0.35em}",
        "{\\footnotesize SECONDARY SAKT limited archive. CharacterLength here is not Family~C confirmatory. OracleEmpDiff was SECONDARY\\_NOT\\_EXECUTABLE\\_FROM\\_FROZEN\\_IMPLEMENTATION.}",
    ]
    (OUT / "secondary_sakt_limited.tex").write_text("\n".join(lines) + "\n")


def wrap_wide_tables() -> None:
    for name in [
        "family_a.tex",
        "family_b.tex",
        "family_c.tex",
        "family_d_xes.tex",
        "family_d_junyi.tex",
        "family_d_dbe.tex",
        "rq1_surface.tex",
        "dataset_universes.tex",
        "six_cell_matrix.tex",
        "secondary_sakt_limited.tex",
        "secondary_unseen_auc.tex",
        "secondary_limited_gru_means.tex",
    ]:
        p = OUT / name
        if not p.exists():
            continue
        t = p.read_text()
        if "resizebox" in t:
            continue
        t2 = t.replace("\\scriptsize\n", "")
        p.write_text("\\resizebox{\\textwidth}{!}{%\n" + t2.rstrip() + "\n}\n")




def ensure_junyi_exact_scalars() -> None:
    path = OUT / "junyi_sensitivity.tex"
    if not path.exists():
        return
    txt = path.read_text()
    if "0.8254259684261771" in txt:
        return
    nl = chr(10)
    path.write_text(
        txt.rstrip()
        + nl + r"\parspace{0.35em}" + nl
        + r"{ootnotesize Exact frozen scalars: repeated learner--item rate $=0.8254259684261771$; "
        + r"FO vs legacy item-error Spearman $=0.7167309868945884$.}" + nl
    )

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    rewrite_chronology()
    rewrite_repro()
    secondary_summaries()
    ensure_junyi_exact_scalars()
    wrap_wide_tables()
    print("P4D2 postprocess complete")
