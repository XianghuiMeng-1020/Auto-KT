#!/usr/bin/env python3
"""Deterministic render of the TLT-3D final study-overview evidence ladder.

Outputs:
  manuscript_tlt/figures/study_overview.pdf
  manuscript_tlt/figures/study_overview.svg

Designed near IEEE figure* textwidth so matplotlib pt ≈ print pt.
Figure-only; display values are PI-authorized.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "manuscript_tlt" / "figures"
PDF_PATH = OUT_DIR / "study_overview.pdf"
SVG_PATH = OUT_DIR / "study_overview.svg"

FIG_W_IN = 7.16
FIG_H_IN = 3.85

C_BG = "#FFFFFF"
C_AUTH_FILL = "#EEF3F8"
C_AUTH_EDGE = "#2F4A6D"
C_STAGE_FILL = "#FFFFFF"
C_STAGE_EDGE = "#2F4A6D"
C_LEFT_FILL = "#F4F6F8"
C_LEFT_EDGE = "#4A5568"
C_SYN_FILL = "#FFF8F0"
C_SYN_EDGE = "#8B5A2B"
C_TAKE_FILL = "#F2F2F2"
C_TAKE_EDGE = "#333333"
C_TEXT = "#1A1A1A"
C_MUTED = "#4A5568"
C_HEAD = "#1B3A5C"
C_ARROW = "#2F4A6D"
C_SYN_ARROW = "#8B5A2B"


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _rect(ax, x, y, w, h, *, fc, ec, lw=1.0, ls="-", z=2, hatch=None, alpha=1.0):
    ax.add_patch(
        Rectangle(
            (x, y),
            w,
            h,
            facecolor=fc,
            edgecolor=ec,
            linewidth=lw,
            linestyle=ls,
            hatch=hatch,
            alpha=alpha,
            zorder=z,
            clip_on=False,
        )
    )


def _arrow(ax, x1, y1, x2, y2, *, color=C_ARROW, lw=1.2, ls="-"):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=9,
            linewidth=lw,
            color=color,
            linestyle=ls,
            zorder=4,
            clip_on=False,
        )
    )


def _txt(
    ax,
    x,
    y,
    s,
    *,
    size=8.0,
    weight="normal",
    color=C_TEXT,
    ha="center",
    va="center",
    style="normal",
):
    ax.text(
        x,
        y,
        s,
        fontsize=size,
        fontweight=weight,
        fontstyle=style,
        color=color,
        ha=ha,
        va=va,
        zorder=6,
        clip_on=True,
        family="DejaVu Sans",
        linespacing=1.12,
    )


def draw(ax) -> None:
    # ---- A. Left prior ----
    lx, lw = 0.010, 0.125
    _rect(ax, lx, 0.300, lw, 0.655, fc=C_LEFT_FILL, ec=C_LEFT_EDGE, lw=1.15)
    _txt(ax, lx + lw / 2, 0.925, "A. Content-derived\nprior", size=7.5, weight="bold", color=C_MUTED)
    _txt(ax, lx + lw / 2, 0.860, "ITEM CONTENT", size=8.0, weight="bold", color=C_HEAD)
    _txt(ax, lx + lw / 2, 0.815, "(visible stem /\noptions)", size=7.5, color=C_MUTED)
    _arrow(ax, lx + lw / 2, 0.780, lx + lw / 2, 0.735)
    _rect(ax, lx + 0.008, 0.575, lw - 0.016, 0.145, fc="#FFFFFF", ec=C_LEFT_EDGE, lw=0.95)
    _txt(ax, lx + lw / 2, 0.675, "GPT-4o-mini", size=7.8, weight="bold")
    _txt(ax, lx + lw / 2, 0.615, "GPT-5.4", size=7.8, weight="bold")
    _arrow(ax, lx + lw / 2, 0.560, lx + lw / 2, 0.505)
    _rect(ax, lx + 0.008, 0.340, lw - 0.016, 0.150, fc="#FFFFFF", ec=C_LEFT_EDGE, lw=0.95)
    _txt(ax, lx + lw / 2, 0.445, "CONTENT-\nDERIVED\nDIFFICULTY", size=7.5, weight="bold", color=C_HEAD)
    _txt(ax, lx + lw / 2, 0.360, "item-side prior", size=7.5, color=C_MUTED, style="italic")
    _arrow(ax, lx + lw + 0.002, 0.415, 0.152, 0.415, lw=1.3)

    # ---- B. Authentic header + stages (wide) ----
    _txt(ax, 0.155, 0.965, "B. Authentic evidence ladder", size=7.8, weight="bold", color=C_MUTED, ha="left")
    _rect(ax, 0.152, 0.855, 0.838, 0.095, fc=C_AUTH_FILL, ec=C_AUTH_EDGE, lw=1.25)
    _txt(ax, 0.571, 0.920, "AUTHENTIC LEARNER EVIDENCE", size=9.0, weight="bold", color=C_HEAD)
    _txt(
        ax,
        0.571,
        0.880,
        "FIRST-OBSERVED PRIMARY ITEMS  ·  XES3G5M: 3,265  ·  Junyi Academy: 169  ·  DBE-KT22: 166",
        size=7.5,
        weight="bold",
    )

    stages = [
        (
            "1. Visible-content\ncharacterization",
            "Surface-feature\nassociations;\ncross-model\nρ = .657–.776\n\nlength-related\nfeatures\nprominent",
        ),
        (
            "2. Authentic learner\ncorrespondence",
            "5/6 Holm-supported\n\nSpearman ρ =\n.103–.354\n\nheld-out first-\nobserved\nlearner error",
        ),
        (
            "3. Incremental learner\ninformation",
            "1/6 supported\n\nbeyond shared\ntransparent\nitem features",
        ),
        (
            "4. Response-limited\nKT",
            "0/12 Holm-supported\n\nGRU confirmatory\nk = 0,1,3,5,10,20\n\nStandard +\nRandom-\nResampledScore",
        ),
        (
            "5. Genuine unseen-\nitem KT",
            "3 positive +\n2 negative\nisolated Holm-\nsupported effects\n0 full distinctive\ntriplets\n\nGRU + SAKT\nStandard +\nRandom-\nPermutedScore +\nCharacterLength\n0 target; shared UNK",
        ),
    ]

    sx0, sw, gap = 0.152, 0.155, 0.014
    sy, sh = 0.300, 0.525
    header_h = 0.095
    for i, (title, body) in enumerate(stages):
        x = sx0 + i * (sw + gap)
        _rect(ax, x, sy, sw, sh, fc=C_STAGE_FILL, ec=C_STAGE_EDGE, lw=1.05)
        _rect(ax, x, sy + sh - header_h, sw, header_h, fc=C_AUTH_FILL, ec=C_STAGE_EDGE, lw=1.05)
        ax.plot(
            [x, x + sw],
            [sy + sh - header_h, sy + sh - header_h],
            color=C_STAGE_EDGE,
            lw=0.9,
            zorder=5,
        )
        _txt(ax, x + sw / 2, sy + sh - header_h / 2, title, size=7.6, weight="bold", color=C_HEAD)
        _txt(ax, x + sw / 2, sy + (sh - header_h) / 2, body, size=7.5, color=C_TEXT)
        if i < 4:
            _arrow(ax, x + sw + 0.001, sy + sh / 2, x + sw + gap - 0.001, sy + sh / 2, lw=1.15)

    _txt(
        ax,
        0.571,
        0.275,
        "Progressive outcomes:  5/6  →  1/6  →  0/12  →  isolated 3+/2−  with  0 full distinctive triplets",
        size=7.5,
        weight="bold",
        color=C_HEAD,
    )

    # ---- C. Synthetic branch as lower separate strip ----
    _rect(ax, 0.010, 0.125, 0.980, 0.125, fc=C_SYN_FILL, ec=C_SYN_EDGE, lw=1.3, ls="--")
    _rect(ax, 0.010, 0.125, 0.980, 0.125, fc="none", ec=C_SYN_EDGE, lw=0, hatch="///", alpha=0.14, z=3)
    _txt(
        ax,
        0.020,
        0.188,
        "C. Controlled synthetic alignment  (GSM8K simulation — not authentic learner data)",
        size=7.6,
        weight="bold",
        color=C_SYN_EDGE,
        ha="left",
    )
    # horizontal flow boxes
    boxes = [
        (0.02, 0.135, 0.16, "GSM8K\nsimulation"),
        (0.22, 0.135, 0.20, "Signal decoupling\n(shared structure removed)"),
        (0.46, 0.135, 0.22, "ρ = .962 → .007\nas structure is removed"),
        (0.72, 0.135, 0.25, "Generator recovery is not\nauthentic learner validation"),
    ]
    for x, y, w, lab in boxes:
        _rect(ax, x, y, w, 0.070, fc="#FFFFFF", ec=C_SYN_EDGE, lw=0.95, ls="--")
        _txt(ax, x + w / 2, y + 0.035, lab, size=7.5, weight="bold" if "ρ =" in lab or "Generator" in lab else "normal")
    _arrow(ax, 0.185, 0.170, 0.215, 0.170, color=C_SYN_ARROW, lw=1.0)
    _arrow(ax, 0.425, 0.170, 0.455, 0.170, color=C_SYN_ARROW, lw=1.0)
    _arrow(ax, 0.685, 0.170, 0.715, 0.170, color=C_SYN_ARROW, lw=1.0)

    # ---- Takeaway ----
    _rect(ax, 0.010, 0.012, 0.980, 0.095, fc=C_TAKE_FILL, ec=C_TAKE_EDGE, lw=1.2)
    _txt(
        ax,
        0.50,
        0.070,
        "Support at an earlier evidence stage does not automatically establish stronger deployment evidence.",
        size=8.0,
        weight="bold",
    )
    _txt(
        ax,
        0.50,
        0.032,
        "Correspondence ≠ incremental information ≠ response-limited utility ≠ distinctive unseen-item utility",
        size=7.5,
        color=C_MUTED,
        style="italic",
    )


def render() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig, ax = plt.subplots(figsize=(FIG_W_IN, FIG_H_IN), dpi=120)
    fig.patch.set_facecolor(C_BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    ax.set_facecolor(C_BG)
    fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
    draw(ax)
    fig.savefig(PDF_PATH, format="pdf", bbox_inches=None, pad_inches=0, facecolor=C_BG)
    fig.savefig(SVG_PATH, format="svg", bbox_inches=None, pad_inches=0, facecolor=C_BG)
    plt.close(fig)


if __name__ == "__main__":
    render()
    print(f"Wrote {PDF_PATH}")
    print(f"Wrote {SVG_PATH}")
    print(f"PDF SHA256: {file_sha256(PDF_PATH)}")
    print(f"SVG SHA256: {file_sha256(SVG_PATH)}")
