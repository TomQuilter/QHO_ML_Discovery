"""Build the ML-versus-lattice Word summary for lambda = 0.01, 0.1, 0.3, 0.5."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
OUT_DOC = ROOT / "reports" / "ML_vs_lattice_lambda_summary.docx"

SCAN = ROOT / "runs" / "20261001_113934_MU_lambda0_0p01_T2to4_dt0.2_N16" / "MU_lambda0_0p01_T2to4_vs_lattice_mehler.csv"
UP = ROOT / "runs" / "lambda_up_0p3_to_1" / "results.csv"
DENSE = {
    0.01: ROOT / "Re__Complex_Time_Code_" / "py_scripts_extracted" / "exact_check_Time_Loop_lambda0p01_T0to8_dt0.01.csv",
    0.1: ROOT / "Re__Complex_Time_Code_" / "py_scripts_extracted" / "exact_check_Time_Loop_lambda0p1_T0to8_dt0.01.csv",
}
LAM01_5 = ROOT / "runs" / "20261001_143307_MU_multithimble_lambda0.1_N16" / "MU_lambda0p1_T2to4_dt0.2_5thimble_vs_lattice.csv"
BIG_05 = ROOT / "runs" / "lambda0p5_T3p6_bigger_nets" / "summary.json"

NAVY = "#1f4e79"
AMBER = "#d97706"
RED = "#b91c1c"
GREEN = "#15803d"
GREY = "#94a3b8"


def cnum(re, im):
    return complex(float(re), float(im))


def rel(a, b):
    return abs(a - b) / abs(b)


def fmt_c(z, nd=4):
    sign = "-" if z.imag < 0 else "+"
    return f"{z.real:+.{nd}f} {sign} {abs(z.imag):.{nd}f}i"


def pct(x):
    p = 100.0 * x
    if p >= 10:
        return f"{p:.0f}%"
    text = f"{p:.1f}".rstrip("0").rstrip(".")
    return f"{text}%"


def load_scan(lam):
    rows = []
    with SCAN.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if abs(float(row["lambda"]) - lam) > 1e-12:
                continue
            rows.append({
                "T": float(row["T"]),
                "ml": cnum(row["K_ML_re"], row["K_ML_im"]),
                "lat": cnum(row["K_lattice_re"], row["K_lattice_im"]),
                "err": float(row["complex_rel_err_vs_lattice"]),
                "one": float(row["one_thimble_err_vs_lattice"]),
                "ess_b": row["bounce_ESS_over_N"],
                "status": row["status"],
            })
    rows.sort(key=lambda r: r["T"])
    return rows


def load_up(lam):
    rows = []
    with UP.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if abs(float(row["lambda"]) - lam) > 1e-12 or row["used_for_scan"] != "True":
                continue
            fres = [float(x) for x in row["family_K_re"].split(";") if x]
            fims = [float(x) for x in row["family_K_im"].split(";") if x]
            fess = [float(x) for x in row["family_ESS"].split(";") if x]
            k0 = cnum(row["K_trivial_re"], row["K_trivial_im"])
            families = [cnum(a, b) for a, b in zip(fres, fims)]
            lat = cnum(row["K_lattice_re"], row["K_lattice_im"])
            k3 = k0 + 2 * families[0]
            k5 = k0 + 2 * sum(families[:2], 0j) if len(families) >= 2 else k3
            rows.append({
                "T": float(row["T"]),
                "ml": k3,
                "k5": k5,
                "lat": lat,
                "err": rel(k3, lat),
                "err5": rel(k5, lat),
                "trivial_ess": float(row["trivial_ESS_over_N"]),
                "families": families,
                "family_ess": fess,
                "n": int(row["n_thimbles"]),
                "status": row["status"],
            })
    rows.sort(key=lambda r: r["T"])
    return rows


def load_dense(path):
    t, re, im = [], [], []
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            tv = float(row["T"])
            if 1.999 <= tv <= 4.001:
                t.append(tv)
                re.append(float(row["lattice_K_re"]))
                im.append(float(row["lattice_K_im"]))
    return np.asarray(t), np.asarray(re), np.asarray(im)


def style_axes(ax, ylabel):
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.set_xlim(1.9, 4.1)


def finish(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def on_scale(z, limit=8.0):
    return np.isfinite(z.real) and np.isfinite(z.imag) and abs(z) <= limit


def ml_line(rows, limit=8.0):
    """Connected ML line, broken where a point leaves the axis."""
    ts, re, im, skipped = [], [], [], []
    for row in rows:
        z = row["ml"]
        if not on_scale(z, limit):
            skipped.append(row["T"])
            if ts and np.isfinite(ts[-1]):
                ts.append(np.nan)
                re.append(np.nan)
                im.append(np.nan)
            continue
        ts.append(row["T"])
        re.append(z.real)
        im.append(z.imag)
    return ts, re, im, skipped


def plot_compare(path, title, lattice, rows, extras=None, note=None):
    fig, axes = plt.subplots(2, 1, figsize=(10.2, 8.8), sharex=True)
    lt, lre, lim = lattice
    axes[0].plot(lt, lre, color=NAVY, lw=2.2, label="Lattice", zorder=2)
    axes[1].plot(lt, lim, color=NAVY, lw=2.2, zorder=2)
    ts, re, im, skipped = ml_line(rows)
    axes[0].plot(ts, re, color=AMBER, lw=2.0, marker="o", ms=7,
                 label="Three saddles", zorder=3)
    axes[1].plot(ts, im, color=AMBER, lw=2.0, marker="o", ms=7, zorder=3)
    for item in extras or []:
        kw = dict(color=item["color"], marker=item["marker"], ms=10, ls="none", zorder=4)
        if item.get("label"):
            kw["label"] = item["label"]
        axes[0].plot([item["T"]], [item["k"].real], **kw)
        axes[1].plot([item["T"]], [item["k"].imag], **{k: v for k, v in kw.items() if k != "label"})
    style_axes(axes[0], r"Re $K$")
    style_axes(axes[1], r"Im $K$")
    axes[0].set_title(title)
    axes[1].set_xlabel("T")
    axes[0].legend(loc="best", frameon=False)
    if skipped and note is None:
        listed = ", ".join(f"{t:.1f}" for t in skipped)
        note = f"Off the scale, not drawn: T = {listed}."
    if note:
        fig.text(0.01, 0.005, note, fontsize=9, color="#444444")
        fig.tight_layout(rect=(0, 0.035, 1, 1))
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
    else:
        finish(fig, path)
    return skipped


def add_p(doc, text, *, bold=False, italic=False, size=11, space_after=8):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.bold = bold
    run.italic = italic
    run.font.size = Pt(size)
    run.font.color.rgb = RGBColor(0x1F, 0x29, 0x37)
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.space_before = Pt(0)
    return p


def add_pic(doc, path, caption):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(str(path), width=Inches(6.3))
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = cap.add_run(caption)
    run.italic = True
    run.font.size = Pt(9)
    run.font.color.rgb = RGBColor(0x47, 0x55, 0x69)
    cap.paragraph_format.space_after = Pt(12)


def add_table(doc, headers, data):
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    for i, header in enumerate(headers):
        cell = table.rows[0].cells[i]
        cell.text = ""
        run = cell.paragraphs[0].add_run(header)
        run.bold = True
        run.font.size = Pt(9)
    for row in data:
        cells = table.add_row().cells
        for i, value in enumerate(row):
            cells[i].text = ""
            run = cells[i].paragraphs[0].add_run(value)
            run.font.size = Pt(9)
    doc.add_paragraph().paragraph_format.space_after = Pt(6)


def at(rows, t):
    return next(r for r in rows if abs(r["T"] - t) < 1e-9)


def lat_pts(rows):
    return (np.array([r["T"] for r in rows]),
            np.array([r["lat"].real for r in rows]),
            np.array([r["lat"].imag for r in rows]))


def five_markers(rows, times):
    markers = []
    labeled = False
    for t in times:
        row = at(rows, t)
        markers.append({
            "T": t,
            "k": row["k5"],
            "color": GREEN,
            "marker": "D",
            "label": "Five saddles" if not labeled else None,
        })
        labeled = True
    return markers


def lam01_three(scan_rows):
    """Three saddles. After the caustic the pair comes from the later search."""
    late = {}
    with LAM01_5.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            t = float(row["T"])
            if t < 3.15:
                continue
            k0 = cnum(row["K_trivial_re"], row["K_trivial_im"])
            b1 = cnum(row["K_bounce_re"], row["K_bounce_im"])
            b2 = cnum(row["K_bounce2_re"], row["K_bounce2_im"])
            bounce = b2 if abs(b2) > abs(b1) else b1
            lat = cnum(row["K_lattice_re"], row["K_lattice_im"])
            k3 = k0 + 2 * bounce
            late[t] = {"T": t, "ml": k3, "lat": lat, "err": rel(k3, lat)}
    rows = []
    for row in scan_rows:
        if row["T"] >= 3.15 and row["T"] in late:
            rows.append(late[row["T"]])
        else:
            rows.append(row)
    return rows


def main():
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })

    s01 = load_scan(0.01)
    s10 = lam01_three(load_scan(0.1))
    u03 = load_up(0.3)
    u05 = load_up(0.5)
    big = json.loads(BIG_05.read_text(encoding="utf-8"))

    plot_compare(FIG / "lam001.png", r"$\lambda = 0.01$", load_dense(DENSE[0.01]), s01)
    plot_compare(FIG / "lam01.png", r"$\lambda = 0.1$", load_dense(DENSE[0.1]), s10)
    plot_compare(
        FIG / "lam03.png", r"$\lambda = 0.3$", lat_pts(u03), u03,
        extras=five_markers(u03, [3.6, 3.8]),
    )
    plot_compare(
        FIG / "lam05.png", r"$\lambda = 0.5$", lat_pts(u05), u05,
        extras=five_markers(u05, [2.0]) + [{
            "T": 3.6,
            "k": cnum(*big["K5"]),
            "color": GREEN,
            "marker": "D",
            "label": "Five saddles, larger network",
        }],
    )

    doc = Document()
    for section in doc.sections:
        section.top_margin = Inches(0.75)
        section.bottom_margin = Inches(0.75)
        section.left_margin = Inches(0.85)
        section.right_margin = Inches(0.85)

    doc.add_heading("Three saddles against the lattice", level=0)
    add_p(
        doc,
        "N = 16, η = 0.05, T = 2 to 4. The amber line is three saddles: the one centered at zero, plus one opposite pair. The navy line is the lattice. A green marker is five saddles, drawn only where that extra pair improves the result.",
        size=12,
    )
    add_p(doc, "Three saddles track the lattice away from the caustics. Around T ≈ 3 they leave the curve, or the samples diverge and drop off the scale.")

    doc.add_heading("λ = 0.01. On the lattice except at the T = 3 caustic.", level=1)
    add_pic(doc, FIG / "lam001.png", "Three saddles. The miss is T = 3.0.")
    add_p(doc, f"T = 3.0 is {pct(at(s01, 3.0)['err'])} off. Every other time is within {pct(max(r['err'] for r in s01 if abs(r['T'] - 3.0) > 1e-9))}. From T = 3.2 only the zero-centered saddle is present, and it stays on the lattice.")

    doc.add_heading("λ = 0.1. On the lattice except at the caustic.", level=1)
    add_pic(doc, FIG / "lam01.png", "Three saddles. T = 3.2–4.0 include the pair found in the later search.")
    late = [r for r in s10 if r["T"] >= 3.2]
    add_p(doc, f"T = 2.0–2.6 stay within {pct(max(r['err'] for r in s10 if r['T'] <= 2.6))}, and T = 3.2–4.0 stay within {pct(max(r['err'] for r in late))}. T = 2.8 is {pct(at(s10, 2.8)['err'])} off, and T = 3.0 leaves the scale. A fifth saddle leaves these results unchanged.")

    doc.add_heading("λ = 0.3. Three saddles work at the early times and at T = 3.4. The caustic does not.", level=1)
    add_pic(doc, FIG / "lam03.png", "Green markers: five saddles at T = 3.6 and T = 3.8.")
    add_p(doc, f"T = 2.0 and 2.2 are within {pct(max(at(u03, t)['err'] for t in (2.0, 2.2)))}, and T = 3.4 is {pct(at(u03, 3.4)['err'])}. T = 2.4 is {pct(at(u03, 2.4)['err'])} off, T = 2.6 is {pct(at(u03, 2.6)['err'])} off, and T = 3.2 and T = 4 are far off the curve. T = 2.8 and 3.0 leave the scale. Five saddles bring T = 3.6 from {pct(at(u03, 3.6)['err'])} to {pct(at(u03, 3.6)['err5'])}, and T = 3.8 from {pct(at(u03, 3.8)['err'])} to {pct(at(u03, 3.8)['err5'])}.")

    doc.add_heading("λ = 0.5. Three saddles stay off the lattice. Five saddles land at T = 2.0.", level=1)
    add_pic(doc, FIG / "lam05.png", "Green markers: five saddles at T = 2.0, and five saddles with a larger network at T = 3.6.")
    add_p(doc, f"Three saddles are {pct(at(u05, 2.0)['err'])} off at T = 2.0; five saddles bring that to {pct(at(u05, 2.0)['err5'])}. Elsewhere three saddles miss or leave the scale: T = 2.2 is {pct(at(u05, 2.2)['err'])}, T = 2.4 is {pct(at(u05, 2.4)['err'])}, T = 3.4 is {pct(at(u05, 3.4)['err'])}, T = 3.6 is {pct(at(u05, 3.6)['err'])} and T = 4.0 is {pct(at(u05, 4.0)['err'])}. Five saddles move T = 3.4 to {pct(at(u05, 3.4)['err5'])} and T = 3.6 to {pct(at(u05, 3.6)['err5'])}. A larger network at T = 3.6 reaches {pct(big['err5'])}.")

    targets = [
        OUT_DOC,
        OUT_DOC.with_name("ML_vs_lattice_lambda_summary_concise.docx"),
        OUT_DOC.with_name("ML_vs_lattice_three_saddles.docx"),
    ]
    for target in targets:
        try:
            doc.save(target)
            print(f"Wrote {target}")
            break
        except PermissionError:
            continue
    else:
        raise PermissionError("All report filenames are open")


if __name__ == "__main__":
    main()


