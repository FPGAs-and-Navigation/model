"""Generate the LaTeX tables and number macros of the report from the result files in runs/.

  python -m btp.latex_tables [--out report/generated]

Anything not run yet is typeset as "pending", so the report always compiles and can be rebuilt
as results arrive (scripts/build_report.sh).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

PENDING = r"\textit{pending}"
STAGES = (("vision", "Vision only"), ("vision+mag", "Vision + magnetometer"),
          ("vision+mag+grav", "Vision + mag + gravity"), ("coarse", "Cell classifier (V+M+G)"),
          ("coarse_balanced", "Balanced classifier (V+M+G)"))


def load(p):
    try:
        with open(p) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def m(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    num = f"{x:.1f}" if x < 100 else f"{x:,.0f}".replace(",", "{,}")
    return num + "\\,m"


def pct(x):
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.0f}\\%"


def w3(w):
    return f"{w['vision']:.2f} / {w['mag']:.2f} / {w['grav']:.2f}"


def table(cols, header, rows, caption, label, note=None):
    body = "\n".join(" & ".join(r) + r" \\" for r in rows) if rows else \
        r"\multicolumn{%d}{c}{%s} \\" % (cols.count("l") + cols.count("r") + cols.count("c"), PENDING)
    out = [r"\begin{table}[t]", r"\centering", r"\small", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
           r"\begin{adjustbox}{max width=\linewidth}",
           rf"\begin{{tabular}}{{{cols}}}", r"\toprule", header + r" \\", r"\midrule", body, r"\bottomrule",
           r"\end{tabular}", r"\end{adjustbox}"]
    if note:
        out += [r"\par\smallskip", rf"\begin{{minipage}}{{0.95\linewidth}}\footnotesize {note}\end{{minipage}}"]
    out.append(r"\end{table}")
    return "\n".join(out) + "\n"


def fix_stats(run_dir, tag):
    """Fix / no-fix statistics of a c2f run, from its metrics JSON or (older runs) its predictions CSV."""
    r = load(run_dir / f"c2f_{tag}_metrics.json")
    if r is None:
        return None, None
    f = r.get("final (fix / no fix)")
    if f is None:
        try:
            d = pd.read_csv(run_dir / f"c2f_{tag}_predictions.csv")
            acc = d.dsmac_accepted.astype(bool).values
            e = d.final_error_m.values[acc]
            f = dict(fix_rate=float(acc.mean()), wrong_fix_rate=float((e > 25).sum() / len(d)),
                     median=float(np.median(e)) if len(e) else None,
                     within_10m=float((e < 10).mean()) if len(e) else None,
                     p90=float(np.percentile(e, 90)) if len(e) else None)
        except (OSError, AttributeError):
            f = None
    return r, f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="report/generated")
    a = ap.parse_args()
    cfg = config.load()
    runs = config.path(cfg, "run_dir")
    out = config.ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    macros = {}

    # ---- 1. regression baselines (slide-6 model alone)
    rows = []
    for name, label in STAGES:
        r = load(runs / name / "test_metrics.json")
        v = load(runs / name / "best_val.json")
        if not r:
            continue
        al = r["all"]
        alone = [m(al[k]["median"]) if k in al else "--" for k in ("vision", "mag", "grav")]
        rows.append([label, m(v["fused"]["median"]) if v else "--", m(al["fused"]["median"]),
                     m(al["fused"]["p90"]), *alone, w3(al["weights"])])
        key = name.replace("+", "").replace("_", "")
        macros[f"Res{key}Median"] = m(al["fused"]["median"])
        if v:
            macros[f"Res{key}ValMedian"] = m(v["fused"]["median"])
    (out / "tab_baseline.tex").write_text(table(
        "lrrrrrrc", r"Model & Val.\ (2016--20) & Test (2022) & Test p90 & Vision alone & Mag alone & Grav alone & "
        r"$w_v / w_m / w_g$", rows,
        r"The slide-6 model on its own: median localisation error of the fused output on the validation lines "
        r"(training-year imagery) and the test lines (held-out 2022 imagery), the error of each modality's "
        r"auxiliary head alone, and the mean UDML weights.", "tab:baseline"))

    # ---- 2. UDML response to degraded sensors
    rows = []
    for name, label in STAGES:
        r = load(runs / name / "test_metrics.json")
        if not r or "noisy" not in r:
            continue
        nz = r["noisy"]
        rows.append([label, w3(r["all"]["weights"])]
                    + [w3(nz[k]["weights"]) if k in nz else "--" for k in ("vision", "mag", "grav")])
    (out / "tab_udml.tex").write_text(table(
        "lcccc", r"Model & Clean & Noisy image & Noisy mag & Noisy grav", rows,
        r"Mean UDML weights $w_v / w_m / w_g$ when one input is corrupted at test time (image: $\sigma=1$, "
        r"profiles: $\sigma=2$ in normalised units). A working uncertainty-aware fusion lowers the weight of "
        r"the corrupted modality.", "tab:udml"))

    # ---- 3. single-shot pipeline, stage by stage
    rows, fixrows = [], []
    c2f_runs = (("vision", "vision", "Vision regressor"), ("vision+mag", "fused", "V+M regressor"),
                ("vision+mag+grav", "fused", "V+M+G regressor"), ("coarse", "fused", "Cell classifier"),
                ("coarse", "vision", "Cell classifier, vision prior"))
    for d, tag, label in c2f_runs:
        r, f = fix_stats(runs / d, tag)
        if not r:
            continue
        g = lambda k: r.get(k, {}).get("median")
        rows.append([label, pct(r["topk_hit_rate"]), m(g("coarse (top-1 cell centre)")), m(g("tercom mag")),
                     m(g("tercom grav")), m(g("tercom mag+grav")),
                     m(r.get("dsmac with true hypothesis", {}).get("median"))])
        if f:
            fixrows.append([label, pct(f["fix_rate"]), f"{100 * f['wrong_fix_rate']:.2f}\\%", m(f.get("median")),
                            pct(f.get("within_10m")), m(f.get("p90"))])
            if d == "vision":
                macros["ResSSFixRate"] = pct(f["fix_rate"])
                macros["ResSSFixMedian"] = m(f.get("median"))
    (out / "tab_c2f_stages.tex").write_text(table(
        "lrrrrrr", r"Prior & Right cell in top-10 & Coarse & + TERCOM mag & + TERCOM grav & + TERCOM both & "
        r"LoFTR at truth", rows,
        r"Single-shot coarse-to-fine pipeline on the 3{,}063 test windows: median error after each stage, "
        r"and the image matcher's error when given a hypothesis at the true position.", "tab:c2fstages"))
    (out / "tab_c2f_fix.tex").write_text(table(
        "lrrrrr", r"Prior & Fix rate & Wrong fixes & Median of fixes & Fixes $<$10\,m & p90 of fixes", fixrows,
        r"Single-shot output with the \emph{no-fix} policy: a position is reported only when the LoFTR match "
        r"has at least 15 RANSAC inliers. Wrong fixes are reported positions more than 25\,m off, as a share "
        r"of all frames.", "tab:c2ffix"))

    # ---- 4. matchers
    rows = []
    for tag, views in (("tilt0.0_rot2.0", "easy"), ("tilt10.0_rot8.0", "realistic")):
        r = load(runs / "phase2" / f"matchers_{tag}.json")
        for k, name in (("ncc", "NCC (high-pass)"), ("loftr", "LoFTR"), ("lightglue", "DISK + LightGlue")):
            if r and k in r:
                v = r[k]
                rows.append([views, name, m(v["median"]), pct(v["within_5m"]), pct(v["within_10m"]),
                             f"{v['ms_per_match']:.0f}"])
    (out / "tab_matchers.tex").write_text(table(
        "llrrrr", r"Views & Matcher & Median & $<$5\,m & $<$10\,m & ms/match", rows,
        r"Image matchers searching $\pm$250\,m around a hypothesis 100\,m from the truth (300 validation "
        r"windows, 2022 query vs 2020 map). Realistic views add $\sigma=8^\circ$ heading, $\sigma=7\%$ scale "
        r"and up to $\pm10^\circ$ camera tilt.", "tab:matchers"))

    # ---- 5. retrieval
    rows = []
    for tag, name in (("zeroshot_vit_small_patch14_dinov2", "Zero-shot DINOv2 ViT-S/14"),
                      ("zeroshot_vit_base_patch14_dinov2", "Zero-shot DINOv2 ViT-B/14"),
                      ("finetuned", "Fine-tuned slide-6 ViT (InfoNCE)")):
        r = load(runs / "phase2" / f"retrieval_{tag}.json")
        if r:
            for k, v in r.items():
                rows.append([name, k.replace("_", " "), pct(v["R@1"]), pct(v["R@5"]), pct(v["R@10"]), pct(v["R@25"])])
    (out / "tab_retrieval.tex").write_text(table(
        "llrrrr", r"Embedder & Queries & R@1 & R@5 & R@10 & R@25", rows,
        r"Retrieval against the embedded 2020 reference map (tiles every 100\,m). A query counts as a hit "
        r"when a retrieved tile lies within 250\,m of the truth.", "tab:retrieval"))

    # ---- 6. single-shot variants (phase 2 and 3)
    rows = []
    for d, tag, name in (("coarse", "fused_ncc_real", "Classifier, NCC"),
                         ("coarse", "fused_loftr_real", "Classifier, LoFTR"),
                         ("coarse", "fused+retrieval_loftr_real", "Classifier $\\times$ retrieval, LoFTR"),
                         ("coarse", "fused+retrieval_loftr_easy", "\\quad same, easy views"),
                         ("coarse", "p3_seq5", "\\quad + sequence prior (5 frames)"),
                         ("coarse", "p3_color", "\\quad + colour alignment"),
                         ("coarse", "p3_top30", "\\quad + top-30 cells"),
                         ("coarse", "p3_all", "\\quad + all three"),
                         ("coarse_balanced", "p3_all", "Balanced classifier + all three"),
                         ("coarse", "p3_retrieval_only", "No slide-6 model: retrieval only")):
        r, f = fix_stats(runs / d, tag)
        if r and f:
            rows.append([name, pct(r["topk_hit_rate"]), pct(f["fix_rate"]), f"{100 * f['wrong_fix_rate']:.2f}\\%",
                         m(f.get("median")), pct(f.get("within_10m"))])
    (out / "tab_single_shot.tex").write_text(table(
        "lrrrrr", r"Configuration & Right cell in top-$k$ & Fix rate & Wrong fixes & Median of fixes & Fixes $<$10\,m",
        rows, r"Single-shot variants on the test windows (realistic views unless noted).", "tab:singleshot"))

    # ---- 7. particle filter
    rows = []
    track_rows = (("phase2", "full", "\\textbf{Full: classifier$\\times$retrieval + TERCOM + LoFTR, UDML}"),
                  ("phase2", "full_easy", "\\quad easy views"), ("phase2", "no_retrieval", "\\quad no retrieval"),
                  ("phase2", "ncc", "\\quad NCC instead of LoFTR"), ("phase2", "no_image", "\\quad no image matching"),
                  ("phase2", "no_mag", "\\quad no magnetometer"), ("phase2", "no_grav", "\\quad no gravity"),
                  ("phase2", "equal_weights", "\\quad equal weights instead of UDML"),
                  ("phase2", "magnoise_udml", "\\quad noisy mag, UDML"), ("phase2", "magnoise_equal", "\\quad noisy mag, equal"),
                  ("phase2", "vision_regressor", "\\quad regressor prior instead of classifier"),
                  ("phase3", "bias", "+ DR bias estimation"),
                  ("phase3", "sparse", "Fix every 4th frame, 3$^\\circ$ DR bias"),
                  ("phase3", "sparse_bias", "\\quad + DR bias estimation"),
                  ("phase3", "balanced", "Balanced classifier + bias est."),
                  ("phase3", "balanced_equal", "\\quad equal weights"),
                  ("phase3", "balanced_magnoise_udml", "\\quad noisy mag, UDML"),
                  ("phase3", "balanced_magnoise_equal", "\\quad noisy mag, equal"),
                  ("phase3", "best", "\\textbf{Best: balanced + bias est.\\ + colour alignment}"),
                  ("phase3", "retrieval_only", "No slide-6 model: retrieval prior, equal weights"))
    for ph, tag, name in track_rows:
        r = load(runs / ph / f"track_{tag}.json")
        if not r:
            continue
        al, cv = r["all"], r["converged_after_10_steps"]
        rows.append([name, m(al["median"]), pct(al["within_10m"]), pct(al["within_25m"]), m(al["p90"]),
                     pct(cv["within_10m"]), f"{r['flights_with_divergence']}/{r['flights']}"])
        if tag in ("full", "best"):
            macros[f"ResPF{tag.capitalize()}Median"] = m(al["median"])
            macros[f"ResPF{tag.capitalize()}Within"] = pct(al["within_10m"])
    (out / "tab_track.tex").write_text(table(
        "lrrrrrr", r"Configuration & Median & $<$10\,m & $<$25\,m & p90 & $<$10\,m after 10 steps & Lost flights",
        rows, r"Particle-filter tracking over the 20 held-out test flights (3{,}063 steps, 50\,m apart), "
        r"realistic views.", "tab:track"))

    # ---- macros for the text (defaults to "pending")
    defaults = ["ResvisionMedian", "ResvisionmaggravMedian", "ResvisionmaggravValMedian", "ResSSFixRate",
                "ResSSFixMedian", "ResPFFullMedian", "ResPFFullWithin", "ResPFBestMedian", "ResPFBestWithin"]
    lines = [f"\\newcommand{{\\{k}}}{{{macros.get(k, PENDING)}}}" for k in defaults]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in macros.items() if k not in defaults]
    (out / "macros.tex").write_text("\n".join(lines) + "\n")
    print(f"wrote tables and {len(macros)} macros to {out}")


if __name__ == "__main__":
    main()
