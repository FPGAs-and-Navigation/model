"""Collect the phase-1 and phase-2 results into runs/phase2/report.md (+ report.json).

  python -m btp.phase2_report            # phase 2 report
  python -m btp.phase2_report --phase3   # phase 3 report (runs/phase3/report.md)
"""
import argparse
import json

from . import config

STAGES = ("vision", "vision+mag", "vision+mag+grav", "coarse")


def load(p):
    try:
        with open(p) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def pct(x):
    return "–" if x is None else f"{100 * x:.0f}%"


def m(x):
    return "–" if x is None else (f"{x:.1f} m" if x < 100 else f"{x:.0f} m")


def track_rows(out, rows):
    lines = ["| configuration | median | within 10 m | within 25 m | p90 | after 10 steps: within 10 m | "
             "flights diverged | UDML weights (v / m / g) |", "|---|---|---|---|---|---|---|---|"]
    data = {}
    for tag, name in rows:
        r = load(out / f"track_{tag}.json")
        if not r:
            continue
        a, c = r["all"], r["converged_after_10_steps"]
        w = r["mean_udml_weights"]
        lines.append(f"| {name} | {m(a['median'])} | {pct(a['within_10m'])} | {pct(a['within_25m'])} | {m(a['p90'])} | "
                     f"{pct(c['within_10m'])} | {r['flights_with_divergence']}/{r['flights']} | "
                     f"{w['vision']:.2f} / {w['mag']:.2f} / {w['grav']:.2f} |")
        data[f"track_{tag}"] = r
    return lines, data


def phase3(cfg):
    runs = config.path(cfg, "run_dir")
    out = runs / "phase3"
    out.mkdir(parents=True, exist_ok=True)
    lines, data = ["# Phase 3 results: literature-driven fixes", "",
                   "Test flight lines, 2022 imagery, realistic camera views, unless noted.", ""]

    lines += ["## 1. Does balanced training (OGM + vision dropout) fix UDML?", "",
              "| model | fused median | vision / mag / grav alone | weights clean | weights, noisy vision | "
              "weights, noisy mag | weights, noisy grav |", "|---|---|---|---|---|---|---|"]
    fmt = lambda w: f"{w['vision']:.2f} / {w['mag']:.2f} / {w['grav']:.2f}"
    for name in ("coarse", "coarse_balanced"):
        r = load(runs / name / "test_metrics.json")
        if not r:
            continue
        a, nz = r["all"], r.get("noisy", {})
        alone = " / ".join(m(a[k]["median"]) for k in ("vision", "mag", "grav") if k in a)
        lines.append(f"| {name} | {m(a['fused']['median'])} | {alone} | {fmt(a['weights'])} | "
                     + " | ".join(fmt(nz[k]["weights"]) if k in nz else "–" for k in ("vision", "mag", "grav")) + " |")
        data[f"udml_{name}"] = r
    lines += ["", "A working UDML should lower a modality's weight when that modality is made noisy.", ""]

    lines += ["## 2. Single-shot pipeline (fix / no fix)", "",
              "| configuration | fix rate | wrong fixes (% of frames) | median of fixes | right area in top-k |",
              "|---|---|---|---|---|"]
    for run_dir, tag, name in (("coarse", "p3_base", "classifier×retrieval (phase-2 setup)"),
                               ("coarse", "p3_seq5", "+ sequence prior (5 frames)"),
                               ("coarse", "p3_color", "+ colour alignment"),
                               ("coarse", "p3_top30", "+ top-30 candidate cells"),
                               ("coarse", "p3_all", "+ all three"),
                               ("coarse_balanced", "p3_all", "balanced classifier + all three")):
        r = load(runs / run_dir / f"c2f_{tag}_metrics.json")
        if r and "final (fix / no fix)" in r:
            f = r["final (fix / no fix)"]
            lines.append(f"| {name} | {pct(f['fix_rate'])} | {100 * f['wrong_fix_rate']:.2f}% | "
                         f"{m(f.get('median'))} | {pct(r['topk_hit_rate'])} |")
            data[f"c2f_{run_dir}_{tag}"] = r

    lines += ["", "## 3. Particle filter", ""]
    tl, td = track_rows(out, (
        ("base", "phase-2 filter (classifier×retrieval, LoFTR, UDML weights)"),
        ("bias", "+ dead-reckoning bias estimation"),
        ("sparse", "image fix only every 4th frame, 3° DR heading bias"),
        ("sparse_bias", "same, + bias estimation"),
        ("balanced", "balanced classifier + bias estimation"),
        ("balanced_equal", "balanced classifier, equal weights"),
        ("balanced_magnoise_udml", "balanced, noisy magnetometer (2σ), UDML weights"),
        ("balanced_magnoise_equal", "balanced, noisy magnetometer (2σ), equal weights"),
        ("best", "**best: balanced + bias estimation + colour alignment**")))
    lines += tl
    data.update(td)
    (out / "report.md").write_text("\n".join(lines) + "\n")
    with open(out / "report.json", "w") as f:
        json.dump(data, f, indent=1)
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase3", action="store_true")
    a = ap.parse_args()
    cfg = config.load()
    if a.phase3:
        return phase3(cfg)
    runs = config.path(cfg, "run_dir")
    out = runs / "phase2"
    lines, data = ["# Phase 2 results: slide-6 model + retrieval + particle filter + learned matching", ""], {}
    lines += ["All numbers are on held-out test flight lines with 2022 imagery (never seen in training); "
              "the onboard reference map is NAIP 2020. 'Realistic' camera views add ±8° heading, ±7% "
              "altitude and ±10° tilt; 'easy' views are nearly perfect orthophoto crops.", ""]

    lines += ["## 1. Slide-6 model alone (baseline)", "",
              "| run | head | median | p90 | within 100 m | UDML weights (v / m / g) |", "|---|---|---|---|---|---|"]
    for s in STAGES:
        r = load(runs / s / "test_metrics.json")
        if not r:
            continue
        a = r["all"]
        w = a["weights"]
        lines.append(f"| {s} | {'classifier' if s == 'coarse' else 'regression'} | {m(a['fused']['median'])} | "
                     f"{m(a['fused']['p90'])} | {pct(a['fused']['within_100m'])} | "
                     f"{w['vision']:.2f} / {w['mag']:.2f} / {w['grav']:.2f} |")
        data[f"baseline_{s}"] = a["fused"]

    lines += ["", "## 2. Image matchers at the true position (val lines)", "",
              "| views | matcher | median | within 5 m | within 10 m | ms/match |", "|---|---|---|---|---|---|"]
    for tag, name in (("tilt0.0_rot2.0", "easy"), ("tilt10.0_rot8.0", "realistic")):
        r = load(out / f"matchers_{tag}.json")
        for k in ("ncc", "loftr", "lightglue"):
            if r and k in r:
                v = r[k]
                lines.append(f"| {name} | {k} | {m(v['median'])} | {pct(v['within_5m'])} | {pct(v['within_10m'])} | "
                             f"{v['ms_per_match']:.0f} |")
        data[f"matchers_{name}"] = r

    lines += ["", "## 3. Coarse localisation: is the right place among the candidates?", "",
              "| method | R@1 | R@10 | (within 250 m) |", "|---|---|---|---|"]
    for tag, name in (("zeroshot_vit_small_patch14_dinov2", "retrieval, zero-shot DINOv2 ViT-S"),
                      ("zeroshot_vit_base_patch14_dinov2", "retrieval, zero-shot DINOv2 ViT-B"),
                      ("finetuned", "retrieval, fine-tuned slide-6 ViT")):
        r = load(out / f"retrieval_{tag}.json")
        if r:
            t = next(v for k, v in r.items() if k.startswith("test"))
            lines.append(f"| {name} | {pct(t['R@1'])} | {pct(t['R@10'])} | |")
            data[f"retrieval_{tag}"] = t
    for tag in ("fused_loftr_real", "fused+retrieval_loftr_real"):
        r = load(runs / "coarse" / f"c2f_{tag}_metrics.json")
        if r:
            lines.append(f"| classifier{' x retrieval' if 'retrieval' in tag else ''} (top-{cfg['coarse']['topk']} cells) "
                         f"| – | {pct(r['topk_hit_rate'])} | |")

    lines += ["", "## 4. Single-shot pipeline (each position independently)", "",
              "A position is reported only when the image match is confident; otherwise the output is "
              "'no fix' and the vehicle continues on its IMU.", "",
              "| configuration | fix rate | wrong fixes (>25 m, % of frames) | median of fixes | fixes within 10 m |",
              "|---|---|---|---|---|"]
    for tag, name in (("fused_ncc_real", "classifier → TERCOM → NCC (realistic)"),
                      ("fused_loftr_real", "classifier → TERCOM → LoFTR (realistic)"),
                      ("fused+retrieval_loftr_real", "classifier×retrieval → TERCOM → LoFTR (realistic)"),
                      ("fused+retrieval_loftr_easy", "classifier×retrieval → TERCOM → LoFTR (easy)")):
        r = load(runs / "coarse" / f"c2f_{tag}_metrics.json")
        if r:
            f = r.get("final (fix / no fix)")
            if f:
                lines.append(f"| {name} | {pct(f['fix_rate'])} | {100 * f['wrong_fix_rate']:.2f}% | "
                             f"{m(f.get('median'))} | {pct(f.get('within_10m'))} |")
            data[f"c2f_{tag}"] = r

    lines += ["", "## 5. Particle filter over whole flights (the combined system)", "",
              "| configuration | median | within 10 m | within 25 m | p90 | after 10 steps: within 10 m | flights diverged | "
              "UDML weights (v / m / g) |", "|---|---|---|---|---|---|---|---|"]
    rows = (("full", "**full: classifier×retrieval + TERCOM + LoFTR, UDML weights**"),
            ("full_easy", "full, easy views"), ("no_retrieval", "without retrieval"), ("ncc", "NCC instead of LoFTR"),
            ("no_image", "no image matching (geophysics only)"), ("no_mag", "no magnetometer"),
            ("no_grav", "no gravity"), ("equal_weights", "equal weights instead of UDML"),
            ("magnoise_udml", "noisy magnetometer (2σ), UDML weights"),
            ("magnoise_equal", "noisy magnetometer (2σ), equal weights"),
            ("vision_regressor", "slide-6 vision regressor as prior (instead of classifier)"))
    for tag, name in rows:
        r = load(out / f"track_{tag}.json")
        if not r:
            continue
        a, c = r["all"], r["converged_after_10_steps"]
        w = r["mean_udml_weights"]
        lines.append(f"| {name} | {m(a['median'])} | {pct(a['within_10m'])} | {pct(a['within_25m'])} | {m(a['p90'])} | "
                     f"{pct(c['within_10m'])} | {r['flights_with_divergence']}/{r['flights']} | "
                     f"{w['vision']:.2f} / {w['mag']:.2f} / {w['grav']:.2f} |")
        data[f"track_{tag}"] = r

    lines += ["", "Track plots: `runs/phase2/track_full_flight*.png`. Per-step CSVs: `runs/phase2/track_*.csv`."]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    with open(out / "report.json", "w") as f:
        json.dump(data, f, indent=1)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
