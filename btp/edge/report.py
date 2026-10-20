"""Edge tolerance table: accuracy vs hardware and flight speed.

Position outputs come from the fast loop (dead reckoning + magnetometer/gravity particle update +
networks every 10th step). Image matching runs in the background at whatever rate the hardware
allows; a fix then arrives every v * latency metres, so with 50 m evaluation steps the filter
gets a fix every ceil(v * latency / 50) steps. Accuracy at that spacing comes from
scripts/edge_sweep.sh (runs/edge/track_*.json); latencies were measured on 4 E-cores of an
i7-12700H. Raspberry Pi figures are scaled estimates, not measurements.

  python -m btp.edge.report
"""
import json
import math

from .. import config

# Measured on 4 E-cores (i7-12700H, 3.5 GHz Gracemont), seconds per image match, +-60 m search.
LATENCY_S = {"loftr_s1.0": 1.22, "loftr_s0.75": 0.68, "loftr_s0.5": 0.31, "ldsmac": 0.06}
LABEL = {"loftr_s1.0": "LoFTR, full resolution", "loftr_s0.75": "LoFTR, 0.75x resolution",
         "loftr_s0.5": "LoFTR, 0.5x resolution", "ldsmac": "Learned DSMAC (ours, 111k params)"}
# Rough single-thread-equivalent slowdown vs 4 E-cores (assumption, not measured).
HARDWARE = {"4 E-cores (measured)": 1.0, "Raspberry Pi 5 (est. 2x slower)": 2.0,
            "Raspberry Pi 4 (est. 4x slower)": 4.0}
SPEEDS = (15, 30)            # m/s
STEPS = (1, 2, 4, 8)          # fix spacings that were evaluated (x 50 m)


def load(p):
    try:
        with open(p) as f:
            return json.load(f)
    except OSError:
        return None


def main():
    runs = config.path(config.load(), "run_dir") / "edge"
    acc = {}
    for m in LATENCY_S:
        for k in STEPS:
            r = load(runs / f"track_{m}_e{k}.json")
            if r:
                acc[(m, k)] = r
    lines = ["# Edge tolerance: accuracy vs matcher, hardware and speed", "",
             "Particle filter over the 20 test flights (2022 imagery, realistic camera views), edge settings: "
             "1,000 particles, fused Numba kernel, networks every 10th step, dead-reckoning bias estimation.", "",
             "## Accuracy vs fix spacing", "",
             "| matcher | E-core latency | fix every 50 m | 100 m | 200 m | 400 m |", "|---|---|---|---|---|---|"]
    for m, lat in LATENCY_S.items():
        cells = []
        for k in STEPS:
            r = acc.get((m, k))
            cells.append(f"{r['all']['median']:.1f} m / {100 * r['all']['within_10m']:.0f}%" if r else "–")
        lines.append(f"| {LABEL[m]} | {1000 * lat:.0f} ms | " + " | ".join(cells) + " |")
    lines += ["", "Cells: median error / share of steps within 10 m.", ""]
    nr = load(runs / "track_noimage.json")
    if nr:
        lines += [f"No image matching at all (magnetometer + gravity only): median {nr['all']['median']:.1f} m, "
                  f"{100 * nr['all']['within_10m']:.0f}% within 10 m.", ""]

    lines += ["## What each hardware/speed combination gets", "",
              "| hardware | matcher | " + " | ".join(f"{v} m/s" for v in SPEEDS) + " |",
              "|---|---|" + "---|" * len(SPEEDS)]
    for hw, f in HARDWARE.items():
        for m, lat in LATENCY_S.items():
            cells = []
            for v in SPEEDS:
                k_needed = max(1, math.ceil(v * lat * f / 50))
                k = next((s for s in STEPS if s >= k_needed), None)
                r = acc.get((m, k)) if k else None
                cells.append(f"fix every {v * lat * f:.0f} m → {r['all']['median']:.1f} m / "
                             f"{100 * r['all']['within_10m']:.0f}%" if r else "–")
            lines.append(f"| {hw} | {LABEL[m]} | " + " | ".join(cells) + " |")
    (runs / "tolerance.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
