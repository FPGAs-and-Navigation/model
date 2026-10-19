"""Build training grids and held-out sensor windows from the USGS 2016 flight-line CSVs.

The magnetic and gravity grids are interpolated from TRAINING flight lines only.
Validation/test samples are the raw readings of held-out lines, so evaluation never
sees the grids it was trained on (avoids the "inverse crime").

Outputs in data/processed/:
  extent.json        common map extent (UTM 11N)
  geo_grids.npz      mag, gD, GDD, heights and masks on the 25 m grid
  stats.json         per-channel normalisation (from training samples)
  windows_{val,test}.npz   64-step sensor profiles along held-out lines
  qc/geophysics.png  grids with the line split overlaid
"""
import argparse
import json

import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

from . import config
from .geo import Extent

GRAV_CSV = "geophysics/agg_2021/Gravity_Gradiometry_Data.csv"
MAG_CSV = "geophysics/agg_2021/Magnetic_Data.csv"
GRAV_COLS = {"Line": "line", "Easting": "e", "Northing": "n", "Height": "h",
             "gD_Fourier_0": "gD", "GDD_Fourier_0": "GDD"}
MAG_COLS = {"Line": "line", "Easting": "e", "Northing": "n", "ALT_RADAR": "h", "MAG_RMI": "mag"}
SPLIT_EVERY, VAL_IDX, TEST_IDX = 8, 3, 7
SEGMENT_GAP = 100.0      # m; a jump larger than this starts a new segment of a line
HOLDOUT_RADIUS = 60.0    # m; training samples this close to a held-out line are dropped
MASK_RADIUS = 300.0      # m; grid cells further than this from training data are invalid
GRAV_SMOOTH = 7          # 8 Hz gradiometer samples (~3.4 m apart) averaged to suppress noise


def read(raw, rel, cols):
    df = pd.read_csv(raw / rel, usecols=list(cols))
    df = df.rename(columns=cols)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna().reset_index(drop=True)
    df["line"] = df["line"].astype(int)
    df["base"] = df["line"] // 10          # 10010/10011 are reflights of line 1001
    return df


def segments(df):
    """Split a line's samples (file order = flight order) at gaps into continuous segments."""
    jump = np.hypot(np.diff(df.e.values), np.diff(df.n.values)) > SEGMENT_GAP
    seg = np.concatenate([[0], np.cumsum(jump)])
    return [g for _, g in df.groupby(seg, sort=False)]


def assign_split(grav, mag):
    """Hold out traverse lines by spatial order: of every 8 adjacent lines, one is val and one test.

    Line numbering: 1001-1883 traverse (az 70, 200 m apart), 2033-2083 infill traverse between
    them, 1901-1914 tie lines (az 20). Tie lines cross everything and stay in training.
    The split counts only the 200 m traverse lines; each infill line follows the traverse
    line just below it, so held-out ground has the same line spacing everywhere.
    """
    allb = set(grav.base) | set(mag.base)
    split = {b: "train" for b in allb}
    c = grav[grav.base < 1900].groupby("base")[["e", "n"]].mean()
    az = np.radians(70 + 90)                       # across-line direction
    across = (c.e * np.sin(az) + c.n * np.cos(az)).sort_values()
    for i, b in enumerate(across.index):
        if i % SPLIT_EVERY == VAL_IDX:
            split[b] = "val"
        elif i % SPLIT_EVERY == TEST_IDX:
            split[b] = "test"
    infill = grav[grav.base >= 2000].groupby("base")[["e", "n"]].mean()
    for b, (e, n) in infill.iterrows():
        a = e * np.sin(az) + n * np.cos(az)
        below = across[across < a]
        if len(below):
            split[b] = split[below.index[-1]]
    grav["split"] = grav.base.map(split)
    mag["split"] = mag.base.map(split)
    return split


def smooth_grav(grav):
    for c in ("gD", "GDD"):
        grav[c] = grav.groupby("line")[c].transform(
            lambda s: s.rolling(GRAV_SMOOTH, center=True, min_periods=1).mean())


def make_extent(grav, mag, res):
    e = np.concatenate([grav.e, mag.e]); n = np.concatenate([grav.n, mag.n])
    pad = 500.0
    snap = lambda v, f: float(f(v / res) * res)
    return Extent(snap(e.min() - pad, np.floor), snap(n.min() - pad, np.floor),
                  snap(e.max() + pad, np.ceil), snap(n.max() + pad, np.ceil))


def grid(df, cols, ext, res):
    H, W = ext.shape(res)
    xs = ext.x0 + (np.arange(W) + 0.5) * res
    ys = ext.y1 - (np.arange(H) + 0.5) * res
    gx, gy = np.meshgrid(xs, ys)
    pts = np.c_[df.e.values, df.n.values]
    dist, _ = cKDTree(pts).query(np.c_[gx.ravel(), gy.ravel()], distance_upper_bound=MASK_RADIUS)
    mask = np.isfinite(dist).reshape(H, W)
    out = {}
    for c in cols:
        g = griddata(pts, df[c].values, (gx, gy), method="linear")
        mask &= np.isfinite(g)
        g = np.where(np.isfinite(g), g, np.nanmean(g))
        out[c] = gaussian_filter(g, 1.0).astype(np.float32)
    return out, mask


def drop_near_holdout(train, held):
    d, _ = cKDTree(np.c_[held.e, held.n]).query(np.c_[train.e, train.n])
    return train[d > HOLDOUT_RADIUS]


def windows(grav, mag, split, L, step, stride):
    """Resample each held-out segment to `step` m and cut L-step windows ending every `stride` steps."""
    out = {k: [] for k in ("e", "n", "heading", "mag", "h_mag", "gD", "GDD", "h_grav", "line",
                                 "track_e", "track_n")}
    mag_s = mag[mag.split == split]
    for line, g in grav[grav.split == split].groupby("line", sort=False):
        m_line = mag_s[mag_s.base == line // 10]
        for seg in segments(g):
            s = np.concatenate([[0], np.cumsum(np.hypot(np.diff(seg.e.values), np.diff(seg.n.values)))])
            if s[-1] < L * step or len(m_line) < 2:
                continue
            sk = np.arange(0, s[-1], step)
            r = {c: np.interp(sk, s, seg[c].values) for c in ("e", "n", "h", "gD", "GDD")}
            # Magnetic samples (~27 m apart) are placed on the gravity track by nearest point.
            d, idx = cKDTree(np.c_[seg.e, seg.n]).query(np.c_[m_line.e, m_line.n])
            near = d < 30.0
            if near.sum() < 2:
                continue
            ms = s[idx[near]]
            order = np.argsort(ms)
            ms = ms[order]
            for c, dst in (("mag", "mag"), ("h", "h_mag")):
                v = m_line[c].values[near][order]
                r[dst] = np.interp(sk, ms, v, left=np.nan, right=np.nan)
            for k in range(L - 1, len(sk), stride):
                sl = slice(k - L + 1, k + 1)
                if np.isnan(r["mag"][sl]).any():
                    continue
                de, dn = r["e"][k] - r["e"][k - L + 1], r["n"][k] - r["n"][k - L + 1]
                out["e"].append(r["e"][k]); out["n"].append(r["n"][k])
                out["heading"].append(np.arctan2(de, dn))
                out["mag"].append(r["mag"][sl]); out["h_mag"].append(r["h_mag"][sl])
                out["gD"].append(r["gD"][sl]); out["GDD"].append(r["GDD"][sl])
                out["h_grav"].append(r["h"][sl]); out["line"].append(line)
                # Track relative to the current position (what onboard dead reckoning provides).
                out["track_e"].append(r["e"][sl] - r["e"][k]); out["track_n"].append(r["n"][sl] - r["n"][k])
    return {k: np.asarray(v, dtype=np.int32 if k == "line" else np.float32) for k, v in out.items()}


def qc_plot(fn, grids, ext, grav, mag):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(21, 8))
    ex = (ext.x0, ext.x1, ext.y0, ext.y1)
    for ax, (k, mk) in zip(axs, (("mag", "mask_mag"), ("gD", "mask_grav"), ("GDD", "mask_grav"))):
        g = np.where(grids[mk], grids[k], np.nan)
        lo, hi = np.nanpercentile(g, [2, 98])
        ax.imshow(g, extent=ex, cmap="RdBu_r", vmin=lo, vmax=hi)
        df = mag if k == "mag" else grav
        for sp, col in (("val", "lime"), ("test", "k")):
            d = df[df.split == sp]
            ax.plot(d.e.values[::5], d.n.values[::5], ",", color=col)
        ax.set_title(f"{k} (training lines only; green=val, black=test)")
    fig.tight_layout()
    fig.savefig(fn, dpi=80)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    cfg = config.load(a.config, a.set)
    raw, out = config.path(cfg, "raw_dir"), config.path(cfg, "data_dir")
    (out / "qc").mkdir(parents=True, exist_ok=True)
    res, L, step = cfg["grid_res"], cfg["seq"]["len"], cfg["seq"]["step"]

    grav, mag = read(raw, GRAV_CSV, GRAV_COLS), read(raw, MAG_CSV, MAG_COLS)
    print(f"gravity {len(grav)} samples / {grav.line.nunique()} lines, magnetic {len(mag)} / {mag.line.nunique()}")
    split = assign_split(grav, mag)
    smooth_grav(grav)
    for sp in ("train", "val", "test"):
        print(f"  {sp}: {sum(v == sp for v in split.values())} base lines")

    ext = make_extent(grav, mag, res)
    ext.save(out / "extent.json")
    print("extent", ext, "grid", ext.shape(res))

    held = pd.concat([grav[grav.split != "train"][["e", "n"]], mag[mag.split != "train"][["e", "n"]]])
    g_tr = drop_near_holdout(grav[grav.split == "train"], held)
    m_tr = drop_near_holdout(mag[mag.split == "train"], held)
    print(f"training samples after holdout buffer: gravity {len(g_tr)}, magnetic {len(m_tr)}")

    gg, mask_grav = grid(g_tr, ("gD", "GDD", "h"), ext, res)
    mg, mask_mag = grid(m_tr, ("mag", "h"), ext, res)
    grids = dict(mag=mg["mag"], h_mag=mg["h"], gD=gg["gD"], GDD=gg["GDD"], h_grav=gg["h"],
                 mask_mag=mask_mag, mask_grav=mask_grav)
    np.savez_compressed(out / "geo_grids.npz", **grids)
    print(f"valid area: magnetic {mask_mag.sum() * res**2 / 1e6:.0f} km2, gravity {mask_grav.sum() * res**2 / 1e6:.0f} km2")

    stats = {c: [float(df[c].mean()), float(df[c].std())]
             for df, cs in ((m_tr, ("mag",)), (g_tr, ("gD", "GDD"))) for c in cs}
    stats["h"] = [float(g_tr.h.mean()), float(g_tr.h.std())]
    with open(out / "stats.json", "w") as f:
        json.dump(stats, f, indent=1)

    for sp, stride in (("val", cfg["train"]["val_stride"]), ("test", 2)):
        w = windows(grav, mag, sp, L, step, stride)
        np.savez_compressed(out / f"windows_{sp}.npz", **w)
        print(f"{sp}: {len(w['e'])} windows from {len(np.unique(w['line']))} lines")

    qc_plot(out / "qc" / "geophysics.png", grids, ext, grav, mag)


if __name__ == "__main__":
    main()
