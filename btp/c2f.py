"""Coarse-to-fine localisation on held-out test lines.

  1. coarse  - the network (UDML) gives a probability for every 250 m map cell
  2. TERCOM  - the measured magnetic/gravity profiles are matched against the training maps
               at every position in the top-k cells (and a few compass headings); the
               likelihood is combined with the coarse prior, so geophysics picks among
               the places vision thinks are plausible
  3. DSMAC   - the aerial image is matched (normalised cross-correlation) against the
               onboard reference map around the best TERCOM hypotheses

  python -m btp.c2f runs/coarse/best.pt                 # classifier prior (fused UDML)
  python -m btp.c2f runs/coarse/best.pt --prior vision  # vision-only prior, geophysics only in TERCOM
  python -m btp.c2f runs/vision/best.pt                 # a regression model works too (Gaussian prior)
  python -m btp.c2f runs/coarse/best.pt --retrieval runs/retrieval/best.pt   # classifier x retrieval

Writes c2f_<prior>_metrics.json and c2f_<prior>_predictions.csv next to the checkpoint.
"""
import argparse
import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from . import config, matchers
from .config import MODALITIES
from .dataset import HeldOutSet, MapData, available_years, test_year
from .geo import bilinear, cell_centers, to_latlon
from .models import Localizer
from .retrieval import Retriever
from .train import to_dev

CHANNELS = ("mag", "gD", "GDD")
VARIANTS = {"mag": ("mag",), "grav": ("gD", "GDD"), "mag+grav": CHANNELS}


# ---------------------------------------------------------------- coarse prior
def log_prior(out, centers_m, ext, source, cell):
    """(B, C) log-probabilities over cells from a classifier or a regression model."""
    k = None if source == "fused" else MODALITIES.index(source)
    if "logits" in out:
        logits = out["logits"] if k is None else out["aux_logits"][:, k]
        return torch.log_softmax(logits.float(), -1).cpu().numpy()
    pos = out["pos"] if k is None else out["aux_pos"][:, k]
    lv = out["pos_logvar"] if k is None else out["aux_logvar"][:, k]
    e, n = ext.denormalize(*pos.float().cpu().numpy().T)
    sigma = np.maximum(np.exp(0.5 * lv.float().cpu().numpy()) * ext.half, cell / 2)
    d2 = (centers_m[None, :, 0] - e[:, None]) ** 2 + (centers_m[None, :, 1] - n[:, None]) ** 2
    lp = -d2 / (2 * sigma[:, None] ** 2)
    return lp - np.logaddexp.reduce(lp, axis=1, keepdims=True)


# ---------------------------------------------------------------- TERCOM
class ProfileMatcher:
    def __init__(self, data, cfg):
        self.d, self.c = data, cfg["tercom"]
        self.res = cfg["grid_res"]
        self.cell = cfg["coarse"]["cell"]
        _, (self.rows, self.cols) = cell_centers(data.ext, self.cell)
        g = data.grids
        self.maps = {c: g[c] for c in CHANNELS}
        self.mask = g["mask_mag"] & g["mask_grav"]
        self.sigma = None

    def sample_points(self, e, n):
        """Map values at points and, per point, whether it is inside the mapped area."""
        r, c = self.d.ext.to_pix(e, n, self.res)
        H, W = self.mask.shape
        ri, ci = np.clip(r.astype(int), 0, H - 1), np.clip(c.astype(int), 0, W - 1)
        inside = (r >= 0) & (c >= 0) & (r < H) & (c < W) & self.mask[ri, ci]
        return {k: bilinear(m, r, c) for k, m in self.maps.items()}, inside

    def sample(self, e, n):
        """Map values at points; invalid where any point leaves the maps."""
        vals, inside = self.sample_points(e, n)
        return vals, inside.all(-1)

    def calibrate(self, w):
        """Noise level per channel: residual std between real readings and the map at the true track."""
        e, n = w["e"][:, None] + w["track_e"], w["n"][:, None] + w["track_n"]
        vals, ok = self.sample(e, n)
        self.sigma = {}
        for k in CHANNELS:
            r = w[k][ok] - vals[k][ok]
            r -= r.mean(-1, keepdims=True)
            self.sigma[k] = float(np.median(r.std(-1)))
        return self.sigma

    def candidates(self, lp):
        """Lattice positions covering the top-k cells plus a margin."""
        top = np.argsort(lp)[::-1][: self.topk]
        half = self.cell / 2 + self.c["margin"]
        off = np.arange(-half, half + 1e-6, self.c["step"])
        ext = self.d.ext
        pts = []
        for idx in top:
            r, c = divmod(int(idx), self.cols)
            ce, cn = ext.x0 + (c + 0.5) * self.cell, ext.y1 - (r + 0.5) * self.cell
            pe, pn = np.meshgrid(ce + off, cn + off)
            pts.append(np.c_[pe.ravel(), pn.ravel()])
        pts = np.unique(np.round(np.concatenate(pts) / self.c["step"]).astype(np.int64), axis=0)
        return pts * self.c["step"]

    def cell_of(self, e, n):
        ext = self.d.ext
        r = np.clip(((ext.y1 - n) // self.cell).astype(int), 0, self.rows - 1)
        c = np.clip(((e - ext.x0) // self.cell).astype(int), 0, self.cols - 1)
        return r * self.cols + c

    def match(self, w, i, lp, rng):
        """Returns per-variant (best position, top hypotheses, posterior scores)."""
        pts = self.candidates(lp)
        te, tn = w["track_e"][i], w["track_n"][i]
        dist = np.hypot(te, tn)
        drift = rng.normal(0, 1, 2) * self.c["ins_drift"]
        te, tn = te + dist * drift[0], tn + dist * drift[1]      # dead-reckoning error
        hd = np.radians(self.c["headings"])
        cs, sn = np.cos(hd)[:, None], np.sin(hd)[:, None]
        re, rn = cs * te + sn * tn, -sn * te + cs * tn            # (H, L) rotated tracks
        e = pts[:, 0, None, None] + re[None]
        n = pts[:, 1, None, None] + rn[None]
        vals, ok = self.sample(e, n)                               # (P, H, L), (P, H)
        cost = {}
        for k in CHANNELS:
            r = w[k][i] - vals[k]
            r -= r.mean(-1, keepdims=True)                         # unknown sensor offset
            cost[k] = (r ** 2).sum(-1) / self.sigma[k] ** 2
        prior = lp[self.cell_of(pts[:, 0], pts[:, 1])][:, None]
        res = {}
        for name, chans in VARIANTS.items():
            ll = -0.5 * sum(cost[k] for k in chans) / self.c["temper"]
            post = np.where(ok, ll + prior, -np.inf).max(1)        # best heading per position
            if not np.isfinite(post).any():
                post = prior[:, 0].copy()
            res[name] = (pts, post)
        return res

    def hypotheses(self, pts, post, n, min_sep=100.0):
        order = np.argsort(post)[::-1]
        out = []
        for j in order:
            if not np.isfinite(post[j]):
                break
            if all(np.hypot(*(pts[j] - p)) >= min_sep for p in out):
                out.append(pts[j])
            if len(out) == n:
                break
        return out


# ---------------------------------------------------------------- DSMAC
class ImageMatcher:
    """Matches the UAV image against the onboard reference map with a matcher from btp/matchers.py."""

    def __init__(self, data, cfg, query_year, dev="cuda"):
        self.d, self.c = data, cfg["dsmac"]
        self.res, self.size = cfg["img"]["res"], cfg["img"]["size"]
        self.query = data.image(query_year)
        self.ref = data.image(self.c["ref_year"])
        self.fn = matchers.build(self.c["method"], dev)
        self.min_score = self.c["min_score"][self.c["method"]]

    def gray(self, img, r0, c0, side):
        H, W = img.shape[:2]
        if r0 < 0 or c0 < 0 or r0 + side > H or c0 + side > W:
            return None
        return np.asarray(img[r0:r0 + side, c0:c0 + side, :3], dtype=np.float32).mean(-1)

    def query_image(self, e, n, rng):
        """The UAV's view: test-year imagery with heading, altitude and tilt errors."""
        win = 320
        r, c = self.d.ext.to_pix(e, n, self.res)
        g = self.gray(self.query, int(round(r)) - win // 2, int(round(c)) - win // 2, win)
        if g is None:
            return None
        return matchers.simulate_view(g, self.size, rng, self.c["rot_sigma"], self.c["scale_sigma"],
                                      self.c.get("tilt_deg", 0))

    def match(self, q, e, n, radius=None):
        """Best match within `radius` m of (e, n): returns (e, n, confidence)."""
        rad = int((radius or self.c["radius"]) / self.res)
        side = self.size + 2 * rad
        r, c = self.d.ext.to_pix(e, n, self.res)
        r0, c0 = int(round(r)) - side // 2, int(round(c)) - side // 2
        ref = self.gray(self.ref, r0, c0, side)
        if ref is None:
            return e, n, -1.0
        x, y, conf = self.fn(q, ref)
        ext = self.d.ext
        return ext.x0 + (c0 + x) * self.res, ext.y1 - (r0 + y) * self.res, float(conf)


# ---------------------------------------------------------------- driver
def summary(err):
    err = np.asarray(err)
    return dict(median=float(np.median(err)), mean=float(err.mean()), p90=float(np.percentile(err, 90)),
                **{f"within_{t}m": float((err < t).mean()) for t in (10, 25, 50, 100, 250)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--prior", default="fused", choices=["fused", "none", *MODALITIES])
    ap.add_argument("--retrieval", default=None, help="retrieval checkpoint; its prior multiplies the model's")
    ap.add_argument("--set", nargs="*", default=[], help="overrides for the coarse/tercom/dsmac settings")
    ap.add_argument("--limit", type=int, default=None, help="evaluate only this many windows")
    ap.add_argument("--tag", default=None, help="name for the output files (default: the prior)")
    a = ap.parse_args()
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    base = config.load("configs/base.yaml", a.set)
    for k in ("coarse", "tercom", "dsmac", "retrieval"):
        cfg[k] = base[k]
    cfg["model"]["pretrained"] = False
    dev = torch.device("cuda")
    data = MapData(cfg, available_years(cfg))
    model = Localizer(cfg, data.ext).to(dev).eval()
    model.load_state_dict(ck["model"])
    year = test_year(cfg)
    ds = HeldOutSet(data, "test", year)
    if a.limit:
        keep = np.linspace(0, len(ds) - 1, a.limit).astype(int)
        ds.w = {k: v[keep] for k, v in ds.w.items()}
    w = ds.w
    cell = cfg["coarse"]["cell"]
    centers_m, _ = cell_centers(data.ext, cell)

    tercom = ProfileMatcher(data, cfg)
    tercom.topk = cfg["coarse"]["topk"]
    sig = tercom.calibrate(HeldOutSet(data, "val", year).w)
    print(f"test windows {len(ds)}, imagery {year}, reference map {cfg['dsmac']['ref_year']}, prior {a.prior}")
    print("profile noise (map vs real readings): " + ", ".join(f"{k} {v:.2f}" for k, v in sig.items()))
    dsmac = ImageMatcher(data, cfg, year)
    retriever = Retriever(a.retrieval, data, cfg) if a.retrieval else None
    tag = a.tag or a.prior + ("+retrieval" if retriever else "")
    rng = np.random.default_rng(0)

    rows = []
    loader = DataLoader(ds, 64, num_workers=cfg["train"]["workers"], pin_memory=True)
    i0 = 0
    with torch.no_grad():
        for batch in loader:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(to_dev(batch, dev))
            lps = (log_prior(out, centers_m, data.ext, a.prior, cell) if a.prior != "none"
                   else np.zeros((len(batch["avail"]), len(centers_m))))
            if retriever is not None:
                lps = lps + retriever.log_prior(batch["img"])
            for b in range(len(lps)):
                i = i0 + b
                te, tn = float(w["e"][i]), float(w["n"][i])
                lp = lps[b]
                true_cell = tercom.cell_of(np.array([te]), np.array([tn]))[0]
                top = np.argsort(lp)[::-1]
                row = dict(e=te, n=tn, line=int(w["line"][i]),
                           topk_hit=bool(true_cell in top[: tercom.topk]),
                           coarse_e=centers_m[top[0], 0], coarse_n=centers_m[top[0], 1])
                res = tercom.match(w, i, lp, rng)
                for name, (pts, post) in res.items():
                    j = int(np.argmax(post))
                    row[f"tercom_{name}_e"], row[f"tercom_{name}_n"] = pts[j]
                pts, post = res["mag+grav"]
                hyps = tercom.hypotheses(pts, post, cfg["tercom"]["hypotheses"])
                q = dsmac.query_image(te, tn, rng)
                best = (row["tercom_mag+grav_e"], row["tercom_mag+grav_n"], -1.0)
                if q is not None:
                    for h in hyps:
                        m = dsmac.match(q, *h)
                        if m[2] > best[2]:
                            best = m
                    oe, on, osc = dsmac.match(q, te, tn)          # matcher precision with a perfect hypothesis
                    row.update(oracle_e=oe, oracle_n=on, oracle_score=osc)
                # Only a confident image match is reported as a position fix. Otherwise the output
                # is "no fix" (NaN) and the vehicle keeps flying on its IMU; `fallback_*` keeps the
                # old behaviour (report the profile-matching estimate anyway) for comparison.
                accept = best[2] >= dsmac.min_score
                row.update(dsmac_score=best[2], dsmac_accepted=accept,
                           final_e=best[0] if accept else np.nan, final_n=best[1] if accept else np.nan,
                           fallback_e=best[0] if accept else row["tercom_mag+grav_e"],
                           fallback_n=best[1] if accept else row["tercom_mag+grav_n"])
                rows.append(row)
            i0 += len(lps)
            print(f"  {i0}/{len(ds)}", end="\r", flush=True)

    df = pd.DataFrame(rows)
    err = lambda p: np.hypot(df[f"{p}_e"] - df.e, df[f"{p}_n"] - df.n)
    stages = {"coarse (top-1 cell centre)": "coarse", **{f"tercom {v}": f"tercom_{v}" for v in VARIANTS},
              "always answer (no-fix -> tercom estimate)": "fallback"}
    report = dict(prior=tag, imagery_year=year, ref_year=cfg["dsmac"]["ref_year"], n=len(df),
                  profile_sigma=sig, topk_hit_rate=float(df.topk_hit.mean()),
                  dsmac_accept_rate=float(df.dsmac_accepted.mean()))
    print(f"\ntrue cell in top-{tercom.topk}: {report['topk_hit_rate']:.1%}; "
          f"DSMAC accepted {report['dsmac_accept_rate']:.1%}")
    for label, p in stages.items():
        s = summary(err(p))
        report[label] = s
        print(f"{label:36s} median {s['median']:7.1f} m  mean {s['mean']:7.1f} m  p90 {s['p90']:7.1f} m  "
              f"<10m {s['within_10m']:.0%}  <25m {s['within_25m']:.0%}  <100m {s['within_100m']:.0%}")
    fix = df.dsmac_accepted.values
    fe = err("final")[fix]
    s = dict(fix_rate=float(fix.mean()), no_fix_rate=float(1 - fix.mean()),
             wrong_fix_rate=float((fe > 25).sum() / len(df)), **summary(fe) if fix.any() else {})
    report["final (fix / no fix)"] = s
    print(f"{'final: fixes only':36s} fix rate {s['fix_rate']:.1%}  wrong fixes (>25 m) {s['wrong_fix_rate']:.2%} "
          f"of frames | fixes: median {s.get('median', np.nan):.1f} m  <10m {s.get('within_10m', 0):.1%}  "
          f"p90 {s.get('p90', np.nan):.1f} m")
    if "oracle_e" in df:
        s = summary(err("oracle").dropna())
        report["dsmac with true hypothesis"] = s
        print(f"{'dsmac with true hypothesis':36s} median {s['median']:7.1f} m  p90 {s['p90']:7.1f} m  "
              f"<10m {s['within_10m']:.0%}")

    out_dir = a.ckpt.rsplit("/", 1)[0]
    with open(f"{out_dir}/c2f_{tag}_metrics.json", "w") as f:
        json.dump(report, f, indent=1)
    for p in ("coarse", "tercom_mag+grav", "final", "fallback"):
        df[f"{p}_lat"], df[f"{p}_lon"] = to_latlon(df[f"{p}_e"], df[f"{p}_n"])
    df["lat"], df["lon"] = to_latlon(df.e, df.n)
    df["final_error_m"] = err("final")              # NaN where there is no fix
    df.to_csv(f"{out_dir}/c2f_{tag}_predictions.csv", index=False)
    print(f"wrote {out_dir}/c2f_{tag}_metrics.json and _predictions.csv")


if __name__ == "__main__":
    main()
