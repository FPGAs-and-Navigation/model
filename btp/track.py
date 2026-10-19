"""Particle-filter tracking along held-out flights: the slide-6 model + TERCOM + image matching over time.

Each held-out line segment is one flight; consecutive windows (50 m apart) are its steps.
Per step:
  - the slide-6 model sees (image, mag, grav) and returns UDML weights w_v, w_m, w_g and a
    coarse cell prior (classifier head; optionally multiplied by the retrieval prior)
  - particles move by the dead-reckoned displacement (with drift and a per-flight bias)
  - each particle is weighted by how well the measured mag/grav profiles match the maps
    along its track, scaled by w_m, w_g (`--weights udml`) or equally (`--weights equal`)
  - the image is matched against the reference map around the particle cloud; a confident
    match re-weights particles (robust Gaussian, scaled by w_v) and seeds new ones there
  - a few particles are re-drawn from the coarse prior so a lost filter can recover
  - systematic resampling when the effective sample size drops below half

  python -m btp.track runs/coarse/best.pt [--retrieval runs/retrieval/best.pt]
        [--weights udml|equal] [--no-mag] [--no-grav] [--no-image] [--mag-noise 2] [--flights 3]

Writes runs/<outdir>/track_<tag>.json (+ per-step CSV and plots of the first flights).
"""
import argparse
import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from . import config
from .c2f import CHANNELS, ImageMatcher, ProfileMatcher, log_prior
from .config import MODALITIES
from .dataset import HeldOutSet, MapData, available_years, test_year
from .evaluate import Perturbed
from .geo import cell_centers
from .models import Localizer
from .retrieval import Retriever
from .train import to_dev

MOD_OF = {"mag": "mag", "gD": "grav", "GDD": "grav"}


class Filter:
    def __init__(self, cfg, data, tercom, dsmac, centers, use, weights, rng):
        self.c, self.d, self.tercom, self.dsmac = cfg["track"], data, tercom, dsmac
        self.centers, self.use, self.weights, self.rng = centers, use, weights, rng
        self.cell = cfg["coarse"]["cell"]

    def sample_prior(self, lp, n):
        p = np.exp(lp - lp.max())
        p /= p.sum()
        idx = self.rng.choice(len(p), n, p=p)
        return self.centers[idx] + self.rng.uniform(-self.cell / 2, self.cell / 2, (n, 2))

    def geo_loglik(self, x, w, i):
        """Profile match of window i placed at every particle (P,), per channel.

        Only the part of the history inside the mapped area is scored (rescaled to the full
        length), so flights near the edge of the survey are not penalised; `ok` is whether
        the particle itself is on the map.
        """
        te, tn = w["track_e"][i], w["track_n"][i]
        vals, inside = self.tercom.sample_points(x[:, 0, None] + te[None], x[:, 1, None] + tn[None])
        n_in = inside.sum(-1)
        frac = np.where(n_in >= 8, len(te) / np.maximum(n_in, 1), 0.0)
        out = {}
        for k in CHANNELS:
            r = np.where(inside, w[k][i][None] - vals[k], 0.0)
            r -= np.where(inside, r.sum(-1, keepdims=True) / np.maximum(n_in, 1)[:, None], 0.0)
            out[k] = -0.5 * frac * (r ** 2).sum(-1) / self.tercom.sigma[k] ** 2 / self.c["geo_temper"]
        return out, inside[:, -1]

    def gains(self, udml_w):
        """Per-modality likelihood scaling: K * w_k from UDML (equal weights -> 1), clipped."""
        if self.weights == "equal":
            return {m: 1.0 for m in MODALITIES}
        return {m: float(np.clip(3 * udml_w[j], self.c["gain_min"], self.c["gain_max"]))
                for j, m in enumerate(MODALITIES)}

    def clusters(self, x, logw, n=3, sep=300.0):
        order = np.argsort(logw)[::-1]
        out = []
        for j in order[:500]:
            if all(np.hypot(*(x[j] - p)) >= sep for p in out):
                out.append(x[j])
            if len(out) == n:
                break
        return out

    def estimate(self, x, logw):
        """Weighted mean of the mode holding the most weight (not the single best particle,
        which can be a lone re-injected particle far from the cloud)."""
        wt = np.exp(logw - logw.max())
        best, best_mass = None, -1.0
        for h in self.clusters(x, logw, n=5, sep=200.0):
            m = np.hypot(*(x - h).T) < 100
            if wt[m].sum() > best_mass:
                best, best_mass = m, wt[m].sum()
        return (x[best] * wt[best, None]).sum(0) / wt[best].sum(), float(best_mass / wt.sum())

    def run(self, w, idx, lps, udml_w, q_imgs, dr_bias):
        c, N = self.c, self.c["particles"]
        x = self.sample_prior(lps[0], N)
        logw = np.zeros(N)
        # Optional augmented state: each particle carries its own guess of the dead-reckoning
        # heading/scale bias (as in marginalised-PF magnetic navigation), so the filter learns the
        # bias from the fixes and drifts less between them.
        est_bias = c.get("estimate_bias", False)
        sd_b = np.array([np.radians(c["dr_heading_bias_deg"]), c["dr_scale_bias"]])
        bias = self.rng.normal(0, 1, (N, 2)) * sd_b if est_bias else np.zeros((N, 2))
        rows = []
        for t, i in enumerate(idx):
            if t > 0:
                j = idx[t - 1]
                u = np.array([w["e"][i] - w["e"][j], w["n"][i] - w["n"][j]])
                # Dead reckoning: per-flight scale/heading bias plus random drift.
                u = dr_bias @ u
                dist = np.hypot(*u)
                if est_bias:                                  # undo each particle's bias guess
                    ca, sa = np.cos(bias[:, 0]), np.sin(bias[:, 0])
                    uc = np.stack([ca * u[0] - sa * u[1], sa * u[0] + ca * u[1]], 1) / (1 + bias[:, 1:2])
                    bias += self.rng.normal(0, 1, (N, 2)) * sd_b * 0.02
                else:
                    uc = u
                x = x + uc + self.rng.normal(0, c["motion_sigma"] + c["drift"] * dist, (N, 2))
                k = int(c["reinject"] * N)
                if k:
                    sl = self.rng.choice(N, k, replace=False)
                    x[sl] = self.sample_prior(lps[t], k)
                    if est_bias:
                        bias[sl] = self.rng.normal(0, 1, (k, 2)) * sd_b
                    logw[sl] = np.log(np.mean(np.exp(logw - logw.max()))) + logw.max()
            g = self.gains(udml_w[t])
            ll, ok = self.geo_loglik(x, w, i)
            for k_, v in ll.items():
                if self.use[MOD_OF[k_]]:
                    logw += g[MOD_OF[k_]] * v
            logw = np.where(ok, logw, logw - 10)                 # particle itself is off the map
            # Robust coarse prior: log(1 + C p) rewards cells the model likes but never punishes a
            # cell below a uniform guess, so a confidently wrong prior can't drag the cloud away.
            lp_cell = lps[t][self.tercom.cell_of(x[:, 0], x[:, 1])] + np.log(len(lps[t]))
            logw += c["prior_weight"] * np.logaddexp(0.0, lp_cell)

            matched = None
            if self.use["vision"] and q_imgs[t] is not None and t % c["image_every"] == 0:
                spread = np.sqrt(np.average(((x - x.mean(0)) ** 2).sum(1), weights=np.exp(logw - logw.max())))
                rad = float(np.clip(2.5 * spread, c["image_radius_min"], self.dsmac.c["radius"]))
                best = (0, 0, -1.0)
                for h in self.clusters(x, logw):
                    m = self.dsmac.match(q_imgs[t], *h, radius=rad)
                    if m[2] > best[2]:
                        best = m
                if best[2] >= self.dsmac.min_score:
                    matched = np.array(best[:2])
                    d2 = ((x - matched) ** 2).sum(1)
                    s2 = c["image_sigma"] ** 2
                    # Robust: a wrong confident match can't wipe out the rest of the cloud.
                    logw += g["vision"] * np.log(c["image_outlier"] + np.exp(-0.5 * d2 / s2))
                    k = int(c["image_seed"] * N)
                    order = np.argsort(logw)
                    sl = order[:k]                                     # replace the worst particles
                    x[sl] = matched + self.rng.normal(0, c["image_sigma"], (k, 2))
                    bias[sl] = bias[self.rng.choice(order[-k:], k)]    # with the best particles' bias
                    logw[sl] = logw.max()

            logw -= logw.max()
            wt = np.exp(logw)
            ess = wt.sum() ** 2 / (wt ** 2).sum()
            est, mode_frac = self.estimate(x, logw)
            b_est = float(np.degrees(np.average(bias[:, 0], weights=wt)))
            if ess < N / 2:
                ri = systematic(wt / wt.sum(), self.rng)
                x = x[ri] + self.rng.normal(0, 2.0, (N, 2))
                bias = bias[ri]
                logw = np.zeros(N)
            rows.append(dict(e=float(w["e"][i]), n=float(w["n"][i]), pe=est[0], pn=est[1], ess=ess,
                             mode_frac=mode_frac, image_fix=matched is not None,
                             bias_deg=b_est,
                             true_bias_deg=float(np.degrees(np.arctan2(dr_bias[0, 1], dr_bias[0, 0]))),
                             w_vision=udml_w[t][0], w_mag=udml_w[t][1], w_grav=udml_w[t][2]))
        return pd.DataFrame(rows)


def systematic(p, rng):
    """Indices for systematic resampling."""
    N = len(p)
    pos = (rng.random() + np.arange(N)) / N
    return np.minimum(np.searchsorted(np.cumsum(p), pos), N - 1)


def summarise(df):
    err = df.error_m.values
    conv = df[df.step >= 10].error_m.values
    fl = df.groupby("flight")
    first_fix = fl.apply(lambda g: g.step[g.error_m < 25].min() if (g.error_m < 25).any() else np.nan)
    diverged = fl.apply(lambda g: int(((g.error_m > 100).rolling(5).sum() >= 5).any()))
    s = lambda e: dict(median=float(np.median(e)), mean=float(e.mean()), p90=float(np.percentile(e, 90)),
                       **{f"within_{t}m": float((e < t).mean()) for t in (10, 25, 100)})
    return dict(all=s(err), converged_after_10_steps=s(conv), flights=int(df.flight.nunique()),
                steps=len(df), median_steps_to_first_fix=float(np.nanmedian(first_fix)),
                never_fixed=int(first_fix.isna().sum()), flights_with_divergence=int(diverged.sum()),
                image_fix_rate=float(df.image_fix.mean()),
                mean_udml_weights=dict(vision=float(df.w_vision.mean()), mag=float(df.w_mag.mean()),
                                       grav=float(df.w_grav.mean())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", help="slide-6 checkpoint (classifier head preferred)")
    ap.add_argument("--retrieval", default=None)
    ap.add_argument("--prior", default="fused", choices=["fused", "none", *MODALITIES],
                    help="'none': no slide-6 model prior (retrieval only, if given)")
    ap.add_argument("--weights", default="udml", choices=["udml", "equal"])
    ap.add_argument("--no-mag", action="store_true")
    ap.add_argument("--no-grav", action="store_true")
    ap.add_argument("--no-image", action="store_true")
    ap.add_argument("--mag-noise", type=float, default=0.0, help="extra mag noise, in signal std units")
    ap.add_argument("--split", default="test")
    ap.add_argument("--flights", type=int, default=None, help="only the first N flights")
    ap.add_argument("--plots", type=int, default=4)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--outdir", default="phase2", help="folder under runs/ for the outputs")
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    base = config.load("configs/base.yaml", a.set)
    for k in ("coarse", "tercom", "dsmac", "retrieval", "track"):
        cfg[k] = base[k]
    cfg["img"]["color_align"] = base["img"].get("color_align", "none")   # evaluation-time option
    cfg["model"]["pretrained"] = False
    cfg["train"]["workers"] = base["train"]["workers"]
    dev = torch.device("cuda")
    data = MapData(cfg, available_years(cfg))
    model = Localizer(cfg, data.ext).to(dev).eval()
    model.load_state_dict(ck["model"])
    year = test_year(cfg)
    ds = HeldOutSet(data, a.split, year)
    w = {k: v.copy() for k, v in ds.w.items()}
    rng = np.random.default_rng(0)
    if a.mag_noise:
        sd = data.stats["mag"][1] * a.mag_noise
        w["mag"] = w["mag"] + rng.normal(0, sd, w["mag"].shape).astype(np.float32)
        ds_model = Perturbed(ds, noise={"mag": a.mag_noise})
    else:
        ds_model = ds

    tercom = ProfileMatcher(data, cfg)
    tercom.calibrate(HeldOutSet(data, "val", year).w)
    dsmac = ImageMatcher(data, cfg, year)
    retriever = Retriever(a.retrieval, data, cfg) if a.retrieval else None
    centers, _ = cell_centers(data.ext, cfg["coarse"]["cell"])
    use = {"vision": not a.no_image, "mag": not a.no_mag, "grav": not a.no_grav}
    tag = a.tag or "_".join([a.prior + ("+retrieval" if retriever else ""), a.weights]
                            + [f"no{m}" for m, u in use.items() if not u]
                            + ([f"magnoise{a.mag_noise:g}"] if a.mag_noise else []))
    out_dir = config.path(cfg, "run_dir") / a.outdir
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{a.split}: imagery {year}, ref map {cfg['dsmac']['ref_year']}, matcher {cfg['dsmac']['method']}, "
          f"tag {tag}")

    flights = list(dict.fromkeys(w["seg"].tolist()))[: a.flights]
    all_rows = []
    for fi, seg in enumerate(flights):
        idx = np.flatnonzero(w["seg"] == seg)
        lps, udml_w = [], []
        with torch.no_grad():
            for batch in DataLoader(Subset(ds_model, idx), 64, num_workers=4):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = model(to_dev(batch, dev))
                lp = (log_prior(out, centers, data.ext, a.prior, cfg["coarse"]["cell"]) if a.prior != "none"
                      else np.full((len(batch["avail"]), len(centers)), -np.log(len(centers))))
                if retriever is not None:
                    lp = lp + retriever.log_prior(batch["img"])
                    lp -= np.logaddexp.reduce(lp, axis=1, keepdims=True)
                lps.append(lp)
                udml_w.append(out["w"].float().cpu().numpy())
        lps, udml_w = np.concatenate(lps), np.concatenate(udml_w)
        q_imgs = [dsmac.query_image(float(w["e"][i]), float(w["n"][i]), rng) for i in idx]
        b = cfg["track"]
        ang = np.radians(rng.normal(0, b["dr_heading_bias_deg"]))
        sc = 1 + rng.normal(0, b["dr_scale_bias"])
        dr_bias = sc * np.array([[np.cos(ang), np.sin(ang)], [-np.sin(ang), np.cos(ang)]])
        f = Filter(cfg, data, tercom, dsmac, centers, use, a.weights, rng)
        df = f.run(w, idx, lps, udml_w, q_imgs, dr_bias)
        df["flight"], df["step"], df["line"] = int(seg), np.arange(len(df)), w["line"][idx]
        df["error_m"] = np.hypot(df.pe - df.e, df.pn - df.n)
        all_rows.append(df)
        print(f"  flight {fi + 1}/{len(flights)} (line {int(w['line'][idx[0]])}, {len(idx)} steps): "
              f"median {df.error_m.median():.1f} m, after 10 steps {df.error_m[10:].median():.1f} m, "
              f"image fixes {df.image_fix.mean():.0%}", flush=True)
        if fi < a.plots:
            plot(df, data, cfg, year, out_dir / f"track_{tag}_flight{seg}.png")

    df = pd.concat(all_rows)
    rep = dict(tag=tag, split=a.split, imagery_year=year, ckpt=a.ckpt, retrieval=a.retrieval, **summarise(df))
    df.to_csv(out_dir / f"track_{tag}.csv", index=False)
    with open(out_dir / f"track_{tag}.json", "w") as fh:
        json.dump(rep, fh, indent=1)
    s, cv = rep["all"], rep["converged_after_10_steps"]
    print(f"{tag}: median {s['median']:.1f} m, <10m {s['within_10m']:.0%}, <25m {s['within_25m']:.0%}, "
          f"p90 {s['p90']:.0f} m | after 10 steps: median {cv['median']:.1f} m, <10m {cv['within_10m']:.0%}, "
          f"p90 {cv['p90']:.0f} m | diverged flights {rep['flights_with_divergence']}/{rep['flights']}")


def plot(df, data, cfg, year, fn):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pad = 400
    x0, x1 = df.e.min() - pad, df.e.max() + pad
    y0, y1 = df.n.min() - pad, df.n.max() + pad
    res, ext = cfg["img"]["res"], data.ext
    r0, c0 = ext.to_pix(x0, y1, res)
    r1, c1 = ext.to_pix(x1, y0, res)
    r0, c0 = max(0, int(r0)), max(0, int(c0))
    k = max(1, int((c1 - c0) / 2000))
    img = np.asarray(data.image(year)[r0:int(r1):k, c0:int(c1):k, :3])
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(14, 12), gridspec_kw=dict(height_ratios=[3, 1]))
    ax.imshow(img, extent=(ext.x0 + c0 * res, ext.x0 + c0 * res + img.shape[1] * k * res,
                           ext.y1 - r0 * res - img.shape[0] * k * res, ext.y1 - r0 * res))
    ax.plot(df.e, df.n, "-", color="lime", lw=2, label="true track")
    ax.plot(df.pe, df.pn, ".-", color="red", ms=3, lw=0.8, label="particle-filter estimate")
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
    ax.legend()
    ax.set_title(f"flight on line {int(df.line.iloc[0])}: median {df.error_m.median():.1f} m")
    ax2.semilogy(df.step, df.error_m.clip(lower=0.1), label="error (m)")
    ax2.plot(df.step[df.image_fix], df.error_m[df.image_fix].clip(lower=0.1), "g.", ms=4, label="image fix")
    ax2.set_xlabel("step (50 m)"); ax2.legend(); ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fn, dpi=80)
    plt.close(fig)


if __name__ == "__main__":
    main()
