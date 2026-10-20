"""Image matchers for the DSMAC step: where in a reference-map region is the query image's centre?

Every matcher takes grayscale float32 images (0-255) and returns (x, y, confidence), with
(x, y) the query centre in reference-region pixel coordinates.
  ncc        high-pass normalised cross-correlation; confidence = peak-to-sidelobe ratio
  loftr      kornia LoFTR (outdoor) + RANSAC similarity; confidence = inlier count
  lightglue  kornia DISK + LightGlue + RANSAC similarity; confidence = inlier count

Benchmark against the true position on held-out windows (no network needed):
  python -m btp.matchers --n 300 [--set dsmac.tilt_deg=10 ...]
"""
import argparse
import json
import time

import cv2
import numpy as np
import torch

from . import config


# ---------------------------------------------------------------- NCC
def highpass(img, sigma=8.0):
    g = img.astype(np.float32)
    g = g - cv2.GaussianBlur(g, (0, 0), sigma)
    return g / (cv2.GaussianBlur(np.abs(g), (0, 0), sigma) + 1.0)


def psr(m, x, y, peak, excl=8):
    """Peak-to-sidelobe ratio: how far the peak stands out from the rest of the NCC map.

    Absolute NCC is low across years (shadows, vegetation), but a correct match still
    has a distinct peak, so this separates right from wrong matches far better.
    """
    side = np.ones(m.shape, bool)
    side[max(0, y - excl):y + excl + 1, max(0, x - excl):x + excl + 1] = False
    rest = m[side]
    return float((peak - rest.mean()) / (rest.std() + 1e-6))


def subpixel(m, x, y):
    def fit(a, b, c):
        d = a - 2 * b + c
        return 0.0 if d == 0 else 0.5 * (a - c) / d
    H, W = m.shape
    dx = fit(m[y, x - 1], m[y, x], m[y, x + 1]) if 0 < x < W - 1 else 0.0
    dy = fit(m[y - 1, x], m[y, x], m[y + 1, x]) if 0 < y < H - 1 else 0.0
    return dx, dy


class NCC:
    def __call__(self, q, ref):
        ncc = cv2.matchTemplate(highpass(ref), highpass(q), cv2.TM_CCOEFF_NORMED)
        _, peak, _, (px, py) = cv2.minMaxLoc(ncc)
        dx, dy = subpixel(ncc, px, py)
        h, w = q.shape
        return px + dx + w / 2, py + dy + h / 2, psr(ncc, px, py, peak)


# ---------------------------------------------------------------- learned
def fit_similarity(kq, kr, q_shape):
    """RANSAC similarity from query to reference keypoints; maps the query centre."""
    if len(kq) < 4:
        return 0.0, 0.0, 0.0
    M, inl = cv2.estimateAffinePartial2D(kq, kr, method=cv2.RANSAC, ransacReprojThreshold=4.0,
                                         maxIters=2000, confidence=0.999)
    if M is None:
        return 0.0, 0.0, 0.0
    h, w = q_shape
    x, y = M @ np.array([w / 2, h / 2, 1.0])
    return float(x), float(y), float(inl.sum())


def to_tensor(img, dev):
    return torch.from_numpy(img / 255.0).float()[None, None].to(dev)


class LoFTR:
    def __init__(self, dev="cuda"):
        import kornia.feature as KF
        self.dev = dev
        self.net = KF.LoFTR(pretrained="outdoor").to(dev).eval()

    @torch.no_grad()
    def __call__(self, q, ref):
        out = self.net({"image0": to_tensor(q, self.dev), "image1": to_tensor(ref, self.dev)})
        keep = out["confidence"] > 0.3
        kq = out["keypoints0"][keep].cpu().numpy()
        kr = out["keypoints1"][keep].cpu().numpy()
        return fit_similarity(kq, kr, q.shape)


class LightGlue:
    def __init__(self, dev="cuda", n_feat=2048):
        import kornia.feature as KF
        self.KF, self.dev, self.n = KF, dev, n_feat
        self.disk = KF.DISK.from_pretrained("depth").to(dev).eval()
        self.lg = KF.LightGlueMatcher("disk").to(dev).eval()

    def features(self, img):
        t = to_tensor(img, self.dev).repeat(1, 3, 1, 1)
        f = self.disk(t, self.n, pad_if_not_divisible=True)[0]
        lafs = self.KF.laf_from_center_scale_ori(f.keypoints[None],
                                                 torch.ones(1, len(f.keypoints), 1, 1, device=self.dev))
        return f.keypoints, f.descriptors, lafs

    @torch.no_grad()
    def __call__(self, q, ref):
        kq, dq, lq = self.features(q)
        kr, dr, lr = self.features(ref)
        _, idx = self.lg(dq, dr, lq, lr, hw1=torch.tensor(q.shape), hw2=torch.tensor(ref.shape))
        idx = idx.cpu().numpy()
        return fit_similarity(kq.cpu().numpy()[idx[:, 0]], kr.cpu().numpy()[idx[:, 1]], q.shape)


class XFeat:
    """XFeat (Potje et al., CVPR 2024): a CPU-friendly detector/descriptor. Unlike LoFTR, the reference
    map's features do not depend on the query, so onboard they can be precomputed offline.
    `star=True` uses its semi-dense matching (XFeat*), which is slower but denser."""

    def __init__(self, dev="cuda", top_k=1024, ref_top_k=4096, star=False):
        self.xf = torch.hub.load("verlab/accelerated_features", "XFeat", pretrained=True, top_k=top_k,
                                 trust_repo=True)
        self.xf.dev = torch.device(dev)
        self.xf.net = self.xf.net.to(dev).eval()
        self.dev, self.top_k, self.ref_top_k, self.star = dev, top_k, ref_top_k, star

    @torch.no_grad()
    def __call__(self, q, ref):
        tq, tr = to_tensor(q, self.dev), to_tensor(ref, self.dev)
        if self.star:
            kq, kr = self.xf.match_xfeat_star(tq, tr, top_k=self.ref_top_k)
        else:
            a = self.xf.detectAndCompute(tq, top_k=self.top_k)[0]
            b = self.xf.detectAndCompute(tr, top_k=self.ref_top_k)[0]
            i, j = self.xf.match(a["descriptors"], b["descriptors"], min_cossim=0.82)
            kq, kr = a["keypoints"][i].cpu().numpy(), b["keypoints"][j].cpu().numpy()
        return fit_similarity(np.asarray(kq, np.float32), np.asarray(kr, np.float32), q.shape)


def build(method, dev="cuda"):
    def ldsmac():
        from .edge.ldsmac import Matcher
        return Matcher(config.ROOT / "runs/edge/ldsmac.pt", dev)
    return {"ncc": NCC, "loftr": lambda: LoFTR(dev), "lightglue": lambda: LightGlue(dev),
            "xfeat": lambda: XFeat(dev), "xfeat_star": lambda: XFeat(dev, star=True),
            "ldsmac": ldsmac}[method]()


def call(fn, q, ref):
    """Run a matcher on RGB float images; grayscale matchers get the channel mean."""
    if getattr(fn, "rgb", False):
        return fn(np.clip(q, 0, 255).astype(np.uint8), np.clip(ref, 0, 255).astype(np.uint8))
    return fn(q.mean(-1) if q.ndim == 3 else q, ref.mean(-1) if ref.ndim == 3 else ref)


# ---------------------------------------------------------------- query simulation
def simulate_view(win, size, rng, rot_sigma, scale_sigma, tilt_deg):
    """The UAV's view from a north-up orthophoto window: heading, altitude and camera-tilt errors."""
    n = win.shape[0]
    ang = rng.normal(0, rot_sigma)
    scale = 1 + rng.normal(0, scale_sigma)
    M = np.vstack([cv2.getRotationMatrix2D((n / 2, n / 2), ang, 1 / scale), [0, 0, 1]])
    if tilt_deg > 0:
        # Homography of a camera rotated by small pitch/roll angles: K R K^-1 (focal ~ window size).
        tx, ty = np.radians(rng.uniform(-tilt_deg, tilt_deg, 2))
        Rx = np.array([[1, 0, 0], [0, np.cos(tx), -np.sin(tx)], [0, np.sin(tx), np.cos(tx)]])
        Ry = np.array([[np.cos(ty), 0, np.sin(ty)], [0, 1, 0], [-np.sin(ty), 0, np.cos(ty)]])
        K = np.array([[n, 0, n / 2], [0, n, n / 2], [0, 0, 1]])
        M = K @ Rx @ Ry @ np.linalg.inv(K) @ M
        # Keep the point below the UAV at the image centre (the IMU attitude gives that pixel).
        c = M @ np.array([n / 2, n / 2, 1.0])
        M = np.array([[1, 0, n / 2 - c[0] / c[2]], [0, 1, n / 2 - c[1] / c[2]], [0, 0, 1]]) @ M
    g = cv2.warpPerspective(win, M, (n, n), flags=cv2.INTER_LINEAR)
    o = (n - size) // 2
    return g[o:o + size, o:o + size]


# ---------------------------------------------------------------- benchmark
def main():
    from .dataset import HeldOutSet, MapData, available_years, test_year

    ap = argparse.ArgumentParser()
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--methods", default="ncc,loftr,lightglue")
    ap.add_argument("--offset", type=float, default=100.0, help="m between true position and search centre")
    ap.add_argument("--out", default="runs/phase2/matchers")
    ap.add_argument("--scale", type=float, default=1.0, help="downsample both images before matching (cheaper)")
    ap.add_argument("--radius", type=float, default=None, help="search radius in m (default dsmac.radius)")
    a = ap.parse_args()
    cfg = config.load("configs/base.yaml", a.set)
    d = cfg["dsmac"]
    data = MapData(cfg, available_years(cfg))
    year = test_year(cfg)
    w = HeldOutSet(data, a.split, year).w
    idx = np.linspace(0, len(w["e"]) - 1, a.n).astype(int)
    res, size = cfg["img"]["res"], cfg["img"]["size"]
    rad = int((a.radius or d["radius"]) / res)
    query_img, ref_img = data.image(year), data.image(d["ref_year"])
    rng = np.random.default_rng(0)

    cases = []
    for i in idx:
        e, n = float(w["e"][i]), float(w["n"][i])
        r, c = data.ext.to_pix(e, n, res)
        win = np.asarray(query_img[int(r) - 160:int(r) + 160, int(c) - 160:int(c) + 160, :3], np.float32)
        q = simulate_view(win, size, rng, d["rot_sigma"], d["scale_sigma"], d.get("tilt_deg", 0))
        # Search region centred on a hypothesis `offset` m from the truth in a random direction.
        th = rng.uniform(0, 2 * np.pi)
        he, hn = e + a.offset * np.cos(th), n + a.offset * np.sin(th)
        hr, hc = data.ext.to_pix(he, hn, res)
        side = size + 2 * rad
        r0, c0 = int(round(hr)) - side // 2, int(round(hc)) - side // 2
        ref = np.asarray(ref_img[r0:r0 + side, c0:c0 + side, :3], np.float32)
        if ref.shape[:2] != (side, side):
            continue
        cases.append((e, n, q, ref, r0, c0))

    report = dict(split=a.split, query_year=year, ref_year=d["ref_year"], n=len(cases), offset_m=a.offset,
                  rot_sigma=d["rot_sigma"], scale_sigma=d["scale_sigma"], tilt_deg=d.get("tilt_deg", 0))
    for m in a.methods.split(","):
        fn = build(m)
        t0, errs, confs = time.time(), [], []
        for e, n, q, ref, r0, c0 in cases:
            if a.scale != 1:
                qs = cv2.resize(q, None, fx=a.scale, fy=a.scale, interpolation=cv2.INTER_AREA)
                rs = cv2.resize(ref, None, fx=a.scale, fy=a.scale, interpolation=cv2.INTER_AREA)
                x, y, conf = call(fn, qs, rs)
                x, y = x / a.scale, y / a.scale
            else:
                x, y, conf = call(fn, q, ref)
            pe, pn = data.ext.x0 + (c0 + x) * res, data.ext.y1 - (r0 + y) * res
            errs.append(np.hypot(pe - e, pn - n))
            confs.append(conf)
        errs, confs = np.array(errs), np.array(confs)
        ok = errs < 15
        report[m] = dict(median=float(np.median(errs)), within_5m=float((errs < 5).mean()),
                         within_10m=float((errs < 10).mean()), within_25m=float((errs < 25).mean()),
                         ms_per_match=1000 * (time.time() - t0) / len(cases),
                         conf_correct_p10=float(np.percentile(confs[ok], 10)) if ok.any() else None,
                         conf_wrong_p90=float(np.percentile(confs[~ok], 90)) if (~ok).any() else None)
        print(f"{m:10s} median {report[m]['median']:7.1f} m  <5m {report[m]['within_5m']:.0%}  "
              f"<10m {report[m]['within_10m']:.0%}  <25m {report[m]['within_25m']:.0%}  "
              f"{report[m]['ms_per_match']:.0f} ms  conf right p10 {report[m]['conf_correct_p10']}  "
              f"wrong p90 {report[m]['conf_wrong_p90']}", flush=True)
    out = config.ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    report.update(scale=a.scale, radius_m=a.radius or d["radius"])
    tag = f"_tilt{report['tilt_deg']}_rot{report['rot_sigma']}"
    if a.scale != 1 or a.radius or a.offset != 100.0:
        tag += f"_s{a.scale}_r{a.radius or d['radius']:.0f}_o{a.offset:.0f}"
    with open(f"{out}{tag}.json", "w") as f:
        json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
