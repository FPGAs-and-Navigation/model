"""Learned DSMAC: a tiny CNN turns the UAV image and the reference map into robust feature maps;
the query's features are correlated over the map's features within the search window.

Classic DSMAC correlates raw pixels, which breaks when the scene changes between imagery years.
Here the features are trained on perfectly aligned cross-year pairs (2016/2018 queries with
simulated heading/altitude/tilt errors vs the 2020 map), so the correlation peak survives those
changes. The network is conv/BN/ReLU only (INT8- and FPGA-friendly), and the map side can be
precomputed offline.

  python -m btp.edge.ldsmac train [--steps 20000]       # on kratos
  python -m btp.matchers --methods ldsmac,loftr ...     # benchmark (see matchers.py)
"""
import argparse
import math
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .. import config

STRIDE = 4          # feature-map stride in pixels (5 m at 1.25 m/px)


def conv(cin, cout, s=1):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, s, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class FeatNet(nn.Module):
    """3 x H x W image -> C x H/4 x W/4 unit-norm features."""

    def __init__(self, width=(16, 32, 64), dim=32):
        super().__init__()
        a, b, c = width
        self.body = nn.Sequential(conv(3, a), conv(a, a), conv(a, b, 2), conv(b, b), conv(b, c, 2), conv(c, c),
                                  conv(c, c), nn.Conv2d(c, dim, 1))

    def forward(self, x):
        return F.normalize(self.body(x), dim=1)


def prep(rgb):
    """uint8 H x W x 3 -> 1 x 3 x H x W float, per-image standardised (removes global colour shifts)."""
    x = rgb.astype(np.float32)
    x = (x - x.mean((0, 1))) / (x.std((0, 1)) + 1e-3)
    return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))[None]


def correlate(fq, fr):
    """Score of every template position: (B,C,h,w) over (B,C,H,W) -> (B, H-h+1, W-w+1), in [-1, 1]."""
    B, C, h, w = fq.shape
    out = F.conv2d(fr.reshape(1, B * C, *fr.shape[-2:]), fq, groups=B)
    return out[0] / (h * w)


class LDSMAC(nn.Module):
    def __init__(self, **kw):
        super().__init__()
        self.net = FeatNet(**kw)
        self.log_scale = nn.Parameter(torch.tensor(math.log(30.0)))

    def forward(self, q, r):
        return correlate(self.net(q), self.net(r)) * self.log_scale.exp()


class Matcher:
    """matchers.py interface: grayscale is not used; pass RGB (H, W, 3) uint8 images.

    Returns (x, y, confidence) with (x, y) the query centre in reference-pixel coordinates and the
    confidence the peak-to-sidelobe ratio of the correlation surface.
    """
    rgb = True

    def __init__(self, ckpt, dev="cpu"):
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        self.model = LDSMAC(**ck.get("kw", {})).to(dev).eval()
        self.model.load_state_dict(ck["model"])
        self.dev = dev

    @torch.no_grad()
    def __call__(self, q, ref):
        s = self.model(prep(q).to(self.dev), prep(ref).to(self.dev))[0].float().cpu().numpy()
        py, px = np.unravel_index(np.argmax(s), s.shape)
        conf = psr(s, px, py)
        dx, dy = subpixel(s, px, py)
        h, w = q.shape[:2]
        # Template top-left at feature cell (px, py) -> query centre in reference pixels.
        return (px + dx) * STRIDE + w / 2, (py + dy) * STRIDE + h / 2, conf


def psr(s, x, y, excl=2):
    side = np.ones(s.shape, bool)
    side[max(0, y - excl):y + excl + 1, max(0, x - excl):x + excl + 1] = False
    rest = s[side]
    return float((s[y, x] - rest.mean()) / (rest.std() + 1e-6)) if rest.size else 0.0


def subpixel(s, x, y):
    def fit(a, b, c):
        d = a - 2 * b + c
        return 0.0 if d >= 0 else float(np.clip(0.5 * (a - c) / d, -0.5, 0.5))
    H, W = s.shape
    dx = fit(s[y, x - 1], s[y, x], s[y, x + 1]) if 0 < x < W - 1 else 0.0
    dy = fit(s[y - 1, x], s[y, x], s[y + 1, x]) if 0 < y < H - 1 else 0.0
    return dx, dy


# ---------------------------------------------------------------- training
class PairSampler:
    """Cross-year pairs at random positions: (query view, reference window, true template cell)."""

    def __init__(self, cfg, rng):
        from ..dataset import MapData, available_years
        from ..matchers import simulate_view
        self.sim = simulate_view
        self.d = MapData(cfg, available_years(cfg))
        self.cfg, self.rng = cfg, rng
        ref = cfg["dsmac"]["ref_year"]
        self.qyears = [y for y in cfg["train_years"] if y != ref and (self.d.dir / f"naip_{y}.npy").exists()]
        self.ref = self.d.image(ref)
        self.cells = np.argwhere(self.d.valid)
        self.res, self.size = cfg["img"]["res"], cfg["img"]["size"]

    def window(self, img, e, n, side):
        r, c = self.d.ext.to_pix(e, n, self.res)
        r0, c0 = int(round(r)) - side // 2, int(round(c)) - side // 2
        return np.asarray(img[r0:r0 + side, c0:c0 + side, :3])

    def __call__(self, radius_px):
        rng, d = self.rng, self.cfg["dsmac"]
        i, j = self.cells[rng.integers(len(self.cells))]
        e = self.d.ext.x0 + (j + rng.random()) * self.cfg["grid_res"]
        n = self.d.ext.y1 - (i + rng.random()) * self.cfg["grid_res"]
        year = self.qyears[rng.integers(len(self.qyears))]
        q = self.sim(self.window(self.d.image(year), e, n, 320).astype(np.float32), self.size, rng,
                     d["rot_sigma"], d["scale_sigma"], d.get("tilt_deg", 0))
        # The reference window is centred on a hypothesis up to radius - 16 px from the truth.
        off = rng.uniform(-(radius_px - 16), radius_px - 16, 2) * self.res
        side = self.size + 2 * radius_px
        ref = self.window(self.ref, e + off[0], n + off[1], side)
        # True template top-left in reference pixels, then in feature cells.
        tx = (side / 2 - off[0] / self.res) - self.size / 2
        ty = (side / 2 + off[1] / self.res) - self.size / 2
        return np.clip(q, 0, 255).astype(np.uint8), ref, (tx / STRIDE, ty / STRIDE)


def soft_target(shape, tx, ty, sigma=1.0):
    H, W = shape
    yy, xx = np.mgrid[0:H, 0:W]
    t = np.exp(-((xx - tx) ** 2 + (yy - ty) ** 2) / (2 * sigma ** 2))
    return t / t.sum()


class Batches(torch.utils.data.Dataset):
    """Each item is a whole batch with one search radius; workers build their own samplers."""

    def __init__(self, cfg, steps, batch):
        self.cfg, self.steps, self.batch, self.s = cfg, steps, batch, None

    def __len__(self):
        return self.steps

    def __getitem__(self, idx):
        rng = np.random.default_rng((torch.initial_seed() + idx * 7919) % 2**63)
        if self.s is None:
            self.s = PairSampler(self.cfg, rng)
        self.s.rng = rng
        radius = int(rng.choice([48, 96, 160]))          # 60 / 120 / 200 m windows
        side = self.s.size + 2 * radius
        qs, rs, ts = [], [], []
        while len(qs) < self.batch:
            q, r, t = self.s(radius)
            if r.shape[:2] != (side, side):
                continue
            qs.append(prep(q)); rs.append(prep(r)); ts.append(t)
        n = (side - self.s.size) // STRIDE + 1
        tgt = np.stack([soft_target((n, n), *t) for t in ts])
        return torch.cat(qs), torch.cat(rs), torch.from_numpy(tgt).float(), torch.tensor(ts)


def train(a):
    cfg = config.load()
    dev = torch.device("cuda")
    torch.manual_seed(0)
    model = LDSMAC().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=a.steps, pct_start=0.05)
    out = config.path(cfg, "run_dir") / "edge"
    out.mkdir(parents=True, exist_ok=True)
    dl = torch.utils.data.DataLoader(Batches(cfg, a.steps, a.batch), None, num_workers=16, pin_memory=True)
    print(f"{sum(p.numel() for p in model.parameters()):,} parameters, reference {cfg['dsmac']['ref_year']}",
          flush=True)
    t0, run = time.time(), 0.0
    res = cfg["img"]["res"]
    for step, (q, r, tgt, ts) in enumerate(dl):
        q, r, tgt = q.to(dev, non_blocking=True), r.to(dev, non_blocking=True), tgt.to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            s = model(q, r).float()
        loss = -(tgt.flatten(1) * F.log_softmax(s.flatten(1), -1)).sum(-1).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step(); sched.step()
        run = 0.98 * run + 0.02 * loss.item() if step else loss.item()
        if step % 500 == 0 or step == a.steps - 1:
            with torch.no_grad():
                pk = s.flatten(1).argmax(-1).cpu().numpy()
                W = s.shape[-1]
                err = np.hypot(pk % W - ts[:, 0].numpy(), pk // W - ts[:, 1].numpy()) * STRIDE * res
            print(f"step {step}/{a.steps} loss {run:.3f} batch-median err {np.median(err):.1f} m "
                  f"<10m {np.mean(err < 10):.0%} ({time.time() - t0:.0f}s)", flush=True)
            torch.save(dict(model=model.state_dict(), kw={}, step=step), out / "ldsmac.pt")
    print(f"saved {out / 'ldsmac.pt'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train"])
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=32)
    a = ap.parse_args()
    train(a)


if __name__ == "__main__":
    main()
