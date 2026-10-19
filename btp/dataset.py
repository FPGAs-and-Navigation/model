"""Samples of one UAV "moment": aerial image + magnetic profile + gravity profile -> position.

Training samples are generated on the fly at random continuous positions, with a synthetic
straight-ish track sampled from the training-line grids. Validation/test samples use the
real raw readings of held-out flight lines (see prepare_geophysics.py).

Each sample is a dict:
  img    (3, S, S) float, ImageNet-normalised RGB, north-up
  mag    (4, L)  [RMI, height, cos(heading), sin(heading)], normalised
  grav   (5, L)  [gD, GDD, height, cos, sin], normalised
  avail  (3,) bool  which of vision/mag/grav are present (modality dropout)
  target (2,) normalised (E, N) in [-1, 1];  en (2,) metres
"""
import json
import math

import numpy as np
import torch
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as TF
from torch.utils.data import Dataset

from . import config
from .geo import Extent, bilinear

IMNET_MEAN, IMNET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


class MapData:
    """Everything shared by the datasets; arrays are opened lazily so each worker memmaps its own."""

    def __init__(self, cfg, years):
        """`years` are the imagery years training crops are drawn from."""
        self.cfg = cfg
        self.dir = config.path(cfg, "data_dir")
        self.ext = Extent.load(self.dir / "extent.json")
        with open(self.dir / "stats.json") as f:
            self.stats = json.load(f)
        self.years = list(years)
        g = np.load(self.dir / "geo_grids.npz")
        self.grids = {k: g[k] for k in g.files}
        # Positions must be usable in every prepared year, so train/val/test share one footprint.
        valid = self.grids["mask_mag"] & self.grids["mask_grav"]
        for f in sorted(self.dir.glob("img_valid_*.npy")):
            valid &= np.load(f)
        self.valid = valid
        self._img = {}

    def image(self, year):
        if year not in self._img:
            self._img[year] = np.load(self.dir / f"naip_{year}.npy", mmap_mode="r")
        return self._img[year]

    def ref_color_stats(self):
        """Per-channel mean/std of the onboard reference map (dsmac.ref_year), from a sparse sample."""
        if not hasattr(self, "_ref_stats"):
            ref = self.image(self.cfg["dsmac"]["ref_year"])[::50, ::50, :3].reshape(-1, 3).astype(np.float32)
            ref = ref[ref.max(1) > 0]
            self._ref_stats = ref.mean(0), ref.std(0)
        return self._ref_stats

    def align_color(self, rgb):
        """Instance colour alignment: give the image the reference map's per-channel mean/std.

        Uses only the image itself plus fixed map statistics, so it is available onboard; it
        removes most of the global brightness/colour shift between imagery years.
        """
        m, sd = self.ref_color_stats()
        x = rgb.astype(np.float32)
        x = (x - x.mean((0, 1))) / (x.std((0, 1)) + 1e-3) * sd + m
        return np.clip(x, 0, 255).astype(np.uint8)

    def norm(self, key, v):
        m, s = self.stats[key]
        return (v - m) / s


class Sampler:
    """Image crops and sensor-profile assembly shared by train and eval datasets."""

    def __init__(self, data, train):
        self.d, self.train = data, train
        c = data.cfg["img"]
        self.res, self.size, self.win = c["res"], c["size"], c["window"]
        self.color = T.Compose([
            T.RandomApply([T.ColorJitter(0.35, 0.35, 0.25, 0.03)], p=0.8),
            T.RandomApply([T.GaussianBlur(5, (0.1, 1.5))], p=0.3),
            T.RandomGrayscale(p=0.05),
        ])

    def crop(self, year, e, n, rng):
        img = self.d.image(year)
        r, c = self.d.ext.to_pix(e, n, self.res)
        r0, c0 = int(round(r)) - self.win // 2, int(round(c)) - self.win // 2
        win = np.asarray(img[r0:r0 + self.win, c0:c0 + self.win, :3])
        if self.d.cfg["img"].get("color_align", "none") == "instance":
            win = self.d.align_color(win)
        t = torch.from_numpy(np.ascontiguousarray(win)).permute(2, 0, 1)
        side = self.size
        if self.train:
            jit = self.d.cfg["img"]
            t = TF.rotate(t, float(rng.uniform(-jit["rot_jitter"], jit["rot_jitter"])),
                          interpolation=T.InterpolationMode.BILINEAR)
            side = int(round(self.size * (1 + rng.uniform(-jit["scale_jitter"], jit["scale_jitter"]))))
        t = TF.center_crop(t, [side, side])
        if side != self.size:
            t = TF.resize(t, [self.size, self.size], antialias=True)
        if self.train:
            t = self.color(t)
        t = TF.to_dtype(t, torch.float32, scale=True)
        return TF.normalize(t, IMNET_MEAN, IMNET_STD)

    def profiles(self, mag, h_mag, gD, GDD, h_grav, heading):
        d = self.d
        L = len(mag)
        cs = np.full(L, math.cos(heading), np.float32), np.full(L, math.sin(heading), np.float32)
        m = np.stack([d.norm("mag", mag), d.norm("h", h_mag), *cs])
        g = np.stack([d.norm("gD", gD), d.norm("GDD", GDD), d.norm("h", h_grav), *cs])
        return torch.from_numpy(m.astype(np.float32)), torch.from_numpy(g.astype(np.float32))

    def avail(self, rng):
        mods = self.d.cfg["modalities"]
        a = np.array([m in mods for m in config.MODALITIES])
        if self.train:
            p = self.d.cfg["train"]["mod_dropout"]
            if isinstance(p, dict):                       # per-modality, e.g. drop vision more often
                p = np.array([p.get(m, 0.0) for m in config.MODALITIES])
            drop = a & (rng.random(3) < p)
            if (a & ~drop).any():
                a &= ~drop
        return torch.from_numpy(a)

    def target(self, e, n):
        u, v = self.d.ext.normalize(e, n)
        return (torch.tensor([u, v], dtype=torch.float32),
                torch.tensor([e, n], dtype=torch.float64))


class TrainSet(Dataset):
    def __init__(self, data):
        self.d, self.s = data, Sampler(data, train=True)
        cfg = data.cfg
        self.n = cfg["train"]["samples_per_epoch"]
        self.L, self.step = cfg["seq"]["len"], cfg["seq"]["step"]
        self.cells = np.argwhere(data.valid)
        self.res = cfg["grid_res"]

    def __len__(self):
        return self.n

    def _track(self, rng):
        """Synthetic flight history ending at a random point; returns None if it leaves the maps."""
        cfg, d = self.d.cfg, self.d
        i, j = self.cells[rng.integers(len(self.cells))]
        e = d.ext.x0 + (j + rng.random()) * self.res
        n = d.ext.y1 - (i + rng.random()) * self.res
        hdg = rng.uniform(-math.pi, math.pi)
        # Walk backwards from the current position with a slowly wandering heading.
        turns = np.cumsum(rng.normal(0, math.radians(cfg["seq"]["turn_sigma_deg"]), self.L - 1))
        steps = self.step * (1 + rng.uniform(-1, 1) * cfg["seq"]["speed_jitter"]) * np.ones(self.L - 1)
        h = hdg + turns
        de, dn = -np.sin(h) * steps, -np.cos(h) * steps
        es = e + np.concatenate([[0], np.cumsum(de)])[::-1]
        ns = n + np.concatenate([[0], np.cumsum(dn)])[::-1]
        r, c = d.ext.to_pix(es, ns, self.res)
        ri, ci = r.astype(int), c.astype(int)
        H, W = d.valid.shape
        if (ri < 0).any() or (ci < 0).any() or (ri >= H).any() or (ci >= W).any():
            return None
        g = d.grids
        if not (g["mask_mag"][ri, ci].all() and g["mask_grav"][ri, ci].all()):
            return None
        hdg_track = math.atan2(es[-1] - es[0], ns[-1] - ns[0])
        return e, n, hdg_track, {k: bilinear(g[k], r, c) for k in ("mag", "h_mag", "gD", "GDD", "h_grav")}

    def _noise(self, v, key, rng):
        nz = self.d.cfg["noise"][key]
        v = v * (1 + rng.uniform(-1, 1) * nz.get("gain", 0)) + rng.normal(0, nz.get("offset", 0))
        return v + rng.normal(0, nz["sigma"], v.shape)

    def __getitem__(self, idx):
        rng = np.random.default_rng((torch.initial_seed() + idx * 7919) % 2**63)
        t = None
        while t is None:
            t = self._track(rng)
        e, n, hdg, p = t
        mag_t, grav_t = self.s.profiles(
            self._noise(p["mag"], "mag", rng), self._noise(p["h_mag"], "height", rng),
            self._noise(p["gD"], "gD", rng), self._noise(p["GDD"], "GDD", rng),
            self._noise(p["h_grav"], "height", rng), hdg)
        year = self.d.years[rng.integers(len(self.d.years))]
        target, en = self.s.target(e, n)
        return dict(img=self.s.crop(year, e, n, rng), mag=mag_t, grav=grav_t,
                    avail=self.s.avail(rng), target=target, en=en)


class HeldOutSet(Dataset):
    """Windows of real readings from held-out lines; the image comes from `year`."""

    def __init__(self, data, split, year):
        self.d, self.s, self.year = data, Sampler(data, train=False), year
        w = np.load(data.dir / f"windows_{split}.npz")
        r, c = data.ext.to_pix(w["e"], w["n"], data.cfg["grid_res"])
        ok = data.valid[r.astype(int), c.astype(int)] if len(w["e"]) else np.zeros(0, bool)
        self.w = {k: w[k][ok] for k in w.files}

    def __len__(self):
        return len(self.w["e"])

    def __getitem__(self, i):
        w = self.w
        mag_t, grav_t = self.s.profiles(w["mag"][i], w["h_mag"][i], w["gD"][i], w["GDD"][i],
                                        w["h_grav"][i], float(w["heading"][i]))
        target, en = self.s.target(float(w["e"][i]), float(w["n"][i]))
        return dict(img=self.s.crop(self.year, float(w["e"][i]), float(w["n"][i]), None),
                    mag=mag_t, grav=grav_t, avail=self.s.avail(None), target=target, en=en,
                    line=int(w["line"][i]))


def available_years(cfg):
    d = config.path(cfg, "data_dir")
    return [y for y in cfg["train_years"] if (d / f"naip_{y}.npy").exists()]


def test_year(cfg):
    d = config.path(cfg, "data_dir")
    y = cfg["test_year"]
    if (d / f"naip_{y}.npy").exists():
        return y
    fb = available_years(cfg)[0]
    print(f"warning: test year {y} not prepared; using {fb} imagery for the test split")
    return fb
