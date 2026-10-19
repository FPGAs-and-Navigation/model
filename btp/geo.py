"""Common map frame (UTM 11N, NAD83) shared by imagery, geophysics and labels."""
import json
from dataclasses import asdict, dataclass

import numpy as np
from pyproj import Transformer

CRS = "EPSG:26911"
_to_ll = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True)


@dataclass
class Extent:
    """Axis-aligned box in metres; x0/y1 is the north-west corner."""
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def center(self):
        return (self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2

    @property
    def half(self):
        return max(self.x1 - self.x0, self.y1 - self.y0) / 2

    def shape(self, res):
        return round((self.y1 - self.y0) / res), round((self.x1 - self.x0) / res)

    def to_pix(self, e, n, res):
        """Fractional (row, col) of a point, with pixel (0, 0) covering the NW corner."""
        return (self.y1 - np.asarray(n)) / res, (np.asarray(e) - self.x0) / res

    def normalize(self, e, n):
        cx, cy = self.center
        return (np.asarray(e) - cx) / self.half, (np.asarray(n) - cy) / self.half

    def denormalize(self, u, v):
        cx, cy = self.center
        return cx + np.asarray(u) * self.half, cy + np.asarray(v) * self.half

    def save(self, fn):
        with open(fn, "w") as f:
            json.dump(asdict(self), f, indent=1)

    @classmethod
    def load(cls, fn):
        with open(fn) as f:
            return cls(**json.load(f))


def to_latlon(e, n):
    lon, lat = _to_ll.transform(np.asarray(e), np.asarray(n))
    return lat, lon


def bilinear(grid, row, col):
    """Sample grid[..., H, W] at fractional cell-centre coordinates (row, col)."""
    H, W = grid.shape[-2:]
    r = np.clip(np.asarray(row) - 0.5, 0, H - 1.001)
    c = np.clip(np.asarray(col) - 0.5, 0, W - 1.001)
    r0, c0 = np.floor(r).astype(int), np.floor(c).astype(int)
    fr, fc = r - r0, c - c0
    g00, g01 = grid[..., r0, c0], grid[..., r0, c0 + 1]
    g10, g11 = grid[..., r0 + 1, c0], grid[..., r0 + 1, c0 + 1]
    return (g00 * (1 - fr) * (1 - fc) + g01 * (1 - fr) * fc
            + g10 * fr * (1 - fc) + g11 * fr * fc)
