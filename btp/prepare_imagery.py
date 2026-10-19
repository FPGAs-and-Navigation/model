"""Mosaic each NAIP year onto the common extent at img.res (1.25 m) as a (H, W, 4) uint8 memmap.

Outputs in data/processed/:
  naip_<year>.npy        RGBN mosaic, 0 = no data
  img_valid_<year>.npy   bool on the 25 m grid: a full image crop centred there is inside the data
  qc/naip_<year>.jpg     thumbnail
"""
import argparse
import glob

import numpy as np
import rasterio
from PIL import Image
from rasterio.enums import Resampling
from rasterio.merge import merge
from rasterio.vrt import WarpedVRT
from scipy.ndimage import binary_erosion

from . import config
from .geo import CRS, Extent


def mosaic(files, ext, res):
    srcs = [rasterio.open(f) for f in files]
    # NAIP quarter-quads have no nodata tag but a black collar; treat 0 as nodata so
    # overlapping tiles don't paint their collars over each other.
    vrts = [WarpedVRT(s, crs=CRS, src_nodata=0, nodata=0, resampling=Resampling.average) for s in srcs]
    arr, _ = merge(vrts, bounds=(ext.x0, ext.y0, ext.x1, ext.y1), res=res, nodata=0,
                   resampling=Resampling.average)
    for v in vrts:
        v.close()
    for s in srcs:
        s.close()
    return arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--years", nargs="*", type=int)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    cfg = config.load(a.config, a.set)
    raw, out = config.path(cfg, "raw_dir"), config.path(cfg, "data_dir")
    ext = Extent.load(out / "extent.json")
    res, cell = cfg["img"]["res"], cfg["grid_res"]
    k = round(cell / res)                      # image pixels per grid cell
    H, W = ext.shape(res)
    years = a.years or [*cfg["train_years"], cfg["test_year"]]

    for y in years:
        files = sorted(glob.glob(str(raw / f"naip_{y}" / "*.tif")))
        dst = out / f"naip_{y}.npy"
        if not files:
            print(f"{y}: no tiles in {raw / f'naip_{y}'} (run scripts/sync_data.sh once it is downloaded)")
            continue
        if dst.exists() and not a.force:
            print(f"{y}: {dst.name} exists")
            continue
        print(f"{y}: mosaicking {len(files)} tiles -> {H}x{W} px at {res} m")
        arr = mosaic(files, ext, res)
        assert arr.shape[1:] == (H, W), arr.shape
        mm = np.lib.format.open_memmap(dst.with_suffix(".tmp.npy"), "w+", np.uint8, (H, W, 4))
        for b in range(4):
            mm[..., b] = arr[b]
        mm.flush()
        del mm
        dst.with_suffix(".tmp.npy").rename(dst)

        has = arr.max(0) > 0
        frac = has.reshape(H // k, k, W // k, k).mean((1, 3))
        # Erode by enough cells that a rotated, scaled crop window stays inside the data.
        r = int(np.ceil(cfg["img"]["window"] * res / 2 / cell))
        valid = binary_erosion(frac > 0.99, iterations=r)
        np.save(out / f"img_valid_{y}.npy", valid)
        print(f"{y}: {valid.sum() * cell**2 / 1e6:.0f} km2 usable crop centres")

        thumb = arr[:3, ::16, ::16].transpose(1, 2, 0)
        Image.fromarray(np.ascontiguousarray(thumb)).save(out / "qc" / f"naip_{y}.jpg", quality=85)
        del arr


if __name__ == "__main__":
    main()
