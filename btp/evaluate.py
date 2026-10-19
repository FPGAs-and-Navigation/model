"""Evaluate a checkpoint on held-out test lines with held-out-year imagery.

  python -m btp.evaluate runs/full/best.pt

Writes next to the checkpoint:
  test_metrics.json   fused + per-modality errors, modality-drop ablations, UDML weights,
                      and how the weights react when a sensor is made noisy
  test_predictions.csv  true/predicted lat, lon per window
"""
import argparse
import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .config import MODALITIES
from .dataset import HeldOutSet, MapData, available_years, test_year
from .geo import to_latlon
from .models import Localizer
from .train import evaluate


class Perturbed(torch.utils.data.Dataset):
    """Wraps a dataset to drop modalities or add extra noise (in normalised units)."""

    def __init__(self, ds, drop=(), noise=None):
        self.ds, self.drop, self.noise = ds, drop, noise or {}

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        s = dict(self.ds[i])
        a = s["avail"].clone()
        for m in self.drop:
            a[MODALITIES.index(m)] = False
        s["avail"] = a
        g = torch.Generator().manual_seed(i)
        for m, sigma in self.noise.items():
            if m == "vision":
                s["img"] = s["img"] + sigma * torch.randn(s["img"].shape, generator=g)
            else:
                x = s[m].clone()
                x[0] += sigma * torch.randn(x.shape[1], generator=g)
                s[m] = x
        return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--split", default="test")
    ap.add_argument("--year", type=int, default=None, help="imagery year (default: config test_year)")
    a = ap.parse_args()
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    cfg["model"]["pretrained"] = False
    dev = torch.device("cuda")
    data = MapData(cfg, available_years(cfg))
    model = Localizer(cfg, data.ext).to(dev)
    model.load_state_dict(ck["model"])
    year = a.year or test_year(cfg)
    ds = HeldOutSet(data, a.split, year)
    load = lambda d: DataLoader(d, 128, num_workers=cfg["train"]["workers"], pin_memory=True)
    print(f"{a.split}: {len(ds)} windows, imagery {year}, modalities {model.mods}")

    res, preds = evaluate(model, load(ds), data.ext, dev)
    report = {"split": a.split, "imagery_year": year, "epoch": ck["epoch"], "all": res}
    print(f"fused: median {res['fused']['median']:.1f} m, mean {res['fused']['mean']:.1f} m, "
          f"p90 {res['fused']['p90']:.1f} m; weights {res['weights']}")
    for m in model.mods:
        print(f"  {m} alone: median {res[m]['median']:.1f} m")

    if len(model.mods) > 1:
        report["drop"] = {}
        for m in model.mods:
            r, _ = evaluate(model, load(Perturbed(ds, drop=(m,))), data.ext, dev)
            report["drop"][m] = r
            print(f"  without {m}: median {r['fused']['median']:.1f} m, weights {r['weights']}")
        report["noisy"] = {}
        for m in model.mods:
            sigma = 1.0 if m == "vision" else 2.0          # in normalised units (std of the signal)
            r, _ = evaluate(model, load(Perturbed(ds, noise={m: sigma})), data.ext, dev)
            report["noisy"][m] = r
            print(f"  noisy {m}: median {r['fused']['median']:.1f} m, weights {r['weights']}")

    out = a.ckpt.rsplit("/", 1)[0]
    with open(f"{out}/{a.split}_metrics.json", "w") as f:
        json.dump(report, f, indent=1)
    lat, lon = to_latlon(preds[:, 0], preds[:, 1])
    plat, plon = to_latlon(preds[:, 2], preds[:, 3])
    pd.DataFrame(dict(line=ds.w["line"], lat=lat, lon=lon, pred_lat=plat, pred_lon=plon,
                      error_m=np.hypot(preds[:, 2] - preds[:, 0], preds[:, 3] - preds[:, 1]))
                 ).to_csv(f"{out}/{a.split}_predictions.csv", index=False)
    print(f"wrote {out}/{a.split}_metrics.json and {a.split}_predictions.csv")


if __name__ == "__main__":
    main()
