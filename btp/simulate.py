"""Fly a held-out test line: feed each (aerial shot, sensor history) in order, output lat/lon.

  python -m btp.simulate runs/full/best.pt [--line 10850]

Writes sim_line<id>.csv (true and predicted lat/lon per step) and sim_line<id>.png
(trajectory over the imagery) next to the checkpoint.
"""
import argparse

import matplotlib
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .dataset import HeldOutSet, MapData, available_years, test_year  # noqa: E402
from .geo import to_latlon  # noqa: E402
from .models import Localizer  # noqa: E402
from .train import to_dev  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--line", type=int, default=None, help="test line id (default: longest)")
    a = ap.parse_args()
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    cfg["model"]["pretrained"] = False
    dev = torch.device("cuda")
    data = MapData(cfg, available_years(cfg))
    model = Localizer(cfg, data.ext).to(dev).eval()
    model.load_state_dict(ck["model"])
    year = test_year(cfg)
    ds = HeldOutSet(data, "test", year)
    lines = ds.w["line"]
    line = a.line or int(pd.Series(lines).value_counts().index[0])
    idx = np.flatnonzero(lines == line)
    print(f"flying test line {line}: {len(idx)} steps, imagery {year}")

    rows = []
    with torch.no_grad():
        for s in range(0, len(idx), 64):
            items = [ds[i] for i in idx[s:s + 64]]
            batch = {k: torch.stack([torch.as_tensor(it[k]) for it in items]) for k in items[0]}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(to_dev(batch, dev))
            e, n = data.ext.denormalize(*out["pos"].float().cpu().numpy().T)
            sd = np.exp(0.5 * out["pos_logvar"].float().cpu().numpy()) * data.ext.half
            w = out["w"].float().cpu().numpy()
            for j, it in enumerate(items):
                rows.append(dict(e=it["en"][0].item(), n=it["en"][1].item(), pe=e[j], pn=n[j], sigma_m=sd[j],
                                 w_vision=w[j, 0], w_mag=w[j, 1], w_grav=w[j, 2]))
    df = pd.DataFrame(rows)
    df["lat"], df["lon"] = to_latlon(df.e, df.n)
    df["pred_lat"], df["pred_lon"] = to_latlon(df.pe, df.pn)
    df["error_m"] = np.hypot(df.pe - df.e, df.pn - df.n)
    out = a.ckpt.rsplit("/", 1)[0]
    df.to_csv(f"{out}/sim_line{line}.csv", index=False)
    print(f"median error {df.error_m.median():.1f} m, mean {df.error_m.mean():.1f} m")

    pad = 600
    x0, x1 = min(df.e.min(), df.pe.min()) - pad, max(df.e.max(), df.pe.max()) + pad
    y0, y1 = min(df.n.min(), df.pn.min()) - pad, max(df.n.max(), df.pn.max()) + pad
    res, ext = cfg["img"]["res"], data.ext
    r0, c0 = ext.to_pix(x0, y1, res)
    r1, c1 = ext.to_pix(x1, y0, res)
    r0, c0 = max(0, int(r0)), max(0, int(c0))
    k = max(1, int((c1 - c0) / 2000))
    img = np.asarray(data.image(year)[r0:int(r1):k, c0:int(c1):k, :3])
    fig, ax = plt.subplots(figsize=(14, 14 * img.shape[0] / max(1, img.shape[1])))
    ax.imshow(img, extent=(ext.x0 + c0 * res, ext.x0 + c0 * res + img.shape[1] * k * res,
                           ext.y1 - r0 * res - img.shape[0] * k * res, ext.y1 - r0 * res))
    ax.plot(df.e, df.n, "-", color="lime", lw=2, label="true track")
    ax.plot(df.pe, df.pn, ".", color="red", ms=5, label="predicted")
    for _, r in df.iloc[::5].iterrows():
        ax.plot([r.e, r.pe], [r.n, r.pn], "-", color="yellow", lw=0.5)
    ax.set_title(f"test line {line}, imagery {year}: median error {df.error_m.median():.0f} m")
    ax.legend()
    fig.savefig(f"{out}/sim_line{line}.png", dpi=90, bbox_inches="tight")
    print(f"wrote {out}/sim_line{line}.csv and .png")


if __name__ == "__main__":
    main()
