"""Retrieval coarse step: embed the UAV image and look it up in an embedded reference map.

The reference map (dsmac.ref_year, the "onboard" imagery) is cut into 224 px tiles on a
`retrieval.stride` lattice. The embedder is the slide-6 model's ViT backbone with a small
projection head: either zero-shot DINOv2 features (AnyLoc-style) or fine-tuned
contrastively by btp/train_retrieval.py (Sample4Geo-style).

  python -m btp.retrieval --zero-shot                       # DINOv2 ViT-S/14, no training
  python -m btp.retrieval --zero-shot --vit vit_base_patch14_dinov2.lvd142m
  python -m btp.retrieval runs/retrieval/best.pt            # fine-tuned

Reports recall@k (any of the top k tiles within `retrieval.hit_radius` m) on val and test.
"""
import argparse
import json

import numpy as np
import timm
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from . import config
from .dataset import HeldOutSet, MapData, Sampler, available_years, test_year


class Embedder(nn.Module):
    """ViT backbone -> [CLS, mean patch] -> (optional projection) -> unit vector."""

    def __init__(self, vit_name, dim=256, pretrained=True, project=True, img_size=224):
        super().__init__()
        self.vit = timm.create_model(vit_name, pretrained=pretrained, num_classes=0, img_size=img_size)
        d = self.vit.num_features * 2
        self.proj = nn.Sequential(nn.Linear(d, 512), nn.GELU(), nn.Linear(512, dim)) if project else nn.Identity()

    def forward(self, img):
        t = self.vit.forward_features(img)
        k = self.vit.num_prefix_tokens
        h = torch.cat([t[:, 0], t[:, k:].mean(1)], -1)
        return F.normalize(self.proj(h).float(), dim=-1)


def load_backbone_from(model, ckpt_path):
    """Initialise the ViT from a trained slide-6 localizer checkpoint (shared backbone)."""
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)["model"]
    vit = {k[len("enc.vision.vit."):]: v for k, v in sd.items() if k.startswith("enc.vision.vit.")}
    missing, _ = model.vit.load_state_dict(vit, strict=False)
    return len(vit), missing


def lattice(data, stride):
    """Reference tile centres: every `stride` m inside the usable footprint."""
    res = data.cfg["grid_res"]
    k = max(1, round(stride / res))
    ii, jj = np.nonzero(data.valid)
    keep = (ii % k == k // 2) & (jj % k == k // 2)
    e = data.ext.x0 + (jj[keep] + 0.5) * res
    n = data.ext.y1 - (ii[keep] + 0.5) * res
    return np.stack([e, n], 1)


class Tiles(Dataset):
    def __init__(self, data, pos, year):
        self.s, self.pos, self.year = Sampler(data, train=False), pos, year

    def __len__(self):
        return len(self.pos)

    def __getitem__(self, i):
        return self.s.crop(self.year, float(self.pos[i, 0]), float(self.pos[i, 1]), None)


class Queries(Dataset):
    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        return self.ds[i]["img"]


@torch.no_grad()
def embed(model, ds, dev, workers=16, batch=256):
    model.eval()
    out = []
    for img in DataLoader(ds, batch, num_workers=workers, pin_memory=True):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(model(img.to(dev, non_blocking=True)).float().cpu())
    return torch.cat(out).numpy()


def recall(q_emb, q_pos, db_emb, db_pos, radius, ks=(1, 5, 10, 25)):
    sims = q_emb @ db_emb.T
    top = np.argsort(-sims, 1)[:, :max(ks)]
    d = np.hypot(db_pos[top, 0] - q_pos[:, None, 0], db_pos[top, 1] - q_pos[:, None, 1])
    hit = d < radius
    return {f"R@{k}": float(hit[:, :k].any(1).mean()) for k in ks}, float(np.median(d[:, 0]))


def cell_log_prior(sims, db_pos, ext, cell, tau):
    """Turn tile similarities (B, T) into log-probabilities over coarse cells (B, C)."""
    from .geo import cell_centers
    _, (rows, cols) = cell_centers(ext, cell)
    r = np.clip(((ext.y1 - db_pos[:, 1]) // cell).astype(int), 0, rows - 1)
    c = np.clip(((db_pos[:, 0] - ext.x0) // cell).astype(int), 0, cols - 1)
    cid = r * cols + c
    lp = np.full((len(sims), rows * cols), -1e9, np.float64)
    z = sims / tau
    for j in np.unique(cid):                       # log-sum-exp of the tiles in each cell
        m = cid == j
        lp[:, j] = np.logaddexp.reduce(z[:, m], axis=1)
    return lp - np.logaddexp.reduce(lp, axis=1, keepdims=True)


class Retriever:
    """Embedded reference map + query embedding, for c2f / the particle filter."""

    def __init__(self, ckpt, data, cfg, dev="cuda"):
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        rc = ck["rcfg"]
        self.model = Embedder(rc["vit"], rc["dim"], pretrained=False, project=rc["project"]).to(dev).eval()
        self.model.load_state_dict(ck["model"])
        self.db_pos, self.db_emb = ck["db_pos"], ck["db_emb"]
        self.tau = cfg["retrieval"]["tau"]
        self.data, self.cfg, self.dev = data, cfg, dev

    @torch.no_grad()
    def log_prior(self, img):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            q = self.model(img.to(self.dev)).float().cpu().numpy()
        sims = q @ self.db_emb.T
        return cell_log_prior(sims, self.db_pos, self.data.ext, self.cfg["coarse"]["cell"], self.tau)


def evaluate(model, data, cfg, dev, years):
    """Recall on val (training-year imagery) and test (held-out year) windows."""
    rc = cfg["retrieval"]
    workers = cfg["train"]["workers"]
    db_pos = lattice(data, rc["stride"])
    db_emb = embed(model, Tiles(data, db_pos, cfg["dsmac"]["ref_year"]), dev, workers)
    res = {}
    for split, year in years.items():
        ds = HeldOutSet(data, split, year)
        q_emb = embed(model, Queries(ds), dev, workers)
        q_pos = np.stack([ds.w["e"], ds.w["n"]], 1)
        r, med = recall(q_emb, q_pos, db_emb, db_pos, rc["hit_radius"])
        res[f"{split}_{year}"] = dict(**r, top1_median_m=med, n=len(ds))
    return res, db_pos, db_emb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", nargs="?")
    ap.add_argument("--zero-shot", action="store_true")
    ap.add_argument("--vit", default=None)
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    cfg = config.load("configs/base.yaml", a.set)
    dev = torch.device("cuda")
    years = available_years(cfg)
    data = MapData(cfg, years)
    if a.zero_shot:
        vit = a.vit or cfg["model"]["vit"]
        model = Embedder(vit, project=False).to(dev)
        tag = "zeroshot_" + vit.split(".")[0]
    else:
        ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
        rc = ck["rcfg"]
        model = Embedder(rc["vit"], rc["dim"], pretrained=False, project=rc["project"]).to(dev)
        model.load_state_dict(ck["model"])
        tag = "finetuned"
    ref = cfg["dsmac"]["ref_year"]
    splits = {"val": [y for y in years if y != ref][-1], "test": test_year(cfg)}   # never the map's own year
    res, _, _ = evaluate(model, data, cfg, dev, splits)
    for k, v in res.items():
        print(f"{tag} {k}: " + "  ".join(f"{m} {x:.1%}" for m, x in v.items() if m.startswith("R@"))
              + f"  top-1 median {v['top1_median_m']:.0f} m")
    out = config.path(cfg, "run_dir") / "phase2"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"retrieval_{tag}.json", "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
