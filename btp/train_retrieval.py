"""Fine-tune the slide-6 ViT for retrieval against the onboard reference map (Sample4Geo-style).

A query is an augmented crop from a training year at a random position. Its positive is the
reference-map tile (dsmac.ref_year) at the nearest database-lattice point, so the network
learns exactly the lookup it does at test time. Loss: symmetric InfoNCE. Hard negatives:
half of the batches are drawn from one small neighbourhood, so the other samples in the
batch are nearby, similar-looking places. Pairs closer than the hit radius are not
treated as negatives.

  python -m btp.train_retrieval [--init runs/coarse/best.pt] [--smoke]
"""
import argparse
import json
import math
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset

from . import config
from .dataset import MapData, Sampler, available_years, test_year
from .retrieval import Embedder, evaluate, load_backbone_from


class Pairs(Dataset):
    """Each item is a whole batch, so neighbourhood batches can be built."""

    def __init__(self, data, years, batch, steps, stride, local_radius, ref_year):
        self.d, self.years, self.B, self.steps = data, years, batch, steps
        self.s_q, self.s_r = Sampler(data, train=True), Sampler(data, train=False)
        self.stride, self.local, self.ref_year = stride, local_radius, ref_year
        self.cells = np.argwhere(data.valid)
        self.res = data.cfg["grid_res"]
        k = max(1, round(stride / self.res))
        self.k = k

    def __len__(self):
        return self.steps

    def snap(self, i, j):
        """Nearest database-lattice cell (same rule as retrieval.lattice), if it is valid."""
        k = self.k
        si, sj = (i // k) * k + k // 2, (j // k) * k + k // 2
        H, W = self.d.valid.shape
        if 0 <= si < H and 0 <= sj < W and self.d.valid[si, sj]:
            return si, sj
        return None

    def __getitem__(self, idx):
        rng = np.random.default_rng((torch.initial_seed() + idx * 104729) % 2**63)
        local = rng.random() < 0.5
        if local:
            ci, cj = self.cells[rng.integers(len(self.cells))]
            r = self.local / self.res
            near = self.cells[(np.abs(self.cells[:, 0] - ci) < r) & (np.abs(self.cells[:, 1] - cj) < r)]
            pool = near if len(near) >= self.B else self.cells
        else:
            pool = self.cells
        q, p, pos = [], [], []
        while len(q) < self.B:
            i, j = pool[rng.integers(len(pool))]
            sn = self.snap(i, j)
            if sn is None:
                continue
            e = self.d.ext.x0 + (j + rng.random()) * self.res
            n = self.d.ext.y1 - (i + rng.random()) * self.res
            pe = self.d.ext.x0 + (sn[1] + 0.5) * self.res
            pn = self.d.ext.y1 - (sn[0] + 0.5) * self.res
            year = self.years[rng.integers(len(self.years))]
            q.append(self.s_q.crop(year, e, n, rng))
            p.append(self.s_r.crop(self.ref_year, pe, pn, None))
            pos.append((pe, pn))
        return torch.stack(q), torch.stack(p), torch.tensor(pos, dtype=torch.float64)


def info_nce(q, p, pos, scale, radius):
    logits = scale * q @ p.T
    d = torch.cdist(pos, pos)
    false_neg = (d < radius) & ~torch.eye(len(q), dtype=torch.bool, device=q.device)
    logits = logits.masked_fill(false_neg, float("-inf"))
    t = torch.arange(len(q), device=q.device)
    return 0.5 * (F.cross_entropy(logits, t) + F.cross_entropy(logits.T, t))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--init", default="runs/coarse/best.pt", help="slide-6 checkpoint whose ViT to start from")
    ap.add_argument("--name", default="retrieval")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    cfg = config.load("configs/base.yaml", a.set)
    rc, t = cfg["retrieval"], cfg["train"]
    if a.smoke:
        rc.update(epochs=1, steps_per_epoch=20)
    run = config.path(cfg, "run_dir") / (a.name + ("_smoke" if a.smoke else ""))
    run.mkdir(parents=True, exist_ok=True)
    with open(run / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f)
    dev = torch.device("cuda")
    torch.manual_seed(t["seed"])

    ref_year = cfg["dsmac"]["ref_year"]
    years = [y for y in available_years(cfg) if y != ref_year] or available_years(cfg)
    data = MapData(cfg, available_years(cfg))
    model = Embedder(cfg["model"]["vit"], rc["dim"], pretrained=True, project=True).to(dev)
    init = config.ROOT / a.init
    if init.exists():
        n, missing = load_backbone_from(model, init)
        print(f"ViT initialised from {a.init} ({n} tensors, {len(missing)} missing)")
    else:
        print(f"{a.init} not found; starting from pretrained DINOv2")
    print(f"queries from {years}, reference map {ref_year}")

    ds = Pairs(data, years, rc["batch"], rc["steps_per_epoch"], rc["stride"], rc["local_radius"], ref_year)
    dl = DataLoader(ds, None, num_workers=t["workers"], pin_memory=True, persistent_workers=True)
    logit_scale = torch.nn.Parameter(torch.tensor(math.log(1 / 0.07), device=dev))
    vit_params = list(model.vit.parameters())
    opt = torch.optim.AdamW([{"params": vit_params, "lr": rc["lr_vit"]},
                             {"params": list(model.proj.parameters()) + [logit_scale], "lr": rc["lr"]}],
                            weight_decay=t["weight_decay"])
    total = rc["epochs"] * rc["steps_per_epoch"]
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, [rc["lr_vit"], rc["lr"]], total_steps=total,
                                                pct_start=max(0.05, 2 / total))
    eval_years = {"val": years[-1], "test": test_year(cfg)}
    best = -1.0
    for epoch in range(rc["epochs"]):
        model.train()
        t0, tot = time.time(), 0.0
        for step, (q, p, pos) in enumerate(dl):
            q, p, pos = q.to(dev, non_blocking=True), p.to(dev, non_blocking=True), pos.to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                eq, ep = model(q), model(p)
            loss = info_nce(eq, ep, pos, logit_scale.exp().clamp(max=100), rc["hit_radius"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item()
            if step % 100 == 0:
                print(f"ep {epoch} it {step}/{len(dl)} loss {loss.item():.3f} scale {logit_scale.exp().item():.1f} "
                      f"{(time.time() - t0) / (step + 1):.2f}s/it", flush=True)
        res, db_pos, db_emb = evaluate(model, data, cfg, dev, eval_years)
        val = res[f"val_{eval_years['val']}"]
        print(f"epoch {epoch} done in {time.time() - t0:.0f}s | loss {tot / len(dl):.3f} | "
              + " | ".join(f"{k}: R@1 {v['R@1']:.1%} R@10 {v['R@10']:.1%}" for k, v in res.items()), flush=True)
        ck = dict(model=model.state_dict(), rcfg=dict(vit=cfg["model"]["vit"], dim=rc["dim"], project=True),
                  db_pos=db_pos, db_emb=db_emb, epoch=epoch, res=res)
        torch.save(ck, run / "last.pt")
        if val["R@10"] > best:
            best = val["R@10"]
            torch.save(ck, run / "best.pt")
            with open(run / "best_val.json", "w") as f:
                json.dump(dict(epoch=epoch, **res), f, indent=1)
    print(f"best val R@10 {best:.1%} -> {run / 'best.pt'}")


if __name__ == "__main__":
    main()
