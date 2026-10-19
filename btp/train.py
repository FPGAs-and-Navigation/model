"""Train the localizer.

  python -m btp.train --name vision --set modalities=[vision]
  python -m btp.train --name full                      # vision + mag + grav
  python -m btp.train --smoke                          # 50 steps + a small validation pass
"""
import argparse
import json
import math
import time

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

from . import config
from .config import MODALITIES
from .dataset import HeldOutSet, MapData, TrainSet, available_years
from .models import Localizer, loss_fn


def to_dev(batch, dev):
    return {k: v.to(dev, non_blocking=True) if torch.is_tensor(v) and k != "en" else v
            for k, v in batch.items()}


def param_groups(model, cfg):
    """Layer-wise LR decay for the ViT (lower blocks change least); full LR elsewhere."""
    t = cfg["train"]
    groups, other = [], []
    vit = model.enc["vision"].vit if "vision" in model.enc else None
    n_blocks = len(vit.blocks) if vit is not None else 0
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if vit is not None and name.startswith("enc.vision.vit."):
            sub = name[len("enc.vision.vit."):]
            if sub.startswith("blocks."):
                depth = int(sub.split(".")[1]) + 1
            elif sub.startswith(("norm", "fc_norm", "head")):
                depth = n_blocks + 1
            else:
                depth = 0                                  # patch/pos embeddings, cls token
            lr = t["lr_vit"] * t["layer_decay"] ** (n_blocks + 1 - depth)
            wd = 0.0 if p.ndim == 1 or "pos_embed" in sub or "token" in sub else t["weight_decay"]
            groups.append({"params": [p], "lr": lr, "weight_decay": wd})
        else:
            other.append(p)
    groups.append({"params": other, "lr": t["lr"], "weight_decay": t["weight_decay"]})
    for g in groups:
        g["base_lr"] = g["lr"]
    return groups


@torch.no_grad()
def evaluate(model, loader, ext, dev):
    """Errors in metres for the fused output and each modality alone, plus mean fusion weights."""
    model.eval()
    errs = {"fused": []}
    errs.update({m: [] for m in model.mods})
    weights, preds = [], []
    for batch in loader:
        b = to_dev(batch, dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(b)
        en = batch["en"].numpy()
        def metres(p):
            e, n = ext.denormalize(p[:, 0], p[:, 1])
            return np.hypot(e - en[:, 0], n - en[:, 1]), e, n
        err, e, n = metres(out["pos"].float().cpu().numpy())
        errs["fused"].append(err)
        preds.append(np.c_[en, e, n])
        aux = out["aux_pos"].float().cpu().numpy()
        for m in model.mods:
            errs[m].append(metres(aux[:, MODALITIES.index(m)])[0])
        weights.append(out["w"].float().cpu().numpy())
    model.train()
    res = {}
    for k, v in errs.items():
        v = np.concatenate(v)
        res[k] = dict(mean=float(v.mean()), median=float(np.median(v)), p90=float(np.percentile(v, 90)),
                      within_25m=float((v < 25).mean()), within_100m=float((v < 100).mean()))
    w = np.concatenate(weights).mean(0)
    res["weights"] = {m: float(w[i]) for i, m in enumerate(MODALITIES)}
    res["n"] = int(sum(len(x) for x in errs["fused"]))
    return res, np.concatenate(preds)


def lr_at(step, total, warmup):
    if step < warmup:
        return (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--name", default=None)
    ap.add_argument("--smoke", action="store_true", help="50 steps on a tiny validation set")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    cfg = config.load(a.config, a.set)
    t = cfg["train"]
    if a.smoke:
        t.update(epochs=1, samples_per_epoch=50 * t["batch"])
    name = a.name or "+".join(cfg["modalities"])
    run = config.path(cfg, "run_dir") / (name + ("_smoke" if a.smoke else ""))
    run.mkdir(parents=True, exist_ok=True)
    with open(run / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f)

    torch.manual_seed(t["seed"])
    np.random.seed(t["seed"])
    dev = torch.device("cuda")
    years = available_years(cfg)
    assert years, "no NAIP mosaics prepared; run btp.prepare_imagery"
    print(f"run {run.name}: modalities {cfg['modalities']}, imagery years {years}")
    data = MapData(cfg, years)
    train_ds = TrainSet(data)
    val_ds = HeldOutSet(data, "val", years[-1])
    if a.smoke:
        val_ds = Subset(val_ds, range(0, len(val_ds), max(1, len(val_ds) // 256)))
    kw = dict(num_workers=t["workers"], pin_memory=True, persistent_workers=True)
    train_dl = DataLoader(train_ds, t["batch"], shuffle=False, drop_last=True, **kw)
    val_dl = DataLoader(val_ds, 128, shuffle=False, **kw)
    print(f"train {len(train_ds)} samples/epoch, val {len(val_ds)} windows")

    model = Localizer(cfg).to(dev)
    opt = torch.optim.AdamW(param_groups(model, cfg), betas=(0.9, 0.999))
    steps_per_epoch = len(train_dl)
    total = t["epochs"] * steps_per_epoch
    start_epoch, best, step = 0, float("inf"), 0
    if a.resume and (run / "last.pt").exists():
        ck = torch.load(run / "last.pt", map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        start_epoch, best, step = ck["epoch"] + 1, ck["best"], ck["step"]
        print(f"resumed from epoch {start_epoch}")

    tb = SummaryWriter(run / "tb")
    for epoch in range(start_epoch, t["epochs"]):
        t0, agg = time.time(), {}
        for i, batch in enumerate(train_dl):
            f = lr_at(step, total, t["warmup_steps"])
            for g in opt.param_groups:
                g["lr"] = g["base_lr"] * f
            b = to_dev(batch, dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(b)
            loss, logs = loss_fn(out, b["target"], cfg)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            for k, v in logs.items():
                agg[k] = agg.get(k, 0) + v
            if step % 50 == 0:
                for k, v in logs.items():
                    tb.add_scalar(f"train/{k}", v, step)
                w = out["w"].float().mean(0)
                for j, m in enumerate(MODALITIES):
                    tb.add_scalar(f"train/w_{m}", w[j].item(), step)
            if i % 200 == 0:
                print(f"ep {epoch} it {i}/{steps_per_epoch} loss {logs['loss']:.3f} nll {logs['nll']:.3f} "
                      f"lr {opt.param_groups[-1]['lr']:.2e} {(time.time() - t0) / (i + 1):.2f}s/it", flush=True)

        res, _ = evaluate(model, val_dl, data.ext, dev)
        for k, v in res.items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    tb.add_scalar(f"val_{k}/{kk}", vv, epoch)
        med = res["fused"]["median"]
        print(f"epoch {epoch} done in {time.time() - t0:.0f}s | train loss {agg['loss'] / steps_per_epoch:.3f} | "
              f"val median {med:.1f} m, mean {res['fused']['mean']:.1f} m, p90 {res['fused']['p90']:.1f} m | "
              + " ".join(f"{m} {res[m]['median']:.0f}m" for m in model.mods)
              + " | w " + " ".join(f"{m}={v:.2f}" for m, v in res["weights"].items()), flush=True)
        ck = dict(model=model.state_dict(), opt=opt.state_dict(), epoch=epoch, step=step, best=min(best, med), cfg=cfg)
        torch.save(ck, run / "last.pt")
        if med < best:
            best = med
            torch.save(ck, run / "best.pt")
            with open(run / "best_val.json", "w") as f:
                json.dump(dict(epoch=epoch, **res), f, indent=1)
    print(f"best val median {best:.1f} m -> {run / 'best.pt'}")


if __name__ == "__main__":
    main()
