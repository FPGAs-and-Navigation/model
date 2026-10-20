"""Per-component GPU latency at batch size 1 (laptop RTX 3060 as a Jetson stand-in).

  .venv-gpu/bin/python -m btp.edge.gpu_profile [--out runs/edge/gpu_profile.json]

Times the classifier and retrieval ViTs (FP32 / FP16), LoFTR (FP32 / FP16) and the particle
filter's geophysics likelihood (plain PyTorch on GPU vs the fused Triton kernel), and checks the
Triton kernel against the Numba CPU kernel.
"""
import argparse
import json

import numpy as np
import torch

from ..geo import Extent
from .profile import EDGE, load_classifier, load_loftr, load_retrieval


def cuda_time(fn, warmup=5, reps=30):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(reps):
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b))
    return float(np.median(ts))


def torch_geo(pos, te, tn, y, maps, mask, inv, ext, res):
    """Unfused GPU reference (what you'd write without a custom kernel)."""
    e = pos[:, 0:1] + te[None]
    n = pos[:, 1:2] + tn[None]
    r = (ext.y1 - n) / res
    c = (e - ext.x0) / res
    H, W = mask.shape
    ri, ci = r.long().clamp(0, H - 1), c.long().clamp(0, W - 1)
    inside = (r >= 0) & (c >= 0) & (r < H) & (c < W) & mask[ri, ci].bool()
    rr = (r - 0.5).clamp(0, H - 1.001)
    cc = (c - 0.5).clamp(0, W - 1.001)
    r0, c0 = rr.long(), cc.long()
    fr, fc = rr - r0, cc - c0
    n_in = inside.sum(-1).clamp(min=1)
    out = []
    for k in range(3):
        m = maps[k]
        v = (m[r0, c0] * (1 - fr) * (1 - fc) + m[r0, c0 + 1] * (1 - fr) * fc
             + m[r0 + 1, c0] * fr * (1 - fc) + m[r0 + 1, c0 + 1] * fr * fc)
        d = torch.where(inside, y[k][None] - v, 0)
        d = torch.where(inside, d - d.sum(-1, keepdim=True) / n_in[:, None], 0)
        out.append((d * d).sum(-1) * inv[k] * te.shape[0] / n_in)
    return torch.stack(out, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    dev = torch.device("cuda")
    res = {"gpu": torch.cuda.get_device_name(0)}
    torch.backends.cudnn.benchmark = True

    with torch.inference_mode():
        m, cfg = load_classifier()
        m = m.to(dev)
        L = cfg["seq"]["len"]
        batch = dict(img=torch.randn(1, 3, 224, 224, device=dev), mag=torch.randn(1, 4, L, device=dev),
                     grav=torch.randn(1, 5, L, device=dev), avail=torch.ones(1, 3, dtype=torch.bool, device=dev))
        res["classifier_fp32_ms"] = cuda_time(lambda: m(batch))
        with torch.autocast("cuda", dtype=torch.float16):
            res["classifier_fp16_ms"] = cuda_time(lambda: m(batch))
        r, _ = load_retrieval()
        r = r.to(dev)
        res["retrieval_fp32_ms"] = cuda_time(lambda: r(batch["img"]))
        with torch.autocast("cuda", dtype=torch.float16):
            res["retrieval_fp16_ms"] = cuda_time(lambda: r(batch["img"]))

        net = load_loftr().to(dev)
        for ref_px in (320, 624):
            q, rr = torch.rand(1, 1, 224, 224, device=dev), torch.rand(1, 1, ref_px, ref_px, device=dev)
            res[f"loftr_fp32_224_vs_{ref_px}_ms"] = cuda_time(lambda: net({"image0": q, "image1": rr}), 3, 10)
            with torch.autocast("cuda", dtype=torch.float16):
                res[f"loftr_fp16_224_vs_{ref_px}_ms"] = cuda_time(lambda: net({"image0": q, "image1": rr}), 3, 10)

    # geophysics likelihood: torch vs Triton, checked against the Numba CPU kernel
    from .kernels import geo_cost as numba_geo
    from .triton_kernels import geo_cost as triton_geo
    ext = Extent.load(EDGE / "extent.json")
    g = np.load(EDGE / "geo_grids.npz")
    maps_np = np.stack([g[k] for k in ("mag", "gD", "GDD")]).astype(np.float32)
    mask_np = g["mask_mag"] & g["mask_grav"]
    maps, mask = torch.from_numpy(maps_np).to(dev), torch.from_numpy(mask_np.astype(np.uint8)).to(dev)
    inv = torch.tensor([1 / 12.6 ** 2, 1 / 0.28 ** 2, 1 / 37.2 ** 2], device=dev)
    rng = np.random.default_rng(0)
    te_np = np.linspace(-1575, 0, 64) * 0.94
    tn_np = np.linspace(-1575, 0, 64) * 0.34
    y_np = (rng.normal(0, 1, (3, 64)) * np.array([[100], [5], [100]])).astype(np.float32)
    te, tn = torch.tensor(te_np, device=dev, dtype=torch.float32), torch.tensor(tn_np, device=dev, dtype=torch.float32)
    y = torch.from_numpy(y_np).to(dev)
    for N in (1000, 3000, 10000, 100000):
        pos_np = np.c_[rng.uniform(ext.x0, ext.x1, N), rng.uniform(ext.y0, ext.y1, N)]
        pos = torch.from_numpy(pos_np).to(dev)
        ct, _ = triton_geo(pos, te, tn, y, maps, mask, inv, ext, 25.0)
        rows, cols = ext.to_pix(pos_np[:, 0:1] + te_np[None], pos_np[:, 1:2] + tn_np[None], 25.0)
        cn, _ = numba_geo(maps_np, mask_np, rows, cols, y_np, inv.cpu().numpy(), 8)
        rel = float(np.max(np.abs(ct.cpu().numpy() - cn) / (np.abs(cn) + 1)))
        res[f"geo_N{N}_torch_ms"] = cuda_time(lambda: torch_geo(pos.float(), te, tn, y, maps, mask, inv, ext, 25.0))
        res[f"geo_N{N}_triton_ms"] = cuda_time(lambda: triton_geo(pos, te, tn, y, maps, mask, inv, ext, 25.0))
        res[f"geo_N{N}_triton_vs_numba_max_rel_diff"] = rel

    print(json.dumps(res, indent=1))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
