"""Per-component CPU latency of the localisation pipeline at batch size 1.

Run pinned to a few slow cores to emulate edge hardware, e.g. 4 E-cores of an i7-12700H:
  taskset -c 12-15 .venv/bin/python -m btp.edge.profile --threads 4

Models are read from data/edge/ (stripped checkpoints, see the phase-4 notes in README).
"""
import argparse
import json
import time

import numpy as np
import torch

from .. import config
from ..geo import Extent

EDGE = config.ROOT / "data" / "edge"


def bench(fn, warmup=3, reps=10):
    for _ in range(warmup):
        fn()
    t = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        t.append(time.perf_counter() - t0)
    return 1000 * float(np.median(t))


def load_classifier(name="coarse"):
    from ..models import Localizer
    ck = torch.load(EDGE / f"{name}.pt", map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    cfg["model"]["pretrained"] = False
    m = Localizer(cfg, Extent.load(EDGE / "extent.json")).eval()
    m.load_state_dict(ck["model"])
    return m, cfg


def load_retrieval():
    from ..retrieval import Embedder
    ck = torch.load(EDGE / "retrieval.pt", map_location="cpu", weights_only=False)
    rc = ck["rcfg"]
    m = Embedder(rc["vit"], rc["dim"], pretrained=False, project=rc["project"]).eval()
    m.load_state_dict(ck["model"])
    return m, ck["db_emb"].astype(np.float32)


def load_loftr():
    import kornia.feature as KF
    net = KF.LoFTR(pretrained=None).eval()
    sd = torch.load(EDGE / "loftr_outdoor.ckpt", map_location="cpu", weights_only=False)["state_dict"]
    net.load_state_dict(sd)
    return net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    res = {"threads": a.threads}

    with torch.inference_mode():
        m, cfg = load_classifier()
        L = cfg["seq"]["len"]
        batch = dict(img=torch.randn(1, 3, 224, 224), mag=torch.randn(1, 4, L), grav=torch.randn(1, 5, L),
                     avail=torch.ones(1, 3, dtype=torch.bool))
        res["classifier_full_ms"] = bench(lambda: m(batch))
        res["classifier_vit_only_ms"] = bench(lambda: m.enc["vision"](batch["img"]))
        res["classifier_profiles_ms"] = bench(lambda: (m.enc["mag"](batch["mag"]), m.enc["grav"](batch["grav"])))

        r, db = load_retrieval()
        res["retrieval_embed_ms"] = bench(lambda: r(batch["img"]))
        q = np.random.randn(1, db.shape[1]).astype(np.float32)
        res["retrieval_search_ms"] = bench(lambda: np.argpartition(-(q @ db.T), 30, axis=1))
        res["retrieval_db_tiles"] = int(len(db))

        net = load_loftr()
        for ref_px in (624, 384, 320):          # search radius 250 m, 100 m, 60 m at 1.25 m/px
            q_img, r_img = torch.rand(1, 1, 224, 224), torch.rand(1, 1, ref_px, ref_px)
            res[f"loftr_224_vs_{ref_px}_ms"] = bench(lambda: net({"image0": q_img, "image1": r_img}), 2, 5)

    # Particle-filter geophysics likelihood: N particles x 64 track samples x 3 map channels.
    from ..geo import bilinear
    g = np.load(EDGE / "geo_grids.npz")
    maps = [g[k] for k in ("mag", "gD", "GDD")]
    H, W = maps[0].shape
    for N in (3000, 1000, 300):
        r = np.random.uniform(10, H - 10, (N, 64))
        c = np.random.uniform(10, W - 10, (N, 64))
        y = np.random.randn(64)

        def step():
            out = 0.0
            for mp in maps:
                res_ = y[None] - bilinear(mp, r, c)
                res_ -= res_.mean(-1, keepdims=True)
                out = out + (res_ ** 2).sum(-1)
            return out
        res[f"pf_geo_likelihood_{N}_ms"] = bench(step)

    print(json.dumps(res, indent=1))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
