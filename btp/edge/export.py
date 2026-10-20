"""Export the networks to ONNX, quantise them to INT8 and time them with ONNX Runtime on CPU.

  python -m btp.edge.export export            # writes data/edge/onnx/*.onnx (fp32 + int8)
  taskset -c 12-15 python -m btp.edge.export bench --threads 4

Models: the slide-6 classifier (image + profiles -> cell logits, UDML weights), the retrieval
embedder (image -> embedding) and the learned-DSMAC feature network (image -> feature map).
"""
import argparse
import json
import time

import numpy as np
import torch
from torch import nn

from .. import config
from .profile import EDGE, load_classifier, load_retrieval

ONNX = EDGE / "onnx"


class ClassifierWrapper(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, img, mag, grav):
        avail = torch.ones(img.shape[0], 3, dtype=torch.bool)
        out = self.m(dict(img=img, mag=mag, grav=grav, avail=avail))
        return out["logits"], out["w"]


def export():
    ONNX.mkdir(parents=True, exist_ok=True)
    m, cfg = load_classifier()
    L = cfg["seq"]["len"]
    args = (torch.randn(1, 3, 224, 224), torch.randn(1, 4, L), torch.randn(1, 5, L))
    torch.onnx.export(ClassifierWrapper(m).eval(), args, ONNX / "classifier.onnx", opset_version=17,
                      input_names=["img", "mag", "grav"], output_names=["logits", "w"], dynamo=False)
    r, _ = load_retrieval()
    torch.onnx.export(r.eval(), (torch.randn(1, 3, 224, 224),), ONNX / "retrieval.onnx", opset_version=17,
                      input_names=["img"], output_names=["emb"], dynamo=False)
    ck = config.ROOT / "runs/edge/ldsmac.pt"
    if not ck.exists():
        ck = EDGE / "ldsmac.pt"
    if ck.exists():
        from .ldsmac import LDSMAC
        sd = torch.load(ck, map_location="cpu", weights_only=False)
        ld = LDSMAC(**sd.get("kw", {})).eval()
        ld.load_state_dict(sd["model"])
        torch.onnx.export(ld.net, (torch.randn(1, 3, 224, 224),), ONNX / "ldsmac_feat.onnx", opset_version=17,
                          input_names=["img"], output_names=["feat"],
                          dynamic_axes={"img": {2: "h", 3: "w"}, "feat": {2: "fh", 3: "fw"}}, dynamo=False)
    from onnxruntime.quantization import QuantType, quantize_dynamic
    for f in sorted(ONNX.glob("*.onnx")):
        if f.stem.endswith("_int8"):
            continue
        quantize_dynamic(str(f), str(f.with_name(f.stem + "_int8.onnx")), weight_type=QuantType.QInt8)
    for f in sorted(ONNX.glob("*.onnx")):
        print(f"{f.name:28s} {f.stat().st_size / 1e6:6.1f} MB")


def bench(threads):
    import onnxruntime as ort
    torch.set_num_threads(threads)
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    rng = np.random.default_rng(0)

    def t(f, n=15):
        for _ in range(3):
            f()
        ts = []
        for _ in range(n):
            t0 = time.perf_counter()
            f()
            ts.append(time.perf_counter() - t0)
        return 1000 * float(np.median(ts))

    res = {"threads": threads}
    feeds = {"classifier": {"img": rng.normal(size=(1, 3, 224, 224)).astype(np.float32),
                            "mag": rng.normal(size=(1, 4, 64)).astype(np.float32),
                            "grav": rng.normal(size=(1, 5, 64)).astype(np.float32)},
             "retrieval": {"img": rng.normal(size=(1, 3, 224, 224)).astype(np.float32)},
             "ldsmac_feat": {"img": rng.normal(size=(1, 3, 320, 320)).astype(np.float32)}}
    for name, feed in feeds.items():
        outs = {}
        for v in ("", "_int8"):
            f = ONNX / f"{name}{v}.onnx"
            if not f.exists():
                continue
            s = ort.InferenceSession(str(f), so, providers=["CPUExecutionProvider"])
            res[f"{name}{v or '_fp32'}_ms"] = t(lambda: s.run(None, feed))
            outs[v] = s.run(None, feed)[0]
        if "" in outs and "_int8" in outs:
            a, b = outs[""].ravel(), outs["_int8"].ravel()
            res[f"{name}_int8_vs_fp32_cosine"] = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
            if name == "classifier":
                ta, tb = np.argsort(-outs[""][0])[:10], np.argsort(-outs["_int8"][0])[:10]
                res["classifier_int8_top10_overlap"] = len(set(ta) & set(tb)) / 10
    print(json.dumps(res, indent=1))
    out = config.ROOT / "runs/edge"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"onnx_bench_t{threads}.json", "w") as f:
        json.dump(res, f, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["export", "bench"])
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    export() if a.cmd == "export" else bench(a.threads)


if __name__ == "__main__":
    main()
