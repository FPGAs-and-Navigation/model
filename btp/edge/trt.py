"""TensorRT engines for the networks (the Jetson deployment path).

  .venv-gpu/bin/python -m btp.edge.trt build      # data/edge/onnx/*.onnx -> data/edge/trt/*.engine (FP16)
  .venv-gpu/bin/python -m btp.edge.trt bench      # latency + agreement with ONNX Runtime FP32

The same ONNX files build on a Jetson with JetPack's TensorRT; engines are device-specific and
must be rebuilt there.
"""
import argparse
import json
import time

import numpy as np
import tensorrt as trt
import torch

from .. import config
from .export import ONNX

TRT_DIR = ONNX.parent / "trt"
LOGGER = trt.Logger(trt.Logger.WARNING)


def export_fp16():
    """Export FP16 ONNX graphs straight from half-precision models on the GPU, so every tensor and
    constant is FP16 (TensorRT 11 builds strongly-typed networks: precision comes from the graph)."""
    from .export import ClassifierWrapper
    from .ldsmac import LDSMAC
    from .profile import load_classifier, load_retrieval
    m, cfg = load_classifier()
    L = cfg["seq"]["len"]
    h = lambda *s: torch.randn(*s, device="cuda", dtype=torch.float16)
    torch.backends.mha.set_fastpath_enabled(False)      # the fused CUDA attention kernel has no ONNX op
    with torch.no_grad():
        torch.onnx.export(ClassifierWrapper(m).cuda().half().eval(), (h(1, 3, 224, 224), h(1, 4, L), h(1, 5, L)),
                          ONNX / "classifier_fp16.onnx", opset_version=17, input_names=["img", "mag", "grav"],
                          output_names=["logits", "w"], dynamo=False)
        r, _ = load_retrieval()
        torch.onnx.export(r.cuda().half().eval(), (h(1, 3, 224, 224),), ONNX / "retrieval_fp16.onnx",
                          opset_version=17, input_names=["img"], output_names=["emb"], dynamo=False)
        sd = torch.load(config.ROOT / "runs/edge/ldsmac.pt", map_location="cpu", weights_only=False)
        ld = LDSMAC(**sd.get("kw", {})).eval()
        ld.load_state_dict(sd["model"])
        torch.onnx.export(ld.net.cuda().half(), (h(1, 3, 320, 320),), ONNX / "ldsmac_feat_fp16.onnx",
                          opset_version=17, input_names=["img"], output_names=["feat"],
                          dynamic_axes={"img": {2: "h", 3: "w"}, "feat": {2: "fh", 3: "fw"}}, dynamo=False)


def build(onnx_path, engine_path, fp16=True, shapes=None):
    builder = trt.Builder(LOGGER)
    net = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parser = trt.OnnxParser(net, LOGGER)
    if not parser.parse(open(onnx_path, "rb").read()):
        raise RuntimeError("\n".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    cfg = builder.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    if shapes:   # dynamic inputs: name -> (min, opt, max)
        prof = builder.create_optimization_profile()
        for name, (mn, op, mx) in shapes.items():
            prof.set_shape(name, mn, op, mx)
        cfg.add_optimization_profile(prof)
    eng = builder.build_serialized_network(net, cfg)
    if eng is None:
        raise RuntimeError(f"TensorRT build failed for {onnx_path}")
    engine_path.write_bytes(bytes(eng))
    return engine_path


class Engine:
    """Run a TensorRT engine on torch CUDA tensors (no extra copies)."""

    def __init__(self, path):
        self.engine = trt.Runtime(LOGGER).deserialize_cuda_engine(path.read_bytes())
        self.ctx = self.engine.create_execution_context()
        self.names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.inputs = [n for n in self.names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        self.outputs = [n for n in self.names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
        self.stream = torch.cuda.current_stream()

    def __call__(self, **feeds):
        for n in self.inputs:
            want = torch.float16 if self.engine.get_tensor_dtype(n) == trt.float16 else torch.float32
            t = feeds[n].to(want).contiguous()
            feeds[n] = t
            self.ctx.set_input_shape(n, tuple(t.shape))
            self.ctx.set_tensor_address(n, t.data_ptr())
        outs = {}
        for n in self.outputs:
            shape = tuple(self.ctx.get_tensor_shape(n))
            dt = torch.float32 if self.engine.get_tensor_dtype(n) == trt.float32 else torch.float16
            outs[n] = torch.empty(shape, device="cuda", dtype=dt)
            self.ctx.set_tensor_address(n, outs[n].data_ptr())
        self.ctx.execute_async_v3(self.stream.cuda_stream)
        return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "bench"])
    a = ap.parse_args()
    TRT_DIR.mkdir(parents=True, exist_ok=True)
    nets = {"classifier": None, "retrieval": None,
            "ldsmac_feat": {"img": ((1, 3, 128, 128), (1, 3, 320, 320), (1, 3, 1024, 1024))}}
    if a.cmd == "build":
        export_fp16()
        for name, shapes in nets.items():
            t0 = time.time()
            build(ONNX / f"{name}_fp16.onnx", TRT_DIR / f"{name}_fp16.engine", True, shapes)
            print(f"built {name}_fp16.engine in {time.time() - t0:.0f}s")
        return

    import onnxruntime as ort
    rng = np.random.default_rng(0)
    feeds = {"classifier": {"img": rng.normal(size=(1, 3, 224, 224)), "mag": rng.normal(size=(1, 4, 64)),
                            "grav": rng.normal(size=(1, 5, 64))},
             "retrieval": {"img": rng.normal(size=(1, 3, 224, 224))},
             "ldsmac_feat": {"img": rng.normal(size=(1, 3, 320, 320))}}
    res = {"gpu": torch.cuda.get_device_name(0)}
    for name, feed in feeds.items():
        feed = {k: v.astype(np.float32) for k, v in feed.items()}
        eng = Engine(TRT_DIR / f"{name}_fp16.engine")
        tf = {k: torch.from_numpy(v).cuda() for k, v in feed.items()}
        for _ in range(10):
            eng(**dict(tf))
        torch.cuda.synchronize()
        ts = []
        for _ in range(50):
            t0 = time.perf_counter()
            out = eng(**dict(tf))
            torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
        res[f"{name}_trt_fp16_ms"] = 1000 * float(np.median(ts))
        ref = ort.InferenceSession(str(ONNX / f"{name}.onnx"), providers=["CPUExecutionProvider"]).run(None, feed)[0]
        got = out[eng.outputs[0]].float().cpu().numpy()
        a_, b_ = ref.ravel(), got.ravel()
        res[f"{name}_trt_vs_fp32_cosine"] = float(a_ @ b_ / (np.linalg.norm(a_) * np.linalg.norm(b_)))
        if name == "classifier":
            res["classifier_trt_top10_overlap"] = len(set(np.argsort(-ref[0])[:10]) & set(np.argsort(-got[0])[:10])) / 10
    print(json.dumps(res, indent=1))
    out = config.ROOT / "runs/edge"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "trt_bench.json", "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
