"""LoFTR with its CNN backbone (ResNet-FPN, ~60% of LoFTR's GPU time) compiled to TensorRT FP16.

The coarse/fine transformers and the matching stay in PyTorch (FP16): their output size depends on
the data, which TensorRT engines don't handle well. The backbone is a plain CNN with dynamic input
size, so it compiles cleanly.

  .venv-gpu/bin/python -m btp.edge.loftr_trt build
  then dsmac.method=loftr_trt (see matchers.py)
"""
import argparse

import torch
from torch import nn

from .trt import TRT_DIR, Engine, build
from .export import ONNX

ENGINE = TRT_DIR / "loftr_backbone_fp16.engine"


def export_and_build():
    import kornia.feature as KF
    net = KF.LoFTR(pretrained="outdoor").cuda().half().eval()
    onnx_path = ONNX / "loftr_backbone_fp16.onnx"
    x = torch.rand(1, 1, 320, 320, device="cuda", dtype=torch.float16)
    with torch.no_grad():
        torch.onnx.export(net.backbone, (x,), onnx_path, opset_version=17, input_names=["img"],
                          output_names=["feat_c", "feat_f"],
                          dynamic_axes={"img": {2: "h", 3: "w"}, "feat_c": {2: "hc", 3: "wc"},
                                        "feat_f": {2: "hf", 3: "wf"}}, dynamo=False)
    TRT_DIR.mkdir(parents=True, exist_ok=True)
    build(onnx_path, ENGINE, True, {"img": ((1, 1, 64, 64), (1, 1, 320, 320), (1, 1, 800, 800))})
    print(f"built {ENGINE}")


class TRTBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.eng = Engine(ENGINE)

    def forward(self, x):
        out = self.eng(img=x)
        return out["feat_c"], out["feat_f"]


def patch(loftr_net):
    """Replace a kornia LoFTR's backbone with the TensorRT engine (in place)."""
    loftr_net.backbone = TRTBackbone()
    return loftr_net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build"])
    ap.parse_args()
    export_and_build()


if __name__ == "__main__":
    main()
