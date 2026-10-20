# GPU / Jetson path: results

Measured on a laptop **RTX 3060 (Ampere, 6 GB)** as a Jetson Orin stand-in (same GPU generation;
TensorRT is the Jetson runtime). Jetson figures at the bottom are scaled estimates, not measurements.

## Components (batch 1)
| component | before | optimised | accuracy effect |
|---|---|---|---|
| Particle update, 1,000 / 10,000 particles | 0.67 / 3.39 ms (PyTorch on GPU) | **0.17 / 0.31 ms** (fused Triton kernel) | matches the CPU kernel to 0.007% |
| Classifier (ViT-S + UDML) | 5.8 ms FP32, 4.4 ms FP16 | **1.6 ms** TensorRT FP16 | identical top-10 cells (cosine 0.9999999) |
| Retrieval ViT | 4.6 ms FP32, 2.5 ms FP16 | **1.2 ms** TensorRT FP16 | cosine 0.999995 |
| LoFTR, ±60 m tracking | 36 ms FP32, 22 ms FP16 | **14 ms** (TensorRT FP16 backbone + FP16 matching) | identical: 2.0 m median, 98% within 10 m |
| LoFTR, ±250 m first fix | 92 ms FP32, 71 ms FP16 | **44 ms** | 94% → 93% within 10 m (FP16) |

## End to end: 20 held-out test flights, realistic camera views
Everything runs on every step: networks, image matching, particle update. 1,000 particles, bias estimation.

| system | median | within 10 m | flights lost | rate |
|---|---|---|---|---|
| Desktop reference (phase 2, RTX 5070 Ti, FP32, 3,000 particles) | 1.4 m | 98% | 1/20 | – |
| **GPU edge pipeline on RTX 3060 laptop** | **1.4 m** | **98%** (99% after 10 steps) | 1/20 | **38.7 Hz** (25.8 ms/step) |

Per step: image matching 19.5 ms, networks 4.5 ms, particle update 0.4 ms, the rest of the filter about 1.5 ms.

## Jetson estimates (assumptions, not measured)
Batch-1 inference here is mostly limited by memory bandwidth and kernel latency.
- **Bandwidth:** RTX 3060 laptop 336 GB/s; Jetson AGX Orin 204 GB/s; Orin Nano 68–102 GB/s.
- **GPU size:** AGX Orin has about half the 3060's GPU (2,048 vs 3,840 CUDA cores); Orin Nano about a quarter of that.

| device | assumed slowdown vs RTX 3060 | everything on every step | with asynchronous matching (as on the CPU) |
|---|---|---|---|
| Jetson AGX Orin | 1.5–2.5× | ~15–25 Hz | position output ≫ 30 Hz, fix every 0.6–1 m at 15 m/s |
| Jetson Orin Nano | 4–8× | ~5–10 Hz | fix every 1.5–3 m at 15 m/s |

Either way, the fix spacing stays far below the 50 m at which the CPU sweep already reached full accuracy.
