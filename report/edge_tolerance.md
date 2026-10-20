# Edge tolerance: accuracy vs matcher, hardware and speed

Particle filter over the 20 test flights (2022 imagery, realistic camera views), edge settings: 1,000 particles, fused Numba kernel, networks every 10th step, dead-reckoning bias estimation.

## Accuracy vs fix spacing

| matcher | E-core latency | fix every 50 m | 100 m | 200 m | 400 m |
|---|---|---|---|---|---|
| LoFTR, full resolution | 1220 ms | 1.4 m / 99% | 1.6 m / 96% | 2.0 m / 94% | 2.8 m / 86% |
| LoFTR, 0.75x resolution | 680 ms | 1.6 m / 98% | 1.8 m / 95% | 2.4 m / 88% | 3.1 m / 80% |
| LoFTR, 0.5x resolution | 310 ms | 2.2 m / 89% | 2.4 m / 84% | 3.3 m / 77% | 5.0 m / 66% |
| Learned DSMAC (ours, 111k params) | 60 ms | 3.7 m / 76% | 4.3 m / 72% | 5.9 m / 66% | 10.8 m / 48% |

Cells: median error / share of steps within 10 m.

No image matching at all (magnetometer + gravity only): median 37.4 m, 4% within 10 m.

## What each hardware/speed combination gets

| hardware | matcher | 15 m/s | 30 m/s |
|---|---|---|---|
| 4 E-cores (measured) | LoFTR, full resolution | fix every 18 m → 1.4 m / 99% | fix every 37 m → 1.4 m / 99% |
| 4 E-cores (measured) | LoFTR, 0.75x resolution | fix every 10 m → 1.6 m / 98% | fix every 20 m → 1.6 m / 98% |
| 4 E-cores (measured) | LoFTR, 0.5x resolution | fix every 5 m → 2.2 m / 89% | fix every 9 m → 2.2 m / 89% |
| 4 E-cores (measured) | Learned DSMAC (ours, 111k params) | fix every 1 m → 3.7 m / 76% | fix every 2 m → 3.7 m / 76% |
| Raspberry Pi 5 (est. 2x slower) | LoFTR, full resolution | fix every 37 m → 1.4 m / 99% | fix every 73 m → 1.6 m / 96% |
| Raspberry Pi 5 (est. 2x slower) | LoFTR, 0.75x resolution | fix every 20 m → 1.6 m / 98% | fix every 41 m → 1.6 m / 98% |
| Raspberry Pi 5 (est. 2x slower) | LoFTR, 0.5x resolution | fix every 9 m → 2.2 m / 89% | fix every 19 m → 2.2 m / 89% |
| Raspberry Pi 5 (est. 2x slower) | Learned DSMAC (ours, 111k params) | fix every 2 m → 3.7 m / 76% | fix every 4 m → 3.7 m / 76% |
| Raspberry Pi 4 (est. 4x slower) | LoFTR, full resolution | fix every 73 m → 1.6 m / 96% | fix every 146 m → 2.0 m / 94% |
| Raspberry Pi 4 (est. 4x slower) | LoFTR, 0.75x resolution | fix every 41 m → 1.6 m / 98% | fix every 82 m → 1.8 m / 95% |
| Raspberry Pi 4 (est. 4x slower) | LoFTR, 0.5x resolution | fix every 19 m → 2.2 m / 89% | fix every 37 m → 2.2 m / 89% |
| Raspberry Pi 4 (est. 4x slower) | Learned DSMAC (ours, 111k params) | fix every 4 m → 3.7 m / 76% | fix every 7 m → 3.7 m / 76% |
