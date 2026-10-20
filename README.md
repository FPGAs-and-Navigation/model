# Multimodal GPS-denied UAV localization (BTP)

This is the model from the midterm deck (slide 6), trained on the Mountain Pass, CA survey. It takes three inputs:
- an aerial image,
- a magnetometer history,
- a gravity-gradiometer history.

It outputs the UAV's latitude and longitude.

```
image ──ViT (DINOv2 ViT-S/14)──► (μ_v, Σ_v) ┐
mag profile ──1D-CNN─────────► (μ_m, Σ_m) ├─► UDML: ρ (uncertainty), α (dependency),
grav profile ──1D-CNN────────► (μ_g, Σ_g) ┘   w = softmax(log α − log ρ) ─► Σ w_k z_k ─► FC head ─► lat, lon
```

## Coarse-to-fine pipeline (`btp/c2f.py`)
Regressing lat/lon directly levels off at hundreds of metres, so the full system works in three steps:
1. **Coarse (learned, UDML):** the network gives a probability for every 250 m map cell (`model.head: classify`).
2. **Profile matching (TERCOM):** the measured 1.6 km magnetic/gravity profiles are matched against the maps.
   - It searches every position in the top-k cells, at a few compass headings, allowing for dead-reckoning drift.
   - The match likelihood is combined with the coarse probabilities. In effect, vision proposes places and geophysics decides among them.
3. **Image matching (DSMAC):** the aerial image is correlated against the onboard reference map (NAIP 2020) around the best hypotheses.
   - The images are high-pass filtered and compared with normalised cross-correlation.
   - A match is accepted only if its peak stands out enough (peak-to-sidelobe ratio).

Test numbers on 300 windows with 2022 imagery, using a mid-training snapshot of the vision regressor (epoch 9) as the coarse prior:

| stage | median | <25 m | <100 m |
|---|---|---|---|
| coarse guess (vision) | 629 m | 0 % | 2 % |
| + mag/grav profile matching | 123 m | 7 % | 43 % |
| + image matching | **9.9 m** | 55 % | 58 % |
| image matching, given a hypothesis near the truth | 3.8 m | 77 % within 10 m | |

The remaining error is the tail where the coarse step misses, which is what the cell classifier is for.

## Phase 2: the combined system (`btp/track.py`)
The slide-6 model stays at the core, and three published ideas are wrapped around it:
- **Learned image matching (`btp/matchers.py`):** LoFTR replaces NCC.
  - At the true position with realistic camera views (±8° heading, ±7% altitude, ±10° tilt), LoFTR is 95% within 10 m; NCC is 30%.
- **Retrieval (`btp/retrieval.py`, `btp/train_retrieval.py`):** the slide-6 ViT plus a projection head embeds the 2020 reference map.
  - It is fine-tuned Sample4Geo-style, with neighbourhood hard negatives.
  - Its prior multiplies the UDML classifier's prior.
- **Particle filter over the flight:** at every step, particles move by dead reckoning.
  - They are weighted by TERCOM profile matching (scaled by the UDML weights w_m and w_g).
  - A confident LoFTR fix re-weights them (scaled by w_v).
  - A robust coarse-prior term and re-injection let a lost filter recover.

`scripts/phase2.sh` runs every evaluation and ablation, then writes `runs/phase2/report.md`.

## Phase 3: fixes from the literature (`scripts/phase3.sh`, off by default)
- **Balanced training against vision dominance:** `train.ogm_alpha`, OGM from Peng et al. (CVPR 2022), plus a per-modality `train.mod_dropout` dict.
- **SeqSLAM-style sequence prior:** `coarse.seq_len` averages the motion-compensated coarse priors of the last k frames.
- **Instance colour alignment to the onboard map:** `img.color_align: instance`.
- **Particles estimate the dead-reckoning heading/scale bias:** `track.estimate_bias`, the augmented-state idea from marginalised-PF magnetic navigation.

## Phase 4: running on cheap hardware (`btp/edge/`, `scripts/edge_sweep.sh`)
- **Target:** a Raspberry Pi-class CPU, emulated on 4 E-cores of an i7-12700H.
- **Design:** position outputs come from a fast loop (dead reckoning plus the magnetometer/gravity particle update, with INT8 ONNX networks every 10th step). That loop runs at about 45–49 Hz on 4 E-cores. Image matching runs in the background, and accuracy depends on how many metres pass between fixes.
- **Optimisations:**
  - fused Numba likelihood kernel (50× faster, identical numbers),
  - ONNX Runtime with INT8 for the ViTs (1.8× faster, 4× smaller, same top-10 cells),
  - non-spinning thread pools so PyTorch, Numba and ONNX Runtime can share 4 cores,
  - 1,000 particles,
  - networks only every 10th step.
- **Findings:** see `report/edge_tolerance.md` for the accuracy/latency table.
  - Full-resolution LoFTR (1.2 s per match on the E-cores) still gives 1.4 m and 99% within 10 m, because one fix per 50 m of flight is enough.
  - Slower hardware or faster flight degrades gracefully: with a fix every 400 m, it's 2.8 m and 86%.
  - Our 111k-parameter learned DSMAC is 20× faster but less accurate across imagery years (3.7 m, 76%).

## Phase 5: GPU / Jetson path (`btp/edge/triton_kernels.py`, `btp/edge/trt.py`, `btp/edge/loftr_trt.py`)
- **Particle update:** a fused Triton kernel (4–11× faster than PyTorch on the GPU, same numbers).
- **Networks:** TensorRT FP16 engines (1.2–1.6 ms; built from FP16 ONNX exported from half-precision models, since TensorRT 11 is strongly typed).
- **LoFTR:** its backbone runs as a TensorRT engine, and the rest of LoFTR in FP16.
- **Result:** on a laptop RTX 3060, the full pipeline (everything every step) runs at 38.7 Hz with the desktop system's accuracy (1.4 m, 98% within 10 m). See `report/gpu_results.md`.
- **Running it:** `.venv-gpu/bin/python -m btp.track data/edge/coarse.pt --retrieval data/edge/retrieval.pt --backend trt --set dsmac.method=loftr_trt track.kernel=triton track.particles=1000 track.estimate_bias=true`

## Data
- **Imagery:** NAIP 0.6 m imagery for 2016, 2018, 2020 and 2022. It is resampled to 1.25 m/px; one crop is 224 px, about 280 m across.
- **Geophysics:** the USGS 2016 HeliFALCON survey (DOI 10.5066/P9SQV3SB):
  - gravity `gD_Fourier_0` and `GDD_Fourier_0` (no terrain correction, i.e. what an aircraft senses),
  - magnetic `MAG_RMI`.

All data is put in one UTM 11N (EPSG:26911) frame.

**Honest evaluation:**
- **Held-out flight lines.** Of every 8 adjacent traverse lines, 1 is used for validation and 1 for test. The magnetic and gravity *training maps are gridded without them*. Validation and test use the *raw readings* of those lines.
- **Held-out imagery year.** The test images come from 2022, while training uses 2016, 2018 and 2020.
- **Training samples** are generated on the fly at random continuous positions. Each gets a synthetic 1.6 km track over the training maps, plus sensor noise, offsets, image colour/scale/rotation augmentation and modality dropout.

## Layout
| path | what |
|---|---|
| `btp/prepare_geophysics.py` | line split, training grids, held-out sensor windows, QC plot |
| `btp/prepare_imagery.py` | NAIP mosaic per year (memmap) and the valid-crop mask |
| `btp/dataset.py` | training sampler and held-out windows |
| `btp/models/` | encoders, UDML core, the localizer and the loss |
| `btp/train.py` | training: AdamW, layer-wise LR decay on the ViT, bf16, TensorBoard |
| `btp/evaluate.py` | test metrics, per-modality errors, ablations, weight response to noise |
| `btp/simulate.py` | flies a held-out line and plots the true vs predicted track |
| `btp/c2f.py` | coarse → TERCOM → DSMAC pipeline with per-stage errors |
| `btp/matchers.py` | NCC / LoFTR / LightGlue matchers, camera-view simulation, benchmark |
| `btp/retrieval.py`, `btp/train_retrieval.py` | reference-map retrieval (zero-shot and fine-tuned) |
| `btp/track.py` | particle filter over held-out flights |
| `btp/phase2_report.py` | collects everything into `runs/phase2/report.md` |
| `configs/base.yaml` | all settings; override with `--set key.sub=value` |
| `scripts/` | `deploy.sh` (here → kratos), `sync_data.sh` (CoW data copy), `train_staged.sh` |

## Running (on kratos, in `~/BTP_impl`)
```bash
scripts/sync_data.sh                      # reflink-copy ~/BTP/data → data/raw (no extra disk, read-only)
.venv/bin/python -m btp.prepare_geophysics
.venv/bin/python -m btp.prepare_imagery   # every year whose download has finished
.venv/bin/python -m btp.train --smoke     # 50-step sanity check
scripts/autorun.sh --install              # cron-driven, resumes after reboots (see the script header)
scripts/train_staged.sh                   # or directly: vision → +mag → +grav → coarse, each evaluated, simulated, run through c2f
tensorboard --logdir runs
```
Outputs go to `runs/<name>/`:
- `best.pt`
- `best_val.json`
- `test_metrics.json`
- `test_predictions.csv` (lat/lon)
- `sim_line*.png`

The code is edited in this repo and pushed to kratos with `scripts/deploy.sh`. Data, runs and the venv live only on kratos.

To set up the environment:
```bash
uv venv -p 3.12 .venv
uv pip install -p .venv/bin/python torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install -p .venv/bin/python -e .
```
