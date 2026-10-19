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
| `configs/base.yaml` | all settings; override with `--set key.sub=value` |
| `scripts/` | `deploy.sh` (here → kratos), `sync_data.sh` (CoW data copy), `train_staged.sh` |

## Running (on kratos, in `~/BTP_impl`)
```bash
scripts/sync_data.sh                      # reflink-copy ~/BTP/data → data/raw (no extra disk, read-only)
.venv/bin/python -m btp.prepare_geophysics
.venv/bin/python -m btp.prepare_imagery   # every year whose download has finished
.venv/bin/python -m btp.train --smoke     # 50-step sanity check
scripts/train_staged.sh                   # vision → +mag → +grav, each evaluated and simulated
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
