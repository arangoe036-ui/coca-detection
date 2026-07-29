# Coca Detection + Hotspot Map (Colombia)

A two-part system that maps where coca is grown in Colombia from **free satellite
imagery** (Sentinel-2 + Sentinel-1), validated against Colombia's official coca
statistics:

1. **Engine** — an ML model (U-Net, satellite-pretrained encoder) that predicts coca
   **density/presence** per pixel.
2. **Map UI** — a Leaflet/MapLibre web map showing coca hotspots by municipality with a
   year slider; zooming into a hotspot streams a high-res satellite basemap for human
   visual inspection.

> **Framing.** This is a **monitoring & analysis** tool measuring *extent and change*.
> Outputs are **aggregate** (density / municipal hotspots), **not** an enforcement
> "target list." A small amount of coca is legally cultivated by some Indigenous
> communities. Visual confirmation is done by a human reviewing public basemap imagery
> ("algorithm flags, analyst verifies").

See [`coca-detection-build-plan.md`](../coca-detection-build-plan.md) for the full plan
(the single source of truth). This README covers setup and status.

## Status

- [x] **P0 — Scaffold** (repo structure, env, `config/default.yaml`, `.gitignore`, README).
- [x] **P1 — Data** (proven end-to-end on a smoke slice). P1a ✅ imagery (Planetary Computer, 18-band GeoTIFF). P1b ✅ labels (Socrata `v3rx-q7t3` density grid → aligned mask; `acs4-3wgp` validation table). P1c ✅ tiling + spatial block split. *Next: scale to full AOI / 4 seasons for a real split (time-costly — confirm first).*
- [x] **P2 — Baseline model.** U-Net (18-ch). Two tasks via `model.task`: **regression** (coca fraction → calibrated hectares, current default) and segmentation (binary presence). Trained on the real geographic split. *Remaining lever: geo-pretrained encoder weights.*
- [x] **P3 — Evaluation.** Current density-regression model (`final_multiyear.pt`), held-out test blocks, **cell-level** presence-IoU **0.474** / F1 **0.636** (thr 0.02, per-year mean across 2019–2024). Labels are ~1 km cells burned uniformly (`src/data/labels.py:107`), so this is agreement with which **1 km cells** contain coca, **not** field-level localization. *(The earlier `IoU 0.665 / F1 0.799 / AP 0.877, thr 0.504` were the superseded single-year-2023 **binary-segmentation** metrics — a different task and threshold. See [`docs/BASELINE_LADDER_RESULTS.md`](docs/BASELINE_LADDER_RESULTS.md) for the full baseline-ladder comparison.)*
- [x] **P4 — Inference & outputs.** Density raster + municipal choropleth (GeoJSON/CSV) + footprint polygons + density-map PNG. **Calibrated area: predicted 46,843 ha vs official 39,815 ha (1.18×). Per-municipality: Tibú 0.98, El Tarra 0.95, Teorama 1.05 of official.** (Binary-presence baseline was 9.2× — density regression fixed the calibration.)
- [x] **P5 — Map UI.** Leaflet console (`ui/`): municipal choropleth (YlOrBr) over CARTO light + Esri imagery crossfade, density COG overlay with opacity/threshold sliders, draw-a-box→hectares, dark side panel (totals, ranked hotspots with fly-to, legend, honest caveats). Serve with `python -m http.server` from `ui/`.

### v2 (engine fixes + interactive map)
- [x] **A — Feathered inference** (Hann-window blending; seams gone) + **presence gating & recalibration** (haze removed 100%→29%; calibrated 39,815 ha).
- [x] **B — Interactive map** (COG density overlay, sliders, draw-to-sum, year selector + nowcast badge).
- [x] **C — Generalization test** (2022 = 1.02 ✅, 2024 = 0.71 ❌ growth-year undercount → triggered v2.1).

### v2.1 (multi-year retrain + LOYO) — final
- [x] **Multi-year model** (2019–2024 pooled), **per-year normalization**, **frozen calibration**, **6-fold leave-one-year-out** validation. See [`docs/v2.1_loyo_results.md`](docs/v2.1_loyo_results.md).
- **Verdict (accepted ceiling):** LOYO out-of-year ratios span **0.59–1.48 (mean 0.95, ±0.27)** — unbiased but wide. **Location & municipal ranking are reliable; within-year calibrated totals are reliable; absolute hectares for a censusless year carry a ~±40% band.** 2026 can only be an *exploratory* nowcast, never a validated figure.

**Not pursued** (would need paid data or large compute for uncertain gain): Phase-5 temporal U-TAE, geo-pretrained encoder, plot-level detection (paid sub-meter imagery + hand labels, ~$15–60k).

## Setup

Requires Python ≥ 3.11. Using [`uv`](https://docs.astral.sh/uv/):

```bash
cd coca-detection
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt        # heavy: torch, torchgeo, earthengine-api, rasterio…
# torch may need a platform-specific index: https://pytorch.org/get-started/locally/
```

Verify the config loads (the P0 smoke test — needs only PyYAML):

```bash
python -m src.utils
```

Imagery comes from the **Microsoft Planetary Computer** STAC API (free, no GEE, no
signup). Run the P1a smoke export (small AOI, one season):

```bash
python -m src.data.stac_export --quick     # -> data/imagery/catatumbo_2023_s0_quick.tif
```

## Layout

```
config/default.yaml   # all tunables (region, year, bands, model, training, paths)
src/data/             # gee_export, labels, tiling, dataset, masks
src/models/           # unet, temporal (stretch), losses
src/{train,evaluate,infer}.py
ui/                   # Phase 5 Leaflet/MapLibre app
notebooks/            # 01 explore GEE · 02 inspect tiles · 03 inspect predictions
data/  outputs/  ui/data/   # gitignored — imagery/labels/tiles/rasters/checkpoints
```

## Guardrails

- **Free data only** for the core build (Sentinel-2 + Sentinel-1 via GEE).
- **Spatial** train/val/test split — never a random pixel split.
- Class imbalance → **Dice+Focal** loss, oversample positive tiles, evaluate with IoU/F1.
- **Smallest end-to-end slice first**; prove each phase's acceptance check before scaling.
- **Never commit** imagery / labels / outputs / checkpoints.

## Before running P1+ — confirm (plan §13)

Region, target year, GEE auth, ODC layer export path, basemap choice, and Phase-6 scope.
These are unresolved; `config/default.yaml` holds **placeholder** defaults marked `# TBD`.
```
