# Coca Detection + Hotspot Map (Colombia)

A two-part system that maps where coca is grown in Colombia from **free satellite
imagery** (Sentinel-2 + Sentinel-1), validated against Colombia's official coca
statistics:

1. **Engine** — an ML model (U-Net with a ResNet-34 encoder) that predicts coca
   **density/presence** per pixel.
2. **Map UI** — a Leaflet/MapLibre web map showing coca hotspots by municipality with a
   year slider; zooming into a hotspot streams a high-res satellite basemap for human
   visual inspection.

> **Framing.** This is a **monitoring & analysis** tool measuring *extent and change*.
> Outputs are **aggregate** (density / municipal hotspots), **not** an enforcement
> "target list." A small amount of coca is legally cultivated by some Indigenous
> communities. Visual confirmation is done by a human reviewing public basemap imagery
> ("algorithm flags, analyst verifies").

**Resolution floor (harm reduction).** Public artifacts are **75 m blurred rasters and
municipality-level aggregates only**. A plot-level 20 m footprint file (2,065 polygons) exists
locally, is gitignored, and has **never appeared in any commit** — verified across full history.
It is targeting-usable, so the resolution gap is a deliberate choice rather than a limitation.

Planning docs live in [`docs/`](docs/) — [`START_HERE.md`](START_HERE.md) for current state,
[`docs/BASELINE_PREREG.md`](docs/BASELINE_PREREG.md) for the pre-registered decision rules.
*(This previously pointed at `../coca-detection-build-plan.md` as "the single source of truth" — a
parent-directory path that has never been in the repo and never can be. Corrected 2026-08-10.)*

## Result (2026-08-18) — measured on gen4, both nulls won

Two pre-registered no-skill nulls were run against this system. **Both beat the model**, and
each was only discovered because it was actually computed:

| question | model | no-model null | verdict |
|---|--:|--:|---|
| **How much** coca? (Track B, A12) | — | historical mean of past censuses | null wins; counting **out of scope** |
| **Where** is coca? (Track A, A16) | presence-IoU **0.725** | previous census carried forward: **0.931** | null wins **6/6**; not publishable as a model result |

- **Against imagery baselines the U-Net wins 6/6** — 0.725 mean presence-IoU vs a context-free
  random forest 0.346 and an NDVI threshold 0.262. Spatial context genuinely helps *when imagery
  is all you have*.
- **Against the previous census it loses 0/6**, by 0.163–0.289 IoU. Coca is a perennial and the
  labels are ~1 km census cells, so the set of occupied cells barely moves: "it is where it was"
  already scores 0.907–0.955 and there is almost no headroom above it. The rule was frozen in
  advance (prereg A16/A19) and the U-Net was **not** retuned in response.
- The floor is **label-informed** — it reuses official census cells and opens no satellite image.
  So this is not "the model cannot see coca"; it is that **free 20 m imagery adds nothing over
  reusing the last census at the granularity the census publishes.** That is the right test,
  because between censuses you always *have* the last census.

Every figure above traces to a row in `outputs/metrics/baseline_ladder.jsonl`
(`data_generation: gen4`, 42 rows). Fold-by-fold tables, the symmetry audit and what survives as
a claim: [`docs/BASELINE_LADDER_RESULTS.md`](docs/BASELINE_LADDER_RESULTS.md).

**What this project is, stated honestly:** a rigorous demonstration that free satellite imagery
does not improve on Colombia's existing coca census at 1 km granularity — plus three real data
defects found along the way, two of which cancelled into a *publishable-looking* number, and a
map UI that reproduces the official spatial pattern from free data. The negative results are the
deliverable, and they are backed; the sections below are the record of how they were reached.

## ⚠ Status of the numbers below (2026-08-10)

**The P3/P4/v2.1 figures in this section were computed on contaminated imagery** and are
retained, not deleted, per the project's no-overwrite rule. ESA Processing Baseline 04.00
introduced `BOA_ADD_OFFSET = -1000` on 2022-01-25; the export did `DN/10000` without
subtracting it, so all **2022–2024 reflectances were inflated by ~0.1**, and every
leave-one-year-out fold trained on those years.

The offset is now **fixed and verified**: the +0.100 visible-band step at 2021→2022 is down
to **+0.0013** on regenerated data (`scripts/acceptance_6_6b.py`), and all six years plus
3,450 tiles have been rebuilt. ~~**The corrected re-run of these numbers is pending.**~~ **It landed 2026-08-18 on gen4 — see Result above.** Until you have read that block, treat every figure below as indicative of *direction* only, not magnitude — see
[`docs/BASELINE_LADDER_RESULTS.md`](docs/BASELINE_LADDER_RESULTS.md) for the detailed
blast-radius annotation.

**Two findings that do not depend on the corrected re-run**, both from pre-registered tests:

- **The model localizes but does not count.** It beats a context-free random forest and an
  NDVI threshold at deciding which 1 km cells contain coca (6/6 folds), but for total
  hectares in a censusless year it does **not** beat predicting the historical mean of past
  official counts.
- **Why counting fails is now measured, not guessed.** Coca is ~3.6% of this AOI and the
  entire 2020→2024 swing is 0.8% of its area, putting its contribution to any AOI-wide
  statistic **20–30× below** ordinary year-to-year weather variation. This is a
  signal-to-noise limit, not a fixable modelling choice — so the planned auxiliary-input
  retrain was **cancelled** rather than attempted. See
  [`docs/a12_level_signal.md`](docs/a12_level_signal.md).

## Status

- [x] **P0 — Scaffold** (repo structure, env, `config/default.yaml`, `.gitignore`, README).
- [x] **P1 — Data** (proven end-to-end on a smoke slice). P1a ✅ imagery (Planetary Computer, 18-band GeoTIFF). P1b ✅ labels (Socrata `v3rx-q7t3` density grid → aligned mask; `acs4-3wgp` validation table). P1c ✅ tiling + spatial block split. *Next: scale to full AOI / 4 seasons for a real split (time-costly — confirm first).*
- [x] **P2 — Baseline model.** U-Net (18-ch). Two tasks via `model.task`: **regression** (coca fraction → calibrated hectares, current default) and segmentation (binary presence). Trained on the real geographic split. **Encoder weights — corrected 2026-08-10:** the encoder is **randomly initialised**, not pretrained. `config` sets `encoder_weights: ssl4eo`, which `src/models/unet.py` routes into `geo_keys` → passes `encoder_weights=None` to `smp.Unet`, and `_load_geo_encoder_weights()` is still a **no-op that only emits a warning** (`P2 TODO`). So the config choice silently *disables* the ImageNet weights the model would otherwise receive. Two untaken levers, cheapest first: set `encoder_weights: imagenet` for real pretrained weights, or wire genuine geo-pretrained weights (SSL4EO/Prithvi/Clay).
- [x] **P3 — Evaluation. RE-MEASURED 2026-08-18 on gen4** (see Result at the top: U-Net 0.725 IoU, beats RF/NDVI 6/6, loses to the persistence floor 0/6). The retraction below stands as written — the numbers it withdraws were never adjusted, they were replaced by a fresh run on rebuilt data. This line
  previously reported cell-level presence-IoU **0.474** / F1 **0.636**. Those came from folds that
  (a) trained on offset-inflated 2022–2024, (b) used composites with 25.3% of the 2019/2020 AOI
  silently blank, and (c) used a tile→block assignment that put 13.7% of test pixels into training.
  The metrics artifact they cite no longer exists, so they cannot even be inspected. **Treat 0.474
  as deleted, not as a number awaiting adjustment.**
  What survives is the *scope* statement, which was always correct: labels are ~1 km cells burned
  uniformly (`src/data/labels.py:107`), so any metric here is agreement with **which ~1 km cells**
  contain coca — never field-level localization. Re-measurement is pre-registered
  ([`docs/BASELINE_PREREG.md`](docs/BASELINE_PREREG.md)) and now includes a **persistence null**
  (A16) that the original comparison never had: coca is a perennial, so "it is where it was last
  year" must be beaten before spatial skill can be claimed at all.
- [x] **P4 — Inference & outputs.** Density raster + municipal choropleth (GeoJSON/CSV) + density-map PNG.
  **Figures removed 2026-08-10, not caveated.** This line previously published "predicted 46,843 ha
  vs official 39,815 ha (1.18×)" and per-municipality ratios "Tibú 0.98, El Tarra 0.95, Teorama 1.05".
  Those numbers appear in **no artifact and no other document** — the run that produced them was
  deleted — and the three ratios contradict the tracked `ui/data/municipal_coca_2023.csv`, which
  gives Tibú 0.963, El Tarra 0.974 and **Teorama 0.895** (flipping it from over- to
  under-predicting). The "46,843" also mixed two different official totals: the municipal table
  sums to 43,058 while 39,815 is the grid total. Publishing three municipalities also violated
  this project's own rule (`START_HERE.md`: *publish all 10, not the best 3*) — the real range is
  **0.05 to 1.69**. Hectare reporting is now out of scope entirely; see Status above.
- [x] **P5 — Map UI.** Leaflet console (`ui/`): municipal choropleth (YlOrBr) over CARTO light + Esri imagery crossfade, density COG overlay with opacity/threshold sliders, draw-a-box→hectares, dark side panel (totals, ranked hotspots with fly-to, legend, honest caveats). Serve with `python -m http.server` from `ui/`.

### v2 (engine fixes + interactive map)
- [x] **A — Feathered inference** (Hann-window blending; seams gone) + **presence gating & recalibration** (haze removed 100%→29%; calibrated 39,815 ha).
- [x] **B — Interactive map** (COG density overlay, sliders, draw-to-sum, year selector + nowcast badge).
- [x] **C — Generalization test** (2022 = 1.02 ✅, 2024 = 0.71 ❌ growth-year undercount → triggered v2.1).

### v2.1 (multi-year retrain + LOYO) — final
- [x] **Multi-year model** (2019–2024 pooled), **per-year normalization**, **frozen calibration**, **6-fold leave-one-year-out** validation. See [`docs/v2.1_loyo_results.md`](docs/v2.1_loyo_results.md).
- **Verdict RETRACTED 2026-08-10.** This previously read: *"LOYO out-of-year ratios span 0.59–1.48
  (mean 0.95, ±0.27) — unbiased but wide … within-year calibrated totals are reliable."* That
  spread is now known to be substantially **two data defects cancelling**, not model variance: the
  offset bug inflated 2022–2024 while the blank-coverage bug suppressed 2019–2021, and the two
  year-groups are exactly complementary. "Unbiased but wide" was the *symptom* of the corruption,
  which is why it looked publishable. Also, "within-year calibrated totals are reliable" describes
  a circular fit — the scalar is fitted to the same year's census (`src/infer.py:218`), so the 2023
  total matching official is arithmetic, not evidence. Hectares are now out of scope.

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
data/  outputs/            # gitignored — imagery/labels/tiles/rasters/checkpoints
ui/data/                   # NOT gitignored: the small municipal aggregates + 75 m COGs
                           # the map needs are tracked, by explicit filename allow-list
```

## Guardrails

- **Free data only** for the core build (Sentinel-2 L2A + Sentinel-1 RTC via the Microsoft
  Planetary Computer STAC API — no Google Earth Engine, no signup).
- **Spatial** train/val/test split — never a random pixel split.
- Class imbalance → **Dice+Focal** loss, oversample positive tiles, evaluate with IoU/F1.
- **Smallest end-to-end slice first**; prove each phase's acceptance check before scaling.
- **Never commit** imagery / labels / model outputs / checkpoints. The exception is `ui/data/`,
  which IS tracked — but by an explicit filename allow-list in `.gitignore`, not by extension.
  It previously whitelisted `*.geojson` and `*_cog.tif`, which could not distinguish municipal
  polygons from field polygons or 75 m from 20 m; the never-publish artifact is itself a
  `.geojson`. Adding any new artifact there must be deliberate (`git add -f`).

## Configuration status

*(This section previously said the region, auth, and export paths were "unresolved
placeholders marked `# TBD`". That was stale P0-era text — corrected 2026-08-10.)*

The pipeline is **configured and running end to end**. Region is Catatumbo (Norte de
Santander), `utm_epsg: 32618`, 20 m grid, 18 channels, 2019–2024. Imagery needs **no
credentials** — Planetary Computer signs asset URLs anonymously. Labels come from the public
Socrata endpoint on `datos.gov.co` (no API key).

Residual `# TBD` markers in `config/default.yaml` refer to *untaken options*, not missing
setup: an alternative AOI (`tumaco`) and the encoder-weights lever noted under P2.

**Rebuild order** — labels → imagery → tiles → **split** → train → evaluate:

```powershell
python -m src.data.stac_export --full --year 2019   # per year; ~12 sub-tiles each
# re-tile. Writes the SUPERSEDED band split, so it refuses to clobber a gen4 index
# without the flag, and archives the gen4 pair before replacing it.
python -m src.data.multiyear --years 2019 2020 2021 2022 2023 2024 --force-band-split-index
# TERMINAL STEP, not optional: assign the A20 stratified macro-block folds and verify them.
python -m src.data.multiyear --rebuild-block-folds
```

**The second command's output is not usable on its own.** It assigns splits with the gen3
contiguous-band rule, whose test fold on this AOI holds **zero coca pixels** — every
`presence_iou` on it is `0/0` printed as `0.000` — while the config still stamps `gen4` on every
metrics row. `--rebuild-block-folds` is what produces a split that is both leak-free and
signal-bearing, and it is the only path that runs the verification (prereg **A20**). If you
already have tiles on disk you need only that last command; it reassigns the existing tiles and
does not re-tile.

Verified on Windows 2026-08-10: 6 annual mosaics (~8.5 GB) and 3,450 tiles (9.35 GB), with
the label totals reproducing the official census exactly for all six years and the
block→split assignment identical across years (0 inconsistencies). The gen4 split (2026-08-14)
keeps 436 of the 575 tile positions — 292 train / 64 val / 80 test, 2,616 rows over six years.
```
