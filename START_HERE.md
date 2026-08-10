# Start here — picking this up on another machine

## What this project is

Maps illegal coca cultivation in Colombia's **Catatumbo** region from **free** satellite imagery
(Sentinel-1 RTC + Sentinel-2 L2A, via Microsoft Planetary Computer — no signup), and validates
predictions against **Colombia's official government coca census**. A U-Net predicts coca density
per 20 m pixel; a Leaflet map shows municipal hotspots and year-over-year change.

The distinguishing feature is the validation: most ML projects check against a random split of
their own labels. This checks against an independent government survey.

## ⚠ Honest current state — read before trusting any number

**The project is mid-debug. Published figures in `README.md` are contaminated.**

What is **established**:
- **Spatially the U-Net wins**: presence-IoU **0.474** vs a context-free random forest 0.260 vs
  an NDVI threshold 0.169, 6/6 folds. Spatial context genuinely helps.
- **At counting it loses**: `aoi_ratio` mean 0.95, std 0.271. It fails a pre-registered ≥5/6 bar
  against **N2**, a historical-mean null that never opens a satellite image.
- **Coca is detected by disturbance, not greenness**: the SWIR/burn-ratio complex dominates
  (NBR, B12, B11 — one physical signal), while **NDVI ranks 15/18**.

What was **found and fixed**:
- **A real data bug.** ESA Processing Baseline 04.00 (2022-01-25) introduced
  `BOA_ADD_OFFSET = -1000`; the code did `DN/10000` without subtracting it, so **all 2022–2024
  Sentinel-2 reflectances were inflated by ~0.1**. Every visible band steps +0.100 at exactly
  2021→2022 while SAR stays flat. Indices are computed *before* normalization, so per-year
  z-scoring could not undo it.
- **Fixing it does not recover counting.** 2022 went 1.48 → 1.75 (still over-predicting).

What was **rejected**:
- "The bad years were the cloudy years" — **false**. 2020, the second-worst fold, has the *best*
  coverage of all six years on both sensors. This killed a planned 30-hour sensor-addition effort
  before it was built.

**Consequence: every LOYO fold trained on contaminated 2022–2024 data, so all published numbers
need a clean re-run.** `docs/BASELINE_LADDER_RESULTS.md` is annotated, not overwritten.

## Scope limits that must survive into any write-up

- **Labels are ~1 km and burned uniformly** (`labels.py:107-115` writes one constant density into
  every 20 m pixel of a cell). So **IoU 0.474 is agreement with 1 km *cells*, not with fields.**
  Nothing here can validate field-level detail.
- **The 2023 calibration is partly circular** — the global scalar was fit on 2023, so the 2023
  total matching official is partly by construction. The per-municipality *distribution* is still
  real evidence.
- **Publish all 10 municipalities, not the best 3.** The full range is **0.05 to 1.69**. Large
  municipalities are accurate; small ones are not.

## 🔒 Never publish

`outputs/catatumbo_2023_coca.geojson` — **2,065 plot-level 20 m field polygons. Targeting-usable.**
It is gitignored (`/outputs/`) and has never been committed. Keep it that way; never `git add -f`.

Public artifacts stay at **75 m** blurred COGs + **municipality-level** aggregates (10 features).
That resolution gap is a deliberate harm-reduction choice, not an accident. Framing stays
monitoring/statistics — **not** an enforcement target list.

## Windows: yes, this transfers

I checked. **No code changes needed.**

- No hardcoded POSIX paths in any tracked file
- No unix-only shell calls
- 23 files use `pathlib`, which is portable by design

The only friction is the geospatial stack. **Use conda-forge, not pip** — `rasterio`, `GDAL`,
`geopandas`, `fiona`, and `pyproj` are painful to pip-install on Windows and trivial via conda.

```powershell
conda create -n coca -c conda-forge python=3.11 rasterio geopandas shapely pyproj fiona
conda activate coca
pip install torch --index-url https://download.pytorch.org/whl/cu121   # CUDA build
pip install -r requirements.txt
```

**A Windows box with an NVIDIA GPU will train faster than the Mac did** — CUDA beats MPS here,
so this is an upgrade, not a compromise.

## The 24 GB of data is NOT in the repo — and doesn't need to be

The repo is ~7.6 MB across 74 files. Everything heavy regenerates from free sources:

| What | Size | How to rebuild |
|---|---|---|
| Sentinel composites | ~11 GB | `python -m src.data.stac_export` (Planetary Computer, free, no signup) |
| Training tiles | ~11 GB | `python -m src.data.tiling` |
| Labels | small | Socrata API on `datos.gov.co` (public, no key) |
| Model checkpoints | ~190 MB | retrain, ~minutes per fold |

**Rebuild order:** labels → imagery → tiles → train → evaluate. Each stage is a
`python -m src.<module>` entrypoint with argparse and documented acceptance criteria.

⚠ **When you regenerate, apply the Sentinel-2 offset fix.** Subtract 1000 from post-2022-01-25
scenes **before** computing indices, and key it on the **processing-baseline metadata**, not the
acquisition date — 2022 is a mixed year, and scenes acquired earlier but reprocessed later carry
the offset too.

## Where to pick up

`docs/PHASE6_9_MASTER_PLAN.md` is the live plan. Next four steps, cheapest first:

1. **Gate/quantile re-test** on the saved fixed-2022 fold — free, ~6 min. `gate_threshold: 0.05`
   is a *fixed absolute* cut applied to per-year-normalized outputs, so it removes a different
   fraction of predicted mass every year. Likely a second latent bug.
2. **Re-run the RF + NDVI baselines** on corrected data — nearly free, no GPU. Resolves whether
   "NDVI is the floor" was partly the offset bug.
3. **Retrain the multiyear model + clean Track A** — one training run. Buys back the *positive*
   spatial result. Track A never needed LOYO.
4. **Track B 6-fold LOYO** — expensive; grind opportunistically and report honest n.

**Expected landing:** the offset was a genuine correctness fix that doesn't recover counting →
route counting to a **hybrid anchor** (historical mean sets the total, U-Net distributes it
spatially), which also removes the 2023 circularity. The defensible claim becomes:

> Reliably shows **where** coca is and which municipalities rank highest. Does **not** claim to
> count total hectares for a censusless year — a historical average does that as well, and here
> is the measured reason why.

## Repo layout

```
src/data/      STAC export, labels, tiling, multiyear composites
src/models/    U-Net (resnet34 encoder), losses; temporal model is an unbuilt stretch goal
src/train.py   single-split training
src/train_loyo.py   6-fold leave-one-year-out + frozen calibration  ← the important one
src/evaluate.py     metrics
ui/            Leaflet map (vanilla JS, no build step)
docs/          plans, prereg, results — PHASE6_9_MASTER_PLAN.md is live
```

**Data facts worth not rediscovering:** 18 channels stacked as **s2(10) + indices(6) + s1(2)** —
*not* the YAML declaration order. NDVI is channel **10**, VV **16**, VH **17**. Tiles are
`(18, 256, 256)` float32 with a `(256, 256)` float32 density mask. Blocks are 10 km and split
consistently across years (verified: 0 inconsistencies), which is what makes the LOYO folds clean.
