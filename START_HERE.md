# Start here — picking this up on another machine

## What this project is

Maps illegal coca cultivation in Colombia's **Catatumbo** region from **free** satellite imagery
(Sentinel-1 RTC + Sentinel-2 L2A, via Microsoft Planetary Computer — no signup), and validates
predictions against **Colombia's official government coca census**. A U-Net predicts coca density
per 20 m pixel; a Leaflet map shows municipal hotspots and year-over-year change.

The distinguishing feature is the validation: most ML projects check against a random split of
their own labels. This checks against an independent government survey.

## ⚠ Honest current state — read before trusting any number

> ### UPDATE 2026-08-10 — the data blocker is cleared
>
> - **The offset fix is implemented and verified.** Keyed on the `s2:processing_baseline`
>   metadata field, applied per scene before indices. Model-free acceptance (Phase 6.6b)
>   **PASSED**: the +0.100 visible-band step at 2021→2022 is now **+0.0013**
>   (`scripts/acceptance_6_6b.py`).
> - **All data regenerated on the corrected pipeline:** 6 annual mosaics (~8.5 GB) and
>   3,450 tiles (9.35 GB). Label totals reproduce the official census **exactly** for all
>   six years; block→split assignment is identical across years (0 inconsistencies).
> - **A12 ran and CLOSED the planned retrain.** No per-year input statistic tracks official
>   hectares. (An earlier version of this line said "best correctly-signed r = +0.355" —
>   **that is retracted**: on a fixed all-years-valid footprint B11 flips to −0.621 and B12 to
>   −0.709, so **no candidate is correctly signed at all**. The verdict is unchanged and if
>   anything firmer. See the correction block in `docs/a12_level_signal.md`.)
>   Coca is ~3.6% of the AOI and the whole
>   2020→2024 swing is 0.8% of its area, so its contribution to any AOI-wide statistic is
>   **20–30× below** interannual weather variation. The counting failure is a
>   signal-to-noise wall, not a modelling choice — so **per-year normalization is refuted
>   as the mechanism** rather than merely unproven, and both the aux-input design and the
>   mixed-normalization fallback are dead. Counting routes to the Phase 6.3 hybrid anchor.
>   See [`docs/a12_level_signal.md`](docs/a12_level_signal.md).
> - ~~**Still pending:** baseline-ladder re-run and one Track A retrain on the corrected
>   tiles.~~ **Both landed 2026-08-18 on gen4** — see the 2026-08-18 update under "Where to
>   pick up". Every metric in `outputs/metrics/baseline_ladder.jsonl` is now gen4 and backed;
>   any figure elsewhere in this file that is not marked gen4 is superseded.
>
> One caveat that cannot be discharged on this machine: §8.1 requires diffing the new
> block→split index against the old one, and `data/` was never committed, so **no old index
> exists here**. Verified instead that the assignment is deterministic *and*
> order-independent given `seed: 42`, and that all six mosaics share identical dimensions.
> "Identical to the split behind the published Phase 3 numbers" is therefore an assumption,
> not a verified fact.

**The measurement phase is over as of 2026-08-18.** `README.md` now leads with the gen4
result; the contaminated sections there are annotated, not deleted.

What is **established (gen4, backed by `baseline_ladder.jsonl`)**:
- **Against imagery baselines the U-Net wins**: presence-IoU **0.725** vs a context-free random
  forest **0.346** and an NDVI threshold **0.262**, 6/6 folds. Spatial context genuinely helps
  when imagery is all you have. *(The gen1 figures previously on this line — 0.474 / 0.260 /
  0.169 — are retracted, not adjusted: they were computed under three defects.)*
- **Against the previous census it loses 0/6.** The A16 persistence floor is **0.931** mean
  (0.907–0.955 per fold) and the U-Net is 0.163–0.289 IoU below it in every fold. The spatial
  claim is **not publishable as a model result** (frozen rule; no retuning in response).
- **At counting it loses**: it fails a pre-registered ≥5/6 bar against **N2**, a historical-mean
  null that never opens a satellite image. *(The specific gen1 numbers "mean 0.95, std 0.271"
  are retracted — that spread was substantially the F1/F2 defects cancelling.)*
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
  every 20 m pixel of a cell). So **every IoU here — 0.725 for the U-Net, 0.931 for the
  persistence floor — is agreement with 1 km *cells*, not with fields.** Nothing here can
  validate field-level detail. This is also *why* the floor is so high: at 1 km granularity the
  set of occupied cells barely moves year to year.
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

## Windows: it transfers, but THREE code fixes were required

*(Corrected 2026-08-10 after actually doing it. This section previously claimed "no code
changes needed" and gave install commands that fail on current hardware. Both were wrong.)*

Portability of paths was fine — no hardcoded POSIX paths, no unix-only shell calls, 23 files
use `pathlib`. But three **runtime** defects surfaced only on Windows, all in
`src/data/stac_export.py`, and none of them announce themselves:

1. **dask's threaded scheduler deadlocks** with rasterio/GDAL reads. The export hangs
   **forever at ~0% CPU with zero bytes written and no error** — it looks like a slow
   download, indefinitely. Observed: 80 minutes hung; 28 seconds with
   `dask.config.set(scheduler="synchronous")`. Not a thread-count issue — 8 workers also
   hung. **Parallelism must come from one PROCESS per year**, never threads.
2. **No retry on transient reads.** Azure blob range reads intermittently return 0 bytes
   (`TIFFFillTile:Read error ... got 0 bytes, expected 407674`), and a single dropped read
   killed an entire year. Fired 11 times over one full 6-year rebuild.
3. **Resume validation must detect truncation.** A GeoTIFF truncated mid-write still opens
   and still reports the correct band count, so existence and `count` are not sufficient —
   read a far-corner pixel of the last band.

### The venv can break with no code change (hit 2026-08-18)

`.venv\Scripts\python.exe` started failing with **`uv trampoline failed to spawn Python child
process - entity not found (os error 2)`**, and *nothing in the project could run*. The cause is
not the project: uv installs its managed interpreter behind a **junction**
(`...\uv\python\cpython-3.12-windows-x86_64-none` -> `cpython-3.12.8-...`), and Windows can stop
traversing it (`Get-ChildItem`: *"the path cannot be traversed because it contains an untrusted
mount point"*), so the launcher cannot find its base Python. The 5.2 GB of installed packages are
fine - only the launcher is broken, so **do not rebuild the venv.**

Fix applied here: point `.venv/pyvenv.cfg`'s `home` at the concrete versioned directory and
replace the uv trampoline with a real `python.exe` copied from it (the old trampoline is kept as
`.venv/Scripts/python.trampoline.exe.bak`, the config as `pyvenv.cfg.bak`). Verify with
`.venv\Scripts\python.exe -c "import torch; print(torch.cuda.get_arch_list())"` - `sm_120` must
appear - and `python -m pytest tests/ -q`.

### Environment that actually works (verified 2026-08-10)

**conda is not required.** Modern pip wheels for the geospatial stack install cleanly on
Windows; this box has no conda at all. `uv` + pip wheels gave rasterio 1.5.1, geopandas
1.1.4, pyogrio 0.13 with zero build steps.

⚠ **Do not use the cu121 wheel.** An RTX 5080 is Blackwell (**sm_120**); cu121 ships no
sm_120 kernels. Use **cu128 or newer**, then verify `sm_120` appears in
`torch.cuda.get_arch_list()` — a mismatch here degrades silently rather than erroring.

```powershell
uv venv .venv --python 3.12
uv pip install --python .venv\Scripts\python.exe torch --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.get_arch_list())"
```

Verified working: torch 2.11.0+cu128, `sm_120` present, real matmul on device, and all 13
core modules importing clean against pandas 3.0 / numpy 2.5 / geopandas 1.1.

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

> ### UPDATE 2026-08-18 — steps 2, 3 and A16 are DONE; the verdict is in
>
> The gen4 measurement chain has been run end to end and the numbers are in
> `outputs/metrics/baseline_ladder.jsonl` (42 rows, `data_generation: gen4`):
>
> * **U-Net 0.725 mean presence-IoU**, beating the random forest (0.346) and NDVI (0.262)
>   **6/6 folds** — so "NDVI is the floor" was *not* an artifact of the offset bug.
> * **The A16 persistence floor is 0.931** (0.907–0.955 per fold) and the U-Net beats it in
>   **0/6 folds**, by 0.163–0.289 IoU. Per the frozen rule the spatial claim is **not
>   publishable as a model result** and the U-Net is **not** retuned in response.
> * Read [`docs/BASELINE_LADDER_RESULTS.md`](docs/BASELINE_LADDER_RESULTS.md) (top block)
>   before touching anything: it states precisely what the null is (label-informed, opens no
>   imagery) and what still survives as a claim.
>
> **What is left** is write-up and product, not measurement:
>
> 1. **A17 municipal ranking** — the second headline deliverable, and the one claim that can
>    still be positive. Its design problem is still unsolved: `src/infer.py:municipal_hectares`
>    aggregates a full-AOI prediction including trained-on blocks, so rank by mean predicted
>    *density* over test-block pixels only. Report ρ **with** the adjacent-inversion count, n=8,
>    and against the previous census's ranking as the null — which, given A16, will be strong.
> 2. **Regenerate the UI artifacts** — everything in `ui/data/` is dated 2026-08-09, i.e. built
>    from corrupted data and a stale checkpoint. Blocked on `rio_cogeo`, which is not installed
>    and not in `requirements.txt` (defect O4), so no code here reproduces the tracked COG.
> 3. **Figures** — the strongest is 2019's blank Sentinel-2 region with valid Sentinel-1 VV
>    underneath: one image showing the bug, why it hid, and the fix.
> 4. **Front-door rewrite** — lead with the two nulls that won and the two bugs that cancelled.
> 5. **Housekeeping** — ~24 GB reclaimable (`data/tiles_gen2_leaky`, `data/tiles_gen3_misaligned`,
>    12 stale `_sub_*.tif`, `UNALIGNED_*.tif`).
>
> Do **not** reopen the model. A16's rule forbids retuning in response to its own outcome, and
> the encoder-pretraining lever (O2) is registered as a separate ablation — taking it now would
> read as chasing the verdict.

`docs/PHASE6_9_MASTER_PLAN.md` was the live plan. **Revised 2026-08-10** — steps 1 and the
aux-retrain branch are closed; the data is rebuilt, so start at step 2:

1. ~~**Gate/quantile re-test**~~ — **DONE** (A10/A11). Gate exonerated as the primary cause;
   see `docs/gate_sweep_2022.md` and `docs/quantile_gate_2022.md`, both marked SUPERSEDED
   because they were computed on buggy-corrected data.
   ~~**Aux-input retrain**~~ — **CANCELLED** by A12; the level signal does not exist.
2. **Re-run the RF + NDVI baselines** on corrected data — nearly free, no GPU. Resolves whether
   "NDVI is the floor" was partly the offset bug. **← start here**
3. **Retrain the multiyear model + clean Track A** — one training run. Buys back the *positive*
   spatial result. Track A never needed LOYO.
4. **Phase 6.3 hybrid anchor** — historical mean sets the total, the U-Net distributes it
   spatially. Now the *designated* counting route rather than a fallback, and it removes the
   2023 calibration circularity.
5. **Track B 6-fold LOYO** — expensive; grind opportunistically and report honest n.

Cheap lever noticed 2026-08-10 and not yet taken: the encoder is **randomly initialised**, not
pretrained — `config`'s `encoder_weights: ssl4eo` routes to `geo_keys`, which passes `None` to
`smp.Unet` while `_load_geo_encoder_weights()` remains a no-op warning. Setting
`encoder_weights: imagenet` would give real pretrained weights for free. Treat it as a
pre-registered ablation, not a silent change.

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
