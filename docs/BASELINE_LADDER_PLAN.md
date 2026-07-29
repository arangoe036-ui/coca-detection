# Baseline Ladder — execution plan (P6-A)

## Goal

Establish whether the U-Net is *justified* by comparing it against progressively
simpler methods on **identical LOYO folds**, and persist every number to disk.

Right now `IoU 0.665 / F1 0.799 / AP 0.877` live only as README prose, and nothing
in this repo compares the U-Net to a simpler method. "Baseline" in the current code
(`src/models/__init__.py:1`, `src/train.py:158`) means *"the baseline U-Net vs. the
stretch temporal model"* — it is not a comparison baseline. NDVI appears only as an
input feature (`src/data/stac_export.py:109`), never as a competing predictor.

A reviewer's first question about any deep-learning geospatial project is **"would a
threshold or a random forest have done this?"** This plan answers it with measured
numbers instead of an assumption.

## Prime directives

1. **Never fit anything on the held-out year.** Thresholds, hyperparameters,
   normalization stats, and the calibration scalar are all fit on train years only.
   This mirrors `fit_scalar` in `src/train_loyo.py:142`, which already receives only
   train rows — preserve that property exactly.
2. **Reuse the existing fold logic. Do not reimplement it.** All three methods must
   consume the same `read_index` / `year_norm_stats` / block-split machinery from
   `src/train_loyo.py`. If you write a second splitting routine, the comparison is
   void.
3. **Do not modify the U-Net, the losses, or the tiling.** This is an additive
   experiment. `src/models/unet.py`, `src/models/losses.py`, and `src/tiling.py`
   are read-only for this work.
4. **Pre-register the interpretation before looking at results** (Phase 0.3). Every
   outcome must already have a meaning assigned, so the conclusion cannot be fit to
   the data after the fact.
5. **A tie is a publishable result, not a failure.** If a random forest matches the
   U-Net, that is a genuine finding about label resolution — report it, don't bury it.
6. **Persist everything.** No metric may exist only in stdout or prose.

## Known facts (verified — do not re-derive)

- Tiles: `data/tiles/{year}/tile_XXXXX.npz`, keys `image` `(18, 256, 256) float32`
  and `mask` `(256, 256) float32`. The mask is a **coca area fraction** (density),
  not a binary label — the task is `regression` (`config/default.yaml:79`).
- Index: `data/tiles/multiyear_index.csv`, columns
  `tile_id, year, split, block, x, y, pos_frac, npz`. 3,450 tiles, 110 blocks.
  Block→split assignment is consistent across years (verified: 0 inconsistencies).
- Declared channel groups: `s2_bands: [B02,B03,B04,B08,B05,B06,B07,B8A,B11,B12]`,
  `s1_bands: [vv,vh]`, `indices: [NDVI,EVI,SAVI,NDWI,NDRE,NBR]`
  (`config/default.yaml:31-33`) → 10 + 2 + 6 = 18.
- Existing LOYO entrypoint: `run_loyo(cfg, years, epochs, patience)`
  (`src/train_loyo.py:176`), with `train_fold`, `fit_scalar`, `test_year_ratio`.
- Existing metric helpers: `_metrics_at(probs, targets, thr, t_thr=0.5)` and
  `_evaluate_regression(cfg, probs, targets)` in `src/evaluate.py`.

---

## Phase 0 — Verify, then pre-register (do not skip)

**0.1 Confirm the channel order empirically.** The YAML lists `s2_bands`,
`s1_bands`, `indices` in that order, but the `in_channels` comment
(`config/default.yaml:83`) says `len(s2)+len(indices)+len(s1)`. These disagree about
where S1 sits. Find the code that stacks the array (start in `src/data/`, likely
`stac_export.py` / the tiling writer) and determine the true index of **NDVI** and of
**vv/vh**. Print per-channel min/median/max for one tile as a sanity check: NDVI must
land in roughly [-1, 1], S1 backscatter is typically negative in dB. **Write the
confirmed mapping into this file before proceeding.** Every later phase depends on it.

**0.2 Confirm what U-Net numbers already exist per fold.** Read
`docs/v2.1_loyo_results.md`. It reports per-year AOI ratios (0.59–1.48, mean 0.95
± 0.27). Determine whether **per-fold MAE / RMSE / presence-IoU / F1** were ever
saved, or only the ratios. This decides the scope below.

**0.3 Write `docs/BASELINE_PREREG.md` — before running anything.** State, in advance:
- the exact metric set (below) and which is primary,
- the decision rule for "the U-Net is justified",
- what you will conclude in each case:
  - U-Net clearly best on both spatial (IoU/F1) and magnitude (MAE/ratio spread) →
    deep learning earned; report the margin.
  - U-Net wins spatially but ties on magnitude → the model localizes better than it
    quantifies; scope claims to location/ranking only.
  - Random forest ties or wins → **the honest headline is that ~1km-resolution labels
    do not support a segmentation model.** Report it as the finding.
  - NDVI threshold alone is competitive → the task is largely a vegetation-index
    problem; say so plainly.
- Commit this file before Phase 2 produces any number.

**Acceptance:** channel mapping written down and verified; scope decided; prereg
committed.

### Phase 0 findings (verified 2026-07-29 — do not re-derive)

**0.1 — Channel order (RESOLVED, empirically confirmed).** The apparent
"config disagrees with itself" is a red herring: the YAML *declaration* order
(`s2_bands`, then `s1_bands`, then `indices`, lines 31–33) is NOT the stacking
order. The array is stacked by `band_order()` (`src/data/stac_export.py:149-151`)
as **s2_bands + indices + s1_bands**, which matches the `in_channels` comment
(`config/default.yaml:83`: `len(s2)+len(indices)+len(s1)`). Tiles inherit this
order verbatim — `multiyear.build_year` writes `img.read(...)` straight from the
annual raster (`src/data/multiyear.py:62,71`), no reordering.

Confirmed 18-channel index → band map:

| idx | band | idx | band | idx | band |
|----:|------|----:|------|----:|------|
| 0 | B02 | 6 | B07 | 12 | SAVI |
| 1 | B03 | 7 | B8A | 13 | NDWI |
| 2 | B04 | 8 | B11 | 14 | NDRE |
| 3 | B08 | 9 | B12 | 15 | NBR |
| 4 | B05 | 10 | **NDVI** | 16 | **VV** |
| 5 | B06 | 11 | EVI | 17 | **VH** |

Empirical sanity check (tile `2019_00030`, `pos_frac=0.28`):
idx 0–9 reflectances all in [0,1]; **idx 10 NDVI** median 0.851, range
[−0.228, 0.900] ⊂ [−1,1] ✓; **idx 16 VV** median −6.63 dB, **idx 17 VH** median
−12.74 dB (both negative, as expected for backscatter in dB) ✓. Reproduce with
`python scripts/verify_channels.py` (run from repo root, venv active).
→ **Baselines use channel index 10 for NDVI.**

**0.2 — What U-Net per-fold numbers exist (RESOLVED).** `run_loyo`
(`src/train_loyo.py:176-201`) persists **nothing to disk** — it prints and returns
`(test_year, ratio, pred_ha, off_ha, scalar)`. The only surviving per-fold record
is `docs/v2.1_loyo_results.md`, which has, per held-out year: **`aoi_ratio`,
predicted_ha, official_ha, scalar, and `val MAE`**. Critically, that `val MAE`
(~0.02 every fold) is measured on the *train years' VAL blocks*, **not** on the
held-out year — it is not comparable to a held-out-year MAE and must not be used
as one. No held-out-year MAE / RMSE / presence-IoU / presence-F1 for the U-Net
exists anywhere on disk.

**Scope decision:** the only metric that (a) exists for the U-Net per fold and
(b) is measured on the held-out year is **`aoi_ratio`** → it is the **primary**
comparison metric. Held-out-year tile metrics (MAE/RMSE/bias/presence-IoU/F1) are
free for the two baselines but require a 6-fold U-Net re-run to obtain for the
U-Net. Per the plan's Phase-4 fork, we start ratio-only (free, all three methods
directly comparable) and defer the re-run decision to the user after Phase 3. Any
table mixing metric sets will label the restriction explicitly.

Feasibility note: `aoi_ratio` via `test_year_ratio` needs (i) the full annual
raster per year — all present at `data/imagery/catatumbo_{2019..2024}_annual_full.tif`
— and (ii) the official held-out-year hectares, which are already frozen in the
v2.1 table (36,296 / 35,277 / 38,285 / 37,965 / 39,815 / 44,240 ha), so no network
call is required.

---

## Phase 1 — Metrics persistence (~1–2h)

Create `src/metrics_io.py` with a single writer used by every method:

```
write_run(out_dir, method, fold_year, metrics: dict, config_snapshot: dict) -> Path
```

Each run appends one JSON object to `outputs/metrics/baseline_ladder.jsonl` with:
`method` (`ndvi_threshold` | `random_forest` | `unet`), `fold_year`,
`n_train_tiles`, `n_test_tiles`, `calibration_scalar`, `fit_years`, `git_commit`,
`timestamp`, and a `metrics` block.

**Fixed metric set for all three methods** — compute with the *existing*
`src/evaluate.py` helpers, do not write new metric math:
- `mae`, `rmse`, `bias` on the density fraction
- `presence_iou`, `presence_f1` via `_metrics_at` (same `thr`, same `t_thr`)
- `aoi_ratio` = predicted / official hectares for the held-out year, via
  `test_year_ratio`

Also add `--emit-metrics` to `src/evaluate.py` so the existing U-Net path writes
`outputs/metrics/metrics.json` instead of only printing. This retires the
"prose-only headline numbers" problem regardless of how the rest goes.

**Acceptance:** one dummy run appears correctly in the JSONL; `evaluate.py
--emit-metrics` produces a real `metrics.json`.

---

## Phase 2 — Baseline A: spectral-index threshold (~2–3h)

New file `src/baselines/ndvi_threshold.py`.

Method: read the NDVI channel (index confirmed in 0.1). Predict density as a
piecewise-linear ramp of NDVI — or, simplest defensible form, a binary presence mask
above a threshold `t`, converted to a density by multiplying by a constant fitted on
train years.

**Fitting rule (critical):** sweep `t` over a grid and select the value minimizing
train-year MAE **using only train-year TRAIN blocks**. The val blocks may be used for
the sweep; the held-out year must never be touched. Then apply the existing
`fit_scalar` procedure on train years to get the calibration scalar, exactly as the
U-Net does.

Run all 6 LOYO folds. Write one JSONL row per fold.

Expect this to be weak. That is the point — it establishes the floor.

**Acceptance:** 6 rows in the JSONL; the selected `t` differs per fold (proving it is
refit, not hardcoded); a written assertion that no held-out-year pixel influenced `t`.

---

## Phase 3 — Baseline B: random forest on pixel spectra (~3–4h)

New file `src/baselines/pixel_rf.py`. This is the baseline that matters — it is the
one that might actually win.

- Features: the 18 channels **per pixel**, no spatial context. Optionally add
  `[x, y]` within-tile position as an ablation, but the headline RF must be
  context-free — that is what isolates "does spatial structure help?"
- Sampling: 3,450 tiles × 65,536 px ≈ 226M pixels is too many. Subsample to
  ~2–5M pixels per fold, stratified so positive-density pixels are not swamped
  (use `pos_frac` in the index to weight tile selection). **Fix the RNG seed and
  record it.**
- Model: `sklearn.ensemble.RandomForestRegressor` (the task is regression). Modest
  depth; record all hyperparameters. Tune only on train-year val blocks.
- Prediction: predict per pixel, reshape to `(256, 256)`, then run the **same**
  `fit_scalar` + `test_year_ratio` path as the other two methods.

Run all 6 folds. Write one JSONL row per fold.

**Acceptance:** 6 rows; feature importances saved per fold (this is free
interpretability — if NDVI and the SWIR bands dominate, that is a real finding worth
a sentence); confirmation that the RF never saw a held-out-year pixel.

---

## Phase 4 — The comparison table (~1–2h)

`src/baselines/compare.py` reads the JSONL and emits
`outputs/metrics/ladder_table.md` + a matching CSV:

| method | MAE | RMSE | presence IoU | presence F1 | AOI ratio mean ± std | folds |
|---|---|---|---|---|---|---|

Rules:
- Report **per-fold rows and the aggregate**, never the aggregate alone.
- Report `mean ± std` across folds, plus min/max. With n=6 do **not** report a
  p-value or a confidence interval that implies more power than 6 folds give.
- If the U-Net numbers for some metrics don't exist per fold (Phase 0.2), either
  re-run `run_loyo` with Phase-1 metrics wired in (costly but complete — 6 trainings)
  or restrict the table to `aoi_ratio`, which you already have, and **label the
  restriction explicitly**. Do not silently compare different metric sets.

**Acceptance:** the table reproduces from the JSONL with one command; every cell
traces to a JSONL row.

---

## Phase 5 — Write it up honestly (~2h)

Create `docs/BASELINE_LADDER_RESULTS.md`:
- The table, then the pre-registered reading from `BASELINE_PREREG.md` applied
  verbatim — quote it, then state which case fired.
- One paragraph on what the RF feature importances imply.
- The limitations, stated plainly: 6 folds is a small n; labels are ~1km resolution
  while predictions are 20m; the RF is context-free by design.

Then fix the two standing integrity items, which belong in the same commit:
1. **Publish all 10 municipalities** in the README, not the best 3. The full range is
   **0.05 to 1.69** (`ui/data/municipal_coca_2023.csv`); the README currently shows
   only Tibú / El Tarra / Teorama. State that large municipalities are accurate and
   small ones are not — the aggregate claim survives, since Tibú alone is ~53% of AOI
   total. Also refresh the stale numbers (README says 0.98/0.95/1.05; the shipped CSV
   says 0.96/0.97/0.90).
2. **Disclose the calibration circularity.** `calibration.json` has
   `fit_year: 2023`, so the 2023 AOI total matching official is partly by
   construction. The per-municipality *distribution* is still real evidence — one
   global scalar cannot force per-municipality agreement — but the 2023 total is not
   independent validation. Say so before a reviewer finds it.

**Acceptance:** results doc committed; README table complete and current; circularity
disclosed.

---

## Guardrails

- **Do not publish** `outputs/catatumbo_2023_coca.geojson` — 2,065 plot-level 20m
  field polygons, targeting-usable. It is gitignored; keep it that way and never
  `git add -f` it. Public artifacts stay at the 75m COG + municipal-aggregate level.
- **Do not build the U-TAE temporal model** (`src/models/temporal.py:13`). It is
  out of scope here and chases accuracy instead of credibility.
- **Do not tune anything toward a nicer table.** If the RF wins, that is the result.
- Keep `data/` and `.venv` out of git (already ignored — verify before any commit).

## Suggested order of work

Phase 0 → 1 → 2 → 3 → 4 → 5. Stop and report after **Phase 0.3** (prereg committed)
and again after **Phase 3** (both baselines run), so the interpretation is agreed
before the write-up.
