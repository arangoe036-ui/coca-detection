# Known defects

**Why this file exists.** Between 2026-07-29 and 2026-08-10 this project found three defects
that each invalidated published results, and each fix surfaced more questions than it closed.
That is a good sign about the auditing and a bad sign about ever shipping. The rule below stops
the search becoming infinite.

## The rule

> A newly found defect is **measured for blast radius and documented here — not fixed** —
> unless it flips the **sign of a published number** or the **direction of a conclusion**.
> Everything else ships as a stated limitation.
>
> No new pre-registered question is opened until the current correction cycle produces a number.

A shipped project with an honest defect list is worth more than an unshipped perfect one, and
the list is itself the evidence of rigour. What is *not* acceptable is a defect that is known
and silently omitted from the docs.

## Fixed (2026-08-10) — kept here because published numbers depended on them

| # | Defect | Effect | Evidence |
|---|---|---|---|
| F1 | S2 `BOA_ADD_OFFSET` ignored (ESA Processing Baseline 04.00, 2022-01-25) | all 2022–2024 reflectance inflated ~0.1; every LOYO fold trained on it | step at 2021→2022 now **+0.0013** vs +0.1000 (`scripts/acceptance_6_6b.py`) |
| F2 | `scl_mask_classes` omitted SCL 0 (NO_DATA) / 1 (SATURATED) | nodata took a median of exactly 0.0 — **25.3% of the 2019/2020 AOI silently blank**, 12.2% in 2021, 0.7% in 2022–2024 | `scripts/blank_footprint.py`; S1 VV 97% valid underneath proves real S2 loss, not edge fill |
| F3 | Tile→block split assigned by **top-left corner** while the 256 px footprint spans blocks | **13.67% of test pixels were also in training** (623,840 px); asymmetric — only the U-Net has the receptive field to exploit it | now 0 px, asserted by `tests/test_split_isolation.py` |
| F4 | dask threaded scheduler deadlocks with rasterio/GDAL on Windows | export hung indefinitely at ~0% CPU, zero bytes, no error | synchronous scheduler; 28 s vs 80 min hang |
| F5 | No retry on transient Azure reads | one dropped range read killed a whole year's export | retry with backoff; absorbed 11 failures over a full rebuild |
| F6 | Resume validated band **count** only | a truncated GeoTIFF still opens and reports 18 bands | far-corner pixel read + geospatial identity + filter-config tags |
| F7 | Expired Planetary Computer SAS tokens were unrecoverable | `planetary_computer` reuses a token within a process, so a "fresh" catalog returns the SAME expired token → HTTP 403 forever. **Three workers each stopped ONE SECOND before their token's `se=` expiry** (23:02:50/23:02:49, 22:00:31/22:00:30, 22:46:20/22:46:19), losing ~6 h | expired creds now exit `75`; `scripts/run_export.ps1` relaunches with fresh credentials and resumes from completed sub-tiles |
| F8 | Machine slept after 15 min idle and hard-crashed (Windows event 41) | turned a recoverable stall into 9 lost hours | `powercfg` standby/hibernate/disk-timeout set to never on AC |

**F1 and F2 anti-correlated across years** — F1 inflated 2022–2024, F2 suppressed 2019–2021, and
the two year-groups are exactly complementary. The published headline "unbiased but wide, mean
0.95, std 0.271" was substantially the two defects cancelling. The corruption did not produce an
obviously broken number; it produced a *publishable* one. That is why it survived.

## Open — documented, not fixed

| # | Defect | Why not fixed | Impact on published claims |
|---|---|---|---|
| O1 | The 2020 LOYO residual (−0.41) has never been fully explained | F2 is a strong candidate (2020 is joint-worst blank fraction, lowest NDVI/NBR, highest visible reflectance — a starved region averaged over hazy scenes) and will probably dissolve on re-run. Not worth pre-empting. | none once counting is out of scope |
| O2 | `encoder_weights: ssl4eo` resolves to a **randomly initialised** encoder — `geo_keys` passes `None` to smp and the geo loader is a no-op warning | fixing it changes two things at once and would confound the corrected re-run. Registered as a separate ablation. | model is weaker than the config implies; no claim depends on pretraining |
| O3 | Spatial autocorrelation across split boundaries is **mitigated, not eliminated** — dropping one tile position leaves a **192 px = 3.84 km** gap (corrected 2026-08-14; this row previously said "~5.1 km", which was one tile *width*, not the gap between two kept tiles) and coca autocorrelates past 10 km. **Worse under gen4/A20:** the per-boundary gap is unchanged, but **51.1% of kept tiles now border a different `block_fold` and 37.8% a different split, against 10.0% under gen3's bands.** The 192 px is also not a floor — `_windows` clamps the final row/column, so the grid admits a 59 px (1.18 km) gap if a boundary lands there; it is re-measured and bar-checked every build | a wider buffer costs more tiles than the AOI can spare (A20 shows the arithmetic); the honest fix is disclosure | "zero pixel overlap" is asserted; **"leakage-free" must never be claimed**. A20 strengthens this row rather than resolving it |
| O4 | `rio_cogeo` is not in `requirements.txt` and is not installed, so `write_cog` cannot run | out of the current cycle | **no code in this repo reproduces the tracked `ui/data/density_2023_cog.tif`** |
| O5 | `src/nowcast.py` never sets `cfg["year"]`, so its outputs carried the wrong year's census | its artifacts were deleted and hectares are out of scope; the code remains | none — outputs removed |
| O6 | The 2023 calibration scalar is fitted to the same year's census (`src/infer.py:218`) | inherent to the method | the 2023 total matching official is arithmetic, not evidence. Hectares out of scope |
| O7 | Test split lands at ~10% of tiles rather than the 15% target after the leak fix | bands are whole 500 px block columns, so achievable sizes are coarse | 50 test tiles/year, 300 across six years — thin, disclose n |
| O8 | `_subtile_bboxes` can emit a zero-width sub-tile for some bbox/step combinations | production geometry verified unaffected | none for Catatumbo |

## Retracted claims

Numbers withdrawn rather than adjusted, because the runs behind them no longer exist:

- **`presence-IoU 0.474` / RF `0.260` / NDVI `0.169`, 6/6 folds** — computed under F1+F2+F3.
- **`predicted 46,843 ha vs official 39,815 ha (1.18×)`** and per-municipality `Tibú 0.98 /
  El Tarra 0.95 / Teorama 1.05` — present in no artifact; the tracked CSV gives 0.963 / 0.974 /
  **0.895**, and only the best 3 of 10 were published against this project's own rule.
- **"LOYO 0.59–1.48, mean 0.95 ±0.27, unbiased but wide"** — the F1/F2 cancellation artifact.
- **A12's B11/B12 correlations** (`+0.355`, `+0.050`) — footprint artifacts; on a fixed footprint
  both flip sign (−0.621, −0.709). A12's *verdict* is unchanged.

## Not yet measured, and it could sink the headline

**There is no no-skill floor for the spatial claim.** Coca is a perennial, so "it is where it was
last year" is a strong predictor and has never been computed. Registered as **A16** with the
decision rule fixed in advance. If the U-Net cannot beat persistence in ≥5/6 folds, the spatial
result is not publishable as a model result — the same trap that caught the counting claim, where
a no-model historical mean won and was only discovered because someone ran it.

---

## BLOCKER (2026-08-13) — B1: the registered test split contains no coca

> **RESOLVED at the data level 2026-08-14 — read this note before the section below.** The
> section is left intact (annotate, never overwrite) because it is the record of how the defect
> was found and why it survived. What has changed:
>
> - The split was rebuilt under prereg amendment **A20** (stratified 5×4 macro-blocks in
>   tile-index space → 6 dual-objective folds → 4/1/1 roll-up, `block_fold` recorded per tile).
>   `project.data_generation` is now **`gen4`**.
> - The `test` split now holds **1.13–1.24 M positive pixels in every one of the six years**
>   (gen3: 0), and so does every individual `block_fold`. Splits and folds remain **exactly
>   pixel-disjoint**.
> - The invariant whose absence caused this — *every split must contain positive labels in
>   every year* — is now asserted in code (`src.data.tiling.verify_split_pixels`, which raises)
>   and in `tests/test_block_folds.py`, including tests that prove the guard fires on a
>   deliberately blanked split. (A test count was quoted here on 2026-08-14 and was wrong twice;
>   it is a short-half-life fact and does not belong in a permanent register — run the suite.)
> - Fix-path steps 1–3 and 5 (agree the design, register it, add the invariant, reassign) are
>   **done**. Step 4 — the `src/baselines/persistence.py` degeneracy guard (S2) — is **NOT
>   done**, and neither is `compare._dedup_last` keying on `data_generation`. Until those land,
>   nothing stops a *future* all-negative target from printing `0.000` again, and nothing stops
>   a gen3 row being deduplicated against a gen4 one.
> - **The blast radius is unchanged and still binding: every Track A number computed on gen3 is
>   void.** No retrain has been run on gen4 and `outputs/metrics/baseline_ladder.jsonl` still
>   does not exist, so there is nothing yet to compare. gen4 also trains on **22.1% fewer train
>   rows** (2250 → 1752), so a gen4 result is not comparable to a gen3 one for that reason too.


Found by a `reviewer` audit, confirmed independently. On the gen3 build:

| split | tiles | tiles with any coca | positive pixels | x columns |
|---|--:|--:|--:|---|
| train | 2250 | 1668 | 76,171,742 | 0 … 3136 |
| val | 450 | **76** | 1,336,641 | 3584 … 4032 |
| **test** | 300 | **0** | **0** | 4704, 4795 |

`src/evaluate.py:47` computes `iou = tp / (tp + fp + fn + 1e-6)`, so an empty target with an
empty prediction returns **`0.000` with no warning**. Every `presence_iou` on the `test` split —
U-Net, random forest, NDVI **and** the A16 persistence nulls, all of which use those same rows —
is `0/0` dressed as a measurement.

**This is not a pre-existing defect. It was introduced by the fix for F3** (the top-left-corner
leak). The replacement rule cuts contiguous west→east bands at 70/15/15 by cumulative tile
count. It removed the leak — the 0.00% train∩test overlap is real and independently verified —
but coca in Catatumbo is a central blob and the eastern margin is empty. Positive-pixel fraction
pooled over all six years:

```
x=   0  0.291   x=1568 0.699   x=2912 0.519   x=3808 0.017   x=4704 0.000
x= 896  0.387   x=2016 0.729   x=3136 0.322   x=4032 0.099   x=4795 0.000
y=   0  0.000   y=1120 0.254   y=2688 0.487   y=4256 0.624   y=5372 0.300
```

Steep in **both** axes, so a north–south band split would also have emptied the test fold.
Contiguous bands cannot work for this AOI at all.

**Blast radius.** Sizing the void by re-running the identical code on `val` (a diagnostic only —
`val` is not the registered split and must never become the published metric) puts the honest
persistence floor at IoU **0.66–0.84** per fold, against the project's best-ever U-Net figure of
0.474 (itself retracted, gen1). So the floor was understated by roughly **0.8 IoU**, entirely in
the direction that flatters the model. The danger is asymmetric: a self-consistent gen3 run reads
as 0/6 ("not publishable"), but pairing a gen3 persistence `0.000` against a differently-sourced
U-Net number manufactures a **false 6/6**. `src/baselines/compare.py:144` `_dedup_last` keys on
`(method, track, fold_year)` and **ignores `data_generation`**, which is exactly that mechanism.

**Why it survived.** Two checks were run on the new split rule — leak-freeness and label totals
matching the census — and both passed. Neither can detect an empty fold: the leak check is about
overlap, and the census check sums over *all* splits. `tests/test_split_isolation.py` has 23
tests, all about isolation; `tests/test_data_invariants.py` has no invariant on how labels are
*distributed* across splits. **A split has to be both leak-free and signal-bearing, and only the
first was ever asserted.**

**Fix path** (does *not* require re-tiling — `split` is a column in
`data/tiles/multiyear_index.csv`; imagery, tiles and labels are all fine):
1. Settle a 2D scattered/stratified block design with a buffer, via the `advisor`.
2. Register it as prereg amendment **A20** *before* recomputing anything.
3. Add the missing invariant to `tests/`: **every split must contain positive labels in every
   year.**
4. Add the degeneracy guard to `src/baselines/persistence.py` (reviewer item S2) so an
   all-negative target raises instead of printing `0.000`.
5. Reassign splits (minutes), retrain (~15 min), then run A16.

Supersedes the O7 entry above, which describes the test split as thin (~10%) but not as **empty
of labels**.

## Open (2026-08-13) — O9: training is not reproducible run-to-run

Two consecutive `python -m src.train_loyo --final --epochs 30 --patience 6` runs on identical
gen3 data gave calibration scalars **1.060** and **1.642**, different best epochs and different
`val_mae` (0.0065 vs 0.00754), despite `set_seed(cfg["project"]["seed"])` being called. Not
investigated (scope freeze). Consequence for write-ups: **no single-run scalar is quotable as
*the* value**, and any scalar that appears in prose needs either a seed-averaged figure or an
explicit "one run" caveat.

## Open (2026-08-13) — O10: `--smoke` wrote into production paths

`python -m src.train_loyo --smoke` runs 2 epochs on a single year and wrote its results to
`outputs/metrics/loyo_corrected.jsonl` — the file the v2.1 LOYO ratio table is read from — and
saved `outputs/checkpoints/loyo_fold_2023.pt` / `loyo_fold_2024.pt`, the **exact** filenames a
real 6-fold run produces and that a downstream Track B script would load. Two meaningless ratios
(1.61, 1.33) sat in the production sink with nothing marking them as a wiring test. All four
artifacts moved to `outputs/DISCARDED/` with a README. The code change routing `--smoke` to
`loyo_smoke.jsonl` / `SMOKE_fold_*.pt` is **not yet made** — verify before the next smoke run.
