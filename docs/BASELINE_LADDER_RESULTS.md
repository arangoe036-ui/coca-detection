# Baseline Ladder — Results

> ## ⚠ NUMBERS BELOW ARE CONTAMINATED — annotation added 2026-08-10 (Phase 6.6d)
>
> Every figure in this document was computed **before** the Sentinel-2 baseline-04.00
> offset was fixed. `stac_export.py` did `DN/10000` without subtracting
> `BOA_ADD_OFFSET = -1000`, so **all 2022–2024 reflectances were inflated by ~0.1**.
> Per prime directive #3 this document is **annotated, not overwritten**; the corrected
> re-run will be published beside it.
>
> **Scope of the damage:**
> - **Every LOYO fold is affected, including the clean-year folds**, because each one
>   *trains* on 2022–2024.
> - **"NDVI is the floor" is suspect.** NDVI is the most corrupted channel — adding a
>   constant to both NIR and Red compresses the ratio — so part of its deficit may be
>   the bug rather than physics.
> - **"SWIR/NBR dominates, NDVI 15/18" is suspect.** NBR is also an index computed from
>   offset reflectances. The ranking may shift.
> - **What likely survives:** all three methods saw the *same* corruption, so the
>   *direction* U-Net > RF > NDVI is probably robust. The **magnitudes are not.**
>
> **Status of the fix (2026-08-10):** the offset correction is implemented, keyed on the
> `s2:processing_baseline` metadata field and applied per scene before indices, and it
> **passed** the model-free Phase 6.6b acceptance check — the +0.100 visible-band step at
> 2021→2022 is now **+0.0013** (`scripts/acceptance_6_6b.py`). All six years of imagery
> and 3,450 tiles have been regenerated on the corrected pipeline. The baseline ladder
> re-run on that data is **pending** and will replace this table.
>
> See [`a12_level_signal.md`](a12_level_signal.md) for what the corrected data then showed
> about the counting failure.

Generated from `outputs/metrics/baseline_ladder.jsonl` (42 runs; reproduce the
tables with `python -m src.baselines.compare`). Interpretation is the
pre-registration in [`BASELINE_PREREG.md`](BASELINE_PREREG.md), applied verbatim —
no reading was chosen after seeing numbers. Method: [`BASELINE_LADDER_PLAN.md`](BASELINE_LADDER_PLAN.md).

**Scope note read this first (prereg A4).** Official labels are a ~1 km density
grid, and `src/data/labels.py:107-115` burns one constant fraction
`coca_ha / cell_ha` into *every* 20 m pixel of each cell. Supervision is therefore
**uniform within each 1 km cell**. Consequently every spatial metric below is
**cell-level** — agreement with *which 1 km cells* contain coca — and nothing in
this repo can validate sub-cell (field) placement against 1 km labels. This
reframes, it does not retract, the spatial result. Phase 7 replaces this
supervision with a cell-aggregate loss.

---

## Headline

The pre-registration split the question into two tracks, and they land on opposite
sides — which is exactly why the two-track design mattered:

- **Spatial (Track A, cell level):** the U-Net is **justified as a localizer.** It
  beats the context-free random forest on cell-level presence-IoU in **6/6** years
  (0.474 vs 0.260; NDVI 0.169). Spatial context genuinely helps decide which 1 km
  cells contain coca.
- **Magnitude (Track B, out-of-year hectares):** the U-Net **does not beat the
  historical-mean null N2** (paired 2W/0T/4L, mean δ = −0.141 — it fails the
  pre-registered ≥5/6 bar). The random forest ties the U-Net (3W/1T/2L). *For
  estimating hectares in a censusless year, no imagery model beats predicting the
  average of past official counts* — **but see §5.3: that bar is low because the
  official series barely moves, and N2 is blind to change by construction.**

Pre-registered cases that fire (prereg §6): **Case 0** (magnitude: U-Net not better
than N2) for Track B, and the **spatial win** of Case 2 for Track A. Case 3 (RF
ties spatially) and Case 4 (NDVI competitive) do **not** fire — NDVI is the floor.
Net reading, quoting the prereg Case 2: *"the model localizes better than it
quantifies; scope claims to location/ranking, not absolute out-of-year hectares."*

---

## Track A — cell-level spatial skill (held-out `test` blocks, per-year mean)

Fit on all-years `train` blocks, evaluated on all-years spatially-held-out `test`
blocks. Predictions are calibrated to the density scale by a train-only constant
`a` (prereg A5) so all three methods are compared on the same footing; `a ≈ 1.0`
for RF, `1.22` for the U-Net, `0.047` for the NDVI ramp.

| metric | **U-Net** | Random forest | NDVI ramp |
|---|--:|--:|--:|
| presence **IoU** | **0.474** | 0.260 | 0.169 |
| presence **F1** | **0.636** | 0.412 | 0.288 |
| **MAE** (density) | **0.0164** | 0.0261 | 0.0313 |
| **RMSE** | **0.0293** | 0.0366 | 0.0416 |

Per-year presence-IoU: U-Net 0.312 / 0.352 / 0.529 / 0.584 / 0.550 / 0.519
(2019→2024); it exceeds the best baseline in **6/6** years → **U-Net wins
spatially (prereg ≥5/6 rule)**. The ladder is monotone: U-Net > RF > NDVI on every
metric.

> **5.2 / A5 calibration note.** Before this fix the NDVI ramp scored MAE 0.5458 — a
> scale artifact (it predicts a 0–1 ramp, not a density), not a finding. Under the
> unified density-scale convention its MAE is 0.0313, comparable to the others,
> and the IoU ranking is unchanged. Track A now uses one convention for all three
> methods; Track B applies each method's `fit_scalar` to reach hectares.

---

## Track B — out-of-year magnitude (`aoi_ratio` = predicted / official ha)

`aoi_ratio` sums over the AOI, so it is **structurally incapable of judging spatial
skill** (that is Track A's job). The U-Net row is the frozen v2.1 LOYO result and
is **not** altered by any Phase-5 fix.

| held-out year | U-Net | Random forest | NDVI ramp | **N2 hist. mean** | N1 const. |
|---|--:|--:|--:|--:|--:|
| 2019 | 0.95 | 0.66 | 1.41 | 1.08 | 1.23 |
| 2020 | 0.59 | 0.59 | 1.43 | 1.11 | 1.27 |
| 2021 | 0.90 | 0.56 | 1.68 | 1.01 | 1.15 |
| 2022 | 1.48 | 1.09 | 0.91 | 1.02 | 1.16 |
| 2023 | 1.01 | 1.35 | 0.89 | 0.96 | 1.10 |
| 2024 | 0.79 | 1.06 | 0.73 | 0.85 | 0.97 |
| **mean** | 0.95 | 0.88 | 1.17 | **1.01** | 1.14 |
| **std S** | 0.271 | 0.298 | 0.346 | **0.085** | 0.097 |
| **C = \|mean−1\|** | 0.047 | 0.116 | 0.174 | **0.006** | 0.145 |
| **K ∈ [.85,1.15]** | 3/6 | 2/6 | 2/6 | **5/6** | 3/6 |
| **band** | 0.59–1.48 | 0.56–1.35 | 0.73–1.68 | 0.85–1.11 | 0.97–1.27 |

**Paired verdicts (prereg §5/A3; δ = |ratio−1|(X) − |ratio−1|(U-Net); W/T/L counts
U-Net wins/ties/losses; tie = |δ| ≤ 0.001):**

- **U-Net vs Random forest:** 3W / 1T / 2L, mean δ = +0.072 → **indistinguishable
  at n=6** (RF ties). *The 2020 fold is an exact numerical tie (both 0.59, δ =
  0.000) — reported as a tie, not a U-Net win (5.4).*
- **U-Net vs NDVI ramp:** 5W / 0T / 1L, mean δ = +0.120 → **U-Net clearly better**
  (NDVI is the floor).
- **U-Net vs N2 (historical mean):** 2W / 0T / 4L, mean δ = −0.141 → U-Net **fails**
  the ≥5/6 bar → **Case 0: the U-Net earns nothing over a no-model historical
  average on magnitude.** (We do not claim N2 "clearly beats" the U-Net — N2 is
  closer in 4/6, short of 5/6; the point is the U-Net is not justified here.)
- **U-Net vs N1 (constant density):** 3W / 0T / 3L → indistinguishable.

### 5.3 — How much did the target actually move? (context for N2's win)

Official Catatumbo coca hectares, 2019–2024:

| year | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 |
|---|--:|--:|--:|--:|--:|--:|
| official ha | 36,296 | 35,277 | 38,285 | 37,965 | 39,815 | 44,240 |

mean 38,646 · std 2,890 · **coefficient of variation 0.075** · range 35,277–44,240
(23% of mean) · year-over-year moves −2.8%, +8.5%, −0.8%, +4.9%, **+11.1%**.

**N2's std (0.085) ≈ the target's own CV (0.075) — this is near-definitional.** N2
predicts the mean, so its ratio spread is essentially the series' coefficient of
variation. Because official area is stable (CV 7.5%), "predict the average" is
automatically close, and N2's win is **partly structural rather than skill**. Two
consequences, both stated plainly:

1. The indictment of the U-Net on magnitude is **weaker than a bare "a no-model
   baseline wins" headline** — the bar N2 sets is low precisely because the quantity
   is stable.
2. **N2 predicts zero change by construction** and is therefore blind to turning
   points. Its worst year is **2024 (0.85)** — exactly the largest real move
   (+11.1%) — consistent with a null that lags a genuine increase. A monitoring
   system exists to catch such moves; `aoi_ratio` on a near-stable series cannot
   discriminate that capability. Phase 6.4 tests change-direction directly.

### 5.4 — Feature importances (one physical signal, not three)

RF Gini importance, mean over 6 folds: **NBR 0.107, B12 0.102, B11 0.070**, VV 0.070,
B02 0.065, VH 0.065; **NDVI ranks 15/18**. NBR is *derived from* NIR (B08) and SWIR
(B12), and B11/B12 are the two SWIR bands — so NBR+B12+B11 are **one physical
signal** (the SWIR / burn-and-bare-soil disturbance complex ≈ 0.28 combined), not
three independent findings. RF Gini also splits importance across correlated
features, so the ranking is **indicative, not exact**. Reading (prereg §9): coca is
detected by *disturbance* (bare soil, senescence, burn scars from slash-and-burn),
**not greenness** — which is why the NDVI-only baseline is the floor.

---

## 5.1 — IoU reconciliation (why the README changed)

The README previously reported `IoU 0.665 / F1 0.799 / AP 0.877 (thr 0.504 tuned on
val)`. That is a **stale P0–P4-era figure**: the AP and a validation-tuned threshold
are produced only by the *binary-segmentation* evaluation path
(`_evaluate_segmentation`), and the checkpoint it came from (`best.pt`) is the
single-year-2023 model. The shipped pipeline is multiyear **density regression**
(`final_multiyear.pt`); its authoritative **cell-level** presence-IoU is the Track A
figure, **0.474** (thr 0.02, per-year mean). The README now carries this one
scoped figure and marks 0.665 as superseded. The two numbers are not a
contradiction — they are different tasks, thresholds, and models.

---

## Limitations (stated plainly)

- **n = 6 folds.** No p-values or confidence intervals; the paired ≥5/6 sign rule is
  the strongest claim the data support.
- **Labels are ~1 km and uniformly burned.** All spatial claims are cell-level;
  field-level detail is unvalidated (prereg A4). The 2,065-polygon 20 m artifact
  overclaims and stays unpublished.
- **The RF is context-free by design** — that is what isolates the value of spatial
  context, not a tuned competitor.
- **Track B magnitude comparison is against a stable target** (§5.3); N2's stability
  is partly definitional, and change-detection (Phase 6.4) is the fairer test of the
  product's actual purpose.
- **2023 calibration circularity** remains in the frozen scalar (`fit_year: 2023`);
  Phase 6.3's year-aware anchor is designed to remove it.
