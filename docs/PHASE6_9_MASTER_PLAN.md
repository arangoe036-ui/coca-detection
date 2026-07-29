# Phases 6–9 — Master plan (supersedes PHASE6_COUNTING_PLAN.md as the entry point)

`docs/PHASE6_COUNTING_PLAN.md` remains authoritative for the **detail** of Phase 5 and
Phase 6. This file is the entry point: it keeps those phases by reference and adds
Phase 7 (supervision), Phase 8 (sensor fusion), and Phase 9 (consolidation).

## The three problems, and what fixes each

| # | Problem | Evidence | Fix | Phase |
|---|---|---|---|---|
| 1 | Model can't count out-of-year | `aoi_ratio` std 0.271, loses to N2 null | coverage gate + hybrid anchor | 6 |
| 2 | **Per-pixel supervision is wrong** | `labels.py:107-115` burns one constant density into every 20 m pixel of each 1 km cell | aggregate-consistent loss | **7** |
| 3 | Degraded imagery in some years | 2020 cloud; 2022/2024 post-S1B | add HLS + PALSAR | **8** |

**Critical framing discovered in review:** because labels are uniform within 1 km cells,
the Phase 3 result "U-Net IoU 0.466 vs RF 0.260" means the U-Net is better at
identifying **which 1 km cells** contain coca. It is **not** evidence of field-level
localization, and nothing in this repo can validate field-level detail against 1 km
labels. All claims must be scoped accordingly, and the 2,065-polygon artifact
(`outputs/catatumbo_2023_coca.geojson`) implicitly overclaims — it stays unpublished.

---

## Phase 5 — Blocking fixes (see PHASE6_COUNTING_PLAN.md §5)

Unchanged and still first: reconcile IoU 0.466 vs README 0.665; unify the calibration
convention across tracks (NDVI Track A MAE 0.5458 implies Track A is uncalibrated);
publish the official-hectares series beside Track B; fix RF "4/6" → 3W/1T/2L and merge
NBR/B12/B11 into one physical signal.

**Add one item:**

**5.5 State the label-resolution scope limit.** Add a short section to the results doc
explaining that labels are ~1 km and uniformly burned, therefore all spatial claims are
at cell level. This reframes — not retracts — the Track A win.

---

## Phase 6 — Counting (see PHASE6_COUNTING_PLAN.md §6.1–6.4)

Unchanged: 6.1 coverage diagnostic (**report back here**), 6.2 coverage gate if
supported, 6.3 hybrid anchor + spatial allocation, 6.4 change detection as a dated
prereg amendment.

6.1 now serves a second purpose: it is the **before** measurement for Phase 8, and it
tells you *which* sensor to prioritize. Measure S2 clear-observation counts and S1 pass
counts **separately per year** — do not conflate them. Lead worth testing: the three
out-of-band U-Net years are 2020, 2022, 2024; 2020 was cloudy and 2022/2024 sit inside
the S1B outage (failed Dec 2021, S1C launched Dec 2024). But 2023 scored 1.01 on S1A
alone, so a pure-S1 story does not fit — measure, don't assume.

**Skip 6.5 (robustness training) from the old plan.** It is absorbed into the Phase 7–8
ablation ladder below, where it is measured properly as one increment.

---

## Phase 6.5 — Diagnose the magnitude mechanism (~4–6h, mostly no GPU) — NEW

**Why this exists:** Phase 6.1 rejected coverage. The lesson is not just "coverage is
innocent" — it is *do not build on an unverified mechanism*. "Uniform labels cause the
counting failure" is also currently a hunch. Diagnose before intervening again.

**Leading suspect — per-year normalization.** `train_loyo.py:180` computes
`year_norm_stats` over **all** tiles including the test year, and line 165 normalizes
the held-out year with **its own** mean/std. Per-year z-scoring removes each year's
absolute level, so the model cannot perceive "this year is more disturbed overall than
usual" — that signal is subtracted out before it sees anything.

This predicts exactly the observed split: within-year relative patterns survive
(localization works, IoU 0.474, 6/6) while cross-year absolute levels are destroyed
(counting fails), errors are uncorrelated with coverage, the sign is inconsistent, and
model variance (0.271) exceeds the target's own variance (official CV 0.075).

Tests, cheapest first:

- **6.5a (~1h, no GPU).** Correlate the per-year normalization parameters (mean/std for
  key channels: NBR, B12, B11, NDVI) against per-year `|ratio − 1|` and against signed
  error. If 2020's means sit above the pooled mean in a direction consistent with
  under-prediction, and 2022's below, the mechanism is confirmed with no inference runs.
- **6.5b (~2h, inference only).** Swap-stats sensitivity: re-run `test_year_ratio` with
  **train-pooled** stats instead of test-year stats. Report the change in predicted
  total per fold. Note this is a train/test normalization mismatch — it measures
  *sensitivity*, not a fix. Large movement ⇒ normalization is the dominant lever.
- **6.5c (~2h).** Decompose the 2020 and 2022 errors by municipality/block. Concentrated
  ⇒ a specific confusion (investigate what changed in those cells). Diffuse ⇒ a global
  level shift, consistent with normalization.
- **6.5d (~1h).** Sweep the presence gate inside `fit_scalar`. If `aoi_ratio` is
  hypersensitive to the gate threshold, the gate is the amplifier converting small
  density shifts into ~40% total swings.

**Pre-register the outcomes before running**, as with 6.1:
- Normalization implicated ⇒ add a **normalization rung** to the Phase 9 ladder
  (retrain with pooled/global stats, or keep per-year stats and feed the year's absolute
  level back as auxiliary scalar inputs). Cheaper than the sensor build.
- Gate implicated ⇒ fix the gate; report how much of the original 0.271 it explained.
- Neither implicated ⇒ supervision (Phase 7) becomes the leading suspect on elimination
  rather than assumption.

**Acceptance:** committed `docs/magnitude_diagnostic.md` with all four tests, the
pre-registered outcome that fired, and an explicit statement of what remains unexplained.

---

## Phase 7 — Fix the supervision (~12–16h)

### 7.1 Aggregate-consistent loss

Today `labels.py` burns `coca_ha / cell_ha` into every pixel of a 1 km cell, so the
model is trained to predict the cell mean everywhere and is **penalized** for
concentrating prediction where fields actually are.

Replace per-pixel regression with a **cell-aggregate loss**: for each 1 km cell
overlapping the tile, penalize the difference between the *sum* of predicted density in
that cell and the cell's official hectares. The model becomes free to place coca
anywhere inside the cell as long as the total matches.

Implementation notes:
- Keep the existing `mask` for evaluation; add a **cell-id raster** so the loss can
  group pixels by cell. Generate it alongside the mask in `labels.py`
  (`rasterize` with the cell id instead of the fraction) — this is additive, do not
  break the existing path.
- Cells partially overlapping a tile edge must be excluded from the loss or weighted by
  the overlap fraction, or the aggregate target is wrong. Handle explicitly.
- Add a small regularizer (e.g. total-variation or an L1 sparsity term) so the model
  concentrates rather than dispersing arbitrarily. Pre-register its weight; do not tune
  it toward a nicer table.
- Keep the existing loss available behind config so the two are comparable.

### 7.2 How to validate sub-cell structure when labels are 1 km

Cell totals remain directly checkable. Sub-cell placement is not checkable against the
labels, so use independent evidence — and be explicit that each is a proxy:
- **Held-out cell-total accuracy** (rigorous, primary).
- **Prediction concentration** — what fraction of predicted density falls in the top
  10% of pixels within a cell? Uniform supervision should score near 0.10; genuine
  localization should be much higher. Report before/after.
- **Independent clearing products** — agreement with Hansen Global Forest Change or
  GLAD alerts as a proxy for the disturbance signal the RF found dominant
  (SWIR/NBR). Not ground truth for coca; report as corroboration only.
- **Visual comparison against high-resolution imagery** (Phase 8.4) on a handful of
  cells, published as figures with the caveat that it is qualitative.

**Acceptance:** cell-id raster generated; both losses selectable; cell-total accuracy
and concentration reported before/after; every proxy labeled as a proxy.

---

## Phase 8 — Sensor fusion (~20–30h)

Add free, complementary sensors. **All must be resampled onto the exact existing 20 m
grid**, reusing the reference-raster pattern already in `labels.py::rasterize_mask`.

### 8.1 Non-negotiable comparability rule

Regenerating tiles **must preserve the existing block→split assignment exactly**
(keyed on `block` / `x` / `y` in `data/tiles/multiyear_index.csv`). If blocks move
between train/val/test, every comparison to Phase 3 is void and the leakage guarantee
is broken. Assert this programmatically: after regeneration, diff the new index against
the old on `(tile_id, block, split)` and fail loudly on any mismatch.

### 8.2 Coverage channel (cheap, do first)

Add per-pixel **valid-observation count** as an input channel, separately for optical
and SAR. The model currently cannot distinguish *"no coca here"* from *"I could not see
this pixel"* — this is the minimal fix and it also feeds Phase 6.2's gate.

### 8.3 HLS — RATIONALE DEAD AFTER PHASE 6.1. DROP.

HLS was justified by the premise that 2020 lacked clear looks. Phase 6.1 measured
**32.7 clear observations/pixel in 2020 — the best of all six years** — while 2020 is
the second-worst fold. Optical coverage across all years is 25.5–32.7 clear looks/pixel,
which is ample. Adding more optical looks addresses a bottleneck that does not exist.
**Do not build this.** Record the reason so the decision is auditable.

### 8.4 ALOS/PALSAR annual mosaics (25 m, L-band SAR) — rationale SURVIVES, but screen first

JAXA's free global annual mosaics. **The coverage argument for this is also dead** —
Phase 6.1 showed S1 passes only dropped ~15% post-S1B, not ~50%, and 2023 succeeded on
the same SAR coverage that 2022 failed on. What survives is an **information** argument,
which is different and stronger: L-band reads vegetation *structure*, not greenness, so
it is new physics rather than more looks. Given the RF found the SWIR/disturbance complex
dominant (NBR/B12/B11) with NDVI ranking 15/18, a structure-sensitive sensor is
well-motivated on signal grounds.

**Screen before building.** Add PALSAR for a single fold and check whether the L-band
channels rank meaningfully in an RF feature importance against the existing 18. One day
of work to decide on twenty. Proceed to the full build only if they do. Verify 2019–2024
availability first.

### 8.5 Optional — high-resolution imagery for validation only

If accessible (e.g. Planet/NICFI ~4.8 m over the tropics — **verify current access
terms, the programme has changed over time**), use it for **qualitative validation of
sub-cell placement** (Phase 7.2), not for training. Sharper inputs against 1 km labels
buy better-looking maps, not better supervision. Treat as a nice-to-have.

### 8.6 Re-verify the channel map after every addition

Phase 0's lesson: the YAML declaration order is **not** the stacking order
(`band_order()`, `stac_export.py:149-151` → s2 + indices + s1). Every time a sensor is
added, re-run `scripts/verify_channels.py`, print per-channel median/min/max, and update
the documented index table. A silently mis-indexed channel produces plausible garbage.

---

## Phase 9 — The ablation ladder and consolidation (~6–8h + compute)

Mirror the baseline ladder: **one change per rung, each measured on identical folds.**

| Config | Change | Question it answers |
|---|---|---|
| **V0** | current (already measured) | reference |
| **V1** | + coverage channel + observation-dropout augmentation | does knowing what it couldn't see fix the bad years? |
| **V2** | V1 + aggregate-consistent loss | was the uniform-label supervision the real ceiling? |
| **V3** | V2 + HLS | do more optical looks fix the cloud year? |
| **V4** | V3 + PALSAR | does L-band fix the SAR-gap years? |

Report every rung on **both tracks** (Track A cell-level spatial, Track B `aoi_ratio`
per fold vs the N2/N1 nulls), plus Phase 7.2's concentration metric.

**Compute honesty:** 5 configs × 6 folds = 30 trainings. If that is too slow, screen
V1–V4 on a **pre-registered subset of folds** (e.g. 2020, 2022, 2023 — the two failures
plus one control), then run full 6-fold LOYO only for V0 and the winner. State the
screening protocol in advance; never pick the folds after seeing results.

**Pre-register before running the ladder:** what counts as an improvement (paired
≥5/6 folds, as in the Phase 3 rule), and what each outcome means — including the case
where **no rung helps**, which would be a strong finding: free multi-sensor imagery plus
corrected supervision still cannot count out-of-year, so the ceiling is the 1 km labels.

**Acceptance:** one table, all five configs, both tracks, every cell traceable to a
persisted JSONL row; a stated verdict per rung; the "nothing helped" case reported as
readily as a win.

---

## Guardrails (unchanged, restated)

- **Never publish** `outputs/catatumbo_2023_coca.geojson` (2,065 plot-level 20 m
  polygons, targeting-usable) — and note it overclaims relative to 1 km labels. Public
  artifacts stay at 75 m COG + municipal aggregate.
- Every threshold and every new question is pre-registered as a dated amendment
  **before** it is computed. Nothing already published gets overwritten.
- Gates use input-side quantities only — never labels, ratios, or errors.
- Report gated **and** ungated, and state n every time (gating leaves n=4).
- Do not build the U-TAE temporal model (`src/models/temporal.py:13`).
- Keep `data/` and `.venv` out of git. Disk will roughly double — expected and accepted.

## Order of work and reporting points

**5.1–5.5 → 6.1 → [6.2] → 6.3 → 6.4 → 7 → 8 → 9**

Stop and report after: **6.1** (coverage hypothesis — decides whether Phase 8 is even
aimed at the right problem), **7** (did fixing supervision move cell-total accuracy and
concentration?), and **9** (the full ladder).
