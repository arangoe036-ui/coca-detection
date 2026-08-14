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

## Phase 6.6 — Fix the Sentinel-2 baseline-04.00 offset (NEW — now the top priority)

Phase 6.5 found the root cause: `stac_export.py:89` does `DN/10000` without subtracting
the `BOA_ADD_OFFSET = -1000` that ESA introduced with **Processing Baseline 04.00
(effective 2022-01-25)**. Correct conversion is `(DN − 1000)/10000`. All 2022–2024 S2
reflectances are inflated by ~0.1. Evidence: every visible band steps +0.100 at exactly
2021→2022 (= 1000/10000) while SAR is flat, and NDVI compresses 0.68→0.50, which is what
adding a constant to both NIR and Red does to a normalized ratio.

Mechanism: per-year mean-centering silently removes the constant from *raw* bands, but
the six index channels are nonlinear functions of the offset reflectances and are
computed **before** normalization — so no normalization can undo it. The offset must be
subtracted before indices are computed.

**This is a correctness bug, not an optimization. It precedes all remaining phases.**

### 6.6a Implement the correction
- Key the correction on the **processing-baseline / `BOA_ADD_OFFSET` metadata field**,
  not acquisition date. Scenes acquired pre-2022-01-25 but reprocessed later carry
  baseline 04.00 and the offset; a date rule mis-classifies them. Fall back to the date
  rule only where metadata is unavailable.
- Apply **per scene**, not per year — 2022 composites are a mix.
- Subtract **before** computing indices.

### 6.6b Regenerate and verify (model-free — do before any training)
- Regenerate 2022–2024 tiles (plus 2025/2026 nowcast inputs), **asserting the
  block→split assignment is byte-identical** to the previous index (§8.1 hard rule).
- Re-run `scripts/verify_channels.py` and update the documented channel table.
- **Acceptance without any model:** the +0.100 step at 2021→2022 disappears, and per-year
  NDVI returns to ~0.68 for 2022–2024. If it doesn't, the fix is wrong — stop.

### 6.6c Pre-register the differential prediction, then re-run LOYO
The sharp, falsifiable test: the fix should **improve 2022–2024 folds specifically and
leave 2019–2021 roughly unchanged.** If 2019–2021 shift materially, something else is
also wrong. Register this before re-running. Report how much of the original 0.271 spread
the fix removes.

**Known residual:** 2020 is a clean pre-offset year and still misses by −0.41. The offset
cannot explain it. Do not let a successful fix obscure this — it remains the open
question, and per-year normalization stays the leading suspect for it.

### 6.6d Audit the blast radius and annotate published results
The offset affects everything post-Jan-2022, including work already written up. Per prime
directive #3, **annotate — do not silently overwrite.** Add a dated caveat to
`BASELINE_LADDER_RESULTS.md` noting it was computed on contaminated 2022–2024 data, and
flag specifically that:
- **"NDVI is the floor"** is suspect — NDVI is the most corrupted channel, so part of its
  deficit may be the bug rather than physics.
- **"SWIR/NBR dominates, NDVI 15/18"** is suspect — NBR is also an index, also corrupted.
  The ranking may shift.
- Every LOYO fold is affected, including clean-year folds, since each **trains** on
  2022–2024.
- The method comparison remains internally fair (all methods saw the same corruption), so
  the *direction* U-Net > RF > NDVI is likely robust; the magnitudes are not.

Then re-run the baseline ladder after the fix and publish both tables side by side.

Also audit and re-derive: `calibration.json` (`fit_year: 2023`, fit on contaminated
data), `ui/data/density_*_cog.tif` for 2022+, `ui/data/municipal_coca_2023.csv` and the
0.05–1.69 range, the README's 2023 figures, the 2025/2026 nowcast, and
`outputs/catatumbo_2023_coca.geojson`.

**Acceptance:** model-free checks pass; differential prediction registered before the
re-run; LOYO re-run reported; published results annotated, not overwritten; blast-radius
audit committed.

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

### 8.3 HLS — STILL NOT BUILT, but the recorded reason was WRONG. Corrected 2026-08-11.

**The old rationale is known-false and is retained here only so the error is auditable:**
*"Phase 6.1 measured 32.7 clear observations/pixel in 2020 — the best of all six years …
optical coverage is ample. Adding more optical looks addresses a bottleneck that does not
exist."*

That reasoning used **mean/median clear-observations per pixel, which cannot detect a region
with ZERO observations.** In fact **25.3% of the 2020 AOI had no valid Sentinel-2 data at all**
(`docs/coverage_by_year.md`, annotated). A year can post the highest mean while a quarter of it
is unobserved, because the observed remainder is densely covered — which is exactly what
happened. So "coverage is ample" was false, and the coverage bottleneck was real.

**The decision not to build HLS nevertheless stands**, on a different and now-tested basis:
prereg A13 reinstated 8.3 only if the gap exceeded 15% after correcting our own filtering. It
was measured at **0.72%** (2020 and 2021, down from 25.25% and 12.18%). The gap was
self-inflicted — a scene-level `cloud_cover_max: 40` discarding whole partly-clear scenes — and
it closed in config. Adding a second sensor would buy nothing.

**Reopen condition, fixed in advance:** any year exceeding **5% blank** after the corrected
re-export. Enforced by `tests/test_data_invariants.py::test_blank_fraction_under_bar`, so this
reopens automatically rather than on someone's recollection.

The lesson worth keeping is not about HLS: **a decision to skip 30 hours of work rested on a
metric that was structurally blind to the thing it was measuring.** Prefer a metric that can
express zero.

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

---

## Phase 10 — 2026 forecast (DEFERRED — revisit only after Phase 9 consolidation)

Placeholder so the intent is not lost. **Do not start this before Phase 9 is done.**

Naive version to avoid: publishing a 2026 hectare total. That is exactly the capability
Track B measured as failing (loses to the N2 historical mean), from a partial year of
imagery, in a year that already produced a phantom decline
(`docs/coverage_check.md:14`), with no official census to check against until roughly
late 2027.

Version worth building, when the time comes:
1. **Forecast the ranking, not the magnitude** — which municipalities rank highest and
   where growth concentrates. This is the validated capability (spatial, 6/6).
2. **Magnitude as an interval from the anchor**, not the network — the Phase 6.3 hybrid,
   with the measured ±40% and an explicit statement that the total's accuracy comes from
   the historical anchor.
3. **Pre-register it with a commit hash and timestamp before the truth exists**, and
   state the scoring rule in advance (top-3 municipalities, direction of change, interval
   coverage). Score it publicly when SIMCI publishes, win or lose.
4. **Handle the partial year explicitly** — declared data cutoff, the existing coverage
   correction, and the encoded prior that coca is a standing perennial and does not halve
   mid-year. Consider a provisional mid-2026 nowcast plus a full-year estimate once the
   composite completes in early 2027.
5. **Ethics line:** municipality-level and aggregate only, never plot-level. A
   backward-looking map is monitoring; a forward-looking one drifts toward targeting.
   Frame as policy statistics, not operational intelligence — consistent with the
   existing 75 m / municipal-aggregate stance.

6. **Policy regime change breaks the anchor's core assumption — raised 2026-08-10.** A new
   Colombian government is expected to push coca *down*, which makes a genuine turning point
   likely rather than hypothetical. This matters because the hybrid anchor is a **historical
   mean**, and §5.3 of the ladder results already establishes that N2 **predicts zero change
   by construction** and is therefore blind to turning points — its single worst year was
   **2024 (0.85)**, exactly the largest real move (+11.1%). An anchor that lagged a real
   increase will lag a policy-driven decrease the same way, in the opposite direction.

   Consequences that must be designed for, not discovered afterwards:
   - **Do not publish a magnitude forecast that leans on stationarity.** Either state the
     stationarity assumption and its failure mode explicitly, or widen the interval to admit
     a policy-driven break — and say which was chosen, in advance.
   - **Rankings are more robust than totals, but not immune.** A *uniform* proportional
     decline leaves the ranking intact while destroying the total, which is the main argument
     for forecasting rank. But eradication and substitution programmes are typically
     **spatially targeted**, and targeted reduction reshuffles rank directly. So the ranking
     claim's robustness depends on whether the intervention is broad or concentrated —
     an empirical question, not something to assume.
   - **Consider making the direction claim the headline instead**, pre-registered with the
     policy reasoning stated: it is falsifiable, it is honest about resting partly on
     policy expectation rather than imagery, and it does not pretend the anchor can see a
     break it structurally cannot.
   - **Record the policy context with the forecast**, so a future reader can tell whether a
     miss came from the model, from the anchor's stationarity assumption, or from a policy
     shift nobody could have read out of Sentinel pixels.

The model does not need to win. If the baseline beats it again, report that — consistency
with the earlier findings is the credibility.

---

## Order of work and reporting points

**5.1–5.5 → 6.1 → [6.2] → 6.3 → 6.4 → 7 → 8 → 9**

Stop and report after: **6.1** (coverage hypothesis — decides whether Phase 8 is even
aimed at the right problem), **7** (did fixing supervision move cell-total accuracy and
concentration?), and **9** (the full ladder).
