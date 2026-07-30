# Baseline Ladder — Pre-registration

**Status:** FROZEN before any baseline produced a number. Committed at the end of
Phase 0, before Phase 2. Written 2026-07-29.

**One-line purpose:** decide, in advance, what each possible comparison outcome
*means*, so the conclusion cannot be reverse-fit to the numbers. Whatever the
folds return, the reading below is applied verbatim.

This file governs the interpretation. If a later phase needs to deviate from any
rule here, the deviation must be written into this file *and* flagged in the
results doc — silent changes void the pre-registration.

---

## Amendments (2026-07-29, logged BEFORE any baseline result)

Three amendments were made after the initial freeze, during user review of the
decision rules, **before any baseline model produced a number**. This is the
legitimate window to amend (no results exist to fit to). Each is logged with its
reason; the body below (§3, §5, §7) is revised to match.

**A1 — Add two no-skill nulls (reason: the ratio table is uninterpretable without
them).** `aoi_ratio` is a sum over the whole AOI, so all spatial accuracy cancels,
and `fit_scalar` normalizes the total on train years — so a predictor with *zero*
spatial or temporal skill can still post a good ratio, and S=0.27 is "good or bad
against *what*?" Two zero-cost nulls fix this:
- **N1 — constant density:** predict the train-years' mean density *uniformly*
  across the AOI, then run the **identical** `fit_scalar → test_year_ratio` path.
  No spatial skill, no temporal skill. (Gate handling: `TAU=0.05` gating a spatially
  uniform field is degenerate — it is all-or-nothing — so for N1 the gate is
  disabled and this is stated wherever N1 appears. N1 is the "does the pipeline
  itself add anything spatial" companion; it should land ≈ N2.)
- **N2 — historical mean (no model at all):** predict held-out-year hectares =
  mean of the *other* years' official hectares. This is the sharpest null: does the
  U-Net beat simply knowing the historical average? **N2 is the primary null the
  headline rests on.** N2 is deterministic from the frozen official hectares (§4),
  so its values are *already fixed* and are reported in Track B; the decision rule
  (§5) is set independent of them.

**A2 — Split into two labeled tracks; get the spatial comparison for FREE (reason:
`aoi_ratio` structurally cannot answer "is a segmentation model warranted" — that
question is spatial).** The held-out-year U-Net tile metrics do NOT require a
6-fold re-run. `outputs/checkpoints/final_multiyear.pt` exists; the block split
holds out `test` blocks spatially across *all* years, so evaluating that checkpoint
on the `test` blocks is leakage-free. Fit both baselines on the same `train` blocks,
evaluate all three on the same `test` blocks → a fair three-way spatial comparison
for one inference pass.
- **Track A — spatial (free):** `mae, rmse, presence_iou, presence_f1` on the `test`
  blocks. U-Net from the existing checkpoint. Answers *does spatial structure help?*
  → this is the track that speaks to "is the U-Net justified."
- **Track B — temporal (free):** `aoi_ratio` per LOYO fold + nulls. U-Net from the
  frozen v2.1 table. Answers *does it transfer across years?*
- Wherever the ratio (Track B) table appears, state explicitly that it is
  structurally incapable of judging spatial skill — that is Track A's job.

**A3 — Decide on paired per-fold differences, not a margin on a noisy std (reason:
the std of a std at n=6 is ≈ σ/√(2(n−1)) ≈ 0.27/√10 ≈ 0.085 > the 0.05 margin, so
`S<0.22` could trip on noise).** All methods run on the *same* folds, so pair them.
For each method compute per-fold `d_f = |ratio_f − 1|`; compare methods by the
paired delta `δ_f = d_f(other) − d_f(U-Net)` fold by fold. Report the mean paired
delta and all 6 signed `δ_f`, and require a **consistent sign in ≥5 of 6 folds** for
a "clearly better/worse" call. Pairing cancels the year-to-year difficulty that
inflates the variance. `C`, `S`, `K` remain **descriptive only** — the decision no
longer rests on a margin smaller than the noise.

---

## Amendments — Phase 5/6 (2026-07-29, logged BEFORE recomputing anything)

Logged during the Phase 5/6 build, each **before** the number it governs is
recomputed (hard rule: pre-register thresholds and new questions as dated
amendments). The Phase 3 `aoi_ratio` verdict (§4, Track B) stays published exactly
as it came out — these amendments *add* analyses and fix presentation, they never
overwrite it.

**A4 — Track A spatial metrics are CELL-LEVEL, not field-level (scope, not a rule
change).** `src/data/labels.py:107-115` burns one constant density
`coca_ha / cell_ha` into *every* 20 m pixel of each ~1 km official cell (uniform
by construction). Therefore Track A's presence-IoU/F1 measure agreement with which
**1 km cells** contain coca, not field-level localization — and nothing in this
repo can validate sub-cell placement against 1 km labels. Every spatial claim is
scoped to cell level hereafter. The 2,065-polygon 20 m artifact
(`outputs/catatumbo_2023_coca.geojson`) implicitly overclaims and stays unpublished.
This reframes, not retracts, the Track A win. Phase 7 replaces this supervision.

**A5 — Unified calibration convention for Track A (pre-registered before
recompute).** In Track A, predictions are compared on the **density-fraction
scale**. Trained regressors (U-Net, RF) already output fractions; the NDVI index
ramp does not, which is why its raw Track A MAE was 0.5458 (a scale artifact, not a
finding). Fix: map every method's prediction to the density scale by a single
train-years-only constant `a = Σ(train target) / Σ(train prediction)` (ungated,
computed on a fixed-seed train sample), then compute MAE/RMSE/bias and presence-
IoU/F1 (thr 0.02) on `a · prediction`. This is the same linear-calibration idea as
Track B's `fit_scalar`, applied to each track's own target scale (density for A,
hectares for B). `a` is recorded per method per run; before/after is reported. For
U-Net/RF `a ≈ 1` (already fraction-scale); for NDVI `a ≈ 0.08`.

**A6 — IoU reconciliation (5.1).** README's `IoU 0.665 / AP 0.877 (thr 0.504 tuned
on val)` is a **stale P0–P4-era figure**: it is the original single-year-2023
segmentation-style evaluation (`best.pt`; AP + a val-tuned threshold are produced
only by `_evaluate_segmentation`). The shipped pipeline is multiyear density
regression (`final_multiyear.pt`); its authoritative **cell-level** presence-IoU is
the Track A number (thr 0.02, per-year mean on held-out test blocks). The README is
corrected to carry one authoritative, clearly-scoped figure; the 0.665 is marked
superseded, not presented as the current headline.

**A7 — Phase 6 coverage hypothesis (new question, pre-registered before 6.1).**
Question: *are the two out-of-band U-Net folds (2020, 2022 — plus a check on 2024)
the degraded-imagery years rather than a model failure?* 6.1 is a **descriptive**
diagnostic at n=6 (no p-values, no regression significance): per year, number of
scenes and mean/median clear **S2** observations per pixel, and **S1** pass count,
measured **separately** (not conflated). Pre-registered readings: (i) 2020 & 2022
clearly lowest coverage → hypothesis supported; (ii) coverage unrelated to
|ratio−1| → hypothesis rejected (a real finding: the misses are a model problem);
(iii) partial → gate only the supported mechanism. Any coverage **threshold** for a
6.2 gate is a separate amendment, chosen from the coverage distribution **alone**
(input-side only, never from labels/ratios/errors) and logged before any ratio is
recomputed.

---

## Amendment — Phase 6.5 (2026-07-29, logged BEFORE computing the diagnostic)

**A8 — Magnitude-mechanism diagnostic (new question, pre-registered before any
number).** Phase 6.1 ruled out coverage as the cause of the out-of-year magnitude
failure (U-Net `aoi_ratio` std 0.271 ≫ official CV 0.075). Before intervening again
we diagnose the mechanism rather than assume it. Leading suspect: per-year
z-scoring (`src/train_loyo.py:165` normalizes the held-out year by its own
mean/std; `:180` computes per-year stats), which erases each year's **absolute
level**, so the model cannot perceive "this year is more disturbed overall than
usual." This would predict the exact observed split (localization works, counting
fails, errors uncorrelated with coverage, inconsistent sign, variance > target CV).

Tests (all n=6, descriptive — no p-values):
- **6.5a** correlate per-year normalization parameters (mean/std of the disturbance
  channels NBR/B12/B11 and NDVI) against per-year signed error and `|ratio−1|`; and
  check whether U-Net `predicted_ha` tracks `official_ha` at all (if the level
  signal is destroyed, it should not).
- **6.5b** swap-stats sensitivity: predict each year under its **own** stats vs
  **pooled** (all-year) stats and report the change in summed gated U-Net density.
  Uses `final_multiyear.pt` (per-fold LOYO models were not saved), so it measures
  *sensitivity*, not a fix; documented as such.
- **6.5c** decompose the 2020 & 2022 discrepancy by spatial block: concentrated ⇒ a
  specific confusion; diffuse ⇒ a global level shift consistent with normalization.
- **6.5d** sweep the `fit_scalar` presence gate `TAU`: if the total is hypersensitive
  to it, the gate is an amplifier converting small density shifts into large total
  swings.

Pre-registered outcomes (more than one may fire; report each test's contribution
and state what remains unexplained):
- **Normalization implicated** (6.5a correlation and/or `pred_ha` not tracking
  `official_ha` and/or 6.5b large swap-stats movement) ⇒ add a **normalization rung**
  to the Phase 9 ladder (retrain with pooled/global stats, or keep per-year stats and
  feed the year's absolute level back as auxiliary scalar inputs) — cheaper than the
  sensor build.
- **Gate implicated** (6.5d hypersensitivity) ⇒ fix the gate; report how much of the
  0.271 it explains.
- **Neither** ⇒ supervision (Phase 7) becomes the leading magnitude suspect **by
  elimination**, not assumption.

No already-published result is overwritten; the frozen Track B `aoi_ratio` verdict
stands. Phase 7's **spatial** justification is independent of this diagnostic (the
~1 km uniform labels cap validation at cell level regardless, A4); only Phase 7's
**magnitude** rationale competes with the normalization hypothesis.

---

## Amendment — Phase 6.6 (2026-07-29, logged BEFORE the LOYO re-run)

**A9 — Differential prediction for the S2-offset fix (registered before recompute).**
Phase 6.6 corrects the baseline-04.00 offset (+0.1 reflectance in 2022–2024),
recomputing indices from corrected reflectance. The sharp, falsifiable test is
**differential**, not "did the spread drop":

- **Prediction:** re-running LOYO on corrected data should **improve the 2022–2024
  folds specifically** (their `|aoi_ratio − 1|` decreases) and **leave 2019–2021
  roughly unchanged**. Formally: mean `|Δ ratio|` over {2022,2023,2024} ≫ mean
  `|Δ ratio|` over {2019,2020,2021}, and ≥2 of the 3 offset folds move toward 1.0.
- **Comparison baseline:** the published v2.1 per-fold ratios (0.95 / 0.59 / 0.90 /
  1.48 / 1.01 / 0.79). Caveat: v2.1 was produced in a possibly different
  environment, and MPS training is not bit-deterministic, so small clean-year moves
  may be run noise rather than the fix; a matched in-environment uncorrected control
  would net this out and can follow if attribution is unclear.
- **Coupling note:** every fold *trains* on 2022–2024, but per-year z-scoring makes
  the raw-band correction invisible after normalization (a constant shift is removed
  either way); only the six **index** channels change. So clean-year folds may move
  a little via changed 2022–2024 training indices, but should move less than the
  offset folds whose **test** indices are corrected.
- **Outcomes:** (i) offset folds improve, clean roughly unchanged, spread drops → the
  offset was a real magnitude cause; report how much of 0.271 it removes. (ii) offset
  folds do **not** improve → the offset was a genuine correctness bug but **not** the
  magnitude driver; report that plainly (a fixed bug that didn't move the metric is
  still a result). (iii) clean folds move as much as offset folds → confound or a
  second problem; investigate before claiming the fix worked.
- **Known residual (do not obscure):** 2020 is a clean pre-offset year and misses by
  −0.41; the offset cannot explain it. It should stay mostly bad; per-year
  normalization remains the leading suspect for that fold specifically.

No published result is overwritten; v2.1 and `BASELINE_LADDER_RESULTS.md` stand and
are annotated (6.6d), and the corrected re-run is published beside them.

---

## Amendment — Phase 6.6 gate diagnostic (2026-07-29, logged BEFORE the TAU sweep)

**A10 — Is the collapse the presence gate? (registered before computing).** The
corrected re-run destabilized the offset folds in *both* directions (2022 1.48→0.08,
2023 1.01→1.33) while clean folds were unchanged. Hypothesis: the presence gate is a
**bug**, not a design choice — `TAU=0.05` (`config/default.yaml:124`) is a *fixed
absolute* threshold applied to the output of a **per-year-normalized** model, so it
removes a *different fraction* of predicted mass each year; correcting the data
shifted the offset years' output distributions relative to where the gate + frozen
scalar were calibrated. This is a defect of the same class as the offset and must be
fixed in the **same** re-run (the correction is what exposed it).

**Test:** sweep `TAU` on the 2022 fold (re-trained; per-fold models were not saved)
and recompute `aoi_ratio(TAU)`, refitting the calibration scalar at each `TAU` (both
gated identically, train years only). Pre-registered branches:
- **Ratio recovers toward ~1 as TAU→0** ⇒ gate confirmed. Fix in 6.6: replace the
  fixed absolute gate with a **quantile of the train-year prediction distribution**
  (fit on train years only — input/train-side, never labels/test/errors). Normalization
  stays a Phase 9 ablation rung (it is a design choice: transductive robustness vs.
  preserving absolute level).
- **Ratio does NOT recover** ⇒ gate exonerated; normalization becomes a *correctness*
  issue and moves into 6.6 alongside the offset fix.
- **Partial recovery** ⇒ fix the gate in 6.6 and record in the Phase 9 prereg that the
  normalization rung is expected to carry the remainder.

**Reporting rule (do not overwrite the collapse):** when the gate is fixed, keep the
three-row progression, one change each — (a) v2.1 contaminated + fixed TAU
(annotated), (b) corrected + fixed TAU = **the collapse**, (c) corrected + quantile
TAU = new baseline. Row (b) is the co-adaptation finding and stays published.

Compute note: folds 2023/2024 are **not** to be re-run until the gate question is
settled (2023 already completed opportunistically: 1.33).

---

## Amendment — Phase 6.6 quantile gate (2026-07-29, logged BEFORE the test)

**A11 — Scale-invariant quantile gate (registered before computing).** The fixed
`TAU=0.05` gate cuts a different mass fraction each year because the per-year-
normalized model output shifts year to year (A10). Replacement gate: keep, in every
year, the **top `f_keep` fraction of pixels by predicted density**, where `f_keep` is
fit on **train years only** = the mean per-train-year fraction of pixels with
predicted density ≥ 0.05 (the current gate's average train keep-rate). The threshold
is then the `(1 − f_keep)` quantile of *each year's own* prediction distribution, so
the same fraction is gated every year — scale-invariant, input/output-side, no
label/test leakage. The calibration scalar is refit on the quantile-gated train
predictions.

**Test:** apply it post-hoc to the saved corrected 2022 fold (model
`loyo_fold`/`loyo_2022_corrected.pt` + cached test density) and recompute `aoi_ratio`.
Decision (same thresholds as A10): 2022 ratio **≥ 0.75** ⇒ the collapse was the gate
→ normalization reverts to a Phase 9 design-choice ablation; **< 0.40** ⇒ the model's
raw output collapsed and no gate fixes it → retrain once with the normalization fix
(**A8 option 2 preferred: per-year stats + absolute-level auxiliary inputs**, which
restores the discarded level as explicit features, keeps per-year transductive
robustness the 2026 nowcast relies on, and avoids the pooled-stats collapse 6.5b
warned of); **0.40–0.75** ⇒ fix the gate now and note the Phase 9 normalization rung
carries the remainder. Prior from A10: ungated (TAU=0) already only reaches 0.20, so
a gate that keeps ≤100% of pixels is not expected to recover — but the scalar refit
under a quantile gate can move it either way, so it is measured, not assumed.

---

## 1. What is being tested

Whether the U-Net is *scientifically justified* over two simpler predictors —
(A) a spectral-index (NDVI) threshold and (B) a context-free per-pixel random
forest — on the **identical** leave-one-year-out (LOYO) folds the U-Net already
uses (`src/train_loyo.py`). The honest headline the study is willing to publish
is that **a random forest ties the U-Net → ~1 km-resolution labels do not support
a 20 m segmentation model**, if that is what the numbers say.

## 2. Folds and leakage discipline (frozen)

- Folds: hold out each of 2019–2024 in turn; train on the other five years'
  **train** blocks; the held-out year is the test fold. Reuse `read_index`,
  `year_norm_stats`, and the block→split assignment from `src/train_loyo.py`
  unchanged. No second splitting routine.
- **Nothing** — NDVI threshold `t`, RF hyperparameters, per-year normalization
  stats, or the calibration scalar `s` — is fit using any held-out-year pixel.
  Thresholds/hyperparameters may be selected on the **train years' val blocks**;
  `s` is fit on train years via the existing `fit_scalar` (`train_loyo.py:142`).
- Each method uses its **own** calibration scalar, fit on train years by the same
  `fit_scalar` procedure. A good `aoi_ratio` is therefore not automatic — the
  scalar corrects the train-years AOI total, and the held-out year still tests
  out-of-year transfer.

## 3. Metric set (frozen; computed only with existing `src/evaluate.py` helpers)

Organized into two tracks (A2). All methods report the **same** metrics within each
track, computed only with the existing `_evaluate_regression` / `_metrics_at`
helpers (`evaluate.py:41,68`) — no new metric math.

### Track A — spatial skill (the "is a U-Net warranted?" track)

Domain: predicted density fraction (pre-scalar) vs. the mask fraction, on the
**`test` blocks** (spatially held out across all years; leakage-free per A2).
Methods: NDVI threshold, RF, and the U-Net checkpoint `final_multiyear.pt` — all
fit on the **`train` blocks**, evaluated on the **`test` blocks**.
- `mae`, `rmse`, `bias`.
- `presence_iou`, `presence_f1` via `_metrics_at(probs, targets, thr=0.02,
  t_thr=0.0)` — "present" = predicted fraction > 0.02; truth present = any coca.

**This is the track that answers whether spatial structure helps.** `aoi_ratio`
(Track B) is a sum over the AOI and is structurally *incapable* of judging spatial
skill; that limitation is stated wherever the Track B table appears.

### Track B — out-of-year magnitude transfer

Domain: absolute hectares, full held-out-year AOI, per LOYO fold.
- `aoi_ratio` = predicted ha / official ha for the held-out year, via the same
  `test_year_ratio` full-raster + gate (`TAU=0.05`) + `px_ha` + scalar path
  (`train_loyo.py:162`). Official ha are the frozen v2.1 figures (§4).
Methods: NDVI threshold, RF, U-Net (frozen v2.1 table), plus nulls **N1** and
**N2** (A1).

### PRIMARY metric: `aoi_ratio` (Track B), adjudicated per §5

Justification: it is the deliverable (out-of-year absolute hectares) and the only
held-out-year metric on disk for the U-Net. Track A's tile metrics are the
**spatial-justification** evidence. Per fold, `aoi_ratio` quality is described
(not decided — see §5/A3) by:
- **Centering** `C = |mean(ratio) − 1.0|` — systematic over/under-count.
- **Stability** `S = std(ratio)` across folds, with the min–max band.
- Companion count `K` = folds with ratio ∈ **[0.85, 1.15]** (the desired ±15 % band).

`C`, `S`, `K` are **descriptive**; the decision uses paired per-fold deltas (§5).

## 4. U-Net reference numbers (frozen, from `docs/v2.1_loyo_results.md`)

Per-fold `aoi_ratio` (Track B): 2019 = 0.95, 2020 = 0.59, 2021 = 0.90, 2022 = 1.48,
2023 = 1.01, 2024 = 0.79.
→ **mean 0.95, C = 0.05, S = 0.27, band 0.59–1.48, K = 3/6** (2019, 2021, 2023).

For Track A (spatial), the U-Net's `mae`/`presence_*` are obtained fresh from
`final_multiyear.pt` on the `test` blocks (A2), not from disk. The v2.1 "val MAE"
(~0.02) is a train-years val-block number and is **excluded** from all comparisons
(it is not a held-out spatial-test number).

Null N2 is deterministic from the frozen official hectares above; its per-fold
values are therefore already fixed and are reported (not pre-stated here) in the
Track B results, where the §5 rule — set independently of them — is applied.

## 5. Decision rule — "is the U-Net justified?" (frozen; paired, per A3)

With n = 6 folds we report `mean ± std`, min, max, and K **descriptively**. We
**do not** compute p-values or confidence intervals — 6 folds do not support them,
and (A3) the std of a std at n=6 (≈0.085) exceeds any sensible margin, so no
decision rests on a margin over `S`.

**Track B (PRIMARY, magnitude) — paired per-fold rule.** All methods share the 6
folds. For each method compute per-fold `d_f = |ratio_f − 1|`. Compare method X
against the U-Net by the paired delta `δ_f = d_f(X) − d_f(U-Net)` (δ_f > 0 ⇒ the
U-Net is closer to 1 on that fold). Report the mean paired delta and all 6 signed
`δ_f`.
- **U-Net clearly better than X:** `δ_f > 0` in **≥ 5 of 6** folds.
- **X clearly better than the U-Net:** `δ_f < 0` in **≥ 5 of 6** folds.
- Otherwise **indistinguishable at n=6** (a tie — Case 3/4 territory).

Applied to every baseline **and to both nulls**. The decisive question A1 poses:
**does the U-Net beat N2 (historical mean) by the ≥5/6 rule?** If not, the model
adds nothing over the historical average for the censusless-year deliverable.

**Track A (spatial justification) — paired rule on the test blocks.** Per-year (the
6 years give 6 paired points on the shared `test` blocks) compute each method's
`presence_iou` and `mae`; the U-Net **wins spatially** iff its `presence_iou`
exceeds the best baseline's in **≥ 5 of 6** years (and likewise lower `mae`). A
weaker/mixed result is a spatial tie — and a spatial tie is the strongest possible
evidence that the labels don't carry segmentation-grade spatial information.

## 6. Outcome cases → conclusions applied verbatim

The two tracks answer two different questions; the write-up states the reading for
each. Track B has a **dominating headline case** that is checked first:

- **Case 0 — the U-Net does not beat N2 (historical mean) on Track B** (i.e. N2 is
  not clearly worse by the ≥5/6 rule). **This dominates every other Track-B
  reading.** *For estimating hectares in a censusless year, the U-Net does not
  improve on simply predicting the historical average of past official counts — the
  imagery-based model adds nothing to the magnitude deliverable.* Report this as the
  central finding; it reframes the project from "I trained a segmentation model" to
  "I measured whether ~1 km labels can support one, and for the magnitude task they
  cannot beat a no-model baseline." Track A is then the only place the U-Net could
  still be justified (as a *localizer*, not a *counter*).

If the U-Net *does* clearly beat N2, adjudicate the remaining cases:

- **Case 1 — U-Net clearly best on both tracks.** Deep learning is earned. Report
  the paired margins (Track B δ, Track A ΔIoU). The segmentation framing stands.

- **Case 2 — U-Net wins spatially (Track A) but ties on magnitude (Track B).** Now
  testable for free via `final_multiyear.pt` (A2). The model **localizes** better
  than it **quantifies**; scope every claim to coca *location / ranking*, and state
  plainly that absolute out-of-year hectares are no better than a simpler predictor.

- **Case 3 — a random forest ties or beats the U-Net on Track A (spatial).**
  **This is the headline, not a failure:** *at ~1 km label resolution, a
  context-free per-pixel random forest matches a spatial segmentation model even on
  the spatial metrics — the labels do not carry enough spatial information to
  justify a U-Net.* Report it as the finding; do not tune the U-Net in response.

- **Case 4 — the NDVI threshold alone is competitive on Track A.** The task is
  largely a vegetation-index problem; say so plainly, and treat both the RF and the
  U-Net as unjustified complexity.

Precedence: **Case 0 is checked first and dominates the magnitude story.** Among
1–4, if RF and NDVI both tie the U-Net spatially, Case 4 dominates (simplest
sufficient method wins the framing). If no baseline/null ties or beats the U-Net on
either track, Case 1 fires.

## 7. Scope of what is run (updated per A2 — no re-run needed)

Per A2 the spatial (Track A) comparison is obtained for free from the existing
`final_multiyear.pt` checkpoint evaluated on the `test` blocks — the 6-fold LOYO
re-run originally contemplated is **not required** and is **not** being run. All
three methods (+ the U-Net checkpoint) are fit on the `train` blocks and evaluated
on the `test` blocks for Track A; Track B reuses the frozen v2.1 LOYO ratios plus
the two nulls. No U-Net weights, losses, or tiling are modified.

## 8. Integrity constraints (frozen)

- No tuning of anything toward a nicer table. If a baseline wins, that is the result.
- Baselines' `t` / hyperparameters differ per fold by construction (proof they are
  refit, not hardcoded); each fold records a written assertion that no held-out-year
  pixel influenced them.
- RF RNG seed fixed and recorded; feature importances saved per fold.
- Every reported number traces to a row in `outputs/metrics/baseline_ladder.jsonl`.

## 9. Random-forest feature-importance reading (frozen intent)

RF importances are reported descriptively, not as a decision input. Pre-registered
reading: if NDVI (idx 10) and the SWIR bands (B11 idx 8 / B12 idx 9) dominate, that
corroborates Case 4 (a spectral-index problem); if importance is spread across many
bands with no spatial term available, it underlines that per-pixel spectra alone
carry most of the signal the labels can express.
