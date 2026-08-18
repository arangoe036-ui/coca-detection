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

## Amendment — Phase 6.6 level-signal check + aux design (2026-07-29, before computing)

**A12 — Does a per-year input statistic actually carry the cross-year level signal?
(gates the retrain).** Before spending the one retrain on absolute-level aux inputs,
verify the signal exists. On the CORRECTED data, compute each candidate per-year
statistic (mean and std of NBR, B12, B11, NDVI) and correlate it (Pearson, n=6,
descriptive — no p-values) against official hectares.
- **Decision:** if a candidate (esp. NBR/B12 **mean**) tracks official total with a
  clear, directionally-sensible correlation, that is a real level signal → proceed to
  the retrain using *that* feature; include **std** only for candidates whose std also
  correlates. If **nothing** correlates → **DO NOT RETRAIN** — the aux inputs would
  restore a signal that does not exist; report that and route counting to the Phase 6.3
  hybrid anchor instead.

**Aux-input spec (only if A12 supports it), pinned to avoid year-ID memorization:**
- **2–4 scalars maximum** (not 16–34). With only 6 distinct per-year values (LOYO sees
  5), many correlated per-year scalars let the model use them as a **year ID** and
  memorize a 5-row lookup, extrapolating arbitrarily on the held-out year — leakage-free
  but harmful. Default: NBR per-year mean+std (+ B12 mean only if A12 supports).
- **Anomaly-coded vs the TRAIN-year pool:** `aux_y = (stat_y − pool_mean_train) /
  pool_std_train` (pool stats train-only → label-free; an unseen year lands in an
  interpretable range, not out-of-distribution). Broadcast as constant planes;
  `in_channels` 18 → 20 or 21.

**Memorization falsification test (pre-registered, run with the retrain):** shuffle the
aux values across years — if performance holds, the model used them as a year ID, not a
level signal (invalidates the aux approach). Cheap variant: predictions must vary
*smoothly* under synthetic aux perturbation.

**Zero-dimensional-cost alternative (fallback if aux risks memorization):** mixed
normalization — per-year z-scoring for most channels, but **pooled train-year stats for
the 1–2 disturbance channels**, so level enters through existing channels with no new
inputs and no year-ID risk (couples level and spatial pattern in that channel).

---

## RESULT — A13 verdict (2026-08-11): PASS on the full composite

The registered metric has now been computed on complete annual composites, not the probe box,
so the procedural deviation logged below is **discharged**.

| year | blank % before | blank % after | reading |
|---|--:|--:|---|
| 2020 | **25.25** | **0.72** | remaining fraction is benign mosaic edge (S1 also invalid under it: 0.3%) |
| 2021 | **12.18** | **0.72** | same |

**A13's `< 5%` branch fires:** the missing quarter of the AOI was caused by **our own
scene-level cloud filter**, not by a gap in the archive. Both years land on the same 0.72% that
the already-clean years (2022–2024) had, i.e. full recovery. Consequences per the registered
rule: proceed with the full re-export; **Phase 8.3 (HLS/Landsat) is NOT reinstated** — the
`> 15%` archive branch did not fire, so honour the rule rather than the intuition. What must
change is 8.3's *recorded rationale* in `PHASE6_9_MASTER_PLAN.md` ("optical coverage is ample"),
which is known-false; replace it with "the gap was self-inflicted and closed in config; reopens
if any year exceeds 5% blank after re-export."

**A15 also passes so far** (2020↔2021 adjacent pair): no visible-band median step above the
registered 0.010 bar. Both A13 and A15 are now *enforced in code* by
`tests/test_data_invariants.py`, which passed 5/5 against the real composites — they are no
longer checks someone has to remember to run.

Reproduce: `python scripts/blank_footprint.py`, `python -m pytest tests/test_data_invariants.py`.

**Operational note, recorded because it cost a night.** The 2026-08-10 run died when Planetary
Computer SAS tokens expired: `planetary_computer` reuses a token within a process, so retrying
returned the same expired token and HTTP 403 forever. Three workers each stopped one second
before their token's `se=` expiry. Raising `cloud_cover_max` to 80 tripled per-sub-tile time and
made crossing a token window inevitable — a correctness fix with a throughput side-effect nobody
priced. Now handled: expired credentials exit 75 and `scripts/run_export.ps1` relaunches with
fresh credentials, resuming from completed sub-tiles. See `KNOWN_DEFECTS.md` F7.

---

## Amendment — location-only scope + the missing spatial null (2026-08-10, BEFORE any re-run)

**Scope narrowed by the owner, 2026-08-10:** the deliverable is now **where coca is** (presence
map + municipal ranking). **Total hectares are out of scope.** Track B's counting question, the
Phase 6.3 hybrid anchor, the calibration scalar and the nowcast are all retired as deliverables.
This is a reduction in claims, not a change in method.

**A16 — the spatial claim needs a no-skill floor, and it does not currently have one.**
Track B's central lesson was that a no-model null (N2, the historical mean) beat the network at
counting, and that this was only discovered because the null was actually run. **Track A has no
equivalent null.** The random forest and NDVI threshold are *imagery* baselines; neither is a
no-skill floor. Coca is a standing perennial and fields persist year to year, so the obvious
floor is prior location — and it has never been computed.

Register two persistence baselines, to be run on the identical corrected, leak-free folds:
- **P1 — last-observed cell mask.** Predict cell presence in year *t* as the presence mask of
  the most recent available train year. No imagery is read at all.
- **P2 — historical per-cell presence frequency.** For each ~1 km cell, the mean presence over
  all train years, thresholded at the same cut used for the other methods.

**Fair-comparison note, stated in advance so it cannot be argued afterwards:** P1/P2 consume
*past labels*, whereas the U-Net/RF/NDVI consume *imagery*. That is not an unfair advantage —
it is the operationally honest comparison, exactly as N2 was for counting: any real deployment
would already hold the last published census. A model that cannot beat "it is where it was" adds
nothing over consulting the previous survey.

**Decision, fixed now (same paired ≥5/6 rule as the original prereg):**
- U-Net beats the **better** of P1/P2 on cell-level presence IoU in **≥5/6 folds** ⇒ the spatial
  claim is earned and publishable as a model result.
- **3–4/6** ⇒ report as indistinguishable from persistence. The publishable claim shrinks to
  "reproduces the official spatial pattern," with persistence named as an equally good method.
- **≤2/6** ⇒ **the spatial claim is not publishable as a model result.** Report it as a negative
  result with the same prominence as the counting one. Do not retune the U-Net in response.

**A17 — municipal ranking is now a headline deliverable, so its metric is fixed in advance.**
Report Spearman ρ between predicted and official municipal hectares, **top-2 and top-3 exact
hit-rate**, and the count of adjacent inversions — per year, never pooled, always with n stated
(n=8 municipalities have official values; 2 of 10 do not). With n=8, ρ is one swap away from a
materially different value, so **ρ alone must never be quoted without the inversion count.**
Persistence applies here too: also report the ranking obtained by simply reusing the previous
census's ranking. Same ≥5/6-style reading.

**A18 — hard floor on published resolution.** `src/nowcast.py:write_cog` takes `long_side` as a
**pixel count**, so the 75 m figure is arithmetic on the current AOI, never asserted. For any AOI
narrower than ~30 km the same code upsamples past native 20 m and writes it to a tracked path
with no error, and `config` already offers `region: tumaco` as a live switch. **Any code path
writing a raster to a tracked directory must assert ground resolution ≥ 75 m and fail loudly
otherwise.** Emergent compliance is not compliance.

---

## Amendment — A13 deviation + directional predictions (2026-08-10, logged BEFORE the export lands)

**A13 procedural deviation, recorded rather than buried.** A13's decision rule required
recomputing the blank fraction **on the new composite**. The decision to proceed to a full
re-export was in fact taken on a **7.4 km probe box** placed at the deepest point of 2020's
blank region (100% blank pre-fix, 16.8 km from the nearest valid pixel), which returned
**0.00% blank / 100% valid S2** with physically sensible band medians. That is strong evidence
but it is *not* the registered metric. Per A13's own terms a silent substitution would void the
pre-registration, so it is logged here instead, and the formal gate is restored: **2020 is being
exported first, and its full-composite blank fraction is the number A13's thresholds apply to.**
The remaining five years run concurrently rather than idling the machine; if the 2020 composite
misses the <5% bar they are killed and A13 is re-adjudicated on the composite figure.

**A14 — predicted DIRECTION of the fix's effect (the strongest available test).** Track A
per-year cell-level presence-IoU and the blank fraction line up almost rank-perfectly:

| year | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 |
|---|--:|--:|--:|--:|--:|--:|
| blank % | 25.3 | 25.3 | 12.2 | 0.7 | 0.7 | 0.7 |
| Track A IoU | **0.312** | **0.352** | 0.529 | 0.584 | 0.550 | 0.519 |

The two most-blanked years are the two worst spatial years. So the defect plausibly explains the
*spatial* pattern too, not only the counting failure. **Registered prediction, before the data
exists:** on corrected data **2019 and 2020 Track A IoU rise substantially toward the 0.52–0.58
band** occupied by the clean-footprint years, while 2022–2024 stay roughly unchanged.
- **Both rise and the spread narrows** ⇒ blank coverage was a dominant driver of the apparent
  per-year spatial variation, and the aggregate IoU 0.474 was depressed by our own preprocessing.
- **They do not rise** ⇒ the blank-coverage explanation for the spatial pattern is **wrong**, and
  2019/2020 are genuinely harder years for reasons still unidentified. Report as such.
- **They rise past 0.58** ⇒ suspicious; check for label/mask misalignment introduced by the
  re-export before claiming an improvement.

**A15 — guard against swapping one artifact for another.** Raising `cloud_cover_max` 40 → 80
roughly doubles admitted scenes (152 → 289 for 2020). 2022–2024 were **already 99.3% covered**
under the old setting, so for them the change adds only marginal, hazier scenes and SCL does not
catch all thin cirrus/haze. **Acceptance check, threshold fixed now:** for 2022–2024 the median
of B02/B03/B04 must move by **< 0.010** versus the archived offset-corrected composites
(`data/imagery/prefix_A13/`). For scale, the entire observed interannual range of B02 medians is
0.0077, and the corrected 2021→2022 step is +0.0012. A shift beyond 0.010 therefore exceeds all
natural year-to-year variation and would mean the relaxed filter has injected a **new**
year-correlated quality artifact — the same class of defect as the two already found. If it
fires: stop, do not re-run the ladder, and test an intermediate threshold (60).
**Also re-run `scripts/acceptance_6_6b.py` afterwards rather than assuming the offset verdict
carries over** — more admitted scenes means more mixed-processing-baseline scenes in 2022.

**Data generations, to be stamped in every metrics record and doc header from now on:**
`gen1` original (offset bug + blank coverage), `gen2` offset-fixed only (2026-08-10, archived in
`data/imagery/prefix_A13/`), `gen3` offset + coverage fixed (this export). Cross-generation
comparison without an explicit generation label is the next confusion waiting to happen.

---

## Amendment — blank-coverage pilot (2026-08-10, logged BEFORE the export)

**A13 — Is the missing Sentinel-2 footprint caused by our own filtering, or by the archive?**

**The defect.** `config/default.yaml`'s `scl_mask_classes: [3,8,9,10,11]` omits SCL **0
(NO_DATA)** and **1 (SATURATED)**, so a pixel whose annual series is mostly nodata takes a
reflectance median of *exactly* 0.0 and (via the `clip(min=0)` index guard) indices of exactly
0.0. Measured fraction of the AOI affected: **2019 25.3%, 2020 25.3%, 2021 12.2%, 2022–2024
0.7%**. Verified to be real S2 loss rather than mosaic edge fill — inside 2019's blank region
Sentinel-1 VV is 97.2% valid at −7.14 dB, whereas in 2022's 0.7% region VV is 0% valid.

**Two candidate causes, and they imply different fixes.**
1. *Our filtering.* `cloud_cover_max: 40` rejects **whole scenes** by `eo:cloud_cover`, so a
   45%-cloudy scene is discarded even where it is clear. In a cloudier year this can starve a
   region of observations entirely. Per-pixel SCL masking makes scene-level pre-filtering
   largely redundant, so this is fixable in config.
2. *The archive.* There may simply be too few usable acquisitions over that region in those
   years, in which case no amount of filtering relaxation helps and the remedy is a second
   optical sensor.

**Pilot (cheap, one year).** Set `scl_mask_classes` to include 0 and 1, raise
`cloud_cover_max` to 80, and re-export **2020 only** — the worst LOYO fold and joint-worst
blank fraction. Recompute the blank fraction on the new composite. No model, no training.

**Decision, fixed before computing** (metric = % of AOI with all ten S2 reflectance bands
non-valid; note the corrected pipeline makes nodata NaN rather than 0, so the test is on
validity, *not* on `!= 0`):
- **< 5%** ⇒ our own scene-level filtering was the dominant cause. Proceed to the full 6-year
  re-export, then re-tile and re-run the ladder.
- **> 15%** ⇒ the archive lacks usable acquisitions there. Config cannot fix it. **Phase 8.3
  (HLS / Landsat-harmonised) is reinstated as a genuine requirement rather than the
  coverage-driven idea that Phase 6.1 deleted**, and the full re-export is deferred until that
  design decision is made.
- **5–15%** ⇒ partial. Report both, proceed with the full re-export, and keep Phase 8.3 open
  with the residual gap stated.

**Recorded in advance as already-established consequences, so they are not presented later as
new discoveries:** Phase 6.1 rejected the coverage hypothesis using mean/median clear
observations per pixel, which cannot detect a zero-observation region, so that rejection does
not hold (`docs/coverage_by_year.md`, annotated). The blank-coverage years (2019–2021) are
exactly the non-offset years and the offset-inflated years (2022–2024) are exactly the
clean-footprint years, so the two defects act in opposite directions on opposite year groups
and the v2.1 LOYO headline (mean 0.95, std 0.271) is confounded. **The conclusion that the
model cannot out-count the N2 historical-mean null is therefore provisional and must be
re-measured.**

**Not bundled into this fix:** an observation-count / validity input channel (Phase 8.2) is now
well motivated, but it changes `in_channels` from 18 and would break comparability with every
existing measurement. It is a separate pre-registered rung *after* the data correction. Same
for `encoder_weights: imagenet` (the encoder is currently randomly initialised).

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

---

## Amendment — A19: how A16's persistence comparison is scored (2026-08-13, BEFORE it is computed)

A16 registered the persistence nulls and the ≥5/6 decision rule but left three things
under-specified. Each is fixed here, in writing, *before* any persistence number exists,
because every one of them is a place where a later choice could be made to favour the model.

**1. "Cell-level presence IoU" is operationalised as the existing pixel-level presence IoU.**
`src/evaluate.py:_metrics_at(probs, targets, thr=0.02, t_thr=0.0)` — the same function, the
same two thresholds already used for `presence_iou` by the U-Net, RF and NDVI arms. The label
mask is rasterised from the official ~1 km grid and is therefore **constant within a cell**, so
a pixel-level IoU against it *is* a cell IoU, area-weighted by how much of each cell falls
inside the test blocks. Rationale for not building a second, unweighted per-cell metric: it
would be a *different* number for the three arms already computed, and having two candidate
granularities on the table is exactly the freedom this document exists to remove. One metric,
one implementation, all five methods.

**2. P1's year for the 2019 fold.** "Most recent available train year" does not exist for 2019
(no 2018 data). P1 for 2019 therefore uses **2020** — the nearest available year, which is in
the *future*. This is stated as a deliberate choice, not an oversight: P2 is already
non-causal by construction (it averages all five other years, including later ones), and
handing persistence a non-causal year can only make the null **stronger** and the model's bar
**harder**. It is never the direction that flatters the model.

**3. The persistence score is the per-fold maximum over three variants**, not one:
- `persistence_last_year` — P1, the prior year's presence mask.
- `persistence_freq_ever` — P2 thresholded literally at the registered 0.02 cut, i.e. "coca in
  this cell in **any** train year".
- `persistence_freq_majority` — P2 at 0.5, i.e. "coca in this cell in **most** train years".

P2's field is a *frequency*, and applying a *density* cut of 0.02 to it collapses to "ever
present" — a very inclusive mask (high recall, low precision). The majority variant is the
opposite failure mode. Rather than pick the one that happens to score lower, **the fold's
persistence score is the best of the three**, so adding variants can only raise the bar the
U-Net has to clear. Declaring the max in advance is the anti-shopping form of this choice.

**Applies unchanged from A16:** U-Net beats the persistence score in ≥5/6 folds ⇒ publishable
as a model result; 3–4/6 ⇒ indistinguishable from persistence; ≤2/6 ⇒ **not publishable as a
model result**, reported as a negative result with equal prominence. No retuning in response.

**One asymmetry recorded for the record, in the model's disfavour:** the U-Net's Track A
`presence_iou` is computed after the A5 density-calibration multiplier `a` (fit on train blocks
only, so leakage-free), which shifts its effective presence cut to 0.02/a. `a` is close to 1,
so the effect is small, but it is not zero and it is not being removed — the calibrated path is
the published pipeline and re-deriving an uncalibrated variant for this one comparison would be
a second bite at the metric.

---

## Amendment — A20: the evaluation split is rebuilt on stratified macro-blocks (2026-08-14, BEFORE any metric is recomputed)

Written before any metric was recomputed on the new split. The gen4 index
(`data/tiles/multiyear_index.csv`, `data_generation: gen4`) exists as of this amendment;
**no metric row has been produced from it**, and `outputs/metrics/baseline_ladder.jsonl` does
not exist.

**On the honesty of the bars below, stated plainly rather than claimed.** The *design* — 5×4
macro-blocks in tile-index space, dual-objective greedy into 6 folds, no buffer, 4/1/1 roll-up
— was fixed before it was run. The *numeric bars* in §"Acceptance bars" were **written after
the first measurement**, with round-number headroom above it (4.0 pp against 2.24/2.14 pp
measured; a 70% retention floor against 75.8% measured). They are therefore honest guards
against a future regression, and they are **not** evidence that the design was validated
against a threshold set blind. What *is* pre-registered in the strict sense is that both the
leak-free **and** the signal-bearing conditions must hold at all — that pair is the thing gen3
lacked and no measurement can move it. Amendment A19's standard (fix the rule before the number
exists) is met for the decision rules; it is only partially met for these bars, and saying so is
cheaper than being caught implying otherwise.

### Why this amendment exists

The gen3 split rule — contiguous west→east bands of 10 km blocks, ~70/15/15 by tile count —
was itself the fix for defect F3 (splits assigned by a tile's top-left corner while its 256 px
footprint spanned two blocks, putting 13.67% of test pixels inside train tiles). Bands genuinely
removed that leak: train ∩ test pixel overlap is 0, verified. **They also produced a `test`
split containing zero coca pixels** in all six years (blocker B1). `src/evaluate.py:47` is
`tp / (tp + fp + fn + 1e-6)`, so an all-negative target with an all-negative prediction returns
`0.000` with no warning. Every Track A `presence_iou` on gen3 — U-Net, RF, NDVI **and** the A16
persistence nulls, which read the same rows — is `0/0` dressed as a measurement. gen3 Track A is
void, not weak.

**Contiguous bands cannot work for this AOI in any orientation.** The positive-pixel fraction
pooled over all six years falls off steeply in *both* axes:

```
x=   0  0.291   x=1568 0.699   x=2912 0.519   x=3808 0.017   x=4704 0.000
x= 896  0.387   x=2016 0.729   x=3136 0.322   x=4032 0.099   x=4795 0.000
y=   0  0.000   y=1120 0.254   y=2688 0.487   y=4256 0.624   y=5372 0.300
```

Coca in Catatumbo is a west-central blob with empty margins east and north. An east→west band
split empties the eastern band; **a north–south band split would have emptied the northern one
for exactly the same reason.** Rotating the bands is not a fix; the contiguity is the defect.

### The registered design

Every position below is a tile position from `src.data.tiling._windows(5051, 5628, 256, 224)`:
**23 × 25 = 575** positions, identical in all six years.

1. **Macro-blocks in tile-index space.** Cut the 23 sorted-unique x positions into **5** blocks
   and the 25 y positions into **4**, via `np.linspace(0, n, k + 1).round().astype(int)` — edges
   `x = [0, 5, 9, 14, 18, 23]`, `y = [0, 6, 12, 19, 25]`, giving **20 macro-blocks**. Cutting in
   *tile-index* space, never in pixel space, is what makes every macro-block a whole number of
   existing tile positions; a pixel-space edge would land mid-position and reintroduce the F3
   straddler problem at a second granularity.
2. **Block weight = coca content.** A macro-block's weight is the sum over its positions of the
   **maximum positive-pixel count across the six years**, counted exactly from the `.npz` mask
   arrays (the index CSV's `pos_frac` is rounded to 4 dp and is not exact enough to weight a
   block). The max over years, rather than a per-year or mean value, keeps the assignment
   invariant to which years happen to be built — a fold must not move when a year is added.
3. **Dual-objective greedy into 6 folds.** Sort macro-blocks by descending weight and give each
   to the fold currently minimising the **joint normalised load**
   `coca[f]/total_coca + tiles[f]/total_tiles`. This is a least-loaded/LPT greedy, fully
   deterministic, no RNG, order-independent. Balancing coca *alone* (a fixed snake) met the coca
   objective but produced fold sizes from 27 to 204 tiles, which makes fold rotation for the A17
   municipal ranking meaningless; the joint objective is registered instead.
4. **Overlap resolution, no buffer.** `stride_px` (224) < `tile_px` (256), so neighbouring
   positions — diagonals included — share 32 px. Two positions share pixels iff
   `abs(dx) < 256 and abs(dy) < 256`. Where such a pair straddles a fold boundary, one of the two
   is dropped: repeatedly drop the position with the most still-unresolved cross-fold conflicts,
   ties to the **lowest** coca weight (so signal is preserved), then to the **lowest `(x, y)`**.
   **No buffer is added beyond that single dropped position** — see the trade-off below.

   The `(x, y)` direction is part of the registered rule, and the reason is *not* "for
   determinism" — every total order is deterministic. Six orderings were swept, and **all six
   clear every acceptance bar in this amendment**, so the choice is a tie-break among acceptable
   options and not a tuned result. Ascending `(x, y)` is registered because it is the best of the
   six on **both** registered balance objectives:

   | drop order | kept | retention | tile dev | coca dev | kept coca weight | share of AOI coca | tiles on a fold boundary | min gap | bars |
   |---|--:|--:|--:|--:|--:|--:|--:|--:|:--:|
   | **`(x, y)` ascending** | **436** | **75.8%** | **2.14 pp** | **2.24 pp** | 10,641,551 | 73.89% | 51.1% | 192 px | pass |
   | `(x, y)` descending | 438 | 76.2% | 2.51 pp | 3.27 pp | 10,687,946 | 74.21% | 49.3% | 192 px | pass |
   | `(y, x)` ascending | 437 | 76.0% | 2.56 pp | 3.46 pp | 10,715,837 | 74.40% | 49.9% | 192 px | pass |
   | `(y, x)` descending | 438 | 76.2% | 2.28 pp | 2.85 pp | 10,577,970 | 73.45% | 49.1% | 192 px | pass |
   | window index ascending | 437 | 76.0% | 2.56 pp | 3.46 pp | 10,715,837 | 74.40% | 49.9% | 192 px | pass |
   | window index descending | 438 | 76.2% | 2.28 pp | 2.85 pp | 10,577,970 | 73.45% | 49.1% | 192 px | pass |

   The two "window index" rows are **identical** to the `(y, x)` rows, not independent evidence:
   `_windows` yields y-outer/x-inner, so window order *is* `(y, x)` ascending. Four distinct
   orderings were therefore compared, under six labels. The registered ordering costs **two
   positions** (436 vs 438) and 0.3 pp of retained coca weight, and buys 0.14–0.42 pp on tile
   balance and 0.61–1.22 pp on coca balance. It also has the highest boundary-adjacency of the
   four (51.1% vs 49.1–49.9%), which is disclosed below rather than netted off.
5. **Roll-up, with the fold id retained.** 4 folds → `train`, 1 → `val`, 1 → `test`. The
   per-position fold id is written to the index as **`block_fold`** so the folds can be **rotated**
   later and every part of the AOI can be scored out-of-fold for the A17 municipal ranking (which
   is the open contamination problem in `src/infer.py:municipal_hectares`). The column is
   deliberately **not** named `fold`: `src/metrics_io.py` and `src/baselines/compare.py` already
   use `fold_year` for the leave-one-year-out fold and `compare._dedup_last` keys on it, so a
   `fold` collision could silently de-duplicate rows from different spatial folds against each
   other. `block_fold` is the name in the CSV column, the meta sidecar and the info dict.

Implementation: `src.data.tiling.assign_block_folds` / `split_for_block_fold` /
`split_pixel_report` / `verify_split_pixels`, driven by
`python -m src.data.multiyear --rebuild-block-folds`, which is **the terminal step of any
rebuild** — `build_all` (the re-tiling path) still writes a band split and now refuses to
overwrite a block-fold index without an explicit `--force-band-split-index`. The gen3 band rule
(`assign_splits`, `_band_cuts`, `_col_split`) is annotated as superseded and **kept**, with its
tests, as the record of the F3 fix. Each superseded index and sidecar is archived beside the new
one rather than overwritten: `multiyear_index_gen3_bands.csv` (bands) and
`multiyear_index_gen4_xy_desc.csv` (the descending-tie-break variant above), with matching
`_meta_*.json`.

**No re-tiling was required** and the net position count is worth stating plainly, because the
retrain that follows is sized by it. All 575 window positions already exist on disk as
`data/tiles/<year>/tile_{k:05d}.npz`, so the 75 positions gen3 discarded as band straddlers cost
nothing to reconsider — but only **65 of those 75 survive** the gen4 overlap resolution, while
129 positions gen3 had kept are dropped. The net:

| | gen3 (bands) | gen4 (A20) | change |
|---|--:|--:|--:|
| tile positions | 500 | **436** | **−12.8%** |
| index rows (× 6 years) | 3000 | 2616 | −12.8% |
| train rows | 2250 | **1752** | **−22.1%** |
| val rows | 450 | 384 | −14.7% |
| test rows | 300 | **480** | **+60.0%** |

**gen4 is a smaller training set and a larger, non-empty test set.** That is the trade: 498
fewer training rows in exchange for a test fold that can be measured at all. Any comparison of a
gen4 training run against a gen3 one is confounded by the 22% smaller train set as well as by
the different ground, which is one more reason no gen3 number may be mixed with a gen4 one.

### What it costs — per-boundary separation is unchanged, but far more tiles sit on a boundary

This is the trade the amendment is buying and it is stated in full rather than summarised.

- **Per-boundary separation is UNCHANGED from the band rule.** Dropping one lattice position
  leaves a gap of `2 × stride − tile = 192 px = 3.84 km` at 20 m. The band rule's boundaries were
  the same 192 px, because it dropped exactly one position-width of straddlers too. Nothing about
  the *quality* of a single boundary changed. Measured on the gen4 index:
  `min_cross_fold_gap_px = 192`.
- **192 px is a MEASUREMENT, not a floor the rule guarantees — disclosed here because it looks
  like one.** `_windows` clamps the final row and column to the raster edge, so the tile lattice
  is not uniform: on this AOI the last x step is **91 px** (4704 → 4795) and the last y step is
  **220 px** (5152 → 5372). The smallest *non-overlapping* separation the grid admits is
  therefore 315 px in x — a gap of **59 px (1.18 km)** — and 444 px in y, a gap of 188 px. The
  current index measures 192 px only because no fold boundary happens to fall on the clamped
  column; nothing in the algorithm makes that so, and a future rebuild could land there. The gap
  is therefore re-measured every build and checked against bar 5 below, never assumed
  (`test_edge_clamped_positions_can_sit_closer_than_one_stride` pins the geometry;
  `test_real_index_separation_re_derived_from_the_csv` re-derives the achieved gap from the index
  rather than from the sidecar that build wrote).
- **The number of tiles sitting against a boundary changed a great deal.** Measured on the real
  grid: **51.1% of kept tiles (223 / 436) have a kept tile of a different `block_fold` at that
  minimum 192 px separation**, and **37.8% (165 / 436) have a kept tile of a different
  train/val/test *split* that close. Under the gen3 band rule the same measure was 10.0%
  (50 / 500).** A stratified split has far more boundary per unit area than three bands; that is
  arithmetic, not a tuning choice. Coca autocorrelates well beyond 3.8 km, so **more of the
  evaluation set is now near training ground than before.** This is the accepted cost of buying
  measurability, and it is disclosed here rather than discovered later.
- **`"leakage-free" must still never be claimed.** What is asserted is exactly what is measured:
  **zero shared pixels.** Spatial autocorrelation across a fold boundary is mitigated, not
  eliminated — and it is mitigated *less* than under gen3. Defect O3 in `KNOWN_DEFECTS.md`
  stands and is strengthened, not resolved, by this amendment.
- **Why no buffer was added.** A buffer wide enough to matter against >10 km autocorrelation
  would have to drop 2–3 tile positions at every one of the 20 macro-block boundaries. On a
  23 × 25 grid that removes most of the AOI: the single-position drop already costs 139 of 575
  positions (24.2%), and each extra ring costs more than the last because the stratified layout
  has more boundary. The result would be a `test` split too small to carry the n the A16/A17
  claims need — which is the same failure mode as gen3, arrived at from the other direction. The
  honest position is a *narrow* separation that is **disclosed and quantified** in preference to
  a wide one that leaves nothing to measure. The 192 px figure (with the 59 px caveat above), the
  51.1% / 37.8% adjacency fractions and the retention figure must appear wherever the split is
  described.
- **Retention is not neutral with respect to coca.** The kept set is 75.8% of positions but
  carries **73.9%** of the AOI's coca weight: dropping is driven by conflict count, and dense
  ground sits in the interior where more boundaries pass. The weight tie-break only chooses
  between tiles that are *equally* entangled.

### Measured outcome (recorded here, before any model or baseline is run)

Retention **436 / 575 positions kept = 75.8%**, 139 dropped for overlap, 0 unresolved cross-fold
overlaps. Per `block_fold`:

| `block_fold` | 0 | 1 | 2 | 3 | 4 | 5 | max dev from 1/6 |
|---|--:|--:|--:|--:|--:|--:|--:|
| tiles | 68 | 68 | 74 | 82 | 64 | 80 | **2.14 pp** |
| coca share (weight basis) | 0.177 | 0.164 | 0.182 | 0.186 | 0.144 | 0.147 | **2.24 pp** |
| coca share (union basis, for contrast) | 0.183 | 0.167 | 0.182 | 0.177 | 0.141 | 0.151 | 2.61 pp |

**"Coca share" names two different quantities in this project and they must not be swapped.**
The **weight basis** — sum over a fold's kept positions of the *max positive-pixel count across
years* — is what `assign_block_folds` optimises, what the meta sidecar records, and **what the
balance bar below is stated on**. The **union basis** — per-year positive pixels over the union
of a fold's footprint, summed over years — is the honest count of coca a fold contains and is
what the table further down reports. They differ because the weight basis sums per tile and so
carries the 32 px seam double-count by construction. `coca_share_basis` in the sidecar names
which one it holds, and `test_real_index_coca_weight_agrees_with_the_meta_sidecar` asserts the
sidecar and a fresh recount from the masks agree.

Roll-up at rotation 0 (`train` = folds 0–3, `val` = 4, `test` = 5): **292 / 64 / 80** positions
= 67.0 / 14.7 / 18.3 %, i.e. 1752 / 384 / 480 rows over six years. Union positive pixels per
split per year — the headline count is over the **union** of pixels a split covers, because tiles
overlap by 32 px and summing per-tile counts double-counts the seams by a measured **1.22–1.26×**:

| split | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 |
|---|--:|--:|--:|--:|--:|--:|
| train | 5,478,774 | 5,368,729 | 5,635,038 | 5,800,534 | 5,597,597 | 5,740,250 |
| val | 1,187,976 | 1,086,492 | 1,116,667 | 1,108,164 | 1,051,885 | 1,115,583 |
| **test** | **1,163,585** | **1,131,382** | **1,184,234** | **1,237,117** | **1,193,604** | **1,240,760** |

The gen3 row this replaces was `test = 0` in all six years.

### Acceptance bars the split must clear — leak-free AND signal-bearing

Both halves are registered. gen3 asserted only the first, which is exactly why an empty fold
shipped. All five are enforced in code and fail loudly (`AssertionError`, never a printed
`0.000`); each is covered by a test that is proven to fire on a deliberately broken input.

1. **Leak-free.** Exactly **0** shared pixels between any two of the six `block_fold`s, and
   therefore between any two of `train`/`val`/`test`. Not "small", not "negligible" — zero.
   (`verify_split_pixels`; `test_real_index_splits_are_pixel_disjoint`.)
2. **Signal-bearing.** **Every split contains > 0 positive pixels in every one of the six
   years**, and so does every `block_fold` (because the folds are rotated for out-of-fold A17
   inference, each must be independently measurable). This is the invariant whose absence let
   B1 ship. (`test_real_index_every_split_has_positives_in_every_year`,
   `test_real_index_every_block_fold_has_positives_in_every_year`.)
3. **Balanced on both objectives.** Per-fold share of tiles and per-fold share of coca **on the
   weight basis defined above** each within **4.0 pp** of 1/6. Measured **2.1407** (tiles) and
   **2.2431** (coca), i.e. margins of 1.86 and 1.76 pp. (`test_real_index_fold_balance`, which
   asserts the weight basis explicitly so it cannot silently become the union basis — those same
   folds deviate by 2.6131 pp on the union basis.)
4. **Retention.** At least **70%** of the tile positions kept. Measured **436 / 575 = 75.8%**.
   (`test_real_index_retention_above_the_registered_floor`, which counts the denominator from the
   `.npz` files on disk rather than from the sidecar the same run wrote.)
5. **Separation.** Minimum cross-fold gap of at least 192 px, **re-derived from the index CSV**
   rather than read back from the sidecar, with the adjacency fractions above disclosed alongside
   any number computed on this split. Measured 192 px. Note this is a *check*, not an invariant:
   the clamped edge column admits 59 px (see above), so a rebuild that lands a boundary there
   fails this bar and must be rejected rather than explained away.
   (`test_real_index_separation_re_derived_from_the_csv`.)

A split that fails any bar is not used, and no metric computed on it is reported.

**One rebuild path bypassed all five and has been closed.** `build_all` — the re-tiling entry
point the README documented — writes the superseded band split, never calls the verifier, and
leaves `data_generation: gen4` stamped on every subsequent metrics row: following the project's
own documented rebuild command would have silently reinstated the coca-free test fold under a
gen4 label, which is precisely the false-6/6 mechanism this amendment exists to prevent. It now
refuses to overwrite a block-fold index unless `--force-band-split-index` is passed, archives the
superseded pair when it is, marks its own sidecar `verified: false` /
`requires_block_fold_rebuild: true`, and prints the reassignment command. The README rebuild
order names `--rebuild-block-folds` as the terminal step.

### What A20 does NOT change

The metric, the thresholds, the tracks and every decision rule stand exactly as registered:
A16's ≥5/6 persistence rule, A19's max-of-three persistence score and `_metrics_at(thr=0.02,
t_thr=0.0)`, A17's ranking metrics with n=8 and the adjacent-inversion count, A18's 75 m
resolution floor. This amendment changes *which ground* the test set is, and nothing about how a
number computed on it is read.

Two consequences follow and are binding. **(a)** Every Track A figure produced on gen3 is void
and may not be quoted, mixed or compared — in particular a gen3 persistence `0.000` paired
against a gen4 U-Net number would manufacture a false 6/6, and `compare._dedup_last` keys on
`(method, track, fold_year)` while **ignoring `data_generation`**, which is exactly that
mechanism (open defect, tracked, not fixed here). **(b)** `project.data_generation` is bumped to
**`gen4`**, so every row written to `outputs/metrics/baseline_ladder.jsonl` carries the stamp;
any row without it, or with `gen3`, is untrustworthy by default.

---

## Amendment — A21: the metric refuses to score a comparison it cannot make (2026-08-14, BEFORE any metric is recomputed)

A20 fixed the *split* that made `presence_iou` unmeasurable. A21 fixes the *metric*, so that if
a split like that ever recurs the number stops rather than prints. Registered here because it
touches the pre-registered metric path, and written before any number is recomputed on gen4.

**The mechanism, stated exactly.** `src/evaluate.py:_metrics_at` computes
`iou = tp / (tp + fp + fn + 1e-6)`, and likewise for precision and recall. When the target
contains no positive above `t_thr`, all three counts are 0 and every metric returns **`0.000`
with no warning** — arithmetically indistinguishable from a method that genuinely detected
nothing. On gen3 that is precisely what happened, six folds running.

**Where the guard goes, and why not only in the null.** `_metrics_at` is the single shared
implementation. Every Track A arm reaches it:

```
track_a._per_year_write -> common.regression_metrics -> evaluate._evaluate_regression:77 -> _metrics_at
baselines.persistence:102 ------------------------------------------------------------> _metrics_at
evaluate._evaluate_segmentation:96 ---------------------------------------------------> _metrics_at
```

Guarding only `src/baselines/persistence.py` would have covered the **safer** side. The
asymmetric danger recorded in A20 and in `KNOWN_DEFECTS.md` B1 is a self-consistent null paired
against a differently-sourced model number: a persistence arm that crashes is loud and harmless,
whereas a U-Net, RF or NDVI arm silently posting `0.000` against a live null is how a false 6/6
gets built. The guard therefore lives in `_metrics_at` itself and protects all five arms.

**What it asserts.** `_metrics_at` raises `DegenerateComparison` (a subclass of `AssertionError`,
so existing handlers still catch it) when:

1. either input array is **empty** — an empty split is the absence of a measurement, not a score
   of zero; or
2. the target contains **no positive above `t_thr`** — nothing to be right or wrong about, and
   no returned value would be defensible.

**What it deliberately does NOT reject: an all-zero prediction.** This was decided explicitly
rather than by omission. With `tp = 0` and `fn > 0`, `iou = 0 / fn = 0` *exactly* — the epsilon
is immaterial and the zero is earned. Three reasons it is recorded rather than raised:

- "This method detects nothing on this ground" is a real result, and A16's decision rule already
  provides for a model that loses to its null. A project whose stated stance is that "a
  rigorously-established negative result is acceptable" cannot make its own negative results
  unreportable.
- The failure A16 guards against is a *non-measurement* wearing a measurement's clothes. A zero
  against a target that does contain positives is not that.
- Raising would crash a legitimately all-zero persistence variant out of A19's **max-of-three**,
  which would **lower** the U-Net's bar. A19 states that adding variants may only raise the bar;
  a guard that can lower it is worse than the problem.
  Instead `src/baselines/persistence.py` records `pred_all_zero` and `n_target_positive` in each
  metrics row, so a genuine `0.000` is distinguishable from the rejected `0/0` in the sink and
  not only in prose.

**It changes no value on valid input, and that is demonstrated, not asserted.** The arithmetic is
untouched; only two early raises were added. A 16-case battery spanning both production cuts
(`thr=0.02, t_thr=0.0` and `thr=0.5, t_thr=0.5`), exact-boundary inputs, all-correct, all-wrong,
all-zero-prediction and sparse fields at the AOI's ~3.6% positive rate was evaluated before and
after: **all 16 bit-identical** on every one of the four returned metrics.
`tests/test_degeneracy_guard.py` pins four of those cases as exact float constants captured from
the pre-guard implementation, so any future drift in this A19-frozen function fails the suite.

**Method-specific preflight for the persistence null.** Two things `_metrics_at` cannot check for
it are asserted in `src/baselines/persistence.py:preflight`, which validates **all six folds
before the first metrics row is written**: each fold's `test` row set is non-empty, and every
sibling year the rule needs (P1's source year plus all five P2 years) exists at every test
position. `_presence` already raised on a missing sibling, but mid-run, after earlier folds had
been appended to the append-only sink — leaving a partial record nothing downstream could
detect. Hoisting the same check makes the module write six folds or none.

**Consequence to accept in advance.** If a future split fails these conditions, the baselines
**crash instead of producing a table**. That is the intended behaviour and must not be worked
around by catching the exception, lowering `t_thr`, or substituting `nan`/`0.0` — the correct
response is to fix the split under A20 and rerun. Nothing about A16's ≥5/6 rule, A19's
max-of-three, or the metric's cuts is changed by this amendment.
