# Phase 6 — Recover out-of-year counting (honestly)

## Where we are

Phase 3 produced a **split verdict**:

- **Spatial (Track A):** U-Net wins 6/6 on presence-IoU (0.466 vs RF 0.260 vs NDVI
  0.183). Spatial context genuinely helps. Deep learning justified as a *localizer*.
- **Magnitude (Track B):** U-Net `aoi_ratio` mean 0.95, **std 0.271**, K 3/6. It
  fails the pre-registered ≥5/6 bar against the **N2 historical-mean null** (closer in
  only 2/6, mean δ = −0.141). RF ties. For counting hectares in a censusless year,
  no imagery model beat "predict the average of past official counts."
- **RF feature importances:** the SWIR/burn-ratio complex dominates (NBR 0.11,
  B12 0.10, B11 0.07); **NDVI ranks 15/18**. Coca is detected by *disturbance*
  (bare soil, burn scars), not greenness.

## The Phase 6 hypothesis

**The two failing folds are the two degraded-imagery years, not a model failure.**

`docs/v2.1_loyo_results.md:33-35` already hypothesizes this: 2020 = heavy cloud
cover; 2022 = sparse Sentinel-1 after S1B failed late 2021. And `docs/coverage_check.md`
already contains a working coverage metric (mean clear S2 observations per pixel) that
was used to correctly reject a false 2026 decline.

Dropping those two folds leaves 0.95 / 0.90 / 1.01 / 0.79 → **mean 0.91, std 0.093**,
which is comparable to N2's 0.085. If coverage explains the misses, the honest claim
becomes *"the model can count when imagery is adequate, and detects when it isn't"* —
a conditioned claim that is stronger engineering than an unconditioned one.

**This hypothesis must be tested, not assumed. It may be wrong.**

---

## Prime directives

1. **Any gate must be input-side only.** Coverage thresholds are computed from
   observation counts — never from labels, ratios, or errors. A gate that touches the
   outcome is cherry-picking, full stop.
2. **Pre-register every threshold and every new question before computing it**, as a
   dated amendment in `docs/BASELINE_PREREG.md` with the reason stated.
3. **Never replace a reported result.** The Phase 3 `aoi_ratio` verdict stays published
   exactly as it came out. Phase 6 *adds* analyses; it does not overwrite an
   inconvenient one.
4. **Always report gated AND ungated side by side.** Never publish only the gated table.
5. **State the power limit.** Gating leaves n=4. Four numbers cannot support strong
   claims — say so explicitly wherever they appear.
6. **Do not tune the U-Net to improve a table.** Model changes are allowed only where
   this plan calls for them (6.4), and results are reported whichever way they land.

---

## Phase 5 first — three unresolved items block the write-up

Do these before any Phase 6 work.

**5.1 Reconcile the IoU contradiction.** Track A measured U-Net presence IoU
**0.466**; `README.md` claims **0.665**. Same model, same task. Determine which is
correct (different threshold? different split? stale figure from an older config?) and
fix the wrong one. Shipping two contradictory IoUs in one repo destroys the document's
credibility precisely where it is trying to earn it.

**5.2 Unify the calibration convention across tracks.** NDVI's Track A MAE is
**0.5458** against a true density of ~0.01–0.02 — i.e. it predicts ~50% coverage
everywhere. That is a scaling artifact, not a finding, and it implies **Track A is
computed on uncalibrated predictions while Track B is calibrated.** Pick one
convention, apply it to all three methods in both tracks, document it.

**5.3 Publish the official-hectares series (2019–2024) beside the Track B table.**
N2 wins automatically when the target is stable, so N2's std of 0.085 is
uninterpretable without knowing how much official area actually moved. If official
area barely moved, N2's win is near-definitional and the indictment of the U-Net is
much weaker — say so. If it moved a lot and N2 still won, the result is sharper. Also
note that N2 predicts **zero change by construction** and so is structurally blind to
turning points; its worst year (2024, 0.85) is consistent with lagging a real decline.

**5.4 Two precision fixes.** The RF "4/6" includes an **exact tie** (2020: both
ratios 0.59, δ = 0.000) → report as **3 wins / 1 tie / 2 losses**. And NBR/B12/B11 are
**one physical signal, not three** (NBR is derived from NIR and SWIR); note also that
RF importances split across correlated features, so the ranking is indicative.

**Acceptance:** one IoU in the repo; one calibration convention; official-ha series
published; counts and feature framing corrected.

---

## Phase 6.1 — Test the coverage hypothesis (~4–6h)

Extend the `docs/coverage_check.md` methodology to **all six LOYO years**. Produce, per
year: number of scenes, mean and median clear S2 observations per pixel, and an
equivalent S1 pass count (which should show the post-S1B drop from late 2021).

Then plot the six years as **coverage vs |ratio − 1|** and report the relationship
descriptively (n=6 — no p-values, no regression line pretending to significance).

Three possible outcomes, all reportable:
- **2020 and 2022 are clearly the lowest-coverage years** → hypothesis supported,
  proceed to 6.2.
- **Coverage is unrelated to error** → hypothesis rejected. This is a real finding:
  the misses are a model/generalization problem, not a data problem. Report it, skip
  6.2, and go to 6.3.
- **Partially supported** (e.g. 2020 yes, 2022 no) → report the split and gate only on
  the supported mechanism.

**Acceptance:** a committed `docs/coverage_by_year.md` with the per-year table, the
plot, and an explicit statement of which outcome fired.

---

## Phase 6.2 — Coverage-gated counting (~3–4h, only if 6.1 supports it)

**Pre-register the threshold first**, in units of clear observations per pixel, chosen
from the coverage distribution alone — before recomputing any ratio.

Implement an abstention path: when a year falls below threshold, the system returns
`insufficient_coverage` and **no hectare estimate**, rather than a silent bad number.
Wire this into `src/evaluate.py` / the ratio path and surface it in the UI the same way
`ui/app.js:187` already refuses to print a bare number for a censusless year.

Report a single table with **both** columns: all-6-fold and gated, for every method
including the nulls. Label n for each.

**Acceptance:** gate threshold committed before results; both tables published; the
n=4 power limitation stated in the same paragraph as the gated numbers.

---

## Phase 6.3 — Hybrid: anchor the total, allocate it spatially (~4h)

Use each method where it demonstrably wins:

- **Magnitude anchor:** N2 (mean of prior official counts) or a simple trend fit sets
  the held-out year's total.
- **Spatial allocation:** the U-Net distributes that total across pixels/municipalities.

This is a natural extension of the existing frozen scalar (`fit_scalar`,
`src/train_loyo.py:142`), except the anchor becomes year-aware instead of pinned to
2023. Note the side benefit: it **removes the 2023 calibration circularity**, because
the anchor no longer derives from the year being predicted.

Evaluate the hybrid on both tracks. Expect: magnitude ≈ N2 (by construction) plus the
spatial distribution N2 cannot produce. Report it as a *product design* conclusion, not
a modelling victory — be explicit that the total's accuracy comes from the anchor.

**Acceptance:** hybrid scored on both tracks; the write-up states plainly which
component earns which claim.

---

## Phase 6.4 — Change detection: the question N2 cannot answer (~4–5h)

**This is a NEW question added after seeing Phase 3 results. It must be logged as a
dated prereg amendment, with the reason, before any number is computed. The original
`aoi_ratio` verdict remains published unchanged.** Justification to state in the
amendment: a monitoring system exists to detect change, and N2 predicts zero change by
construction, so `aoi_ratio` on a stable series cannot discriminate the capability the
product is for.

For each of the 5 year-over-year transitions, evaluate:
- **Direction accuracy** — does the model get the sign of the change right? N2 scores
  zero by definition.
- **Magnitude of change** — MAE on Δ hectares, model vs N2.
- Pre-register the bar (e.g. ≥4/5 correct directions) *before* computing.

Optional, cheap, and honest: also report a **persistence-with-trend** null (extrapolate
the linear trend of prior official counts) so the comparison isn't against a
deliberately weak straw man.

**Acceptance:** amendment committed before results; direction table for all 5
transitions; the trend null included.

---

## Phase 6.5 — Robustness training (~10–12h, optional but highest ML value)

The version that removes the need for a gate. Two changes:

1. **Add a coverage channel** — per-pixel valid-observation count as a 19th input
   channel. The model currently cannot distinguish *"no coca here"* from *"I could not
   see this pixel."* Give it that information.
2. **Simulate observation dropout during training** — randomly degrade tiles to mimic
   cloudy years and missing SAR passes, so the model learns to compensate rather than
   under-predicting on degraded inputs.

Requires a LOYO re-run to evaluate (6 trainings). Report before/after for all six
folds. **If it does not help, publish that too** — a failed, well-measured robustness
intervention is still a result.

**Acceptance:** before/after table across all 6 folds; honest reporting either way.

---

## Phase 6.6 — Write-up (~2h)

Update `docs/BASELINE_LADDER_RESULTS.md` (or a companion `PHASE6_RESULTS.md`) with the
conditioned claim, if the evidence supports it:

> This system estimates coca area for a year with no census **provided satellite
> coverage is adequate**, and detects and declines when it is not. Under adequate
> coverage it matches a historical-average baseline on magnitude while additionally
> providing the spatial distribution, which that baseline cannot produce. It localizes
> coca substantially better than any simpler method (IoU 0.47 vs 0.26 context-free).

Every clause must trace to a committed number. If 6.1 rejected the hypothesis, write
the unconditioned version instead — *"the model localizes but does not count"* — and
say what would be needed to fix it.

---

## Guardrails

- **Never publish** `outputs/catatumbo_2023_coca.geojson` (2,065 plot-level 20m field
  polygons, targeting-usable). Gitignored — keep it that way; never `git add -f`.
  Public artifacts stay at 75m COG + municipal aggregate.
- **Do not build the U-TAE temporal model** (`src/models/temporal.py:13`).
- Do not tune toward a nicer table. Report what fires.
- Keep `data/` and `.venv` out of git.

## Order of work and reporting points

**5.1–5.4 → 6.1 → [6.2 if supported] → 6.3 → 6.4 → 6.6**, with 6.5 optional after.

Stop and report after **6.1** (the hypothesis test — this determines whether the rest
of the plan is even valid) and again after **6.4**.
