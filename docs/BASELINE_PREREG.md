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

All three methods report the **same** metrics per fold. Two domains:

**Density fraction (pre-scalar), per held-out-year tile** — matches
`_evaluate_regression` exactly (`evaluate.py:68`):
- `mae`, `rmse`, `bias` = mean(pred − target), on the predicted density fraction
  vs. the mask fraction.
- `presence_iou`, `presence_f1` via `_metrics_at(probs, targets, thr=0.02,
  t_thr=0.0)` (`evaluate.py:41`) — "present" = predicted fraction > 0.02; ground
  truth present = cell has any coca.

**Absolute hectares (uses the scalar), full held-out-year AOI:**
- `aoi_ratio` = predicted ha / official ha for the held-out year, via the same
  `test_year_ratio` full-raster + gate (`TAU=0.05`) + `px_ha` + scalar path
  (`train_loyo.py:162`). Official ha are the frozen v2.1 figures (§0.2 of the plan).

### PRIMARY metric: `aoi_ratio`

Justification: it is the only metric that is (a) computed on the **held-out year**
(not train-years val blocks), (b) available for the U-Net **already on disk**
(v2.1 table), and (c) the project's actual deliverable — an out-of-year absolute-
hectares estimate. The tile metrics (`mae`/`rmse`/`presence_*`) are **secondary**
and, for the U-Net, exist only if the optional 6-fold re-run is run (§7).

`aoi_ratio` quality is summarized per method by two frozen summaries across the 6
folds:
- **Centering** `C = |mean(ratio) − 1.0|` — systematic over/under-count. Lower better.
- **Stability** `S = std(ratio)` across folds, reported with the min–max band.
  Lower better. This is the quantity the whole v2.x line has been trying to shrink.
- Companion count `K` = number of folds with ratio ∈ **[0.85, 1.15]** (the
  originally-desired ±15 % band). Higher better.

## 4. U-Net reference numbers (frozen, from `docs/v2.1_loyo_results.md`)

Per-fold `aoi_ratio`: 2019 = 0.95, 2020 = 0.59, 2021 = 0.90, 2022 = 1.48,
2023 = 1.01, 2024 = 0.79.
→ **mean 0.95, C = 0.05, S = 0.27, band 0.59–1.48, K = 3/6** (2019, 2021, 2023).

The U-Net has **no** held-out-year `mae`/`presence_*` on disk; the v2.1 "val MAE"
(~0.02) is a train-years val-block number and is **excluded** from all
comparisons (it is not the held-out year).

## 5. Decision rule — "is the U-Net justified?" (frozen margins)

With n = 6 folds we report `mean ± std`, min, max, and K. We **do not** compute
p-values or confidence intervals — 6 folds do not support them. Adjudication uses
these pre-committed margins on the PRIMARY metric:

Compare each baseline's (C, S, K) against the U-Net's (0.05, 0.27, 3):

- **Baseline clearly WORSE** (U-Net justified on magnitude): baseline `S` is
  larger than the U-Net's by **> 0.05**, or baseline `K` is lower by **≥ 2** folds.
- **Practical TIE on magnitude:** `|S_baseline − 0.27| ≤ 0.05` **and**
  `|C_baseline − 0.05| ≤ 0.05` **and** `|K_baseline − 3| ≤ 1`. i.e. neither method
  is meaningfully better-centered or more stable out-of-year.
- **Baseline clearly BETTER:** baseline `S` **< 0.22** (lower by > 0.05) **and**
  baseline `C ≤ 0.10` (not achieved by drifting the mean).

Secondary/spatial (only adjudicated if the U-Net re-run supplies its tile metrics,
§7): the U-Net **wins spatially** iff its mean `presence_iou` across folds exceeds
the best baseline's by **> 0.05** absolute. A smaller gap is a spatial tie.

The margin 0.05 is chosen once, here, and is not revisited after seeing results.

## 6. Outcome cases → conclusions applied verbatim

Exactly one case fires; the results doc quotes the matching paragraph and states
which fired.

- **Case 1 — U-Net clearly best on magnitude (and, if re-run, spatially too).**
  Deep learning is earned. Report the margin (ΔS, ΔK, and ΔIoU if available). The
  segmentation framing stands.

- **Case 2 — U-Net wins spatially but ties on magnitude** (requires the re-run to
  establish the spatial win). The model **localizes** better than it **quantifies**;
  scope every claim to coca *location / ranking*, and state plainly that absolute
  out-of-year hectares are no better than a simpler predictor.

- **Case 3 — a random forest ties or beats the U-Net on the PRIMARY metric.**
  **This is the headline, not a failure:** *at ~1 km label resolution, a
  context-free per-pixel random forest matches a spatial segmentation model on
  out-of-year absolute hectares — the labels do not carry enough spatial
  information to justify a U-Net.* Report it as the finding; do not tune the U-Net
  in response. This reframes the project from "I trained a segmentation model" to
  "I measured whether 1 km labels can support one."

- **Case 4 — the NDVI threshold alone is competitive** (NDVI baseline reaches a
  practical tie with the U-Net on the primary metric). The task is largely a
  vegetation-index problem; say so plainly, and treat both the RF and the U-Net as
  unjustified complexity for the magnitude deliverable.

If the RF and NDVI cases both apply, Case 4 dominates (the simplest sufficient
method wins the framing). If neither baseline reaches a tie or better, Case 1
fires (with Case 2's spatial caveat only if the re-run was run).

## 7. What is NOT yet decided (and how it changes adjudication)

The U-Net's held-out-year tile metrics require re-running `run_loyo` with Phase-1
metrics wired in (6 trainings). This decision is **deferred to the user after
Phase 3**. Consequences, pre-committed:
- If the re-run is **not** done: adjudication uses the PRIMARY metric
  (`aoi_ratio`) only. Cases 1/3/4 can still fire; **Case 2 cannot** (it needs the
  spatial win) and will be marked "not testable without the re-run."
- If the re-run **is** done: the secondary spatial rule in §5 is added, enabling
  Case 2. The re-run must reuse the frozen folds/scalars — it may not change the
  U-Net, losses, or tiling, and its numbers may not be tuned toward a nicer table.

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
