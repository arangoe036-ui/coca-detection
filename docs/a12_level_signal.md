# Phase 6.6 (A12) — Does a per-year input statistic carry the cross-year level signal?

> ## ⚠ CORRECTION 2026-08-10 (same day, found in code review) — verdict stands, two rows do not
>
> The composites carry a **year-varying Sentinel-2 data gap** that this analysis did not
> control for. `config/default.yaml`'s `scl_mask_classes: [3,8,9,10,11]` omits SCL **0
> (NO_DATA)** and **1 (SATURATED)**, so nodata-dominated pixels take a reflectance median of
> *exactly* 0.0. Fraction of the AOI affected: **2019 25.3%, 2020 25.3%, 2021 12.2%,
> 2022–2024 0.7%**. Confirmed to be real S2 loss rather than mosaic edge: inside 2019's zero
> region Sentinel-1 VV is 97.2% valid at −7.14 dB, whereas inside 2022's 0.7% region VV is 0%
> valid (that one *is* edge fill).
>
> Consequence: the `a[a != 0.0]` filter in `scripts/a12_level_signal.py` measured a
> **different spatial footprint every year** (74.7% / 74.8% / 87.8% / 99.3% / 99.3% / 99.3%),
> and footprint size is itself correlated with the target. Recomputed on the fixed
> all-years-valid footprint (70.6% of AOI):
>
> | candidate | as published | fixed footprint | |
> |---|--:|--:|---|
> | B11.mean | +0.355 | **−0.621** | sign flip |
> | B12.mean | +0.050 | **−0.709** | sign flip |
> | NDVI.mean | +0.632 | +0.634 | robust |
> | NBR.mean | +0.486 | +0.544 | robust |
>
> **What changes:** the B11/B12 rows in the table below, and the sentence claiming the SWIR
> means "carry the right sign but are weak". On a fixed footprint both SWIR means carry the
> **wrong** sign. **What does not change:** no candidate is both strong and correctly signed,
> so the pre-registered branch still fires — **DO NOT RETRAIN** — and the signal-to-noise
> argument is untouched, since it never depended on these correlations.
>
> **Also corrected:** the leave-one-out column below is reported but **not enforced** by the
> decision rule, and NDVI/NBR were excluded by *sign and magnitude alone*. Describing LOO as
> part of the decision rule, and crediting it with preventing a false write-up, overstates
> its role. It is a diagnostic here, not a gate.
>
> **Latent trap for any re-run:** with the offset fix now subtracting 1000 DN, a
> nodata-dominated pixel in 2022+ lands at **−0.1**, not 0.0, and slips past an `!= 0` filter
> entirely. Verified 0.0000% negatives today, so nothing is currently affected — but the
> `!= 0` idiom must be replaced with a real validity mask, not trusted.

**Date:** 2026-08-10. Run on the **corrected** data (Phase 6.6 `BOA_ADD_OFFSET` fix; the
model-free 6.6b acceptance check passed the same day — visible-band step at 2021→2022 is
**+0.0013** against the bug's +0.1000). Interpretation is the pre-registration
([`BASELINE_PREREG.md`](BASELINE_PREREG.md) amendment A12), applied verbatim.

Reproduce: `python scripts/a12_level_signal.py` (acceptance check:
`python scripts/acceptance_6_6b.py`).

## Decision rule, fixed before computing

- **Candidates:** per-year mean and std of NBR, B12, B11, NDVI over the annual mosaic.
- **Target:** official Catatumbo coca hectares 2019–2024 (§5.3 of the ladder results).
- **Statistic:** Pearson *r*, n=6, **descriptive only — no p-values** (prereg A12).
- **Directional prior**, from the RF finding that coca is detected by disturbance rather
  than greenness (SWIR/NBR complex dominant, NDVI 15/18): **B11 +, B12 +, NBR −, NDVI −**.
  A large correlation with the wrong sign is *not* evidence.
- **Bar:** |r| ≥ 0.70 **and** the sign matches the prior. The prereg said "clear and
  directionally-sensible" without a number, so the number was fixed here in advance.
- **Robustness:** leave-one-year-out *r*, because 2024 is a +11.1% outlier in the official
  series and at n=6 one point can manufacture a correlation.

## Results

| candidate | r | prior | sign ok | \|r\| ≥ .70 | LOO r range |
|---|--:|:--:|:--:|:--:|:--|
| B11.mean | +0.355 | + | yes | no | [+0.14, +0.61] |
| B12.mean | +0.050 | + | yes | no | [−0.37, +0.36] |
| NDVI.mean | +0.631 | − | **no** | no | [+0.44, +0.97] |
| NBR.mean | +0.483 | − | **no** | no | [+0.22, +0.97] |
| B11.std | −0.632 | none | — | no | [−0.78, −0.53] |
| B12.std | −0.627 | none | — | no | [−0.93, −0.46] |
| NBR.std | −0.437 | none | — | no | [−0.86, −0.20] |
| NDVI.std | +0.068 | none | — | no | [−0.04, +0.15] |

No candidate clears the bar. The two physically-motivated candidates (the SWIR disturbance
bands) carry the right sign but are weak — B12 is essentially zero. NDVI and NBR are the
strongest means but point the **wrong way**: in years with more coca the AOI reads *greener*
and *less burned*.

## Why the wrong sign is expected, and why more years would not help

Coca is ~3.6% of this AOI (≈40,000 ha of ≈1,099,000 ha), and the whole 2020→2024 swing is
+8,963 ha = **0.8% of the area**. Against the observed interannual spread of the statistics:

| statistic | observed range across years | max plausible coca contribution | ratio |
|---|--:|--:|--:|
| NDVI.mean | 0.0486 | ~0.0025 (0.8% area × ΔNDVI 0.3) | ~5% |
| B11.mean | 0.0130 | ~0.0004 (0.8% area × ΔB11 0.05) | ~3% |

The coca term sits **20–30× below** the year-to-year variation, which is regional weather and
phenology. The positive NDVI correlation is therefore best read as *wetter years are greener
and also happened to carry more coca* — a confound, not a signal. **This is a
signal-to-noise limit, not a small-sample limit: adding years would not recover it.**

The LOO ranges justify having pre-registered that check: NDVI.mean and NBR.mean both reach
r = +0.97 when a single year is dropped. Without the sign prior and the LOO column fixed in
advance, either could have been written up as a level signal.

## Verdict (A12): NO LEVEL SIGNAL → **DO NOT RETRAIN**

Per the pre-registered branch: aux inputs would restore a signal that does not exist, so the
one retrain is not spent, and **counting routes to the Phase 6.3 hybrid anchor** (historical
mean sets the total, the U-Net distributes it spatially). That also removes the 2023
calibration circularity.

Three consequences, stated explicitly:

1. **The aux-input design is closed.** 2–4 anomaly-coded scalars, the memorization shuffle
   test, the `in_channels` 18→20/21 change — none of it gets built.
2. **The "zero-dimensional-cost" mixed-normalization fallback is closed too**, which A12 did
   not spell out. Pooled train-year stats for the disturbance channels rest on the *same*
   premise as the aux inputs — that the year's absolute level says something about total
   hectares. It does not, so the delivery mechanism is irrelevant.
3. **The per-year-normalization mechanism is refuted, not merely unproven.** Phase 6.5's
   leading suspect was that per-year z-scoring discards a useful cross-year level. A12 shows
   there was no useful level to discard. This is a stronger outcome than an inconclusive test.

## Still open

**The 2020 residual.** 2020 is a clean pre-offset year and still missed by −0.41; neither the
offset fix nor A12 explains it. One lead worth a cheap look: on the corrected mosaics 2020 has
the **lowest NDVI (0.7929) and lowest NBR (0.5922) of the three clean years** together with the
*highest* visible reflectance — the signature of more bare soil or residual haze — despite
Phase 6.1 measuring 2020 as having the **best** optical coverage of all six years (32.7 clear
observations/pixel). Coverage was already rejected as the cause; scene *quality* at equal
coverage has not been tested.
