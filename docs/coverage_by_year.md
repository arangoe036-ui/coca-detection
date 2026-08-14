# Phase 6.1 — Per-year coverage vs out-of-year error (Catatumbo, 2019–2024)

> ## ⚠ THE REJECTION BELOW USED A COVERAGE METRIC BLIND TO THE ACTUAL DEFECT — 2026-08-10
>
> This document rejected the coverage hypothesis using **mean and median clear observations
> per pixel**. Neither statistic can detect a region with *zero* observations. Measured
> directly on the composites, the fraction of the AOI where Sentinel-2 has **no valid data at
> all** (reflectance median exactly 0, because `scl_mask_classes` omits SCL 0/1) is:
>
> | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 |
> |--:|--:|--:|--:|--:|--:|
> | **25.3%** | **25.3%** | **12.2%** | 0.7% | 0.7% | 0.7% |
>
> Confirmed real S2 loss, not mosaic edge: inside 2019's blank region Sentinel-1 VV is 97.2%
> valid at −7.14 dB. A year with a quarter of its AOI unobserved can still post the *highest*
> mean clear/px (2020: 32.7) because the observed remainder is densely covered — which is
> exactly what happened. **Fact #1 below ("2020 has the single best coverage of all six
> years") is an artifact of averaging over a hole.**
>
> **Argument #3 inverts.** It reasoned that the error direction was backwards for a coverage
> mechanism. Re-ranked on blank-fraction: 2019 (25.3%) → 0.95, 2020 (25.3%) → **0.59**,
> 2021 (12.2%) → 0.90 all **under-predict**, mean 0.81; the three clean-footprint years give
> 1.48 / 1.01 / 0.79, mean 1.09. Under-prediction where the imagery is blank is precisely the
> predicted direction, and the most-affected year is the worst fold.
>
> **A confound this exposes:** the blank-coverage years (2019–2021) are exactly the
> *non-offset* years, and the clean-footprint years (2022–2024) are exactly the
> *offset-inflated* years. The two defects push predictions in **opposite** directions on
> **opposite** year groups. So the v2.1 LOYO headline — "unbiased but wide, mean 0.95, std
> 0.271", presented as the model's accepted ceiling — is substantially two data bugs
> cancelling, not measured model variance.
>
> **Consequences that must be revisited, not assumed:**
> - "Skip Phase 6.2 (coverage gate)" rested on 2020 passing any input-side threshold. On
>   blank-fraction it would not.
> - "Phase 8 sensor additions are aimed at the wrong problem" and the Phase 8.3 **HLS
>   deletion** both rested on optical coverage being ample. HLS (Landsat-harmonised) fills
>   exactly these gaps. That rationale is **live again**.
> - "The indicated lever is Phase 7 (supervision)" followed from "coverage is not the
>   bottleneck", which is no longer established.
> - **Most important: the headline counting failure is now provisional.** Whether the model
>   beats the N2 historical-mean null must be re-measured after both defects are fixed. It may
>   still lose — but the existing measurement cannot settle it.
>
> Nothing here is overwritten; the original analysis stands below as it was computed.

Optical (Sentinel-2 clear observations) and SAR (Sentinel-1 passes) measured
**separately** per full calendar year (prereg A7). `|ratio−1|` is the U-Net's
frozen LOYO out-of-year error (`docs/v2.1_loyo_results.md`). n=6 — descriptive
only, no p-values, no fitted line.

| year | S2 scenes | S2 clear/px (mean) | S2 median | S1 passes | S1 obs/px (mean) | U-Net ratio | \|ratio−1\| |
|---|--:|--:|--:|--:|--:|--:|--:|
| 2019 | 447 | 32.4 | 28 | 154 | 37.3 | 0.95 | 0.05 |
| 2020 | 444 | 32.7 | 28 | 164 | 38.4 | 0.59 | 0.41 |
| 2021 | 457 | 26.7 | 23 | 139 | 37.3 | 0.90 | 0.10 |
| 2022 | 592 | 25.5 | 23 | 128 | 33.2 | 1.48 | 0.48 |
| 2023 | 743 | 27.1 | 25 | 141 | 36.5 | 1.01 | 0.01 |
| 2024 | 438 | 27.7 | 25 | 136 | 37.2 | 0.79 | 0.21 |

## Rankings (for the descriptive read)
- Lowest S2 clear coverage → highest: 2022 < 2021 < 2023 < 2024 < 2019 < 2020
- Lowest S1 passes → highest: 2022 < 2023 < 2024 < 2019 < 2021 < 2020
- Largest \|ratio−1\| → smallest: 2022 > 2020 > 2024 > 2021 > 2019 > 2023

The two worst U-Net years are **2022** (|err|=0.48) and **2020** (|err|=0.41).

## Outcome (prereg A7): hypothesis **REJECTED** — coverage does not explain the errors

The pre-registered hypothesis was *"the two out-of-band folds are the
degraded-imagery years."* The data contradicts it. Three decisive facts:

1. **2020 — the 2nd-worst year (|err| 0.41) — has the single best coverage of all
   six years**, on *both* sensors (S2 32.7 clear/px, the highest; S1 164 passes and
   38.4 obs/px, also the highest). The long-standing "2020 was cloudy" speculation
   (`v2.1_loyo_results.md:33`) is empirically **false** for this AOI.
2. **2023 — the best year (|err| 0.01) — ran on essentially the same post-S1B SAR
   coverage as 2022, the worst year** (141 vs 128 passes; 36.5 vs 33.2 obs/px). Near-
   identical SAR input, opposite outcomes → SAR passes do not drive the error, exactly
   as the pure-S1 story was warned against.
3. **The error direction is backwards for a coverage mechanism.** Fewer looks would
   plausibly cause *under*-prediction. Instead the **lowest**-coverage year (2022)
   **over**-predicts (ratio 1.48) while the **best**-covered year (2020)
   **under**-predicts (0.59). The sign is inconsistent with "less data → less coca."

Rankings confirm no monotone relationship: worst-error 2022 is the lowest-coverage
year (the one point consistent with the hypothesis), but 2nd-worst 2020 is the
*highest*-coverage year, and best-error 2023 sits mid-pack on both axes. At most this
is **partial and confounded** for 2022 alone — and 2022 is not separable from a model
effect, since 2023 succeeded on the same SAR coverage. The honest read is
**Outcome (ii): coverage is unrelated to out-of-year error → the misses are a
model / generalization problem, not a data problem.**

Secondary empirical correction: the S1B outage (failed Dec 2021) did **not** halve
S1 passes over Catatumbo — 2022–2024 (S1A only) had 128–141 passes vs 139–164 in
2019–2021, a ~15% reduction, not ~50%. The "2022/2024 crippled by the SAR gap"
premise is not supported by the pass counts.

### Consequences for the plan

- **Skip Phase 6.2 (coverage gate):** a gate cannot cleanly separate good from bad
  years when the 2nd-worst year (2020) has the best coverage and would pass any
  input-side threshold. Gating here would discard nothing useful and explain nothing.
- **Phase 8 sensor additions (HLS, PALSAR) are aimed at the wrong problem.** They add
  *coverage*; coverage is not the bottleneck. Adding looks will not fix a model whose
  worst-tied year is already its best-covered. Their expected value should be
  reconsidered before spending the compute/disk.
- **The indicated lever is Phase 7 (supervision).** The out-of-year magnitude failure
  is a model/generalization problem; the uniform-within-1 km-cell labels
  (`labels.py:107-115`) remain the leading structural suspect. This is consistent with
  Track B's finding that the model cannot count out-of-year better than a historical
  mean — a data-*coverage* fix was never going to address a label/supervision ceiling.

n=6 throughout; this is a descriptive read, not a significance test.
