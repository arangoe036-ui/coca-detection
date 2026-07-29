# Phase 6.1 — Per-year coverage vs out-of-year error (Catatumbo, 2019–2024)

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
