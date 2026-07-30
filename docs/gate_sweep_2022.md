# Phase 6.6 — Presence-gate (TAU) sweep on the 2022 fold (corrected data)

Prereg: A10. Re-trained 2022 LOYO fold on corrected data; scalar refit at each TAU (train years only). Official 2022 = 37,965 ha. v2.1 (contaminated, TAU=0.05) = 1.48; corrected @ TAU=0.05 = 0.08 (the collapse).

| TAU | scalar | predicted ha | aoi_ratio | pixels kept |
|---|--:|--:|--:|--:|
| 0.000 | 1.30 | 7,678 | 0.202 | 100.0% |
| 0.001 | 1.30 | 7,635 | 0.201 | 95.0% |
| 0.002 | 1.32 | 6,971 | 0.184 | 62.7% |
| 0.005 | 1.36 | 5,052 | 0.133 | 14.1% |
| 0.010 | 1.38 | 4,457 | 0.117 | 7.7% |
| 0.020 | 1.42 | 3,865 | 0.102 | 4.6% |
| 0.030 | 1.45 | 3,467 | 0.091 | 3.4% |
| 0.050 | 1.56 | 2,704 | 0.071 | 1.9% |
| 0.080 | 1.81 | 1,730 | 0.046 | 0.8% |
| 0.100 | 2.06 | 964 | 0.025 | 0.4% |

- aoi_ratio at **TAU=0** = **0.202**, at operational **TAU=0.05** = **0.071**.
  → **Gate EXONERATED as the primary cause** (per A10: no recovery toward ~1 at
  TAU→0 — 2022 is still a ~5× under-prediction with the gate fully off).

Reading (A10), with the nuance the numbers demand — **two distinct defects**:

1. **Primary = normalization / distribution shift (a correctness issue → moves into
   6.6).** Even ungated, the model trained on corrected data outputs almost no coca
   density for held-out corrected 2022 (raw summed ≈ 5,900 ha vs official 37,965).
   The clean held-out folds calibrate fine (~0.94), so the model handles clean years
   but not the corrected offset year — the correction shifted 2022's per-year-
   normalized input distribution into a regime the model maps to ~zero. This is the
   co-adaptation the offset fix exposed, and it is the dominant effect.
2. **Secondary = the gate is genuinely mis-designed** (as hypothesized), just not the
   root. A *fixed absolute* `TAU=0.05` on a per-year-normalized output whose
   distribution moves year to year cuts a different mass fraction each year — here it
   removes **98% of corrected-2022 mass** and drops the ratio a further ~3× (0.20→0.07).
   The right form is a **train-year quantile** gate (fit on train years only,
   input/train-side). It should still be fixed in 6.6 (it materially aggravates the
   collapse), but on its own it recovers only 0.07→0.20, not to ~1.

Net: the gate fix is necessary but not sufficient; the normalization correctness
issue is the primary lever for out-of-year counting and is now a 6.6 item, not a
Phase 9 ablation. n=1 offset fold (2022); 2023 independently worsened (1.01→1.33),
consistent with distribution-shift instability rather than a uniform gate effect.
