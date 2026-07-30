# Phase 6.6 (A11) — Quantile gate on the corrected 2022 fold (post-hoc)

Scale-invariant gate: keep the top `f_keep` fraction of pixels per year (`f_keep`=0.201, fit on train years as the mean keep-rate at 0.05); scalar refit on quantile-gated train predictions. Official 2022 = 37,965 ha.

| gate | 2022 aoi_ratio | scalar |
|---|--:|--:|
| ungated (TAU=0) | 0.202 | 1.30 |
| fixed TAU=0.05 (the collapse) | 0.071 | 1.56 |
| **quantile (top 20.1%)** | **0.172** | 1.63 |

v2.1 (contaminated, fixed TAU=0.05) 2022 ratio = 1.48 for reference.

**Verdict (A11):** gate NOT the fix (<0.40): raw output collapsed -> retrain with normalization.
