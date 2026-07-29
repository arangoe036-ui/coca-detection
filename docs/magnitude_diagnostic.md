# Phase 6.5 — Magnitude-mechanism diagnostic

Pre-registration: `BASELINE_PREREG.md` A8. n=6 throughout, descriptive (no p-values). The frozen Track B `aoi_ratio` verdict is unchanged.

## Outcome (prereg A8) — which mechanism fired

**Root cause identified: an uncorrected Sentinel-2 processing-baseline 04.00 offset (~+0.1 reflectance, visible-band jump +0.102 at 2021→2022), masked by per-year normalization.** Both pre-registered mechanisms are implicated, and they compound:

- **Normalization — IMPLICATED (dominant).** Swapping own-year → pooled stats moves the total by a mean 132% (6.5b), and the predicted total barely tracks the truth (corr(pred,official)=+0.22, pred CV 0.28 vs official 0.07). Per-year z-scoring is load-bearing precisely because it is silently correcting the ~0.1 S2 offset — which is why naive pooled stats collapse the post-2022 years to ~0 (6.5b). The fix is not 'use pooled stats'; it is **correct the offset first**, then revisit normalization.
- **Gate — IMPLICATED (secondary amplifier).** Cross-year CV rises 0.21 → 0.33 from ungated to the operational TAU=0.05 (6.5d) — ~1.6×. But the base instability (CV 0.21 even ungated) predates the gate.
- **6.5c**: the 2020 & 2022 discrepancies are **diffuse** (top-5 of 110 blocks hold only 23% / 31%), consistent with a global level shift, not a local confusion.

**What remains unexplained / next step.** Even ungated with per-year stats the cross-year CV (0.21) still exceeds the target's own CV (0.075), so normalization is not the whole story — but the S2 offset is a concrete, cheap data bug that must be fixed and re-measured **before** any normalization rung, supervision change (Phase 7), or sensor build. **Reordered priority: (1) correct the S2 baseline offset in `stac_export.py`, regenerate 2022–2024 tiles preserving the block→split assignment (hard rule), re-verify channels; (2) re-run LOYO and see how much of the 0.271 spread it removes; (3) then decide on normalization/gate/supervision.** Phase 7's *spatial* justification is unaffected (cell-level, A4); its *magnitude* rationale is now downstream of the offset fix.

## 6.5a — Per-year normalization parameters vs out-of-year error

Per-year z-scoring mean/std for the disturbance channels the RF found dominant, and NDVI. Correlations across the 6 years (descriptive, n=6).

| channel | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | r(mean, signed err) | r(mean, \|err\|) | r(std, signed err) |
|---|---|---|---|---|---|---|---|---|---|
| B11 (mean) | 0.138 | 0.142 | 0.159 | 0.284 | 0.292 | 0.291 | +0.50 | +0.09 | -0.58 |
| B12 (mean) | 0.068 | 0.070 | 0.077 | 0.188 | 0.192 | 0.193 | +0.50 | +0.11 | -0.63 |
| NDVI (mean) | 0.558 | 0.556 | 0.676 | 0.499 | 0.508 | 0.492 | -0.28 | -0.28 | -0.53 |
| NBR (mean) | 0.400 | 0.401 | 0.486 | 0.373 | 0.378 | 0.364 | -0.22 | -0.29 | -0.55 |
| VV (mean) | -7.164 | -6.989 | -6.813 | -6.937 | -7.031 | -7.057 | +0.16 | +0.27 | +0.05 |
| VH (mean) | -13.253 | -13.053 | -12.914 | -12.854 | -12.925 | -12.946 | +0.41 | +0.32 | -0.10 |

- **corr(predicted_ha, official_ha) = +0.22** — the model's total does NOT track the true level.
- predicted-ha CV = 0.284 vs official-ha CV = 0.075: the model's total is MORE variable than the truth (it adds spread rather than tracking the modest real signal).
- signed errors: 2019=-0.05, 2020=-0.41, 2021=-0.10, 2022=+0.48, 2023=+0.01, 2024=-0.21.

### 6.5a(ii) — Absolute reflectance level by year (all S2 bands)

| band | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | Δ 2021→2022 |
|---|---|---|---|---|---|---|---|
| B02 | 0.029 | 0.031 | 0.032 | 0.132 | 0.133 | 0.136 | +0.100 |
| B03 | 0.045 | 0.047 | 0.053 | 0.157 | 0.159 | 0.159 | +0.104 |
| B04 | 0.031 | 0.033 | 0.035 | 0.136 | 0.137 | 0.139 | +0.101 |
| B08 | 0.224 | 0.228 | 0.266 | 0.414 | 0.428 | 0.416 | +0.148 |
| B05 | 0.073 | 0.075 | 0.086 | 0.197 | 0.201 | 0.199 | +0.111 |
| B06 | 0.185 | 0.187 | 0.223 | 0.359 | 0.373 | 0.361 | +0.137 |
| B07 | 0.225 | 0.228 | 0.269 | 0.416 | 0.433 | 0.419 | +0.147 |
| B8A | 0.246 | 0.250 | 0.293 | 0.444 | 0.462 | 0.448 | +0.151 |
| B11 | 0.138 | 0.142 | 0.159 | 0.284 | 0.292 | 0.291 | +0.125 |
| B12 | 0.068 | 0.070 | 0.077 | 0.188 | 0.192 | 0.193 | +0.110 |

- **Every reflectance band steps up by ~+0.10 at 2021→2022** (visible B02/B03/B04 mean jump = +0.102); SAR VV is flat (-0.124 dB). This is the **Sentinel-2 processing-baseline 04.00 radiometric offset** (2022-01-25: `reflectance = (DN − 1000)/10000`). `stac_export.py:89` divides by 10000 **without** subtracting 1000, so **2022–2024 S2 inputs are inflated by ~0.1 reflectance** relative to 2019–2021. Per-year z-scoring re-centres each year and thereby **masks** this discontinuity — the model never sees comparable absolute levels across the boundary.
- This is a concrete, cheap **data bug**, not an inherent modelling limit.

## 6.5b — Swap-stats sensitivity (own-year vs pooled stats, `final_multiyear.pt`)

Per-year summed gated U-Net density (ha, tile-based) under each year's own normalization vs a single pooled (all-year) normalization. This measures how much the *normalization choice alone* moves the total — a sensitivity probe (uses the final all-years model, not per-fold LOYO models), not a fix.

| year | own-stats total | pooled-stats total | Δ% |
|---|--:|--:|--:|
| 2019 | 16,429 | 58,190 | +254% |
| 2020 | 18,708 | 52,498 | +181% |
| 2021 | 36,497 | 56,861 | +56% |
| 2022 | 44,295 | 65 | -100% |
| 2023 | 37,713 | 78 | -100% |
| 2024 | 40,624 | 72 | -100% |

- Mean |Δ| from swapping normalization = **132%** (range 56–254%). Large ⇒ the total is highly sensitive to the per-year normalization, i.e. normalization is a dominant lever on magnitude.

## 6.5c — Spatial decomposition of the 2020 & 2022 discrepancy (by block)

Per block, |predicted − mask| ha over that block's tiles. If a few blocks dominate ⇒ a specific confusion; if spread evenly ⇒ a global level shift (consistent with normalization).

- **2020**: 110 blocks; top-5 blocks hold **23%** of the total absolute discrepancy (diffuse).
- **2022**: 110 blocks; top-5 blocks hold **31%** of the total absolute discrepancy (diffuse).

## 6.5d — Presence-gate (`TAU`) sensitivity of the total

Per-year summed gated density (ha) as `TAU` varies, and the cross-year coefficient of variation of the total at each gate. If the CV balloons with the gate, the gate amplifies small density shifts into large total swings.

| TAU | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | cross-year CV |
|---|---|---|---|---|---|---|---|
| 0.00 | 30,989 | 32,391 | 47,341 | 54,569 | 48,487 | 51,688 | 0.208 |
| 0.02 | 22,169 | 23,771 | 41,386 | 47,687 | 41,409 | 44,591 | 0.273 |
| 0.05 | 16,429 | 18,708 | 36,497 | 44,295 | 37,713 | 40,624 | 0.333 |
| 0.10 | 6,030 | 9,439 | 17,625 | 34,179 | 27,343 | 29,577 | 0.505 |
| 0.15 | 2,549 | 4,778 | 8,128 | 18,310 | 12,954 | 14,935 | 0.545 |

- Cross-year CV at TAU=0.00 is 0.208; at the operational TAU=0.05 it is 0.333. The gate amplifies cross-year spread.
