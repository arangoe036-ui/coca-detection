"""A12 - does a per-year INPUT statistic carry the cross-year level signal?

Pre-registered gate on the retrain (BASELINE_PREREG.md, amendment A12). Run on the
CORRECTED data (Phase 6.6 offset fix, acceptance PASSED 2026-08-10).

Decision rule, fixed BEFORE computing:
  * computed: per-year mean AND std of NBR, B12, B11, NDVI over the annual mosaic
  * target: official Catatumbo coca hectares 2019-2024
  * Pearson r, n=6, DESCRIPTIVE ONLY - no p-values (prereg A12)
  * directional prior (coca = disturbance, not greenness; RF found SWIR/NBR dominant
    with NDVI 15/18):  B11 +, B12 +, NBR -, NDVI -
  * bar: |r| >= 0.70 AND the sign matches the prior. Wrong sign is not evidence.
  * DECISIVE CANDIDATES ARE THE MEANS ONLY. The bar requires a prior sign, and a std has
    no defensible one (spread can widen or narrow as coca grows), so a std can never pass
    by construction. std rows are computed and reported; any std reaching |r| >= 0.70 is
    surfaced explicitly in the verdict block as a reported-but-not-decisive finding, so
    that a strong un-prior-ed correlation cannot go unseen.
  * REPORTED DIAGNOSTIC, NOT A GATE: leave-one-year-out r, printed for every candidate
    because 2024 is a +11.1% outlier and with n=6 one point can manufacture a
    correlation. It is never enforced by the code and never excluded a candidate
    (correction recorded in docs/a12_level_signal.md).
  * if a mean candidate passes -> proceed to the retrain using THAT feature
  * if nothing passes         -> DO NOT RETRAIN; route counting to the Phase 6.3 hybrid
                                 anchor and report the null.

Caveat recorded up front: train_loyo.py computes its year_norm_stats over TILES, not
over the full-AOI mosaic. This uses the mosaic (tiles do not exist yet). The per-year
LEVEL is what is being tested, which the mosaic represents directly, but the numbers
are not byte-identical to what a retrain would see.
"""
import numpy as np
import rasterio
from pathlib import Path

IMG = Path("data/imagery")
YEARS = [2019, 2020, 2021, 2022, 2023, 2024]
# docs/BASELINE_LADDER_RESULTS.md section 5.3
OFFICIAL = {2019: 36296, 2020: 35277, 2021: 38285, 2022: 37965, 2023: 39815, 2024: 44240}
# 1-based rasterio band numbers; order is s2(10) + indices(6) + s1(2)
BANDS = {"B11": 9, "B12": 10, "NDVI": 11, "NBR": 16}
PRIOR = {"B11": +1, "B12": +1, "NDVI": -1, "NBR": -1}
R_BAR = 0.70

names = (IMG / "catatumbo_2019_annual_full.bands.txt").read_text().split()
for nm, bidx in BANDS.items():
    assert names[bidx - 1] == nm, f"band map drift: expected {nm} at {bidx}, got {names[bidx-1]}"
print("band map verified against .bands.txt\n")

stats = {nm: {"mean": [], "std": []} for nm in BANDS}
for yr in YEARS:
    p = IMG / f"catatumbo_{yr}_annual_full.tif"
    with rasterio.open(p) as src:
        for nm, bidx in BANDS.items():
            a = src.read(bidx, masked=True).compressed()   # full resolution, one band
            a = a[np.isfinite(a)]
            a = a[a != 0.0]                                # AOI-edge fill stored as 0
            stats[nm]["mean"].append(float(a.mean()))
            stats[nm]["std"].append(float(a.std()))

y = np.array([OFFICIAL[k] for k in YEARS], dtype=float)


def pearson(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    return float(np.corrcoef(a, b)[0, 1])


print("Official ha:", "  ".join(f"{k}={OFFICIAL[k]:,}" for k in YEARS))
print(f"target mean={y.mean():,.0f}  std={y.std(ddof=0):,.0f}  CV={y.std(ddof=0)/y.mean():.3f}\n")

hdr = f"{'candidate':14s}{'r':>8s}{'prior':>7s}{'sign_ok':>9s}{'|r|>=.70':>10s}{'LOO r range':>22s}   values"
print(hdr); print("-" * len(hdr))

results = []
for nm in BANDS:
    for kind in ("mean", "std"):
        x = stats[nm][kind]
        r = pearson(x, y)
        prior = PRIOR[nm] if kind == "mean" else 0   # no directional prior for std
        sign_ok = (prior == 0) or (np.sign(r) == prior)
        loo = [pearson([v for j, v in enumerate(x) if j != i],
                       [v for j, v in enumerate(y) if j != i]) for i in range(len(x))]
        passes = abs(r) >= R_BAR and sign_ok and prior != 0
        results.append((f"{nm}.{kind}", r, prior, sign_ok, passes, min(loo), max(loo)))
        pstr = {1: "+", -1: "-", 0: "none"}[prior]
        vals = " ".join(f"{v:.4f}" for v in x)
        print(f"{nm+'.'+kind:14s}{r:8.3f}{pstr:>7s}{str(sign_ok):>9s}{str(abs(r)>=R_BAR):>10s}"
              f"{f'[{min(loo):+.2f}, {max(loo):+.2f}]':>22s}   {vals}")

print()
winners = [x for x in results if x[4]]
strong_wrong_sign = [x for x in results if abs(x[1]) >= R_BAR and not x[3]]
# A no-prior (std) candidate clears |r| but can never pass, and sign_ok is True for it, so
# it lands in neither list above. Surface it or a strong std correlation would be invisible.
strong_no_prior = [x for x in results if abs(x[1]) >= R_BAR and x[2] == 0]

print("=" * 78)
if winners:
    print("A12 VERDICT: SIGNAL FOUND -> retrain is justified")
    for w in winners:
        print(f"   {w[0]}: r={w[1]:+.3f}, LOO range [{w[5]:+.2f}, {w[6]:+.2f}]")
    print("   Per A12 aux spec: 2-4 scalars max, anomaly-coded vs TRAIN-year pool,")
    print("   plus the pre-registered shuffle/memorization falsification test.")
else:
    print("A12 VERDICT: NO LEVEL SIGNAL -> DO NOT RETRAIN")
    print("   No mean candidate clears |r|>=0.70 with the directionally-sensible sign.")
    print("   Per prereg A12: aux inputs would restore a signal that does not exist.")
    print("   Route counting to the Phase 6.3 hybrid anchor instead.")
if strong_wrong_sign:
    print("\n   NOTE - strong but WRONG-SIGNED (not evidence, prior was fixed in advance):")
    for s in strong_wrong_sign:
        print(f"     {s[0]}: r={s[1]:+.3f} (prior expected {'+' if s[2]>0 else '-' if s[2]<0 else 'none'})")
if strong_no_prior:
    print("\n   REPORTED BUT NOT DECISIVE - |r|>=0.70 with NO directional prior:")
    for s in strong_no_prior:
        print(f"     {s[0]}: r={s[1]:+.3f}, LOO range [{s[5]:+.2f}, {s[6]:+.2f}]")
    print("     Cannot pass the bar by construction (the rule needs a prior sign), and no")
    print("     prior was defensible for a spread. Recorded as a finding, not as evidence;")
    print("     if this is to drive anything it needs its own pre-registration.")
print("=" * 78)
