"""Phase 6.6b model-free acceptance check for the Sentinel-2 baseline-04.00 offset fix.

Compares per-band medians of the annual mosaics. The decisive test is differential:
every visible band stepped +0.100 at exactly 2021->2022 under the bug (= 1000/10000),
so after a correct fix that step must vanish. Run with no model, no training.

Fill pixels at the AOI edge are stored as 0 rather than nodata, so zeros are excluded;
medians would otherwise be dragged toward 0.
"""
import numpy as np
import rasterio
from pathlib import Path

IMG = Path("data/imagery")
names = (IMG / "catatumbo_2019_annual_full.bands.txt").read_text().split()
CLEAN = (2019, 2020, 2021)    # pre-offset (baseline 04.00 starts 2022-01-25)
FIXED = (2022, 2023, 2024)    # baseline 04.00 era, previously inflated ~+0.1

med = {}
for yr in (2019, 2020, 2021, 2022, 2023, 2024):
    p = IMG / f"catatumbo_{yr}_annual_full.tif"
    if not p.exists():
        continue
    # The band map below is 2019's sidecar applied to every year's array by INDEX, so a
    # year exported with a different band set would be silently mislabelled. Verify per
    # year, the same way scripts/a12_level_signal.py verifies its band map.
    side = IMG / f"catatumbo_{yr}_annual_full.bands.txt"
    assert side.exists(), f"{yr}: missing {side.name}; cannot verify band order"
    yr_names = side.read_text().split()
    assert yr_names == names, (f"band map drift: {yr} bands {yr_names} != 2019 bands {names}")
    with rasterio.open(p) as src:
        assert src.count == len(names), \
            f"band map drift: {yr} raster has {src.count} bands, sidecar lists {len(names)}"
        arr = src.read(out_shape=(src.count, src.height // 8, src.width // 8), masked=True)
    vals = []
    for i in range(len(names)):
        b = arr[i].compressed()
        b = b[np.isfinite(b)]
        b = b[b != 0.0]
        vals.append(float(np.median(b)) if b.size else float("nan"))
    med[yr] = vals
    arr = None

years = sorted(med)
print("PHASE 6.6b MODEL-FREE ACCEPTANCE CHECK")
print(f"clean (pre-offset): {CLEAN}   previously contaminated: {FIXED}")
print(f"years available: {years}\n")

hdr = "band    " + "".join(f"{y:>9}" for y in years) + "   d(22-21)"
print(hdr)
print("-" * len(hdr))
for i, nm in enumerate(names):
    row = f"{nm:6s}  " + "".join(f"{med[y][i]:9.4f}" for y in years)
    d = med[2022][i] - med[2021][i] if 2022 in med and 2021 in med else float("nan")
    print(row + f"{d:11.4f}")

vis = [names.index(b) for b in ("B02", "B03", "B04")]
# The whole acceptance test is the 2021->2022 step, so a missing year is not a degraded
# run, it is no run at all: fail loudly instead of averaging an empty list into nan.
missing = [y for y in (2021, 2022) if y not in med]
if missing:
    raise SystemExit(f"cannot run the acceptance check: mosaic missing for {missing} "
                     f"(years found: {years}). The decisive test is the 2021->2022 step; "
                     f"export the missing year(s) first.")
steps = [med[2022][i] - med[2021][i] for i in vis]
print("\nVISIBLE-BAND STEP 2021->2022  (bug produced ~+0.1000; must now be ~0)")
for b, s in zip(("B02", "B03", "B04"), steps):
    print(f"   {b}: {s:+.4f}")
mean_step = float(np.mean(steps))
print(f"   mean: {mean_step:+.4f}")

ndvi = names.index("NDVI")
clean_ndvi = float(np.mean([med[y][ndvi] for y in CLEAN if y in med]))
print(f"\nNDVI   clean-year baseline (this basis) = {clean_ndvi:.4f}")
for y in years:
    tag = "clean" if y in CLEAN else "fixed"
    print(f"       {y} ({tag}) = {med[y][ndvi]:.4f}   delta_vs_clean = {med[y][ndvi] - clean_ndvi:+.4f}")
print("   NB the doc's '~0.68' is a different aggregation basis; the like-for-like")
print("   comparison is against the clean-year value measured identically.")

ok_step = abs(mean_step) < 0.02
ok_ndvi = all(abs(med[y][ndvi] - clean_ndvi) < 0.10 for y in FIXED if y in med)
print(f"\nSTEP TEST : {'PASS' if ok_step else 'FAIL'}   |mean visible step| = {abs(mean_step):.4f}  (threshold 0.02)")
print(f"NDVI TEST : {'PASS' if ok_ndvi else 'FAIL'}   all fixed years within 0.10 of clean baseline")
print(f"\nVERDICT   : {'ACCEPT - offset fix is correct' if (ok_step and ok_ndvi) else 'REJECT - do not proceed to training'}")
