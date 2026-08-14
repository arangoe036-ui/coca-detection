"""Report the fraction of the AOI where Sentinel-2 carries no usable data.

Prereg A13 metric. Works on composites from BOTH pipeline versions:
  * pre-fix  (`scl_mask_classes` omitting SCL 0/1): nodata surfaces as exactly 0.0
  * post-fix (0/1 masked):                          nodata surfaces as NaN

so "blank" = all ten S2 reflectance bands are non-finite OR all exactly zero. Sentinel-1 is
reported alongside because it distinguishes the two causes: S1 valid under an S2 blank means
real S2 loss, S1 invalid too means genuine mosaic edge fill.

    python scripts/blank_footprint.py              # every year found
    python scripts/blank_footprint.py --year 2020  # one year
"""
import argparse
import numpy as np
import rasterio
from pathlib import Path

IMG = Path("data/imagery")
D = 8  # decimation; blank regions are large and contiguous, so this is ample


def blank_stats(path: Path) -> dict:
    names = (path.parent / f"{path.stem}.bands.txt").read_text().split()
    n_s2 = 10  # s2(10) + indices(6) + s1(2)
    with rasterio.open(path) as src:
        assert src.count == len(names), f"band count {src.count} != {len(names)} in sidecar"
        h, w = src.height // D, src.width // D
        s2 = src.read(list(range(1, n_s2 + 1)), out_shape=(n_s2, h, w), masked=True).filled(np.nan)
        vv = src.read(names.index("VV") + 1, out_shape=(h, w), masked=True).filled(np.nan)

    nonfinite = ~np.isfinite(s2)
    blank = nonfinite.all(axis=0) | (np.nan_to_num(s2, nan=0.0) == 0.0).all(axis=0)
    vv_ok = np.isfinite(vv) & (vv != 0.0)
    under = vv_ok[blank]
    return {
        "blank_pct": 100 * float(blank.mean()),
        "nan_pct": 100 * float(nonfinite.all(axis=0).mean()),
        "zero_pct": 100 * float((np.nan_to_num(s2, nan=1.0) == 0.0).all(axis=0).mean()),
        "s1_valid_under_blank_pct": 100 * float(under.mean()) if under.size else float("nan"),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=None)
    args = ap.parse_args()

    paths = sorted(IMG.glob("catatumbo_*_annual_full.tif"))
    if args.year:
        paths = [p for p in paths if str(args.year) in p.name]
    if not paths:
        raise SystemExit("no annual mosaics found")

    print(f"{'year':6s}{'BLANK %':>10s}{'(as NaN)':>10s}{'(as zero)':>11s}"
          f"{'S1 valid under blank':>23s}   reading")
    for p in paths:
        yr = p.stem.split("_")[1]
        s = blank_stats(p)
        if np.isnan(s["s1_valid_under_blank_pct"]):
            reading = "no blank pixels"
        elif s["s1_valid_under_blank_pct"] > 50:
            reading = "REAL S2 DATA LOSS"
        else:
            reading = "mosaic edge fill (benign)"
        print(f"{yr:6s}{s['blank_pct']:9.2f}%{s['nan_pct']:9.2f}%{s['zero_pct']:10.2f}%"
              f"{s['s1_valid_under_blank_pct']:22.1f}%   {reading}")

    print("\nPrereg A13 decision bar (2020 pilot): <5% -> our filtering was the cause, proceed to")
    print("full re-export; >15% -> the archive lacks acquisitions, Phase 8.3 (HLS) reinstated;")
    print("5-15% -> partial, proceed but keep 8.3 open.")
