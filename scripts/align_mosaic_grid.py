"""Pad a mosaic onto the reference annual grid so every year is pixel-identical.

WHY. 2023 had to be exported with `--subtile-deg 0.15` because at 0.3 deg its ~365 scenes per
sub-tile meant a single transient read failure (which restarts the whole sub-tile) made it never
finish. But `_subtile_bboxes` derives its edge overlap from the step (`ov = step_deg * 0.02`),
so a smaller step produces a smaller union: 2023 came out 5018x5595 at left=686380/top=1029420
versus 5051x5628 at left=686060/top=1029740 for every other year — 320 m (=16 px) offset and 33
px smaller.

That is not a leak (blocks and splits stayed consistent) but it breaks the property the
leave-one-year-out folds rest on: a given block must cover the SAME ground every year. 44 of 500
tile positions differed.

320 m is an exact multiple of the 20 m grid, so the two rasters share a lattice and the fix is
an exact pixel-aligned pad — no resampling, no interpolation. The added margin is written as
NaN because 2023 genuinely has no data there: the smaller sub-tile union never covered it.

    python scripts/align_mosaic_grid.py --year 2023            # dry run
    python scripts/align_mosaic_grid.py --year 2023 --apply
"""
import argparse
import shutil
from pathlib import Path

import numpy as np
import rasterio

IMG = Path("data/imagery")
REF_YEAR = 2019  # any year on the majority grid
RES_TOL = 1e-6


def main(year: int, apply: bool) -> int:
    ref_p = IMG / f"catatumbo_{REF_YEAR}_annual_full.tif"
    tgt_p = IMG / f"catatumbo_{year}_annual_full.tif"
    for p in (ref_p, tgt_p):
        if not p.exists():
            raise SystemExit(f"missing {p}")

    with rasterio.open(ref_p) as ref, rasterio.open(tgt_p) as tgt:
        rb, tb = ref.bounds, tgt.bounds
        res = abs(ref.transform.a)
        print(f"reference {REF_YEAR}: {ref.width}x{ref.height} left={rb.left} top={rb.top} res={res}")
        print(f"target    {year}: {tgt.width}x{tgt.height} left={tb.left} top={tb.top} "
              f"res={abs(tgt.transform.a)}")

        if (ref.width, ref.height) == (tgt.width, tgt.height) and rb == tb:
            print("already aligned — nothing to do")
            return 0
        if abs(abs(tgt.transform.a) - res) > RES_TOL or ref.crs != tgt.crs:
            raise SystemExit("resolution or CRS differ — an exact pad is not possible")
        if tgt.count != ref.count:
            raise SystemExit(f"band count differs ({tgt.count} vs {ref.count})")

        dx = (tb.left - rb.left) / res
        dy = (rb.top - tb.top) / res
        if abs(dx - round(dx)) > 1e-6 or abs(dy - round(dy)) > 1e-6:
            raise SystemExit(f"offset not a whole number of pixels (dx={dx}, dy={dy}) — "
                             "grids are not co-registered, padding would misalign data")
        col_off, row_off = int(round(dx)), int(round(dy))
        print(f"offset: {col_off} px right, {row_off} px down "
              f"({col_off*res:.0f} m, {row_off*res:.0f} m)")
        if col_off < 0 or row_off < 0 or col_off + tgt.width > ref.width \
                or row_off + tgt.height > ref.height:
            raise SystemExit("target is not contained by the reference extent — cannot pad")

        added = (ref.width * ref.height - tgt.width * tgt.height) / (ref.width * ref.height)
        print(f"padding adds {100*added:.2f}% of the canvas as NaN (no data was exported there)")
        if not apply:
            print("\nDRY RUN — pass --apply to write")
            return 0

        profile = ref.profile.copy()
        profile.update(count=ref.count, dtype="float32", compress="deflate", nodata=float("nan"))
        backup = tgt_p.with_name(f"UNALIGNED_{tgt_p.name}")
        shutil.copy2(tgt_p, backup)
        print(f"backed up original -> {backup.name}")

        out = tgt_p.with_suffix(".aligned.tif")
        with rasterio.open(out, "w", **profile) as dst:
            for b in range(1, ref.count + 1):
                canvas = np.full((ref.height, ref.width), np.nan, dtype="float32")
                canvas[row_off:row_off + tgt.height,
                       col_off:col_off + tgt.width] = tgt.read(b).astype("float32")
                dst.write(canvas, b)
        # carry the band-name sidecar across unchanged
        side = tgt_p.parent / f"{tgt_p.stem}.bands.txt"
        if side.exists():
            shutil.copy2(side, out.parent / f"{out.stem}.bands.txt")

    tgt_p.unlink()
    out.replace(tgt_p)
    (out.parent / f"{out.stem}.bands.txt").unlink(missing_ok=True)

    with rasterio.open(tgt_p) as chk, rasterio.open(ref_p) as ref:
        ok = (chk.width, chk.height) == (ref.width, ref.height) and chk.bounds == ref.bounds
        print(f"\nresult: {chk.width}x{chk.height} bounds={chk.bounds}")
        print("ALIGNED" if ok else "*** STILL MISALIGNED ***")
        return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--apply", action="store_true")
    raise SystemExit(main(ap.parse_args().year, ap.parse_args().apply))
