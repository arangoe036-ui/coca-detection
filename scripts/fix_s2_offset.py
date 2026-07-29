"""Phase 6.6a — correct the Sentinel-2 baseline-04.00 offset in EXISTING artifacts.

The offset is a known additive constant (+1000 DN = +0.1 reflectance) present in
2022+ S2 data (confirmed model-free in Phase 6.5: +0.100 step at 2021->2022 in every
visible band, SAR flat). So we correct in place rather than re-export: subtract 0.1
from the 10 S2 reflectance bands and RECOMPUTE the 6 index channels from the corrected
reflectances (indices are nonlinear, so they must be recomputed, not shifted). SAR
(bands 16,17) is untouched.

In-place correction is chosen because (a) it is exact for our data — empirically the
contamination is confined to 2022-2024 (2019-2021 reflectances are clean-low), so the
year selection is validated by the model-free check in 6.6b — and (b) it leaves the
tile geometry and therefore `multiyear_index.csv` (block->split) BYTE-IDENTICAL, which
the §8.1 hard rule requires. It is mathematically equivalent to the fixed
`stac_export.py` for pure post-2022 composites (median(x-0.1) = median(x)-0.1).

    python scripts/fix_s2_offset.py --years 2022 2023 2024
    python scripts/fix_s2_offset.py --years 2025 2026 --rasters-only   # nowcast inputs

Backs up originals to data/_precorrection_backup/ and refuses to double-correct.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import rasterio

from src.utils import load_config

OFFSET = 0.1          # reflectance units (= 1000 DN / 10000)
N_REFL = 10           # bands 0-9 are S2 reflectance (band_order: s2 + indices + s1)
EPS = 1e-6
MARKER = "data/.s2_offset_fixed.json"


def recompute_indices(img):
    """Indices 10-15 from corrected reflectances — identical formulas to
    stac_export.add_indices. img: (18,H,W)."""
    B02, B03, B04, B08 = img[0], img[1], img[2], img[3]
    B05, B12 = img[4], img[9]
    ndvi = (B08 - B04) / (B08 + B04 + EPS)
    evi = 2.5 * (B08 - B04) / (B08 + 6 * B04 - 7.5 * B02 + 1 + EPS)
    savi = 1.5 * (B08 - B04) / (B08 + B04 + 0.5 + EPS)
    ndwi = (B03 - B08) / (B03 + B08 + EPS)
    ndre = (B08 - B05) / (B08 + B05 + EPS)
    nbr = (B08 - B12) / (B08 + B12 + EPS)
    return np.stack([ndvi, evi, savi, ndwi, ndre, nbr]).astype("float32")


def correct_stack(img):
    """Subtract the offset from reflectance bands (finite pixels only) and recompute
    indices. Returns a corrected copy; SAR bands 16,17 untouched."""
    out = img.astype("float32").copy()
    for c in range(N_REFL):
        m = np.isfinite(out[c])
        out[c][m] = out[c][m] - OFFSET
    out[10:16] = recompute_indices(out)
    return out


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_marker():
    p = Path(MARKER)
    return json.loads(p.read_text()) if p.exists() else {"corrected_years": []}


def fix_tiles(cfg, year, backup_root):
    tiles_dir = Path(cfg["paths"]["tiles_dir"]) / str(year)
    npzs = sorted(tiles_dir.glob("tile_*.npz"))
    if not npzs:
        print(f"[fix] {year}: no tiles at {tiles_dir}"); return 0
    bdir = backup_root / "tiles" / str(year); bdir.mkdir(parents=True, exist_ok=True)
    for f in npzs:
        shutil.copy2(f, bdir / f.name)
        d = np.load(f)
        np.savez_compressed(f, image=correct_stack(d["image"]), mask=d["mask"])
    print(f"[fix] {year}: corrected {len(npzs)} tiles (backup -> {bdir})")
    return len(npzs)


def fix_raster(cfg, year, backup_root):
    region = cfg["aoi"]["region"]
    path = Path(f"data/imagery/{region}_{year}_annual_full.tif")
    if not path.exists():
        print(f"[fix] {year}: no raster {path}"); return False
    bdir = backup_root / "imagery"; bdir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, bdir / path.name)
    tmp = path.with_suffix(".fixed.tif")
    with rasterio.open(path) as src:
        prof = src.profile
        with rasterio.open(tmp, "w", **prof) as dst:
            for _, win in src.block_windows(1):
                arr = src.read(window=win)  # (18, wh, ww)
                dst.write(correct_stack(arr), window=win)
    tmp.replace(path)
    print(f"[fix] {year}: corrected raster {path.name} (backup -> {bdir/path.name})")
    return True


def main():
    ap = argparse.ArgumentParser(description="Fix S2 baseline-04.00 offset in place.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--years", type=int, nargs="+", required=True)
    ap.add_argument("--rasters-only", action="store_true",
                    help="correct only annual rasters (e.g. 2025/2026 nowcast inputs)")
    args = ap.parse_args()
    cfg = load_config(args.config)

    marker = load_marker()
    already = set(marker["corrected_years"])
    todo = [y for y in args.years if y not in already]
    skip = [y for y in args.years if y in already]
    if skip:
        print(f"[fix] REFUSING to re-correct already-fixed years: {skip}")
    if not todo:
        print("[fix] nothing to do."); return

    backup_root = Path("data/_precorrection_backup")
    idx = Path(cfg["paths"]["tiles_dir"]) / "multiyear_index.csv"
    idx_hash_before = _sha(idx) if idx.exists() else None

    for y in todo:
        if not args.rasters_only:
            fix_tiles(cfg, y, backup_root)
        fix_raster(cfg, y, backup_root)

    # §8.1 hard rule: block->split assignment (the index) must be untouched.
    idx_hash_after = _sha(idx) if idx.exists() else None
    assert idx_hash_before == idx_hash_after, \
        "multiyear_index.csv changed — block->split assignment violated!"
    print(f"[fix] OK: multiyear_index.csv byte-identical ({idx_hash_after[:12]}...)")

    marker["corrected_years"] = sorted(already | set(todo))
    marker["offset_reflectance"] = OFFSET
    Path(MARKER).write_text(json.dumps(marker, indent=2))
    print(f"[fix] marked corrected years: {marker['corrected_years']}")


if __name__ == "__main__":
    main()
