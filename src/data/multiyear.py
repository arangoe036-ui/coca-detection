"""v2.1 Phase 1 — build a pooled multi-year dataset (2019–2024).

For each year: rasterize the official coca fraction label aligned to that year's
imagery, tile the stack+mask, and append the tiles to a master index tagged with
the year. The spatial block split is reused unchanged across years (blocks derive
from pixel position on the identical AOI grid, assigned with a fixed seed), so a
given geographic block stays in the same fold every year — no spatial leakage and
clean leave-one-year-out folds.

    python -m src.data.multiyear --years 2019 2020 2021 2022 2023 2024

Requires each year's imagery at data/imagery/<region>_<year>_annual_full.tif.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import rasterio

from src.data.labels import fetch_coca_grid, rasterize_mask
from src.data.tiling import _assign_blocks, _windows
from src.utils import ensure_dirs, load_config

# Official coca-grid field per year (naming is irregular on Socrata v3rx-q7t3).
GRID_FIELD = {2019: "areacoca_2019", 2020: "areacoca_2020", 2021: "areacoca_2021",
              2022: "coca2022_", 2023: "areacoca2023", 2024: "areacoca2024"}

INDEX_NAME = "multiyear_index.csv"


def _year_cfg(cfg: dict, year: int) -> dict:
    """Config view with year + its irregular grid field."""
    return {**cfg, "year": year,
            "labels": {**cfg["labels"], "coca_grid_year_field": GRID_FIELD[year]}}


def build_year(cfg: dict, year: int) -> list[dict]:
    """Rasterize the year's label mask and tile it; return tile records (year-tagged)."""
    region = cfg["aoi"]["region"]
    img_path = f"data/imagery/{region}_{year}_annual_full.tif"
    if not Path(img_path).exists():
        raise FileNotFoundError(f"Missing imagery for {year}: {img_path} (export it first).")

    ycfg = _year_cfg(cfg, year)
    gdf = fetch_coca_grid(ycfg, tuple(cfg["aoi"]["bbox"]))
    mask_path = rasterize_mask(ycfg, gdf, img_path)

    tpx = cfg["tiling"]["tile_px"]
    stride = cfg["tiling"]["stride_px"]
    block_px = int(round(cfg["tiling"]["split_block_km"] * 1000 / cfg["imagery"]["resolution_m"]))
    out_dir = Path(cfg["paths"]["tiles_dir"]) / str(year)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    with rasterio.open(img_path) as img, rasterio.open(mask_path) as msk:
        for (x, y) in _windows(img.width, img.height, tpx, stride):
            win = rasterio.windows.Window(x, y, tpx, tpx)
            stack = img.read(window=win).astype("float32")
            m = msk.read(1, window=win)
            if stack.shape[1:] != (tpx, tpx):
                continue
            if np.isfinite(stack).all(axis=0).mean() < 0.5:
                continue
            block = f"{x // block_px}_{y // block_px}"
            i = len(records)
            npz = out_dir / f"tile_{i:05d}.npz"
            np.savez_compressed(npz, image=stack, mask=m)
            records.append({"tile_id": f"{year}_{i:05d}", "year": year, "block": block,
                            "x": x, "y": y, "pos_frac": float((m > 0).mean()),
                            "npz": f"{year}/{npz.name}"})
    print(f"[multiyear] {year}: {len(records)} tiles")
    return records


def build_all(cfg: dict, years: list[int]) -> Path:
    ensure_dirs(cfg)
    all_recs = []
    for y in years:
        all_recs.extend(build_year(cfg, y))

    # Assign whole spatial blocks to splits ONCE (same across years) for clean LOYO.
    assign = _assign_blocks([r["block"] for r in all_recs], cfg["tiling"]["split_ratios"],
                            cfg["project"]["seed"])
    index_path = Path(cfg["paths"]["tiles_dir"]) / INDEX_NAME
    counts = {}
    with open(index_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["tile_id", "year", "split", "block", "x", "y", "pos_frac", "npz"])
        for r in all_recs:
            split = assign[r["block"]]
            counts[(r["year"], split)] = counts.get((r["year"], split), 0) + 1
            w.writerow([r["tile_id"], r["year"], split, r["block"], r["x"], r["y"],
                        f"{r['pos_frac']:.4f}", r["npz"]])
    print(f"[multiyear] wrote {index_path}  ({len(all_recs)} tiles)")
    for y in years:
        print(f"           {y}: " + " ".join(f"{s}={counts.get((y, s), 0)}" for s in ("train", "val", "test")))
    return index_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build pooled multi-year dataset (v2.1 Phase 1).")
    ap.add_argument("--config", default=None)
    ap.add_argument("--years", type=int, nargs="+", default=[2019, 2020, 2021, 2022, 2023, 2024])
    args = ap.parse_args()
    build_all(load_config(args.config), args.years)
