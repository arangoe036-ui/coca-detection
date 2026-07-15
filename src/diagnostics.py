"""v2.1 Phase 0 — stable-core diagnostic (domain shift vs coverage gap).

Compares the 2023 model's predicted density on the PERSISTENT CORE (cells that
were official coca in both a base year and a test year) across those two years.
If the test year reads clearly lower on the *same known-coca ground*, the miss is
a domain/calibration shift on newer imagery — not just missed new fields.

    python -m src.diagnostics --base 2023 --test 2024

Finding (2023 vs 2024): core mean 0.0975 -> 0.0774 (ratio 0.79) => domain shift
dominates the 2024 undercount (AOI ratio 0.71). Phase 2 normalization matters most.
"""

from __future__ import annotations

import argparse

import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.warp import Resampling, reproject

from src.data.labels import fetch_coca_grid
from src.utils import load_config

_GRID_FIELD = {2021: "areacoca_2021", 2022: "coca2022_", 2023: "areacoca2023", 2024: "areacoca2024"}


def _read_aligned(path: str, transform, width, height, crs):
    with rasterio.open(path) as ds:
        if (ds.width, ds.height) == (width, height) and ds.transform == transform:
            return ds.read(1)
        out = np.zeros((height, width), "float32")
        reproject(ds.read(1), out, src_transform=ds.transform, src_crs=ds.crs,
                  dst_transform=transform, dst_crs=crs, resampling=Resampling.average)
        return out


def _official_mask(cfg, field, transform, width, height, crs):
    g = fetch_coca_grid({**cfg, "labels": {**cfg["labels"], "coca_grid_year_field": field}},
                        tuple(cfg["aoi"]["bbox"])).to_crs(crs)
    return rasterize([(geom, 1) for geom in g.geometry], out_shape=(height, width),
                     transform=transform, fill=0, dtype="uint8") > 0


def stable_core_check(cfg, base_year: int, test_year: int) -> dict:
    region = cfg["aoi"]["region"]
    with rasterio.open(f"outputs/{region}_{base_year}_coca_density.tif") as d:
        pred_base, T, W, H, crs = d.read(1), d.transform, d.width, d.height, d.crs
    pred_test = _read_aligned(f"outputs/{region}_{test_year}_coca_density.tif", T, W, H, crs)

    core = _official_mask(cfg, _GRID_FIELD[base_year], T, W, H, crs) & \
        _official_mask(cfg, _GRID_FIELD[test_year], T, W, H, crs)
    mb, mt = float(pred_base[core].mean()), float(pred_test[core].mean())
    ratio = mt / mb if mb else float("nan")
    verdict = "DOMAIN SHIFT (Phase 2 normalization matters most)" if ratio < 0.9 \
        else "coverage gap (Phase 1 multi-year extent matters most)"
    print(f"[diag] persistent core: {int(core.sum()):,} px")
    print(f"[diag] core mean density {base_year}={mb:.4f} {test_year}={mt:.4f} ratio={ratio:.2f} -> {verdict}")
    return {"core_px": int(core.sum()), "mean_base": mb, "mean_test": mt, "ratio": ratio, "verdict": verdict}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="v2.1 Phase 0 stable-core diagnostic.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--base", type=int, default=2023)
    ap.add_argument("--test", type=int, default=2024)
    args = ap.parse_args()
    stable_core_check(load_config(args.config), args.base, args.test)
