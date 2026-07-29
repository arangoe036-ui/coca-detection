"""Phase 6.1 — per-year satellite coverage diagnostic (all six LOYO years).

Extends src/coverage.py to 2019–2024 and measures OPTICAL and SAR coverage
SEPARATELY (prereg A7): they degrade for different reasons in different years and
conflating them hides the mechanism. Per year and full calendar window:
  - S2: number of scenes, mean/median CLEAR observations per pixel (SCL-based).
  - S1: number of RTC passes (scenes, S1C excluded as in training) and mean
    per-pixel valid VV observations.
Then relate coverage to the U-Net's out-of-year error |aoi_ratio − 1| (frozen v2.1),
descriptively — n=6, no p-values, no regression line (prereg A7).

    python -m src.coverage_by_year        # writes docs/coverage_by_year.md

Network: Planetary Computer STAC (same backend as stac_export).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.coverage import CLEAR_SCL
from src.data.stac_export import get_catalog
from src.utils import load_config

YEARS = [2019, 2020, 2021, 2022, 2023, 2024]

# Frozen U-Net LOYO aoi_ratio (docs/v2.1_loyo_results.md) — the error we explain.
UNET_RATIO = {2019: 0.95, 2020: 0.59, 2021: 0.90, 2022: 1.48, 2023: 1.01, 2024: 0.79}


def s2_stats(cfg, year, res_m=200):
    from odc.stac import load as odc_load
    cat = get_catalog(cfg)
    dr = f"{year}-01-01/{year}-12-31"
    items = list(cat.search(collections=[cfg["imagery"]["s2_collection"]],
                            bbox=cfg["aoi"]["bbox"], datetime=dr).items())
    ds = odc_load(items, bands=["SCL"], bbox=cfg["aoi"]["bbox"],
                  crs=f"EPSG:{cfg['aoi']['utm_epsg']}", resolution=res_m,
                  chunks={}, groupby="solar_day")
    clear = ds["SCL"].isin(CLEAR_SCL).sum(dim="time").compute().values.astype("float32")
    return {"s2_scenes": len(items), "s2_mean_clear": float(clear.mean()),
            "s2_median_clear": float(np.median(clear))}


def s1_stats(cfg, year, res_m=200):
    from odc.stac import load as odc_load
    cat = get_catalog(cfg)
    dr = f"{year}-01-01/{year}-12-31"
    items = list(cat.search(collections=[cfg["imagery"]["s1_collection"]],
                            bbox=cfg["aoi"]["bbox"], datetime=dr).items())
    # exclude S1C (malformed on PC + absent from 2019–2024 training) — matches stac_export
    items = [it for it in items if not it.id.upper().startswith("S1C")]
    ds = odc_load(items, bands=["vv"], bbox=cfg["aoi"]["bbox"],
                  crs=f"EPSG:{cfg['aoi']['utm_epsg']}", resolution=res_m,
                  chunks={}, groupby="solar_day")
    valid = (ds["vv"] > 0).sum(dim="time").compute().values.astype("float32")
    return {"s1_passes": len(items), "s1_mean_obs": float(valid.mean()),
            "s1_median_obs": float(np.median(valid))}


def run(cfg):
    rows = {}
    for y in YEARS:
        s2 = s2_stats(cfg, y)
        s1 = s1_stats(cfg, y)
        err = abs(UNET_RATIO[y] - 1.0)
        rows[y] = {**s2, **s1, "ratio": UNET_RATIO[y], "abs_err": err}
        print(f"[cov] {y}: S2 scenes={s2['s2_scenes']:3d} clear/px={s2['s2_mean_clear']:5.1f} "
              f"| S1 passes={s1['s1_passes']:3d} obs/px={s1['s1_mean_obs']:5.1f} "
              f"| ratio={UNET_RATIO[y]:.2f} |err|={err:.2f}", flush=True)

    # rank years by each coverage axis (lowest first) and by error (highest first)
    by_s2 = sorted(YEARS, key=lambda y: rows[y]["s2_mean_clear"])
    by_s1 = sorted(YEARS, key=lambda y: rows[y]["s1_mean_obs"])
    by_err = sorted(YEARS, key=lambda y: -rows[y]["abs_err"])

    lines = [
        "# Phase 6.1 — Per-year coverage vs out-of-year error (Catatumbo, 2019–2024)",
        "",
        "Optical (Sentinel-2 clear observations) and SAR (Sentinel-1 passes) measured",
        "**separately** per full calendar year (prereg A7). `|ratio−1|` is the U-Net's",
        "frozen LOYO out-of-year error (`docs/v2.1_loyo_results.md`). n=6 — descriptive",
        "only, no p-values, no fitted line.",
        "",
        "| year | S2 scenes | S2 clear/px (mean) | S2 median | S1 passes | S1 obs/px (mean) | U-Net ratio | \\|ratio−1\\| |",
        "|---|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for y in YEARS:
        r = rows[y]
        lines.append(f"| {y} | {r['s2_scenes']} | {r['s2_mean_clear']:.1f} | "
                     f"{r['s2_median_clear']:.0f} | {r['s1_passes']} | {r['s1_mean_obs']:.1f} | "
                     f"{r['ratio']:.2f} | {r['abs_err']:.2f} |")
    lines += [
        "",
        "## Rankings (for the descriptive read)",
        f"- Lowest S2 clear coverage → highest: {' < '.join(str(y) for y in by_s2)}",
        f"- Lowest S1 passes → highest: {' < '.join(str(y) for y in by_s1)}",
        f"- Largest \\|ratio−1\\| → smallest: {' > '.join(str(y) for y in by_err)}",
        "",
        "The two worst U-Net years are "
        f"**{by_err[0]}** (|err|={rows[by_err[0]]['abs_err']:.2f}) and "
        f"**{by_err[1]}** (|err|={rows[by_err[1]]['abs_err']:.2f}).",
    ]
    doc = Path("docs/coverage_by_year.md")
    doc.write_text("\n".join(lines) + "\n")
    print(f"[cov] wrote {doc}")
    return rows


if __name__ == "__main__":
    run(load_config())
