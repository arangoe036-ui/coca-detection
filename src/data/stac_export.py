"""P1a — Export Sentinel-2 + Sentinel-1 seasonal composites + indices as a GeoTIFF.

Implements plan §6.1 using the **Microsoft Planetary Computer** STAC API (free, no
GEE, no signup) instead of Earth Engine. Builds cloud-masked Sentinel-2 seasonal
median composites, appends spectral indices, adds Sentinel-1 RTC (VV/VH) composites,
and writes a stacked GeoTIFF in the AOI's UTM CRS.

Band order in the output raster (matches model.in_channels):
    s2_bands (reflectance, float32) + indices + s1_bands (dB)

Smoke test (plan §3 "smallest end-to-end slice"):
    python -m src.data.stac_export --quick
    -> exports one season over aoi.quick_bbox; verify it opens in rasterio.

Full export:
    python -m src.data.stac_export            # all seasons over aoi.bbox
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.utils import ensure_dirs, load_config

# Assets that live at native 20 m on Planetary Computer's Sentinel-2 L2A.
_S2_20M = {"B05", "B06", "B07", "B8A", "B11", "B12"}

# ESA Processing Baseline 04.00 (effective 2022-01-25) set BOA_ADD_OFFSET = -1000 for
# L2A, so the correct DN->reflectance conversion is (DN - 1000)/10000. Scenes below
# baseline 04.00 have no offset. (Phase 6.6.)
_BOA_OFFSET_DN = -1000.0
_BASELINE_04 = 4.0
_BASELINE_04_DATE = "2022-01-25"


def boa_offset_dn(item) -> float:
    """DN offset to ADD before /10000 for one S2 item: -1000 if processing baseline
    >= 04.00, else 0. Keyed on the `s2:processing_baseline` metadata field (a
    reprocessed pre-2022 scene carries 04.00 and the offset); date only as fallback."""
    props = getattr(item, "properties", {}) or {}
    base = props.get("s2:processing_baseline")
    try:
        bnum = float(base)
    except (TypeError, ValueError):
        bnum = None
    if bnum is not None:
        return _BOA_OFFSET_DN if bnum >= _BASELINE_04 else 0.0
    return _BOA_OFFSET_DN if str(props.get("datetime", "")) >= _BASELINE_04_DATE else 0.0


def get_catalog(cfg: dict):
    """Open the PC STAC catalog with automatic asset signing."""
    import planetary_computer as pc
    from pystac_client import Client

    return Client.open(cfg["imagery"]["stac_url"], modifier=pc.sign_inplace)


def seasonal_ranges(year: int, n: int) -> list[tuple[str, str]]:
    """Split a calendar year into `n` contiguous date ranges (YYYY-MM-DD)."""
    edges = np.linspace(0, 365, n + 1).astype(int)
    base = np.datetime64(f"{year}-01-01")
    out = []
    for i in range(n):
        start = base + np.timedelta64(int(edges[i]), "D")
        end = base + np.timedelta64(int(edges[i + 1]) - 1, "D")
        out.append((str(start), str(end)))
    return out


def _search(catalog, collection: str, bbox, date_range, query=None):
    return list(
        catalog.search(
            collections=[collection],
            bbox=bbox,
            datetime=f"{date_range[0]}/{date_range[1]}",
            query=query,
        ).items()
    )


def build_s2_composite(catalog, cfg: dict, bbox, date_range):
    """Cloud-masked Sentinel-2 seasonal median composite -> xarray DataArray (reflectance)."""
    from odc.stac import load as odc_load

    img = cfg["imagery"]
    items = _search(
        catalog, img["s2_collection"], bbox, date_range,
        query={"eo:cloud_cover": {"lt": img["cloud_cover_max"]}},
    )
    if not items:
        raise RuntimeError(f"No Sentinel-2 scenes for {date_range} over {bbox}.")

    bands = list(img["s2_bands"]) + ["SCL"]
    ds = odc_load(
        items, bands=bands, bbox=bbox,
        crs=f"EPSG:{cfg['aoi']['utm_epsg']}", resolution=img["resolution_m"],
        chunks={"x": 1024, "y": 1024}, groupby="solar_day",
    )

    # SCL cloud/shadow/snow mask -> NaN, then median over time (skipna).
    scl = ds["SCL"]
    bad = scl.isin(img["scl_mask_classes"])
    refl_bands = img["s2_bands"]
    import pandas as pd
    import xarray as xr

    # Per-solar-day BOA additive offset (baseline 04.00 = -1000 DN), aligned to the
    # loaded time axis and applied PER SCENE BEFORE /10000 and before indices (6.6a).
    day_off = {str(pd.Timestamp(it.properties["datetime"]).date()): boa_offset_dn(it)
               for it in items}
    off = np.array([day_off.get(str(pd.Timestamp(t).date()), 0.0)
                    for t in ds["time"].values], dtype="float32")
    off_da = xr.DataArray(off, dims=["time"], coords={"time": ds["time"]})
    scaled = {}
    for b in refl_bands:
        # L2A (DN + offset) -> surface reflectance [0,1]; masked pixels become NaN.
        da = (ds[b].where(~bad).astype("float32") + off_da) / 10000.0
        scaled[b] = da.median(dim="time", skipna=True)

    comp = xr.concat([scaled[b] for b in refl_bands], dim="band")
    comp = comp.assign_coords(band=refl_bands)
    return comp


def add_indices(s2: "object", cfg: dict):
    """Compute the configured spectral indices from the S2 reflectance composite."""
    import xarray as xr

    def band(name):
        # drop the scalar 'band' coord so index arithmetic concats cleanly
        return s2.sel(band=name).drop_vars("band")

    # Floor reflectance at 0 so the baseline-offset correction (which can drive dark
    # pixels slightly negative) cannot blow indices up via near-zero denominators.
    b = {n: band(n).clip(min=0.0) for n in cfg["imagery"]["s2_bands"]}
    eps = 1e-6
    defs = {
        "NDVI": (b["B08"] - b["B04"]) / (b["B08"] + b["B04"] + eps),
        "EVI": 2.5 * (b["B08"] - b["B04"]) / (b["B08"] + 6 * b["B04"] - 7.5 * b["B02"] + 1 + eps),
        "SAVI": 1.5 * (b["B08"] - b["B04"]) / (b["B08"] + b["B04"] + 0.5 + eps),
        "NDWI": (b["B03"] - b["B08"]) / (b["B03"] + b["B08"] + eps),
        "NDRE": (b["B08"] - b["B05"]) / (b["B08"] + b["B05"] + eps),
        "NBR": (b["B08"] - b["B12"]) / (b["B08"] + b["B12"] + eps),
    }
    # Clip to physical ranges (bounded ratios in [-1,1]; EVI/SAVI wider) — dark-pixel guard.
    clip = {"NDVI": (-1, 1), "EVI": (-1, 2.5), "SAVI": (-1.5, 1.5),
            "NDWI": (-1, 1), "NDRE": (-1, 1), "NBR": (-1, 1)}
    wanted = cfg["imagery"]["indices"]
    arr = xr.concat([defs[i].clip(*clip[i]) for i in wanted], dim="band").assign_coords(band=wanted)
    return arr


def build_s1_composite(catalog, cfg: dict, bbox, date_range):
    """Sentinel-1 RTC VV/VH seasonal median composite in dB -> xarray DataArray."""
    from odc.stac import load as odc_load
    import xarray as xr

    img = cfg["imagery"]
    items = _search(catalog, img["s1_collection"], bbox, date_range)
    # Exclude Sentinel-1C granules: some 2025 S1C RTC assets on Planetary Computer are
    # malformed (unreadable tiffs), and the 2019-2024 training data had no S1C anyway
    # (it launched late 2024), so dropping it keeps inputs consistent with the model.
    items = [it for it in items if not it.id.upper().startswith("S1C")]
    if not items:
        raise RuntimeError(f"No Sentinel-1 scenes for {date_range} over {bbox}.")

    ds = odc_load(
        items, bands=img["s1_bands"], bbox=bbox,
        crs=f"EPSG:{cfg['aoi']['utm_epsg']}", resolution=img["resolution_m"],
        chunks={"x": 1024, "y": 1024}, groupby="solar_day",
    )
    out = []
    for pol in img["s1_bands"]:
        lin = ds[pol].where(ds[pol] > 0)
        db = 10.0 * np.log10(lin)
        out.append(db.median(dim="time", skipna=True))
    comp = xr.concat(out, dim="band").assign_coords(band=[p.upper() for p in img["s1_bands"]])
    return comp


def band_order(cfg: dict) -> list[str]:
    img = cfg["imagery"]
    return list(img["s2_bands"]) + list(img["indices"]) + [p.upper() for p in img["s1_bands"]]


def build_stack(catalog, cfg: dict, bbox, date_range):
    """Build the eager 18-band composite stack (S2 reflectance + indices + S1 dB)."""
    import xarray as xr

    s2 = build_s2_composite(catalog, cfg, bbox, date_range)
    idxs = add_indices(s2, cfg)
    s1 = build_s1_composite(catalog, cfg, bbox, date_range)
    stack = xr.concat([s2, idxs, s1], dim="band").assign_coords(band=band_order(cfg))
    return stack.compute()


def _subtile_bboxes(bbox, step_deg: float):
    """Grid a lon/lat bbox into sub-bboxes of ~step_deg (with tiny overlap)."""
    import numpy as np

    min_lon, min_lat, max_lon, max_lat = bbox
    lons = list(np.arange(min_lon, max_lon, step_deg)) + [max_lon]
    lats = list(np.arange(min_lat, max_lat, step_deg)) + [max_lat]
    ov = step_deg * 0.02
    out = []
    for i in range(len(lons) - 1):
        for j in range(len(lats) - 1):
            out.append((round(lons[i] - ov, 4), round(lats[j] - ov, 4),
                        round(lons[i + 1] + ov, 4), round(lats[j + 1] + ov, 4)))
    return out


def export_full(cfg: dict) -> Path:
    """Full-AOI annual composite via TILED export + mosaic (robust to URL expiry).

    Each sub-tile gets a FRESH signed catalog and is computed eagerly (low memory),
    so no single read outlives its signed URL and RAM stays bounded. Sub-tiles are
    then mosaicked into one GeoTIFF for the downstream pipeline.
    """
    import rasterio
    import rioxarray  # noqa: F401
    from rasterio.merge import merge as rio_merge

    ensure_dirs(cfg)
    out_dir = Path(cfg["paths"]["imagery_dir"])
    names = band_order(cfg)
    date_range = (f"{cfg['year']}-01-01", f"{cfg['year']}-12-31")
    subs = _subtile_bboxes(cfg["aoi"]["bbox"], cfg["imagery"]["full_subtile_deg"])
    print(f"[stac] FULL tiled export: {len(subs)} sub-tiles @ {cfg['imagery']['resolution_m']}m, "
          f"annual {date_range}", flush=True)

    sub_paths = []
    for k, sb in enumerate(subs):
        catalog = get_catalog(cfg)  # fresh sign per sub-tile
        try:
            stack = build_stack(catalog, cfg, list(sb), date_range)
        except RuntimeError as e:
            print(f"[stac]  sub {k+1}/{len(subs)} {sb} SKIPPED ({e})", flush=True)
            continue
        sp = out_dir / f"_sub_{cfg['aoi']['region']}_{cfg['year']}_{k:02d}.tif"
        stack.rio.to_raster(sp, driver="GTiff", compress="deflate")
        sub_paths.append(sp)
        print(f"[stac]  sub {k+1}/{len(subs)} {sb} -> {sp.name} shape={tuple(stack.shape)}", flush=True)

    if not sub_paths:
        raise RuntimeError("No sub-tiles exported.")

    print(f"[stac] mosaicking {len(sub_paths)} sub-tiles ...", flush=True)
    srcs = [rasterio.open(p) for p in sub_paths]
    mosaic, transform = rio_merge(srcs)
    profile = srcs[0].profile.copy()
    profile.update(height=mosaic.shape[1], width=mosaic.shape[2], transform=transform,
                   count=mosaic.shape[0], compress="deflate")
    out_path = out_dir / f"{cfg['aoi']['region']}_{cfg['year']}_annual_full.tif"
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(mosaic)
    for s in srcs:
        s.close()
    for p in sub_paths:  # tidy intermediates
        p.unlink()
    (out_dir / f"{out_path.stem}.bands.txt").write_text("\n".join(names))
    print(f"[stac] wrote {out_path}  bands={mosaic.shape[0]}  shape={tuple(mosaic.shape)}", flush=True)
    return out_path


def export(cfg: dict, quick: bool = False, season_idx: int | None = None,
           full: bool = False) -> Path:
    """Build a stacked composite GeoTIFF. Returns the output path.

    Modes: full -> export_full() (tiled full-AOI annual mosaic); otherwise a
    single eager seasonal composite over quick_bbox (quick) or aoi.bbox.
    """
    if full:
        return export_full(cfg)

    import rioxarray  # noqa: F401

    img = cfg["imagery"]
    bbox = cfg["aoi"]["quick_bbox"] if quick else cfg["aoi"]["bbox"]
    seasons = seasonal_ranges(cfg["year"], img["n_seasonal_composites"])
    idx = 0 if (quick and season_idx is None) else (season_idx or 0)
    date_range = seasons[idx]
    tag = "quick" if quick else "season"

    print(f"[stac] backend={img['backend']} mode={tag} bbox={bbox} range={date_range}", flush=True)
    ensure_dirs(cfg)
    catalog = get_catalog(cfg)
    stack = build_stack(catalog, cfg, bbox, date_range)

    out_dir = Path(cfg["paths"]["imagery_dir"])
    out_path = out_dir / f"{cfg['aoi']['region']}_{cfg['year']}_{idx}_{tag}.tif"
    stack.rio.to_raster(out_path, driver="GTiff", compress="deflate")
    (out_dir / f"{out_path.stem}.bands.txt").write_text("\n".join(band_order(cfg)))
    print(f"[stac] wrote {out_path}  bands={stack.shape[0]}  shape={tuple(stack.shape)}", flush=True)
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Export S2+indices+S1 composite via Planetary Computer (P1a).")
    ap.add_argument("--config", default=None)
    ap.add_argument("--quick", action="store_true", help="small quick_bbox, one season (smoke test)")
    ap.add_argument("--season", type=int, default=None, help="season index (0..n-1)")
    ap.add_argument("--full", action="store_true",
                    help="full aoi.bbox, annual median composite, memory-safe lazy write")
    ap.add_argument("--year", type=int, default=None, help="override target year (v2 Part C/D)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.year:
        cfg["year"] = args.year
    export(cfg, quick=args.quick, season_idx=args.season, full=args.full)
