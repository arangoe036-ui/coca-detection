"""P1b — Pull Colombia's official coca data and rasterize a weak label mask.

Implements plan §6.2. Resolved §13 Q4: the ODC ArcGIS hosts are dead, but the
same data is published on Socrata (datos.gov.co):

  * Training labels — resource `v3rx-q7t3`: a ~1 km coca-DENSITY grid (WGS84
    GeoJSON) with per-year hectares (field ``areacoca<year>``). Rasterized onto
    the exact imagery grid -> continuous density or binary presence mask.
  * Validation table — resource `acs4-3wgp`: municipal coca hectares by DANE code
    (``codmpio``), one column per year (``_<year>``). Used for the P3 area check.

Both are public (no account). The grid is finer than municipal aggregation, which
is ideal for the segmentation task.

    python -m src.data.labels --reference data/imagery/catatumbo_2023_s0_quick.tif
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.utils import ensure_dirs, load_config


def _year_field(cfg: dict) -> str:
    f = cfg["labels"]["coca_grid_year_field"]
    return f or f"areacoca{cfg['year']}"


def _val_field(cfg: dict) -> str:
    f = cfg["labels"]["validation_year_field"]
    return f or f"_{cfg['year']}"


def fetch_coca_grid(cfg: dict, bbox_lonlat: tuple[float, float, float, float]):
    """Fetch coca-density grid cells (hectares > 0) over a lon/lat bbox as a GeoDataFrame.

    ``bbox_lonlat`` = (min_lon, min_lat, max_lon, max_lat). Socrata's
    ``within_box`` takes (lat_max, lon_min, lat_min, lon_max).
    """
    from urllib.parse import urlencode

    import geopandas as gpd

    lab = cfg["labels"]
    yf = _year_field(cfg)
    min_lon, min_lat, max_lon, max_lat = bbox_lonlat
    where = (
        f"within_box({lab['coca_grid_geom_field']},{max_lat},{min_lon},{min_lat},{max_lon})"
        f" AND {yf}>0"
    )
    params = urlencode({
        "$select": f"{lab['coca_grid_id_field']},{yf},{lab['coca_grid_geom_field']}",
        "$where": where,
        "$limit": 50000,
    })
    url = f"{lab['socrata_domain']}/resource/{lab['coca_grid_resource_id']}.geojson?{params}"
    gdf = gpd.read_file(url)
    gdf = gdf.rename(columns={yf: "coca_ha"})
    gdf["coca_ha"] = gdf["coca_ha"].astype("float32")
    if gdf.crs is None:
        gdf = gdf.set_crs(4326)
    print(f"[labels] coca grid cells fetched: {len(gdf)}  total_ha={gdf['coca_ha'].sum():.1f}")
    return gdf


def fetch_validation_table(cfg: dict, departamento: str = "NORTE DE SANTANDER"):
    """Fetch municipal coca hectares (DANE-coded) for the target year -> DataFrame."""
    import pandas as pd

    lab = cfg["labels"]
    vf = _val_field(cfg)
    url = (
        f"{lab['socrata_domain']}/resource/{lab['validation_resource_id']}.json"
        f"?$select=codmpio,municipio,{vf}&departamento={departamento.replace(' ', '%20')}"
    )
    df = pd.read_json(url)
    df = df.rename(columns={vf: "coca_ha"})
    print(f"[labels] validation rows ({departamento}): {len(df)}  total_ha={df['coca_ha'].sum():.1f}")
    return df


def rasterize_mask(cfg: dict, gdf, reference_raster: str) -> Path:
    """Rasterize the coca grid onto the EXACT grid of ``reference_raster``.

    Reprojects grid polygons to the imagery CRS and burns either continuous
    hectares (``mask_type: continuous``) or binary presence. Writes a 1-band
    GeoTIFF aligned pixel-for-pixel with the imagery.

    Acceptance (P1b): mask aligns with imagery (same CRS/transform/shape); the
    positive-pixel fraction is reported for a quick QA in 02_inspect_tiles.ipynb.
    """
    import numpy as np
    import rasterio
    from rasterio.features import rasterize

    lab = cfg["labels"]
    with rasterio.open(reference_raster) as ref:
        transform, width, height, crs = ref.transform, ref.width, ref.height, ref.crs

    g = gdf.to_crs(crs)
    if lab["mask_type"] == "binary":
        shapes = [(geom, 1) for geom, ha in zip(g.geometry, g["coca_ha"])
                  if ha >= lab["presence_threshold_ha"]]
        dtype, fill = "uint8", 0
    elif lab["mask_type"] == "fraction":
        # coca FRACTION of each cell = coca_ha / cell_area_ha, clipped to [0,1].
        # Burning this makes sum(fraction * pixel_area) == coca hectares (density regression).
        shapes = []
        for geom, ha in zip(g.geometry, g["coca_ha"]):
            cell_ha = geom.area / 1e4  # UTM CRS -> m^2 -> ha
            frac = min(max(float(ha) / cell_ha, 0.0), 1.0) if cell_ha > 0 else 0.0
            shapes.append((geom, frac))
        dtype, fill = "float32", 0.0
    else:  # continuous density (hectares per cell)
        shapes = [(geom, float(ha)) for geom, ha in zip(g.geometry, g["coca_ha"])]
        dtype, fill = "float32", 0.0

    mask = rasterize(
        shapes, out_shape=(height, width), transform=transform,
        fill=fill, dtype=dtype, all_touched=False,
    )
    pos_frac = float((mask > 0).mean())
    if lab["mask_type"] == "fraction":
        px_area_ha = (cfg["imagery"]["resolution_m"] ** 2) / 1e4
        implied_ha = float(mask.sum()) * px_area_ha
        print(f"[labels] fraction mask: mean_frac(where>0)={mask[mask>0].mean():.3f} "
              f"implied_total_ha={implied_ha:,.0f} (should ≈ official grid-sum)")

    out_dir = Path(cfg["paths"]["labels_dir"])
    ensure_dirs(cfg)
    out_path = out_dir / (Path(reference_raster).stem + "_cocamask.tif")
    with rasterio.open(
        out_path, "w", driver="GTiff", height=height, width=width, count=1,
        dtype=dtype, crs=crs, transform=transform, compress="deflate",
    ) as dst:
        dst.write(mask, 1)
    print(f"[labels] wrote {out_path}  type={lab['mask_type']}  positive_pixels={100*pos_frac:.2f}%")
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Fetch official coca data + rasterize mask (P1b).")
    ap.add_argument("--config", default=None)
    ap.add_argument("--reference", required=True, help="imagery GeoTIFF to align the mask to")
    ap.add_argument("--quick", action="store_true", help="use aoi.quick_bbox for the grid fetch")
    args = ap.parse_args()
    cfg = load_config(args.config)
    bbox = cfg["aoi"]["quick_bbox"] if args.quick else cfg["aoi"]["bbox"]
    gdf = fetch_coca_grid(cfg, tuple(bbox))
    fetch_validation_table(cfg)
    rasterize_mask(cfg, gdf, args.reference)
