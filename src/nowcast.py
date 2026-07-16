"""v2.2 Part B — exploratory nowcast for a censusless year (2026).

Loads the final all-years model, normalizes the target year by ITS OWN per-channel
stats, applies the FROZEN train-years calibration scalar, gates, and produces a
density raster + downsampled COG + municipal aggregation. Because no official
census exists, EVERY hectares figure is reported as a range using the LOYO band
(BAND=0.40, i.e. ±40%), and a change-vs-reference-year direction is computed
(more reliable than either absolute level).

    python -m src.nowcast --year 2026 --reference 2024

Outputs: ui/data/density_<year>_cog.tif, ui/data/municipal_coca_<year>.geojson/.csv.
DO NOT present any figure here as validated. Lead with location, not magnitude.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np
import rasterio
import torch

from src.infer import municipal_hectares, predict_raster
from src.models.unet import build_unet
from src.train import pick_device
from src.utils import ensure_dirs, load_config

BAND = 0.40  # LOYO out-of-year magnitude band (±40%)


def year_stats_from_raster(image_path: str, factor: int = 8):
    """Per-channel mean/std from a decimated read (input normalization only)."""
    with rasterio.open(image_path) as ds:
        arr = ds.read(out_shape=(ds.count, ds.height // factor, ds.width // factor)).astype("float64")
    c = arr.shape[0]
    flat = arr.reshape(c, -1)
    mean = np.zeros(c, "float32"); std = np.ones(c, "float32")
    for i in range(c):
        v = flat[i][np.isfinite(flat[i])]
        if v.size:
            mean[i] = v.mean(); std[i] = max(v.std(), 1e-6)
    return mean, std


def write_cog(density: np.ndarray, profile, out_path: Path, long_side: int = 1500):
    from affine import Affine
    from rasterio.enums import Resampling
    from rio_cogeo.cogeo import cog_translate
    from rio_cogeo.profiles import cog_profiles

    h, w = density.shape
    factor = max(h, w) / long_side
    nh, nw = int(h / factor), int(w / factor)
    with rasterio.open(tempfile.mktemp(suffix=".tif"), "w", driver="GTiff", height=h, width=w,
                       count=1, dtype="float32", crs=profile["crs"], transform=profile["transform"],
                       nodata=0) as src:
        src.write(density.astype("float32"), 1)
        full = src.name
    with rasterio.open(full) as ds:
        small = ds.read(1, out_shape=(nh, nw), resampling=Resampling.average)
        tr = ds.transform * Affine.scale(w / nw, h / nh)
    tmp = tempfile.mktemp(suffix=".tif")
    with rasterio.open(tmp, "w", driver="GTiff", height=nh, width=nw, count=1, dtype="float32",
                       crs=profile["crs"], transform=tr, nodata=0) as d:
        d.write(small.astype("float32"), 1)
    cog_translate(tmp, out_path, cog_profiles.get("deflate"), quiet=True)


def nowcast(cfg, year: int, reference: int):
    device = pick_device()
    ensure_dirs(cfg)
    region = cfg["aoi"]["region"]
    img_path = f"data/imagery/{region}_{year}_annual_full.tif"
    if not Path(img_path).exists():
        raise FileNotFoundError(f"Missing {year} imagery: {img_path}")

    ck = torch.load(Path(cfg["paths"]["checkpoints_dir"]) / "final_multiyear.pt",
                    map_location=device, weights_only=False)
    model = build_unet(cfg).to(device)
    model.load_state_dict(ck["model"])
    scalar, tau = ck["scalar"], ck["tau"]

    mean, std = year_stats_from_raster(img_path)   # 2026 normalized by its own stats
    dens, profile = predict_raster(cfg, img_path, model, mean, std, device)
    dens = dens * (dens >= tau)
    px_ha = (cfg["imagery"]["resolution_m"] ** 2) / 1e4
    est_ha = float(dens.sum()) * px_ha * scalar

    # density raster + COG for the UI
    out_dir = Path(cfg["paths"]["outputs_dir"])
    dens_path = out_dir / f"{region}_{year}_coca_density.tif"
    pp = profile.copy(); pp.update(count=1, dtype="float32", compress="deflate")
    with rasterio.open(dens_path, "w", **pp) as d:
        d.write((dens * scalar).astype("float32"), 1)
    ui_dir = Path(cfg["paths"]["ui_data_dir"]); ui_dir.mkdir(parents=True, exist_ok=True)
    write_cog(dens * scalar, profile, ui_dir / f"density_{year}_cog.tif")

    # municipal aggregation (official_ha will be NaN — no census); reproject to WGS84 inside.
    per = municipal_hectares(cfg, dens * scalar, profile)
    per["nowcast"] = True

    # change vs reference year (directional — more reliable than absolute level)
    ref_geo = ui_dir / f"municipal_coca_{reference}.geojson"
    if ref_geo.exists():
        import geopandas as gpd
        ref = gpd.read_file(ref_geo)[["name", "predicted_ha"]].rename(columns={"predicted_ha": "ref_ha"})
        per = per.merge(ref, on="name", how="left")
        def direction(r):
            if not r.get("ref_ha") or np.isnan(r["ref_ha"]):
                return "new"
            ch = (r["predicted_ha"] - r["ref_ha"]) / r["ref_ha"]
            return "up" if ch > 0.15 else "down" if ch < -0.15 else "stable"
        per["change_dir"] = per.apply(direction, axis=1)
        per["ref_year"] = reference
    per.to_file(ui_dir / f"municipal_coca_{year}.geojson", driver="GeoJSON")
    per.drop(columns="geometry").to_csv(ui_dir / f"municipal_coca_{year}.csv", index=False)

    lo, hi = est_ha * (1 - BAND), est_ha * (1 + BAND)
    import datetime
    kind = "PARTIAL-YEAR" if year >= datetime.date.today().year else "FULL-YEAR"
    print(f"\n[nowcast] {year} ({kind}, EXPLORATORY — no official census):")
    print(f"[nowcast]   estimated coca ≈ {est_ha:,.0f} ha  (±{int(BAND*100)}% band: {lo:,.0f}–{hi:,.0f} ha)")
    print(f"[nowcast]   *** not a validated figure; model under-reads growth years ***")
    top = per.sort_values("predicted_ha", ascending=False).head(6)
    print(f"[nowcast]   top municipalities (est ha · change vs {reference}):")
    for _, r in top.iterrows():
        arrow = {"up": "↑", "down": "↓", "stable": "→", "new": "＋"}.get(r.get("change_dir", ""), "")
        print(f"            {r['municipio']:<16} ≈{r['predicted_ha']:>7,.0f} ha  {arrow}")
    return est_ha


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Exploratory nowcast for a censusless year (v2.2 B).")
    ap.add_argument("--config", default=None)
    ap.add_argument("--year", type=int, default=2026)
    ap.add_argument("--reference", type=int, default=2024)
    args = ap.parse_args()
    nowcast(load_config(args.config), args.year, args.reference)
