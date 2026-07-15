"""P4 — Inference & map outputs (plan §9).

Sliding-window inference across the full-AOI imagery raster (overlaps averaged)
-> coca-probability GeoTIFF; threshold + morphological open/close + min-mapping-
unit -> binary mask; vectorize -> polygons; spatial-join to municipality
boundaries (GADM) -> hectares per municipality.

Also runs the **area sanity check** (plan §8): total predicted hectares vs the
official coca figure summed over the same AOI (from labels.fetch_coca_grid).

Exports: probability GeoTIFF, coca GeoJSON, and for the UI
``ui/data/municipal_coca.geojson`` + ``.csv``.

    python -m src.infer --image data/imagery/catatumbo_2023_annual_full.tif \
                        --checkpoint outputs/checkpoints/best.pt --threshold 0.5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from src.data.labels import fetch_coca_grid
from src.models.unet import build_unet
from src.train import pick_device
from src.utils import ensure_dirs, load_config


def hann2d(n: int) -> np.ndarray:
    """2D Hann window (near-zero at edges, 1 in the center) for feathered blending."""
    w = np.hanning(n)
    return np.outer(w, w).astype("float32") + 1e-6


@torch.no_grad()
def predict_raster(cfg: dict, image_path: str, model, mean, std, device) -> tuple:
    """Sliding-window inference -> (prob[H,W] float32, profile).

    v2 A1: overlaps are blended with a 2D Hann window (weight ~0 at tile edges,
    1 at center) instead of hard-averaged, so tile seams/grid artifacts vanish.
    """
    import rasterio

    win = cfg["infer"]["window_px"]
    ov = cfg["infer"]["window_overlap_px"]
    step = max(win - ov, 1)
    m = np.asarray(mean).reshape(-1, 1, 1)
    s = np.asarray(std).reshape(-1, 1, 1)
    weight = hann2d(win)

    with rasterio.open(image_path) as ds:
        H, W = ds.height, ds.width
        profile = ds.profile
        acc = np.zeros((H, W), dtype="float32")
        wsum = np.zeros((H, W), dtype="float32")
        model.eval()
        xs = list(range(0, max(W - win, 0) + 1, step)) + ([W - win] if W > win else [])
        ys = list(range(0, max(H - win, 0) + 1, step)) + ([H - win] if H > win else [])
        for y in sorted(set(ys)):
            for x in sorted(set(xs)):
                w = rasterio.windows.Window(x, y, min(win, W - x), min(win, H - y))
                arr = ds.read(window=w).astype("float32")
                arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
                arr = (arr - m) / s
                ph, pw = win - arr.shape[1], win - arr.shape[2]
                if ph or pw:
                    arr = np.pad(arr, ((0, 0), (0, ph), (0, pw)))
                t = torch.from_numpy(arr[None]).to(device)
                prob = torch.sigmoid(model(t))[0, 0].cpu().numpy()
                hh, ww = w.height, w.width
                acc[y:y + hh, x:x + ww] += prob[:hh, :ww] * weight[:hh, :ww]
                wsum[y:y + hh, x:x + ww] += weight[:hh, :ww]
        prob = acc / np.maximum(wsum, 1e-6)
    return prob, profile


def postprocess(cfg: dict, prob: np.ndarray, threshold: float) -> np.ndarray:
    """Threshold + morphological open/close + min-mapping-unit -> binary mask."""
    from scipy import ndimage

    k = cfg["infer"]["morph_kernel_px"]
    mask = (prob > threshold).astype("uint8")
    struct = np.ones((k, k), dtype=bool)
    mask = ndimage.binary_opening(mask, structure=struct)
    mask = ndimage.binary_closing(mask, structure=struct).astype("uint8")
    # min-mapping-unit: drop connected components below min area
    px_area_ha = (cfg["imagery"]["resolution_m"] ** 2) / 10000.0
    min_px = int(cfg["infer"]["min_polygon_ha"] / px_area_ha)
    lab, n = ndimage.label(mask)
    if n:
        sizes = ndimage.sum(np.ones_like(lab), lab, index=range(1, n + 1))
        small = {i + 1 for i, sz in enumerate(sizes) if sz < min_px}
        if small:
            mask[np.isin(lab, list(small))] = 0
    return mask


def vectorize_and_join(cfg: dict, mask: np.ndarray, profile) -> tuple:
    """Vectorize mask -> polygons; spatial-join to GADM municipalities -> per-mpio hectares."""
    import geopandas as gpd
    import rasterio.features
    from shapely.geometry import shape

    transform, crs = profile["transform"], profile["crs"]
    geoms = [shape(g) for g, v in rasterio.features.shapes(mask, mask=mask > 0, transform=transform) if v == 1]
    coca = gpd.GeoDataFrame(geometry=geoms, crs=crs)
    coca["area_ha"] = coca.geometry.area / 10000.0

    muni = gpd.read_file(cfg["labels"]["boundaries_url"]).to_crs(crs)
    muni = muni.rename(columns={"NAME_2": "municipio", "NAME_1": "departamento", "GID_2": "gid"})
    joined = gpd.overlay(coca, muni[["municipio", "departamento", "gid", "geometry"]], how="intersection")
    joined["area_ha"] = joined.geometry.area / 10000.0
    per_mpio = joined.groupby(["gid", "municipio", "departamento"], as_index=False)["area_ha"].sum()
    return coca, per_mpio, muni


def area_sanity_check(cfg: dict, predicted_ha: float) -> float:
    """Compare total predicted hectares to the official grid-sum over the AOI (plan §8)."""
    gdf = fetch_coca_grid(cfg, tuple(cfg["aoi"]["bbox"]))
    official = float(gdf["coca_ha"].sum())
    ratio = predicted_ha / official if official else float("nan")
    print(f"[infer] AREA SANITY: predicted={predicted_ha:,.0f} ha  "
          f"official(grid {cfg['year']})={official:,.0f} ha  ratio={ratio:.2f} "
          f"(expect a ballpark match, not exact)")
    return official


def _norm_name(s: str) -> str:
    """Normalize a municipality name for joining: strip accents/spaces, uppercase."""
    import unicodedata
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return "".join(ch for ch in s if ch.isalnum()).upper()


def official_municipal_ha(cfg: dict) -> dict:
    """Official coca hectares by municipality for the target year (Socrata acs4-3wgp).

    Returns {normalized_municipio_name: hectares}. Used to enrich predictions with
    the ground-truth figure + ratio for the UI.
    """
    import pandas as pd

    lab = cfg["labels"]
    yf = f"_{cfg['year']}"
    url = (f"{lab['socrata_domain']}/resource/{lab['validation_resource_id']}.json"
           f"?$select=municipio,{yf}&$limit=5000")
    df = pd.read_json(url)
    df = df.dropna(subset=[yf])
    return {_norm_name(m): float(h) for m, h in zip(df["municipio"], df[yf])}


def municipal_hectares(cfg: dict, density: np.ndarray, profile):
    """Zonal sum of predicted density (fraction) per municipality -> hectares,
    enriched with the official figure + ratio, and UI-ready property names.

    Rasterizes GADM municipalities onto the prediction grid and sums
    fraction*pixel_area within each. Returns a GeoDataFrame with columns
    name / municipio / departamento / year / predicted_ha / official_ha / ratio.
    """
    import geopandas as gpd
    import rasterio.features

    crs = profile["crs"]
    px_area_ha = (cfg["imagery"]["resolution_m"] ** 2) / 1e4
    minx, miny, maxx, maxy = cfg["aoi"]["bbox"]

    muni = gpd.read_file(cfg["labels"]["boundaries_url"]).to_crs(crs)
    muni = muni.rename(columns={"NAME_2": "municipio", "NAME_1": "departamento", "GID_2": "gid"})
    aoi = gpd.GeoDataFrame(geometry=gpd.GeoSeries.from_wkt(
        [f"POLYGON(({minx} {miny},{maxx} {miny},{maxx} {maxy},{minx} {maxy},{minx} {miny}))"]
    ), crs=4326).to_crs(crs)
    muni = muni[muni.intersects(aoi.union_all())].reset_index(drop=True)

    ids = rasterio.features.rasterize(
        ((geom, i + 1) for i, geom in enumerate(muni.geometry)),
        out_shape=density.shape, transform=profile["transform"], fill=0, dtype="int32",
    )
    sums = np.bincount(ids.ravel(), weights=density.ravel(), minlength=len(muni) + 1)
    muni["predicted_ha"] = sums[1:] * px_area_ha
    muni = muni[muni["predicted_ha"] > 1.0].reset_index(drop=True)

    off = official_municipal_ha(cfg)
    muni["official_ha"] = [off.get(_norm_name(m), float("nan")) for m in muni["municipio"]]
    muni["ratio"] = muni["predicted_ha"] / muni["official_ha"]
    muni["name"] = muni["municipio"]
    muni["year"] = cfg["year"]
    return muni[["gid", "name", "municipio", "departamento", "year",
                 "predicted_ha", "official_ha", "ratio", "geometry"]]


def gate_and_calibrate(cfg, density, fit_official_ha=None):
    """v2 A2: gate the density to kill the background haze, then rescale by a
    calibration constant k so the AOI total matches the official figure.

    ``fit_official_ha`` given  -> FIT k = official / gated_total and save it.
    ``fit_official_ha`` None   -> LOAD the saved k (reuse across years).
    Returns (gated_density, stats).
    """
    import json

    px_area_ha = (cfg["imagery"]["resolution_m"] ** 2) / 1e4
    tau = cfg["infer"]["gate_threshold"]
    calib_path = Path(cfg["paths"]["outputs_dir"]) / Path(cfg["infer"]["calibration_path"]).name

    raw_total = float(density.sum()) * px_area_ha
    raw_nonzero = float((density > 0).mean())
    gated = density * (density >= tau)
    gated_total = float(gated.sum()) * px_area_ha
    gated_nonzero = float((gated > 0).mean())

    if fit_official_ha is not None:
        k = fit_official_ha / gated_total if gated_total > 0 else 1.0
        calib_path.write_text(json.dumps(
            {"gate_threshold": tau, "k": k, "fit_year": cfg["year"],
             "fit_official_ha": fit_official_ha, "gated_total_prescale_ha": gated_total}, indent=2))
        print(f"[infer] A2 FIT calibration k={k:.3f} (saved -> {calib_path.name})")
    else:
        if not calib_path.exists():
            raise FileNotFoundError(f"No calibration at {calib_path}; run a fit year (2023) with --fit-calibration first.")
        cal = json.loads(calib_path.read_text())
        k = cal["k"]
        print(f"[infer] A2 LOADED calibration k={k:.3f} (fit on {cal['fit_year']})")

    gated *= k
    final_total = float(gated.sum()) * px_area_ha
    stats = {"raw_total_ha": raw_total, "raw_nonzero_frac": raw_nonzero,
             "gated_nonzero_frac": gated_nonzero, "k": k, "final_total_ha": final_total}
    print(f"[infer] A2 gate@{tau}: nonzero {100*raw_nonzero:.0f}% -> {100*gated_nonzero:.0f}%  "
          f"total {raw_total:,.0f} -> {final_total:,.0f} ha (k={k:.2f})")
    return gated, stats


def _infer_regression(cfg, prob, profile, out_dir, fit_calibration=False) -> None:
    """Density-regression outputs: gated + calibrated hectares + municipal choropleth + footprint."""
    import geopandas as gpd
    import rasterio.features
    from shapely.geometry import shape

    px_area_ha = (cfg["imagery"]["resolution_m"] ** 2) / 1e4

    # v2 A2: gate haze + calibrate. Fit k on a year with official data; else reuse.
    official = None
    if fit_calibration:
        official = float(fetch_coca_grid(cfg, tuple(cfg["aoi"]["bbox"]))["coca_ha"].sum())
    prob, gate_stats = gate_and_calibrate(cfg, prob, fit_official_ha=official)
    predicted_ha = float(prob.sum()) * px_area_ha

    # overwrite the density raster with the gated+calibrated version
    dens_path = out_dir / f"{cfg['aoi']['region']}_{cfg['year']}_coca_density.tif"
    pp = profile.copy(); pp.update(count=1, dtype="float32", compress="deflate")
    import rasterio
    with rasterio.open(dens_path, "w", **pp) as dst:
        dst.write(prob.astype("float32"), 1)

    per_mpio = municipal_hectares(cfg, prob, profile)
    ui_dir = Path(cfg["paths"]["ui_data_dir"]); ui_dir.mkdir(parents=True, exist_ok=True)
    # Year-suffixed files let the UI's year selector enumerate available years (spec §4/§8).
    for stem in ("municipal_coca", f"municipal_coca_{cfg['year']}"):
        per_mpio.to_file(ui_dir / f"{stem}.geojson", driver="GeoJSON")
        per_mpio.drop(columns="geometry").to_csv(ui_dir / f"{stem}.csv", index=False)

    # Presence footprint (fraction > threshold) for the map overlay.
    thr = cfg["infer"]["presence_fraction"]
    mask = (prob > thr).astype("uint8")
    geoms = [shape(g) for g, v in rasterio.features.shapes(
        mask, mask=mask > 0, transform=profile["transform"]) if v == 1]
    foot = gpd.GeoDataFrame(geometry=geoms, crs=profile["crs"])
    foot.to_file(out_dir / f"{cfg['aoi']['region']}_{cfg['year']}_coca.geojson", driver="GeoJSON")

    print(f"[infer] CALIBRATED predicted coca = {predicted_ha:,.0f} ha "
          f"across {len(per_mpio)} municipalities; footprint polys={len(foot)}")
    official = area_sanity_check(cfg, predicted_ha)
    top = per_mpio.sort_values("predicted_ha", ascending=False).head(5)
    print("[infer] top municipalities (predicted ha):")
    for _, r in top.iterrows():
        print(f"         {r['municipio']:<16} {r['predicted_ha']:>8,.0f} ha")


def infer(cfg: dict, image_path: str, checkpoint: str, threshold: float | None = None,
          fit_calibration: bool = False) -> None:
    import rasterio

    ensure_dirs(cfg)
    device = pick_device()
    ck = torch.load(checkpoint, map_location=device, weights_only=False)
    model = build_unet(cfg).to(device)
    model.load_state_dict(ck["model"])
    thr = threshold if threshold is not None else cfg["eval"]["threshold"]

    prob, profile = predict_raster(cfg, image_path, model, ck["mean"], ck["std"], device)
    out_dir = Path(cfg["paths"]["outputs_dir"])

    if cfg.get("model", {}).get("task") == "regression":
        # _infer_regression writes the gated+calibrated density raster itself.
        _infer_regression(cfg, prob, profile, out_dir, fit_calibration=fit_calibration)
        return

    prob_path = out_dir / f"{cfg['aoi']['region']}_{cfg['year']}_coca_prob.tif"
    pp = profile.copy()
    pp.update(count=1, dtype="float32", compress="deflate")
    with rasterio.open(prob_path, "w", **pp) as dst:
        dst.write(prob.astype("float32"), 1)
    print(f"[infer] wrote prob raster -> {prob_path}")

    mask = postprocess(cfg, prob, thr)
    coca, per_mpio, _ = vectorize_and_join(cfg, mask, profile)
    predicted_ha = float(coca["area_ha"].sum())
    coca.to_file(out_dir / f"{cfg['aoi']['region']}_{cfg['year']}_coca.geojson", driver="GeoJSON")
    ui_dir = Path(cfg["paths"]["ui_data_dir"]); ui_dir.mkdir(parents=True, exist_ok=True)
    per_mpio["year"] = cfg["year"]
    per_mpio.to_csv(ui_dir / "municipal_coca.csv", index=False)
    print(f"[infer] wrote coca polygons ({len(coca)}) + municipal CSV. total={predicted_ha:,.0f} ha")
    area_sanity_check(cfg, predicted_ha)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Inference & map outputs (P4).")
    ap.add_argument("--config", default=None)
    ap.add_argument("--image", required=True)
    ap.add_argument("--checkpoint", default="outputs/checkpoints/best.pt")
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--fit-calibration", action="store_true",
                    help="v2 A2: fit the density->hectares calibration k on this year's "
                         "official figure and save it (do this on 2023; reuse for other years).")
    args = ap.parse_args()
    infer(load_config(args.config), args.image, args.checkpoint, args.threshold,
          fit_calibration=args.fit_calibration)
