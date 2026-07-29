"""Shared plumbing for the baseline ladder — folds, normalization, calibration,
full-raster scoring, and metric assembly.

Everything here delegates the *decisions* to existing code:
- folds / index / per-year norm stats / gating: ``src.train_loyo``
- density-regression metrics: ``src.evaluate._evaluate_regression`` (no new math)
- calibration: a faithful generalization of ``src.train_loyo.fit_scalar`` to any
  per-pixel ``predict`` callable (same TAU gate, same num/den ratio, train-years
  only). This is a generalization, not a second implementation of the split.

NDVI (idx 10), VV (16), VH (17) — confirmed empirically in Phase 0.1.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio

from src.evaluate import _evaluate_regression
from src.train_loyo import TAU, read_index, year_norm_stats

YEARS = [2019, 2020, 2021, 2022, 2023, 2024]
NDVI_IDX = 10

# Frozen official held-out hectares (docs/v2.1_loyo_results.md) — the same
# denominators the U-Net's aoi_ratio used. Kept here so Track B needs no network.
OFFICIAL_HA = {2019: 36296.0, 2020: 35277.0, 2021: 38285.0,
               2022: 37965.0, 2023: 39815.0, 2024: 44240.0}


# ----------------------------------------------------------------------------- folds
def load_index(cfg) -> list[dict]:
    return read_index(cfg)


def all_stats(cfg, rows) -> dict:
    """Per-(year,channel) mean/std over ALL rows — identical to train_loyo
    (inputs only, unsupervised, leakage-free)."""
    return year_norm_stats(cfg, rows)


def rows_for(rows, years, split) -> list[dict]:
    yrs = {int(y) for y in years}
    return [r for r in rows if int(r["year"]) in yrs and r["split"] == split]


def _tile(cfg, r):
    npz = np.load(Path(cfg["paths"]["tiles_dir"]) / r["npz"])
    img = np.nan_to_num(npz["image"].astype("float32"), nan=0.0, posinf=0.0, neginf=0.0)
    mask = npz["mask"].astype("float32")
    return img, mask


def _norm(img, mean, std):
    return (img - mean.reshape(-1, 1, 1)) / std.reshape(-1, 1, 1)


# ------------------------------------------------------------------- pixel sampling
def sample_train_pixels(cfg, rows, stats, n_target, seed, normalize_inputs=True):
    """Stratified per-pixel sample for RF training. Positive-density pixels are
    kept up to half of each tile's quota so they are not swamped (plan §3). Fixed
    seed, recorded by the caller. Returns (X[n,C], y[n], n_pos, n_neg)."""
    rng = np.random.default_rng(seed)
    per_tile = max(2, n_target // max(len(rows), 1))
    Xs, Ys = [], []
    npos = nneg = 0
    for r in rows:
        img, mask = _tile(cfg, r)
        if normalize_inputs:
            m, s = stats[int(r["year"])]
            img = _norm(img, m, s)
        feat = img.reshape(img.shape[0], -1).T          # (npx, C)
        tgt = mask.reshape(-1)
        pos = np.flatnonzero(tgt > 0)
        neg = np.flatnonzero(tgt == 0)
        k_pos = min(len(pos), per_tile // 2)
        k_neg = min(len(neg), per_tile - k_pos)
        sel = np.concatenate([
            rng.choice(pos, k_pos, replace=False) if k_pos else np.empty(0, int),
            rng.choice(neg, k_neg, replace=False) if k_neg else np.empty(0, int),
        ])
        Xs.append(feat[sel]); Ys.append(tgt[sel])
        npos += k_pos; nneg += k_neg
    return np.concatenate(Xs), np.concatenate(Ys), npos, nneg


# --------------------------------------------------------------- prediction collect
def collect_pixels(cfg, rows, stats, predict_pixels, normalize_inputs=True):
    """Flatten (pred_density, target) over `rows` for a per-pixel predictor.
    predict_pixels: (npx, C) -> (npx,) density fraction."""
    ps, ts = [], []
    for r in rows:
        img, mask = _tile(cfg, r)
        if normalize_inputs:
            m, s = stats[int(r["year"])]
            img = _norm(img, m, s)
        feat = img.reshape(img.shape[0], -1).T
        ps.append(np.asarray(predict_pixels(feat), dtype="float32"))
        ts.append(mask.reshape(-1))
    return np.concatenate(ps), np.concatenate(ts)


def collect_tiles(cfg, rows, stats, predict_tile):
    """Flatten (pred_density, target) for a spatial (tile) predictor, e.g. the U-Net.
    predict_tile: normalized (C,H,W) -> (H,W) density."""
    ps, ts = [], []
    for r in rows:
        img, mask = _tile(cfg, r)
        m, s = stats[int(r["year"])]
        pred = predict_tile(_norm(img, m, s))
        ps.append(np.asarray(pred, dtype="float32").reshape(-1))
        ts.append(mask.reshape(-1))
    return np.concatenate(ps), np.concatenate(ts)


def regression_metrics(cfg, probs, targets) -> dict:
    """Density-regression metrics via the EXISTING evaluate helper (no new math).
    Returns mae, rmse, bias, presence_iou, presence_f1 (+ tile ha_ratio)."""
    m = _evaluate_regression(cfg, np.asarray(probs), np.asarray(targets))
    return {"mae": m["mae"], "rmse": m["rmse"], "bias": m["bias"],
            "presence_iou": m["presence_iou"], "presence_f1": m["presence_f1"],
            "ha_ratio_tile": m["ha_ratio_test"]}


# -------------------------------------------------------------------- calibration/B
def fit_scalar_generic(cfg, rows, stats, predict_pixels, *, gate=True,
                       normalize_inputs=True) -> float:
    """Frozen calibration scalar over TRAIN-year tiles — a faithful generalization
    of train_loyo.fit_scalar (same TAU gate, same s = sum(official)/sum(gated pred),
    train years only). `gate=False` for a spatially-uniform null (gating a constant
    is degenerate, prereg A1)."""
    num = den = 0.0
    for r in rows:
        img, mask = _tile(cfg, r)
        if normalize_inputs:
            m, s = stats[int(r["year"])]
            img = _norm(img, m, s)
        feat = img.reshape(img.shape[0], -1).T
        pred = np.asarray(predict_pixels(feat), dtype="float64")
        if gate:
            pred = pred * (pred >= TAU)
        num += float(mask.sum()); den += float(pred.sum())
    return num / den if den > 0 else 1.0


def full_raster_ratio(cfg, test_year, stats, predict_pixels, scalar, *, gate=True,
                      normalize_inputs=True, valid_only=False, chunk_rows=512):
    """Predicted/official hectares for a held-out year — the pixel-wise analogue of
    train_loyo.test_year_ratio (same gate, px_ha, scalar; official from OFFICIAL_HA).
    Reads the full annual raster in row-chunks so memory stays bounded.

    valid_only: zero predictions on nodata pixels (all input bands ~0). The U-Net /
    NDVI / RF paths leave this False and let the TAU gate remove nodata (matching
    how the frozen v2.1 U-Net ratio was computed); the uniform null sets it True,
    since gating a constant is degenerate so it must be confined to the AOI."""
    region = cfg["aoi"]["region"]
    path = f"data/imagery/{region}_{test_year}_annual_full.tif"
    m, s = stats[test_year]
    px_ha = (cfg["imagery"]["resolution_m"] ** 2) / 1e4
    total = 0.0
    with rasterio.open(path) as ds:
        H, W = ds.height, ds.width
        for y0 in range(0, H, chunk_rows):
            win = rasterio.windows.Window(0, y0, W, min(chunk_rows, H - y0))
            raw = np.nan_to_num(ds.read(window=win).astype("float32"),
                                nan=0.0, posinf=0.0, neginf=0.0)
            valid = (np.abs(raw).sum(0) > 0).reshape(-1) if valid_only else None
            arr = _norm(raw, m, s) if normalize_inputs else raw
            feat = arr.reshape(arr.shape[0], -1).T
            pred = np.asarray(predict_pixels(feat), dtype="float64")
            if gate:
                pred = pred * (pred >= TAU)
            if valid is not None:
                pred = pred * valid
            total += float(pred.sum())
    pred_ha = total * px_ha * scalar
    off = OFFICIAL_HA[test_year]
    return pred_ha, off, (pred_ha / off if off else float("nan"))
