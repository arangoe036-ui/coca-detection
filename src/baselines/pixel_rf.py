"""Baseline B — context-free per-pixel random forest (plan Phase 3).

The baseline that might actually win. Features = the 18 channels PER PIXEL, no
spatial context — this is what isolates "does spatial structure help?" Inputs are
the SAME per-year-normalized 18 channels the U-Net consumes (fairness). Trained on
a stratified pixel subsample (positive pixels not swamped), fixed RNG seed. Density
is predicted per pixel, then run through the SAME fit_scalar -> ratio path.

    python -m src.baselines.pixel_rf          # Track B (6 LOYO folds)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from src.baselines import common as C
from src.metrics_io import metrics_dir, write_run
from src.utils import load_config

SEED = 42
N_PIXELS = 3_000_000
BAND_NAMES = ["B02", "B03", "B04", "B08", "B05", "B06", "B07", "B8A", "B11", "B12",
              "NDVI", "EVI", "SAVI", "NDWI", "NDRE", "NBR", "VV", "VH"]
RF_KWARGS = dict(n_estimators=100, max_depth=16, min_samples_leaf=20,
                 max_features="sqrt", n_jobs=-1, random_state=SEED)


class PixelRF:
    def __init__(self, rf):
        self.rf = rf

    def predict_pixels(self, feat):
        return self.rf.predict(feat)

    @classmethod
    def fit(cls, cfg, train_rows, stats, seed=SEED, n_pixels=N_PIXELS):
        X, y, npos, nneg = C.sample_train_pixels(cfg, train_rows, stats, n_pixels,
                                                 seed, normalize_inputs=True)
        rf = RandomForestRegressor(**{**RF_KWARGS, "random_state": seed})
        rf.fit(X, y)
        info = {"n_pixels": int(len(y)), "n_pos_px": int(npos), "n_neg_px": int(nneg),
                "seed": int(seed), **RF_KWARGS}
        return cls(rf), info


def _save_importances(cfg, tag, importances):
    d = metrics_dir(cfg) / "rf_importances"
    d.mkdir(parents=True, exist_ok=True)
    ranked = sorted(zip(BAND_NAMES, [float(i) for i in importances]),
                    key=lambda kv: -kv[1])
    (d / f"{tag}.json").write_text(json.dumps(dict(ranked), indent=2))
    return ranked[:4]


def run_track_b(cfg, years=None):
    years = years or C.YEARS
    rows = C.load_index(cfg)
    stats = C.all_stats(cfg, rows)
    print(f"[rf] Baseline B — pixel RF, Track B (6 LOYO folds), seed={SEED}")
    for test_year in years:
        train_years = [y for y in years if y != test_year]
        assert test_year not in train_years, "held-out year leaked into fit set"
        tr = C.rows_for(rows, train_years, "train")
        mdl, info = PixelRF.fit(cfg, tr, stats, seed=SEED)
        top = _save_importances(cfg, f"loyo_{test_year}", mdl.rf.feature_importances_)
        s = C.fit_scalar_generic(cfg, tr, stats, mdl.predict_pixels, gate=True,
                                 normalize_inputs=True)
        p_ha, off, ratio = C.full_raster_ratio(cfg, test_year, stats, mdl.predict_pixels,
                                               s, gate=True, normalize_inputs=True)
        write_run(cfg, "random_forest", test_year, {"aoi_ratio": ratio}, track="B",
                  n_train_tiles=len(tr), n_test_tiles=0, calibration_scalar=s,
                  fit_years=train_years,
                  extra={**info, "pred_ha": p_ha, "official_ha": off,
                         "top_features": top,
                         "leakage_assert": f"held-out {test_year} absent from fit years {train_years}"})
        print(f"  {test_year}: s={s:.2f} ratio={ratio:.3f}  top={[f'{n}:{v:.2f}' for n,v in top]}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Baseline B: pixel random forest.")
    ap.add_argument("--config", default=None)
    run_track_b(load_config(ap.parse_args().config))
