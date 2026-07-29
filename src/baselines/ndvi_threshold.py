"""Baseline A — spectral-index (NDVI) threshold (plan Phase 2).

Method (simplest defensible form, plan §2): predict presence where raw NDVI
(channel 10, confirmed Phase 0.1) exceeds a threshold t, and convert presence to a
density by multiplying by a constant c = mean train density among above-threshold
pixels. t is swept on a grid and selected by MINIMUM MAE on the train-years' VAL
blocks; c is fit on the train-years' TRAIN blocks. The held-out year is never
touched (prereg §2). Uses RAW NDVI (not per-year normalized) by design — this is
the naive vegetation-index floor.

    python -m src.baselines.ndvi_threshold          # Track B (6 LOYO folds)
"""

from __future__ import annotations

import numpy as np

from src.baselines import common as C
from src.metrics_io import write_run
from src.utils import load_config

T_GRID = np.round(np.linspace(0.0, 0.9, 19), 3)   # raw-NDVI thresholds


class NDVIThreshold:
    """density(pixel) = (NDVI > t) * c."""

    def __init__(self, t: float, c: float):
        self.t, self.c = float(t), float(c)

    def predict_pixels(self, feat):
        return (feat[:, C.NDVI_IDX] > self.t).astype("float64") * self.c

    @classmethod
    def fit(cls, cfg, train_rows, val_rows, seed=42, n_pixels=2_000_000):
        """Fit c on TRAIN pixels per t; select t by min MAE on VAL pixels. Threshold
        SELECTION uses a stratified pixel sample (fast, low-memory, seed recorded);
        the calibration scalar and full-raster ratio still use the full data."""
        Xtr, ytr, *_ = C.sample_train_pixels(cfg, train_rows, None, n_pixels, seed,
                                             normalize_inputs=False)
        Xva, yva, *_ = C.sample_train_pixels(cfg, val_rows, None, n_pixels, seed + 1,
                                             normalize_inputs=False)
        ndvi_tr, ndvi_va = Xtr[:, C.NDVI_IDX], Xva[:, C.NDVI_IDX]
        best = None
        for t in T_GRID:
            a_tr = ndvi_tr > t
            c = float(ytr[a_tr].sum() / a_tr.sum()) if a_tr.any() else 0.0
            mae = float(np.abs((ndvi_va > t).astype("float64") * c - yva).mean())
            if best is None or mae < best[0]:
                best = (mae, float(t), c)
        _, t, c = best
        return cls(t, c), {"selected_t": t, "fitted_c": c, "val_mae_at_t": best[0],
                           "select_seed": seed, "select_n_pixels": int(len(ytr))}


def run_track_b(cfg, years=None):
    years = years or C.YEARS
    rows = C.load_index(cfg)
    stats = C.all_stats(cfg, rows)
    print("[ndvi] Baseline A — NDVI threshold, Track B (6 LOYO folds)")
    for test_year in years:
        train_years = [y for y in years if y != test_year]
        assert test_year not in train_years, "held-out year leaked into fit set"
        tr = C.rows_for(rows, train_years, "train")
        va = C.rows_for(rows, train_years, "val")
        mdl, info = NDVIThreshold.fit(cfg, tr, va)
        # calibration scalar on train-year TRAIN blocks (raw NDVI, no per-year norm)
        s = C.fit_scalar_generic(cfg, tr, stats, mdl.predict_pixels, gate=True,
                                 normalize_inputs=False)
        p_ha, off, ratio = C.full_raster_ratio(cfg, test_year, stats, mdl.predict_pixels,
                                               s, gate=True, normalize_inputs=False)
        write_run(cfg, "ndvi_threshold", test_year, {"aoi_ratio": ratio}, track="B",
                  n_train_tiles=len(tr), n_test_tiles=0, calibration_scalar=s,
                  fit_years=train_years,
                  extra={**info, "pred_ha": p_ha, "official_ha": off,
                         "leakage_assert": f"held-out {test_year} absent from fit years {train_years}"})
        print(f"  {test_year}: t={info['selected_t']:.2f} c={info['fitted_c']:.4f} "
              f"s={s:.2f} ratio={ratio:.3f}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Baseline A: NDVI threshold.")
    ap.add_argument("--config", default=None)
    run_track_b(load_config(ap.parse_args().config))
