"""Baseline A — spectral-index (NDVI) ramp (plan Phase 2).

Method (plan §2 "piecewise-linear ramp of NDVI"): density(pixel) =
clip((NDVI − t)/(T_HI − t), 0, 1) on RAW NDVI (channel 10, confirmed Phase 0.1),
so density rises from 0 at threshold t to 1 near the NDVI ceiling T_HI. The
threshold t is selected by MAXIMUM presence-F1 on the train-years' VAL blocks; the
held-out year is never touched (prereg §2). Uses RAW NDVI (not per-year normalized)
by design — the naive vegetation-index floor.

Why F1, not MAE (logged deviation from the plan's "minimize MAE"): coca density is
extremely imbalanced (mean fraction ~0.04), so an MAE-selected threshold degenerates
to the all-zero predictor (empirically t→0.90, predicting ~nothing; with a constant
below the TAU=0.05 gate the ratio collapses to 0). F1 gives a genuine detection
threshold. This repairs a degenerate estimator; it does not tune toward the U-Net
comparison. The ramp (0→1) also survives the TAU gate, unlike a small constant.

    python -m src.baselines.ndvi_threshold          # Track B (6 LOYO folds)
"""

from __future__ import annotations

import numpy as np

from src.baselines import common as C
from src.metrics_io import write_run
from src.utils import load_config

T_GRID = np.round(np.linspace(0.20, 0.85, 14), 3)   # raw-NDVI detection thresholds
T_HI = 0.90                                          # ramp saturates near NDVI max
PRESENCE_THR = 0.02                                  # matches _metrics_at default


class NDVIThreshold:
    """density(pixel) = clip((NDVI − t)/(T_HI − t), 0, 1)."""

    def __init__(self, t: float):
        self.t = float(t)

    def predict_pixels(self, feat):
        ndvi = feat[:, C.NDVI_IDX]
        return np.clip((ndvi - self.t) / (T_HI - self.t), 0.0, 1.0)

    @classmethod
    def fit(cls, cfg, train_rows, val_rows, seed=42, n_pixels=2_000_000):
        """Select t by MAX presence-F1 on a stratified VAL-block pixel sample (fast,
        low-memory, seed recorded). Fitting touches train years only."""
        Xva, yva, *_ = C.sample_train_pixels(cfg, val_rows, None, n_pixels, seed + 1,
                                             normalize_inputs=False)
        ndvi = Xva[:, C.NDVI_IDX]
        truth = yva > 0
        best = None
        for t in T_GRID:
            pred = np.clip((ndvi - t) / (T_HI - t), 0.0, 1.0) > PRESENCE_THR
            tp = int((pred & truth).sum()); fp = int((pred & ~truth).sum())
            fn = int((~pred & truth).sum())
            f1 = 2 * tp / (2 * tp + fp + fn + 1e-6)
            if best is None or f1 > best[0]:
                best = (f1, float(t))
        return cls(best[1]), {"selected_t": best[1], "val_f1_at_t": best[0],
                              "select_metric": "presence_f1", "t_hi": T_HI,
                              "select_seed": seed, "select_n_pixels": int(len(yva))}


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
        print(f"  {test_year}: t={info['selected_t']:.2f} (F1={info['val_f1_at_t']:.3f}) "
              f"s={s:.3f} ratio={ratio:.3f}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Baseline A: NDVI threshold.")
    ap.add_argument("--config", default=None)
    run_track_b(load_config(ap.parse_args().config))
