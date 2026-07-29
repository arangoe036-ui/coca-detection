"""No-skill nulls for Track B (prereg A1).

N1 — constant density: predict the train-years' mean pixel density UNIFORMLY over
     the AOI, then run the identical fit_scalar -> ratio path. Gating a uniform
     field is degenerate (all-or-nothing), so the gate is disabled for N1; it
     should land ~ N2 and confirms the pipeline itself adds no spatial skill.
N2 — historical mean (no model): predict held-out-year hectares = mean of the
     OTHER years' official hectares. The sharpest null; the headline question is
     whether the U-Net beats it.

    python -m src.baselines.nulls
"""

from __future__ import annotations

import numpy as np

from src.baselines import common as C
from src.metrics_io import write_run
from src.utils import load_config


def train_mean_density(cfg, rows) -> float:
    """Mean coca fraction over all pixels of the given tiles (labels only)."""
    tot = n = 0.0
    for r in rows:
        _, mask = C._tile(cfg, r)
        tot += float(mask.sum()); n += mask.size
    return tot / n if n else 0.0


def run(cfg, years=None):
    years = years or C.YEARS
    rows = C.load_index(cfg)
    stats = C.all_stats(cfg, rows)
    print("[nulls] N1 (constant density, gate off) + N2 (historical mean) — Track B")
    for test_year in years:
        train_years = [y for y in years if y != test_year]
        tr = C.rows_for(rows, train_years, "train")

        # --- N2: historical mean of the other years' official hectares ----------
        pred_ha = float(np.mean([C.OFFICIAL_HA[y] for y in train_years]))
        off = C.OFFICIAL_HA[test_year]
        write_run(cfg, "null_historical_mean", test_year,
                  {"aoi_ratio": pred_ha / off}, track="B",
                  n_train_tiles=len(tr), n_test_tiles=0, calibration_scalar=None,
                  fit_years=train_years,
                  extra={"pred_ha": pred_ha, "official_ha": off, "model": "none"})

        # --- N1: uniform train-mean density through the identical scalar path ----
        c = train_mean_density(cfg, tr)
        const = lambda feat, _c=c: np.full(feat.shape[0], _c, dtype="float64")
        s = C.fit_scalar_generic(cfg, tr, stats, const, gate=False,
                                 normalize_inputs=False)
        p_ha, off, ratio = C.full_raster_ratio(cfg, test_year, stats, const, s,
                                               gate=False, normalize_inputs=False,
                                               valid_only=True)
        write_run(cfg, "null_constant_density", test_year,
                  {"aoi_ratio": ratio}, track="B",
                  n_train_tiles=len(tr), n_test_tiles=0, calibration_scalar=s,
                  fit_years=train_years,
                  extra={"pred_ha": p_ha, "official_ha": off,
                         "uniform_density": c, "gate": "disabled"})
        print(f"  {test_year}: N2 ratio={pred_ha/C.OFFICIAL_HA[test_year]:.3f}  "
              f"N1 ratio={ratio:.3f} (c={c:.5f}, s={s:.2f})")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="No-skill nulls (Track B).")
    ap.add_argument("--config", default=None)
    run(load_config(ap.parse_args().config))
