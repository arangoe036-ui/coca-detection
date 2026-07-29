"""Track A — spatial skill on the spatially-held-out `test` blocks (prereg A2).

The track that answers "is a segmentation model warranted?" All three methods are
fit on the ALL-YEARS `train` blocks and evaluated on the ALL-YEARS `test` blocks
(spatial holdout, leakage-free). Metrics are reported per year (6 paired points)
via the existing evaluate helper — MAE/RMSE/bias/presence-IoU/presence-F1 on the
density fraction (pre-scalar).

- U-Net: the existing `final_multiyear.pt` checkpoint (no re-training).
- NDVI threshold and RF: refit here on the all-years train/val blocks.

    python -m src.baselines.track_a
"""

from __future__ import annotations

import numpy as np
import torch

from src.baselines import common as C
from src.baselines.ndvi_threshold import NDVIThreshold
from src.baselines.pixel_rf import PixelRF, SEED
from src.metrics_io import write_run
from src.models.unet import build_unet
from src.train import pick_device
from src.utils import load_config

CKPT = "outputs/checkpoints/final_multiyear.pt"


def _unet_predictor(cfg, device):
    ck = torch.load(CKPT, map_location=device, weights_only=False)
    model = build_unet(cfg).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    @torch.no_grad()
    def predict_tile(imgn):  # normalized (C,H,W) -> (H,W) density
        t = torch.from_numpy(np.ascontiguousarray(imgn))[None].float().to(device)
        return torch.sigmoid(model(t))[0, 0].cpu().numpy()

    # the checkpoint's own per-year stats (matches how it was trained/normalized)
    stats = {int(y): (np.asarray(m), np.asarray(s)) for y, (m, s) in ck["year_stats"].items()}
    return predict_tile, stats


def _per_year_write(cfg, method, years, collect_fn, n_train):
    """collect_fn(test_rows) -> (probs, targets) for a set of test tiles."""
    rows = C.load_index(cfg)
    for y in years:
        te = C.rows_for(rows, [y], "test")
        probs, tgts = collect_fn(te)
        m = C.regression_metrics(cfg, probs, tgts)
        write_run(cfg, method, y, m, track="A", n_train_tiles=n_train,
                  n_test_tiles=len(te), calibration_scalar=None, fit_years=years)
        print(f"  [{method}] {y}: MAE={m['mae']:.4f} IoU={m['presence_iou']:.3f} "
              f"F1={m['presence_f1']:.3f}")


def run(cfg, years=None):
    years = years or C.YEARS
    device = pick_device()
    rows = C.load_index(cfg)
    stats = C.all_stats(cfg, rows)
    tr = C.rows_for(rows, years, "train")
    va = C.rows_for(rows, years, "val")
    print(f"[track-a] fit on {len(tr)} all-years train tiles; eval per-year on test blocks")

    # --- U-Net (existing checkpoint) ---
    print("[track-a] U-Net (final_multiyear.pt)")
    predict_tile, ck_stats = _unet_predictor(cfg, device)
    _per_year_write(cfg, "unet", years,
                    lambda te: C.collect_tiles(cfg, te, ck_stats, predict_tile),
                    n_train=len(tr))

    # --- NDVI threshold (raw NDVI) ---
    print("[track-a] NDVI threshold")
    ndvi, info = NDVIThreshold.fit(cfg, tr, va)
    print(f"          selected t={info['selected_t']:.2f} c={info['fitted_c']:.4f}")
    _per_year_write(cfg, "ndvi_threshold", years,
                    lambda te: C.collect_pixels(cfg, te, stats, ndvi.predict_pixels,
                                                normalize_inputs=False),
                    n_train=len(tr))

    # --- Random forest (normalized 18ch) ---
    print("[track-a] random forest")
    rf, rinfo = PixelRF.fit(cfg, tr, stats, seed=SEED)
    _per_year_write(cfg, "random_forest", years,
                    lambda te: C.collect_pixels(cfg, te, stats, rf.predict_pixels,
                                                normalize_inputs=True),
                    n_train=len(tr))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Track A — spatial 3-way on test blocks.")
    ap.add_argument("--config", default=None)
    run(load_config(ap.parse_args().config))
