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
CALIB_SEED = 7
CALIB_N = 1_000_000


def _calib_pixels(cfg, rows, stats, predict_pixels, normalize_inputs):
    """A5: density-scale calibration a = Σ(train target)/Σ(train pred), ungated,
    on a fixed-seed train pixel sample."""
    X, y, *_ = C.sample_train_pixels(cfg, rows, stats, CALIB_N, CALIB_SEED,
                                     normalize_inputs=normalize_inputs)
    den = float(np.sum(predict_pixels(X)))
    return float(np.sum(y)) / den if den > 0 else 1.0


def _calib_tiles(cfg, rows, stats, predict_tile, n=200):
    rng = np.random.default_rng(CALIB_SEED)
    sel = rng.choice(len(rows), min(n, len(rows)), replace=False)
    num = den = 0.0
    for i in sel:
        img, mask = C._tile(cfg, rows[i])
        m, s = stats[int(rows[i]["year"])]
        p = predict_tile(C._norm(img, m, s))
        num += float(mask.sum()); den += float(np.asarray(p).sum())
    return num / den if den > 0 else 1.0


def _unet_predictor(cfg, device):
    ck = torch.load(CKPT, map_location=device, weights_only=False)
    # The checkpoint must come from the SAME data generation the rows are stamped with.
    # Nothing else here would notice: the tiles on disk are gen4, `write_run` stamps gen4
    # from the config, and a stale gen3 checkpoint loads and predicts perfectly happily —
    # producing a row that claims to be a gen4 measurement of a model trained on a
    # different split, i.e. one whose train blocks overlap this generation's test blocks.
    # That is the cross-generation pairing `compare._one_generation` rejects, arriving one
    # layer earlier where no filter can see it. Fail loudly instead.
    want = str(cfg["project"]["data_generation"])
    got = str(ck.get("data_generation"))
    if got != want:
        raise AssertionError(
            f"{CKPT} was trained on data_generation={got!r} but the config and the tile "
            f"index are {want!r}. Its train blocks are not this generation's train blocks, "
            "so any metric from it is a cross-generation number. Retrain first: "
            "`python -m src.train_loyo --final --epochs 30 --patience 6`")
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


def _per_year_write(cfg, method, years, collect_fn, n_train, a):
    """collect_fn(test_rows) -> (probs, targets) for a set of test tiles. Predictions
    are calibrated to the density scale by `a` (A5) before metrics."""
    rows = C.load_index(cfg)
    for y in years:
        te = C.rows_for(rows, [y], "test")
        probs, tgts = collect_fn(te)
        m = C.regression_metrics(cfg, np.asarray(probs) * a, tgts)
        write_run(cfg, method, y, m, track="A", n_train_tiles=n_train,
                  n_test_tiles=len(te), calibration_scalar=None, fit_years=years,
                  extra={"track_a_density_calib_a": a, "calib_convention": "A5"})
        print(f"  [{method}] {y}: a={a:.3f} MAE={m['mae']:.4f} IoU={m['presence_iou']:.3f} "
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
    a_u = _calib_tiles(cfg, tr, ck_stats, predict_tile)
    _per_year_write(cfg, "unet", years,
                    lambda te: C.collect_tiles(cfg, te, ck_stats, predict_tile),
                    n_train=len(tr), a=a_u)

    # --- NDVI ramp (raw NDVI) ---
    print("[track-a] NDVI ramp")
    ndvi, info = NDVIThreshold.fit(cfg, tr, va)
    print(f"          selected t={info['selected_t']:.2f} (val F1={info['val_f1_at_t']:.3f})")
    a_n = _calib_pixels(cfg, tr, stats, ndvi.predict_pixels, normalize_inputs=False)
    _per_year_write(cfg, "ndvi_threshold", years,
                    lambda te: C.collect_pixels(cfg, te, stats, ndvi.predict_pixels,
                                                normalize_inputs=False),
                    n_train=len(tr), a=a_n)

    # --- Random forest (normalized 18ch) ---
    print("[track-a] random forest")
    rf, rinfo = PixelRF.fit(cfg, tr, stats, seed=SEED)
    a_r = _calib_pixels(cfg, tr, stats, rf.predict_pixels, normalize_inputs=True)
    _per_year_write(cfg, "random_forest", years,
                    lambda te: C.collect_pixels(cfg, te, stats, rf.predict_pixels,
                                                normalize_inputs=True),
                    n_train=len(tr), a=a_r)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Track A — spatial 3-way on test blocks.")
    ap.add_argument("--config", default=None)
    run(load_config(ap.parse_args().config))
