"""Phase 6.6 gate diagnostic (prereg A10) — sweep the presence gate TAU on the 2022
LOYO fold (corrected data) to test whether the collapse (ratio 1.48->0.08) is the
fixed absolute gate.

The gate + calibration scalar are refit together at each TAU (both gated identically,
train years only), mirroring train_loyo.fit_scalar / test_year_ratio. Per-fold models
are not saved, so the 2022 fold is re-trained here (seed 42 -> reproduces ~0.08 at the
operational TAU=0.05). Heavy artifacts (trained model, train per-TAU sums, test-year
density) are cached so a reaped run resumes without retraining.

    python -m src.gate_sweep       # writes docs/gate_sweep_2022.md
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from src.baselines.common import OFFICIAL_HA
from src.infer import predict_raster
from src.train import pick_device
from src.train_loyo import read_index, train_fold, year_norm_stats
from src.utils import load_config, set_seed

TEST_YEAR = 2022
TRAIN_YEARS = [2019, 2020, 2021, 2023, 2024]
TAUS = [0.0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10]
PX_HA = (20 ** 2) / 1e4
CKPT = Path("outputs/checkpoints/loyo_2022_corrected.pt")
CACHE = Path("outputs/metrics/gate_sweep_2022_cache.npz")


def _phase1(cfg):
    """Train the 2022 fold, then cache: train mask-sum (num), per-TAU train gated
    pred-sum (den), and the full test-year density map. Resumes from cache if present."""
    if CACHE.exists():
        print(f"[gate] using cached artifacts {CACHE}")
        z = np.load(CACHE)
        return float(z["num"]), z["den"], z["test_density"]

    device = pick_device()
    set_seed(cfg["project"]["seed"])
    rows = read_index(cfg)
    stats = year_norm_stats(cfg, rows)
    tr = [r for r in rows if int(r["year"]) in TRAIN_YEARS and r["split"] == "train"]
    va = [r for r in rows if int(r["year"]) in TRAIN_YEARS and r["split"] == "val"]
    print(f"[gate] training 2022 fold ({len(tr)} train / {len(va)} val) ...", flush=True)
    model, vmae = train_fold(cfg, tr, va, stats, device, cfg["train"]["epochs"],
                             cfg["train"]["early_stop_patience"])
    CKPT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict()}, CKPT)
    print(f"[gate] trained (val_mae={vmae:.4f}); saved {CKPT}", flush=True)

    # train-year accumulators: num = sum(mask); den(TAU) = sum(gated pred) over train tiles
    tiles_dir = Path(cfg["paths"]["tiles_dir"])
    num = 0.0
    den = np.zeros(len(TAUS))
    model.eval()
    with torch.no_grad():
        for r in tr:
            npz = np.load(tiles_dir / r["npz"])
            img = np.nan_to_num(npz["image"].astype("float32"), nan=0.0)
            mean, std = stats[int(r["year"])]
            x = torch.from_numpy(((img - mean.reshape(-1, 1, 1)) / std.reshape(-1, 1, 1))[None]).to(device)
            p = torch.sigmoid(model(x))[0, 0].cpu().numpy()
            num += float(npz["mask"].astype("float32").sum())
            for i, t in enumerate(TAUS):
                den[i] += float((p * (p >= t)).sum())
    print(f"[gate] train num={num:,.0f} den(TAU=0.05)={den[TAUS.index(0.05)]:,.0f}", flush=True)

    region = cfg["aoi"]["region"]
    mean, std = stats[TEST_YEAR]
    dens, _ = predict_raster(cfg, f"data/imagery/{region}_{TEST_YEAR}_annual_full.tif",
                             model, mean, std, device)
    dens = dens.astype("float32")
    np.savez_compressed(CACHE, num=num, den=den, test_density=dens, taus=np.array(TAUS))
    print(f"[gate] cached test density {dens.shape} -> {CACHE}", flush=True)
    return num, den, dens


def run(cfg):
    num, den, dens = _phase1(cfg)
    official = OFFICIAL_HA[TEST_YEAR]
    rows_out = []
    for i, t in enumerate(TAUS):
        scalar = num / den[i] if den[i] > 0 else float("nan")
        test_total = float((dens * (dens >= t)).sum()) * PX_HA
        pred_ha = test_total * scalar
        ratio = pred_ha / official
        frac_kept = float((dens >= t).mean())
        rows_out.append((t, scalar, pred_ha, ratio, frac_kept))
        print(f"[gate] TAU={t:.3f} scalar={scalar:6.2f} pred={pred_ha:9,.0f} "
              f"ratio={ratio:.3f} kept={100*frac_kept:.1f}%", flush=True)

    r0 = rows_out[0][3]                      # TAU=0
    r05 = [r for r in rows_out if abs(r[0] - 0.05) < 1e-9][0][3]
    lines = ["# Phase 6.6 — Presence-gate (TAU) sweep on the 2022 fold (corrected data)",
             "",
             "Prereg: A10. Re-trained 2022 LOYO fold on corrected data; scalar refit at "
             "each TAU (train years only). Official 2022 = 37,965 ha. v2.1 (contaminated, "
             "TAU=0.05) = 1.48; corrected @ TAU=0.05 = 0.08 (the collapse).",
             "",
             "| TAU | scalar | predicted ha | aoi_ratio | pixels kept |",
             "|---|--:|--:|--:|--:|"]
    for t, s, p, ratio, fk in rows_out:
        lines.append(f"| {t:.3f} | {s:.2f} | {p:,.0f} | {ratio:.3f} | {100*fk:.1f}% |")
    verdict = ("gate CONFIRMED (ratio recovers toward ~1 as TAU->0)" if r0 >= 0.75
               else "gate EXONERATED (no recovery at TAU->0)" if r0 < 0.4
               else "PARTIAL recovery")
    lines += ["",
              f"- aoi_ratio at **TAU=0** = **{r0:.3f}**, at operational **TAU=0.05** = "
              f"**{r05:.3f}**. → **{verdict}.**",
              "- Reading (A10): if confirmed, the fixed absolute gate is a bug — a "
              "per-year-normalized model's output distribution moves year to year, so a "
              "constant 0.05 cut removes a different mass fraction each year; the fix is a "
              "train-year **quantile** gate. Normalization remains a Phase 9 ablation."]
    Path("docs/gate_sweep_2022.md").write_text("\n".join(lines) + "\n")
    print("[gate] wrote docs/gate_sweep_2022.md")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="TAU gate sweep on the 2022 fold (A10).")
    ap.add_argument("--config", default=None)
    run(load_config(ap.parse_args().config))
