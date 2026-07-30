"""Phase 6.6 (A11) — test the scale-invariant quantile gate on the saved corrected
2022 fold, post-hoc (no retrain).

Quantile gate: keep the top `f_keep` fraction of pixels by predicted density in every
year, where `f_keep` = mean per-train-year fraction of pixels with density >= 0.05
(the current fixed gate's train keep-rate, fit on train years only). Scalar refit on
the quantile-gated train predictions. Compare the 2022 aoi_ratio under: ungated,
fixed TAU=0.05 (the collapse), and the quantile gate.

    python -m src.quantile_gate_test
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.baselines.common import OFFICIAL_HA
from src.models.unet import build_unet
from src.train import pick_device
from src.train_loyo import read_index, year_norm_stats
from src.utils import load_config

TEST_YEAR = 2022
TRAIN_YEARS = [2019, 2020, 2021, 2023, 2024]
PX_HA = (20 ** 2) / 1e4
FIXED_TAU = 0.05
CKPT = "outputs/checkpoints/loyo_2022_corrected.pt"
TEST_CACHE = "outputs/metrics/gate_sweep_2022_cache.npz"


def main(cfg):
    device = pick_device()
    rows = read_index(cfg)
    stats = year_norm_stats(cfg, rows)
    model = build_unet(cfg).to(device)
    model.load_state_dict(torch.load(CKPT, map_location=device, weights_only=False)["model"])
    model.eval()
    tiles = Path(cfg["paths"]["tiles_dir"])

    dens_by_year = {y: [] for y in TRAIN_YEARS}
    num = 0.0
    print("[qgate] inferring train-year densities ...", flush=True)
    with torch.no_grad():
        for r in rows:
            y = int(r["year"])
            if y not in TRAIN_YEARS or r["split"] != "train":
                continue
            npz = np.load(tiles / r["npz"])
            img = np.nan_to_num(npz["image"].astype("float32"), nan=0.0)
            m, s = stats[y]
            x = torch.from_numpy(((img - m.reshape(-1, 1, 1)) / s.reshape(-1, 1, 1))[None]).to(device)
            p = torch.sigmoid(model(x))[0, 0].cpu().numpy().astype("float32")
            dens_by_year[y].append(p.ravel())
            num += float(npz["mask"].astype("float32").sum())
    dens_by_year = {y: np.concatenate(v) for y, v in dens_by_year.items()}

    # f_keep = mean per-train-year fraction of pixels >= 0.05 (fit on train only)
    f_keep = float(np.mean([np.mean(d >= FIXED_TAU) for d in dens_by_year.values()]))
    q = 1.0 - f_keep
    print(f"[qgate] f_keep={f_keep:.4f} (keep top {100*f_keep:.1f}% of pixels) -> quantile q={q:.4f}", flush=True)

    # scalar under the quantile gate: per-year threshold, gated train sum
    gated_train = 0.0
    for y, d in dens_by_year.items():
        thr_y = float(np.quantile(d, q))
        gated_train += float(d[d >= thr_y].sum())
    scalar_q = num / gated_train if gated_train > 0 else float("nan")

    # test year (cached full-raster density)
    dens = np.load(TEST_CACHE)["test_density"].astype("float32").ravel()
    official = OFFICIAL_HA[TEST_YEAR]

    def ratio_fixed(tau):
        s_num = num
        s_den = sum(float(d[d >= tau].sum()) for d in dens_by_year.values())
        sc = s_num / s_den if s_den > 0 else float("nan")
        return float((dens[dens >= tau].sum())) * PX_HA * sc / official, sc

    r_ungated, sc0 = ratio_fixed(0.0)
    r_fixed, sc5 = ratio_fixed(FIXED_TAU)
    thr_test = float(np.quantile(dens, q))
    r_quant = float(dens[dens >= thr_test].sum()) * PX_HA * scalar_q / official

    print(f"[qgate] ungated ratio={r_ungated:.3f} (scalar={sc0:.2f})", flush=True)
    print(f"[qgate] fixed TAU=0.05 ratio={r_fixed:.3f} (scalar={sc5:.2f})", flush=True)
    print(f"[qgate] QUANTILE gate ratio={r_quant:.3f} (scalar={scalar_q:.2f}, "
          f"test_thr={thr_test:.4f})", flush=True)

    verdict = ("gate was the mechanism (>=0.75) -> normalization is a Phase 9 design choice"
               if r_quant >= 0.75 else
               "gate NOT the fix (<0.40): raw output collapsed -> retrain with normalization"
               if r_quant < 0.40 else
               "partial (0.40-0.75): fix gate now; normalization carries the remainder")
    lines = ["# Phase 6.6 (A11) — Quantile gate on the corrected 2022 fold (post-hoc)",
             "",
             "Scale-invariant gate: keep the top `f_keep` fraction of pixels per year "
             f"(`f_keep`={f_keep:.3f}, fit on train years as the mean keep-rate at 0.05); "
             "scalar refit on quantile-gated train predictions. Official 2022 = 37,965 ha.",
             "",
             "| gate | 2022 aoi_ratio | scalar |",
             "|---|--:|--:|",
             f"| ungated (TAU=0) | {r_ungated:.3f} | {sc0:.2f} |",
             f"| fixed TAU=0.05 (the collapse) | {r_fixed:.3f} | {sc5:.2f} |",
             f"| **quantile (top {100*f_keep:.1f}%)** | **{r_quant:.3f}** | {scalar_q:.2f} |",
             "",
             f"v2.1 (contaminated, fixed TAU=0.05) 2022 ratio = 1.48 for reference.",
             "",
             f"**Verdict (A11):** {verdict}."]
    Path("docs/quantile_gate_2022.md").write_text("\n".join(lines) + "\n")
    print(f"[qgate] wrote docs/quantile_gate_2022.md -> {verdict}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="A11 quantile gate test on 2022 fold.")
    ap.add_argument("--config", default=None)
    main(load_config(ap.parse_args().config))
