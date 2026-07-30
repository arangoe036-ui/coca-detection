"""v2.1 Phases 2-4 — multi-year training + leave-one-year-out (LOYO) validation.

Phase 2 (year-robust inputs): each tile is standardized by ITS YEAR's per-channel
mean/std, so absolute reflectance drift between years stops shifting predictions
(the diagnosed domain shift). Test-year inputs use the test year's own stats
(unsupervised, no label leakage).

Phase 3 (LOYO): for each held-out year, train on the other years' TRAIN blocks,
early-stop on their VAL blocks, then predict the held-out year — the honest
"predict an unseen year at these locations" test (= the 2026 use case).

Phase 4 (frozen calibration): a single multiplicative scalar s is fit on the
TRAIN years' labeled coca pixels (sum official / sum predicted, gated) and FROZEN
— never re-fit on the held-out year (that would leak and is impossible for 2026).

    python -m src.train_loyo --years 2019 2020 2021 2022 2023 2024
    python -m src.train_loyo --smoke        # quick 2-year, few-epoch wiring test

Reports the LOYO AOI predicted/official ratio per year; their spread is the
generalization band to attach to the 2026 nowcast.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from src.data.labels import fetch_coca_grid
from src.data.multiyear import GRID_FIELD, INDEX_NAME
from src.infer import predict_raster
from src.models.losses import SigmoidMSELoss
from src.models.unet import build_unet
from src.train import pick_device
from src.utils import load_config, set_seed

TAU = 0.05  # gating threshold (matches v2 A2)


def read_index(cfg):
    with open(Path(cfg["paths"]["tiles_dir"]) / INDEX_NAME, newline="") as fh:
        return list(csv.DictReader(fh))


def year_norm_stats(cfg, rows):
    """Per-(year,channel) mean/std over the given rows. {year: (mean[C], std[C])}."""
    tiles_dir = Path(cfg["paths"]["tiles_dir"])
    acc = {}
    for r in rows:
        y = int(r["year"])
        img = np.load(tiles_dir / r["npz"])["image"].astype("float64")
        c = img.shape[0]
        flat = img.reshape(c, -1)
        fin = np.isfinite(flat)
        flat = np.where(fin, flat, 0.0)
        s, sq, n = acc.get(y, (np.zeros(c), np.zeros(c), np.zeros(c)))
        acc[y] = (s + flat.sum(1), sq + (flat**2).sum(1), n + fin.sum(1))
    stats = {}
    for y, (s, sq, n) in acc.items():
        mean = s / np.maximum(n, 1)
        std = np.sqrt(np.maximum(sq / np.maximum(n, 1) - mean**2, 1e-6))
        stats[y] = (mean.astype("float32"), std.astype("float32"))
    return stats


class MYDataset(Dataset):
    def __init__(self, cfg, rows, stats, augment=False):
        self.tiles_dir = Path(cfg["paths"]["tiles_dir"])
        self.rows = rows
        self.stats = stats
        self.augment = augment

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        npz = np.load(self.tiles_dir / r["npz"])
        img = np.nan_to_num(npz["image"].astype("float32"), nan=0.0, posinf=0.0, neginf=0.0)
        mean, std = self.stats[int(r["year"])]
        img = (img - mean.reshape(-1, 1, 1)) / std.reshape(-1, 1, 1)
        mask = npz["mask"].astype("float32")               # coca fraction target
        if self.augment:
            if np.random.rand() < 0.5: img, mask = img[:, :, ::-1], mask[:, ::-1]
            if np.random.rand() < 0.5: img, mask = img[:, ::-1, :], mask[::-1, :]
        img = torch.from_numpy(np.ascontiguousarray(img))
        mask = torch.from_numpy(np.ascontiguousarray(mask)).unsqueeze(0)
        return img, mask


def _loader(cfg, rows, stats, train):
    ds = MYDataset(cfg, rows, stats, augment=train)
    sampler = None
    if train:
        pos = np.array([float(r["pos_frac"]) > 0 for r in rows], dtype="float64")
        w = np.where(pos > 0, 1.0 / max(pos.sum(), 1), 1.0 / max((~pos.astype(bool)).sum(), 1))
        sampler = WeightedRandomSampler(w, num_samples=len(rows), replacement=True)
    return DataLoader(ds, batch_size=cfg["train"]["batch_size"], sampler=sampler,
                      shuffle=False, num_workers=0)


def _mae(model, loader, device):
    model.eval(); tot, n = 0.0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            p = torch.sigmoid(model(xb))
            tot += (p - yb).abs().mean().item() * xb.size(0); n += xb.size(0)
    return tot / n


def train_fold(cfg, train_rows, val_rows, stats, device, epochs, patience):
    model = build_unet(cfg).to(device)
    loss_fn = SigmoidMSELoss().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    tl, vl = _loader(cfg, train_rows, stats, True), _loader(cfg, val_rows, stats, False)
    best, best_state, wait = 1e9, None, 0
    for ep in range(1, epochs + 1):
        model.train()
        for xb, yb in tl:
            xb, yb = xb.to(device), yb.to(device)
            loss = loss_fn(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        vm = _mae(model, vl, device)
        if vm < best - 1e-5:
            best, best_state, wait = vm, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state:
        model.load_state_dict(best_state)
    return model, best


def fit_scalar(cfg, model, calib_rows, stats, device):
    """Frozen calibration s = sum(official coca fraction) / sum(gated predicted),
    over ALL pixels of the TRAIN years' tiles (an AOI-total match, frozen on train
    years — corrects false-positive flood too; overlap cancels in the ratio;
    never uses the held-out year)."""
    tiles_dir = Path(cfg["paths"]["tiles_dir"])
    num = den = 0.0
    model.eval()
    with torch.no_grad():
        for r in calib_rows:
            npz = np.load(tiles_dir / r["npz"])
            img = np.nan_to_num(npz["image"].astype("float32"), nan=0.0)
            mean, std = stats[int(r["year"])]
            x = torch.from_numpy(((img - mean.reshape(-1, 1, 1)) / std.reshape(-1, 1, 1))[None]).to(device)
            pred = torch.sigmoid(model(x))[0, 0].cpu().numpy()
            pred = pred * (pred >= TAU)
            num += float(npz["mask"].astype("float32").sum()); den += float(pred.sum())
    return num / den if den > 0 else 1.0


def test_year_ratio(cfg, model, test_year, scalar, stats, device):
    region = cfg["aoi"]["region"]
    img_path = f"data/imagery/{region}_{test_year}_annual_full.tif"
    mean, std = stats[test_year]
    dens, _ = predict_raster(cfg, img_path, model, mean, std, device)
    dens = dens * (dens >= TAU)
    px_ha = (cfg["imagery"]["resolution_m"] ** 2) / 1e4
    pred_ha = float(dens.sum()) * px_ha * scalar
    ycfg = {**cfg, "year": test_year,
            "labels": {**cfg["labels"], "coca_grid_year_field": GRID_FIELD[test_year]}}
    official = float(fetch_coca_grid(ycfg, tuple(cfg["aoi"]["bbox"]))["coca_ha"].sum())
    return pred_ha, official, pred_ha / official if official else float("nan")


def run_loyo(cfg, years, epochs, patience, holdout=None):
    """`holdout` (default: all `years`) restricts WHICH held-out folds to run; each
    fold still trains on all OTHER years in `years` (identical fold logic). This lets
    an interrupted run resume specific folds. Each fold is appended to
    outputs/metrics/loyo_corrected.jsonl as it completes, so a kill loses nothing."""
    import json
    from pathlib import Path
    device = pick_device()
    set_seed(cfg["project"]["seed"])
    rows = read_index(cfg)
    stats = year_norm_stats(cfg, rows)  # per-year stats over all tiles (inputs only)
    holdout = holdout or years
    out = Path(cfg["paths"]["outputs_dir"]) / "metrics"; out.mkdir(parents=True, exist_ok=True)
    results = []
    for test_year in holdout:
        train_years = [y for y in years if y != test_year]
        tr = [r for r in rows if int(r["year"]) in train_years and r["split"] == "train"]
        va = [r for r in rows if int(r["year"]) in train_years and r["split"] == "val"]
        print(f"\n[loyo] === hold out {test_year} | train on {train_years} "
              f"({len(tr)} train / {len(va)} val tiles) ===", flush=True)
        model, vmae = train_fold(cfg, tr, va, stats, device, epochs, patience)
        s = fit_scalar(cfg, model, tr, stats, device)
        pred_ha, off_ha, ratio = test_year_ratio(cfg, model, test_year, s, stats, device)
        print(f"[loyo] {test_year}: val_mae={vmae:.4f} scalar={s:.3f} "
              f"pred={pred_ha:,.0f} official={off_ha:,.0f} ratio={ratio:.2f}", flush=True)
        results.append((test_year, ratio, pred_ha, off_ha, s))
        with open(out / "loyo_corrected.jsonl", "a") as fh:
            fh.write(json.dumps({"year": test_year, "ratio": ratio, "pred_ha": pred_ha,
                                 "official_ha": off_ha, "scalar": s, "val_mae": vmae}) + "\n")
        # Persist per-fold weights so post-hoc gate/scalar experiments are free (no
        # retrain) — critical under job reaping. Includes the fold's norm stats.
        ck_dir = Path(cfg["paths"]["checkpoints_dir"]); ck_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "test_year": test_year,
                    "train_years": train_years, "scalar": s, "tau": TAU,
                    "year_stats": {int(y): (m, st) for y, (m, st) in stats.items()}},
                   ck_dir / f"loyo_fold_{test_year}.pt")

    ratios = [r for _, r, *_ in results]
    print("\n[loyo] ===== LOYO out-of-year ratio table =====")
    for y, ratio, p, o, s in results:
        print(f"        {y}: ratio={ratio:.2f}  (pred {p:,.0f} / official {o:,.0f}, s={s:.2f})")
    print(f"[loyo] spread: min={min(ratios):.2f} max={max(ratios):.2f} "
          f"mean={np.mean(ratios):.2f} +/-{np.std(ratios):.2f}")
    return results


def train_final(cfg, years, epochs, patience):
    """Train the FINAL deployment model on ALL years (train blocks), val on all
    years' val blocks, calibration scalar fit on all years. Saves a checkpoint
    (model + per-year norm stats + frozen scalar) for the 2026 nowcast."""
    device = pick_device()
    set_seed(cfg["project"]["seed"])
    rows = read_index(cfg)
    stats = year_norm_stats(cfg, rows)
    tr = [r for r in rows if int(r["year"]) in years and r["split"] == "train"]
    va = [r for r in rows if int(r["year"]) in years and r["split"] == "val"]
    print(f"[final] train on ALL years {years}: {len(tr)} train / {len(va)} val tiles", flush=True)
    model, vmae = train_fold(cfg, tr, va, stats, device, epochs, patience)
    s = fit_scalar(cfg, model, tr, stats, device)
    ckpt_dir = Path(cfg["paths"]["checkpoints_dir"]); ckpt_dir.mkdir(parents=True, exist_ok=True)
    out = ckpt_dir / "final_multiyear.pt"
    torch.save({"model": model.state_dict(),
                "year_stats": {int(y): (m, st) for y, (m, st) in stats.items()},
                "scalar": float(s), "tau": TAU, "years": years}, out)
    print(f"[final] val_mae={vmae:.4f} scalar={s:.3f} -> saved {out}", flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Multi-year LOYO training/validation (v2.1).")
    ap.add_argument("--config", default=None)
    ap.add_argument("--years", type=int, nargs="+", default=[2019, 2020, 2021, 2022, 2023, 2024])
    ap.add_argument("--holdout", type=int, nargs="+", default=None,
                    help="subset of held-out folds to run (default: all --years); "
                         "each still trains on the other --years. For resuming.")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--smoke", action="store_true", help="quick 2-year, 2-epoch wiring test")
    ap.add_argument("--final", action="store_true", help="train the final all-years deployment model")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.smoke:
        run_loyo(cfg, [2023, 2024], epochs=2, patience=2)
    elif args.final:
        train_final(cfg, args.years, args.epochs, args.patience)
    else:
        run_loyo(cfg, args.years, args.epochs, args.patience, holdout=args.holdout)
