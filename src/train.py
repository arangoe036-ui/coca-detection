"""P2 — Training loop (plan §7.3).

AdamW + cosine LR, weighted positive sampler, early stopping on val IoU,
best-checkpoint saving, per-epoch metric logging.

Sanity gate (plan §7.3): first overfit a tiny subset to prove the loop *learns*,
BEFORE any full training or scaling data:

    python -m src.train --overfit-sanity

Full training (after data is scaled to the real AOI):

    python -m src.train
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.data.dataset import build_dataloader
from src.models.losses import build_loss
from src.models.unet import build_unet
from src.utils import ensure_dirs, load_config, set_seed


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def binary_metrics(logits: torch.Tensor, target: torch.Tensor, thr: float = 0.5) -> dict:
    pred = (torch.sigmoid(logits) > thr).float()
    t = target.float()
    tp = (pred * t).sum()
    fp = (pred * (1 - t)).sum()
    fn = ((1 - pred) * t).sum()
    iou = (tp / (tp + fp + fn + 1e-6)).item()
    f1 = (2 * tp / (2 * tp + fp + fn + 1e-6)).item()
    return {"iou": iou, "f1": f1}


@torch.no_grad()
def regression_metrics(logits: torch.Tensor, target: torch.Tensor) -> dict:
    """MAE on the fraction + summed-fraction ratio (proxy for predicted/true hectares)."""
    pred = torch.sigmoid(logits)
    t = target.float()
    mae = (pred - t).abs().mean().item()
    ratio = (pred.sum() / (t.sum() + 1e-6)).item()  # ~ predicted_ha / true_ha on this batch
    return {"mae": mae, "ha_ratio": ratio}


def is_regression(cfg: dict) -> bool:
    return cfg.get("model", {}).get("task") == "regression"


def _run_epoch(cfg, model, loader, loss_fn, device, optimizer=None) -> dict:
    train = optimizer is not None
    model.train(train)
    reg = is_regression(cfg)
    keys = ["mae", "ha_ratio"] if reg else ["iou", "f1"]
    tot_loss, agg, n = 0.0, {k: 0.0 for k in keys}, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        with torch.set_grad_enabled(train):
            logits = model(xb)
            loss = loss_fn(logits, yb)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        bs = xb.size(0)
        tot_loss += loss.item() * bs
        m = regression_metrics(logits.detach(), yb) if reg else binary_metrics(logits.detach(), yb)
        for k in keys:
            agg[k] += m[k] * bs
        n += bs
    return {"loss": tot_loss / n, **{k: v / n for k, v in agg.items()}}


def overfit_sanity(cfg: dict) -> bool:
    """Train on a handful of tiles for many epochs; the loop is healthy if the
    model can (over)fit them — train IoU should climb toward ~1.0 and loss fall.
    """
    device = pick_device()
    n_tiles = cfg["train"]["overfit_sanity_tiles"]
    loader, mean, std = build_dataloader(cfg, "train", overfit_n=n_tiles)
    model = build_unet(cfg).to(device)
    loss_fn = build_loss(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"])

    reg = is_regression(cfg)
    print(f"[sanity] device={device} task={cfg['model'].get('task')} tiles={n_tiles} "
          f"channels={cfg['model']['in_channels']}")
    epochs = 60
    last = {}
    for ep in range(1, epochs + 1):
        last = _run_epoch(cfg, model, loader, loss_fn, device, optimizer=opt)
        if ep % 10 == 0 or ep == 1:
            extra = (f"mae={last['mae']:.4f} ha_ratio={last['ha_ratio']:.2f}"
                     if reg else f"iou={last['iou']:.3f} f1={last['f1']:.3f}")
            print(f"[sanity] epoch {ep:3d}  loss={last['loss']:.4f}  {extra}")

    ok = (last["mae"] < 0.05) if reg else (last["iou"] > 0.85)
    metric = f"MAE={last['mae']:.4f} (gate <0.05)" if reg else f"IoU={last['iou']:.3f} (gate >0.85)"
    print(f"[sanity] {'PASS' if ok else 'FAIL'} — final train {metric}. "
          f"Loop {'learns' if ok else 'is NOT learning — investigate'}.")
    return ok


def train(cfg: dict) -> None:
    """Full training with val early stopping + checkpointing (needs a real split)."""
    device = pick_device()
    ensure_dirs(cfg)
    train_loader, mean, std = build_dataloader(cfg, "train")
    val_loader, _, _ = build_dataloader(cfg, "val", mean=mean, std=std, augment=False)

    model = build_unet(cfg).to(device)
    loss_fn = build_loss(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["train"]["epochs"])

    reg = is_regression(cfg)
    ckpt_dir = Path(cfg["paths"]["checkpoints_dir"])
    # regression: monitor val MAE (lower better); segmentation: val IoU (higher better)
    best, patience = (float("inf") if reg else -1.0), 0
    for ep in range(1, cfg["train"]["epochs"] + 1):
        tr = _run_epoch(cfg, model, train_loader, loss_fn, device, optimizer=opt)
        va = _run_epoch(cfg, model, val_loader, loss_fn, device)
        sched.step()
        if reg:
            print(f"[train] epoch {ep:3d}  train_loss={tr['loss']:.4f} "
                  f"val_mae={va['mae']:.4f} val_ha_ratio={va['ha_ratio']:.2f}")
            improved = va["mae"] < best
        else:
            print(f"[train] epoch {ep:3d}  train_loss={tr['loss']:.4f} "
                  f"val_iou={va['iou']:.3f} val_f1={va['f1']:.3f}")
            improved = va["iou"] > best
        if improved:
            best, patience = (va["mae"] if reg else va["iou"]), 0
            torch.save({"model": model.state_dict(), "mean": mean, "std": std, "cfg": cfg},
                       ckpt_dir / "best.pt")
        else:
            patience += 1
            if patience >= cfg["train"]["early_stop_patience"]:
                print(f"[train] early stop at epoch {ep} (best val {'MAE' if reg else 'IoU'}={best:.4f})")
                break
    print(f"[train] done. best val {'MAE' if reg else 'IoU'}={best:.4f} -> {ckpt_dir/'best.pt'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train baseline coca U-Net (P2).")
    ap.add_argument("--config", default=None)
    ap.add_argument("--overfit-sanity", action="store_true",
                    help="Overfit a tiny subset to prove the loop learns (plan §7.3 gate).")
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["project"]["seed"])
    if args.overfit_sanity:
        overfit_sanity(cfg)
    else:
        train(cfg)
