"""P2 — torch Dataset / DataLoader over the tiled stack + mask (plan §7.3).

Reads the tiles written by ``tiling.py`` (``tiles_index.csv`` + per-tile .npz),
returns (image[C,H,W], mask[1,H,W]) tensors, handles cloud-gap NaNs, applies
per-channel standardization, optional geometric augmentation (flips / 90° rots),
and provides a positive-oversampling sampler for class imbalance.

The label is trained as **presence** (binary): target = (coca_ha > 0), matching
the density/presence framing at the coarse label granularity (plan §6.2).
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from src.utils import load_config


def read_index(cfg: dict) -> list[dict]:
    """Load tiles_index.csv as a list of row dicts."""
    idx_path = Path(cfg["paths"]["tiles_dir"]) / "tiles_index.csv"
    with open(idx_path, newline="") as fh:
        return list(csv.DictReader(fh))


def compute_norm_stats(cfg: dict, rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel mean/std over the given tiles (NaNs ignored). Shape [C]."""
    tiles_dir = Path(cfg["paths"]["tiles_dir"])
    sums = sqs = cnts = None
    for r in rows:
        img = np.load(tiles_dir / r["npz"])["image"].astype("float64")  # [C,H,W]
        c = img.shape[0]
        flat = img.reshape(c, -1)
        finite = np.isfinite(flat)
        flat = np.where(finite, flat, 0.0)
        if sums is None:
            sums, sqs, cnts = (np.zeros(c), np.zeros(c), np.zeros(c))
        sums += flat.sum(1)
        sqs += (flat**2).sum(1)
        cnts += finite.sum(1)
    mean = sums / np.maximum(cnts, 1)
    var = np.maximum(sqs / np.maximum(cnts, 1) - mean**2, 1e-6)
    return mean.astype("float32"), np.sqrt(var).astype("float32")


class CocaTileDataset(Dataset):
    """Yields (image, mask) tensors for a set of index rows."""

    def __init__(self, cfg: dict, rows: list[dict], mean: np.ndarray, std: np.ndarray,
                 augment: bool = False):
        self.cfg = cfg
        self.rows = rows
        self.tiles_dir = Path(cfg["paths"]["tiles_dir"])
        self.mean = mean.reshape(-1, 1, 1)
        self.std = std.reshape(-1, 1, 1)
        self.augment = augment
        self.regression = cfg.get("model", {}).get("task") == "regression"

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        r = self.rows[i]
        npz = np.load(self.tiles_dir / r["npz"])
        img = npz["image"].astype("float32")            # [C,H,W]
        m = npz["mask"].astype("float32")
        # regression -> keep the [0,1] coca fraction; segmentation -> binarize to presence
        mask = m if self.regression else (m > 0).astype("float32")     # [H,W]

        img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
        img = (img - self.mean) / self.std

        if self.augment:
            if np.random.rand() < 0.5:
                img, mask = img[:, :, ::-1], mask[:, ::-1]
            if np.random.rand() < 0.5:
                img, mask = img[:, ::-1, :], mask[::-1, :]
            k = np.random.randint(4)
            if k:
                img = np.rot90(img, k, axes=(1, 2))
                mask = np.rot90(mask, k)

        img = torch.from_numpy(np.ascontiguousarray(img))
        mask = torch.from_numpy(np.ascontiguousarray(mask)).unsqueeze(0)  # [1,H,W]
        return img, mask


def build_dataloader(cfg: dict, split: str, mean=None, std=None, augment=None,
                     overfit_n: int | None = None):
    """Build a DataLoader for a split.

    ``overfit_n`` (sanity gate): take the first N tiles regardless of split — the
    quick AOI is a single spatial block, so all tiles are in ``train``.
    Returns (loader, mean, std) so stats fit on train can be reused for val/test.
    """
    rows = read_index(cfg)
    if overfit_n is not None:
        rows = rows[:overfit_n]
    else:
        rows = [r for r in rows if r["split"] == split]
    if not rows:
        raise RuntimeError(f"No tiles for split={split} (overfit_n={overfit_n}).")

    if mean is None or std is None:
        mean, std = compute_norm_stats(cfg, rows)

    if augment is None:
        augment = (split == "train" and overfit_n is None)
    ds = CocaTileDataset(cfg, rows, mean, std, augment=augment)

    sampler = None
    shuffle = False
    if split == "train" and overfit_n is None and cfg["tiling"]["oversample_positive"]:
        pos = np.array([float(r["pos_frac"]) > 0 for r in rows], dtype="float64")
        w = np.where(pos > 0, 1.0 / max(pos.sum(), 1), 1.0 / max((~pos.astype(bool)).sum(), 1))
        sampler = WeightedRandomSampler(w, num_samples=len(rows), replacement=True)
    elif overfit_n is None:
        shuffle = split == "train"

    loader = DataLoader(
        ds, batch_size=cfg["train"]["batch_size"], sampler=sampler, shuffle=shuffle,
        num_workers=0, drop_last=False,
    )
    return loader, mean, std


if __name__ == "__main__":
    cfg = load_config()
    loader, mean, std = build_dataloader(cfg, "train", overfit_n=cfg["train"]["overfit_sanity_tiles"])
    xb, yb = next(iter(loader))
    print(f"batch image={tuple(xb.shape)} mask={tuple(yb.shape)} "
          f"pos_frac={yb.mean().item():.3f} channels={len(mean)}")
