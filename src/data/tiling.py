"""P1c — Tile the stacked raster + mask into patches with a SPATIAL block split.

Implements plan §6.3. Cuts the imagery stack + aligned label mask into
``tiling.tile_px`` patches (stride ``tiling.stride_px``), then overlays a coarse
grid of ``tiling.split_block_km`` blocks and assigns WHOLE blocks to
train/val/test (~70/15/15). Because whole blocks go to one split, nearby pixels
never leak across splits (plan §3) — the cardinal rule for honest EO metrics.

Each kept tile is written as a compressed .npz (image[C,H,W] float32,
mask[H,W]) plus a row in ``tiles_index.csv`` (tile_id, split, block, pos_frac).

    python -m src.data.tiling \
        --image data/imagery/catatumbo_2023_s0_quick.tif \
        --mask  data/labels/catatumbo_2023_s0_quick_cocamask.tif

Acceptance (P1c): tile counts per split reported.
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

from src.utils import ensure_dirs, load_config


def _windows(width: int, height: int, tile: int, stride: int):
    """Yield (col_off, row_off) for tiles covering the raster (clamped to edges)."""
    xs = list(range(0, max(width - tile, 0) + 1, stride)) or [0]
    ys = list(range(0, max(height - tile, 0) + 1, stride)) or [0]
    if xs[-1] != width - tile and width > tile:
        xs.append(width - tile)
    if ys[-1] != height - tile and height > tile:
        ys.append(height - tile)
    for y in ys:
        for x in xs:
            yield x, y


def _assign_blocks(block_ids: list, ratios: dict, seed: int) -> dict:
    """Randomly assign whole spatial blocks to splits per the ratios (seeded)."""
    blocks = sorted(set(block_ids))
    rng = random.Random(seed)
    rng.shuffle(blocks)
    n = len(blocks)
    n_train = max(int(round(ratios["train"] * n)), 1 if n else 0)
    n_val = int(round(ratios["val"] * n))
    assign = {}
    for i, b in enumerate(blocks):
        if i < n_train:
            assign[b] = "train"
        elif i < n_train + n_val:
            assign[b] = "val"
        else:
            assign[b] = "test"
    return assign


def tile(cfg: dict, image_path: str, mask_path: str) -> Path:
    import numpy as np
    import rasterio

    tc = cfg["tiling"]
    tpx, stride = tc["tile_px"], tc["stride_px"]
    block_px = int(round(tc["split_block_km"] * 1000 / cfg["imagery"]["resolution_m"]))
    ensure_dirs(cfg)
    out_dir = Path(cfg["paths"]["tiles_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(image_path) as img, rasterio.open(mask_path) as msk:
        assert (img.width, img.height) == (msk.width, msk.height), "image/mask shape mismatch"
        assert img.crs == msk.crs and img.transform == msk.transform, "image/mask not aligned"

        records = []
        for (x, y) in _windows(img.width, img.height, tpx, stride):
            win = rasterio.windows.Window(x, y, tpx, tpx)
            stack = img.read(window=win).astype("float32")   # [C, tpx, tpx]
            m = msk.read(1, window=win)                       # [tpx, tpx]
            if stack.shape[1:] != (tpx, tpx):                 # edge (shouldn't happen after clamp)
                continue
            finite = np.isfinite(stack).all(axis=0)
            if finite.mean() < 0.5:                           # skip mostly-nodata tiles
                continue
            # Spatial block id from the tile's top-left pixel.
            block = (x // block_px, y // block_px)
            pos_frac = float((m > 0).mean())
            records.append({"x": x, "y": y, "block": f"{block[0]}_{block[1]}",
                            "pos_frac": pos_frac, "stack": stack, "mask": m})

    if not records:
        raise RuntimeError("No valid tiles produced (all nodata?).")

    assign = _assign_blocks([r["block"] for r in records], tc["split_ratios"],
                            cfg["project"]["seed"])

    index_path = out_dir / "tiles_index.csv"
    counts = {"train": 0, "val": 0, "test": 0}
    with open(index_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["tile_id", "split", "block", "x", "y", "pos_frac", "npz"])
        for i, r in enumerate(records):
            split = assign[r["block"]]
            counts[split] += 1
            tile_id = f"tile_{i:05d}"
            npz = out_dir / f"{tile_id}.npz"
            np.savez_compressed(npz, image=r["stack"], mask=r["mask"])
            w.writerow([tile_id, split, r["block"], r["x"], r["y"],
                        f"{r['pos_frac']:.4f}", npz.name])

    n_blocks = len(set(r["block"] for r in records))
    print(f"[tiling] tiles={len(records)}  blocks={n_blocks}  block_px={block_px}")
    print(f"[tiling] split counts: {counts}")
    if n_blocks < 3:
        print("[tiling] NOTE: <3 spatial blocks — quick AOI too small for a real "
              "geographic split; scale to full aoi.bbox for meaningful train/val/test.")
    print(f"[tiling] wrote index -> {index_path}")
    return index_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Tile + spatial-block split (P1c).")
    ap.add_argument("--config", default=None)
    ap.add_argument("--image", required=True)
    ap.add_argument("--mask", required=True)
    args = ap.parse_args()
    tile(load_config(args.config), args.image, args.mask)
