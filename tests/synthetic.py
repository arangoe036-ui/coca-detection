"""Synthetic tile grids for the split tests — no exported imagery required.

Geometry mirrors production (``config/default.yaml``): 256 px tiles, 224 px
stride, 10 km split blocks at 20 m = 500 px, over a canvas the size and shape of
the real Catatumbo AOI. ``test_split_isolation.test_geometry_matches_config``
asserts these constants still match the config, so the tests cannot silently
drift away from the geometry they claim to protect.

It also carries a synthetic COCA FIELD (``blob_mask`` / ``blob_weights``): a
west-central blob whose density falls off steeply in BOTH axes with an empty
east. That is the shape of the real AOI and the reason gen3's contiguous band
split produced a test fold with zero coca, so the gen4 tests are run on a canvas
that can actually reproduce the failure —
``test_block_folds.test_band_rule_empties_a_split_on_this_canvas`` asserts it
does, exactly as ``test_old_corner_rule_would_leak`` does for the leak.
"""

from __future__ import annotations

import random
from functools import lru_cache

import numpy as np

from src.data.tiling import _windows

TILE_PX = 256
STRIDE_PX = 224
BLOCK_PX = 500                      # split_block_km 10 / resolution_m 20
RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}

# Real AOI grid: the built index has x in [0, 4795], y in [0, 5372] -> 23 x 25
# tile positions, 575 per year, 10-11 block columns.
CANVAS_W, CANVAS_H = 5000, 5500

# gen4 / prereg A20 parameters, mirroring src.data.multiyear.
GRID_SHAPE = (5, 4)
N_BLOCK_FOLDS = 6
SYN_YEARS = [2019, 2020, 2021, 2022, 2023, 2024]

# Smaller canvas for the PIXEL-level verification tests. Rasterising nine boolean
# canvases plus six years of masks at full AOI size costs ~250 MB and thousands of
# tile reads; the invariants being asserted (disjointness, positives in every
# year) do not depend on the canvas being AOI-sized.
SMALL_W, SMALL_H = 2500, 2800

# Coca blob: centre west of the middle and south of it, sigma a fifth of the
# canvas, so the eastern quarter is empty in every year (the real x=4704 / 4795
# columns hold 0.000 positive fraction).
BLOB_CENTER = (0.35, 0.55)
BLOB_SIGMA = (0.22, 0.26)
BLOB_CUT = 0.25


def uniform_grid(width: int = CANVAS_W, height: int = CANVAS_H) -> list[tuple[int, int]]:
    """Every tile position on a fully valid canvas (matches the real 575/year)."""
    return list(_windows(width, height, TILE_PX, STRIDE_PX))


def ragged_grid(width: int = CANVAS_W, height: int = CANVAS_H) -> list[tuple[int, int]]:
    """Elliptical valid area — mimics nodata/cloud dropout thinning the outer columns.

    This is the case that motivated cutting by cumulative tile count rather than
    by column index: the edge columns hold far fewer valid tiles.
    """
    cx, cy = width / 2, height / 2
    tiles = []
    for x, y in _windows(width, height, TILE_PX, STRIDE_PX):
        mx, my = x + TILE_PX / 2, y + TILE_PX / 2
        if ((mx - cx) / cx) ** 2 + ((my - cy) / cy) ** 2 <= 1.0:
            tiles.append((x, y))
    return tiles


def split_masks(tile_xy, splits, width: int = CANVAS_W, height: int = CANVAS_H,
                tile_px: int = TILE_PX) -> dict[str, np.ndarray]:
    """Rasterise each split's tile footprints into a boolean canvas mask."""
    masks = {s: np.zeros((height, width), dtype=bool) for s in ("train", "val", "test")}
    for (x, y), split in zip(tile_xy, splits):
        if split is not None:
            masks[split][y:y + tile_px, x:x + tile_px] = True
    return masks


@lru_cache(maxsize=4)
def _blob_profiles(width: int, height: int):
    """1-D Gaussian profiles; the field is separable so no canvas is materialised."""
    fx = np.exp(-0.5 * ((np.arange(width) - BLOB_CENTER[0] * width)
                        / (BLOB_SIGMA[0] * width)) ** 2)
    fy = np.exp(-0.5 * ((np.arange(height) - BLOB_CENTER[1] * height)
                        / (BLOB_SIGMA[1] * height)) ** 2)
    return fx, fy


def blob_mask(x: int, y: int, year: int, width: int = CANVAS_W, height: int = CANVAS_H,
              tile_px: int = TILE_PX) -> np.ndarray:
    """Label mask for the tile at (x, y) in ``year`` — a coca FRACTION, as in production.

    The cut moves 0.01 per year so the six years are not identical, but every
    year keeps the same blob, which is what makes "positives in every year" a
    meaningful thing to assert.
    """
    fx, fy = _blob_profiles(width, height)
    d = fy[y:y + tile_px, None] * fx[None, x:x + tile_px]
    return np.where(d > BLOB_CUT + 0.01 * (year - min(SYN_YEARS)), d, 0.0).astype("float32")


def blob_weights(tile_xy, width: int = CANVAS_W, height: int = CANVAS_H,
                 years=SYN_YEARS, tile_px: int = TILE_PX) -> dict[tuple[int, int], int]:
    """Per-position coca weight = MAX positive-pixel count across years.

    Same definition ``src.data.multiyear.position_coca_weights`` uses on the real
    npz masks, so the synthetic tests exercise the production weighting.
    """
    return {(x, y): max(int((blob_mask(x, y, yr, width, height, tile_px) > 0).sum())
                        for yr in years)
            for x, y in tile_xy}


def old_corner_rule_splits(tile_xy, block_px: int = BLOCK_PX, ratios: dict = RATIOS,
                           seed: int = 42) -> list[str]:
    """Reproduction of the REMOVED buggy rule, for the defect-detection test only.

    Was ``tiling._assign_blocks``: block id from the tile's TOP-LEFT corner only,
    whole blocks handed to splits by seeded random shuffle. Reproduced here (not
    imported) so the buggy path cannot be called by production code again.
    """
    blocks = sorted({f"{x // block_px}_{y // block_px}" for x, y in tile_xy})
    rng = random.Random(seed)
    rng.shuffle(blocks)
    n = len(blocks)
    n_train = max(int(round(ratios["train"] * n)), 1 if n else 0)
    n_val = int(round(ratios["val"] * n))
    assign = {}
    for i, b in enumerate(blocks):
        assign[b] = "train" if i < n_train else ("val" if i < n_train + n_val else "test")
    return [assign[f"{x // block_px}_{y // block_px}"] for x, y in tile_xy]
