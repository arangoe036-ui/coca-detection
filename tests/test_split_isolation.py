"""Guards on the spatial split: splits must be PIXEL-DISJOINT.

The defect these tests exist for: the split was decided from each tile's top-left
corner block while the tile's 256 px footprint often spanned two blocks, and
blocks were handed to splits by random shuffle. On the real index that put 13.67%
of test pixels inside train tiles.

What is asserted here is ZERO PIXEL OVERLAP between splits. That is NOT
"leakage-free": dropping the straddling tiles leaves only about a one-tile
(~5.1 km at 20 m) gap at each band boundary, and coca autocorrelates well beyond
10 km, so cross-boundary similarity is mitigated, not removed.
"""

from __future__ import annotations

import pytest

from src.data.tiling import assign_splits, footprint_blocks
from src.utils import load_config
from tests.synthetic import (
    BLOCK_PX,
    CANVAS_H,
    CANVAS_W,
    RATIOS,
    STRIDE_PX,
    TILE_PX,
    old_corner_rule_splits,
    ragged_grid,
    split_masks,
    uniform_grid,
)

GRIDS = {"uniform": uniform_grid, "ragged": ragged_grid}


def _assign(grid_name):
    tiles = GRIDS[grid_name]()
    splits, info = assign_splits(tiles, TILE_PX, BLOCK_PX, RATIOS)
    return tiles, splits, info


def test_geometry_matches_config():
    """The synthetic geometry must stay equal to the production geometry."""
    cfg = load_config()
    tc = cfg["tiling"]
    assert tc["tile_px"] == TILE_PX
    assert tc["stride_px"] == STRIDE_PX
    assert round(tc["split_block_km"] * 1000 / cfg["imagery"]["resolution_m"]) == BLOCK_PX
    assert tc["split_ratios"] == RATIOS


@pytest.mark.parametrize("grid_name", list(GRIDS))
def test_no_pixel_overlap_between_splits(grid_name):
    """THE GATE: no pixel belongs to tiles of two different splits."""
    tiles, splits, _ = _assign(grid_name)
    masks = split_masks(tiles, splits, CANVAS_W, CANVAS_H, TILE_PX)
    assert masks["train"].any() and masks["val"].any() and masks["test"].any()
    assert int((masks["train"] & masks["test"]).sum()) == 0
    assert int((masks["train"] & masks["val"]).sum()) == 0
    assert int((masks["val"] & masks["test"]).sum()) == 0


@pytest.mark.parametrize("grid_name", list(GRIDS))
def test_old_corner_rule_would_leak(grid_name):
    """Same canvas, OLD corner+shuffle rule -> non-zero overlap.

    Without this, test_no_pixel_overlap_between_splits could pass on a synthetic
    grid too benign to expose the defect it is meant to catch.
    """
    tiles = GRIDS[grid_name]()
    old = old_corner_rule_splits(tiles)
    masks = split_masks(tiles, old, CANVAS_W, CANVAS_H, TILE_PX)
    train_test = int((masks["train"] & masks["test"]).sum())
    train_val = int((masks["train"] & masks["val"]).sum())
    assert train_test > 0, "old rule showed no train/test overlap — canvas too benign"
    assert train_val > 0, "old rule showed no train/val overlap — canvas too benign"


@pytest.mark.parametrize("grid_name", list(GRIDS))
def test_every_kept_tile_lies_in_one_split(grid_name):
    """Every block a kept tile's footprint touches carries that tile's split."""
    tiles, splits, info = _assign(grid_name)
    block_split = info["block_split"]
    kept = 0
    for (x, y), split in zip(tiles, splits):
        if split is None:
            continue
        kept += 1
        touched = {block_split[f"{bx}_{by}"]
                   for bx, by in footprint_blocks(x, y, TILE_PX, BLOCK_PX)}
        assert touched == {split}, f"tile ({x},{y}) claims {split} but touches {touched}"
    assert kept == info["kept"]
    assert kept + info["dropped_straddler"] == len(tiles)


@pytest.mark.parametrize("grid_name", list(GRIDS))
def test_assignment_deterministic_and_order_independent(grid_name):
    """Same input twice, and the same tiles reversed, give identical assignments."""
    tiles = GRIDS[grid_name]()
    first, info_a = assign_splits(tiles, TILE_PX, BLOCK_PX, RATIOS)
    again, info_b = assign_splits(tiles, TILE_PX, BLOCK_PX, RATIOS)
    rev, info_c = assign_splits(list(reversed(tiles)), TILE_PX, BLOCK_PX, RATIOS)

    assert first == again
    assert info_a == info_b
    assert dict(zip(tiles, first)) == dict(zip(reversed(tiles), rev))
    for key in ("cut_train_val_col", "cut_val_test_col", "kept", "dropped_straddler",
                "block_split"):
        assert info_a[key] == info_c[key]


@pytest.mark.parametrize("grid_name", list(GRIDS))
def test_split_bands_are_contiguous(grid_name):
    """Each split occupies one contiguous run of block-x columns, in train/val/test order."""
    _, _, info = _assign(grid_name)
    cols: dict[str, set[int]] = {"train": set(), "val": set(), "test": set()}
    for block, split in info["block_split"].items():
        cols[split].add(int(block.split("_")[0]))

    for split, cs in cols.items():
        assert cs, f"{split} band has no block columns"
        assert cs == set(range(min(cs), max(cs) + 1)), f"{split} columns not contiguous: {sorted(cs)}"
    assert max(cols["train"]) < min(cols["val"]), "train band must end before val begins"
    assert max(cols["val"]) < min(cols["test"]), "val band must end before test begins"


# Tolerance: 6 percentage points, absolute, per split. The bands are whole 500 px
# block columns and the AOI is ~10 columns wide, so one column is ~10% of the
# tiles; after straddlers are dropped the achievable band sizes move in steps of
# roughly 5 pp. 0.06 is therefore the finest honest tolerance this geometry
# supports — val and test are the tight ones (target 0.15, one column of slack).
RATIO_TOL = 0.06


@pytest.mark.parametrize("grid_name", list(GRIDS))
def test_split_ratios_near_target(grid_name):
    """Kept-TILE fractions land within RATIO_TOL of the configured 0.70/0.15/0.15."""
    _, splits, _ = _assign(grid_name)
    kept = [s for s in splits if s is not None]
    frac = {s: kept.count(s) / len(kept) for s in ("train", "val", "test")}
    for split, target in RATIOS.items():
        assert abs(frac[split] - target) <= RATIO_TOL, f"{split}={frac[split]:.3f} vs {target}"


@pytest.mark.parametrize("grid_name", list(GRIDS))
def test_most_tiles_are_kept(grid_name):
    """The drop rule must cost tiles, but not many — dropping all straddlers keeps 27%."""
    tiles, _, info = _assign(grid_name)
    assert 0 < info["dropped_straddler"] < len(tiles)
    assert info["kept"] / len(tiles) >= 0.80
