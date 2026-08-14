"""Guards on the gen4 stratified macro-block split (prereg A20).

Two things must be true of a split, and gen3 only ever asserted the first:

* **leak-free** — no pixel is in two folds, and therefore none is in two splits.
  ``tests/test_split_isolation.py`` already asserts this for the superseded band
  rule; the same gate is applied here to the macro-block rule.
* **signal-bearing** — every split contains positive labels in EVERY year. Its
  absence is precisely why an empty test fold shipped: `src/evaluate.py:47`
  computes ``tp / (tp + fp + fn + 1e-6)``, so an all-negative target returns
  ``0.000`` with no warning, and every Track A number on gen3 was ``0/0`` dressed
  as a measurement.

The synthetic canvas carries a west-central coca blob that falls off steeply in
both axes (``tests/synthetic.blob_mask``), so
``test_band_rule_empties_a_split_on_this_canvas`` can prove the canvas is not too
benign to reproduce the defect — the same discipline as
``test_old_corner_rule_would_leak``.

The real-index tests read ``data/tiles/`` and SKIP when it is absent, so the
suite stays runnable on a fresh clone. Skipped is not passed.
"""

from __future__ import annotations

import csv
import itertools
import json
from pathlib import Path

import numpy as np
import pytest

from src.data.tiling import (
    BLOCK_FOLD_KEY,
    SPLIT_RULE_BLOCKS,
    SPLITS,
    _gap_px,
    _overlapping_pairs,
    _resolve_overlaps,
    _windows,
    assign_block_folds,
    assign_splits,
    macro_block_edges,
    split_for_block_fold,
    split_pixel_report,
    verify_split_pixels,
)
from src.utils import load_config
from tests.synthetic import (
    BLOCK_PX,
    CANVAS_H,
    CANVAS_W,
    GRID_SHAPE,
    N_BLOCK_FOLDS,
    RATIOS,
    SMALL_H,
    SMALL_W,
    STRIDE_PX,
    SYN_YEARS,
    TILE_PX,
    blob_mask,
    blob_weights,
    ragged_grid,
    uniform_grid,
)

GRIDS = {"uniform": uniform_grid, "ragged": ragged_grid}

#: Synthetic tolerance on the per-fold share of tiles / of coca, in percentage
#: points off the ideal 1/6. Looser than the real grid because a synthetic canvas
#: puts fewer tile positions in each macro-block, so the achievable granularity is
#: coarser. The failure this bounds is gross: a fixed snake assignment produced
#: fold sizes of 27 and 204 on the real grid.
BALANCE_TOL_PP = 8.0

#: Registered A20 acceptance bar for the REAL index, in percentage points off 1/6.
#:
#: The BASIS matters and two different quantities have both been called "coca
#: share" in this project. The bar is stated on the WEIGHT basis — sum over kept
#: positions of the max positive-pixel count across years, which is the quantity
#: ``assign_block_folds`` balances and the one the meta sidecar and A20 report.
#: Measured on the real grid: **2.2431 pp coca (weight) / 2.1407 pp tiles**, i.e.
#: margins of 1.76 and 1.86 pp. For reference the same folds deviate by 2.6131 pp
#: on the per-year UNION positive-pixel basis; that number is reported but is not
#: what the bar is set on.
REAL_BALANCE_TOL_PP = 4.0

#: Registered A20 floor on kept tile positions. Measured on the real grid:
#: 436 / 575 = 75.8% (synthetic grids: 72.1-72.4%).
KEEP_FLOOR = 0.70

#: Registered A20 bar on the gap between two kept tiles in different folds:
#: dropping one LATTICE position leaves ``2 * stride - tile`` = 192 px = 3.84 km
#: at 20 m. This is an acceptance CHECK on the real index, not an invariant of the
#: rule — ``_windows`` clamps the final row/column, which admits gaps as small as
#: 59 px on this AOI. See test_edge_clamped_positions_can_sit_closer_than_one_stride.
MIN_GAP_PX = 2 * STRIDE_PX - TILE_PX


def _assign(grid_name, width=CANVAS_W, height=CANVAS_H):
    tiles = GRIDS[grid_name](width, height)
    weights = blob_weights(tiles, width, height)
    folds, info = assign_block_folds(tiles, TILE_PX, STRIDE_PX, GRID_SHAPE,
                                     N_BLOCK_FOLDS, weights)
    return tiles, folds, info


def _fold_masks(tile_xy, folds, width, height, tile_px=TILE_PX):
    masks: dict[int, np.ndarray] = {}
    for (x, y), f in zip(tile_xy, folds):
        if f is None:
            continue
        m = masks.setdefault(f, np.zeros((height, width), dtype=bool))
        m[y:y + tile_px, x:x + tile_px] = True
    return masks


# --- the macro-block grid ---------------------------------------------------


def test_macro_block_edges_are_whole_tile_positions():
    """Edges live in TILE-INDEX space, so no macro-block can split a tile position.

    Computing them in pixel space is the mistake this guards: a pixel-space edge
    lands mid-position and reintroduces straddlers at a second granularity.
    """
    for n_positions, n_blocks in ((23, 5), (25, 4), (575, 6), (7, 4), (4, 4)):
        edges = macro_block_edges(n_positions, n_blocks)
        assert len(edges) == n_blocks + 1
        assert all(isinstance(e, int) for e in edges)
        assert edges[0] == 0 and edges[-1] == n_positions
        assert edges == sorted(edges)
        sizes = [b - a for a, b in itertools.pairwise(edges)]
        assert sum(sizes) == n_positions
        assert max(sizes) - min(sizes) <= 1, f"uneven macro-blocks {sizes}"


def test_real_aoi_grid_gives_the_registered_macro_block_edges():
    """The 23 x 25 Catatumbo tile grid must cut into the A20 5x4 edges."""
    assert macro_block_edges(23, 5) == [0, 5, 9, 14, 18, 23]
    assert macro_block_edges(25, 4) == [0, 6, 12, 19, 25]


@pytest.mark.parametrize("grid_name", list(GRIDS))
def test_grid_shape_produces_twenty_macro_blocks(grid_name):
    _, _, info = _assign(grid_name)
    assert info["grid_shape"] == list(GRID_SHAPE)
    assert len(info["block_fold"]) == GRID_SHAPE[0] * GRID_SHAPE[1]
    assert sum(info["block_tiles"].values()) == info["n_positions"]
    assert set(info["block_fold"].values()) <= set(range(N_BLOCK_FOLDS))
    assert info["split_rule"] == SPLIT_RULE_BLOCKS


# --- the gate: pixel disjointness -------------------------------------------


@pytest.mark.parametrize("grid_name", list(GRIDS))
def test_no_pixel_overlap_between_block_folds(grid_name):
    """THE GATE: no pixel belongs to tiles of two different block folds."""
    tiles, folds, info = _assign(grid_name)
    masks = _fold_masks(tiles, folds, CANVAS_W, CANVAS_H)
    assert len(masks) == N_BLOCK_FOLDS, "a fold ended up with no tiles at all"
    for a in sorted(masks):
        for b in sorted(masks):
            if a < b:
                assert int((masks[a] & masks[b]).sum()) == 0, f"folds {a}/{b} share pixels"
    assert info["unresolved_cross_fold_overlaps"] == 0


@pytest.mark.parametrize("grid_name", list(GRIDS))
def test_rolled_up_splits_are_pixel_disjoint(grid_name):
    """train/val/test inherit disjointness from the folds they are built from."""
    tiles, folds, _ = _assign(grid_name)
    splits = [None if f is None else split_for_block_fold(f, N_BLOCK_FOLDS) for f in folds]
    masks = {s: np.zeros((CANVAS_H, CANVAS_W), dtype=bool) for s in SPLITS}
    for (x, y), s in zip(tiles, splits):
        if s is not None:
            masks[s][y:y + TILE_PX, x:x + TILE_PX] = True
    assert all(masks[s].any() for s in SPLITS)
    assert int((masks["train"] & masks["val"]).sum()) == 0
    assert int((masks["train"] & masks["test"]).sum()) == 0
    assert int((masks["val"] & masks["test"]).sum()) == 0


@pytest.mark.parametrize("grid_name", list(GRIDS))
def test_folds_are_separated_by_a_positive_gap(grid_name):
    """The STRUCTURAL guarantee: kept tiles in different folds never share a pixel.

    Note what is asserted and what is not. ``> 0`` is guaranteed by the rule. The
    192 px figure A20 quotes is **a measurement of the real index, not a property
    of the algorithm** — see
    ``test_edge_clamped_positions_can_sit_closer_than_one_stride`` for why. On
    these synthetic canvases the minimum is 60 px, and asserting 192 here would be
    asserting a coincidence.
    """
    _, _, info = _assign(grid_name)
    assert info["min_cross_fold_gap_px"] > 0
    assert info["unresolved_cross_fold_overlaps"] == 0


@pytest.mark.parametrize("width,height,min_x_gap,min_y_gap", [
    (5051, 5628, 59, 188),      # the real Catatumbo AOI
    (CANVAS_W, CANVAS_H, 8, 60),
])
def test_edge_clamped_positions_can_sit_closer_than_one_stride(width, height,
                                                               min_x_gap, min_y_gap):
    """``_windows`` clamps the final row/column, so 192 px is NOT a floor anywhere.

    The lattice steps by ``stride`` (224) until the last position, which is pushed
    to ``size - tile`` to cover the raster edge. On the real AOI that makes the
    last x step 91 px and the last y step 220 px, so the smallest NON-OVERLAPPING
    separation available is 315 px in x — a gap of **59 px (1.18 km)**, not 192 px
    (3.84 km). The current index measures 192 px because no fold boundary happens
    to fall on the clamped column; nothing in the rule makes that so, and a future
    rebuild could land there. A20 discloses this, and the 192 px bar is asserted
    on the real index as an acceptance check rather than assumed as an invariant.
    """
    xs = sorted({x for x, _ in _windows(width, height, TILE_PX, STRIDE_PX)})
    ys = sorted({y for _, y in _windows(width, height, TILE_PX, STRIDE_PX)})
    assert xs[-1] == width - TILE_PX and ys[-1] == height - TILE_PX
    assert xs[-1] - xs[-2] < STRIDE_PX, "no clamped final column — premise gone"
    assert min(b - a for a in xs for b in xs if b - a >= TILE_PX) - TILE_PX == min_x_gap
    assert min(b - a for a in ys for b in ys if b - a >= TILE_PX) - TILE_PX == min_y_gap
    assert min(min_x_gap, min_y_gap) < MIN_GAP_PX


def test_fold_rollup_covers_every_fold_and_is_rotatable():
    """4 folds train, 1 val, 1 test — and every fold serves as test under some rotation."""
    for rotation in range(N_BLOCK_FOLDS):
        got = [split_for_block_fold(f, N_BLOCK_FOLDS, rotation) for f in range(N_BLOCK_FOLDS)]
        assert got.count("train") == N_BLOCK_FOLDS - 2
        assert got.count("val") == 1 and got.count("test") == 1
    as_test = {next(f for f in range(N_BLOCK_FOLDS)
                    if split_for_block_fold(f, N_BLOCK_FOLDS, r) == "test")
               for r in range(N_BLOCK_FOLDS)}
    assert as_test == set(range(N_BLOCK_FOLDS)), "some fold can never be held out"


# --- determinism and balance ------------------------------------------------


@pytest.mark.parametrize("grid_name", list(GRIDS))
def test_assignment_deterministic_and_order_independent(grid_name):
    """Same input twice, and the same tiles reversed, give identical assignments."""
    tiles = GRIDS[grid_name]()
    weights = blob_weights(tiles)
    args = (TILE_PX, STRIDE_PX, GRID_SHAPE, N_BLOCK_FOLDS, weights)
    first, info_a = assign_block_folds(tiles, *args)
    again, info_b = assign_block_folds(tiles, *args)
    rev, info_c = assign_block_folds(list(reversed(tiles)), *args)

    assert first == again
    assert info_a == info_b
    assert dict(zip(tiles, first)) == dict(zip(reversed(tiles), rev))
    assert info_a == info_c


def test_assignment_ignores_which_years_were_built():
    """Weights are a MAX over years, so a subset of years must not move a fold.

    LOYO needs the same geographic fold in every year; a weight that depended on
    the year set would silently re-split the AOI when a year is added.
    """
    tiles = uniform_grid()
    args = (TILE_PX, STRIDE_PX, GRID_SHAPE, N_BLOCK_FOLDS)
    all_years, _ = assign_block_folds(tiles, *args, blob_weights(tiles, years=SYN_YEARS))
    subset, _ = assign_block_folds(tiles, *args, blob_weights(tiles, years=SYN_YEARS[:1]))
    assert all_years == subset


@pytest.mark.parametrize("grid_name", list(GRIDS))
def test_fold_balance_within_tolerance(grid_name):
    """Both objectives at once: coca share AND tile count per fold near 1/6.

    Coca-only balancing (a fixed snake) satisfied the first and produced fold
    sizes of 27 to 204 on the real grid, which makes fold rotation for the A17
    municipal ranking meaningless.
    """
    _, _, info = _assign(grid_name)
    assert info["max_coca_share_dev_pp"] <= BALANCE_TOL_PP, info["coca_share_by_block_fold"]
    assert info["max_tile_share_dev_pp"] <= BALANCE_TOL_PP, info["tiles_by_block_fold"]
    n = info["tiles_by_block_fold"]
    assert min(n) >= 0.5 * (sum(n) / N_BLOCK_FOLDS), f"fold sizes too uneven to rotate: {n}"
    assert all(v > 0 for v in info["coca_weight_by_block_fold"]), "a fold has no coca at all"


@pytest.mark.parametrize("grid_name", list(GRIDS))
def test_overlap_resolution_costs_tiles_but_not_most_of_them(grid_name):
    tiles, _, info = _assign(grid_name)
    assert 0 < info["dropped_overlap"] < len(tiles)
    assert info["kept_fraction"] >= KEEP_FLOOR


def test_drop_tie_breaks_toward_the_lower_coca_weight():
    """Equal conflict counts are broken by dropping the EMPTIER tile, so signal survives.

    Note what this does NOT claim. The primary key is the conflict count, and
    coca-dense ground sits in the interior where more fold boundaries pass, so
    the dropped set is on average slightly RICHER in coca than the kept set (real
    grid: 75.8% of positions kept carrying 73.9% of the coca weight). The tie-break
    only decides between tiles that are equally entangled.
    """
    positions = [(0, 0), (STRIDE_PX, 0)]        # 32 px of shared pixels
    pairs = _overlapping_pairs(positions, TILE_PX)
    assert pairs == [(0, 1)]
    assert _resolve_overlaps(positions, [0, 1], [100, 0], pairs) == [True, False]
    assert _resolve_overlaps(positions, [0, 1], [0, 100], pairs) == [False, True]
    # Same fold -> the shared pixels are not a leak -> nothing is dropped.
    assert _resolve_overlaps(positions, [0, 0], [0, 100], pairs) == [True, True]


def test_position_tie_breaks_ascending():
    """Equal conflicts AND equal weight -> drop the LOWEST ``(x, y)`` (A20, v2 rule).

    Not a cosmetic choice: all six swept orderings clear every registered bar, and
    ascending ``(x, y)`` is the one registered because it is best on both balance
    objectives. The direction is therefore part of the rule and is pinned here.
    """
    positions = [(0, 0), (STRIDE_PX, 0)]
    pairs = _overlapping_pairs(positions, TILE_PX)
    assert _resolve_overlaps(positions, [0, 1], [7, 7], pairs) == [False, True]


# --- the canvas is not too benign -------------------------------------------


def test_band_rule_empties_a_split_on_this_canvas():
    """Same canvas, the SUPERSEDED gen3 band rule -> a split with zero coca.

    Without this, the gen4 signal-bearing tests could pass on a synthetic field
    too flat to expose the defect they exist for. It also records the reason
    contiguous bands cannot work for this AOI: the gradient is steep in x AND y.
    """
    tiles = uniform_grid()
    weights = blob_weights(tiles)
    splits, _ = assign_splits(tiles, TILE_PX, BLOCK_PX, RATIOS)
    per_split = {s: 0 for s in SPLITS}
    for p, s in zip(tiles, splits):
        if s is not None:
            per_split[s] += weights[p]
    assert per_split["train"] > 0
    assert min(per_split.values()) == 0, (
        f"band rule left coca in every split ({per_split}) — canvas too benign to "
        "reproduce the gen3 defect")


# --- the verifier itself ----------------------------------------------------


def _small_assignment():
    tiles = uniform_grid(SMALL_W, SMALL_H)
    weights = blob_weights(tiles, SMALL_W, SMALL_H)
    folds, info = assign_block_folds(tiles, TILE_PX, STRIDE_PX, GRID_SHAPE,
                                     N_BLOCK_FOLDS, weights)
    splits = [None if f is None else split_for_block_fold(f, N_BLOCK_FOLDS) for f in folds]
    return tiles, folds, splits, info


def _small_loader(year, x, y):
    return blob_mask(x, y, year, SMALL_W, SMALL_H)


def test_verifier_passes_and_counts_the_union_not_the_seams():
    """Tiles overlap by 32 px, so summing per-tile counts double-counts the seams."""
    tiles, folds, splits, _ = _small_assignment()
    report = verify_split_pixels(tiles, splits, folds, SYN_YEARS, _small_loader,
                                 (SMALL_H, SMALL_W), TILE_PX, verbose=False)
    for s in SPLITS:
        union = sum(report["union_positive_px"][s].values())
        per_tile = sum(report["sum_over_tiles_positive_px"][s].values())
        assert union > 0
        assert per_tile > union, f"{s}: per-tile sum {per_tile} must exceed union {union}"
    assert set(report["split_pixel_intersections"].values()) == {0}
    assert set(report["block_fold_pixel_intersections"].values()) == {0}


def test_verifier_asserts_positives_in_every_split_in_every_year():
    tiles, folds, splits, _ = _small_assignment()
    report = verify_split_pixels(tiles, splits, folds, SYN_YEARS, _small_loader,
                                 (SMALL_H, SMALL_W), TILE_PX, verbose=False)
    for s in SPLITS:
        for year in SYN_YEARS:
            assert report["union_positive_px"][s][year] > 0, f"{s} has no coca in {year}"


def test_verifier_rejects_a_coca_free_split():
    """Prove the guard FIRES — four inert safety checks have already shipped here."""
    tiles, folds, splits, _ = _small_assignment()
    by_pos = dict(zip(tiles, splits))

    def blanked(year, x, y):
        m = _small_loader(year, x, y)
        return np.zeros_like(m) if by_pos[(x, y)] == "test" else m

    with pytest.raises(AssertionError, match="ZERO positive pixels"):
        verify_split_pixels(tiles, splits, folds, SYN_YEARS, blanked,
                            (SMALL_H, SMALL_W), TILE_PX, verbose=False)


def test_verifier_rejects_a_coca_free_block_fold():
    """Folds are rotated into the test role by A17, so an empty FOLD must raise too."""
    tiles, folds, splits, _ = _small_assignment()
    by_pos = dict(zip(tiles, folds))
    victim = max(f for f in folds if f is not None)

    def blanked(year, x, y):
        m = _small_loader(year, x, y)
        return np.zeros_like(m) if by_pos[(x, y)] == victim else m

    with pytest.raises(AssertionError, match=f"{BLOCK_FOLD_KEY}={victim}"):
        verify_split_pixels(tiles, splits, folds, SYN_YEARS, blanked,
                            (SMALL_H, SMALL_W), TILE_PX, verbose=False)


def test_verifier_rejects_overlapping_folds():
    """Two adjacent positions in different folds share 32 px — that must raise."""
    tiles = [(0, 0), (STRIDE_PX, 0), (1024, 0)]
    folds = [0, 1, 2]
    splits = ["train", "test", "val"]
    ones = np.ones((TILE_PX, TILE_PX), dtype="float32")
    with pytest.raises(AssertionError, match="NOT pixel-disjoint"):
        verify_split_pixels(tiles, splits, folds, [2019], lambda *_: ones,
                            (TILE_PX, 1024 + TILE_PX), TILE_PX, verbose=False)


# --- the rebuild path cannot silently revert the split ----------------------


def _fake_gen4_meta(tmp_path, rule=SPLIT_RULE_BLOCKS):
    (tmp_path / "multiyear_index.csv").write_text("tile_id,year,split\n", encoding="utf-8")
    (tmp_path / "multiyear_index_meta.json").write_text(
        json.dumps({"split_rule": rule, "n_folds": N_BLOCK_FOLDS,
                    "archive_tag": "pytest_fake"}), encoding="utf-8")
    return tmp_path


def test_build_all_refuses_to_revert_a_block_fold_index(tmp_path):
    """The documented re-tiling command must not silently reinstate the band split.

    ``build_all`` writes ``assign_splits``, whose test fold on this AOI holds zero
    coca — while the config still stamps gen4 on every metrics row. Following the
    README used to do exactly that. It must refuse, and it must say how to proceed.
    """
    pytest.importorskip("rasterio")
    from src.data import multiyear

    cfg = load_config()
    cfg = {**cfg, "paths": {**cfg["paths"], "tiles_dir": str(_fake_gen4_meta(tmp_path))}}
    with pytest.raises(RuntimeError, match="rebuild-block-folds"):
        multiyear.build_all(cfg, years=[])


def test_build_all_guard_recognises_older_block_fold_rule_versions(tmp_path):
    """A rule-version bump must not open the hole again.

    Keying the guard on the CURRENT rule string would let an index built by an
    earlier version of the same rule be clobbered without a word.
    """
    pytest.importorskip("rasterio")
    from src.data import multiyear

    stale = "stratified_macro_blocks_dual_objective_folds_v1"
    assert stale != SPLIT_RULE_BLOCKS, "pick a rule string that is not the current one"
    cfg = load_config()
    cfg = {**cfg,
           "paths": {**cfg["paths"], "tiles_dir": str(_fake_gen4_meta(tmp_path, stale))}}
    with pytest.raises(RuntimeError, match="rebuild-block-folds"):
        multiyear.build_all(cfg, years=[])


def test_archiving_never_clobbers_and_never_loses_a_superseded_index(tmp_path):
    """Annotate, never overwrite — and only when the rule actually changed."""
    pytest.importorskip("rasterio")
    from src.data import multiyear

    tiles = _fake_gen4_meta(tmp_path)
    original = (tiles / "multiyear_index.csv").read_bytes()

    # Same rule -> nothing archived (a rerun produces the same file).
    assert multiyear._archive_superseded_index(tiles, SPLIT_RULE_BLOCKS) is None
    assert not list(tiles.glob("multiyear_index_pytest_fake.csv"))

    # Different rule -> archived under the sidecar's own tag.
    assert multiyear._archive_superseded_index(tiles, "some_other_rule") == "pytest_fake"
    archived = tiles / "multiyear_index_pytest_fake.csv"
    assert archived.read_bytes() == original

    # An existing archive is never overwritten by a later replacement.
    (tiles / "multiyear_index.csv").write_bytes(b"replaced\n")
    multiyear._archive_superseded_index(tiles, "some_other_rule")
    assert archived.read_bytes() == original


def test_archiving_refuses_an_unrecognised_superseded_rule(tmp_path):
    """Losing a superseded artifact silently is worse than failing the rebuild."""
    pytest.importorskip("rasterio")
    from src.data import multiyear

    tiles = _fake_gen4_meta(tmp_path)
    (tiles / "multiyear_index_meta.json").write_text(
        json.dumps({"split_rule": "some_rule_nobody_recorded"}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="LEGACY_ARCHIVE_TAG"):
        multiyear._archive_superseded_index(tiles, SPLIT_RULE_BLOCKS)


# --- the real index ---------------------------------------------------------


@pytest.fixture(scope="module")
def real_index():
    """Rows of the built multi-year index, or a skip on a fresh clone."""
    pytest.importorskip("rasterio")          # src.data.multiyear imports it at module level
    from src.data import multiyear

    INDEX_NAME, META_NAME = multiyear.INDEX_NAME, multiyear.META_NAME
    REAL_N_FOLDS = multiyear.N_BLOCK_FOLDS

    tiles_dir = Path(load_config()["paths"]["tiles_dir"])
    path = tiles_dir / INDEX_NAME
    if not path.exists():
        pytest.skip("no multi-year index on disk (fresh clone) — nothing to check")
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert rows, f"{path} is empty"
    assert BLOCK_FOLD_KEY in rows[0], (
        f"{path} has no {BLOCK_FOLD_KEY} column — it predates the gen4 A20 rebuild "
        "(python -m src.data.multiyear --rebuild-block-folds)")
    return {"rows": rows, "tiles_dir": tiles_dir, "n_folds": REAL_N_FOLDS,
            "meta": tiles_dir / META_NAME}


@pytest.fixture(scope="module")
def real_measured(real_index):
    """Pixel-level MEASUREMENT of the built index (reads every tile's mask once).

    Deliberately ``split_pixel_report``, which asserts nothing, rather than
    ``verify_split_pixels``, which raises. A raising fixture turns every test that
    depends on it into a fixture ERROR carrying whichever guard fired first, so a
    coca-free split would be reported by the disjointness test's name. Measuring
    here lets each test below fail with its own message. The production guard is
    still exercised, by ``test_real_index_passes_the_production_verifier``.
    """
    rows = real_index["rows"]
    tiles_dir = real_index["tiles_dir"]
    years = sorted({int(r["year"]) for r in rows})
    npz_of = {(int(r["year"]), int(r["x"]), int(r["y"])): r["npz"] for r in rows}
    by_pos = {}
    for r in rows:
        by_pos[(int(r["x"]), int(r["y"]))] = (r["split"], int(r[BLOCK_FOLD_KEY]))
    positions = sorted(by_pos)
    height = max(y for _, y in positions) + TILE_PX
    width = max(x for x, _ in positions) + TILE_PX

    def loader(year, x, y):
        with np.load(tiles_dir / npz_of[(year, x, y)]) as z:
            return z["mask"]

    args = (positions, [by_pos[p][0] for p in positions], [by_pos[p][1] for p in positions],
            years, loader, (height, width), TILE_PX)
    return {"report": split_pixel_report(*args), "args": args}


@pytest.fixture(scope="module")
def real_report(real_measured):
    return real_measured["report"]


def test_real_index_passes_the_production_verifier(real_measured):
    """The guard the rebuild runs must pass on what the rebuild actually wrote."""
    verify_split_pixels(*real_measured["args"], verbose=False)


def test_real_index_splits_are_pixel_disjoint(real_report):
    """THE GATE on the real data: no pixel is in two splits, or in two folds."""
    assert set(real_report["split_pixel_intersections"].values()) == {0}, \
        real_report["split_pixel_intersections"]
    assert set(real_report["block_fold_pixel_intersections"].values()) == {0}, \
        real_report["block_fold_pixel_intersections"]


def test_real_index_every_split_has_positives_in_every_year(real_report):
    """The invariant whose absence let an empty test fold ship (blocker B1)."""
    empty = [(g, y) for g in SPLITS for y in real_report["years"]
             if real_report["union_positive_px"][g][y] == 0]
    assert not empty, (
        "split(s) with zero coca in a year: "
        + ", ".join(f"{g}/{y}" for g, y in empty)
        + " — every presence_iou on them would be 0/0 printed as 0.000")


def test_real_index_every_block_fold_has_positives_in_every_year(real_report):
    """Folds are rotated for out-of-fold A17 inference, so each must be measurable."""
    for g in real_report["groups"]:
        if not g.startswith(BLOCK_FOLD_KEY):
            continue
        for year in real_report["years"]:
            assert real_report["union_positive_px"][g][year] > 0, f"{g} has no coca in {year}"


def test_real_index_fold_balance(real_report, real_index):
    """Registered A20 acceptance bar: both shares within REAL_BALANCE_TOL_PP of 1/6.

    ``coca_weight`` is the WEIGHT basis (max positive pixels per position, summed)
    — the same quantity ``assign_block_folds`` optimises and the meta sidecar and
    A20 quote. The per-year union count is a different number for the same folds
    and is checked separately below so the two can never be swapped silently.
    """
    folds = [g for g in real_report["groups"] if g.startswith(BLOCK_FOLD_KEY)]
    assert len(folds) == real_index["n_folds"]
    bases = {"tiles": {g: real_report["n_tiles"][g] for g in folds},
             "coca (weight basis)": {g: real_report["coca_weight"][g] for g in folds}}
    ideal = 1 / len(folds)
    for name, d in bases.items():
        total = sum(d.values())
        dev = max(abs(v / total - ideal) for v in d.values()) * 100
        assert dev <= REAL_BALANCE_TOL_PP, f"{name} share deviates {dev:.2f} pp: {d}"


def test_real_index_coca_weight_agrees_with_the_meta_sidecar(real_report, real_index):
    """The balance bar and the published shares must be the SAME number.

    They are computed by different code on different inputs: the sidecar's figure
    comes from ``assign_block_folds`` over the full 575-position grid at rebuild
    time, this one from re-reading the kept tiles' masks. Any drift means A20
    quotes a share the split does not have.
    """
    meta = json.loads(real_index["meta"].read_text(encoding="utf-8"))
    from_meta = meta["coca_weight_by_block_fold"]
    from_masks = [real_report["coca_weight"][f"{BLOCK_FOLD_KEY}={i}"]
                  for i in range(real_index["n_folds"])]
    assert from_masks == from_meta
    assert "max positive-pixel count across years" in meta["coca_share_basis"].lower()


def test_real_index_split_matches_its_block_fold(real_index):
    """``split`` must be exactly the roll-up of ``block_fold`` — one source of truth."""
    n_folds = real_index["n_folds"]
    per_pos: dict[tuple[int, int], set] = {}
    for r in real_index["rows"]:
        f = int(r[BLOCK_FOLD_KEY])
        assert 0 <= f < n_folds, f"{BLOCK_FOLD_KEY}={f} outside 0..{n_folds - 1}"
        assert r["split"] == split_for_block_fold(f, n_folds), \
            f"row {r['tile_id']}: split={r['split']} but {BLOCK_FOLD_KEY}={f}"
        per_pos.setdefault((int(r["x"]), int(r["y"])), set()).add((r["split"], f))
    mixed = {p: v for p, v in per_pos.items() if len(v) > 1}
    assert not mixed, f"positions whose fold changes between years: {list(mixed)[:5]}"


def test_real_index_every_position_present_in_every_year(real_index):
    """A position missing in one year breaks the LOYO sibling lookup at run time."""
    rows = real_index["rows"]
    years = sorted({int(r["year"]) for r in rows})
    by_year = {y: {(int(r["x"]), int(r["y"])) for r in rows if int(r["year"]) == y}
               for y in years}
    ref = by_year[years[0]]
    for y in years[1:]:
        assert by_year[y] == ref, f"{y} covers different positions than {years[0]}"
    assert len(rows) == len(ref) * len(years)


def test_real_index_retention_above_the_registered_floor(real_index):
    """A20 bar 4, asserted on the REAL grid rather than only on synthetic ones.

    The denominator is counted from the tiles on disk, not read back out of the
    sidecar the same run wrote: every window position was tiled, so the number of
    ``tile_*.npz`` files in a year IS the position count the split chose from.
    """
    tiles_dir = real_index["tiles_dir"]
    kept = {(int(r["x"]), int(r["y"])) for r in real_index["rows"]}
    years = sorted({int(r["year"]) for r in real_index["rows"]})
    totals = {y: len(list((tiles_dir / str(y)).glob("tile_*.npz"))) for y in years}
    if not all(totals.values()):
        pytest.skip("per-year tile directories absent — cannot count total positions")
    assert len(set(totals.values())) == 1, f"years hold different tile counts: {totals}"
    total = next(iter(totals.values()))
    assert len(kept) <= total
    assert len(kept) / total >= KEEP_FLOOR, (
        f"retention {len(kept)}/{total} = {len(kept) / total:.3f} below the "
        f"registered floor {KEEP_FLOOR}")


def test_real_index_separation_re_derived_from_the_csv(real_index):
    """Re-measure the gap and the adjacency cost from the index, not the sidecar.

    The sidecar's ``min_cross_fold_gap_px`` and ``frac_tiles_at_min_gap`` are
    written by the same run that wrote the CSV, so asserting them against the
    sidecar is self-reporting. A20 states these figures "must appear wherever the
    split is described", which makes them claims about the DATA; they are checked
    here against the data.
    """
    fold_of = {(int(r["x"]), int(r["y"])): int(r[BLOCK_FOLD_KEY]) for r in real_index["rows"]}
    split_of = {(int(r["x"]), int(r["y"])): r["split"] for r in real_index["rows"]}
    positions = sorted(fold_of)

    def touching(assign):
        near = set()
        gaps = []
        for i, a in enumerate(positions):
            for b in positions[i + 1:]:
                if assign[a] == assign[b]:
                    continue
                if abs(a[0] - b[0]) > 2 * TILE_PX or abs(a[1] - b[1]) > 2 * TILE_PX:
                    continue
                g = _gap_px(a, b, TILE_PX)
                gaps.append(g)
                if g <= MIN_GAP_PX:
                    near.add(a)
                    near.add(b)
        return min(gaps), len(near) / len(positions)

    fold_gap, fold_adj = touching(fold_of)
    split_gap, split_adj = touching(split_of)
    assert fold_gap >= MIN_GAP_PX, f"cross-fold gap {fold_gap} px below {MIN_GAP_PX}"
    assert split_gap >= fold_gap
    # Disclosure, not a bar: A20 records that stratifying puts far more tiles
    # against a boundary than the gen3 bands did (10.0%). Bounded loosely so it
    # documents the cost without pinning an exact figure.
    assert 0.25 <= fold_adj <= 0.75, f"{fold_adj:.3f} of kept tiles border another fold"
    assert split_adj <= fold_adj


def test_real_index_meta_records_the_gen4_design(real_index):
    """Provenance: the sidecar must say which rule produced the index, and agree with it."""
    meta_path = real_index["meta"]
    assert meta_path.exists(), f"missing sidecar {meta_path}"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["split_rule"] == SPLIT_RULE_BLOCKS
    assert meta["grid_shape"] == list(GRID_SHAPE)
    assert meta["n_folds"] == real_index["n_folds"]
    assert len(meta["block_fold"]) == GRID_SHAPE[0] * GRID_SHAPE[1]
    positions = {(int(r["x"]), int(r["y"])) for r in real_index["rows"]}
    assert meta["kept"] == len(positions)
    assert meta["kept"] + meta["dropped_overlap"] == meta["n_positions"]
    assert meta["unresolved_cross_fold_overlaps"] == 0
    # Self-reported by construction; the same facts are re-derived from the CSV in
    # test_real_index_separation_re_derived_from_the_csv. This only asserts the
    # sidecar is internally consistent and marked verified.
    assert meta["min_cross_fold_gap_px"] >= MIN_GAP_PX
    assert meta["verified"] is True
    assert meta.get("requires_block_fold_rebuild") is None
