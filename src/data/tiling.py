"""P1c — Tile the stacked raster + mask into patches with a SPATIAL block split.

Implements plan §6.3. Cuts the imagery stack + aligned label mask into
``tiling.tile_px`` patches (stride ``tiling.stride_px``), then assigns whole
spatial blocks to folds/splits.

TWO split rules live here.

* **gen4, current** — ``assign_block_folds``: STRATIFIED MACRO-BLOCKS. A 5x4 grid
  of macro-blocks is cut in TILE-INDEX space (so every macro-block is a whole
  number of existing tile positions), each macro-block is weighted by its coca
  content, and the 20 macro-blocks are greedily assigned to 6 spatial folds that
  balance coca AND tile count jointly. Tiles whose 256 px footprints overlap
  across a fold boundary are then dropped until no pixel is shared. The 6 folds
  roll up to train/val/test 4/1/1 via ``split_for_block_fold``; the per-tile fold
  id is kept so the folds can be ROTATED for out-of-fold inference (A17).
  Registered as prereg **A20**.
* **gen3, superseded** — ``assign_splits`` / ``_band_cuts`` / ``_col_split``:
  contiguous west->east bands of ``tiling.split_block_km`` blocks, ~70/15/15 by
  tile count, straddlers dropped. **Superseded 2026-08-14 by A20 and retained for
  provenance only** (see the note on ``assign_splits``): it is pixel-disjoint, but
  on this AOI it produced a test fold containing ZERO coca pixels, which made
  every Track A number on gen3 unmeasurable. Do not delete — it and its tests are
  the record of the earlier leak fix (defect F3).

Both rules guarantee ZERO PIXEL OVERLAP between splits, which is what this module
asserts (``verify_split_pixels``, ``tests/test_split_isolation.py``,
``tests/test_block_folds.py``). Neither is "leakage-free": dropping one LATTICE
position leaves a gap of ``stride_px * 2 - tile_px`` = 192 px = 3.8 km at 20 m,
and coca autocorrelates well beyond that, so cross-boundary similarity is
MITIGATED, not removed. 192 px is also not a floor — ``_windows`` clamps the
final row/column to the raster edge, so on the real AOI the last x step is 91 px
and a fold boundary falling there would give a 59 px (1.18 km) gap. The gap is
therefore MEASURED per build (``min_cross_fold_gap_px``) and checked against the
A20 bar, never assumed.

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
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path

from src.utils import ensure_dirs, load_config

# Name recorded in index metadata so a built dataset says which rule produced it.
# SUPERSEDED 2026-08-14 by SPLIT_RULE_BLOCKS (prereg A20); kept for provenance so
# a gen3 index still reports the rule that actually produced it.
SPLIT_RULE = "contiguous_x_bands_strict_footprint_v1"

# gen4 rule (prereg A20): stratified macro-blocks -> 6 spatial folds -> 4/1/1.
# v2 (2026-08-14) differs from v1 ONLY in the drop tie-break: ascending (x, y)
# instead of descending, chosen because it is the best of the six swept orderings
# on both registered balance objectives. See ``_resolve_overlaps`` and A20.
SPLIT_RULE_BLOCKS = "stratified_macro_blocks_dual_objective_folds_v2"

#: Fold ids are recorded under this name everywhere — CSV column, meta keys, info
#: dict. NOT ``fold``: ``src/metrics_io.py`` and ``src/baselines/compare.py``
#: already use ``fold_year`` for the leave-one-year-out fold, and
#: ``compare._dedup_last`` keys on it. A ``fold`` collision would silently
#: de-duplicate rows from different spatial folds against each other.
BLOCK_FOLD_KEY = "block_fold"


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


def block_px_from_cfg(cfg: dict) -> int:
    """Split-block edge in pixels (``split_block_km`` at ``resolution_m``).

    One place only: multiyear and tiling must derive the SAME block grid or the
    split silently stops matching the one the index was built with.
    """
    return round(cfg["tiling"]["split_block_km"] * 1000 / cfg["imagery"]["resolution_m"])


def block_id(x: int, y: int, block_px: int) -> str:
    """Canonical block id string for the pixel (x, y) — the CSV ``block`` column."""
    return f"{x // block_px}_{y // block_px}"


def footprint_blocks(x: int, y: int, tile_px: int, block_px: int) -> list[tuple[int, int]]:
    """Every (block_x, block_y) the tile's [x, x+tile_px) x [y, y+tile_px) extent touches.

    The old rule used only the top-left corner's block, so a tile whose 256 px
    footprint crossed a block boundary was filed under one block while its pixels
    sat in two — the mechanism behind the measured train/test pixel overlap.
    """
    bx0, bx1 = x // block_px, (x + tile_px - 1) // block_px
    by0, by1 = y // block_px, (y + tile_px - 1) // block_px
    return [(bx, by) for bx in range(bx0, bx1 + 1) for by in range(by0, by1 + 1)]


def _col_split(bx: int, cut_train: int, cut_val: int) -> str:
    """Split of block column ``bx`` given the two band cut columns (inclusive).

    SUPERSEDED 2026-08-14 by ``assign_block_folds`` (prereg A20). Retained for
    provenance — it is part of the gen3 band rule, not dead scaffolding.
    """
    if bx <= cut_train:
        return "train"
    return "val" if bx <= cut_val else "test"


def _column_tile_counts(positions: Iterable[tuple[int, int]], tile_px: int,
                        block_px: int) -> dict[int, int]:
    """Tiles touching each block column (a straddler counts in both columns)."""
    cols: dict[int, int] = {}
    for x, y in positions:
        for bx in range(x // block_px, (x + tile_px - 1) // block_px + 1):
            cols[bx] = cols.get(bx, 0) + 1
    return cols


def _band_tile_counts(positions: Iterable[tuple[int, int]], tile_px: int, block_px: int,
                      cut_train: int, cut_val: int) -> dict[str, int]:
    """Kept tiles per split (and dropped straddlers) for a candidate pair of cuts."""
    counts = {"train": 0, "val": 0, "test": 0, "dropped": 0}
    for x, y in positions:
        touched = {_col_split(bx, cut_train, cut_val)
                   for bx, _ in footprint_blocks(x, y, tile_px, block_px)}
        counts["dropped" if len(touched) > 1 else touched.pop()] += 1
    return counts


def _band_cuts(tile_xy: Iterable[tuple[int, int]], tile_px: int, block_px: int,
               ratios: dict) -> tuple[int, int, dict[int, int], tuple[int, int]]:
    """Cut the block-x axis into three CONTIGUOUS bands, sized by TILE COUNT.

    SUPERSEDED 2026-08-14 by ``assign_block_folds`` (prereg A20). Retained for
    provenance — it is part of the gen3 band rule, not dead scaffolding.

    Cutting on column index alone skews badly (measured 71/5/24 instead of
    70/15/15) because the outer columns hold far fewer valid tiles. So the columns
    are sorted ascending and weighted by how many tile footprints touch each one.

    Two steps:

    1. Walk the columns accumulating tile counts and note where the cumulative
       fraction first crosses ``train`` and ``train + val`` (``first_crossing``).
    2. Because straddling tiles are then DROPPED, the pre-drop cumulative fraction
       is not the achieved split size — a narrow band loses proportionally far more
       tiles (on the real 10-column Catatumbo grid the first-crossing pair (7, 8)
       yields 85/5/10, not 70/15/15). So the chosen pair is the one whose
       POST-DROP tile fractions are closest to ``ratios`` (min-max deviation,
       ties broken toward more kept tiles then lower columns). Both pairs are
       reported in the index metadata.

    Positions are de-duplicated first, so the bands depend only on the AOI grid
    geometry — the same geographic block lands in the same split every year (LOYO
    stability) regardless of which years were built. Fully deterministic: no RNG.

    Returns (last_train_column, last_val_column, tiles_per_column, first_crossing).
    """
    positions = sorted(set(tile_xy))
    cols = _column_tile_counts(positions, tile_px, block_px)
    if not cols:
        return 0, 0, cols, (0, 0)

    order = sorted(cols)
    total = sum(cols.values())
    cum = 0
    first_train = first_val = None
    for bx in order:
        cum += cols[bx]
        frac = cum / total
        if first_train is None:
            if frac >= ratios["train"]:
                first_train = bx
        elif first_val is None and frac >= ratios["train"] + ratios["val"]:
            first_val = bx
    # Degenerate grids (< 3 columns) never cross a threshold: keep everything in
    # the earlier split rather than raising.
    first_train = order[-1] if first_train is None else first_train
    first_val = order[-1] if first_val is None else first_val
    first_crossing = (first_train, first_val)

    # Candidate pairs: every way to cut the sorted columns into three non-empty
    # contiguous bands. len(order) is ~10, so this is a handful of evaluations.
    best = None
    for i in range(len(order) - 2):
        for j in range(i + 1, len(order) - 1):
            ct, cv = order[i], order[j]
            counts = _band_tile_counts(positions, tile_px, block_px, ct, cv)
            kept = counts["train"] + counts["val"] + counts["test"]
            if not kept:
                continue
            dev = max(abs(counts[s] / kept - ratios[s]) for s in ("train", "val", "test"))
            key = (round(dev, 9), -kept, ct, cv)
            if best is None or key < best[0]:
                best = (key, ct, cv)
    if best is None:                       # < 3 columns: no three-band cut exists
        return first_train, first_val, cols, first_crossing
    return best[1], best[2], cols, first_crossing


def assign_splits(tile_xy: Sequence[tuple[int, int]], tile_px: int, block_px: int,
                  ratios: dict) -> tuple[list[str | None], dict]:
    """Assign tiles to train/val/test by contiguous block-x bands; drop straddlers.

    SUPERSEDED 2026-08-14 by ``assign_block_folds`` (prereg A20). Retained for
    provenance, and still exercised by ``tests/test_split_isolation.py``, because
    it and those tests are the record of the F3 leak fix. It is genuinely
    pixel-disjoint; what it is not is SIGNAL-BEARING — on Catatumbo the coca
    density gradient is steep in BOTH x and y, so the eastern band came out with
    zero positive pixels and every Track A metric on it was 0/0 printed as 0.000.
    New builds must use ``assign_block_folds``.

    ``tile_xy`` is the list of tile top-left pixel corners. Returns
    ``(splits, info)`` where ``splits[i]`` is the split for ``tile_xy[i]`` or
    ``None`` if that tile must be DROPPED because its footprint touches blocks in
    more than one split. Keeping only tiles fully inside one split's blocks is
    what makes the split pixel-disjoint; it does not remove spatial
    autocorrelation across the band boundary, it only widens the gap by ~1 tile.

    Deterministic and order-independent: same input in any order -> same result.
    """
    cut_train, cut_val, cols, first_crossing = _band_cuts(tile_xy, tile_px, block_px, ratios)

    splits: list[str | None] = []
    dropped = 0
    for x, y in tile_xy:
        touched = {_col_split(bx, cut_train, cut_val)
                   for bx, _ in footprint_blocks(x, y, tile_px, block_px)}
        if len(touched) == 1:
            splits.append(touched.pop())
        else:
            splits.append(None)
            dropped += 1

    block_split = {f"{bx}_{by}": _col_split(bx, cut_train, cut_val)
                   for x, y in tile_xy
                   for bx, by in footprint_blocks(x, y, tile_px, block_px)}
    info = {
        "split_rule": SPLIT_RULE,
        "tile_px": tile_px,
        "block_px": block_px,
        "split_ratios_target": dict(ratios),
        "block_columns": len(cols),
        "cut_train_val_col": cut_train,
        "cut_val_test_col": cut_val,
        "cuts_first_crossing": list(first_crossing),
        "tiles_per_block_column": {str(bx): cols[bx] for bx in sorted(cols)},
        "kept": len(splits) - dropped,
        "dropped_straddler": dropped,
        "block_split": dict(sorted(block_split.items())),
        "guarantee": "zero pixel overlap between splits; cross-boundary spatial "
                     "autocorrelation mitigated (~1-tile gap), not eliminated",
    }
    return splits, info


# ---------------------------------------------------------------------------
# gen4 split rule (prereg A20): stratified macro-blocks -> dual-objective folds
# ---------------------------------------------------------------------------


def macro_block_edges(n_positions: int, n_blocks: int) -> list[int]:
    """Macro-block edges in TILE-INDEX space (not pixels).

    Cutting in pixel space would put a macro-block boundary in the middle of a
    tile position and reintroduce the straddler problem at a second granularity.
    Cutting on the tile-index axis makes every macro-block a whole number of
    EXISTING tile positions, so the only thing that ever has to be dropped is a
    tile whose 256 px footprint reaches across a fold boundary.

    Returns ``n_blocks + 1`` ascending edges; block ``b`` owns index positions
    ``[edges[b], edges[b + 1])``.
    """
    import numpy as np

    if n_blocks < 1:
        raise ValueError(f"n_blocks must be >= 1, got {n_blocks}")
    return np.linspace(0, n_positions, n_blocks + 1).round().astype(int).tolist()


def _block_of(edges: Sequence[int], i: int) -> int:
    """Index of the macro-block owning tile-index ``i`` (edges are ascending)."""
    import numpy as np

    return int(np.searchsorted(edges[1:], i, side="right"))


def _overlapping_pairs(positions: Sequence[tuple[int, int]], tile_px: int) -> list[tuple[int, int]]:
    """Every pair of positions whose ``tile_px`` footprints share at least one pixel.

    Two tiles share pixels iff ``abs(dx) < tile_px and abs(dy) < tile_px``.
    """
    return [(a, b)
            for a in range(len(positions))
            for b in range(a + 1, len(positions))
            if abs(positions[a][0] - positions[b][0]) < tile_px
            and abs(positions[a][1] - positions[b][1]) < tile_px]


def _resolve_overlaps(positions: Sequence[tuple[int, int]], folds: Sequence[int],
                      weights: Sequence[int], pairs: Sequence[tuple[int, int]]) -> list[bool]:
    """Greedily drop positions until no two SURVIVING tiles in different folds overlap.

    Repeatedly drop the position involved in the most still-unresolved cross-fold
    conflicts. Ties go to the LOWEST coca weight (dropping an empty tile costs no
    signal), then to the LOWEST ``(x, y)``.

    The ``(x, y)`` direction is not arbitrary and is not merely "for determinism"
    — every total order is deterministic. All six orderings swept for A20
    (``(x, y)`` / ``(y, x)`` / window index, each ascending and descending) clear
    every registered acceptance bar, so this is a tie-break among acceptable
    options rather than a tuned result; ASCENDING ``(x, y)`` is registered because
    it is the best of the six on BOTH registered balance objectives (real grid:
    2.14 pp tile deviation and 2.24 pp coca deviation, against 2.28-2.56 and
    2.85-3.46 for the rest), at a cost of two kept positions. The sweep table is
    in ``docs/BASELINE_PREREG.md`` A20.

    Note that ``positions`` is ``sorted(set(tile_xy))``, so ``(x, y)`` ascending
    and index ascending are the same order here; the WINDOW order emitted by
    ``_windows`` is y-outer/x-inner and is therefore ``(y, x)`` ascending, a
    different ordering that keeps 437 rather than 436.

    Returns a keep flag per position.
    """
    adj: list[list[int]] = [[] for _ in positions]
    for a, b in pairs:
        if folds[a] != folds[b]:
            adj[a].append(b)
            adj[b].append(a)

    alive = [True] * len(positions)
    deg = [len(nb) for nb in adj]
    while any(deg):
        worst = min((k for k in range(len(positions)) if alive[k] and deg[k]),
                    key=lambda k: (-deg[k], weights[k], positions[k]))
        alive[worst] = False
        deg[worst] = 0
        for j in adj[worst]:
            if alive[j]:
                deg[j] -= 1
    return alive


def _gap_px(a: tuple[int, int], b: tuple[int, int], tile_px: int) -> int:
    """L-inf gap in pixels between two ``tile_px`` footprints (0 if they touch/overlap)."""
    return max(max(0, abs(a[0] - b[0]) - tile_px), max(0, abs(a[1] - b[1]) - tile_px))


def assign_block_folds(tile_xy: Sequence[tuple[int, int]], tile_px: int, stride_px: int,
                       grid_shape: tuple[int, int] = (5, 4), n_folds: int = 6,
                       weights: Mapping[tuple[int, int], int] | None = None,
                       ) -> tuple[list[int | None], dict]:
    """Assign tiles to ``n_folds`` STRATIFIED spatial folds (prereg A20, gen4).

    ``tile_xy`` is the list of tile top-left pixel corners; ``grid_shape`` is
    ``(n_blocks_x, n_blocks_y)``; ``weights`` maps a position to its coca weight
    (the max positive-pixel count across years — see
    ``src.data.multiyear.position_coca_weights``). Returns ``(folds, info)`` where
    ``folds[i]`` is the integer ``block_fold`` for ``tile_xy[i]``, or ``None`` if
    that tile is DROPPED because its footprint overlaps a tile in another fold.

    Three steps:

    1. **Macro-blocks.** Cut the sorted-unique x and y tile positions into
       ``grid_shape`` blocks in tile-index space (``macro_block_edges``). Each
       macro-block's weight is the sum of its positions' coca weights.
    2. **Dual-objective greedy.** Sort macro-blocks by descending coca weight and
       give each to the fold currently minimising the JOINT NORMALISED LOAD
       ``coca[f]/total_coca + tiles[f]/total_tiles``. Balancing coca alone (a
       fixed snake) let fold sizes run 27-204 on the real grid, which breaks fold
       rotation for A17; the joint objective keeps both within a few points.
    3. **Overlap resolution.** ``stride_px`` < ``tile_px`` means neighbouring tile
       positions (diagonals included) share ``tile_px - stride_px`` px, so a fold
       boundary that runs between two adjacent positions would put the same pixels
       in two folds. ``_resolve_overlaps`` drops positions until no pixel is
       shared. NO BUFFER is added beyond that, and the resulting separation is
       MEASURED into ``min_cross_fold_gap_px`` rather than assumed: it is one
       dropped lattice position (192 px at the production geometry) only when no
       boundary falls on the row/column ``_windows`` clamps to the raster edge,
       where the step is 91 px in x and the achievable gap drops to 59 px.

    Fully deterministic (no RNG) and order-independent: positions are
    de-duplicated and sorted internally, so the same AOI grid yields the same
    folds in any input order and in any subset of years.
    """
    if not 0 < stride_px <= tile_px:
        raise ValueError(f"need 0 < stride_px <= tile_px, got {stride_px} / {tile_px}")
    if n_folds < 2:
        raise ValueError(f"n_folds must be >= 2, got {n_folds}")

    positions = sorted(set(tile_xy))
    w = {p: int((weights or {}).get(p, 0)) for p in positions}

    xs = sorted({x for x, _ in positions})
    ys = sorted({y for _, y in positions})
    ix = {x: i for i, x in enumerate(xs)}
    iy = {y: i for i, y in enumerate(ys)}
    n_bx, n_by = grid_shape
    edges_x = macro_block_edges(len(xs), n_bx)
    edges_y = macro_block_edges(len(ys), n_by)

    block_of_pos = {p: (_block_of(edges_x, ix[p[0]]), _block_of(edges_y, iy[p[1]]))
                    for p in positions}
    members: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for p in positions:
        members.setdefault(block_of_pos[p], []).append(p)

    blk_w = {b: sum(w[p] for p in ps) for b, ps in members.items()}
    blk_n = {b: len(ps) for b, ps in members.items()}
    total_w = sum(blk_w.values()) or 1        # all-zero weights -> balance tiles only
    total_n = sum(blk_n.values())

    # Least-loaded (LPT) greedy on the joint normalised load. Heaviest block first
    # so the coarsest decision is made while every fold is still empty.
    load_w = [0] * n_folds
    load_n = [0] * n_folds
    block_fold: dict[tuple[int, int], int] = {}
    for b in sorted(members, key=lambda b: (-blk_w[b], b)):
        f = min(range(n_folds), key=lambda i: (load_w[i] / total_w + load_n[i] / total_n, i))
        block_fold[b] = f
        load_w[f] += blk_w[b]
        load_n[f] += blk_n[b]

    fold_of_pos = [block_fold[block_of_pos[p]] for p in positions]
    pairs = _overlapping_pairs(positions, tile_px)
    alive = _resolve_overlaps(positions, fold_of_pos, [w[p] for p in positions], pairs)

    kept_idx = [i for i, ok in enumerate(alive) if ok]
    kept_w = [0] * n_folds
    kept_n = [0] * n_folds
    for i in kept_idx:
        kept_w[fold_of_pos[i]] += w[positions[i]]
        kept_n[fold_of_pos[i]] += 1
    tot_w = sum(kept_w) or 1
    tot_n = sum(kept_n) or 1

    # Separation actually achieved, and how much of the split sits against it.
    # Disclosed rather than assumed: stratifying buys measurability by putting
    # MORE tiles near a fold boundary than a 3-band split does.
    unresolved = sum(1 for a, b in pairs
                     if fold_of_pos[a] != fold_of_pos[b] and alive[a] and alive[b])
    cross = [(_gap_px(positions[a], positions[b], tile_px), a, b)
             for a in kept_idx for b in kept_idx
             if a < b and fold_of_pos[a] != fold_of_pos[b]
             and abs(positions[a][0] - positions[b][0]) <= 2 * tile_px
             and abs(positions[a][1] - positions[b][1]) <= 2 * tile_px]
    min_gap = min((g for g, _, _ in cross), default=None)
    at_min = sorted({i for g, a, b in cross if g == min_gap for i in (a, b)})

    fold_by_pos = dict(zip(positions, fold_of_pos))
    keep_by_pos = dict(zip(positions, alive))
    folds: list[int | None] = [fold_by_pos[p] if keep_by_pos[p] else None for p in tile_xy]

    info = {
        "split_rule": SPLIT_RULE_BLOCKS,
        "tile_px": tile_px,
        "stride_px": stride_px,
        "overlap_px": tile_px - stride_px,
        "grid_shape": [n_bx, n_by],
        "n_folds": n_folds,
        "n_positions": len(positions),
        "n_positions_x": len(xs),
        "n_positions_y": len(ys),
        "block_edges_x_tileidx": edges_x,
        "block_edges_y_tileidx": edges_y,
        "block_tiles": {f"{bx}_{by}": blk_n[(bx, by)] for bx, by in sorted(members)},
        "block_coca_weight": {f"{bx}_{by}": blk_w[(bx, by)] for bx, by in sorted(members)},
        "block_fold": {f"{bx}_{by}": block_fold[(bx, by)] for bx, by in sorted(members)},
        "total_coca_weight": sum(blk_w.values()),
        "kept": len(kept_idx),
        "dropped_overlap": len(positions) - len(kept_idx),
        "kept_fraction": len(kept_idx) / len(positions) if positions else 0.0,
        "tiles_by_block_fold": kept_n,
        "coca_weight_by_block_fold": kept_w,
        "tile_share_by_block_fold": [n / tot_n for n in kept_n],
        "coca_share_by_block_fold": [v / tot_w for v in kept_w],
        "max_tile_share_dev_pp": 100 * max(abs(n / tot_n - 1 / n_folds) for n in kept_n),
        "max_coca_share_dev_pp": 100 * max(abs(v / tot_w - 1 / n_folds) for v in kept_w),
        "unresolved_cross_fold_overlaps": unresolved,
        "min_cross_fold_gap_px": min_gap,
        "tiles_at_min_gap": len(at_min),
        "frac_tiles_at_min_gap": len(at_min) / len(kept_idx) if kept_idx else 0.0,
        "guarantee": "zero pixel overlap between block folds (and therefore between "
                     "splits); no buffer beyond the one dropped tile position, so "
                     "cross-boundary spatial autocorrelation is mitigated, not eliminated",
    }
    return folds, info


def split_for_block_fold(block_fold: int, n_folds: int = 6, rotation: int = 0) -> str:
    """Roll a ``block_fold`` up to a pipeline split: ``n_folds - 2`` train, 1 val, 1 test.

    With ``rotation=0`` the last fold is ``test`` and the second-to-last is
    ``val``. ``rotation`` shifts which fold plays which role, so the same recorded
    ``block_fold`` column supports out-of-fold inference over the WHOLE AOI (A17
    municipal ranking) by re-running with each rotation — which is the reason the
    fold id is stored per tile rather than only the rolled-up split.
    """
    pos = (block_fold - rotation) % n_folds
    if pos == n_folds - 1:
        return "test"
    if pos == n_folds - 2:
        return "val"
    return "train"


SPLITS = ("train", "val", "test")


def split_pixel_report(tile_xy: Sequence[tuple[int, int]], splits: Sequence[str | None],
                       block_folds: Sequence[int | None], years: Sequence[int],
                       mask_loader: Callable[[int, int, int], object],
                       canvas_shape: tuple[int, int], tile_px: int) -> dict:
    """MEASURE the split's pixel-level facts. Asserts nothing — see ``verify_split_pixels``.

    Measurement and adjudication are separate so a caller (notably the test suite)
    can assert one invariant at a time and surface its own message, instead of
    every invariant collapsing into whichever guard happens to raise first.

    ``mask_loader(year, x, y)`` returns that tile's label mask; anything > 0 is a
    positive pixel. Because tiles OVERLAP by ``tile_px - stride_px``, a positive
    count summed over tiles double-counts the seams — every headline count here is
    therefore over the UNION of pixels a split covers, on a canvas of
    ``canvas_shape`` ``(height, width)``. The per-tile sum is reported alongside
    only so the size of the double-count is visible.

    Two different quantities in this report are both a "coca total" and they must
    not be confused:

    * ``union_positive_px[group][year]`` — pixels, per year, over the union. The
      honest count of how much coca a group actually contains.
    * ``coca_weight[group]`` — sum over the group's positions of the MAX
      positive-pixel count across years, i.e. the quantity
      ``assign_block_folds`` balances and the one the A20 balance bar is stated
      on. It is a per-tile sum, so it includes the seam double-count by
      construction; that is deliberate, because it is a weighting, not a census.
    """
    import numpy as np

    h, w = canvas_shape
    kept = [(x, y, s, f) for (x, y), s, f in zip(tile_xy, splits, block_folds) if s is not None]
    if not kept:
        raise ValueError("no kept tiles — nothing to measure")

    fold_ids = sorted({f for _, _, _, f in kept})
    groups: dict[str, object] = {}
    for name in SPLITS:
        groups[name] = np.zeros((h, w), dtype=bool)
    for f in fold_ids:
        groups[f"{BLOCK_FOLD_KEY}={f}"] = np.zeros((h, w), dtype=bool)

    for x, y, s, f in kept:
        groups[s][y:y + tile_px, x:x + tile_px] = True
        groups[f"{BLOCK_FOLD_KEY}={f}"][y:y + tile_px, x:x + tile_px] = True

    # --- 4. pixel intersections: strictly zero, both between folds and splits ---
    intersections: dict[str, int] = {}
    for i, a in enumerate(SPLITS):
        for b in SPLITS[i + 1:]:
            intersections[f"{a}&{b}"] = int((groups[a] & groups[b]).sum())
    fold_intersections: dict[str, int] = {}
    for i, fa in enumerate(fold_ids):
        for fb in fold_ids[i + 1:]:
            fold_intersections[f"{fa}&{fb}"] = int(
                (groups[f"{BLOCK_FOLD_KEY}={fa}"] & groups[f"{BLOCK_FOLD_KEY}={fb}"]).sum())

    # --- 1-3. per group: tiles, tiles with coca, union positives, per-tile sum ---
    names = list(groups)
    n_tiles = {g: 0 for g in names}
    for _, _, s, f in kept:
        n_tiles[s] += 1
        n_tiles[f"{BLOCK_FOLD_KEY}={f}"] += 1
    union_pos = {g: {y: 0 for y in years} for g in names}
    sum_pos = {g: {y: 0 for y in years} for g in names}
    tiles_pos = {g: {y: 0 for y in years} for g in names}
    per_tile_max = {(x, y): 0 for x, y, _, _ in kept}

    for year in years:
        pos = np.zeros((h, w), dtype=bool)
        for x, y, s, f in kept:
            m = np.asarray(mask_loader(year, x, y)) > 0
            pos[y:y + tile_px, x:x + tile_px] |= m
            n = int(m.sum())
            per_tile_max[(x, y)] = max(per_tile_max[(x, y)], n)
            for g in (s, f"{BLOCK_FOLD_KEY}={f}"):
                sum_pos[g][year] += n
                tiles_pos[g][year] += int(n > 0)
        for g in names:
            union_pos[g][year] = int((groups[g] & pos).sum())

    coca_weight = {g: 0 for g in names}
    for x, y, s, f in kept:
        for g in (s, f"{BLOCK_FOLD_KEY}={f}"):
            coca_weight[g] += per_tile_max[(x, y)]

    report = {
        "canvas_shape": [h, w],
        "tile_px": tile_px,
        "years": list(years),
        "groups": names,
        "n_tiles": n_tiles,
        "tiles_with_coca": {g: dict(tiles_pos[g]) for g in names},
        "union_positive_px": {g: dict(union_pos[g]) for g in names},
        "sum_over_tiles_positive_px": {g: dict(sum_pos[g]) for g in names},
        "coca_weight": coca_weight,
        "split_pixel_intersections": intersections,
        "block_fold_pixel_intersections": fold_intersections,
    }
    return report


def verify_split_pixels(tile_xy: Sequence[tuple[int, int]], splits: Sequence[str | None],
                        block_folds: Sequence[int | None], years: Sequence[int],
                        mask_loader: Callable[[int, int, int], object],
                        canvas_shape: tuple[int, int], tile_px: int,
                        verbose: bool = True) -> dict:
    """PROVE the split is both leak-free and signal-bearing; return the report.

    Raises ``AssertionError`` (never a warning, never a printed 0.000) if:

    * any pixel belongs to two different block folds, or to two different splits
      — the F3 leak; or
    * any split OR any block fold has ZERO positive pixels in any year — the B1
      blocker, which is what let an empty test fold ship and turned every
      ``presence_iou`` on it into ``0/0`` reported as ``0.000``. Folds are checked
      as well as splits because the folds get ROTATED for out-of-fold A17
      inference, so each one has to be measurable on its own.
    """
    report = split_pixel_report(tile_xy, splits, block_folds, years, mask_loader,
                                canvas_shape, tile_px)
    years = report["years"]
    names = report["groups"]
    union_pos = report["union_positive_px"]

    if verbose:
        _print_split_report(report)

    bad = {k: v for k, v in {**report["split_pixel_intersections"],
                             **report["block_fold_pixel_intersections"]}.items() if v}
    if bad:
        raise AssertionError(f"splits/folds are NOT pixel-disjoint: {bad}")
    empty = [(g, y) for g in names for y in years if union_pos[g][y] == 0]
    if empty:
        raise AssertionError(
            "group has ZERO positive pixels in a year: "
            + ", ".join(f"{g}/{y}" for g, y in empty)
            + " — this is the gen3 B1 blocker; every presence_iou on it would be "
              "0/0 printed as 0.000")
    return report


def _print_split_report(report: dict) -> None:
    """Print the per-split AND per-fold verification table."""
    years = report["years"]
    print(f"[verify] canvas={report['canvas_shape'][1]}x{report['canvas_shape'][0]} px  "
          f"tile={report['tile_px']} px  years={years}")
    print(f"[verify] {'group':<14}{'tiles':>7}{'tiles>0':>9}{'union_pos_px':>15}"
          f"{'sum_over_tiles':>16}{'seam_dbl':>10}{'coca_weight':>14}")
    for g in report["groups"]:
        u = sum(report["union_positive_px"][g].values())
        s = sum(report["sum_over_tiles_positive_px"][g].values())
        t = sum(report["tiles_with_coca"][g].values()) / max(len(years), 1)
        print(f"[verify] {g:<14}{report['n_tiles'][g]:>7}{t:>9.1f}{u:>15,}{s:>16,}"
              f"{(s / u if u else 0):>9.2f}x{report['coca_weight'][g]:>14,}")
    print(f"[verify] union positive px per year ({'  '.join(str(y) for y in years)}):")
    for g in report["groups"]:
        print(f"[verify] {g:<14}" + "".join(f"{report['union_positive_px'][g][y]:>12,}"
                                            for y in years))
    print("[verify] pixel intersections (must all be 0): "
          + "  ".join(f"{k}={v}" for k, v in report["split_pixel_intersections"].items()))
    print("[verify] block_fold pixel intersections (must all be 0): "
          + "  ".join(f"{k}={v}" for k, v in report["block_fold_pixel_intersections"].items()))


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
            # Representative block id (top-left pixel); the SPLIT uses the full
            # footprint via assign_splits, not this single block.
            pos_frac = float((m > 0).mean())
            records.append({"x": x, "y": y, "block": block_id(x, y, block_px),
                            "pos_frac": pos_frac, "stack": stack, "mask": m})

    if not records:
        raise RuntimeError("No valid tiles produced (all nodata?).")

    splits, info = assign_splits([(r["x"], r["y"]) for r in records], tpx, block_px,
                                 tc["split_ratios"])

    index_path = out_dir / "tiles_index.csv"
    counts = {"train": 0, "val": 0, "test": 0}
    with open(index_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["tile_id", "split", "block", "x", "y", "pos_frac", "npz"])
        for i, (r, split) in enumerate(zip(records, splits)):
            if split is None:            # footprint straddles two splits -> dropped
                continue
            counts[split] += 1
            tile_id = f"tile_{i:05d}"
            npz = out_dir / f"{tile_id}.npz"
            np.savez_compressed(npz, image=r["stack"], mask=r["mask"])
            w.writerow([tile_id, split, r["block"], r["x"], r["y"],
                        f"{r['pos_frac']:.4f}", npz.name])

    n_blocks = len(set(r["block"] for r in records))
    print(f"[tiling] tiles={len(records)}  blocks={n_blocks}  block_px={block_px}")
    print(f"[tiling] split rule={info['split_rule']} "
          f"cuts: train<=x{info['cut_train_val_col']} val<=x{info['cut_val_test_col']}")
    print(f"[tiling] kept={info['kept']} dropped_straddler={info['dropped_straddler']}")
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
