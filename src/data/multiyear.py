"""v2.1 Phase 1 — build a pooled multi-year dataset (2019–2024).

For each year: rasterize the official coca fraction label aligned to that year's
imagery, tile the stack+mask, and append the tiles to a master index tagged with
the year. The spatial split is reused unchanged across years (it is a function of
pixel position on the identical AOI grid only), so a given geographic block stays
in the same fold every year — clean leave-one-year-out folds.

gen4 (prereg A20) splits by STRATIFIED MACRO-BLOCKS: a 5x4 grid cut in tile-index
space, greedily assigned to 6 spatial folds balancing coca AND tile count, then
rolled up 4/1/1 to train/val/test (``src.data.tiling.assign_block_folds``). Tiles
whose 256 px footprints overlap across a fold boundary are dropped, which yields
ZERO PIXEL OVERLAP between folds and therefore between splits. It does NOT make
the split "leakage-free": the dropped position leaves a gap of 192 px (3.8 km) at
each boundary on the regular lattice — measured, not guaranteed, since ``_windows``
clamps the final row/column — and coca autocorrelates beyond 10 km, so
cross-boundary similarity is mitigated, not removed.

gen3 used contiguous west->east bands (``assign_splits``); that rule is
SUPERSEDED — it was pixel-disjoint but its eastern band held zero coca, so every
Track A metric on it was 0/0.

``build_all`` (the re-tiling path) STILL WRITES A BAND SPLIT, so on its own it
would silently revert a gen4 index to the defect A20 exists to prevent — and it
would do so while ``config`` still stamps ``gen4`` onto every metrics row. It
therefore now REFUSES to overwrite a block-fold index unless ``force=True``, and
under ``force`` it archives the superseded pair and ends by telling you to run
the reassignment. **``--rebuild-block-folds`` is the terminal step of any rebuild**;
it reassigns the EXISTING tiles at zero re-tiling cost and verifies the result.

    # full rebuild: re-tile, THEN reassign. The second command is not optional.
    python -m src.data.multiyear --years 2019 2020 2021 2022 2023 2024 --force-band-split-index
    python -m src.data.multiyear --rebuild-block-folds

Requires each year's imagery at data/imagery/<region>_<year>_annual_full.tif.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import rasterio

from src.data.labels import fetch_coca_grid, rasterize_mask
from src.data.tiling import (
    BLOCK_FOLD_KEY,
    _windows,
    assign_block_folds,
    assign_splits,
    block_id,
    block_px_from_cfg,
    split_for_block_fold,
    verify_split_pixels,
)
from src.utils import ensure_dirs, load_config

# Official coca-grid field per year (naming is irregular on Socrata v3rx-q7t3).
GRID_FIELD = {2019: "areacoca_2019", 2020: "areacoca_2020", 2021: "areacoca_2021",
              2022: "coca2022_", 2023: "areacoca2023", 2024: "areacoca2024"}

YEARS = [2019, 2020, 2021, 2022, 2023, 2024]

INDEX_NAME = "multiyear_index.csv"
# Sidecar: split rule + kept/dropped bookkeeping + the full A20 design. The index
# CSV columns are otherwise NOT touched (downstream readers depend on them).
META_NAME = "multiyear_index_meta.json"
# gen4 rebuild parameters (prereg A20).
GRID_SHAPE = (5, 4)      # macro-blocks along x, along y
N_BLOCK_FOLDS = 6

# Annotate, never overwrite: a superseded index/sidecar pair is copied to
# ``multiyear_index_<tag>.csv`` before being replaced. New sidecars carry their
# own ``archive_tag``; this map only has to name the ones written before the key
# existed, and must never be edited — it is the record of what was replaced.
ARCHIVE_TAG = "gen4_xy_asc"
LEGACY_ARCHIVE_TAG = {
    "contiguous_x_bands_strict_footprint_v1": "gen3_bands",
    "stratified_macro_blocks_dual_objective_folds_v1": "gen4_xy_desc",
}


def _year_cfg(cfg: dict, year: int) -> dict:
    """Config view with year + its irregular grid field."""
    return {**cfg, "year": year,
            "labels": {**cfg["labels"], "coca_grid_year_field": GRID_FIELD[year]}}


def build_year(cfg: dict, year: int) -> list[dict]:
    """Rasterize the year's label mask and tile it; return tile records (year-tagged)."""
    region = cfg["aoi"]["region"]
    img_path = f"data/imagery/{region}_{year}_annual_full.tif"
    if not Path(img_path).exists():
        raise FileNotFoundError(f"Missing imagery for {year}: {img_path} (export it first).")

    ycfg = _year_cfg(cfg, year)
    gdf = fetch_coca_grid(ycfg, tuple(cfg["aoi"]["bbox"]))
    mask_path = rasterize_mask(ycfg, gdf, img_path)

    tpx = cfg["tiling"]["tile_px"]
    stride = cfg["tiling"]["stride_px"]
    block_px = int(round(cfg["tiling"]["split_block_km"] * 1000 / cfg["imagery"]["resolution_m"]))
    out_dir = Path(cfg["paths"]["tiles_dir"]) / str(year)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    with rasterio.open(img_path) as img, rasterio.open(mask_path) as msk:
        for (x, y) in _windows(img.width, img.height, tpx, stride):
            win = rasterio.windows.Window(x, y, tpx, tpx)
            stack = img.read(window=win).astype("float32")
            m = msk.read(1, window=win)
            if stack.shape[1:] != (tpx, tpx):
                continue
            if np.isfinite(stack).all(axis=0).mean() < 0.5:
                continue
            # Representative block (top-left pixel). The SPLIT is decided later
            # from the tile's whole footprint, not from this single block.
            block = block_id(x, y, block_px)
            i = len(records)
            npz = out_dir / f"tile_{i:05d}.npz"
            np.savez_compressed(npz, image=stack, mask=m)
            records.append({"tile_id": f"{year}_{i:05d}", "year": year, "block": block,
                            "x": x, "y": y, "pos_frac": float((m > 0).mean()),
                            "npz": f"{year}/{npz.name}"})
    print(f"[multiyear] {year}: {len(records)} tiles")
    return records


def _read_meta(tiles_dir: Path) -> dict:
    """The existing index sidecar, or ``{}`` if there is none."""
    path = tiles_dir / META_NAME
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _archive_superseded_index(tiles_dir: Path, new_rule: str) -> str | None:
    """Copy the existing index + sidecar aside before a DIFFERENT rule replaces them.

    Annotate, never overwrite. Nothing is archived when the rule is unchanged (a
    rerun of the same rule produces the same file), and an existing archive is
    never clobbered — the first copy of a superseded artifact is the one that
    matters. Returns the tag used, or None if nothing was archived.
    """
    old = _read_meta(tiles_dir)
    old_rule = old.get("split_rule")
    if not old_rule or old_rule == new_rule:
        return None
    tag = old.get("archive_tag") or LEGACY_ARCHIVE_TAG.get(old_rule)
    if tag is None:
        raise RuntimeError(
            f"refusing to replace an index built under unknown rule {old_rule!r}: it "
            f"carries no archive_tag and is not in LEGACY_ARCHIVE_TAG. Add it there "
            f"(or move the index aside by hand) rather than losing the artifact.")
    for src_name, dst_name in ((INDEX_NAME, f"multiyear_index_{tag}.csv"),
                               (META_NAME, f"multiyear_index_meta_{tag}.json")):
        src_path, dst = tiles_dir / src_name, tiles_dir / dst_name
        if src_path.exists() and not dst.exists():
            dst.write_bytes(src_path.read_bytes())
            print(f"[multiyear] archived superseded artifact -> {dst}")
    return tag


def build_all(cfg: dict, years: list[int], force: bool = False) -> Path:
    """Re-tile every year and write a BAND-SPLIT index (the superseded gen3 rule).

    This is the re-tiling path, and it is **not** a complete rebuild. It writes
    ``split`` using ``assign_splits``, which on this AOI yields a test fold with
    zero coca (blocker B1) — while ``project.data_generation`` still stamps
    ``gen4`` onto every metrics row, which is exactly how a false 6/6 gets
    manufactured. It therefore refuses to overwrite a block-fold index unless
    ``force=True``, and always ends by telling you to run
    ``rebuild_block_fold_index``, which is the terminal step of any rebuild.
    """
    ensure_dirs(cfg)
    tiles_dir = Path(cfg["paths"]["tiles_dir"])
    old_meta = _read_meta(tiles_dir)
    old_rule = old_meta.get("split_rule")
    # ``n_folds`` is written only by rebuild_block_fold_index, so this recognises
    # ANY version of the block-fold rule — not just the one this build ships with,
    # which would let a v1 index be clobbered silently after a rule bump.
    if old_meta.get("n_folds") is not None and not force:
        raise RuntimeError(
            f"{tiles_dir / INDEX_NAME} was built under {old_rule!r} (prereg A20) and "
            f"build_all would replace it with the SUPERSEDED band split, whose test "
            f"fold holds zero coca — while config still stamps "
            f"{cfg['project']['data_generation']!r} on every metrics row. Re-tile only "
            f"if you mean to: pass force=True (--force-band-split-index), then run "
            f"`python -m src.data.multiyear --rebuild-block-folds` to reassign the "
            f"folds and verify them.")

    all_recs = []
    for y in years:
        all_recs.extend(build_year(cfg, y))

    # Assign spatial blocks to splits ONCE (same across years) for clean LOYO, in
    # contiguous block-x bands; tiles straddling two splits are dropped.
    tpx = cfg["tiling"]["tile_px"]
    block_px = int(round(cfg["tiling"]["split_block_km"] * 1000 / cfg["imagery"]["resolution_m"]))
    splits, info = assign_splits([(r["x"], r["y"]) for r in all_recs], tpx, block_px,
                                 cfg["tiling"]["split_ratios"])

    _archive_superseded_index(tiles_dir, info["split_rule"])
    index_path = tiles_dir / INDEX_NAME
    counts: dict[tuple[int, str], int] = {}
    dropped: dict[int, int] = {y: 0 for y in years}
    n_written = 0
    with open(index_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["tile_id", "year", "split", "block", "x", "y", "pos_frac", "npz"])
        for r, split in zip(all_recs, splits):
            if split is None:            # footprint straddles two splits -> dropped
                dropped[r["year"]] = dropped.get(r["year"], 0) + 1
                continue
            counts[(r["year"], split)] = counts.get((r["year"], split), 0) + 1
            n_written += 1
            w.writerow([r["tile_id"], r["year"], split, r["block"], r["x"], r["y"],
                        f"{r['pos_frac']:.4f}", r["npz"]])

    print(f"[multiyear] wrote {index_path}  ({n_written} tiles kept of {len(all_recs)})")
    print(f"[multiyear] split rule={info['split_rule']}  block_px={block_px}  "
          f"cut train|val at block-x {info['cut_train_val_col']}, "
          f"val|test at block-x {info['cut_val_test_col']} "
          f"(of {info['block_columns']} columns)")
    for y in years:
        kept_y = sum(counts.get((y, s), 0) for s in ("train", "val", "test"))
        print(f"           {y}: "
              + " ".join(f"{s}={counts.get((y, s), 0)}" for s in ("train", "val", "test"))
              + f"  kept={kept_y} dropped_straddler={dropped.get(y, 0)}")

    meta = {**info,
            "archive_tag": LEGACY_ARCHIVE_TAG[info["split_rule"]],
            "index_csv": INDEX_NAME,
            "years": list(years),
            "tiles_seen": len(all_recs),
            "tiles_kept": n_written,
            "verified": False,
            "requires_block_fold_rebuild": True,
            "dropped_straddler_by_year": {str(y): dropped.get(y, 0) for y in years},
            "kept_by_year_split": {f"{y}_{s}": counts.get((y, s), 0)
                                   for y in years for s in ("train", "val", "test")}}
    meta_path = Path(cfg["paths"]["tiles_dir"]) / META_NAME
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[multiyear] wrote {meta_path}")
    print("[multiyear] *** THIS INDEX IS A SUPERSEDED BAND SPLIT AND IS NOT USABLE. ***")
    print("[multiyear] *** Its test fold holds zero coca (blocker B1). Run now:      ***")
    print("[multiyear] ***   python -m src.data.multiyear --rebuild-block-folds      ***")
    return index_path


# ---------------------------------------------------------------------------
# gen4 (prereg A20) — reassign the EXISTING tiles, no re-tiling
# ---------------------------------------------------------------------------


def _grid_geometry(cfg: dict, years: list[int]) -> tuple[int, int]:
    """(width, height) of the annual composites, asserted identical across years.

    Tile position ``(x, y)`` means "the same ground in every year" only because
    all six composites share one origin, shape and transform. That is what makes
    a single spatial fold assignment valid for all years, so it is checked rather
    than assumed — this repo has a live history of a re-exported year landing on a
    different origin (``UNALIGNED_catatumbo_2023_annual_full.tif``).
    """
    region = cfg["aoi"]["region"]
    ref = None
    for year in years:
        path = Path(f"data/imagery/{region}_{year}_annual_full.tif")
        if not path.exists():
            raise FileNotFoundError(f"Missing imagery for {year}: {path}")
        with rasterio.open(path) as src:
            got = (src.width, src.height, tuple(src.transform)[:6], str(src.crs))
        if ref is None:
            ref = (year, got)
        elif got != ref[1]:
            raise AssertionError(
                f"{year} grid {got} differs from {ref[0]} {ref[1]} — the same (x, y) "
                "would be different ground in different years")
    return ref[1][0], ref[1][1]


def position_coca_weights(cfg: dict, years: list[int], positions: list[tuple[int, int]],
                          ) -> tuple[dict[tuple[int, int], int], dict[int, list[int]]]:
    """Exact positive-pixel counts per (year, position), recounted from the npz masks.

    ``pos_frac`` in the index CSV is rounded to 4 decimals, which is not exact
    enough to weight a macro-block; every count here comes from the mask array.
    Returns ``(weights, counts_by_year)`` where a position's WEIGHT is the MAX of
    its positive-pixel counts across years — a position that ever held coca is
    worth balancing, and the max is invariant to which years are built.
    """
    tiles_dir = Path(cfg["paths"]["tiles_dir"])
    counts: dict[int, list[int]] = {}
    for year in years:
        per_year = []
        for k in range(len(positions)):
            npz = tiles_dir / str(year) / f"tile_{k:05d}.npz"
            if not npz.exists():
                raise FileNotFoundError(f"{npz} missing — cannot rebuild without re-tiling")
            with np.load(npz) as z:
                per_year.append(int((z["mask"] > 0).sum()))
        counts[year] = per_year
    weights = {p: max(counts[y][k] for y in years) for k, p in enumerate(positions)}
    return weights, counts


def rebuild_block_fold_index(cfg: dict, years: list[int] | None = None,
                             grid_shape: tuple[int, int] = GRID_SHAPE,
                             n_folds: int = N_BLOCK_FOLDS, rotation: int = 0) -> Path:
    """Rebuild ``multiyear_index.csv`` under the gen4 A20 rule WITHOUT re-tiling.

    The gen3 index dropped 75 of the 575 tile positions as band straddlers, but
    ``data/tiles/<year>/tile_{k:05d}.npz`` exists for ALL 575 (``k`` is the index
    into ``tiling._windows``), so those positions are reclaimable for free. This
    walks the window order, recounts every mask exactly, assigns the A20 macro-block
    folds, rolls them up to train/val/test, rewrites the index (adding only the
    ``block_fold`` column — every existing column keeps its name and meaning), and
    then RUNS THE VERIFICATION, which raises rather than returns if the result is
    not both pixel-disjoint and signal-bearing in every year.
    """
    years = list(years or YEARS)
    tiles_dir = Path(cfg["paths"]["tiles_dir"])
    tpx = cfg["tiling"]["tile_px"]
    stride = cfg["tiling"]["stride_px"]
    block_px = block_px_from_cfg(cfg)

    width, height = _grid_geometry(cfg, years)
    positions = list(_windows(width, height, tpx, stride))
    print(f"[gen4] grid {width}x{height} px -> {len(positions)} tile positions "
          f"({len({x for x, _ in positions})} x {len({y for _, y in positions})})")

    weights, counts = position_coca_weights(cfg, years, positions)
    folds, info = assign_block_folds(positions, tpx, stride, grid_shape, n_folds, weights)
    splits = [None if f is None else split_for_block_fold(f, n_folds, rotation) for f in folds]

    print(f"[gen4] rule={info['split_rule']}  grid={info['grid_shape']}  folds={n_folds}")
    print(f"[gen4] block edges (tile index) x={info['block_edges_x_tileidx']} "
          f"y={info['block_edges_y_tileidx']}")
    print(f"[gen4] kept={info['kept']}/{info['n_positions']} "
          f"({info['kept_fraction']:.1%})  dropped_overlap={info['dropped_overlap']}")
    print(f"[gen4] tiles per {BLOCK_FOLD_KEY}: {info['tiles_by_block_fold']}  "
          f"(max dev {info['max_tile_share_dev_pp']:.2f} pp)")
    print(f"[gen4] coca share per {BLOCK_FOLD_KEY}: "
          + " ".join(f"{v:.3f}" for v in info["coca_share_by_block_fold"])
          + f"  (max dev {info['max_coca_share_dev_pp']:.2f} pp)")
    print(f"[gen4] min cross-fold gap={info['min_cross_fold_gap_px']} px; "
          f"{info['tiles_at_min_gap']} kept tiles ({info['frac_tiles_at_min_gap']:.1%}) "
          "sit against a fold boundary")

    # Read the outgoing sidecar BEFORE it is replaced, so a rerun of the same rule
    # (which archives nothing) carries its provenance forward instead of erasing it.
    previous_supersedes = _read_meta(tiles_dir).get("supersedes")
    archived = _archive_superseded_index(tiles_dir, info["split_rule"])
    index_path = tiles_dir / INDEX_NAME
    meta_path = tiles_dir / META_NAME

    counts_by: dict[tuple[int, str], int] = {}
    n_written = 0
    with open(index_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["tile_id", "year", "split", "block", "x", "y", "pos_frac", "npz",
                    BLOCK_FOLD_KEY])
        for year in years:
            for k, (x, y) in enumerate(positions):
                if splits[k] is None:
                    continue
                counts_by[(year, splits[k])] = counts_by.get((year, splits[k]), 0) + 1
                n_written += 1
                w.writerow([f"{year}_{k:05d}", year, splits[k], block_id(x, y, block_px), x, y,
                            f"{counts[year][k] / (tpx * tpx):.4f}",
                            f"{year}/tile_{k:05d}.npz", folds[k]])
    print(f"[gen4] wrote {index_path}  ({n_written} rows = "
          f"{info['kept']} positions x {len(years)} years)")

    index_of = {p: k for k, p in enumerate(positions)}

    def mask_loader(year: int, x: int, y: int):
        with np.load(tiles_dir / str(year) / f"tile_{index_of[(x, y)]:05d}.npz") as z:
            return z["mask"]

    report = verify_split_pixels(positions, splits, folds, years, mask_loader,
                                 (height, width), tpx)

    meta = {**info,
            "archive_tag": ARCHIVE_TAG,
            "data_generation": cfg["project"]["data_generation"],
            "index_csv": INDEX_NAME,
            "index_columns": ["tile_id", "year", "split", "block", "x", "y", "pos_frac",
                              "npz", BLOCK_FOLD_KEY],
            "block_px": block_px,
            "years": list(years),
            "rebuilt_from_existing_tiles": True,
            "verified": True,
            "canvas_px": [width, height],
            "coca_share_basis": "sum over kept positions of the MAX positive-pixel count "
                                "across years (the quantity assign_block_folds balances); "
                                "NOT the per-year union positive-pixel count",
            "fold_rollup": {"rotation": rotation,
                            "n_train_folds": n_folds - 2, "n_val_folds": 1, "n_test_folds": 1,
                            "split_by_block_fold": {str(f): split_for_block_fold(f, n_folds,
                                                                                 rotation)
                                                    for f in range(n_folds)}},
            "kept_by_year_split": {f"{y}_{s}": counts_by.get((y, s), 0)
                                   for y in years for s in ("train", "val", "test")},
            "verification": report,
            "supersedes": ({"archived_as": archived,
                            "index_csv": f"multiyear_index_{archived}.csv",
                            "reason": "gen3 bands: eastern band held zero coca (blocker B1); "
                                      "gen4 v1: drop tie-break was descending (x, y), which is "
                                      "worse on both registered balance objectives than v2's "
                                      "ascending order"}
                           if archived else previous_supersedes)}
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[gen4] wrote {meta_path}")
    return index_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build pooled multi-year dataset (v2.1 Phase 1).")
    ap.add_argument("--config", default=None)
    ap.add_argument("--years", type=int, nargs="+", default=YEARS)
    ap.add_argument("--rebuild-block-folds", action="store_true",
                    help="gen4/A20: reassign the EXISTING tiles to stratified macro-block "
                         "folds and rewrite the index. Does not re-tile. THE TERMINAL STEP "
                         "of any rebuild.")
    ap.add_argument("--force-band-split-index", action="store_true",
                    help="allow re-tiling to replace a gen4 block-fold index with the "
                         "SUPERSEDED band split. Archives the gen4 pair first; you must "
                         "then run --rebuild-block-folds.")
    args = ap.parse_args()
    if args.rebuild_block_folds:
        rebuild_block_fold_index(load_config(args.config), args.years)
    else:
        build_all(load_config(args.config), args.years, force=args.force_band_split_index)
