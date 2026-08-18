"""A17 — municipal ranking on out-of-fold predictions (prereg A17 + A22).

The second headline deliverable: does the model rank municipalities by coca the way the
official census does, and does it do so better than simply **reusing last year's census
ranking**? A16 already showed the previous census is a strong null spatially; A17 asks the
same question of the ranking.

**What A22 settled and why this module exists.** `src/infer.py:municipal_hectares`
aggregates a full-AOI prediction, ~70% of which is ground the model trained on, so a
ranking built that way is partly a memory test. Here every pixel that enters the ranking is
predicted by a model that never saw it:

* six models, one per A20 rotation (`tiling.split_for_block_fold(block_fold, 6, r)`), each
  predicting **only its own `test` fold**. The six test folds are pixel-disjoint and
  exhaust all 436 kept positions, so the union is out-of-fold coverage of the whole AOI.
  Rotation 0's split is the index's own `split` column, so `final_multiyear.pt` **is** the
  rotation-0 model and is reused rather than retrained.
* the compared quantity is **footprint-matched mean density** — mean predicted density over
  a municipality's out-of-fold pixels against the mean official density over the *identical*
  pixel set. Not hectares: out-of-fold coverage is a subset of each municipality, so
  predicted OOF hectares and a whole-municipality census total are different integrals, and
  ratioing them would rank municipalities partly by how much of them lands in the kept grid.
* membership is fixed in advance and depends only on split geometry (official value present
  for the year, and >= `MIN_OOF_PIXELS` out-of-fold pixels) — never on the prediction, which
  was the defect in the old `predicted_ha > 1.0` filter.

    python -m src.rank_municipal --train-rotations    # trains rotations 1..5 (~15 min each)
    python -m src.rank_municipal                      # computes the ranking + verdict
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch

from src.baselines import common as C
from src.data.tiling import split_for_block_fold
from src.infer import _norm_name, official_municipal_ha
from src.metrics_io import metrics_dir, read_runs, write_run
from src.models.unet import build_unet
from src.train import pick_device
from src.train_loyo import fit_scalar, read_index, set_seed, train_fold, year_norm_stats
from src.utils import load_config

YEARS = [2019, 2020, 2021, 2022, 2023, 2024]
N_BLOCK_FOLDS = 6
#: A22 membership floor: 25,000 pixels = 1,000 ha at 20 m. Fixed before any number existed.
MIN_OOF_PIXELS = 25_000
#: Cached AOI-clipped GADM level-2 polygons (gitignored). Rebuilt from the URL if absent.
BOUNDARIES_CACHE = "data/boundaries_aoi.gpkg"
ROT0_CKPT = "outputs/checkpoints/final_multiyear.pt"
TRACK = "A17"
MODEL_ARM = "unet_oof"
NULL_ARM = "prev_census_ranking"


# --------------------------------------------------------------------------- rotations
def rotation_rows(rows, rotation):
    """Split `rows` by A20 rotation `rotation`, returning {split: [rows]}.

    The index's `split` column is rotation 0. Deriving the other five here rather than
    rewriting the CSV keeps one index on disk as the single source of `block_fold`, and
    makes it impossible for a rotation to disagree with the verified assignment.
    """
    out = {"train": [], "val": [], "test": []}
    for r in rows:
        bf = r.get("block_fold")
        if bf in (None, ""):
            raise AssertionError(
                "the tile index has no `block_fold` column — A17 needs the A20 rotation. "
                "Run `python -m src.data.multiyear --rebuild-block-folds` first")
        out[split_for_block_fold(int(bf), N_BLOCK_FOLDS, rotation)].append(r)
    return out


def rotation_ckpt_path(cfg, rotation) -> Path:
    if rotation == 0:
        return Path(ROT0_CKPT)
    return Path(cfg["paths"]["checkpoints_dir"]) / f"a17_rot{rotation}.pt"


def train_rotations(cfg, epochs, patience, rotations=None):
    """Train one model per rotation. Rotation 0 is skipped — it already exists as
    `final_multiyear.pt`, trained on exactly this split (asserted below, not assumed)."""
    device = pick_device()
    rows = read_index(cfg)
    stats = year_norm_stats(cfg, rows)
    for rotation in (rotations or range(N_BLOCK_FOLDS)):
        out = rotation_ckpt_path(cfg, rotation)
        if rotation == 0:
            _assert_rot0_matches_index(cfg, rows)
            print(f"[a17] rotation 0 -> reusing {out} (its split IS the index's split)")
            continue
        if out.exists():
            print(f"[a17] rotation {rotation} -> {out} exists, skipping")
            continue
        set_seed(cfg["project"]["seed"])
        sp = rotation_rows(rows, rotation)
        test_fold = (N_BLOCK_FOLDS - 1 + rotation) % N_BLOCK_FOLDS
        print(f"\n[a17] === rotation {rotation}: block_fold {test_fold} held out "
              f"({len(sp['train'])} train / {len(sp['val'])} val / {len(sp['test'])} test "
              f"rows) ===", flush=True)
        history: list[dict] = []
        model, vmae = train_fold(cfg, sp["train"], sp["val"], stats, device, epochs,
                                patience, history=history)
        scalar = fit_scalar(cfg, model, sp["train"], stats, device)
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(),
                    "year_stats": {int(y): (m, s) for y, (m, s) in stats.items()},
                    "scalar": float(scalar), "rotation": int(rotation),
                    "block_fold_test": int(test_fold),
                    "epochs_requested": int(epochs), "patience": int(patience),
                    "epochs_run": len(history), "history": history,
                    "best_epoch": max((h["epoch"] for h in history if h["improved"]),
                                      default=None),
                    "val_mae": float(vmae),
                    "n_train_tiles": len(sp["train"]), "n_val_tiles": len(sp["val"]),
                    "data_generation": str(cfg["project"]["data_generation"])}, out)
        print(f"[a17] rotation {rotation}: val_mae={vmae:.5f} scalar={scalar:.3f} -> {out}",
              flush=True)


def _assert_rot0_matches_index(cfg, rows):
    """Rotation 0 must reproduce the index's own `split` column exactly.

    Reusing `final_multiyear.pt` as the rotation-0 model is only valid if the rotation-0
    roll-up IS the split that model trained on. That is a claim about two independent
    pieces of state (the CSV column and `split_for_block_fold`'s arithmetic), so it is
    checked rather than assumed — if a future rebuild rolls up differently, the reuse
    would silently score a model on its own training ground.
    """
    bad = [(r["tile_id"], r["split"], split_for_block_fold(int(r["block_fold"]), N_BLOCK_FOLDS, 0))
           for r in rows
           if r["split"] != split_for_block_fold(int(r["block_fold"]), N_BLOCK_FOLDS, 0)]
    if bad:
        raise AssertionError(
            f"rotation 0 does not reproduce the index's `split` column for {len(bad)} rows "
            f"(e.g. {bad[:3]}), so `final_multiyear.pt` cannot be reused as the rotation-0 "
            "model — it may have trained on ground rotation 0 calls test. Train all six "
            "rotations explicitly instead")
    ck = Path(ROT0_CKPT)
    if not ck.exists():
        raise AssertionError(f"{ck} is missing — train it with "
                             "`python -m src.train_loyo --final --epochs 30 --patience 6`")
    got = str(torch.load(ck, map_location="cpu", weights_only=False).get("data_generation"))
    want = str(cfg["project"]["data_generation"])
    if got != want:
        raise AssertionError(
            f"{ck} was trained on data_generation={got!r}, config says {want!r}. Its train "
            "blocks are not this generation's train blocks")


# ----------------------------------------------------------------------- municipalities
def municipal_id_raster(cfg, shape, transform, crs):
    """Rasterise AOI municipalities onto the tile canvas -> (ids[H,W], GeoDataFrame).

    id 0 is "no municipality"; municipality i is i+1, matching the GeoDataFrame's row
    order. Cached locally because the GADM download is ~100 MB and the geometry never
    changes between runs.
    """
    import geopandas as gpd
    import rasterio.features

    cache = Path(BOUNDARIES_CACHE)
    if cache.exists():
        muni = gpd.read_file(cache).to_crs(crs)
    else:
        minx, miny, maxx, maxy = cfg["aoi"]["bbox"]
        muni = gpd.read_file(cfg["labels"]["boundaries_url"]).to_crs(crs)
        muni = muni.rename(columns={"NAME_2": "municipio", "NAME_1": "departamento",
                                    "GID_2": "gid"})
        aoi = gpd.GeoDataFrame(geometry=gpd.GeoSeries.from_wkt(
            [(f"POLYGON(({minx} {miny},{maxx} {miny},{maxx} {maxy},{minx} {maxy},"
             f"{minx} {miny}))")]), crs=4326).to_crs(crs)
        muni = muni[muni.intersects(aoi.union_all())].reset_index(drop=True)
        cache.parent.mkdir(parents=True, exist_ok=True)
        muni.to_file(cache, driver="GPKG")
    ids = rasterio.features.rasterize(
        ((geom, i + 1) for i, geom in enumerate(muni.geometry)),
        out_shape=shape, transform=transform, fill=0, dtype="int32")
    return ids, muni.reset_index(drop=True)


def canvas_geometry(cfg):
    """Shape/transform/crs of the tile canvas, read from a mosaic (all six are identical)."""
    import rasterio
    img_dir = Path(cfg["paths"]["imagery_dir"])
    region = cfg["aoi"]["region"]
    cands = sorted(img_dir.glob(f"{region}_*_annual_full.tif"))
    if not cands:
        raise AssertionError(f"no annual mosaic in {img_dir} to read the canvas geometry from")
    shapes = set()
    for c in cands:
        with rasterio.open(c) as ds:
            shapes.add((ds.height, ds.width, ds.transform.to_gdal(), str(ds.crs)))
    if len(shapes) != 1:
        raise AssertionError(
            f"the {len(cands)} annual mosaics do not share one grid ({len(shapes)} distinct "
            "geometries). A municipal id raster built on one of them would mis-assign pixels "
            "in the others — realign before ranking (scripts/align_mosaic_grid.py)")
    with rasterio.open(cands[0]) as ds:
        return (ds.height, ds.width), ds.transform, ds.crs


# ------------------------------------------------------------------- OOF accumulation
def _predictor(cfg, ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_unet(cfg).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    stats = {int(y): (np.asarray(m), np.asarray(s)) for y, (m, s) in ck["year_stats"].items()}

    @torch.no_grad()
    def predict_tile(imgn):
        t = torch.from_numpy(np.ascontiguousarray(imgn))[None].float().to(device)
        return torch.sigmoid(model(t))[0, 0].cpu().numpy()

    return predict_tile, stats, ck


def accumulate_oof(cfg, ids, n_muni, canvases=None, footprints=None):
    """Sum predicted and official density per (year, municipality) over out-of-fold pixels.

    Returns {(year, muni_idx): [sum_pred, sum_official, n_pixels]}.

    Tiles within a fold overlap by `tile_px - stride_px` (32 px), so a per-year boolean
    canvas tracks which pixels have already been counted: every canvas pixel contributes
    exactly once, regardless of how many tiles cover it. Without it the overlap strips
    would be double-weighted, which biases a *mean* density by an amount that depends on
    tile layout rather than on coca.

    `canvases`: optional {year: float32 array} filled with the same predictions, for the
    published density rasters. Only *fresh* pixels are written, so the raster the map
    serves and the numbers in the A17 table are the identical pixel set — a map that
    disagreed with its own metrics would be worse than no map.

    `footprints`: optional dict receiving {year: bool array} of which pixels were scored.
    The published official-census raster is masked to the same footprint, so the two
    layers a viewer flips between cover identical ground; otherwise the comparison would
    silently include census cells the model was never asked about.
    """
    device = pick_device()
    rows = read_index(cfg)
    H, W = ids.shape
    counted = {y: np.zeros((H, W), dtype=bool) for y in YEARS}
    acc: dict[tuple[int, int], list[float]] = {}
    seen_folds = []
    for rotation in range(N_BLOCK_FOLDS):
        ck_path = rotation_ckpt_path(cfg, rotation)
        if not ck_path.exists():
            raise AssertionError(
                f"missing {ck_path} for rotation {rotation} — A17 needs all six rotations so "
                "every pixel is predicted out-of-fold. Run "
                "`python -m src.rank_municipal --train-rotations`")
        predict_tile, stats, ck = _predictor(cfg, ck_path, device)
        got = str(ck.get("data_generation"))
        want = str(cfg["project"]["data_generation"])
        if got != want:
            raise AssertionError(f"{ck_path}: data_generation={got!r}, config {want!r}")
        te = rotation_rows(rows, rotation)["test"]
        folds = {int(r["block_fold"]) for r in te}
        seen_folds.append(folds)
        print(f"[a17] rotation {rotation}: predicting {len(te)} out-of-fold rows "
              f"(block_fold {sorted(folds)})", flush=True)
        for r in te:
            year = int(r["year"])
            x, y = int(r["x"]), int(r["y"])
            img, mask = C._tile(cfg, r)
            m, s = stats[year]
            pred = np.asarray(predict_tile(C._norm(img, m, s)), dtype="float32")
            h, w = mask.shape
            sub_ids = ids[y:y + h, x:x + w]
            fresh = ~counted[year][y:y + h, x:x + w]
            counted[year][y:y + h, x:x + w] = True
            if canvases is not None:
                win = canvases[year][y:y + h, x:x + w]
                win[fresh] = pred[fresh]
            flat_ids = sub_ids[fresh]
            if not flat_ids.size:
                continue
            n_bins = n_muni + 1
            sp = np.bincount(flat_ids, weights=pred[fresh].astype("float64"), minlength=n_bins)
            so = np.bincount(flat_ids, weights=mask[fresh].astype("float64"), minlength=n_bins)
            sn = np.bincount(flat_ids, minlength=n_bins)
            for mi in np.nonzero(sn)[0]:
                if mi == 0:          # id 0 = outside every municipality
                    continue
                a = acc.setdefault((year, int(mi)), [0.0, 0.0, 0])
                a[0] += float(sp[mi])
                a[1] += float(so[mi])
                a[2] += int(sn[mi])
    if footprints is not None:
        footprints.update(counted)
    union = set().union(*seen_folds)
    if len(union) != N_BLOCK_FOLDS or sum(len(f) for f in seen_folds) != N_BLOCK_FOLDS:
        raise AssertionError(
            f"the six rotations' test folds are not a partition of the six block_folds "
            f"(saw {[sorted(f) for f in seen_folds]}). Coverage would be neither complete "
            "nor out-of-fold everywhere")
    return acc


# ------------------------------------------------------------------------ rank metrics
def _rank_desc(values):
    """Dense ranks, 1 = largest. Ties broken by name order via the caller's stable input,
    which is deterministic but arbitrary — the inversion count exists precisely because a
    single swap matters at this n, so ties are reported rather than smoothed."""
    order = sorted(range(len(values)), key=lambda i: -values[i])
    ranks = [0] * len(values)
    for pos, i in enumerate(order):
        ranks[i] = pos + 1
    return ranks


def _spearman(a, b):
    """Spearman ρ via `scipy.stats.spearmanr` — the reference implementation, deliberately.

    A hand-rolled version was written first and agreed with scipy to 1e-9 on untied input
    but diverged on ties (0.9747 vs 1.0000 on one tied vector), because scipy averages the
    ranks of tied values while dense ranking breaks them arbitrarily. Mean densities are
    continuous, so exact ties are near-impossible here — and "near-impossible, silently
    divergent" is exactly the shape of defect this project keeps finding. `_ties` below
    counts ties so any tie-affected number is visible rather than assumed away.
    """
    from scipy.stats import spearmanr
    return float(spearmanr(np.asarray(a, float), np.asarray(b, float)).statistic)


def _ties(values):
    """How many values are involved in an exact tie (0 = strict order, ranks unambiguous)."""
    _, counts = np.unique(np.asarray(values, float), return_counts=True)
    return int(counts[counts > 1].sum())


def _topk_hit(pred_vals, off_vals, k):
    """|arm top-k ∩ official top-k| / k."""
    pr = _rank_desc(pred_vals)
    orr = _rank_desc(off_vals)
    p = {i for i, r in enumerate(pr) if r <= k}
    o = {i for i, r in enumerate(orr) if r <= k}
    return len(p & o) / k


def _adjacent_inversions(pred_vals, off_vals):
    """Adjacent pairs of the OFFICIAL order whose relative order the arm reverses."""
    order = sorted(range(len(off_vals)), key=lambda i: -off_vals[i])
    pr = _rank_desc(pred_vals)
    return int(sum(pr[order[i]] > pr[order[i + 1]] for i in range(len(order) - 1)))


def arm_metrics(pred_vals, off_vals):
    """A17's four registered quantities plus n, plus the tie counts that qualify them.

    `top2_hit`, `top3_hit` and `adjacent_inversions` use dense ranks, so under a tie their
    value depends on an arbitrary break. `ties_pred`/`ties_official` are therefore reported
    with them: non-zero means those three numbers are tie-sensitive and must be read with
    that in mind. They are 0 on every arm in the current run.
    """
    return {"spearman_rho": _spearman(pred_vals, off_vals),
            "top2_hit": _topk_hit(pred_vals, off_vals, 2),
            "top3_hit": _topk_hit(pred_vals, off_vals, 3),
            "adjacent_inversions": _adjacent_inversions(pred_vals, off_vals),
            "n": len(pred_vals),
            "ties_pred": _ties(pred_vals),
            "ties_official": _ties(off_vals)}


def select_members(acc, year, names, off_ha, *, px_m, min_px=MIN_OOF_PIXELS):
    """A22 §3 membership, applied to one year. Returns the year's rows, name-sorted.

    A municipality is in iff it has an official value for the year **and** at least
    `min_px` out-of-fold pixels. Both conditions depend only on the census and the split
    geometry — never on the prediction. The rule this replaces (`predicted_ha > 1.0` in
    `infer.municipal_hectares`) let the model decide which municipalities it would be
    scored on, which can only ever flatter it.

    Sorted by name, not by either density, so the row order carries no information about
    the answer and the tie-break in `_rank_desc` cannot depend on the arm being scored.
    """
    rows = []
    for i, name in enumerate(names):
        a = acc.get((year, i + 1))
        if a is None:
            continue
        sum_pred, sum_off, npx = a
        if npx < min_px or off_ha.get((year, i + 1)) is None:
            continue
        rows.append({"municipio": name, "muni_idx": i + 1,
                     "pred_density": sum_pred / npx,
                     "official_density": sum_off / npx,
                     "n_oof_px": npx,
                     "pred_oof_ha": sum_pred * (px_m ** 2) / 1e4,
                     "official_muni_ha": off_ha[(year, i + 1)]})
    rows.sort(key=lambda r: r["municipio"])
    return rows


# --------------------------------------------------------------------------- UI artifacts
UI_DIR = Path("ui/data")
#: Particles that GADM runs into the preceding word once camel case is split.
_PARTICLES = r"(de|del|la|las|los|y)"


def display_name(gadm_name: str) -> str:
    """GADM level-2 names arrive with the spaces stripped ("LaPlayadeBelen",
    "SanJosedeCucuta"). Restore them for display only.

    Matching against the census is unaffected either way — `infer._norm_name` strips
    everything but letters — so this is presentation, not data. It is done here rather
    than in JavaScript so the CSV a reader downloads says "El Tarra" too.
    """
    # 1. split camel case: a lowercase (or accented) letter followed by an uppercase one.
    s = re.sub(r"(?<=[a-z\u00e0-\u00ff])(?=[A-Z\u00c0-\u00dd])", " ", gadm_name)
    # 2. detach the Spanish particles step 1 leaves fused to the preceding word
    #    ("Playade Belen" -> "Playa de Belen", "Josede Cucuta" -> "Jose de Cucuta"). The
    #    trailing \b is what keeps "Santander" and "Sardinata" intact - their "de"/"di" is
    #    mid-word, so no word boundary follows it.
    s = re.sub(r"([a-z\u00e0-\u00ff])" + _PARTICLES + r"\b", r"\1 \2", s)
    return re.sub(r"\s+", " ", s).strip()


def official_ha_cached(cfg, year, retries=4, cache_dir="data/official_municipal"):
    """`infer.official_municipal_ha` for one year, with retry and an on-disk cache.

    Six live HTTP calls used to sit in the middle of this module's run, unguarded. A single
    transient failure on datos.gov.co threw `FileNotFoundError` out of `pandas.read_json`
    and destroyed a completed seven-minute inference pass — the same failure mode as defect
    F5 (one dropped Azure read killing a whole year's export), which was fixed for imagery
    and never for labels.

    The cache is per (year, resource) under `data/` (gitignored). The census for a past year
    does not change, so a cache hit is not a staleness risk; it also makes the artifacts
    reproducible on a machine with no network, which the rebuild instructions claim.
    """
    import time

    cache = Path(cache_dir) / f"{cfg['labels']['validation_resource_id']}_{year}.json"
    if cache.exists():
        return {k: float(v) for k, v in json.loads(cache.read_text(encoding="utf-8")).items()}
    last = None
    for attempt in range(retries):
        try:
            off = official_municipal_ha({**cfg, "year": year})
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(off, indent=0), encoding="utf-8")
            return off
        except Exception as e:                      # noqa: BLE001 - any transport failure retries
            last = e
            wait = 2 ** attempt
            print(f"[a17] official {year} fetch failed ({type(e).__name__}), retry in {wait}s")
            time.sleep(wait)
    raise RuntimeError(
        f"could not fetch the official municipal table for {year} after {retries} attempts "
        f"({type(last).__name__}: {last}). The ranking is not computable without it") from last


def year_metrics_from_sink(cfg):
    """Per-year measured numbers for the UI, read from the metrics sink.

    The map states its own verdicts, and prereg section 8 says no reported number may live
    only in prose — a hardcoded rho in a caveat string is prose. So the panel reads the
    same JSONL rows `docs/BASELINE_LADDER_RESULTS.md` is built from, filtered to the
    current data generation, and renders "—" if a row is missing rather than inventing one.
    """
    want = str(cfg["project"]["data_generation"])
    out: dict[str, dict] = {}
    for r in read_runs(cfg):
        if str(r.get("data_generation")) != want:
            continue
        y, m = str(r["fold_year"]), r["metrics"]
        key = {("A", "unet"): "model_iou",
               ("A", "persistence_max"): "null_iou",
               ("A17", MODEL_ARM): "model_rho",
               ("A17", NULL_ARM): "null_rho"}.get((r["track"], r["method"]))
        if not key:
            continue
        d = out.setdefault(y, {})
        d[key] = m.get("presence_iou") if key.endswith("iou") else m.get("spearman_rho")
        if r["track"] == "A17":
            d["model_inv" if r["method"] == MODEL_ARM else "null_inv"] =                 m.get("adjacent_inversions")
            d["n_muni"] = m.get("n")
    return out


def official_canvas(cfg, year, footprint):
    """The official ~1 km census grid, rasterised, masked to the out-of-fold footprint.

    Published so the map can show model and census as the *same* kind of layer at the
    *same* resolution over the *same* ground — which is the only way a viewer can judge
    the claim "reproduces the official pattern" instead of taking it on trust. It is the
    identical raster the model is trained and scored against
    (`data/labels/<region>_<year>_annual_full_cocamask.tif`), and it carries no
    finer-than-1 km information: every 20 m pixel in a cell holds that cell's one value,
    so downsampling it to 75 m discards nothing and reveals nothing.
    """
    import rasterio
    region = cfg["aoi"]["region"]
    path = Path(cfg["paths"]["labels_dir"]) / f"{region}_{year}_annual_full_cocamask.tif"
    if not path.exists():
        raise FileNotFoundError(f"missing label raster {path}")
    with rasterio.open(path) as ds:
        a = ds.read(1).astype("float32")
    if a.shape != footprint.shape:
        raise AssertionError(
            f"{path.name} is {a.shape}, the tile canvas is {footprint.shape} — a mask built "
            "on one grid cannot be applied to the other")
    a[~footprint] = 0.0
    return a


def write_ui_artifacts(cfg, table, canvases, footprints, ids, muni, transform, crs):
    """Regenerate everything the map serves, from the out-of-fold predictions.

    Two honesty problems have to be solved here, both of which the previous artifacts got
    wrong by construction rather than by wording:

    1. **Partial coverage must not read as under-prediction.** Out-of-fold pixels cover
       ~71% of Tibú's area, so its summed out-of-fold hectares are ~half the census total
       for a reason that has nothing to do with the model. `predicted_ha` is therefore the
       out-of-fold mean density extrapolated over the municipality's pixels **inside the
       AOI canvas** — extrapolating across the dropped-tile gaps only, never outside the
       study area. `pred_oof_ha` (measured, no extrapolation) and both coverage fractions
       ship alongside it so the extrapolation is auditable.
    2. **A municipality extending past the AOI still reads low**, because the census counts
       all of it. `canvas_share` reports exactly how much of each municipality the AOI
       contains, so a low ratio can be attributed rather than mistaken for model error.

    Resolution: every raster goes through `nowcast.write_cog`, which asserts the 75 m
    publication floor (A18) and refuses to upsample.
    """
    from src.nowcast import write_cog

    px_ha = (cfg["imagery"]["resolution_m"] ** 2) / 1e4
    canvas_px = np.bincount(ids.ravel(), minlength=len(muni) + 1)
    profile = {"crs": crs, "transform": transform}
    UI_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    for year in YEARS:
        cog = UI_DIR / f"density_{year}_cog.tif"
        write_cog(canvases[year], profile, cog, long_side=1500)
        off_cog = UI_DIR / f"official_{year}_cog.tif"
        write_cog(official_canvas(cfg, year, footprints[year]), profile, off_cog,
                  long_side=1500)
        rows = table[year]
        by_name = {r["municipio"]: r for r in rows}
        model_rank = {r["municipio"]: i + 1 for i, r in
                      enumerate(sorted(rows, key=lambda r: -r["pred_density"]))}
        off_rank = {r["municipio"]: i + 1 for i, r in
                    enumerate(sorted(rows, key=lambda r: -r["official_density"]))}
        gdf = muni.copy()
        recs = []
        for i, name in enumerate(str(n) for n in muni["municipio"]):
            r = by_name.get(name)
            cpx = int(canvas_px[i + 1])
            if r is None:            # in the AOI but not in the ranking (A22 membership)
                recs.append({"predicted_ha": None, "official_ha": None, "ratio": None,
                             "pred_oof_ha": None, "pred_density": None,
                             "official_density": None, "oof_px": 0, "canvas_px": cpx,
                             "oof_share_of_canvas": None, "canvas_share_hint": None,
                             "model_rank": None, "official_rank": None, "in_ranking": False})
                continue
            pred_ha = r["pred_density"] * cpx * px_ha
            recs.append({
                "predicted_ha": pred_ha,
                "official_ha": r["official_muni_ha"],
                "ratio": pred_ha / r["official_muni_ha"] if r["official_muni_ha"] else None,
                "pred_oof_ha": r["pred_oof_ha"],
                "pred_density": r["pred_density"],
                "official_density": r["official_density"],
                "oof_px": int(r["n_oof_px"]),
                "canvas_px": cpx,
                "oof_share_of_canvas": r["n_oof_px"] / cpx if cpx else None,
                "canvas_share_hint": cpx * px_ha,     # AOI hectares of this municipality
                "model_rank": model_rank[name],
                "official_rank": off_rank[name],
                "in_ranking": True})
        for k in recs[0]:
            gdf[k] = [rec[k] for rec in recs]
        gdf["year"] = year
        gdf["name"] = gdf["municipio"]
        gdf["label"] = [display_name(str(n)) for n in gdf["municipio"]]
        keep = ["gid", "name", "label", "municipio", "departamento", "year", "predicted_ha",
                "official_ha", "ratio", "pred_oof_ha", "pred_density", "official_density",
                "oof_px", "canvas_px", "oof_share_of_canvas", "canvas_share_hint",
                "model_rank", "official_rank", "in_ranking"]
        out = gdf.to_crs(4326)[[*keep, "geometry"]]
        gj = UI_DIR / f"municipal_coca_{year}.geojson"
        csv = UI_DIR / f"municipal_coca_{year}.csv"
        gj.write_text(out.to_json(), encoding="utf-8")
        out.drop(columns="geometry").to_csv(csv, index=False, encoding="utf-8")
        written += [cog, off_cog, gj, csv]
        print(f"[a17-ui] {year}: {cog.name} + {off_cog.name} + {gj.name} + {csv.name} "
              f"({int(out['in_ranking'].sum())} ranked of {len(out)} municipalities)")
    # The undated aliases the map falls back to; kept pointing at the latest census year.
    metrics = {"data_generation": str(cfg["project"]["data_generation"]),
               "years": year_metrics_from_sink(cfg),
               # The map fits THIS, not the municipal envelope: two AOI municipalities
               # (Curumani, San Jose de Cucuta) extend far past the study area, so fitting
               # their bounds opened the map two zoom levels too wide with the data as a
               # speck in the middle.
               "aoi_bbox": [float(v) for v in cfg["aoi"]["bbox"]],
               "source": "outputs/metrics/baseline_ladder.jsonl"}
    (UI_DIR / "metrics.json").write_text(json.dumps(metrics, indent=1), encoding="utf-8")
    print(f"[a17-ui] metrics.json for {len(metrics['years'])} years, straight from the sink")
    latest = max(YEARS)
    for suffix in ("geojson", "csv"):
        src = UI_DIR / f"municipal_coca_{latest}.{suffix}"
        (UI_DIR / f"municipal_coca.{suffix}").write_bytes(src.read_bytes())
    print(f"[a17-ui] municipal_coca.{{geojson,csv}} aliased to {latest}")
    return written


# ------------------------------------------------------------------------------- report
def run(cfg, write_ui=False):
    shape, transform, crs = canvas_geometry(cfg)
    ids, muni = municipal_id_raster(cfg, shape, transform, crs)
    print(f"[a17] canvas {shape} {crs}; {len(muni)} municipalities intersect the AOI")
    # One inference pass serves both the metrics and the published rasters, so they cannot
    # drift apart. ~108 MB per year of canvas, only allocated when the rasters are wanted.
    canvases = ({y: np.zeros(shape, dtype="float32") for y in YEARS} if write_ui else None)
    footprints: dict[int, np.ndarray] = {}
    acc = accumulate_oof(cfg, ids, len(muni), canvases=canvases,
                         footprints=footprints if write_ui else None)

    official = {}
    for y in YEARS:
        official[y] = official_ha_cached(cfg, y)
    names = [str(n) for n in muni["municipio"]]
    off_ha = {(y, i + 1): official[y].get(_norm_name(names[i]))
              for y in YEARS for i in range(len(names))}

    # ---- per-year, footprint-matched density table (A22 §2/§3)
    table = {}
    for y in YEARS:
        rowset = select_members(acc, y, names, off_ha, px_m=cfg["imagery"]["resolution_m"])
        table[y] = rowset

    # ---- the null: previous year's OFFICIAL footprint-matched density (2020 for 2019, A19)
    lines = ["# A17 — municipal ranking on out-of-fold predictions",
             "",
             (f"Prereg A17 + A22. `data_generation: {cfg['project']['data_generation']}`. "
             "Every pixel is predicted by a model that never trained on it (six A20 "
             "rotations). Primary quantity is footprint-matched **mean density**; hectares "
             "are shown for continuity only and are **not** footprint-matched."),
             ""]
    verdicts = []
    for y in YEARS:
        cur = table[y]
        if len(cur) < 3:
            raise AssertionError(
                f"{y}: only {len(cur)} municipalities pass A22 membership — a ranking metric "
                "on fewer than 3 is not interpretable. Do not lower MIN_OOF_PIXELS to fix "
                "this; it was fixed in advance")
        prev_y = 2020 if y == 2019 else y - 1
        prev = {r["municipio"]: r["official_density"] for r in table[prev_y]}
        missing = [r["municipio"] for r in cur if r["municipio"] not in prev]
        if missing:
            raise AssertionError(
                f"{y}: municipalities {missing} have no {prev_y} official density, so the "
                "previous-census null cannot be formed for them")
        off_vals = [r["official_density"] for r in cur]
        model_vals = [r["pred_density"] for r in cur]
        null_vals = [prev[r["municipio"]] for r in cur]
        m_model = arm_metrics(model_vals, off_vals)
        m_null = arm_metrics(null_vals, off_vals)
        won = m_model["spearman_rho"] > m_null["spearman_rho"]   # ties -> the null (A22)
        verdicts.append(won)
        extra_common = {"prereg": "A17/A22", "min_oof_px": MIN_OOF_PIXELS,
                        "municipios": [r["municipio"] for r in cur],
                        "quantity": "footprint_matched_mean_density"}
        write_run(cfg, MODEL_ARM, y, m_model, track=TRACK, n_train_tiles=1752,
                  n_test_tiles=len(cur),
                  calibration_scalar=None, fit_years=[yy for yy in YEARS],
                  extra={**extra_common, "n_train_tiles_basis": "per rotation (all six equal)",
                         "rotations": N_BLOCK_FOLDS,
                         "n_oof_px": {r["municipio"]: r["n_oof_px"] for r in cur}})
        write_run(cfg, NULL_ARM, y, m_null, track=TRACK, n_train_tiles=0,
                  n_test_tiles=len(cur), calibration_scalar=None, fit_years=[prev_y],
                  extra={**extra_common, "uses_imagery": False,
                         "null_source_year": prev_y, "is_causal": prev_y < y})
        write_run(cfg, "a17_verdict", y,
                  {"model_beats_null": float(won),
                   "rho_delta": m_model["spearman_rho"] - m_null["spearman_rho"],
                   "n": m_model["n"]},
                  track=TRACK, n_train_tiles=0, n_test_tiles=len(cur),
                  calibration_scalar=None, fit_years=[prev_y],
                  extra={**extra_common, "is_decision_quantity": True,
                         "rule": "A17/A22: model rho > null rho in >=5/6 years",
                         "tie_goes_to": "null"})

        lines += [f"## {y}  (n = {m_model['n']})", "",
                  ("| municipality | official density | model density | prev-census density "
                  "| OOF px | pred OOF ha | official muni ha |"),
                  "|---|--:|--:|--:|--:|--:|--:|"]
        for r in sorted(cur, key=lambda r: -r["official_density"]):
            lines.append(
                f"| {r['municipio']} | {r['official_density']:.5f} | {r['pred_density']:.5f} "
                f"| {prev[r['municipio']]:.5f} | {r['n_oof_px']:,} | {r['pred_oof_ha']:,.0f} "
                f"| {r['official_muni_ha']:,.0f} |")
        lines += ["",
                  "| arm | Spearman ρ | top-2 hit | top-3 hit | adjacent inversions |",
                  "|---|--:|--:|--:|--:|",
                  (f"| model (out-of-fold U-Net) | {m_model['spearman_rho']:.3f} | "
                  f"{m_model['top2_hit']:.2f} | {m_model['top3_hit']:.2f} | "
                  f"{m_model['adjacent_inversions']} |"),
                  (f"| null (reuse {prev_y} census) | {m_null['spearman_rho']:.3f} | "
                  f"{m_null['top2_hit']:.2f} | {m_null['top3_hit']:.2f} | "
                  f"{m_null['adjacent_inversions']} |"),
                  "",
                  (f"Model {'beats' if won else 'does NOT beat'} the null on ρ "
                  f"({m_model['spearman_rho']:+.3f} vs {m_null['spearman_rho']:+.3f})."), ""]
        print(f"[a17] {y}: n={m_model['n']} model rho={m_model['spearman_rho']:.3f} "
              f"(inv {m_model['adjacent_inversions']}) vs null rho="
              f"{m_null['spearman_rho']:.3f} (inv {m_null['adjacent_inversions']}) -> "
              f"{'model' if won else 'null'}")

    wins = sum(verdicts)
    if wins >= 5:
        call = ("**municipal ranking EARNED as a model result** (≥5/6)")
    elif wins >= 3:
        call = ("**INDISTINGUISHABLE from reusing the previous census** — the claim shrinks "
                "to \"reproduces the official ranking\", with the previous census named as an "
                "equally good method (3–4/6)")
    else:
        call = ("**NOT PUBLISHABLE as a model result** — report as a negative result with the "
                "same prominence as A16's (≤2/6). No retuning in response (A22)")
    lines += ["## Verdict (prereg A17/A22, frozen before computing)", "",
              (f"The out-of-fold model beats the previous-census ranking on Spearman ρ in "
              f"**{wins}/6** years → {call}."), "",
              ("> ρ is never quoted without the adjacent-inversion count (A17): at n≈9 one swap "
              "moves ρ materially. Ties count for the null, which is the incumbent."), ""]
    out = metrics_dir(cfg) / "a17_municipal.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n[a17] {wins}/6 -> {call}")
    print(f"[a17] wrote {out}")
    if write_ui:
        write_ui_artifacts(cfg, table, canvases, footprints, ids, muni, transform, crs)
    return wins


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="A17 municipal ranking on out-of-fold folds.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--train-rotations", action="store_true",
                    help="train the per-rotation models (rotation 0 is reused)")
    ap.add_argument("--rotations", type=int, nargs="+", default=None)
    ap.add_argument("--write-ui", action="store_true",
                    help="also regenerate ui/data (75 m COGs + municipal aggregates) from "
                         "the same out-of-fold pass")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=6)
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.train_rotations:
        train_rotations(cfg, args.epochs, args.patience, args.rotations)
    else:
        run(cfg, write_ui=args.write_ui)
