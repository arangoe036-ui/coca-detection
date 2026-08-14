"""P1a — Export Sentinel-2 + Sentinel-1 seasonal composites + indices as a GeoTIFF.

Implements plan §6.1 using the **Microsoft Planetary Computer** STAC API (free, no
GEE, no signup) instead of Earth Engine. Builds cloud-masked Sentinel-2 seasonal
median composites, appends spectral indices, adds Sentinel-1 RTC (VV/VH) composites,
and writes a stacked GeoTIFF in the AOI's UTM CRS.

Band order in the output raster (matches model.in_channels):
    s2_bands (reflectance, float32) + indices + s1_bands (dB)

Smoke test (plan §3 "smallest end-to-end slice"):
    python -m src.data.stac_export --quick
    -> exports one season over aoi.quick_bbox; verify it opens in rasterio.

Full export:
    python -m src.data.stac_export            # all seasons over aoi.bbox
"""

from __future__ import annotations

import argparse
import errno
import math
import os
import re
import time
from pathlib import Path

import numpy as np

from src.utils import ensure_dirs, load_config

# Assets that live at native 20 m on Planetary Computer's Sentinel-2 L2A.
_S2_20M = {"B05", "B06", "B07", "B8A", "B11", "B12"}

# ESA Processing Baseline 04.00 (effective 2022-01-25) set BOA_ADD_OFFSET = -1000 for
# L2A, so the correct DN->reflectance conversion is (DN - 1000)/10000. Scenes below
# baseline 04.00 have no offset. (Phase 6.6.)
_BOA_OFFSET_DN = -1000.0
_BASELINE_04 = 4.0
_BASELINE_04_DATE = "2022-01-25"

# Mosaic nodata. rasterio.merge with nodata=None resolves nodataval to 0 and zero-fills
# any area no source covers, which is indistinguishable from real zero reflectance in a
# tagless raster. NaN is written into the profile AND already fails the isfinite() checks
# downstream (src/data/tiling.py skips tiles that are >50% non-finite).
_MOSAIC_NODATA = float("nan")

# Identity stamped into every sub-tile so a resumed run can prove a leftover was produced
# by the same band stack / resolution as the current config (see _subtile_reject_reason).
_TAG_BANDS = "coca_band_order"
_TAG_RES = "coca_resolution_m"
_TAG_BBOX = "coca_subtile_bbox"
_TAG_CLOUD = "coca_cloud_cover_max"
_TAG_SCL = "coca_scl_mask_classes"

# Failures no amount of backoff can fix — fail fast instead of burning 75s of retries.
_FATAL_ERRNOS = {errno.ENOSPC}


def boa_offset_dn(item) -> float:
    """DN offset to ADD before /10000 for one S2 item: -1000 if processing baseline
    >= 04.00, else 0. Keyed on the `s2:processing_baseline` metadata field (a
    reprocessed pre-2022 scene carries 04.00 and the offset); date only as fallback."""
    props = getattr(item, "properties", {}) or {}
    base = props.get("s2:processing_baseline")
    try:
        bnum = float(base)
    except (TypeError, ValueError):
        bnum = None
    if bnum is not None:
        return _BOA_OFFSET_DN if bnum >= _BASELINE_04 else 0.0
    return _BOA_OFFSET_DN if str(props.get("datetime", "")) >= _BASELINE_04_DATE else 0.0


def get_catalog(cfg: dict):
    """Open the PC STAC catalog with automatic asset signing."""
    import planetary_computer as pc
    from pystac_client import Client

    return Client.open(cfg["imagery"]["stac_url"], modifier=pc.sign_inplace)


def seasonal_ranges(year: int, n: int) -> list[tuple[str, str]]:
    """Split a calendar year into `n` contiguous date ranges (YYYY-MM-DD)."""
    edges = np.linspace(0, 365, n + 1).astype(int)
    base = np.datetime64(f"{year}-01-01")
    out = []
    for i in range(n):
        start = base + np.timedelta64(int(edges[i]), "D")
        end = base + np.timedelta64(int(edges[i + 1]) - 1, "D")
        out.append((str(start), str(end)))
    return out


def _search(catalog, collection: str, bbox, date_range, query=None):
    return list(
        catalog.search(
            collections=[collection],
            bbox=bbox,
            datetime=f"{date_range[0]}/{date_range[1]}",
            query=query,
        ).items()
    )


def build_s2_composite(catalog, cfg: dict, bbox, date_range):
    """Cloud-masked Sentinel-2 seasonal median composite -> xarray DataArray (reflectance)."""
    from odc.stac import load as odc_load

    img = cfg["imagery"]
    items = _search(
        catalog, img["s2_collection"], bbox, date_range,
        query={"eo:cloud_cover": {"lt": img["cloud_cover_max"]}},
    )
    if not items:
        raise RuntimeError(f"No Sentinel-2 scenes for {date_range} over {bbox}.")

    bands = list(img["s2_bands"]) + ["SCL"]
    ds = odc_load(
        items, bands=bands, bbox=bbox,
        crs=f"EPSG:{cfg['aoi']['utm_epsg']}", resolution=img["resolution_m"],
        chunks={"x": 1024, "y": 1024}, groupby="solar_day",
    )

    # SCL cloud/shadow/snow mask -> NaN, then median over time (skipna).
    scl = ds["SCL"]
    bad = scl.isin(img["scl_mask_classes"])
    refl_bands = img["s2_bands"]
    import pandas as pd
    import xarray as xr

    # Per-solar-day BOA additive offset (baseline 04.00 = -1000 DN), aligned to the
    # loaded time axis and applied PER SCENE BEFORE /10000 and before indices (6.6a).
    day_off = {str(pd.Timestamp(it.properties["datetime"]).date()): boa_offset_dn(it)
               for it in items}
    off = np.array([day_off.get(str(pd.Timestamp(t).date()), 0.0)
                    for t in ds["time"].values], dtype="float32")
    off_da = xr.DataArray(off, dims=["time"], coords={"time": ds["time"]})
    scaled = {}
    for b in refl_bands:
        # L2A (DN + offset) -> surface reflectance [0,1]; masked pixels become NaN.
        da = (ds[b].where(~bad).astype("float32") + off_da) / 10000.0
        scaled[b] = da.median(dim="time", skipna=True)

    comp = xr.concat([scaled[b] for b in refl_bands], dim="band")
    comp = comp.assign_coords(band=refl_bands)
    return comp


def add_indices(s2: "object", cfg: dict):
    """Compute the configured spectral indices from the S2 reflectance composite."""
    import xarray as xr

    def band(name):
        # drop the scalar 'band' coord so index arithmetic concats cleanly
        return s2.sel(band=name).drop_vars("band")

    # Floor reflectance at 0 so the baseline-offset correction (which can drive dark
    # pixels slightly negative) cannot blow indices up via near-zero denominators.
    b = {n: band(n).clip(min=0.0) for n in cfg["imagery"]["s2_bands"]}
    eps = 1e-6
    defs = {
        "NDVI": (b["B08"] - b["B04"]) / (b["B08"] + b["B04"] + eps),
        "EVI": 2.5 * (b["B08"] - b["B04"]) / (b["B08"] + 6 * b["B04"] - 7.5 * b["B02"] + 1 + eps),
        "SAVI": 1.5 * (b["B08"] - b["B04"]) / (b["B08"] + b["B04"] + 0.5 + eps),
        "NDWI": (b["B03"] - b["B08"]) / (b["B03"] + b["B08"] + eps),
        "NDRE": (b["B08"] - b["B05"]) / (b["B08"] + b["B05"] + eps),
        "NBR": (b["B08"] - b["B12"]) / (b["B08"] + b["B12"] + eps),
    }
    # Clip to physical ranges (bounded ratios in [-1,1]; EVI/SAVI wider) — dark-pixel guard.
    clip = {"NDVI": (-1, 1), "EVI": (-1, 2.5), "SAVI": (-1.5, 1.5),
            "NDWI": (-1, 1), "NDRE": (-1, 1), "NBR": (-1, 1)}
    wanted = cfg["imagery"]["indices"]
    arr = xr.concat([defs[i].clip(*clip[i]) for i in wanted], dim="band").assign_coords(band=wanted)
    return arr


def build_s1_composite(catalog, cfg: dict, bbox, date_range):
    """Sentinel-1 RTC VV/VH seasonal median composite in dB -> xarray DataArray."""
    from odc.stac import load as odc_load
    import xarray as xr

    img = cfg["imagery"]
    items = _search(catalog, img["s1_collection"], bbox, date_range)
    # Exclude Sentinel-1C granules: some 2025 S1C RTC assets on Planetary Computer are
    # malformed (unreadable tiffs), and the 2019-2024 training data had no S1C anyway
    # (it launched late 2024), so dropping it keeps inputs consistent with the model.
    items = [it for it in items if not it.id.upper().startswith("S1C")]
    if not items:
        raise RuntimeError(f"No Sentinel-1 scenes for {date_range} over {bbox}.")

    ds = odc_load(
        items, bands=img["s1_bands"], bbox=bbox,
        crs=f"EPSG:{cfg['aoi']['utm_epsg']}", resolution=img["resolution_m"],
        chunks={"x": 1024, "y": 1024}, groupby="solar_day",
    )
    out = []
    for pol in img["s1_bands"]:
        lin = ds[pol].where(ds[pol] > 0)
        db = 10.0 * np.log10(lin)
        out.append(db.median(dim="time", skipna=True))
    comp = xr.concat(out, dim="band").assign_coords(band=[p.upper() for p in img["s1_bands"]])
    return comp


def band_order(cfg: dict) -> list[str]:
    img = cfg["imagery"]
    return list(img["s2_bands"]) + list(img["indices"]) + [p.upper() for p in img["s1_bands"]]


def build_stack(catalog, cfg: dict, bbox, date_range):
    """Build the eager 18-band composite stack (S2 reflectance + indices + S1 dB)."""
    import dask
    import xarray as xr

    s2 = build_s2_composite(catalog, cfg, bbox, date_range)
    idxs = add_indices(s2, cfg)
    s1 = build_s1_composite(catalog, cfg, bbox, date_range)
    stack = xr.concat([s2, idxs, s1], dim="band").assign_coords(band=band_order(cfg))
    # dask's threaded scheduler deadlocks with rasterio/GDAL reads on Windows (hangs at
    # ~0% CPU indefinitely, never writes a byte); synchronous is ~30s for the quick
    # export. Parallelism comes from running one PROCESS per year instead. Scoped to
    # this compute so importing this module cannot change anyone else's scheduler.
    with dask.config.set(scheduler="synchronous"):
        return stack.compute()


def _subtile_bboxes(bbox, step_deg: float):
    """Grid a lon/lat bbox into sub-bboxes of ~step_deg (with tiny overlap)."""
    import numpy as np

    min_lon, min_lat, max_lon, max_lat = bbox
    lons = list(np.arange(min_lon, max_lon, step_deg)) + [max_lon]
    lats = list(np.arange(min_lat, max_lat, step_deg)) + [max_lat]
    ov = step_deg * 0.02
    out = []
    for i in range(len(lons) - 1):
        for j in range(len(lats) - 1):
            out.append((round(lons[i] - ov, 4), round(lats[j] - ov, 4),
                        round(lons[i + 1] + ov, 4), round(lats[j + 1] + ov, 4)))
    return out


def _safe_unlink(path: Path) -> bool:
    """Best-effort delete of a stale/partial sub-tile. A transient Windows file lock must
    not turn a wasted recompute into an aborted multi-hour export: the write that follows
    overwrites the file anyway, and a leftover that stays locked just fails validation
    again on the next run."""
    try:
        path.unlink()
        return True
    except OSError as e:
        print(f"[stac]  warn: could not delete {path.name} ({e}); continuing", flush=True)
        return False


def _subtile_tags(cfg: dict, names: list[str], sb) -> dict:
    """Identity metadata written into each sub-tile GeoTIFF."""
    img = cfg["imagery"]
    return {
        _TAG_BANDS: ",".join(names),
        _TAG_RES: f"{float(img['resolution_m']):g}",
        _TAG_BBOX: ",".join(f"{v:.4f}" for v in sb),
        # Band order, resolution and geometry do NOT change when the masking/filter config
        # changes, so without these two a sub-tile built under `cloud_cover_max: 40` and
        # `scl_mask_classes` lacking SCL 0/1 would pass every other check and be merged
        # with new-filter tiles. That is the same silent-mixing failure this stamp exists
        # to prevent, one level up. (Prereg A13.)
        _TAG_CLOUD: f"{img['cloud_cover_max']}",
        _TAG_SCL: ",".join(str(c) for c in img["scl_mask_classes"]),
    }


def _expected_grid(cfg: dict, sb):
    """The (bounds, width, height) a sub-tile SHOULD have: `sb` reprojected into the
    output UTM CRS at the configured resolution. Sub-pixel grid snapping is absorbed by
    the tolerance at the call site."""
    from rasterio.warp import transform_bounds

    res = float(cfg["imagery"]["resolution_m"])
    left, bottom, right, top = transform_bounds(
        "EPSG:4326", f"EPSG:{int(cfg['aoi']['utm_epsg'])}", *sb)
    return (left, bottom, right, top), round((right - left) / res), round((top - bottom) / res)


def _subtile_reject_reason(path: Path, cfg: dict, sb, names: list[str]) -> str | None:
    """None if `path` may be reused for sub-tile `sb`, else a short reason why not.

    Readability is not enough. `_sub_{region}_{year}_{k}.tif` is invariant to
    `aoi.bbox`, `full_subtile_deg`, `resolution_m` and to WHICH bands are stacked, and
    leftovers exist precisely when a previous run crashed — i.e. exactly when the code
    then got changed and re-run. So verify geospatial identity too, and treat a file
    whose identity tag is missing (legacy) as unverifiable, hence NOT reusable.
    """
    import rasterio
    from rasterio.windows import Window

    res = float(cfg["imagery"]["resolution_m"])
    exp_bounds, exp_w, exp_h = _expected_grid(cfg, sb)
    tol = 4 * res  # grid snapping moves an edge by <1 px; 4 px still rejects any regrid
    try:
        with rasterio.open(path) as src:
            if src.count != len(names):
                return f"band count {src.count} != {len(names)}"
            epsg = src.crs.to_epsg() if src.crs else None
            if epsg != int(cfg["aoi"]["utm_epsg"]):
                return f"crs EPSG:{epsg} != EPSG:{cfg['aoi']['utm_epsg']}"
            if not all(math.isclose(r, res, rel_tol=1e-6) for r in src.res):
                return f"resolution {src.res} != {res}m"
            if any(abs(a - b) > tol for a, b in zip(src.bounds, exp_bounds)):
                return (f"bounds {tuple(round(v, 1) for v in src.bounds)} != expected "
                        f"{tuple(round(v, 1) for v in exp_bounds)} (tol {tol:g}m)")
            if abs(src.width - exp_w) > 4 or abs(src.height - exp_h) > 4:
                return f"size {src.width}x{src.height} != expected {exp_w}x{exp_h}"
            tags = {k.lower(): v for k, v in src.tags().items()}
            want = _subtile_tags(cfg, names, sb)
            if _TAG_BANDS not in tags or _TAG_RES not in tags:
                return "no identity tag (written by an older run) — cannot verify"
            if _TAG_CLOUD not in tags or _TAG_SCL not in tags:
                return "no filter-config tag (pre-A13 run) — cannot verify masking"
            for key, label in ((_TAG_BANDS, "band order"), (_TAG_RES, "resolution"),
                               (_TAG_CLOUD, "cloud_cover_max"), (_TAG_SCL, "scl_mask_classes")):
                if tags[key] != want[key]:
                    return f"{label} tag '{tags[key]}' != '{want[key]}'"
            # open() + header checks only read the header, which survives truncation
            # intact, so a half-written tile would still look valid. Touch one pixel at
            # the far corner of the last band — the last data to be written — to force
            # GDAL to read the tail of the file, where truncation actually shows up.
            src.read(src.count, window=Window(src.width - 1, src.height - 1, 1, 1))
            return None
    except Exception as e:
        return f"unreadable ({type(e).__name__}: {str(e)[:80]})"


#: Exit code meaning "credentials expired — relaunch me and I will resume". The supervisor
#: keys on this; see EXPIRED_CREDS_EXIT in the launch loop.
EXPIRED_CREDS_EXIT = 75


def _is_expired_creds(e: BaseException) -> bool:
    """True for an expired/rejected Planetary Computer SAS token (HTTP 403).

    This killed the 2026-08-10 rebuild. PC signs asset URLs with a time-limited token and
    `planetary_computer` reuses it, so calling `get_catalog()` again inside the SAME process
    hands back the SAME expired token — every retry gets another 403. Three workers each
    stopped ONE SECOND before their token's `se=` expiry timestamp.

    Retrying in-process therefore cannot help; only a new process can obtain new credentials.
    So this is fatal *to the process* but explicitly recoverable by relaunching, which is
    what makes it different from MemoryError/ENOSPC below.
    """
    txt = str(e)
    # Match the STATUS CODE POSITION, never a bare "403" anywhere in the text. Exception
    # messages embed the failing asset href, and Sentinel scene IDs contain dates — so
    # `S2A_MSIL2A_20190403T150719` made a plain `"403" in txt` return True for a 404 on any
    # early-April scene. That mislabelled ordinary transient errors (404/429/503) as expired
    # credentials and triggered an immediate relaunch, discarding up to an hour of work.
    if re.search(r"HTTP response code:\s*403\b", txt):
        return True
    if re.search(r"\b403\s+Forbidden\b", txt):
        return True
    # Azure's actual SAS-expiry body: "AuthenticationFailed ... Signature not valid in the
    # specified time frame". There is no literal "expired" in it, so do not test for one.
    return "AuthenticationFailed" in txt


def _is_fatal(e: BaseException, attempt: int = 99) -> bool:
    """True for failures retrying cannot fix, so the caller must re-raise immediately rather
    than sleeping through its backoff.

    Expired credentials are only fatal from the SECOND attempt onward. A one-off 403 does
    happen transiently, and the in-process retry demonstrably recovered from transient read
    errors throughout the successful 2020/2021 exports. An expired token, by contrast, 403s
    again instantly and deterministically — so one immediate no-sleep retry cleanly separates
    the two cases and costs nothing, while treating the first 403 as fatal would throw away
    the whole in-progress sub-tile.
    """
    if isinstance(e, MemoryError):
        return True
    if _is_expired_creds(e) and attempt >= 2:
        return True
    return isinstance(e, OSError) and e.errno in _FATAL_ERRNOS


def log_token_expiry(catalog, cfg: dict) -> None:
    """Print the SAS expiry of a signed asset, so a future stall is diagnosable from the log
    alone. On 2026-08-10 this had to be reverse-engineered out of stack traces."""
    import re
    try:
        items = _search(catalog, cfg["imagery"]["s2_collection"], cfg["aoi"]["quick_bbox"],
                        (f"{cfg['year']}-01-01", f"{cfg['year']}-01-31"))
        if not items:
            return
        href = next(iter(items[0].assets.values())).href
        m = re.search(r"[?&]se=([^&]+)", href)
        if m:
            from urllib.parse import unquote
            print(f"[stac] SAS token expires {unquote(m.group(1))} (UTC) — a sub-tile that "
                  f"outlives this fails 403 and the process exits {EXPIRED_CREDS_EXIT} to be "
                  f"relaunched", flush=True)
    except Exception:
        pass  # diagnostics only; never block an export


def export_full(cfg: dict) -> Path:
    """Full-AOI annual composite via TILED export + mosaic (robust to URL expiry).

    Each sub-tile gets a FRESH signed catalog and is computed eagerly (low memory),
    so no single read outlives its signed URL and RAM stays bounded. Sub-tiles are
    then mosaicked into one GeoTIFF for the downstream pipeline.
    """
    import rasterio
    import rioxarray  # noqa: F401
    from rasterio.merge import merge as rio_merge

    # Azure range reads intermittently return short/0-byte responses; let GDAL retry them
    # internally so most never surface as exceptions. setdefault = outer env still wins.
    # Retry INSIDE GDAL so a dropped range read never propagates up and destroys the whole
    # sub-tile. Raised 5 -> 10 after 2023 failed three times: that year has ~365 scenes over a
    # single sub-tile (vs 169-219 elsewhere), i.e. ~4000 asset reads per attempt, so at any
    # realistic per-read failure rate at least one failure per attempt is near-certain — and
    # one failure restarted everything. RETRY_CODES=ALL matters because these failures are
    # truncated bodies ("got 0 bytes, expected N"), not clean HTTP error codes, and GDAL's
    # default retry list only covers 429/500/502/503/504.
    os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "10")
    os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "2")
    os.environ.setdefault("GDAL_HTTP_RETRY_CODES", "ALL")
    os.environ.setdefault("VSI_CACHE", "TRUE")

    ensure_dirs(cfg)
    out_dir = Path(cfg["paths"]["imagery_dir"])
    names = band_order(cfg)
    date_range = (f"{cfg['year']}-01-01", f"{cfg['year']}-12-31")
    subs = _subtile_bboxes(cfg["aoi"]["bbox"], cfg["imagery"]["full_subtile_deg"])
    print(f"[stac] FULL tiled export: {len(subs)} sub-tiles @ {cfg['imagery']['resolution_m']}m, "
          f"annual {date_range}", flush=True)
    # Emit the SAS expiry up front. Written specifically so a future stall is readable from
    # the log; on 2026-08-10 it had to be reverse-engineered out of stack traces. (It was
    # defined but never called until 2026-08-11 — a diagnostic nobody invokes is not a
    # diagnostic, which is the same class of mistake as a guard behind a failing import.)
    log_token_expiry(get_catalog(cfg), cfg)

    sub_paths = []
    skipped = []
    for k, sb in enumerate(subs):
        sp = out_dir / f"_sub_{cfg['aoi']['region']}_{cfg['year']}_{k:02d}.tif"
        # Resume: reuse a sub-tile from an earlier crashed run, but only if it is really
        # THIS sub-tile — same band stack, same grid — and fully written. Anything else
        # (truncated, regridded, older band order, untagged) is deleted and redone.
        if sp.exists():
            reason = _subtile_reject_reason(sp, cfg, sb, names)
            if reason is None:
                sub_paths.append(sp)
                print(f"[stac]  sub {k+1}/{len(subs)} reusing {sp.name}", flush=True)
                continue
            print(f"[stac]  sub {k+1}/{len(subs)} {sp.name} not reusable ({reason}) "
                  f"-> recomputing", flush=True)
            _safe_unlink(sp)
        # A single dropped HTTP read used to kill the whole year, so retry the whole
        # build+write with backoff (5/10/20/40s) and a FRESH signed catalog per attempt.
        attempts = 5
        # Success flag, NOT `stack is not None`: one attempt can build a stack and then
        # fail inside to_raster (leaving no file), and a LATER attempt can exit via the
        # skip branch — the stale stack would then get a deleted path appended.
        wrote, shape = False, None
        for attempt in range(1, attempts + 1):
            if sp.exists():  # drop a partial write from a previous attempt
                _safe_unlink(sp)
            try:
                catalog = get_catalog(cfg)  # fresh sign + connections per attempt
                stack = build_stack(catalog, cfg, list(sb), date_range)
                stack.rio.to_raster(sp, driver="GTiff", compress="deflate",
                                    tags=_subtile_tags(cfg, names, sb))
                wrote, shape = True, tuple(stack.shape)  # only now does sp hold this data
                break
            except Exception as e:
                # Out of RAM / out of disk will not heal in 40s; abort now. Expired creds are
                # fatal only from attempt 2 (see _is_fatal): one immediate no-sleep retry
                # distinguishes a transient 403 from a dead token, because a dead token 403s
                # again instantly while a blip usually clears.
                if _is_fatal(e, attempt):
                    print(f"[stac]  sub {k+1}/{len(subs)} {sb} FATAL "
                          f"({type(e).__name__}: {e}), not retrying", flush=True)
                    raise
                if _is_expired_creds(e):
                    print(f"[stac]  sub {k+1}/{len(subs)} got 403 on attempt {attempt} — "
                          f"retrying once immediately to tell a transient 403 from an expired "
                          f"token", flush=True)
                    continue  # no sleep: an expired token fails again at once
                # "No Sentinel-{1,2} scenes" = genuinely empty sub-tile; retrying cannot
                # help, so skip it. Match on the MESSAGE, not the type: a RuntimeError
                # raised anywhere else (dask, odc internals) must not be read as "empty",
                # or it silently punches a hole in the mosaic while the annual composite
                # still looks complete. Anything else is assumed transient and retried.
                # KeyboardInterrupt is a BaseException, so Ctrl-C still is not swallowed.
                if isinstance(e, RuntimeError) and "No Sentinel" in str(e):
                    print(f"[stac]  sub {k+1}/{len(subs)} {sb} SKIPPED ({e})", flush=True)
                    break
                if attempt == attempts:
                    # Never skip a failed tile: a hole in the mosaic that still looks
                    # like a complete annual composite is worse than a loud failure,
                    # and resume makes the re-run cheap.
                    print(f"[stac]  sub {k+1}/{len(subs)} {sb} FAILED after {attempts} "
                          f"attempts, aborting export", flush=True)
                    raise
                delay = 5 * 2 ** (attempt - 1)
                print(f"[stac]  sub {k+1}/{len(subs)} read failed (attempt {attempt}/{attempts}), "
                      f"retrying in {delay}s: {str(e)[:120]}", flush=True)
                time.sleep(delay)

        if not wrote:  # skipped: this sub-tile reported no imagery
            skipped.append(k)
            continue
        sub_paths.append(sp)
        print(f"[stac]  sub {k+1}/{len(subs)} {sb} -> {sp.name} shape={shape}", flush=True)

    if skipped:
        # Nothing downstream can see a missing sub-tile: rio_merge zero-fills the hole and
        # the mosaic still has the right bounds, right size and no nodata tag, so an
        # interior gap (this AOI is a 3x4 grid) is an ~8% block of exact zeros with no
        # record but one log line. Every sub-tile of this AOI HAS imagery, so a skip means
        # the search or the config is wrong, not that the ground is empty. Refuse to
        # mosaic; the written sub-tiles stay on disk so the re-run is cheap.
        gaps = "; ".join(f"{i:02d} {subs[i]}" for i in skipped)
        raise RuntimeError(
            f"{len(skipped)}/{len(subs)} sub-tiles reported no Sentinel imagery and were "
            f"skipped: {gaps}. Refusing to mosaic a partial AOI — the gap would be "
            f"indistinguishable from real zero reflectance. Fix the search/config (or the "
            f"AOI) and re-run; the completed sub-tiles will be reused.")
    if not sub_paths:
        raise RuntimeError("No sub-tiles exported.")

    print(f"[stac] mosaicking {len(sub_paths)} sub-tiles ...", flush=True)
    srcs = [rasterio.open(p) for p in sub_paths]
    try:
        # Explicit nodata: the sub-tiles carry none, so merge would otherwise fall back to 0
        # for every pixel no source covers (the lon/lat tile grid does not fill the UTM union
        # rectangle at the corners). NaN keeps such fill tagged and non-finite.
        mosaic, transform = rio_merge(srcs, nodata=_MOSAIC_NODATA)
        profile = srcs[0].profile.copy()
        profile.update(height=mosaic.shape[1], width=mosaic.shape[2], transform=transform,
                       count=mosaic.shape[0], compress="deflate", nodata=_MOSAIC_NODATA)
        out_path = out_dir / f"{cfg['aoi']['region']}_{cfg['year']}_annual_full.tif"
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(mosaic)
    finally:
        # Close in a finally: if merge or the write raises, Windows keeps the sub-tiles
        # locked, so the resume path cannot even delete a bad leftover on the next run.
        for s in srcs:
            s.close()
    for p in sub_paths:  # tidy intermediates (a locked leftover is not worth aborting for)
        _safe_unlink(p)
    (out_dir / f"{out_path.stem}.bands.txt").write_text("\n".join(names))
    print(f"[stac] wrote {out_path}  bands={mosaic.shape[0]}  shape={tuple(mosaic.shape)}", flush=True)
    return out_path


def export(cfg: dict, quick: bool = False, season_idx: int | None = None,
           full: bool = False) -> Path:
    """Build a stacked composite GeoTIFF. Returns the output path.

    Modes: full -> export_full() (tiled full-AOI annual mosaic); otherwise a
    single eager seasonal composite over quick_bbox (quick) or aoi.bbox.
    """
    if full:
        return export_full(cfg)

    import rioxarray  # noqa: F401

    img = cfg["imagery"]
    bbox = cfg["aoi"]["quick_bbox"] if quick else cfg["aoi"]["bbox"]
    seasons = seasonal_ranges(cfg["year"], img["n_seasonal_composites"])
    idx = 0 if (quick and season_idx is None) else (season_idx or 0)
    date_range = seasons[idx]
    tag = "quick" if quick else "season"

    print(f"[stac] backend={img['backend']} mode={tag} bbox={bbox} range={date_range}", flush=True)
    ensure_dirs(cfg)
    catalog = get_catalog(cfg)
    stack = build_stack(catalog, cfg, bbox, date_range)

    out_dir = Path(cfg["paths"]["imagery_dir"])
    out_path = out_dir / f"{cfg['aoi']['region']}_{cfg['year']}_{idx}_{tag}.tif"
    stack.rio.to_raster(out_path, driver="GTiff", compress="deflate")
    (out_dir / f"{out_path.stem}.bands.txt").write_text("\n".join(band_order(cfg)))
    print(f"[stac] wrote {out_path}  bands={stack.shape[0]}  shape={tuple(stack.shape)}", flush=True)
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Export S2+indices+S1 composite via Planetary Computer (P1a).")
    ap.add_argument("--config", default=None)
    ap.add_argument("--quick", action="store_true", help="small quick_bbox, one season (smoke test)")
    ap.add_argument("--season", type=int, default=None, help="season index (0..n-1)")
    ap.add_argument("--full", action="store_true",
                    help="full aoi.bbox, annual median composite (tiled, eager per sub-tile)")
    ap.add_argument("--year", type=int, default=None, help="override target year (v2 Part C/D)")
    ap.add_argument("--subtile-deg", type=float, default=None,
                    help="override imagery.full_subtile_deg. Smaller = more, smaller work "
                         "units. A failed read restarts its WHOLE sub-tile, so a heavy year "
                         "(2023 has ~2x the scenes of any other) may never finish a 0.3 deg "
                         "tile between failures while it finishes 0.15 deg tiles comfortably. "
                         "Does not change the output mosaic grid, only how it is assembled.")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.year:
        cfg["year"] = args.year
    if args.subtile_deg:
        cfg["imagery"]["full_subtile_deg"] = args.subtile_deg
        print(f"[stac] full_subtile_deg overridden to {args.subtile_deg} "
              f"(default {load_config(args.config)['imagery']['full_subtile_deg']})", flush=True)
    try:
        export(cfg, quick=args.quick, season_idx=args.season, full=args.full)
    except Exception as e:
        # Expired PC credentials are unrecoverable IN-PROCESS but fixed by relaunching, so
        # exit with a distinct code the supervisor can act on rather than dying like a real
        # error. Completed sub-tiles stay on disk, so the relaunch resumes rather than restarts.
        if _is_expired_creds(e):
            print(f"[stac] SAS CREDENTIALS EXPIRED — exiting {EXPIRED_CREDS_EXIT} so the "
                  f"supervisor relaunches with fresh credentials; completed sub-tiles are "
                  f"kept and will be reused: {str(e)[:140]}", flush=True)
            raise SystemExit(EXPIRED_CREDS_EXIT)
        raise
