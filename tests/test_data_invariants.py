"""Per-year invariants on the annual composites.

BOTH data defects that invalidated this project's published results would have been caught
here, by two assertions:

* ``BOA_ADD_OFFSET``: every visible band stepped +0.100 at exactly 2021->2022 (= 1000/10000)
  because the DN->reflectance conversion ignored ESA Processing Baseline 04.00.
  -> ``test_no_reflectance_step_between_adjacent_years``
* ``scl_mask_classes`` omitting SCL 0 (NO_DATA) / 1 (SATURATED): nodata pixels took a
  reflectance median of exactly 0.0, silently blanking 25.3% of the AOI in 2019/2020 and
  12.2% in 2021 while 2022-2024 sat at 0.7%.
  -> ``test_blank_fraction_under_bar`` and ``test_blank_fraction_consistent_across_years``

These read the real composites and SKIP when they are absent, so the suite stays runnable on
a fresh clone with no data. Skipped is not passed: the gate only has force once data exists.
"""

from __future__ import annotations

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")

from src.utils import load_config  # noqa: E402

YEARS = [2019, 2020, 2021, 2022, 2023, 2024]
N_S2 = 10          # s2(10) + indices(6) + s1(2)
VISIBLE = ("B02", "B03", "B04")
DECIMATION = 8

#: Prereg A13: a corrected year must be under this. 2019/2020 were at 25.3% before the fix.
BLANK_PCT_BAR = 5.0
#: Prereg A15: adjacent-year median shift beyond this exceeds all natural interannual
#: variation (the observed full range of B02 medians across six years is 0.0077), so it
#: indicates a preprocessing artifact rather than weather. The bug produced +0.1000.
MAX_ADJACENT_STEP = 0.010


def _img_dir():
    return __import__("pathlib").Path(load_config()["paths"]["imagery_dir"])


def _path(year):
    return _img_dir() / f"catatumbo_{year}_annual_full.tif"


def _available():
    return [y for y in YEARS if _path(y).exists()]


@pytest.fixture(scope="module")
def stats():
    years = _available()
    if not years:
        pytest.skip("no annual composites on disk (fresh clone) — nothing to check")
    out = {}
    for y in years:
        p = _path(y)
        names = (p.parent / f"{p.stem}.bands.txt").read_text().split()
        with rasterio.open(p) as src:
            h, w = src.height // DECIMATION, src.width // DECIMATION
            arr = src.read(list(range(1, N_S2 + 1)),
                           out_shape=(N_S2, h, w), masked=True).filled(np.nan)
        blank = (~np.isfinite(arr)).all(axis=0) | (np.nan_to_num(arr, nan=0.0) == 0.0).all(axis=0)
        med = {}
        for i, nm in enumerate(names[:N_S2]):
            v = arr[i][~blank]
            v = v[np.isfinite(v)]
            med[nm] = float(np.median(v)) if v.size else float("nan")
        out[y] = {"blank_pct": 100 * float(blank.mean()), "median": med, "names": names}
    return out


def test_band_order_matches_sidecar_every_year(stats):
    """A silently mis-indexed channel produces plausible garbage, not an error."""
    ref = None
    for y, s in stats.items():
        assert len(s["names"]) == 18, f"{y}: {len(s['names'])} bands, expected 18"
        if ref is None:
            ref = s["names"]
        assert s["names"] == ref, f"{y}: band order differs from {min(stats)}"
    assert ref[:N_S2][0] == "B02" and ref[10] == "NDVI" and ref[16] == "VV"


def test_blank_fraction_under_bar(stats):
    """No year may have a large unobserved region masquerading as data (prereg A13)."""
    bad = {y: s["blank_pct"] for y, s in stats.items() if s["blank_pct"] > BLANK_PCT_BAR}
    assert not bad, (
        f"years exceeding {BLANK_PCT_BAR}% blank: "
        + ", ".join(f"{y}={p:.2f}%" for y, p in sorted(bad.items()))
        + " — this is the defect that blanked 25% of 2019/2020")


def test_blank_fraction_consistent_across_years(stats):
    """A year-VARYING footprint is what made cross-year comparison invalid: each year's
    statistics were computed over a different region, and footprint size correlated with
    the target."""
    if len(stats) < 2:
        pytest.skip("need >=2 years to compare footprints")
    vals = {y: s["blank_pct"] for y, s in stats.items()}
    spread = max(vals.values()) - min(vals.values())
    assert spread <= BLANK_PCT_BAR, (
        f"blank-fraction spread {spread:.2f}pp across years "
        f"({', '.join(f'{y}={p:.2f}%' for y, p in sorted(vals.items()))}) — "
        "years are not measured over the same footprint")


def test_no_reflectance_step_between_adjacent_years(stats):
    """The BOA_ADD_OFFSET signature: a uniform jump in every visible band at one year
    boundary. Weather moves these medians by <0.008 across all six years."""
    years = sorted(stats)
    if len(years) < 2:
        pytest.skip("need >=2 adjacent years")
    failures = []
    for a, b in zip(years, years[1:]):
        if b - a != 1:
            continue
        for nm in VISIBLE:
            ma, mb = stats[a]["median"].get(nm), stats[b]["median"].get(nm)
            if ma is None or mb is None or not (np.isfinite(ma) and np.isfinite(mb)):
                continue
            if abs(mb - ma) > MAX_ADJACENT_STEP:
                failures.append(f"{nm} {a}->{b}: {mb - ma:+.4f}")
    assert not failures, (
        "visible-band medians step by more than "
        f"{MAX_ADJACENT_STEP} between adjacent years: " + "; ".join(failures)
        + " — the offset bug produced +0.1000 here")


def test_reflectance_medians_physically_plausible(stats):
    """Catches a units error or a wrong scale factor outright."""
    for y, s in stats.items():
        red, nir = s["median"].get("B04"), s["median"].get("B08")
        assert 0.0 < red < 0.25, f"{y}: B04 median {red} implausible for vegetated terrain"
        assert 0.10 < nir < 0.60, f"{y}: B08 median {nir} implausible"
        assert nir > red, f"{y}: NIR ({nir}) must exceed red ({red}) over vegetation"
