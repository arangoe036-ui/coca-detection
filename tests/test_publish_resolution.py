"""A18: the publication resolution floor, proven to fire rather than merely written.

This guard protects the one irreversible mistake in the project — publishing geometry fine
enough to target individual fields. It has existed since 2026-08-10 and had **never
executed**, because `rio_cogeo` was not installed and `write_cog` raised
`ModuleNotFoundError` first. The fix at the time was to move the checks above the import;
these tests are what makes that fix verifiable, and they are the reason the dependency is
now pinned in `requirements.txt` instead of being absent.

`nowcast.write_cog` takes `long_side` as a **pixel count**, so "75 m" is arithmetic on the
current AOI's width, not a property of the function. Both failure directions are covered:
too few pixels to downsample (upsampling past native 20 m) and enough pixels but a cut that
still lands finer than the floor.
"""

from __future__ import annotations

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
pytest.importorskip("rio_cogeo")
from affine import Affine

from src.nowcast import MIN_PUBLISH_RES_M, write_cog

# The production canvas: six identical annual mosaics, 20 m, EPSG:32618.
CANVAS_H, CANVAS_W = 5628, 5051
NATIVE_RES = 20.0
PROD_LONG_SIDE = 1500


def _profile(res=NATIVE_RES):
    return {"crs": rasterio.crs.CRS.from_epsg(32618),
            "transform": Affine(res, 0, 686060.0, 0, -res, 1029740.0)}


def test_the_floor_is_75_m():
    """A number the harm-reduction claim rests on, so it is pinned by a test, not a comment."""
    assert MIN_PUBLISH_RES_M == 75.0


def test_refuses_to_upsample_past_native_resolution(tmp_path):
    """A narrower AOI (`region: tumaco` is a live config switch) makes factor < 1."""
    out = tmp_path / "up.tif"
    with pytest.raises(ValueError, match="UPSAMPLE"):
        write_cog(np.zeros((800, 800), "float32"), _profile(), out, long_side=PROD_LONG_SIDE)
    assert not out.exists(), "refused, but still wrote a file"


def test_refuses_an_output_finer_than_the_floor_even_when_downsampling(tmp_path):
    """The subtler case: a genuine downsample (factor 1.5) that still lands at 30 m."""
    out = tmp_path / "fine.tif"
    with pytest.raises(ValueError, match="finer than the 75 m publication floor"):
        write_cog(np.zeros((3000, 3000), "float32"), _profile(), out, long_side=2000)
    assert not out.exists()


def test_production_geometry_writes_a_cog_at_or_above_the_floor(tmp_path):
    """The positive control: the guard is a floor, not a blanket refusal.

    Without this, all three tests above would pass on a `write_cog` that raised
    unconditionally — which is the same defect class as a guard that never fires.
    """
    dens = np.zeros((CANVAS_H, CANVAS_W), "float32")
    dens[2000:2100, 2000:2100] = 0.4
    out = tmp_path / "ok_cog.tif"
    write_cog(dens, _profile(), out, long_side=PROD_LONG_SIDE)
    with rasterio.open(out) as ds:
        assert ds.transform.a >= MIN_PUBLISH_RES_M, f"published at {ds.transform.a:.2f} m"
        assert ds.transform.a == pytest.approx(75.05, abs=0.1)
        assert max(ds.width, ds.height) == PROD_LONG_SIDE
        assert ds.overviews(1), "no overviews — not actually a COG"
        assert float(ds.read(1).max()) > 0, "density averaged away to nothing"


def test_quantised_output_round_trips_within_tolerance(tmp_path):
    """uint16 quantisation is a transport optimisation and must not change what is shown.

    The published rasters are what a remote viewer waits on, so they ship as uint16 holding
    round(fraction * scale) — 2.5 MB instead of 7.6 MB each. The scale is published in
    metrics.json; if writer and reader ever disagree the map divides by the wrong constant
    and renders empty, so the contract is pinned here.
    """
    scale = 10_000
    dens = np.zeros((CANVAS_H, CANVAS_W), "float32")
    dens[1000:1400, 1000:1400] = 0.2818          # the real maximum cover in the gen4 rasters
    dens[2000:2400, 2000:2400] = 0.0137
    out = tmp_path / "q.tif"
    write_cog(dens, _profile(), out, long_side=PROD_LONG_SIDE, quantize_scale=scale)
    with rasterio.open(out) as ds:
        a = ds.read(1)
        assert ds.dtypes[0] == "uint16", ds.dtypes
        assert ds.tags(1).get("DENSITY_SCALE") == str(scale) or \
               ds.tags().get("DENSITY_SCALE") == str(scale), "scale not recorded in the file"
        assert ds.transform.a >= MIN_PUBLISH_RES_M      # the A18 guard still applies
    back = a.astype("float64") / scale
    assert back.max() == pytest.approx(0.2818, abs=1e-4)
    # the smaller patch survives too: quantisation must not erase low-cover ground
    assert (back > 0.01).any()


def test_unquantised_output_is_still_float32(tmp_path):
    """The default path is unchanged — `nowcast` and any other caller keep float32."""
    dens = np.zeros((CANVAS_H, CANVAS_W), "float32")
    dens[1000:1100, 1000:1100] = 0.15
    out = tmp_path / "f.tif"
    write_cog(dens, _profile(), out, long_side=PROD_LONG_SIDE)
    with rasterio.open(out) as ds:
        assert ds.dtypes[0] == "float32"
        assert float(ds.read(1).max()) == pytest.approx(0.15, abs=1e-3)
