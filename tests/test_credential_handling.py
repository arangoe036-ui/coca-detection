"""Guards on expired-credential handling and the publication resolution floor.

Both were shipped untested on 2026-08-11 and a review found real bugs in both:

* ``_is_expired_creds`` matched a bare ``"403"`` anywhere in the message. Exception text
  embeds the failing asset href and Sentinel scene IDs embed dates, so
  ``S2A_MSIL2A_20190403T150719`` made a **404** look like expired credentials — which would
  mislabel an ordinary transient error and trigger an immediate relaunch, discarding up to an
  hour of in-progress work.
* the first 403 was treated as fatal, converting a recoverable blip into a lost sub-tile.

``write_cog``'s 75 m floor was also untested; its checks additionally have to sit BEFORE the
optional imports, because ``rio_cogeo`` is not installed and importing first made the guard
unreachable.
"""

from __future__ import annotations

import errno

import pytest

from src.data.stac_export import (
    EXPIRED_CREDS_EXIT,
    _is_expired_creds,
    _is_fatal,
)

rasterio_errors = pytest.importorskip("rasterio.errors")
RasterioIOError = rasterio_errors.RasterioIOError
WarpOperationError = rasterio_errors.WarpOperationError

# A real href from the 2026-08-10 logs: the scene ID contains "0403".
APRIL_HREF = ("https://sentinel2l2a01.blob.core.windows.net/sentinel2-l2/18/P/YQ/2019/04/03/"
              "S2A_MSIL2A_20190403T150719_N0212_R039_T18PYQ/B02.tif")

EXPIRED = [
    RasterioIOError("HTTP response code: 403"),
    RasterioIOError(f"HTTP response code: 403 for {APRIL_HREF}"),
    Exception("403 Forbidden"),
    Exception("<Error><Code>AuthenticationFailed</Code><Message>Signature not valid in the "
              "specified time frame</Message></Error>"),
]

NOT_EXPIRED = [
    # THE REGRESSION: a 404 whose href contains "0403".
    RasterioIOError(f"HTTP response code: 404 for {APRIL_HREF}"),
    RasterioIOError(f"HTTP response code: 429 for {APRIL_HREF}"),
    RasterioIOError(f"HTTP response code: 503 for {APRIL_HREF}"),
    WarpOperationError("Chunk and warp failed"),
    RasterioIOError("Read failed. See previous exception for details."),
    RuntimeError("No Sentinel-2 scenes for ('2020-01-01', '2020-12-31')"),
    # a byte count that merely contains 403
    RasterioIOError("TIFFFillTile:Read error; got 0 bytes, expected 40312"),
]


@pytest.mark.parametrize("exc", EXPIRED, ids=lambda e: type(e).__name__ + ":" + str(e)[:28])
def test_expired_creds_detected(exc):
    assert _is_expired_creds(exc) is True


@pytest.mark.parametrize("exc", NOT_EXPIRED, ids=lambda e: type(e).__name__ + ":" + str(e)[:28])
def test_non_credential_errors_not_misread(exc):
    """A 404/429/503 on an early-April scene must NOT be reported as expired credentials."""
    assert _is_expired_creds(exc) is False


def test_first_403_is_retryable_but_second_is_fatal():
    """One immediate retry separates a transient 403 from a dead token; the first must not
    abort the sub-tile, the second must not sleep through 40s of pointless backoff."""
    exc = RasterioIOError("HTTP response code: 403")
    assert _is_fatal(exc, attempt=1) is False, "first 403 should be retried once"
    assert _is_fatal(exc, attempt=2) is True, "second 403 means the token is dead"


def test_memory_and_disk_are_always_fatal_even_on_attempt_one():
    assert _is_fatal(MemoryError(), attempt=1) is True
    assert _is_fatal(OSError(errno.ENOSPC, "No space left on device"), attempt=1) is True


def test_transient_errors_never_fatal():
    for exc in (WarpOperationError("Chunk and warp failed"),
                RasterioIOError("Read failed. See previous exception for details.")):
        for attempt in (1, 2, 5):
            assert _is_fatal(exc, attempt=attempt) is False


def test_exit_code_is_stable_and_matches_the_supervisor():
    """The supervisor keys on this integer; a silent change breaks the relaunch contract."""
    from pathlib import Path

    from src.utils import REPO_ROOT

    assert EXPIRED_CREDS_EXIT == 75
    ps = Path(REPO_ROOT, "scripts", "run_export.ps1")
    if not ps.exists():
        pytest.skip("supervisor script absent")
    assert f"= {EXPIRED_CREDS_EXIT}" in ps.read_text(encoding="utf-8"), \
        "scripts/run_export.ps1 no longer agrees with EXPIRED_CREDS_EXIT"


# ---- publication resolution floor (prereg A18) --------------------------------------------

def test_write_cog_refuses_to_upsample_and_to_go_below_75m():
    """The 75 m floor was emergent arithmetic, never asserted. `long_side` is a PIXEL COUNT,
    so a narrower AOI silently produced finer-than-native output on a tracked path."""
    np = pytest.importorskip("numpy")
    affine = pytest.importorskip("affine")
    from pathlib import Path

    from src.nowcast import MIN_PUBLISH_RES_M, write_cog

    def prof(res):
        return {"crs": "EPSG:32618", "transform": affine.Affine(res, 0, 0, 0, -res, 0)}

    assert MIN_PUBLISH_RES_M == 75.0

    # narrow AOI -> factor < 1 -> would upsample past native 20 m
    with pytest.raises(ValueError, match="UPSAMPLE"):
        write_cog(np.zeros((1200, 1000), "float32"), prof(20.0), Path("x.tif"), 1500)

    # full AOI but too many output pixels -> ~37 m, below the floor
    with pytest.raises(ValueError, match="finer than"):
        write_cog(np.zeros((5628, 5051), "float32"), prof(20.0), Path("x.tif"), 3000)


def test_write_cog_guard_runs_before_optional_imports():
    """`rio_cogeo` is not installed. If the guard sat after the imports it would be
    unreachable — ModuleNotFoundError would fire first and nothing would be checked."""
    np = pytest.importorskip("numpy")
    affine = pytest.importorskip("affine")
    from pathlib import Path

    from src.nowcast import write_cog

    prof = {"crs": "EPSG:32618", "transform": affine.Affine(20.0, 0, 0, 0, -20.0, 0)}
    # Must raise ValueError (the guard), NOT ModuleNotFoundError (imports winning the race).
    with pytest.raises(ValueError):
        write_cog(np.zeros((1200, 1000), "float32"), prof, Path("x.tif"), 1500)
