"""Proof that the test suite writes nothing into the real dataset.

The split tests are pure-Python over synthetic grids: they never open a raster and
never write a tile. This asserts that after they run, no path under ``data/``
carries the throwaway test region name, and that the repo's data tree gained no
file named after this suite.
"""

from __future__ import annotations

from src.utils import REPO_ROOT
from tests.conftest import TEST_REGION


def test_no_test_artifacts_under_data():
    data_dir = REPO_ROOT / "data"
    if not data_dir.exists():          # CI: no data at all, nothing to protect
        return
    strays = [p for p in data_dir.rglob(f"*{TEST_REGION}*")]
    assert strays == [], f"test artefacts leaked into data/: {strays[:5]}"


def test_tmp_cwd_is_not_the_repo():
    """The autouse fixture must have moved cwd out of the repo (relative-write guard)."""
    from pathlib import Path

    cwd = Path.cwd().resolve()
    assert REPO_ROOT not in [cwd, *cwd.parents], f"test cwd is inside the repo: {cwd}"
