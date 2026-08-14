"""Test-wide isolation guards.

These tests must never touch the real dataset: a multi-year export writes under
``data/`` while development continues, and an earlier accident destroyed a live
sub-tile because a test ran with the default region/year. Two guards:

* every test runs with its cwd set to a throwaway ``tmp_path``, so any relative
  write lands in pytest's temp dir instead of the repo;
* ``TEST_REGION`` is the only region name tests may ever write under, and
  ``test_isolation.py`` asserts nothing matching it exists below ``data/``.
"""

from __future__ import annotations

import pytest

# Throwaway region name. Nothing in the production config uses it, so a stray
# write is detectable (and never collides with catatumbo/tumaco outputs).
TEST_REGION = "pytest_synthetic_region"


@pytest.fixture(autouse=True)
def _cwd_in_tmp(tmp_path, monkeypatch):
    """Run every test from an empty temp dir (no relative write can hit the repo)."""
    monkeypatch.chdir(tmp_path)
