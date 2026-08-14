"""Guards on the metrics sink.

Five results documents cited ``outputs/metrics/baseline_ladder.jsonl`` while that file did
not exist anywhere on disk, so every published number was unbacked — a reader (or the author
on a new machine) could not check a single figure. The sink itself was fine; nothing asserted
it was actually used, and nothing recorded WHICH generation of the data a row came from.

Three data generations now exist and are not comparable:
  gen1  original (reflectance offset bug + 25% blank coverage + block-straddle leak)
  gen2  offset fixed only
  gen3  offset + coverage fixed, leak-free splits
Without the stamp, a gen1 row and a gen3 row are indistinguishable in the file.
"""

from __future__ import annotations

import json

import pytest

from src.metrics_io import JSONL_NAME, read_runs, write_run
from src.utils import load_config


@pytest.fixture()
def cfg_tmp(tmp_path):
    """Real config, but with outputs redirected into tmp so nothing touches the repo."""
    cfg = load_config()
    cfg["paths"] = dict(cfg["paths"])
    cfg["paths"]["outputs_dir"] = str(tmp_path / "outputs")
    return cfg


def _write(cfg, **kw):
    base = dict(method="unet", fold_year=2020, metrics={"iou": 0.5},
                track="A", n_train_tiles=10, n_test_tiles=2,
                calibration_scalar=None, fit_years=[2019, 2021])
    base.update(kw)
    return write_run(cfg, **base)


def test_write_then_read_round_trips(cfg_tmp):
    path = _write(cfg_tmp)
    assert path.name == JSONL_NAME
    runs = read_runs(cfg_tmp)
    assert len(runs) == 1
    assert runs[0]["method"] == "unet"
    assert runs[0]["metrics"]["iou"] == pytest.approx(0.5)


def test_every_row_records_its_data_generation(cfg_tmp):
    """THE GATE: a number with no generation stamp is not comparable to anything."""
    _write(cfg_tmp)
    row = read_runs(cfg_tmp)[0]
    assert "data_generation" in row, "metrics row has no data_generation stamp"
    assert row["data_generation"] == str(cfg_tmp["project"]["data_generation"])
    assert row["data_generation"], "data_generation must not be empty"


def test_sink_is_append_only(cfg_tmp):
    """Runs accumulate; a re-run must never silently overwrite earlier evidence."""
    _write(cfg_tmp, fold_year=2019)
    _write(cfg_tmp, fold_year=2020)
    _write(cfg_tmp, fold_year=2021)
    runs = read_runs(cfg_tmp)
    assert [r["fold_year"] for r in runs] == [2019, 2020, 2021]


def test_rows_carry_provenance(cfg_tmp):
    """Each row must be traceable to code and time, not just to a number."""
    _write(cfg_tmp)
    row = read_runs(cfg_tmp)[0]
    for field in ("git_commit", "timestamp", "fit_years", "track",
                  "n_train_tiles", "n_test_tiles"):
        assert field in row, f"provenance field {field!r} missing"


def test_rows_are_one_json_object_per_line(cfg_tmp):
    """The file must stay greppable/streamable line-by-line."""
    _write(cfg_tmp, fold_year=2019)
    _write(cfg_tmp, fold_year=2020)
    text = (_write(cfg_tmp, fold_year=2021)).read_text()
    lines = [ln for ln in text.splitlines() if ln.strip()]
    assert len(lines) == 3
    for ln in lines:
        assert json.loads(ln)["method"] == "unet"


def test_read_runs_is_empty_when_absent(cfg_tmp):
    """Absent sink reads as empty rather than raising — but see the module docstring:
    empty means UNBACKED, and no results doc may cite numbers while this is empty."""
    assert read_runs(cfg_tmp) == []
