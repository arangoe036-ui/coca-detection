"""A16's decision rule is computed in code, on one data generation, or not at all.

Three defects converge here, and all three were "the number looked fine" failures:

* the per-fold persistence MAX — the quantity A19 makes *the* persistence score — was
  only ``print``ed, so A16's >=5/6 rule was computed nowhere and no reported figure
  traced to a sink row (reviewer S3, prereg §8);
* ``compare._dedup_last`` keyed on ``(method, track, fold_year)`` and ignored
  ``data_generation``, so a gen4 row silently replaced a gen3 one of the same name.
  Harmless-sounding, but it is the exact mechanism for a **false 6/6**: gen3's test
  split held no coca, so its persistence score is ``0.000``, and pairing that against a
  gen4 U-Net reads as a clean sweep for the model;
* Track B's U-Net row is a frozen **retracted** gen1 dict, so emitting that table
  unconditionally hardcodes the same cross-generation pairing.

Every test below asserts on the verdict *text*, because the verdict is the deliverable —
a rule that computes silently and is written up by hand is the failure mode being fixed.
"""

from __future__ import annotations

import pytest

pytest.importorskip("rasterio")
pytest.importorskip("torch")

from src.baselines.compare import (
    PERSISTENCE_SCORE,
    _dedup_last,
    _one_generation,
    a16_verdict,
)

YEARS = [2019, 2020, 2021, 2022, 2023, 2024]


def row(method, year, iou, gen="gen4", track="A", **extra):
    """One sink record, shaped exactly as `metrics_io.write_run` writes it."""
    return {"data_generation": gen, "method": method, "track": track, "fold_year": year,
            "n_train_tiles": 0, "n_test_tiles": 80, "calibration_scalar": None,
            "fit_years": [y for y in YEARS if y != year], "git_commit": "deadbee",
            "timestamp": "2026-08-18T00:00:00+00:00",
            "metrics": {"presence_iou": iou, "presence_f1": iou, "mae": None},
            "extra": {"argmax_variant": "persistence_freq_ever", **extra}}


def ladder(unet_ious, pers_ious, gen="gen4"):
    """A full Track A pair of arms: {year: iou} for the U-Net and the persistence score."""
    return ([row("unet", y, v, gen) for y, v in unet_ious.items()]
            + [row(PERSISTENCE_SCORE, y, v, gen) for y, v in pers_ious.items()])


def cfg_for(gen):
    return {"project": {"data_generation": gen}}


# --- the >=5/6 rule and its three registered bands ---------------------------

def test_five_of_six_earns_the_spatial_claim():
    runs = ladder({y: 0.70 for y in YEARS} | {2020: 0.10},
                  {y: 0.60 for y in YEARS})
    text = "\n".join(a16_verdict(runs))
    assert "5/6" in text
    assert "EARNED" in text


def test_four_of_six_is_indistinguishable_not_a_win():
    """The band boundary. 4/6 must NOT read as a win — this is the case a hand-written
    write-up is most likely to round up."""
    runs = ladder({y: 0.70 for y in YEARS} | {2020: 0.10, 2021: 0.10},
                  {y: 0.60 for y in YEARS})
    text = "\n".join(a16_verdict(runs))
    assert "4/6" in text
    assert "INDISTINGUISHABLE" in text
    assert "EARNED" not in text


def test_two_of_six_is_not_publishable_as_a_model_result():
    runs = ladder({y: 0.10 for y in YEARS} | {2019: 0.90, 2020: 0.90},
                  {y: 0.60 for y in YEARS})
    text = "\n".join(a16_verdict(runs))
    assert "2/6" in text
    assert "NOT PUBLISHABLE" in text
    assert "retune" in text  # A16 forbids responding by retuning; the text must say so


def test_an_exact_tie_counts_as_a_loss_for_the_unet():
    """The null is the incumbent: "no better than where it was last year" is not skill."""
    runs = ladder({y: 0.60 for y in YEARS}, {y: 0.60 for y in YEARS})
    text = "\n".join(a16_verdict(runs))
    assert "0/6" in text
    assert "NOT PUBLISHABLE" in text


# --- the rule refuses to run on inputs it cannot judge -----------------------

def test_no_persistence_rows_reports_not_computed_not_a_sweep():
    """Absent a floor the answer is "unknown", never "the U-Net won every fold"."""
    runs = [row("unet", y, 0.70) for y in YEARS]
    text = "\n".join(a16_verdict(runs))
    assert "NOT COMPUTED" in text
    assert "EARNED" not in text
    assert "6/6" not in text


def test_missing_fold_states_the_real_denominator():
    unet = {y: 0.70 for y in YEARS}
    pers = {y: 0.60 for y in YEARS if y != 2024}   # one arm short
    text = "\n".join(a16_verdict(ladder(unet, pers)))
    assert "5/5" in text
    assert "not 6" in text          # the denominator is disclosed, not silently rescaled
    assert "excluded (missing arm)" in text


# --- generations must never be paired ---------------------------------------

def test_dedup_keeps_one_row_per_generation():
    """A gen4 re-run supersedes the previous gen4 row, and only that."""
    runs = [row("unet", 2023, 0.40, "gen3"),
            row("unet", 2023, 0.50, "gen4"),
            row("unet", 2023, 0.55, "gen4")]
    kept = _dedup_last(runs)
    assert len(kept) == 2
    by_gen = {r["data_generation"]: r["metrics"]["presence_iou"] for r in kept}
    assert by_gen == {"gen3": 0.40, "gen4": 0.55}


def test_one_generation_excludes_other_generations():
    runs = ladder({y: 0.70 for y in YEARS}, {y: 0.60 for y in YEARS}, gen="gen4")
    runs += [row("unet", 2023, 0.474, "gen1"), row("unet", 2023, 0.0, "gen3")]
    kept, gen, dropped = _one_generation(runs, cfg_for("gen4"))
    assert gen == "gen4"
    assert dropped == {"gen1": 1, "gen3": 1}
    assert all(r["data_generation"] == "gen4" for r in kept)


def test_rows_with_no_generation_stamp_are_dropped():
    """Unstamped rows predate the provenance field and are untrustworthy by default."""
    stamped = row("unet", 2023, 0.50, "gen4")
    unstamped = {k: v for k, v in row("unet", 2022, 0.90).items()
                 if k != "data_generation"}
    kept, _, dropped = _one_generation([stamped, unstamped], cfg_for("gen4"))
    assert kept == [stamped]
    assert sum(dropped.values()) == 1


def test_no_rows_for_the_current_generation_raises():
    runs = [row("unet", y, 0.70, "gen3") for y in YEARS]
    with pytest.raises(SystemExit, match="gen4"):
        _one_generation(runs, cfg_for("gen4"))


def test_a_gen3_persistence_zero_cannot_manufacture_a_false_sweep():
    """The regression this whole file exists for.

    gen3's empty test fold makes its persistence score 0.000. Feed that alongside real
    gen4 U-Net rows: the pipeline must reach "NOT COMPUTED" (no gen4 floor exists yet),
    never "6/6 EARNED".
    """
    runs = [row("unet", y, 0.70, "gen4") for y in YEARS]
    runs += [row(PERSISTENCE_SCORE, y, 0.0, "gen3") for y in YEARS]
    kept, _, dropped = _one_generation(runs, cfg_for("gen4"))
    text = "\n".join(a16_verdict(_dedup_last(kept)))
    assert dropped == {"gen3": 6}
    assert "NOT COMPUTED" in text
    assert "EARNED" not in text


# --- the checkpoint's generation must match the index's ----------------------

def test_track_a_refuses_a_checkpoint_from_another_generation(tmp_path, monkeypatch):
    """A stale checkpoint predicts happily and `write_run` stamps the CURRENT generation,
    so the resulting row claims to be a gen4 measurement of a model whose train blocks
    were gen4's *test* blocks. No downstream filter can see that — the arm has to refuse
    it, and the guard has to be proven to fire (four inert checks have shipped here).
    """
    torch = pytest.importorskip("torch")
    from src.baselines import track_a

    ck = tmp_path / "stale.pt"
    torch.save({"model": {}, "year_stats": {}, "data_generation": "gen3"}, ck)
    monkeypatch.setattr(track_a, "CKPT", str(ck))
    with pytest.raises(AssertionError, match="gen3"):
        track_a._unet_predictor(cfg_for("gen4"), "cpu")


def test_track_a_refuses_a_checkpoint_with_no_generation_stamp(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from src.baselines import track_a

    ck = tmp_path / "unstamped.pt"
    torch.save({"model": {}, "year_stats": {}}, ck)
    monkeypatch.setattr(track_a, "CKPT", str(ck))
    with pytest.raises(AssertionError, match="None"):
        track_a._unet_predictor(cfg_for("gen4"), "cpu")
