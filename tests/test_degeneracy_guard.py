"""Guards on the metric itself: a comparison that cannot be made must not score 0.000.

`src/evaluate.py:_metrics_at` divides by `(tp + fp + fn + 1e-6)`. With no positive
target all three counts are 0, so it returned **0.000 with no warning** —
arithmetically indistinguishable from a method that genuinely detected nothing.
That is how gen3 shipped: its `test` split held no coca, so every `presence_iou`
on it — U-Net, RF, NDVI **and** the A16 persistence nulls, all of which reach this
same function — was a non-measurement dressed as a measurement. A16 exists to stop
an unearned claim, so the guard raises (prereg **A21**).

Two things are asserted here, and the second matters as much as the first:

* the guard FIRES on a degenerate comparison — this repo has shipped four inert
  checks, so a guard that is merely present is not evidence of anything;
* the guard changes NO VALUE on valid input. `_metrics_at` is frozen by A19, and
  the expected numbers below were captured from the implementation *before* the
  guard was added.

These import `src.evaluate`, which imports torch at module scope (its `_collect`
is decorated `@torch.no_grad()`), so they `importorskip` and are inert on the
light-dependency CI runner. They enforce locally, where metrics are actually
computed. Skipped is not passed.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")            # src.evaluate imports it at module scope

from src.evaluate import (
    DegenerateComparison,
    _evaluate_regression,
    _metrics_at,
)

#: The two (thr, t_thr) pairs actually used in production: A19's presence cut for
#: Track A / persistence, and the segmentation path's tuned-threshold default.
PRODUCTION_CUTS = [(0.02, 0.0), (0.5, 0.5)]

#: Captured from `_metrics_at` BEFORE the A21 guard was added, as exact float
#: repr. A19 freezes this function, so any drift in these is a prereg violation,
#: not a rounding detail — hence exact equality rather than a tolerance.
FROZEN = [
    ("all_correct", [0.9, 0.9, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0], 0.5, 0.5,
     {"iou": 0.99999950000025, "f1": 0.9999990000004999,
      "precision": 0.99999950000025, "recall": 0.99999950000025}),
    ("all_wrong", [0.0, 0.0, 0.9, 0.9], [1.0, 1.0, 0.0, 0.0], 0.5, 0.5,
     {"iou": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0}),
    ("pred_all_zero", [0.0] * 10, [1.0] + [0.0] * 9, 0.5, 0.5,
     {"iou": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0}),
    ("thr_exact_boundary", [0.02, 0.021, 0.0], [0.0, 0.5, 1.0], 0.02, 0.0,
     {"iou": 0.499999750000125, "f1": 0.6666657777782964,
      "precision": 0.9999990000010001, "recall": 0.499999750000125}),
]


# --- the guard fires --------------------------------------------------------


@pytest.mark.parametrize("thr,t_thr", PRODUCTION_CUTS)
def test_all_negative_target_raises(thr, t_thr):
    """THE GATE: 0/0 must not come back as 0.000. This is blocker B1's mechanism."""
    probs = np.array([0.0, 0.9, 0.5, 0.01], dtype="float32")
    targets = np.zeros(4, dtype="float32")
    with pytest.raises(DegenerateComparison, match="NO positive pixel"):
        _metrics_at(probs, targets, thr, t_thr=t_thr)


def test_target_positive_only_below_the_cut_still_raises():
    """`t_thr` decides what counts as present — a target under it is still empty.

    The segmentation path uses `t_thr=0.5`, so a fraction-valued mask whose values
    all sit below 0.5 is degenerate *for that call* even though the array is not
    all zero.
    """
    targets = np.full(100, 0.3, dtype="float32")
    with pytest.raises(DegenerateComparison, match="t_thr=0.5"):
        _metrics_at(np.ones(100, dtype="float32"), targets, 0.5, t_thr=0.5)
    # ... and the same array IS valid at the presence cut, which uses t_thr=0.0.
    assert _metrics_at(np.ones(100, dtype="float32"), targets, 0.02, t_thr=0.0)["iou"] > 0


@pytest.mark.parametrize("probs,targets", [
    ([], []),
    ([], [1.0]),
    ([1.0], []),
])
def test_empty_input_raises(probs, targets):
    """An empty split is not a score of zero; it is the absence of a measurement."""
    with pytest.raises(DegenerateComparison, match="empty array"):
        _metrics_at(np.array(probs, dtype="float32"),
                    np.array(targets, dtype="float32"), 0.02, t_thr=0.0)


def test_guard_is_catchable_as_assertion_error():
    """Subclassing AssertionError keeps existing handlers and pytest.raises working."""
    assert issubclass(DegenerateComparison, AssertionError)
    with pytest.raises(AssertionError):
        _metrics_at(np.ones(4, dtype="float32"), np.zeros(4, dtype="float32"), 0.02, t_thr=0.0)


def test_evaluate_regression_propagates_the_guard():
    """The guard must reach the arms through the wrapper they actually call.

    U-Net, RF and NDVI never call `_metrics_at` directly: `track_a._per_year_write`
    -> `common.regression_metrics` -> `_evaluate_regression` -> `_metrics_at`. If
    the wrapper swallowed the error, three of the five arms would still be able to
    publish a silent 0.000.
    """
    cfg = {"model": {"task": "regression"}}
    with pytest.raises(DegenerateComparison):
        _evaluate_regression(cfg, np.zeros(64, dtype="float32"), np.zeros(64, dtype="float32"))


# --- the guard changes nothing ---------------------------------------------


@pytest.mark.parametrize("name,probs,targets,thr,t_thr,expected",
                         FROZEN, ids=[c[0] for c in FROZEN])
def test_valid_input_values_are_unchanged(name, probs, targets, thr, t_thr, expected):
    """A19 freezes this function: exact equality against pre-guard values."""
    got = _metrics_at(np.array(probs, dtype="float64"),
                      np.array(targets, dtype="float64"), thr, t_thr=t_thr)
    assert got == expected, f"{name}: metric drifted under the A21 guard"


def test_all_zero_prediction_is_a_result_not_an_error():
    """A21's deliberate asymmetry: a degenerate RESULT is reported, not rejected.

    `tp = 0` with `fn > 0` gives `iou = 0 / fn = 0` exactly — the epsilon is
    immaterial and the zero is earned. "This method detects nothing here" is a
    real finding that A16's decision rule explicitly provides for. Raising on it
    would also crash a legitimately all-zero persistence variant out of A19's
    max-of-three, which would LOWER the U-Net's bar — the one direction A19
    forbids.
    """
    targets = np.array([1.0, 1.0, 0.0, 0.0], dtype="float32")
    m = _metrics_at(np.zeros(4, dtype="float32"), targets, 0.02, t_thr=0.0)
    assert m == {"iou": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0}


def test_matches_epsilon_free_iou_on_a_production_shaped_field():
    """On valid input the epsilons must be immaterial — checked against the textbook form.

    Deliberately not a pinned magic constant: this recomputes IoU/precision/recall
    from tp/fp/fn with NO epsilon and requires `_metrics_at` to agree. It therefore
    also demonstrates what the epsilons are for — they only ever matter when a
    denominator is zero, which is exactly the case the A21 guard now rejects.
    Sparsity matches the AOI (coca is ~3.6% of pixels).
    """
    rng = np.random.default_rng(20260814)
    n = 250_000
    t = (rng.random(n) < 0.0355).astype("float32") * rng.random(n).astype("float32")
    p = np.clip(t + rng.normal(0, 0.15, n), 0, 1).astype("float32")

    pred, tt = p > 0.02, t > 0.0
    tp = int((pred & tt).sum())
    fp = int((pred & ~tt).sum())
    fn = int((~pred & tt).sum())
    assert tp and fp and fn, "degenerate sample — this test needs all three counts non-zero"

    m = _metrics_at(p, t, 0.02, t_thr=0.0)
    assert m["iou"] == pytest.approx(tp / (tp + fp + fn), rel=1e-9)
    assert m["precision"] == pytest.approx(tp / (tp + fp), rel=1e-9)
    assert m["recall"] == pytest.approx(tp / (tp + fn), rel=1e-9)


# --- persistence-specific preflight ----------------------------------------


@pytest.fixture(scope="module")
def preflight():
    pytest.importorskip("rasterio")             # src.baselines.common imports it
    from src.baselines.persistence import preflight as fn
    return fn


YEARS = [2019, 2020, 2021, 2022, 2023, 2024]


def _grid_rows(years=YEARS, positions=((0, 0), (448, 0), (0, 448)), split="test"):
    """Index-shaped rows: every position present in every year, as gen4 guarantees."""
    return [{"x": str(x), "y": str(y), "year": str(yr), "split": split,
             "npz": f"{yr}/tile_0000{i}.npz"}
            for yr in years for i, (x, y) in enumerate(positions)]


def test_preflight_passes_on_a_complete_grid(preflight):
    p1 = preflight(_grid_rows(), YEARS)
    assert set(p1) == set(YEARS)
    # A19 point 2: 2019 has no prior year, so P1 borrows 2020 (non-causal, on purpose).
    assert p1[2019] == 2020
    assert p1[2024] == 2023


def test_preflight_rejects_an_empty_test_fold(preflight):
    """A fold with no test rows would otherwise reach the metric as an empty array."""
    rows = [r for r in _grid_rows() if r["year"] != "2022"]
    with pytest.raises(AssertionError, match="no test tiles for fold"):
        preflight(rows, YEARS)


def test_preflight_rejects_a_missing_sibling_year(preflight):
    """A position present in five years and absent in one breaks the sibling lookup.

    Reachable without any misalignment: the tile keep filter runs per year, so a
    cloudier rebuild can drop a position in one year only.
    """
    rows = [r for r in _grid_rows()
            if not (r["year"] == "2021" and r["x"] == "448")]
    with pytest.raises(AssertionError, match="sibling tile"):
        preflight(rows, YEARS)


def test_preflight_runs_before_any_row_is_written(preflight):
    """It must validate ALL folds up front, not fail midway through the sink.

    `_presence` already raised on a missing sibling, but only when that fold was
    reached — after earlier folds had been appended to the append-only metrics
    sink, leaving a partial record nothing downstream can detect (item S5). Here
    the LAST fold is broken and the check must still refuse before returning.
    """
    rows = [r for r in _grid_rows()
            if not (r["year"] == "2024" and r["x"] == "0" and r["y"] == "448")]
    with pytest.raises(AssertionError, match="sibling tile"):
        preflight(rows, YEARS)
