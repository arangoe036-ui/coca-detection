"""A17/A22: the ranking is out-of-fold, its membership cannot be tuned, and rho is scipy's.

What is being defended here, in order of how badly each would have flattered the model:

* **out-of-fold coverage.** The whole point of A22 is that no pixel entering the ranking was
  trained on. That rests on two claims — the six A20 rotations partition the six
  `block_fold`s, and rotation 0 is exactly the index's own `split` column (which is what
  makes reusing `final_multiyear.pt` legitimate). Both are asserted in code and tested here.
* **membership.** The rule it replaces let the *prediction* decide which municipalities the
  model was scored on (`predicted_ha > 1.0`). The registered rule depends only on the census
  and the split geometry.
* **the metric.** A hand-rolled Spearman agreed with scipy to 1e-9 on untied input and
  disagreed under ties; ties are now surfaced rather than assumed away.
"""

from __future__ import annotations

import pytest

pytest.importorskip("rasterio")
pytest.importorskip("torch")
pytest.importorskip("scipy")

from src.data.tiling import split_for_block_fold
from src.rank_municipal import (
    MIN_OOF_PIXELS,
    N_BLOCK_FOLDS,
    _adjacent_inversions,
    _spearman,
    _ties,
    _topk_hit,
    arm_metrics,
    rotation_rows,
    select_members,
)

OFFICIAL = [10.0, 8.0, 6.0, 4.0, 2.0]


def _rows(n_folds=N_BLOCK_FOLDS, years=(2019, 2020)):
    """A synthetic index: one position per fold per year, with `split` = rotation 0."""
    out = []
    for y in years:
        for bf in range(n_folds):
            out.append({"tile_id": f"{y}_{bf}", "year": y, "x": bf * 224, "y": 0,
                        "block_fold": bf,
                        "split": split_for_block_fold(bf, n_folds, 0), "npz": "x.npz"})
    return out


# --- out-of-fold coverage --------------------------------------------------

def test_the_six_rotations_partition_the_six_block_folds():
    """Each block_fold is the test fold in exactly one rotation, and every fold is covered.

    If a fold were the test set twice, some ground would be scored twice and some never;
    if none, part of the AOI would be missing from the ranking with nothing to say so.
    """
    rows = _rows()
    test_folds = []
    for r in range(N_BLOCK_FOLDS):
        folds = {int(x["block_fold"]) for x in rotation_rows(rows, r)["test"]}
        assert len(folds) == 1, f"rotation {r} holds out {folds}, expected exactly one fold"
        test_folds += sorted(folds)
    assert sorted(test_folds) == list(range(N_BLOCK_FOLDS))
    assert len(test_folds) == len(set(test_folds))


def test_every_row_is_scored_exactly_once_across_rotations():
    rows = _rows()
    seen = [x["tile_id"] for r in range(N_BLOCK_FOLDS) for x in rotation_rows(rows, r)["test"]]
    assert sorted(seen) == sorted(x["tile_id"] for x in rows)


def test_rotation_zero_is_the_index_split_column():
    """Reusing final_multiyear.pt as rotation 0 is only valid if this holds."""
    for x in _rows():
        assert x["split"] == split_for_block_fold(int(x["block_fold"]), N_BLOCK_FOLDS, 0)


def test_a_test_row_of_one_rotation_is_never_a_train_row_of_the_same_rotation():
    rows = _rows()
    for r in range(N_BLOCK_FOLDS):
        sp = rotation_rows(rows, r)
        train_ids = {x["tile_id"] for x in sp["train"]} | {x["tile_id"] for x in sp["val"]}
        assert not train_ids & {x["tile_id"] for x in sp["test"]}


def test_missing_block_fold_column_raises_rather_than_guessing():
    rows = [{"tile_id": "a", "year": 2019, "x": 0, "y": 0, "block_fold": "", "split": "train"}]
    with pytest.raises(AssertionError, match="block_fold"):
        rotation_rows(rows, 0)


# --- membership ------------------------------------------------------------

def _acc(px_a=MIN_OOF_PIXELS, px_b=MIN_OOF_PIXELS):
    # (sum_pred, sum_official, n_pixels) per (year, muni_idx)
    return {(2023, 1): [10.0, 20.0, px_a], (2023, 2): [5.0, 5.0, px_b]}


def test_membership_excludes_a_municipality_below_the_pixel_floor():
    off = {(2023, 1): 100.0, (2023, 2): 50.0}
    rows = select_members(_acc(px_b=MIN_OOF_PIXELS - 1), 2023, ["A", "B"], off, px_m=20)
    assert [r["municipio"] for r in rows] == ["A"]


def test_membership_excludes_a_municipality_with_no_official_value():
    off = {(2023, 1): 100.0, (2023, 2): None}
    rows = select_members(_acc(), 2023, ["A", "B"], off, px_m=20)
    assert [r["municipio"] for r in rows] == ["A"]


def test_membership_does_not_depend_on_the_prediction():
    """The defect being fixed: `predicted_ha > 1.0` let the model pick its own scorecard."""
    off = {(2023, 1): 100.0, (2023, 2): 50.0}
    zero_pred = {(2023, 1): [0.0, 20.0, MIN_OOF_PIXELS], (2023, 2): [0.0, 5.0, MIN_OOF_PIXELS]}
    huge_pred = {(2023, 1): [1e6, 20.0, MIN_OOF_PIXELS], (2023, 2): [1e6, 5.0, MIN_OOF_PIXELS]}
    a = [r["municipio"] for r in select_members(zero_pred, 2023, ["A", "B"], off, px_m=20)]
    b = [r["municipio"] for r in select_members(huge_pred, 2023, ["A", "B"], off, px_m=20)]
    assert a == b == ["A", "B"]


def test_rows_are_name_sorted_so_row_order_carries_no_answer():
    off = {(2023, 1): 1.0, (2023, 2): 1.0, (2023, 3): 1.0}
    acc = {(2023, i): [float(i), 1.0, MIN_OOF_PIXELS] for i in (1, 2, 3)}
    rows = select_members(acc, 2023, ["Zulia", "Abrego", "Mutiscua"], off, px_m=20)
    assert [r["municipio"] for r in rows] == ["Abrego", "Mutiscua", "Zulia"]


def test_densities_are_means_over_the_same_pixel_set():
    """Both sides divide by the identical n_oof_px — that is what 'footprint-matched' means."""
    off = {(2023, 1): 100.0}
    rows = select_members({(2023, 1): [30.0, 60.0, 100_000]}, 2023, ["A"], off, px_m=20)
    assert rows[0]["pred_density"] == pytest.approx(30.0 / 100_000)
    assert rows[0]["official_density"] == pytest.approx(60.0 / 100_000)
    assert rows[0]["pred_oof_ha"] == pytest.approx(30.0 * 400 / 1e4)


# --- the metrics -----------------------------------------------------------

def test_spearman_matches_scipy_including_under_ties():
    from scipy.stats import spearmanr
    for pred in ([10, 6, 8, 4, 2], [1, 5, 3, 9, 7], [3, 3, 2, 1, 0.5], [10, 8, 6, 4, 2]):
        pv = [float(v) for v in pred]
        assert _spearman(pv, OFFICIAL) == pytest.approx(spearmanr(pv, OFFICIAL).statistic)


def test_perfect_and_reversed_rankings():
    assert arm_metrics(OFFICIAL, OFFICIAL)["spearman_rho"] == pytest.approx(1.0)
    assert arm_metrics(OFFICIAL, OFFICIAL)["adjacent_inversions"] == 0
    rev = arm_metrics(OFFICIAL[::-1], OFFICIAL)
    assert rev["spearman_rho"] == pytest.approx(-1.0)
    assert rev["adjacent_inversions"] == len(OFFICIAL) - 1


def test_one_adjacent_swap_costs_exactly_one_inversion():
    m = arm_metrics([10.0, 6.0, 8.0, 4.0, 2.0], OFFICIAL)
    assert m["adjacent_inversions"] == 1
    assert m["spearman_rho"] == pytest.approx(0.9)
    # ...and it moves rho by 0.1 at n=5, which is why A17 forbids quoting rho alone.
    assert abs(1.0 - m["spearman_rho"]) > 0.05


def test_topk_hit_is_set_overlap_over_k():
    assert _topk_hit([10.0, 8.0, 6.0, 4.0, 2.0], OFFICIAL, 2) == 1.0
    assert _topk_hit([8.0, 10.0, 6.0, 4.0, 2.0], OFFICIAL, 2) == 1.0   # order within k is free
    assert _topk_hit([2.0, 4.0, 6.0, 8.0, 10.0], OFFICIAL, 2) == 0.0
    assert _topk_hit([10.0, 4.0, 6.0, 8.0, 2.0], OFFICIAL, 2) == 0.5


def test_inversions_are_counted_on_adjacent_official_pairs_only():
    """A single far-apart displacement is not the same defect as several adjacent swaps."""
    assert _adjacent_inversions([2.0, 8.0, 6.0, 4.0, 10.0], OFFICIAL) == 2


def test_ties_are_reported_so_tie_sensitive_numbers_are_visible():
    assert _ties([1.0, 2.0, 3.0]) == 0
    assert _ties([1.0, 1.0, 3.0]) == 2
    assert arm_metrics([3.0, 3.0, 2.0, 1.0, 0.5], OFFICIAL)["ties_pred"] == 2
    assert arm_metrics(OFFICIAL, OFFICIAL)["ties_pred"] == 0
