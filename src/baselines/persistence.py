"""Persistence nulls for Track A — the no-skill floor for the SPATIAL claim (prereg A16/A19).

Track B's whole lesson was that a no-model null (N2, the historical mean) beat the network at
counting, and that this was only ever discovered because someone ran the null. Track A had no
equivalent: NDVI and the random forest are *imagery* baselines, not no-skill floors. Coca is a
standing perennial, so the obvious floor for "where is it" is **where it was**.

Three variants, all label-only (no imagery is opened at all):

* ``persistence_last_year``     P1 — the prior train year's presence mask.
* ``persistence_freq_ever``     P2 at the registered 0.02 cut => "present in ANY train year".
* ``persistence_freq_majority`` P2 at 0.5 => "present in MOST train years".

Per A19 the fold's persistence score is the **maximum** of the three: declaring the max in
advance means adding a variant can only make the U-Net's bar harder, never easier. That max is
written to the metrics sink as its own ``persistence_max`` row, because it is the quantity A16's
>=5/6 rule is applied to and prereg §8 allows no reported number to exist only in stdout.
``src.baselines.compare`` reads that row to compute the rule.

The tile grid is identical across years — verified on the gen4 index (prereg A20):
**436 positions x 6 years**, every position present in every year and carrying the same
``block_fold`` and ``split`` in every year (asserted by
``tests/test_block_folds.py::test_real_index_every_position_present_in_every_year`` and
``::test_real_index_split_matches_its_block_fold``). So a test tile's sibling in another year
covers exactly the same ground and the lookup below is exact, not approximate.

(Docstring corrected 2026-08-14: it previously read "the gen3 index: 500 positions x 6 years",
which the A20 rebuild superseded. What it asserts is checked in ``preflight`` below rather than
being taken on trust.)

Guards, per prereg A21: ``preflight`` validates EVERY fold before the first metrics row is
written, and ``src.evaluate._metrics_at`` raises on an all-negative target instead of returning
a 0/0 as 0.000.

    python -m src.baselines.persistence
"""

from __future__ import annotations

import numpy as np

from src.baselines import common as C
from src.evaluate import _metrics_at
from src.metrics_io import write_run
from src.utils import load_config

PRESENCE_THR = 0.02   # same cut as every other arm's presence_iou (src/evaluate._metrics_at)
TARGET_THR = 0.0      # ground truth: the cell has any coca
MAJORITY = 0.5

METHODS = ("persistence_last_year", "persistence_freq_ever", "persistence_freq_majority")

#: A19 makes the per-fold MAXIMUM of the three variants *the* persistence score — the
#: single quantity A16's >=5/6 rule is applied to. It is written to the sink under this
#: method name so the decision number is a row like any other, not a print. Before this
#: existed the three variant rows were persisted and the max was only ``print``ed, which
#: `src/metrics_io.py` forbids ("no metric may live only in stdout or prose") and which
#: left A16's rule computed nowhere (reviewer item S3).
SCORE_METHOD = "persistence_max"


def _by_position(rows) -> dict:
    """{(x, y, year): row}. Keyed on the tile's grid position, not its id, because the
    id is not guaranteed to be comparable across years."""
    return {(r["x"], r["y"], int(r["year"])): r for r in rows}


def _p1_year(test_year: int, train_years: list[int]) -> int:
    """The most recent train year BEFORE test_year; for 2019 there is none, so the nearest
    available year is used instead (A19 point 2 — non-causal, deliberately, since it only
    strengthens the null)."""
    prior = [y for y in train_years if y < test_year]
    if prior:
        return max(prior)
    return min(train_years, key=lambda y: (abs(y - test_year), y))


def _presence(cfg, lookup, x, y, year) -> np.ndarray:
    """Binary presence mask of the tile at grid position (x, y) in `year`."""
    row = lookup.get((x, y, year))
    if row is None:
        raise KeyError(f"no tile at position ({x},{y}) for year {year} — the annual tile "
                       f"grids are not aligned, which invalidates the sibling lookup")
    _, mask = C._tile(cfg, row)
    return (mask > TARGET_THR).astype("float32")


def preflight(rows, years) -> dict:
    """Validate EVERY fold before any metrics row is written. Raises on failure.

    Two assertions specific to this method, neither of which ``_metrics_at`` can
    make for it:

    * each fold's ``test`` row set is non-empty — an empty fold would otherwise
      reach the metric as an empty array;
    * every sibling year the rule needs is present at every test position — P1's
      source year and all five P2 years, for each of the six folds.

    It runs up front, over all folds, deliberately. ``_presence`` already raises a
    ``KeyError`` when a sibling is missing, but it does so *mid-run*, after earlier
    folds have been appended to the append-only metrics sink — leaving a record
    that is partial in a way nothing downstream can see (reviewer item S5). The
    check is the same; hoisting it means the module either writes six folds or
    writes none. Reachable without misalignment: the tile keep filter runs per
    year, so a cloudier rebuild can drop a position in one year only.

    Returns the per-fold ``{test_year: p1_year}`` map so ``run`` need not recompute it.
    """
    lookup = _by_position(rows)
    p1_years, missing, empty = {}, [], []
    for test_year in years:
        train_years = [y for y in years if y != test_year]
        te = C.rows_for(rows, [test_year], "test")
        if not te:
            empty.append(test_year)
            continue
        p1_years[test_year] = _p1_year(test_year, train_years)
        needed = {p1_years[test_year], *train_years}
        for r in te:
            for yr in needed:
                if (r["x"], r["y"], yr) not in lookup:
                    missing.append((r["x"], r["y"], yr))
    if empty:
        raise AssertionError(
            f"no test tiles for fold(s) {empty} — the persistence null would be scored on an "
            "empty row set. Check the split (prereg A20) before reporting anything")
    if missing:
        raise AssertionError(
            f"{len(missing)} (x, y, year) sibling tile(s) missing from the index, e.g. "
            f"{missing[:3]} — the annual tile grids are not aligned, which invalidates the "
            "sibling lookup. Refusing to write a partial record")
    return p1_years


def fold_predictions(cfg, rows, test_year, train_years):
    """Return {method: flat prediction array} plus the flat target, over this year's test
    tiles. Nothing here reads imagery; only label masks are touched."""
    lookup = _by_position(rows)
    te = C.rows_for(rows, [test_year], "test")
    p1_year = _p1_year(test_year, train_years)
    preds = {m: [] for m in METHODS}
    tgts = []
    for r in te:
        x, y = r["x"], r["y"]
        _, mask = C._tile(cfg, r)
        tgts.append(mask.reshape(-1))
        preds["persistence_last_year"].append(_presence(cfg, lookup, x, y, p1_year).reshape(-1))
        freq = np.mean([_presence(cfg, lookup, x, y, ty) for ty in train_years], axis=0)
        # A binary mask is emitted (1.0 where predicted present) rather than the frequency
        # itself: these are presence hypotheses, not density estimates, and feeding a
        # frequency through a density-scaled cut would be a category error.
        preds["persistence_freq_ever"].append(
            (freq > PRESENCE_THR).astype("float32").reshape(-1))
        preds["persistence_freq_majority"].append(
            (freq >= MAJORITY).astype("float32").reshape(-1))
    return ({m: np.concatenate(v) for m, v in preds.items()},
            np.concatenate(tgts), te, p1_year)


def run(cfg, years=None):
    years = years or C.YEARS
    rows = C.load_index(cfg)
    print("[persistence] Track A no-skill floor (A16/A19) — labels only, no imagery")
    p1_years = preflight(rows, years)
    print(f"[persistence] preflight OK: {len(years)} folds, non-empty test sets, "
          f"all sibling years present (P1 sources {p1_years})")
    summary = {}
    for test_year in years:
        train_years = [y for y in years if y != test_year]
        preds, tgt, te, p1_year = fold_predictions(cfg, rows, test_year, train_years)
        # S6: this method fits nothing on train-split tiles — it reads the *test*-split
        # label masks of the five other years. Recording the train-split count here (as
        # every imagery arm does) would overstate its inputs by ~1800 tiles, so the
        # honest value is 0 and the tiles it actually reads go in `extra` below.
        n_source = len(C.rows_for(rows, train_years, "test"))
        best, best_method, best_metrics, per_variant = -1.0, None, None, {}
        for method, p in preds.items():
            m = _metrics_at(p, tgt, PRESENCE_THR, t_thr=TARGET_THR)
            metrics = {"presence_iou": m["iou"], "presence_f1": m["f1"],
                       "presence_precision": m["precision"], "presence_recall": m["recall"],
                       # density metrics are undefined for a binary presence hypothesis
                       "mae": None, "rmse": None, "bias": None}
            # A21: an all-zero prediction is a legitimate result, not a broken
            # input, so _metrics_at returns it rather than raising — but it is
            # recorded here so a genuine 0.000 can never be mistaken for the 0/0
            # non-measurement that the guard now rejects.
            extra = {"uses_imagery": False, "presence_thr": PRESENCE_THR,
                     "target_thr": TARGET_THR, "prereg": "A16/A19/A21",
                     "pred_all_zero": bool(not (p > PRESENCE_THR).any()),
                     "n_target_positive": int((tgt > TARGET_THR).sum()),
                     "reads_train_split_tiles": 0,
                     "n_label_source_tiles": n_source}
            if method == "persistence_last_year":
                extra["p1_source_year"] = p1_year
                extra["p1_is_causal"] = p1_year < test_year
            if method == "persistence_freq_majority":
                extra["majority_thr"] = MAJORITY
            write_run(cfg, method, test_year, metrics, track="A",
                      n_train_tiles=0, n_test_tiles=len(te),
                      calibration_scalar=None, fit_years=train_years, extra=extra)
            print(f"  {test_year} {method:28s} IoU={m['iou']:.3f} F1={m['f1']:.3f} "
                  f"P={m['precision']:.3f} R={m['recall']:.3f}")
            per_variant[method] = m["iou"]
            if m["iou"] > best:
                best, best_method, best_metrics = m["iou"], method, m
        # A19's decision quantity, written as its own row (S3). Its F1/precision/recall
        # are the argmax variant's, not a max over variants — mixing arms per metric
        # would invent a method that was never run.
        write_run(cfg, SCORE_METHOD, test_year,
                  {"presence_iou": best, "presence_f1": best_metrics["f1"],
                   "presence_precision": best_metrics["precision"],
                   "presence_recall": best_metrics["recall"],
                   "mae": None, "rmse": None, "bias": None},
                  track="A", n_train_tiles=0, n_test_tiles=len(te),
                  calibration_scalar=None, fit_years=train_years,
                  extra={"uses_imagery": False, "presence_thr": PRESENCE_THR,
                         "target_thr": TARGET_THR, "prereg": "A16/A19/A21",
                         "is_decision_quantity": True,
                         "aggregation": "max over the three registered variants (A19)",
                         "argmax_variant": best_method,
                         "per_variant_presence_iou": per_variant,
                         "n_target_positive": int((tgt > TARGET_THR).sum()),
                         "reads_train_split_tiles": 0,
                         "n_label_source_tiles": n_source})
        summary[test_year] = best
        print(f"  {test_year} -> persistence score (max of 3) IoU={best:.3f} "
              f"[{best_method}]  [P1 used {p1_year}]")
    print("\n[persistence] per-fold bar the U-Net must beat (prereg A16):")
    for y, v in summary.items():
        print(f"        {y}: IoU={v:.3f}")
    return summary


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Persistence nulls for Track A (A16/A19).")
    ap.add_argument("--config", default=None)
    run(load_config(ap.parse_args().config))
