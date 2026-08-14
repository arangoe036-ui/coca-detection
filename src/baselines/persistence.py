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
advance means adding a variant can only make the U-Net's bar harder, never easier.

The tile grid is identical across years (verified on the gen3 index: 500 positions x 6 years,
and every position's split is the same in every year), so a test tile's sibling in another year
covers exactly the same ground and the lookup below is exact, not approximate.

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
    summary = {}
    for test_year in years:
        train_years = [y for y in years if y != test_year]
        preds, tgt, te, p1_year = fold_predictions(cfg, rows, test_year, train_years)
        n_train = len(C.rows_for(rows, train_years, "train"))
        best = -1.0
        for method, p in preds.items():
            m = _metrics_at(p, tgt, PRESENCE_THR, t_thr=TARGET_THR)
            metrics = {"presence_iou": m["iou"], "presence_f1": m["f1"],
                       "presence_precision": m["precision"], "presence_recall": m["recall"],
                       # density metrics are undefined for a binary presence hypothesis
                       "mae": None, "rmse": None, "bias": None}
            extra = {"uses_imagery": False, "presence_thr": PRESENCE_THR,
                     "target_thr": TARGET_THR, "prereg": "A16/A19"}
            if method == "persistence_last_year":
                extra["p1_source_year"] = p1_year
                extra["p1_is_causal"] = p1_year < test_year
            if method == "persistence_freq_majority":
                extra["majority_thr"] = MAJORITY
            write_run(cfg, method, test_year, metrics, track="A",
                      n_train_tiles=n_train, n_test_tiles=len(te),
                      calibration_scalar=None, fit_years=train_years, extra=extra)
            print(f"  {test_year} {method:28s} IoU={m['iou']:.3f} F1={m['f1']:.3f} "
                  f"P={m['precision']:.3f} R={m['recall']:.3f}")
            best = max(best, m["iou"])
        summary[test_year] = best
        print(f"  {test_year} -> persistence score (max of 3) IoU={best:.3f}  "
              f"[P1 used {p1_year}]")
    print("\n[persistence] per-fold bar the U-Net must beat (prereg A16):")
    for y, v in summary.items():
        print(f"        {y}: IoU={v:.3f}")
    return summary


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Persistence nulls for Track A (A16/A19).")
    ap.add_argument("--config", default=None)
    run(load_config(ap.parse_args().config))
