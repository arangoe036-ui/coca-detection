"""Phase 4 — assemble the comparison tables from baseline_ladder.jsonl.

Emits outputs/metrics/ladder_table.md (+ .csv) with Track A (spatial) and Track B
(temporal aoi_ratio incl. nulls), per-fold rows AND aggregates, plus the
pre-registered paired per-fold deltas vs the U-Net (prereg §5/A3). No p-values or
CIs at n=6 (prereg §5). Every cell traces to a JSONL row.

    python -m src.baselines.compare
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.baselines.common import YEARS
from src.metrics_io import read_runs
from src.utils import load_config

# Frozen U-Net Track B aoi_ratio (docs/v2.1_loyo_results.md) — the only per-fold
# U-Net magnitude numbers on disk (prereg §4).
UNET_B = {2019: 0.95, 2020: 0.59, 2021: 0.90, 2022: 1.48, 2023: 1.01, 2024: 0.79}

B_METHODS = ["unet", "random_forest", "ndvi_threshold",
             "null_historical_mean", "null_constant_density"]
A_METHODS = ["unet", "random_forest", "ndvi_threshold"]
LABEL = {"unet": "U-Net", "random_forest": "Random forest",
         "ndvi_threshold": "NDVI threshold",
         "null_historical_mean": "N2 historical mean",
         "null_constant_density": "N1 constant density"}


def _by(runs, track):
    d = {}
    for r in runs:
        if r["track"] == track:
            d.setdefault(r["method"], {})[r["fold_year"]] = r
    return d


def _fmt(x, p=3):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{p}f}"


def track_b_table(runs) -> list[str]:
    d = _by(runs, "B")
    ratios = {"unet": UNET_B}
    for m in B_METHODS:
        if m == "unet":
            continue
        ratios[m] = {y: d.get(m, {}).get(y, {}).get("metrics", {}).get("aoi_ratio")
                     for y in YEARS}
    L = ["## Track B — out-of-year magnitude (aoi_ratio = predicted / official ha)",
         "",
         "> Track B sums over the AOI, so it is **structurally incapable of judging "
         "spatial skill** — that is Track A's job (prereg A2). U-Net row is the frozen "
         "v2.1 LOYO result.", "",
         "| held-out year | " + " | ".join(LABEL[m] for m in B_METHODS) + " |",
         "|---|" + "---|" * len(B_METHODS)]
    for y in YEARS:
        L.append(f"| {y} | " + " | ".join(_fmt(ratios[m].get(y), 2) for m in B_METHODS) + " |")
    # aggregates
    def agg(m):
        r = np.array([ratios[m][y] for y in YEARS if ratios[m].get(y) is not None], float)
        return r.mean(), r.std(), abs(r.mean() - 1), int(((r >= .85) & (r <= 1.15)).sum()), r.min(), r.max()
    L += ["| **mean** | " + " | ".join(_fmt(agg(m)[0], 2) for m in B_METHODS) + " |",
          "| **std (S)** | " + " | ".join(_fmt(agg(m)[1], 3) for m in B_METHODS) + " |",
          "| **\\|mean−1\\| (C)** | " + " | ".join(_fmt(agg(m)[2], 3) for m in B_METHODS) + " |",
          "| **K in [0.85,1.15]** | " + " | ".join(f"{agg(m)[3]}/6" for m in B_METHODS) + " |",
          "| **band** | " + " | ".join(f"{agg(m)[4]:.2f}–{agg(m)[5]:.2f}" for m in B_METHODS) + " |"]
    # paired deltas vs U-Net
    L += ["", "### Paired per-fold deltas vs U-Net  (δ = |ratio−1|(X) − |ratio−1|(U-Net); δ>0 ⇒ U-Net closer)",
          "", "| year | " + " | ".join(LABEL[m] for m in B_METHODS if m != "unet") + " |",
          "|---|" + "---|" * (len(B_METHODS) - 1)]
    du = {y: abs(UNET_B[y] - 1) for y in YEARS}
    for y in YEARS:
        cells = []
        for m in B_METHODS:
            if m == "unet":
                continue
            v = ratios[m].get(y)
            cells.append("—" if v is None else f"{abs(v-1)-du[y]:+.3f}")
        L.append(f"| {y} | " + " | ".join(cells) + " |")
    verdict = []
    for m in B_METHODS:
        if m == "unet":
            continue
        deltas = [abs(ratios[m][y]-1)-du[y] for y in YEARS if ratios[m].get(y) is not None]
        if not deltas:
            continue
        unet_better = sum(x > 0 for x in deltas)
        x_better = sum(x < 0 for x in deltas)
        call = ("U-Net clearly better (≥5/6)" if unet_better >= 5 else
                f"{LABEL[m]} clearly better (≥5/6)" if x_better >= 5 else
                "indistinguishable at n=6")
        verdict.append(f"- **U-Net vs {LABEL[m]}:** U-Net closer in {unet_better}/6, "
                       f"mean δ = {np.mean(deltas):+.3f} → **{call}**")
    L += ["", "**Paired verdicts (prereg §5, ≥5/6 rule):**", *verdict]
    return L


def track_a_table(runs) -> list[str]:
    d = _by(runs, "A")
    metrics = ["mae", "rmse", "presence_iou", "presence_f1"]
    L = ["## Track A — spatial skill on the held-out `test` blocks (per year)",
         "", "> Fit on all-years train blocks, evaluated on all-years spatially-held-out "
         "test blocks. This is the track that judges whether a segmentation model is "
         "warranted.", ""]
    for mt in metrics:
        L += [f"### {mt}", "",
              "| year | " + " | ".join(LABEL[m] for m in A_METHODS) + " |",
              "|---|" + "---|" * len(A_METHODS)]
        for y in YEARS:
            row = [d.get(m, {}).get(y, {}).get("metrics", {}).get(mt) for m in A_METHODS]
            L.append(f"| {y} | " + " | ".join(_fmt(v, 4 if mt in ("mae","rmse") else 3) for v in row) + " |")
        means = []
        for m in A_METHODS:
            vals = [d.get(m, {}).get(y, {}).get("metrics", {}).get(mt) for y in YEARS]
            vals = [v for v in vals if v is not None]
            means.append(np.mean(vals) if vals else None)
        L.append("| **mean** | " + " | ".join(_fmt(v, 4 if mt in ("mae","rmse") else 3) for v in means) + " |")
        L.append("")
    # paired IoU verdict: U-Net vs best baseline per year
    iou = {m: {y: d.get(m, {}).get(y, {}).get("metrics", {}).get("presence_iou") for y in YEARS}
           for m in A_METHODS}
    wins = 0; n = 0
    for y in YEARS:
        u = iou["unet"].get(y)
        base = [iou[m].get(y) for m in A_METHODS if m != "unet"]
        base = [b for b in base if b is not None]
        if u is None or not base:
            continue
        n += 1; wins += (u > max(base))
    if n:
        call = "U-Net wins spatially (≥5/6)" if wins >= 5 else "spatial tie / not a clear win"
        L += [f"**Paired presence-IoU verdict (prereg §5):** U-Net > best baseline in "
              f"{wins}/{n} years → **{call}**."]
    return L


def _dedup_last(runs):
    """Keep the LAST record per (method, track, fold_year) so a re-run supersedes a
    stale row without rewriting the append-only JSONL."""
    keep = {}
    for r in runs:
        keep[(r["method"], r["track"], r["fold_year"])] = r
    return list(keep.values())


def main(cfg):
    runs = _dedup_last(read_runs(cfg))
    if not runs:
        raise SystemExit("no runs in baseline_ladder.jsonl")
    lines = ["# Baseline ladder — comparison tables",
             "", f"Generated from `outputs/metrics/baseline_ladder.jsonl` "
             f"({len(runs)} runs). Reproduce: `python -m src.baselines.compare`.", ""]
    lines += track_b_table(runs) + ["", "---", ""] + track_a_table(runs)
    out = Path(cfg["paths"]["outputs_dir"]) / "metrics"
    (out / "ladder_table.md").write_text("\n".join(lines) + "\n")
    # CSV: one row per (track, method, year)
    import csv
    with open(out / "ladder_table.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["track", "method", "fold_year", "aoi_ratio", "mae", "rmse",
                    "presence_iou", "presence_f1", "calibration_scalar"])
        for r in runs:
            m = r["metrics"]
            w.writerow([r["track"], r["method"], r["fold_year"], m.get("aoi_ratio"),
                        m.get("mae"), m.get("rmse"), m.get("presence_iou"),
                        m.get("presence_f1"), r.get("calibration_scalar")])
    print(f"[compare] wrote {out/'ladder_table.md'} and .csv from {len(runs)} runs")
    print("\n".join(lines))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Assemble the baseline-ladder tables.")
    ap.add_argument("--config", default=None)
    main(load_config(ap.parse_args().config))
